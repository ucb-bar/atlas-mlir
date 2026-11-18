"""Mixed FP8/BF16 tiles compose selected primitives through exact source edges."""
from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import struct
import tempfile
import unittest

from atlas_mlir_native.binding import (
    AtlasNativeBinding, BF16_BINARY_ATTRS, MXU1_CAP448_POLICY,
    MXU1_CAP448_INPUT_DOMAIN, ROUNDED_PACK_DOMAIN, ROUNDED_PACK_POLICY,
    TARGET_IDENTITY, _mxu0_maps,
)
from atlas_mlir_native.resident_graph import is_staged_mixed, mixed_boundary_request, mixed_memory_plan
from merlin.semantic_compiler.model import (
    ConstantBinding, KernelRequest, SemanticNode, TensorType, pointwise_identity_maps,
)

ROOT = Path(__file__).resolve().parents[1]


def request(*, expose=True, transpose=False, seeded=False):
    raw = TensorType((32, 32), 'bf16', 'raw-bit-pattern')
    fp8 = TensorType((32, 32), 'fp8_e4m3', 'raw-bit-pattern')
    packed = replace(fp8, numerical_policy=ROUNDED_PACK_POLICY)
    attrs = BF16_BINARY_ATTRS['mul_bf16_zero']
    x = SemanticNode('x', 'input', (), raw, (("input_domain", ROUNDED_PACK_DOMAIN),), 'input')
    weight = SemanticNode('weight', 'constant', (), fp8, (("input_domain", MXU1_CAP448_INPUT_DOMAIN),), 'constant')
    scale = SemanticNode('scale', 'constant', (), raw, (attrs[0],), 'constant')
    pack = SemanticNode('packed', 'pack_fp8', ('x',), packed,
                        (("input_domain", ROUNDED_PACK_DOMAIN), ("scale_code", 127)),
                        index_maps=pointwise_identity_maps(2, 1))
    transposes = ()
    if transpose:
        from atlas_mlir_native.binding import _transpose_maps
        transposes = (SemanticNode('transposed', 'transpose', ('packed',), packed, index_maps=_transpose_maps()),)
    from atlas_mlir_native.binding import _mxu0_accumulate_maps
    seed = (SemanticNode('seed', 'constant', (), raw, effect='constant'),) if seeded else ()
    dot = SemanticNode('dot', 'matmul_accumulate' if seeded else 'matmul',
                       ('transposed' if transpose else 'packed', 'weight') + (('seed',) if seeded else ()),
                       replace(raw, numerical_policy=MXU1_CAP448_POLICY),
                       index_maps=_mxu0_accumulate_maps() if seeded else _mxu0_maps())
    output = SemanticNode('output', 'mul_bf16_zero', ('dot', 'scale'), raw, attrs,
                          index_maps=pointwise_identity_maps(2, 2))
    outputs = (pack.id, *((transposes[-1].id,) if transposes else ()), dot.id, output.id) if expose else (output.id,)
    return KernelRequest((x, weight, scale, *seed, pack, *transposes, dot, output), outputs, ('external',)*len(outputs),
                         ((x.id, 'external'),), TARGET_IDENTITY, lowering_policy='diagnostic',
                         constants=(ConstantBinding(weight.id, 'fp8_e4m3-raw', '38'*1024),
                                    ConstantBinding(scale.id, 'bf16-le-raw', '003f'*1024),
                                    *((ConstantBinding('seed', 'bf16-le-raw', '803f'*1024),) if seeded else ())))


def expected_values(source):
    from atlas_mlir_native.pack_lane import pack_bf16_lane
    from atlas_mlir_native.reference import evaluate_mxu1_cap448, evaluate_bf16_binary_zero
    from merlin.semantic_compiler.reference import TensorValue
    values = {
        'x': TensorValue(source.node('x').type, tuple((0x3f80, 0x4000, 0x3e80, 0x3f00)[(r+c)%4] for r in range(32) for c in range(32))),
        'weight': TensorValue(source.node('weight').type, (0x38,)*1024),
        'scale': TensorValue(source.node('scale').type, (0x3f00,)*1024),
    }
    if 'seed' in {node.id for node in source.nodes}:
        values['seed'] = TensorValue(source.node('seed').type, (0x3f80,)*1024)
    for node in source.nodes:
        if node.effect != 'pure':
            continue
        operands = tuple(values[key] for key in node.inputs)
        if node.op == 'pack_fp8':
            values[node.id] = TensorValue(node.type, tuple(pack_bf16_lane(bits, 127) for bits in operands[0].elements))
        elif node.op == 'transpose':
            values[node.id] = TensorValue(node.type, tuple(operands[0].elements[c*32+r] for r in range(32) for c in range(32)))
        elif node.op in {'matmul', 'matmul_accumulate'}:
            values[node.id] = evaluate_mxu1_cap448(node, operands)
        else:
            # The recorded completed-byte transport changes only the policy view.
            raw_operands = tuple(TensorValue(node.type, value.elements) for value in operands)
            values[node.id] = evaluate_bf16_binary_zero(node, raw_operands)
    return values


class MixedTilePlanTest(unittest.TestCase):
    def test_source_partition_transport_and_typed_fp8_edges(self):
        source = request(transpose=True)
        self.assertTrue(is_staged_mixed(source))
        digest = source.digest()
        cuts, _, sizes, _ = mixed_memory_plan(source)
        self.assertEqual([tuple(node.op for node in cut.nodes if node.effect == 'pure') for cut in cuts],
                         [('pack_fp8',), ('transpose',), ('matmul',), ('mul_bf16_zero',)])
        self.assertEqual(sizes['packed'], 1024)
        self.assertEqual(sizes['dot'], 2048)
        body = mixed_boundary_request(cuts[2])
        self.assertEqual(body.node('transposed').type.numerical_policy, 'raw-bit-pattern')
        self.assertEqual(source.node('transposed').type.numerical_policy, ROUNDED_PACK_POLICY)
        self.assertEqual(source.digest(), digest)
        self.assertEqual(body.node('dot'), source.node('dot'))

    def test_sources_stay_immutable_and_completed_tiles_reuse_storage(self):
        source = request(expose=False)
        cuts, slots, sizes, buffers = mixed_memory_plan(source)
        readonly = next(buffer for buffer in buffers if buffer['name'] == '$readonly')
        self.assertTrue(readonly['persistent'])
        self.assertEqual({slots[key] for key in ('x', 'weight', 'scale')}, {0, 1, 2})
        self.assertTrue(all(slots[key] >= 3 for key in ('packed', 'dot', 'output')))
        self.assertLess(len(set(slots[key] for key in ('packed', 'dot', 'output'))), 3)
        self.assertEqual(len(cuts), 3)
        self.assertEqual(sizes['weight'], 1024)

    def test_incompatible_policy_shape_and_single_op_are_not_mixed_staging(self):
        source = request()
        wrong = replace(source.node('packed'), type=replace(source.node('packed').type, numerical_policy='unqualified'))
        self.assertFalse(is_staged_mixed(replace(source, nodes=tuple(wrong if node.id == wrong.id else node for node in source.nodes))))
        wide = replace(source, nodes=tuple(replace(node, type=replace(node.type, shape=(32, 64))) if node.id == 'x' else node for node in source.nodes))
        self.assertFalse(is_staged_mixed(wide))
        from merlin.compile.semantic_regions import extract_semantic_region
        self.assertFalse(is_staged_mixed(extract_semantic_region(source, ('packed',))))


@unittest.skipUnless(os.environ.get('MERLIN_NATIVE_ROOT') and os.environ.get('ATLAS_OOT_BIN_DIR'), 'select native tools')
class MixedTileNativeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from merlin.semantic_compiler.snapshot import build_native_snapshot, packaged_egg_bridge_path
        retained = os.environ.get('ATLAS_MIXED_TEST_OUT')
        cls.temporary = None if retained else tempfile.TemporaryDirectory(dir=os.environ.get('TMPDIR'))
        cls.work = Path(retained or cls.temporary.name)
        cls.work.mkdir(exist_ok=True)
        cls.snapshot = build_native_snapshot(AtlasNativeBinding.profile(), destination=cls.work/'snapshot',
            crate=packaged_egg_bridge_path(), cargo_target_dir=Path(os.environ.get('ATLAS_MXU1_CARGO_TARGET', cls.work/'cargo')),
            source_revision='mixed-tile-native-test')

    @classmethod
    def tearDownClass(cls):
        if cls.temporary:
            cls.temporary.cleanup()

    def test_generated_fragments_compose_and_execute_one_exact_image(self):
        from merlin.semantic_compiler.search import SearchLimits
        from merlin.semantic_compiler.target_binding import verify_native_publication
        for transpose, seeded in ((False, False), (True, False), (False, True)):
            source = request(transpose=transpose, seeded=seeded)
            abi = AtlasNativeBinding.plan_boundary_abi(source)
            destination = self.work/('seeded' if seeded else 'with-transpose' if transpose else 'pack-mxu-multiply')
            manifest = AtlasNativeBinding.compile(self.snapshot, source, fixed_inputs=abi['fixed_inputs'],
                fixed_outputs=tuple(abi['fixed_outputs']), target_source=ROOT, destination=destination,
                limits=SearchLimits(candidate_nodes=256, wall_timeout_s=120))
            verify_native_publication(destination, manifest, engine='merlin_native', request_digest=source.digest(), target_identity=source.target_identity)
            plan = json.loads((destination/'execution_plan.json').read_text())
            self.assertEqual(plan['operation'], 'mixed_tile_staged')
            self.assertEqual(plan['launches_per_invocation'], 1)
            self.assertEqual(plan['backend_verification_tier'], 'verified')
            self.assertTrue(all('boundary_transports' in region for region in plan['regions']))
            self.assertTrue(all(region['source_slice_digest'] for region in plan['regions']))
            self.assertEqual(plan['input_payload_bytes'], [2048, 1024, 2048] + ([2048] if seeded else []))
            self.assertEqual([entry['encoding'] for entry in plan['constant_inputs']], ['fp8_e4m3-raw', 'bf16-le-raw'] + (['bf16-le-raw'] if seeded else []))
            if os.environ.get('ATLAS_MXU1_MODEL') == '1':
                self.execute(source, destination, plan)

    def execute(self, source, destination, plan):
        import dataclasses
        import hashlib
        import torch
        from npu_model.configs import isa_definition as isa
        from npu_model.configs.hardware.default import DefaultHardwareConfig
        from npu_model.logging import LoggerConfig
        from npu_model.simulation import Simulation
        from npu_model.software.program import InstantiableProgram
        from tools.check_program_decode import decode_binary
        from tools.model_instructions import model_instruction
        from atlas_mlir_native.layout import pack_bf16_pair_halves
        from test_mxu1_cap448 import authenticate_model, MODEL_PIN
        torch.set_num_threads(1)
        model_root = Path(os.environ['NPU_MODEL_ROOT'])
        before = authenticate_model(model_root)
        inventory = json.loads((ROOT/'docs/selected-variant-inventory.json').read_text())
        binary = (destination/'program.bin').read_bytes()
        decoded = decode_binary(inventory, binary)
        device = InstantiableProgram([model_instruction(row, inventory, isa) for row in decoded['instructions']])
        device.dram_base = 0x90000000
        top = max(entry['address']+2048 for entry in plan['buffers'])-device.dram_base+2048
        initial = bytearray((i*17+43)%256 for i in range(top))
        values = expected_values(source)
        def payload(key):
            value = values[key]
            return bytes(value.elements) if value.type.dtype == 'fp8_e4m3' else pack_bf16_pair_halves(struct.pack('<1024H', *value.elements))
        for entry in plan['inputs']:
            start = entry['address']-device.dram_base
            initial[start:start+entry['bytes']] = payload(entry['node'])
        expected = bytearray(initial)
        for entry in plan['outputs']:
            start = entry['address']-device.dram_base
            expected[start:start+entry['bytes']] = payload(entry['node'])
        device.memory_regions = [(0, torch.tensor(list(initial), dtype=torch.uint8))]
        config = DefaultHardwareConfig()
        config.arch_state_config = dataclasses.replace(config.arch_state_config, numerics='rtl', randomize_init=True, init_seed=187)
        simulation = Simulation(config, LoggerConfig(filename=str(destination/'model.json')), device, verbose=False)
        state = simulation.core.arch_state
        try:
            state.vmem.fill_(0xcd)
            simulation.run(max_cycles=100000)
            self.assertTrue(state.halted)
            self.assertEqual(simulation.get_stats().runtime_errors, 0)
            actual = bytes(state.read_dram(device.dram_base, len(initial)).tolist())
            self.assertEqual(actual, expected)
            after = authenticate_model(model_root)
            self.assertTrue(all(after.get(name) == digest for name, digest in before.items()))
            report = dict(status='PASS', model_commit=MODEL_PIN, numerics='rtl', model_source_sha256=after,
                program_sha256=hashlib.sha256(binary).hexdigest(), program_words=len(binary)//4,
                cycles=simulation.get_stats().cycles, output_bytes=sum(entry['bytes'] for entry in plan['outputs']),
                full_arena_bytes=len(initial), all_stage_outputs_exact=True, all_non_output_bytes_preserved=True,
                backend_verification_tier='verified')
            (destination/'model-receipt.json').write_text(json.dumps(report,indent=2)+'\n')
        finally:
            state.close()


if __name__ == '__main__':
    unittest.main()
