"""Source-attributed cap448 MXU1 BF16 continuation and native byte checks."""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import random
import struct
import subprocess
import sys
import tempfile
import unittest

from atlas_mlir_native.binding import (
    AtlasNativeBinding, MXU1_CAP448_POLICY, MXU1_CAP448_INPUT_DOMAIN,
    MXU1_CAP448_SEED_DOMAIN, _mxu0_maps, _mxu0_accumulate_maps,
    check_mxu1_cap448_stream,
)
from atlas_mlir_native.reference import mxu1_cap448_lane, evaluate_mxu1_cap448, _mxu1_product_fields
from merlin.semantic_compiler.model import KernelRequest, SemanticNode, TensorType
from merlin.semantic_compiler.reference import TensorValue, evaluate_graph

ROOT = Path(__file__).resolve().parents[1]
MODEL_PIN = '47b8599d8ffbe5ddd96059ffe2b5be2427a410f5'


def request(*, tiles=2, seeded=False):
    fp8 = TensorType((32, 32), 'fp8_e4m3', 'raw-bit-pattern')
    bf16 = TensorType((32, 32), 'bf16', 'raw-bit-pattern')
    output = TensorType((32, 32), 'bf16', MXU1_CAP448_POLICY)
    inputs = tuple(SemanticNode(f'{role}{i}', 'input', (), fp8,
                               (("input_domain", MXU1_CAP448_INPUT_DOMAIN),), 'input')
                   for i in range(tiles) for role in ('a', 'w'))
    if seeded:
        inputs += (SemanticNode('seed', 'input', (), bf16,
                                (("input_domain", MXU1_CAP448_SEED_DOMAIN),), 'input'),)
    contractions = []
    for i in range(tiles):
        continued = seeded or i > 0
        children = (f'a{i}', f'w{i}') + (((f's{i-1}' if i else 'seed'),) if continued else ())
        contractions.append(SemanticNode(f's{i}', 'matmul_accumulate' if continued else 'matmul',
                                        children, output, index_maps=_mxu0_accumulate_maps() if continued else _mxu0_maps()))
    return KernelRequest((*inputs, *contractions), (contractions[-1].id,), ('external',),
                         tuple((node.id, 'external') for node in inputs),
                         AtlasNativeBinding.profile().target_identity, lowering_policy='diagnostic')


def panel(source):
    rng = random.Random(1927)
    palette = (0, 0x08, 0x18, 0x38, 0x40, 0x70, 0x78, 0x7e,
               0x88, 0xb8, 0xc0, 0xf0, 0xf8, 0xfe)
    values = {node.id: TensorValue(node.type, tuple(
        rng.choice(palette) for _ in range(1024))) for node in source.nodes if node.op == 'input' and node.type.dtype == 'fp8_e4m3'}
    if 'seed' in {node.id for node in source.nodes}:
        values['seed'] = TensorValue(source.node('seed').type, (0x4901,) * 1024)
    return values


def authenticate_model(root):
    """Authenticate imported model sources directly against the selected pin."""
    sums = {}
    for name, module in list(sys.modules.items()):
        path = getattr(module, '__file__', None)
        if not (name == 'npu_model' or name.startswith('npu_model.')) or not path:
            continue
        path = Path(path)
        relative = path.relative_to(root)
        raw = path.read_bytes()
        blob = subprocess.check_output(['git', '-C', str(root), 'cat-file', 'blob', f'{MODEL_PIN}:{relative}'])
        if raw != blob:
            raise AssertionError('imported model source differs from selected pin')
        sums[str(relative)] = hashlib.sha256(raw).hexdigest()
    if not sums:
        raise AssertionError('model import source inventory is empty')
    return sums


class MXU1Cap448ScalarTest(unittest.TestCase):
    def test_cap448_zero_ties_seed_and_old_domain(self):
        zero = (0,) * 32
        self.assertEqual(mxu1_cap448_lane(zero, zero), 0)
        # Model47 deliberately differs from normalized exact sum (0x3c00).
        self.assertEqual(mxu1_cap448_lane((0x7e, 0xfe, 0x08) + zero[3:],
                                         (0x7e, 0x7e, 0x30) + zero[3:]), 0x3c01)
        self.assertEqual(mxu1_cap448_lane((0x7e,)*32, (0x38,)*32), 0x4660)  # 14336
        self.assertEqual(mxu1_cap448_lane(zero, zero, 0x3f81), 0x3f81)
        a = (0x18,) + zero[1:]
        self.assertEqual(mxu1_cap448_lane(a, a, 0x3f80), 0x3f80)
        self.assertEqual(mxu1_cap448_lane(a, a, 0x3f81), 0x3f82)
        for bad in (1, 0x80, 0x7f, 0xff):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                mxu1_cap448_lane((bad,) + zero[1:], zero)
        with self.assertRaises(ValueError):
            _mxu1_product_fields(0x78, 0x38)
        for seed in (0x8000, 1, 0x7f80, 0x7fc0):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                mxu1_cap448_lane(zero, zero, seed)

    @unittest.skipUnless(os.environ.get('ATLAS_MXU1_MODEL') == '1', 'select original47 for direct scalar comparison')
    def test_seeded_scalar_against_unmodified_original47(self):
        import torch
        from npu_model.configs.rtl_math import ipt_row
        model_root = Path(os.environ['NPU_MODEL_ROOT'])
        self.assertEqual(subprocess.check_output(['git', '-C', str(model_root), 'rev-parse', 'HEAD'], text=True).strip(), MODEL_PIN)
        torch.set_num_threads(1)
        rng = random.Random(831)
        palette = [0] + [sign | exponent << 3 | mantissa for sign in (0, 128)
                         for exponent in range(1, 16) for mantissa in range(7 if exponent == 15 else 8)]
        cases = []
        for _ in range(200):
            a, w = ([rng.choice(palette) for _ in range(32)] for _ in range(2))
            seed = rng.choice([0, 0x3f80, 0xbf80, 0x7f7f, 0xff7f, 0x0080, 0x8080,
                               rng.randrange(1, 255) << 7 | rng.randrange(128)])
            cases.append((a, w, seed))
        cases.append(([0x7e, 0xfe, 0x08]+[0]*29, [0x7e, 0x7e, 0x30]+[0]*29, 0))
        for i, (a, w, seed) in enumerate(cases):
            got = ipt_row(torch.tensor(a, dtype=torch.uint8).view(torch.float8_e4m3fn),
                          torch.tensor([w], dtype=torch.uint8).view(torch.float8_e4m3fn),
                          torch.tensor([seed], dtype=torch.uint16).view(torch.bfloat16))
            self.assertEqual(mxu1_cap448_lane(tuple(a), tuple(w), seed), int(got.view(torch.uint16).item()), i)


@unittest.skipUnless(os.environ.get('MERLIN_NATIVE_ROOT') and os.environ.get('ATLAS_OOT_BIN_DIR'), 'select native tools')
class MXU1Cap448NativeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from merlin.semantic_compiler.snapshot import build_native_snapshot, packaged_egg_bridge_path
        retained = os.environ.get('ATLAS_MXU1_TEST_OUT')
        cls.temporary = None if retained else tempfile.TemporaryDirectory(dir=os.environ.get('TMPDIR'))
        cls.work = Path(retained or cls.temporary.name)
        cls.work.mkdir(exist_ok=True)
        cls.snapshot = build_native_snapshot(AtlasNativeBinding.profile(), destination=cls.work/"snapshot",
            crate=packaged_egg_bridge_path(), cargo_target_dir=Path(os.environ.get("ATLAS_MXU1_CARGO_TARGET", cls.work/"cargo")),
            source_revision="model47-cap448-mxu1-native-test")

    @classmethod
    def tearDownClass(cls):
        if cls.temporary:
            cls.temporary.cleanup()

    def test_native_k64_and_bf16_seed(self):
        from merlin.semantic_compiler.search import SearchLimits
        from merlin.semantic_compiler.target_binding import verify_native_publication, NativeCompilationError
        records = []
        for seeded in (False, True):
            source = request(seeded=seeded)
            destination = self.work / ('seeded' if seeded else 'reset')
            inputs = {node.id: 3+i for i, node in enumerate(source.nodes) if node.op == 'input'}
            manifest = AtlasNativeBinding.compile(self.snapshot, source, fixed_inputs=inputs, fixed_outputs=(35,),
                target_source=ROOT, destination=destination, limits=SearchLimits(candidate_nodes=256))
            verify_native_publication(destination, manifest, engine='merlin_native', request_digest=source.digest(), target_identity=source.target_identity)
            program = json.loads((destination/'physical-program.json').read_text())
            plan = json.loads((destination/'execution_plan.json').read_text())
            # The pass records state 'timed'; its unconditional success and
            # machine-stream success define the backend's verified tier.
            self.assertEqual(program['timing_state'], 'timed')
            self.assertNotIn('atlas.dram_contract', (destination/'atlas-machine.mlir').read_text())
            self.assertEqual(plan['operation'], 'mxu1_cap448_seeded_ordered')
            source_ids = tuple(inputs)
            arguments = dict(request=source, source_ids=source_ids, input_slots=tuple(inputs.values()), output_slot=35)
            check_mxu1_cap448_stream(program, **arguments)
            changed = copy.deepcopy(program)
            [row for row in changed['instructions'] if row['operation'] == 'atlas.mxu_matmul'][-1]['fields']['accumulate'] = False
            with self.assertRaises(NativeCompilationError):
                check_mxu1_cap448_stream(changed, **arguments)
            if os.environ.get('ATLAS_MXU1_MODEL') == '1':
                records.append(self.execute(source, destination, inputs, plan))
        if records:
            (self.work/'receipt.json').write_text(json.dumps(dict(status='PASS', model_commit=MODEL_PIN, numerics='rtl',
                compiled_by='Merlin native semantic selection and Atlas virtual pipeline', cases=records), indent=2)+'\n')

    def execute(self, source, destination, inputs, plan):
        import torch
        from npu_model.configs import isa_definition as isa
        from npu_model.configs.hardware.default import DefaultHardwareConfig
        from npu_model.logging import LoggerConfig
        from npu_model.simulation import Simulation
        from npu_model.software.program import InstantiableProgram
        from tools.check_program_decode import decode_binary
        from tools.model_instructions import model_instruction
        from atlas_mlir_native.layout import pack_bf16_pair_halves
        torch.set_num_threads(1)
        model_root = Path(os.environ['NPU_MODEL_ROOT'])
        self.assertEqual(subprocess.check_output(['git', '-C', str(model_root), 'rev-parse', 'HEAD'], text=True).strip(), MODEL_PIN)
        model_sources = authenticate_model(model_root)
        binary = (destination/'program.bin').read_bytes()
        inventory = json.loads((ROOT/'docs/selected-variant-inventory.json').read_text())
        decoded = decode_binary(inventory, binary)
        device = InstantiableProgram([model_instruction(row, inventory, isa) for row in decoded['instructions']])
        device.dram_base = 0x90000000
        initial = bytearray((i*17+43)%256 for i in range(37*2048))
        values = panel(source)
        for node, slot in inputs.items():
            value = values[node]
            raw = bytes(value.elements) if value.type.dtype == 'fp8_e4m3' else pack_bf16_pair_halves(struct.pack('<1024H', *value.elements))
            initial[slot*2048:slot*2048+len(raw)] = raw
        result = evaluate_graph(source, values, primitives={('matmul', MXU1_CAP448_POLICY): evaluate_mxu1_cap448,
            ('matmul_accumulate', MXU1_CAP448_POLICY): evaluate_mxu1_cap448})[0]
        output = pack_bf16_pair_halves(struct.pack('<1024H', *result.elements))
        expected = bytearray(initial)
        expected[35*2048:36*2048] = output
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
            stats = simulation.get_stats()
            after = authenticate_model(model_root)
            self.assertTrue(all(after.get(path) == digest for path, digest in model_sources.items()))
            return dict(backend_timing_tier="verified", model_source_sha256=after, source_bookends_verified=True, program_sha256=hashlib.sha256(binary).hexdigest(), program_words=len(binary)//4,
                        cycles=stats.cycles, output_bytes=2048, full_arena_bytes=len(initial), output_exact=True,
                        non_output_guards_preserved=True, output_sha256=hashlib.sha256(output).hexdigest())
        finally:
            state.close()


if __name__ == '__main__':
    unittest.main()
