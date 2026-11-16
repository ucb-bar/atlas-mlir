"""A closed BF16 SiLU expansion retains selected arithmetic and rejection."""
import json
import os
import subprocess
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
from atlas_mlir_native import expanded_silu as native
from atlas_mlir_native.application import AtlasApplicationAdapter
from atlas_mlir_native.binding import DRAM_BASE
from atlas_mlir_native.composition_adapter import AdjacentDeviceAdapter
from merlin.semantic_compiler.codegen import CodegenOptions, codegen_options
from merlin.semantic_compiler.model import SemanticNode, pointwise_identity_maps
from test_adjacent_pointwise import case as exp_case
from test_application_float_arithmetic import policy

ROOT = Path(__file__).resolve().parents[1]


def case(shape=(3, 35)):
    source, _ = exp_case(shape=shape)
    x, neg, literal, ones, exp, add, divide = source.nodes
    x = replace(x, type=replace(x.type, dtype='bf16'))
    widen = SemanticNode('wide', 'bf16_to_f32', ('x',), divide.type,
        (('numeric_semantics', 'bf16-to-f32-exact-v1'),),
        index_maps=pointwise_identity_maps(len(shape), 1))
    neg = replace(neg, op='f32_neg', inputs=('wide',), effect='pure',
        attrs=(('numeric_semantics', 'f32-sign-bit-flip-v1'),), index_maps=widen.index_maps)
    divide = replace(divide, inputs=('wide', add.id))
    out = SemanticNode('out', 'f32_to_bf16', (divide.id,), x.type,
        (('numeric_semantics', 'f32-to-bf16-rne-v1'),), index_maps=widen.index_maps)
    return replace(source, nodes=(x, literal, widen, ones, neg, exp, add, divide, out),
        outputs=('out',), input_storages=(('x', 'external'),))


def adapter():
    from atlas_mlir_native.already_quantized import POLICY as INTAKE
    from atlas_mlir_native.source_exp import POLICY
    selected = {**policy(), 'source_exp': POLICY, 'already_quantized': INTAKE,
                'host_float_arithmetic': ['f32_neg']}
    return AtlasApplicationAdapter(selected, original_source_sha256=selected['source_sha256'])


def expanded_reference(codes):
    from atlas_mlir_native.exp_lane import vexp_bf16_bits
    from atlas_mlir_native.recip_lane import vrecip_bf16_bounded_bits
    from atlas_mlir_native.reference import _f32_add_rne, _round_bf16_product

    result = []
    for code in codes:
        # Sign flip, unlike the macro's zero subtraction, reverses both zeros.
        exp = vexp_bf16_bits(int(code) ^ 0x8000)
        denominator = _f32_add_rne(0x3f800000, exp << 16) >> 16
        reciprocal = vrecip_bf16_bounded_bits(denominator)
        value = _round_bf16_product(int(code), reciprocal) if code & 0x7fff else int(code)
        result.append(value)
    return np.array(result, dtype='<u2')


class ExpandedSiluTest(unittest.TestCase):
    def setUp(self):
        options = codegen_options(CodegenOptions(io_memory_bytes=67108864))
        options.__enter__()
        self.addCleanup(options.__exit__, None, None, None)

    def test_selector_uses_one_closed_region_and_preserves_source(self):
        source = case()
        before = source.record()
        wrapper = AdjacentDeviceAdapter(adapter(), patterns=('pointwise_narrowing', 'expanded_silu'))
        selected, = wrapper.plan(source).selected
        self.assertEqual(selected.nodes, tuple(range(7)))
        self.assertEqual(dict(selected.metadata)['kind'], 'expanded_silu')
        self.assertEqual(tuple(p.buffer for p in selected.inputs), ('x', 'literal'))
        self.assertEqual(tuple(p.buffer for p in selected.outputs), ('out',))
        self.assertEqual(source.record(), before)
        escaped = replace(source, outputs=('exp', 'out'), output_storages=('external',) * 2)
        self.assertFalse(any(len(row.nodes) == 7 for row in wrapper.plan(escaped).selected))

    def test_changed_policies_literal_edges_rounding_and_maps_decline(self):
        source = case()
        base = adapter()
        for field, value in (('pointwise', False), ('pointwise_math', False), ('sigmoid', False),
                             ('source_exp_policy', None), ('qualified_rne_add', 'another-policy')):
            changed = adapter()
            setattr(changed, field, value)
            with self.subTest(field=field), self.assertRaises(ValueError):
                native.prepare(source, changed)
        for index, node in enumerate(source.nodes):
            if node.effect == 'pure' and node.op != 'gather':
                for changed in (replace(node, attrs=()), replace(node, index_maps=())):
                    with self.subTest(node=node.id), self.assertRaises(ValueError):
                        native.prepare(replace(source, nodes=(*source.nodes[:index], changed,
                                                             *source.nodes[index + 1:])), base)
        for changed in (
            replace(source, constants=(replace(source.constants[0], data_hex='00000040'),)),
            replace(source, nodes=(*source.nodes[:-2], replace(source.nodes[-2],
                inputs=('neg', 'add')), source.nodes[-1])),
            replace(source, outputs=('wide', 'out'), output_storages=('external',) * 2),
        ):
            with self.assertRaises(ValueError):
                native.prepare(changed, base)

    def test_exhaustive_bf16_admission_and_selected_arithmetic(self):
        from atlas_mlir_native.pointwise_math import input_specs
        from atlas_mlir_native.reference import evaluate_silu
        from atlas_mlir_native.source_exp import input_spec as exp_spec
        from merlin.semantic_compiler.reference import TensorValue

        group = native.prepare(case(), adapter())
        admitted = lambda code, spec: any(lo <= code <= hi for lo, hi in spec.admitted_code_ranges)
        codes = []
        for code in range(65536):
            original = admitted(code ^ 0x8000, exp_spec()) and admitted(code, input_specs('divide_bf16_lut')[0])
            self.assertEqual(admitted(code, group.operand_specs[0]), original, hex(code))
            if original:
                codes.append(code)
        self.assertEqual(len(codes), 17412)
        group = native.prepare(case((1, len(codes))), adapter())
        result = evaluate_silu(group.request.nodes[-1], (TensorValue(group.request.nodes[0].type, tuple(codes)),))
        np.testing.assert_array_equal(result.elements, expanded_reference(codes))

    @unittest.skipUnless(os.environ.get('ATLAS_RUST_NPU_MODEL') and os.environ.get('ATLAS_OOT_BIN_DIR'),
                         'select Rust simulator and Atlas tools')
    def test_rust_c_abi_all_admitted_codes_tails_and_late_rejection(self):
        from atlas_mlir_native.layout import pack_bf16_pair_halves
        from atlas_mlir_native.pointwise import prepare_plan
        from atlas_mlir_native.pointwise_c import prepare_precision_c_package
        from merlin.semantic_compiler.pointwise_tiling import (
            materialize_pointwise_input,
        )
        from merlin.semantic_compiler.search import SearchLimits
        from test_looped_mnk import build_snapshot
        from test_pointwise_loop import check_source_c_call

        codes = np.array([0, 0x8000, *range(0x2000, 0x4201), *range(0xa000, 0xc201)], dtype='<u2')
        for shape in ((3, 35), (2, 8706)):
            with self.subTest(shape=shape), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                group = native.prepare(case(shape), adapter())
                compiled = root / 'compiled'
                native.compile_group(group, build_snapshot(root), target_source=ROOT,
                                     destination=compiled, limits=SearchLimits())
                exported = native.export_group(group, compiled, 'pointwise', target_source=ROOT)
                self.assertEqual(exported.description['launches_per_invocation'], 1)
                self.assertEqual(exported.description['source_input_indices'], [0])
                record, files = prepare_precision_c_package(group, compiled, 'pointwise',
                    target_source=ROOT, runtime_linkage='embedded', output_dtype='bf16')
                self.assertEqual(record['source_dtype'], 'bf16')
                package = root / 'package'
                package.mkdir()
                for name, data in files.items():
                    (package / name).write_bytes(data)
                execution = json.loads((compiled / 'execution_plan.json').read_text())
                plan = prepare_plan(group.request)
                values = np.resize(codes, shape)
                packed = materialize_pointwise_input(plan, 0, values.tobytes())
                packed = b''.join(pack_bf16_pair_halves(packed[i:i + 2048]) for i in range(0, len(packed), 2048))
                end = execution['output_addresses'][0] + execution['output_bytes'] - DRAM_BASE
                initial = bytearray([0xa5]) * ((end + 4095) // 4096 * 4096)
                at = execution['input_addresses'][0] - DRAM_BASE
                initial[at:at + len(packed)] = packed
                config = root / 'config.json'
                config.write_text(json.dumps({'compute': 'cpu', 'pack_layout': 'register_rows',
                                              'dram_base': DRAM_BASE, 'dram_size': len(initial)}))

                def execute(memory, root=root, compiled=compiled, config=config, execution=execution, end=end):
                    with tempfile.TemporaryDirectory(dir=root) as folder:
                        work = Path(folder)
                        (work / 'input.bin').write_bytes(memory)
                        subprocess.run([os.environ['ATLAS_RUST_NPU_MODEL'], str(compiled / 'program.bin'),
                            '--config', str(config), '--load', f'{DRAM_BASE}:{work / "input.bin"}',
                            '--dump-state', str(work / 'state')], check=True, capture_output=True)
                        actual = b''.join((work / 'state' / f'dram_{a:016x}.bin').read_bytes()
                            for a in range(DRAM_BASE, DRAM_BASE + len(memory), 4096))
                        first = execution['output_addresses'][0] - DRAM_BASE
                        self.assertEqual(actual[:first], memory[:first])
                        self.assertEqual(actual[end:], memory[end:])
                        return actual

                rejected = []
                for code in (0x7fc1, 0xffff, 0x7f80, 0xff80, 0x4201, 0xc201, 0x1fff, 0x9fff, 1):
                    invalid = values.copy()
                    invalid.flat[-1] = code
                    rejected.append((invalid,))
                check_source_c_call(self, package, initial, (compiled / 'program.bin').read_bytes(),
                    execute, [values], expanded_reference(values.flat).tobytes(), rejected,
                    source_dtype='bf16', output_dtype='bf16')


if __name__ == '__main__':
    unittest.main()
