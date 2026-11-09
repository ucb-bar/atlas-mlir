"""Square/reduction fusion moves precisely the old intermediate guard."""
import ctypes
from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
from types import SimpleNamespace
import unittest

from atlas_mlir_native.adjacent_square_reduce import (
    alternatives, compile_group, export_group, input_spec, intermediate_domain_proof, prepare,
    render_combination_guard,
)
from atlas_mlir_native.binding import TARGET_IDENTITY
from atlas_mlir_native.pointwise import input_spec as original_spec
from atlas_mlir_native.reduction import (
    NONNEGATIVE_NORMAL_REDUCTION_POLICY as POSITIVE, REDUCTION_POLICY, input_spec as reduction_spec, request_for,
)
from atlas_mlir_native.reference import _round_bf16_product
from merlin.semantic_compiler.model import KernelRequest, SemanticNode, TensorType, pointwise_identity_maps
from merlin.xdsl_dialects.lowering.global_plan import (
    BufferRepresentation, CycleInterval, RegionAlternative, ValueRepresentation,
)


def admits(spec, code):
    return any(low <= code <= high for low, high in spec.admitted_code_ranges)


def case(shape=(3, 7), axes=None, escape=False):
    axes = (len(shape) - 1,) if axes is None else axes
    raw = request_for(shape, axes)
    x = SemanticNode('x', 'input', (), TensorType(shape, 'f32', 'raw-bit-pattern'), effect='input')
    seed = SemanticNode('seed', 'input', (), replace(raw.nodes[1].type, dtype='f32'), effect='input')
    square = SemanticNode('square', 'f32_mul', ('x', 'x'), x.type,
        (('numeric_semantics', 'ieee754-f32-mul-rne-v1'),), index_maps=pointwise_identity_maps(len(shape), 2))
    reduced = replace(raw.nodes[-1], id='reduced', op='source_f32_reduce_add_unqualified',
        inputs=('square', 'seed'), type=seed.type,
        attrs=(('axes', ','.join(map(str, axes))), ('reduction_order', 'unresolved'),
               ('numeric_semantics', 'source-f32-add-unqualified-v1'), ('scalar_add_order', 'input-accumulator')))
    outputs = ('reduced', 'square') if escape else ('reduced',)
    source = KernelRequest((x, seed, square, reduced), outputs, ('external',) * len(outputs),
                           (('x', 'external'), ('seed', 'external')), TARGET_IDENTITY, 'diagnostic')
    def ports(keys):
        return tuple(BufferRepresentation(key, ValueRepresentation('host_memory', 'dense', 'f32')) for key in dict.fromkeys(keys))
    choices = tuple(RegionAlternative('r' + str(index), (index,), 'source', 'accelerator',
        CycleInterval.unknown('test'), ports(node.inputs), ports((node.id,)), metadata=(('kind', kind),))
        for index, (node, kind) in enumerate(((square, 'pointwise'), (reduced, 'reduction'))))
    return source, SimpleNamespace(reduction_policy=REDUCTION_POLICY), choices


class SquareReductionTest(unittest.TestCase):
    def test_exhaustive_preimage_preserves_both_original_guards(self):
        proof = intermediate_domain_proof()
        self.assertEqual(proof['checked_codes'], 65536)
        self.assertEqual(proof['original_admitted_codes'], 32260)
        self.assertEqual(proof['admitted_preimage_codes'], 16132)
        self.assertEqual(proof['admitted_code_ranges'], [[0, 0], [0x2fb5, 0x4f35],
                                                       [0x8000, 0x8000], [0xafb5, 0xcf35]])
        for code in range(65536):
            expected = False
            if admits(original_spec(), code):
                square = 0 if code & 0x7fff == 0 else _round_bf16_product(code, code)
                expected = admits(reduction_spec(), square)
            self.assertEqual(admits(input_spec(), code), expected, hex(code))

    def test_nonnegative_policy_preimage_is_the_pointwise_domain(self):
        # Every square of a pointwise-admitted BF16 input is +0 or a positive
        # normal code, so the selected reduction guard removes no input code.
        squares = set()
        for code in range(65536):
            if admits(original_spec(), code):
                square = 0 if code & 0x7fff == 0 else _round_bf16_product(code, code)
                self.assertTrue(admits(reduction_spec(reduction_policy=POSITIVE), square), hex(code))
                squares.add(square)
        self.assertEqual((min(squares - {0}), max(squares)), (0x0080, 0x7e80))
        proof = intermediate_domain_proof(POSITIVE)
        self.assertEqual(proof['reduction_policy'], POSITIVE)
        self.assertEqual((proof['original_admitted_codes'], proof['admitted_preimage_codes']), (32260, 32260))
        self.assertEqual(tuple(map(tuple, proof['admitted_code_ranges'])), original_spec().admitted_code_ranges)
        self.assertEqual(input_spec(reduction_policy=POSITIVE).admitted_code_ranges, original_spec().admitted_code_ranges)
        self.assertEqual(intermediate_domain_proof()['reduction_policy'], REDUCTION_POLICY)

    def test_fusion_records_the_selected_reduction_policy(self):
        for policy in (REDUCTION_POLICY, POSITIVE):
            source, _, choices = case()
            adapter = SimpleNamespace(reduction_policy=policy)
            fused = [row for row in alternatives(adapter, source, choices)
                     if dict(row.metadata)['kind'] == 'adjacent_square_reduce']
            self.assertEqual(len(fused), 1)
            group, = adapter._square_groups.values()
            self.assertEqual(group['reduction_policy'], policy)
            self.assertEqual(group['proof'], intermediate_domain_proof(policy))
            record = group['precisions'][1].record()
            self.assertEqual(record['arithmetic'], policy)
            self.assertEqual(record['input_conversion']['policy'], policy)
        # Without a selected reduction policy the separate reduction refuses,
        # and so does its enlargement.
        source, _, choices = case()
        adapter = SimpleNamespace(reduction_policy=None)
        self.assertEqual(alternatives(adapter, source, choices), choices)

    def test_nonnegative_guard_replays_the_selected_square(self):
        import numpy as np

        source, _, _ = case(shape=(1, 32))
        group = prepare(source, SimpleNamespace(reduction_policy=POSITIVE))
        guard = render_combination_guard(group, 'fused', group['inputs'])[0][0]
        codes = [code for code in range(65536) if admits(original_spec(), code)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'guard.c').write_text(guard + '''
uint32_t square(uint32_t code) { return fused_square(code); }
int domain(const uint8_t *data,const uint8_t *seed) { return fused_domain(data,seed); }
''')
            subprocess.run([shutil.which('cc'), '-std=c11', '-O2', '-shared', '-fPIC', '-Wall', '-Wextra', '-Werror',
                            str(root / 'guard.c'), '-o', str(root / 'guard.so')], check=True, capture_output=True)
            library = ctypes.CDLL(str(root / 'guard.so'))
            library.square.argtypes, library.square.restype = [ctypes.c_uint32], ctypes.c_uint32
            library.domain.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
            for code in codes:
                self.assertEqual(library.square(code), 0 if code & 0x7fff == 0 else _round_bf16_product(code, code),
                                 hex(code))
            def domain(data, seed):
                return library.domain(np.array(data, '<u2').tobytes(), np.array([seed], '<u2').tobytes())
            witness = 0xafb1  # nonzero |x| < 2**-31.5: refused by the default reduction domain
            self.assertFalse(admits(input_spec(), witness))
            self.assertTrue(admits(input_spec(reduction_policy=POSITIVE), witness))
            self.assertEqual(domain([witness] + [0] * 31, 0), 0)
            self.assertEqual(domain([0x5f00] * 2 + [0] * 30, 0x3f80), 0)
            # Squares of 2**63 are 2**126; four of them overflow FP32 in the tree.
            self.assertEqual(domain([0x5f00] * 4 + [0] * 28, 0), 1)
            self.assertEqual(domain([0x5f00] * 2 + [0] * 30, 0x7f7f), 1)

    def test_adjacent_pair_requires_identical_layout_and_no_escaping_square(self):
        source, adapter, choices = case()
        selected = [row for row in alternatives(adapter, source, choices)
                    if dict(row.metadata)['kind'] == 'adjacent_square_reduce']
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].nodes, (0, 1))
        self.assertEqual([port.buffer for port in selected[0].inputs], ['x', 'seed'])
        self.assertEqual(len(adapter._square_groups), 1)
        for kwargs in ({'escape': True}, {'shape': (3, 7), 'axes': (0,)}):
            source, adapter, choices = case(**kwargs)
            self.assertEqual(len(alternatives(adapter, source, choices)), 2)
        source, adapter, choices = case()
        choices = (choices[0], replace(choices[1], placement='host'))
        self.assertEqual(len(alternatives(adapter, source, choices)), 2)

    def test_large_square_keeps_original_device_regions(self):
        from merlin.semantic_compiler.codegen import CodegenOptions, codegen_options
        options = codegen_options(CodegenOptions(io_memory_bytes=67108864))
        options.__enter__()
        self.addCleanup(options.__exit__, None, None, None)
        source, adapter, choices = case(shape=(512, 2048))
        self.assertEqual(alternatives(adapter, source, choices), choices)
        self.assertEqual(adapter._square_groups, {})
        with self.assertRaisesRegex(ValueError, 'bounded packed square size'):
            prepare(source, adapter)
        # Fixed64 retains the existing checked enlargement.
        source, adapter, choices = case(shape=(64, 2048))
        self.assertEqual(len(alternatives(adapter, source, choices)), 3)
        # A prefix view would encode unused input values under the old route.
        # Moving the reduction guard to those unused values would reject more
        # calls, so only exact dense self-products may enter this alternative.
        source, adapter, choices = case()
        larger = replace(source.nodes[0], type=TensorType((3, 9), 'f32', 'raw-bit-pattern'))
        source = replace(source, nodes=(larger, *source.nodes[1:]))
        self.assertEqual(len(alternatives(adapter, source, choices)), 2)

    def test_c_preflight_rejects_square_guard_counterexamples_before_writes(self):
        import numpy as np
        from merlin.compile.c_conversions import render_f32_array_c

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            code = render_f32_array_c(input_spec(), 3, 'encode')
            (root / 'guard.c').write_text(code + '\nint check(const float *p,uint8_t *q) { return encode(p,q); }\n')
            subprocess.run([shutil.which('cc'), '-std=c11', '-shared', '-fPIC', '-Wall', '-Wextra', '-Werror',
                            str(root / 'guard.c'), '-o', str(root / 'guard.so')], check=True, capture_output=True)
            check = ctypes.CDLL(str(root / 'guard.so')).check
            check.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            valid = np.array([0.0, -0.0, -1.5], dtype='<f4')
            output = ctypes.create_string_buffer(bytes([0xa7]) * 6)
            self.assertEqual(check(valid.ctypes.data, output), 0)
            self.assertEqual(output.raw[:6], input_spec().encode(valid).tobytes())
            self.assertEqual(struct.unpack('<3H', output.raw[:6])[:2], (0, 0x8000))
            for value in (-2.0**40, -2.0**-40, 2.0**-40, 2.0**40):
                invalid = np.array([0.0, value, -0.0], dtype='<f4')
                encoded = original_spec().encode(invalid)
                code = struct.unpack('<3H', encoded)[1]
                squared = _round_bf16_product(code, code)
                self.assertFalse(any(low <= squared <= high for low, high in reduction_spec().admitted_code_ranges))
                output = ctypes.create_string_buffer(bytes([0xa7]) * 6)
                self.assertEqual(check(invalid.ctypes.data, output), 2)
                self.assertEqual(output.raw[:6], bytes([0xa7]) * 6)

    @unittest.skipUnless(os.environ.get('ATLAS_OOT_BIN_DIR'), 'select default native tools')
    def test_native_composition_c_numeric_roundtrip_and_guards(self):
        self._check_native(REDUCTION_POLICY)

    @unittest.skipUnless(os.environ.get('ATLAS_OOT_BIN_DIR'), 'select default native tools')
    def test_native_nonnegative_composition_receipt_and_guards(self):
        self._check_native(POSITIVE)

    def _check_native(self, policy):
        import numpy as np
        from atlas_mlir_native.binding import DRAM_BASE
        from atlas_mlir_native.layout import pack_bf16_pair_halves
        from merlin.semantic_compiler.search import SearchLimits
        from test_looped_mnk import ROOT, build_snapshot
        from test_pointwise_loop import check_source_c_call

        source, _, _ = case()
        group = prepare(source, SimpleNamespace(reduction_policy=policy))
        nonnegative = policy == POSITIVE
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = build_snapshot(root)
            compiled = root / 'compiled'
            compile_group(group, snapshot, target_source=ROOT, destination=compiled, limits=SearchLimits())
            binary = (compiled / 'program.bin').read_bytes()
            emitted = json.loads((compiled / 'physical-program.json').read_text())
            self.assertEqual(sum(row['operation'] == 'atlas.trap' for row in emitted['instructions']), 1)
            region = export_group(group, compiled, 'pointwise', target_source=ROOT)
            receipt = region.description
            self.assertEqual(receipt['intermediate_domain_proof']['reduction_policy'], policy)
            self.assertEqual(receipt['precision_plan'][1]['arithmetic'], policy)
            self.assertEqual(receipt['outer_input_conversions']['seed']['policy'], policy)
            package = root / 'package'
            package.mkdir()
            for name, payload in region.files.items():
                path = package / name
                path.parent.mkdir(exist_ok=True, parents=True)
                path.write_bytes(payload)
            # The shared source runtime is an explicit exported dependency.
            with (package / 'pointwise.c').open('a') as handle:
                handle.write('\n#include "runtime/atlas_host.c"\n')
            data = (((np.arange(21) % 7) - 3) / 2).astype('<f4').reshape(3, 7)
            data[0, :2] = [0.0, -0.0]
            seed = np.array([0.5, 0.0 if nonnegative else -0.0, 1.0], dtype='<f4')
            if nonnegative:
                # Both squares leave the default reduction domain.
                data[1, :2] = [2.0**-40, 2.0**40]
            squared = (data * data).astype('<f4')
            wanted = (squared.sum(axis=1) + seed).astype('<f4')
            def packed(matrix):
                padded = np.zeros((32, 32), dtype='<u2')
                padded[:matrix.shape[0], :matrix.shape[1]] = matrix
                return pack_bf16_pair_halves(padded.tobytes())
            data_codes = np.frombuffer(input_spec(reduction_policy=policy).encode(data), dtype='<u2').reshape(3, 7)
            seed_codes = np.frombuffer(reduction_spec(reduction_policy=policy).encode(seed), dtype='<u2').reshape(3, 1)
            square_codes = (squared.view('<u4') >> 16).astype('<u2')
            sums = (squared.sum(axis=1).view('<u4') >> 16).astype('<u2')
            result_tile = np.zeros((32, 32), dtype='<u2')
            result_tile[:3, :] = sums[:, None]
            result_tile[:3, 0] = (wanted.view('<u4') >> 16).astype('<u2')
            initial = bytearray([0xa5]) * (group['arena_bytes'] + 2048)
            for key, payload in ((group['data'], packed(data_codes)), (group['seed'], packed(seed_codes))):
                at = group['addresses'][key] - DRAM_BASE
                initial[at:at + len(payload)] = payload
            expected = bytearray(initial)
            for key, payload in ((group['outputs'][0], packed(square_codes)),
                                 (group['outputs'][1], pack_bf16_pair_halves(result_tile.tobytes()))):
                at = group['addresses'][key] - DRAM_BASE
                expected[at:at + len(payload)] = payload

            def execute(memory):
                self.assertEqual(memory, initial)
                if os.environ.get('ATLAS_RUN_NPU_MODEL') == '1':
                    return self._execute_model(compiled, binary, memory, bytes(expected))
                return bytes(expected)

            invalid_small, invalid_large = data.copy(), data.copy()
            invalid_small[0, 1], invalid_large[0, 1] = 2.0**-40, 2.0**40
            invalid = ([invalid_small, seed], [invalid_large, seed])
            if nonnegative:
                # Below the square input domain; FP32 tree overflow; a negative seed.
                invalid_small[0, 1], invalid_large[0, :4] = 2.0**-70, 2.0**63
                invalid = (*invalid, [data, -seed])
            check_source_c_call(self, package, initial, binary, execute, [data, seed], wanted.tobytes(),
                                invalid_values=invalid)

    def _execute_model(self, compiled, binary, memory, expected):
        from test_looped_mnk import ROOT
        input_path, output_path = compiled.parent / 'model-input.bin', compiled.parent / 'model-output.bin'
        input_path.write_bytes(memory)
        script = r"""
import json,sys
from pathlib import Path
from dataclasses import replace
import torch
from npu_model.configs import isa_definition as isa
from npu_model.configs.hardware.default import DefaultHardwareConfig
from npu_model.logging import LoggerConfig
from npu_model.simulation import Simulation
from npu_model.software.program import InstantiableProgram
from tools.check_program_decode import decode_binary
from tools.model_instructions import model_instruction
root,compiled,input_path,output_path=map(Path,sys.argv[1:])
torch.set_num_threads(1)
inventory=json.loads((root/'docs/selected-variant-inventory.json').read_text())
program=InstantiableProgram([model_instruction(row,inventory,isa) for row in
    decode_binary(inventory,(compiled/'program.bin').read_bytes())['instructions']])
program.dram_base=0x90000000
memory=input_path.read_bytes()
program.memory_regions=[(0,torch.tensor(list(memory),dtype=torch.uint8))]
config=DefaultHardwareConfig()
config.arch_state_config=replace(config.arch_state_config,numerics='rtl',randomize_init=True,init_seed=219)
sim=Simulation(config,LoggerConfig(filename=str(compiled.parent/'model.json')),program,verbose=False)
state=sim.core.arch_state
try:
    state.vmem.fill_(0xcd)
    sim.run(max_cycles=500000)
    assert state.halted and sim.get_stats().runtime_errors==0
    output_path.write_bytes(bytes(state.read_dram(program.dram_base,len(memory)).tolist()))
finally:
    state.close()
"""
        import sys
        process = subprocess.run([os.environ.get('IMPORT_MODEL_PYTHON', sys.executable), '-c', script,
                                  str(ROOT), str(compiled), str(input_path), str(output_path)],
                                 text=True, capture_output=True)
        self.assertEqual(process.returncode, 0, process.stderr)
        actual = output_path.read_bytes()
        self.assertEqual(actual, expected)
        return actual


if __name__ == '__main__':
    unittest.main()
