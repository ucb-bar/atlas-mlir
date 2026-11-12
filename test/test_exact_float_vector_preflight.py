"""A vector sufficient-domain proof retains scalar fallback and atomic rejection."""
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from atlas_mlir_native import exact_float_device as device
from atlas_mlir_native.float_virtual import binary_vector_preflight
from merlin.semantic_compiler.codegen import CodegenOptions, codegen_options
from merlin.semantic_compiler.target_binding import NativeCompilationError
from test_exact_float_device import host_reference, request

ROOT = Path(__file__).resolve().parents[1]


class VectorPreflightTest(unittest.TestCase):
    def test_option_is_explicit_and_validated(self):
        with patch.dict(os.environ, ATLAS_EXACT_BINARY_VECTOR_PREFLIGHT='0'):
            self.assertFalse(binary_vector_preflight())
            ordinary = device.virtual_program(device.prepare_plan(request('bf16_add', 1024)))
        with patch.dict(os.environ, ATLAS_EXACT_BINARY_VECTOR_PREFLIGHT='1'):
            self.assertTrue(binary_vector_preflight())
            selected = device.virtual_program(device.prepare_plan(request('bf16_add', 1024)))
        self.assertNotIn('scalar_qualified_check', ordinary)
        self.assertIn('scalar_qualified_check', selected)
        with patch.dict(os.environ, ATLAS_EXACT_BINARY_VECTOR_PREFLIGHT='true'), \
                self.assertRaises(NativeCompilationError):
            binary_vector_preflight()

    def test_every_input_encoding_has_exact_stronger_domain(self):
        import torch
        from npu_model.configs.numerics import RtlNumerics
        # Arithmetic oracle only: no Python instruction simulation. Ordered-bit
        # extrema preserve these bounds for every arrangement of lanes/operands.
        raw = torch.arange(65536, dtype=torch.int32)
        values = raw.to(torch.int16).view(torch.bfloat16)
        absolute = RtlNumerics.maximum(values,
            RtlNumerics.mul(values, torch.full_like(values, -1)))
        bits = absolute.view(torch.int16).to(torch.int32) & 65535
        actual = (bits >= 0x2f80) & (bits <= 0x4f80)
        expected = ((raw & 32767) >= 0x2f80) & ((raw & 32767) <= 0x4f80)
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(int(actual.sum()), 16386)
        # The same ordered absolute value admits every non-NaN encoding for
        # exact widening, including signed zeros, subnormals and infinities.
        self.assertTrue(torch.equal(bits <= 0x7f80, (raw & 32767) <= 0x7f80))

    def test_widening_option_is_explicit_and_validated(self):
        from atlas_mlir_native.float_virtual import cast_vector_preflight
        source = device.prepare_plan(request('bf16_to_f32', 35))
        with patch.dict(os.environ, ATLAS_EXACT_CAST_VECTOR_PREFLIGHT='0'):
            self.assertFalse(cast_vector_preflight())
            ordinary = device.virtual_program(source)
        with patch.dict(os.environ, ATLAS_EXACT_CAST_VECTOR_PREFLIGHT='1'):
            self.assertTrue(cast_vector_preflight())
            selected = device.virtual_program(source)
        self.assertNotIn('col_max', ordinary)
        self.assertIn('col_max', selected)
        with patch.dict(os.environ, ATLAS_EXACT_CAST_VECTOR_PREFLIGHT='true'), \
                self.assertRaises(NativeCompilationError):
            cast_vector_preflight()

    @unittest.skipUnless(os.environ.get('ATLAS_RUST_NPU_MODEL') and os.environ.get('ATLAS_OOT_BIN_DIR'),
                         'select Rust simulator and Atlas tools')
    def test_rust_widening_every_non_nan_code_and_atomic_nan_rejection(self):
        import numpy as np
        runner = os.environ['ATLAS_RUST_NPU_MODEL']
        codes = np.arange(65536, dtype='<u2')
        codes = codes[(codes & 0x7fff) <= 0x7f80]
        np.random.default_rng(1701).shuffle(codes)
        with tempfile.TemporaryDirectory() as directory, codegen_options(CodegenOptions(
                io_memory_bytes=67108864, execution_instruction_limit=128000000,
                exact_float_unroll=True)):
            root = Path(directory)
            for count in (35, 512, 1025, len(codes)):
                with self.subTest(count=count):
                    source = request('bf16_to_f32', count)
                    plan = device.prepare_plan(source)
                    programs = []
                    for enabled in ('0', '1'):
                        compiled = root / f'widen-{count}-{enabled}'
                        with patch.dict(os.environ, ATLAS_EXACT_CAST_VECTOR_PREFLIGHT=enabled):
                            device.compile_source(None, source, target_source=ROOT,
                                destination=compiled, limits=None)
                        programs.append(compiled / 'program.bin')
                    raw = codes[:count].tobytes()
                    self.compare_runs(root, runner, source, plan, programs, [raw],
                                      cycle_ratio=0.8 if count >= 512 else None)
                    for invalid in (0x7f81, 0x7fff, 0xff81, 0xffff):
                        self.compare_runs(root, runner, source, plan, programs,
                            [raw[:-2] + invalid.to_bytes(2, 'little')])

    @unittest.skipUnless(os.environ.get('ATLAS_RUST_NPU_MODEL') and os.environ.get('ATLAS_OOT_BIN_DIR'),
                         'select Rust simulator and Atlas tools')
    def test_rust_execution_covers_fast_fallback_tail_and_late_rejection(self):
        runner = os.environ['ATLAS_RUST_NPU_MODEL']
        with tempfile.TemporaryDirectory() as directory, codegen_options(CodegenOptions(
                io_memory_bytes=67108864, execution_instruction_limit=4000000)):
            root = Path(directory)
            for operation in ('bf16_add', 'bf16_sub'):
                for count in (35, 1024, 2048):
                    with self.subTest(operation=operation, count=count):
                        source = request(operation, count)
                        plan = device.prepare_plan(source)
                        programs = []
                        for enabled in ('0', '1'):
                            compiled = root / f'{operation}-{count}-{enabled}'
                            with patch.dict(os.environ, ATLAS_EXACT_BINARY_VECTOR_PREFLIGHT=enabled):
                                device.compile_source(None, source, target_source=ROOT,
                                    destination=compiled, limits=None)
                            programs.append(compiled / 'program.bin')
                        pairs = [(0x3f80, 0x3f00), (0x8000, 0x8000),
                                 (0x2000, 0x1c00), (1, 127), (0x2f80, 0xcf80),
                                 (0x7f80, 0x3f80), (0x7f7f, 0x7f7f)]
                        for index, pair in enumerate(pairs):
                            raw = [word.to_bytes(2, 'little') * count for word in pair]
                            if count == 2048:
                                raw = [bytes.fromhex('803f') * 1024 + data[2048:] for data in raw]
                            self.compare_runs(root, runner, source, plan, programs, raw,
                                              cycle_ratio=0.5 if index == 0 and count == 1024 else None)
                        for operand in (0, 1):
                            raw = [bytes.fromhex('803f') * count, bytes.fromhex('003f') * count]
                            raw[operand] = raw[operand][:-2] + bytes.fromhex('c1ff')
                            self.compare_runs(root, runner, source, plan, programs, raw)

    def compare_runs(self, root, runner, source, plan, programs, raw, *, cycle_ratio=None):
        expected, status = host_reference(source, raw)
        initial = bytearray([0xa5]) * ((plan.workspace_bytes + 4095) // 4096 * 4096)
        for index, data in enumerate(raw):
            at = index * plan.input_bytes
            initial[at:at + plan.input_bytes] = bytes(plan.input_bytes)
            initial[at:at + len(data)] = data
        cycles = []
        for program in programs:
            with tempfile.TemporaryDirectory(dir=root) as work:
                work = Path(work)
                (work / 'input.bin').write_bytes(initial)
                result = subprocess.run([runner, str(program), '--compute', 'cpu',
                    '--load', f'{plan.io_memory_base}:{work / "input.bin"}',
                    '--dump-state', str(work / 'state'), '--max-cycles', '10000000'],
                    check=True, capture_output=True, text=True)
                cycles.append(json.loads(result.stdout)['cycles'])
                memory = b''.join((work / 'state' / f'dram_{a:016x}.bin').read_bytes()
                    for a in range(plan.io_memory_base, plan.io_memory_base + len(initial), 4096))
                output = plan.output_address - plan.io_memory_base
                code = plan.status_address - plan.io_memory_base
                self.assertEqual(bool(int.from_bytes(memory[code:code + 4], 'little')), bool(status))
                self.assertEqual(memory[output:output + len(expected)], expected)
                self.assertEqual(memory[:output], initial[:output])
                self.assertEqual(memory[code + 32:], initial[code + 32:])
                if status:
                    self.assertEqual(memory[output:code], initial[output:code])
        if cycle_ratio is not None:
            self.assertLess(cycles[1], cycles[0] * cycle_ratio)


if __name__ == '__main__':
    unittest.main()
