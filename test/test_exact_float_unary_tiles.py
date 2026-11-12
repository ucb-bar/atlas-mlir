"""Independent bit-level and publication checks for fixed-offset unary tiles."""
import random
import unittest
import json
import subprocess

from test_exact_float_device import BIN, host_reference, request, scalar_execute


class UnaryTileTest(unittest.TestCase):
    def setUp(self):
        from merlin.semantic_compiler.codegen import CodegenOptions, codegen_options
        from float_isa_c import executor
        executor.cache_clear()
        options = codegen_options(CodegenOptions(exact_float_unroll=True))
        options.__enter__()
        self.addCleanup(options.__exit__, None, None, None)
        self.addCleanup(executor.cache_clear)

    def test_expansion_requires_explicit_option_independently_of_image_cap(self):
        from atlas_mlir_native.exact_float_device import prepare_plan, virtual_program
        from merlin.semantic_compiler.codegen import CodegenOptions, codegen_options
        for operation in ('bf16_to_f32', 'f32_to_bf16', 'bf16_neg', 'f32_neg'):
            plan = prepare_plan(request(operation, 35))
            for cap in (None, 131072):
                with codegen_options(CodegenOptions(program_size_limit_bytes=cap)):
                    self.assertIn('^scalar_lane(', virtual_program(plan))
                with codegen_options(CodegenOptions(program_size_limit_bytes=cap, exact_float_unroll=True)):
                    self.assertNotIn('^scalar_lane(', virtual_program(plan))

    def test_explicit_image_cap_retains_compact_counted_kernels(self):
        from atlas_mlir_native.exact_float_device import prepare_plan, machine_program, virtual_program
        from merlin.semantic_compiler.codegen import CodegenOptions, codegen_options
        with codegen_options(CodegenOptions(program_size_limit_bytes=4096)):
            for operation in ('bf16_to_f32', 'f32_to_bf16', 'bf16_neg', 'f32_neg'):
                plan = prepare_plan(request(operation, 2048))
                self.assertIn('^scalar_lane(', virtual_program(plan))
                machine = machine_program(plan)
                emitted = subprocess.run([str(BIN/'atlas-emit'), '--program-json', '-'],
                                         input=machine, text=True, capture_output=True, check=True)
                self.assertLessEqual(len(json.loads(emitted.stdout)['instructions']) * 4, 4096)

    def compare(self, operation, values, width):
        source = request(operation, len(values))
        raw = [b''.join(value.to_bytes(width, 'little') for value in values)]
        self.assertEqual(scalar_execute(source, raw)[:2], host_reference(source, raw))

    def test_every_bf16_encoding_for_negation_and_every_admitted_widening(self):
        for operation in ('bf16_neg', 'bf16_to_f32'):
            values = [value for value in range(65536)
                      if operation == 'bf16_neg' or value & 32767 <= 32640]
            for start in range(0, len(values), 2048):
                self.compare(operation, values[start:start + 2048], 2)

    def test_f32_rounding_ties_signs_exponents_and_random_payloads(self):
        rng = random.Random(9031)
        values = [(sign << 31) | (exponent << 23) | (mantissa << 16) | low
                  for sign in (0, 1) for exponent in range(255)
                  for mantissa in (0, 1, 126, 127)
                  for low in (0, 0x7fff, 0x8000, 0x8001, 0xffff)]
        values += [rng.randrange(1 << 32) for _ in range(8192)]
        for operation in ('f32_neg', 'f32_to_bf16'):
            admitted = [value for value in values
                        if operation == 'f32_neg' or value & 0x7fffffff <= 0x7f800000]
            for start in range(0, len(admitted), 2048):
                self.compare(operation, admitted[start:start + 2048], 4)

    def test_tails_and_invalid_values_across_tile_and_leaf_boundaries(self):
        from atlas_mlir_native.exact_float_device import MAX_LEAF_ELEMENTS
        for operation, width, invalid in (('bf16_to_f32', 2, 0x7f81),
                                          ('f32_to_bf16', 4, 0x7f800001)):
            for count in (1, 31, 32, 33, 511, 512, 513, 2047, 2048,
                          MAX_LEAF_ELEMENTS[operation] - 1, MAX_LEAF_ELEMENTS[operation]):
                self.compare(operation, [1] * count, width)
                for at in sorted({0, count - 1, min(512, count - 1)}):
                    values = [1] * count
                    values[at] = invalid
                    source = request(operation, count)
                    raw = [b''.join(value.to_bytes(width, 'little') for value in values)]
                    output, status, _ = scalar_execute(source, raw)
                    self.assertNotEqual(status, 0)
                    self.assertEqual(output, bytes([0xa5]) * len(output))

    def test_current_maximum_leaves_have_checked_instruction_bounds(self):
        import tempfile
        from pathlib import Path
        from atlas_mlir_native.exact_float_device import MAX_LEAF_ELEMENTS, compile_source
        from merlin.semantic_compiler.codegen import CodegenOptions, codegen_options
        from test_exact_float_device import ROOT

        for operation in ('bf16_to_f32', 'f32_to_bf16', 'bf16_neg', 'f32_neg'):
            with tempfile.TemporaryDirectory(prefix='atlas-expanded-unary-') as directory, \
                    codegen_options(CodegenOptions(io_memory_bytes=67108864, exact_float_unroll=True)):
                destination = Path(directory) / 'compiled'
                source = request(operation, MAX_LEAF_ELEMENTS[operation])
                manifest = compile_source(None, source, target_source=ROOT,
                                          destination=destination, limits=None)
                bound = json.loads((destination / 'float-instruction-bound.json').read_text())
                self.assertTrue(bound['admitted'])
                self.assertLessEqual(bound['instruction_bound'], 1_000_000)
                self.assertLessEqual(manifest['program_bytes'], 32768 * 4)


if __name__ == '__main__':
    unittest.main()
