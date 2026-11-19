"""A vector sufficient-domain proof retains scalar fallback and atomic rejection."""
from dataclasses import replace
import unittest
from pathlib import Path

from atlas_mlir_native.options import codegen_options, current_codegen_options
from atlas_mlir_native import exact_float_device as device
from atlas_mlir_native.float_virtual import binary_vector_preflight

from test_exact_float_device import request

ROOT = Path(__file__).resolve().parents[1]


class VectorPreflightTest(unittest.TestCase):
    def test_option_is_explicit_and_validated(self):
        with codegen_options(replace(current_codegen_options(), exact_binary_vector_preflight=False)):
            self.assertFalse(binary_vector_preflight())
            ordinary = device.virtual_program(device.prepare_plan(request('bf16_add', 1024)))
        with codegen_options(replace(current_codegen_options(), exact_binary_vector_preflight=True)):
            self.assertTrue(binary_vector_preflight())
            selected = device.virtual_program(device.prepare_plan(request('bf16_add', 1024)))
        self.assertNotIn('scalar_qualified_check', ordinary)
        self.assertIn('scalar_qualified_check', selected)

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
        with codegen_options(replace(current_codegen_options(), exact_cast_vector_preflight=False)):
            self.assertFalse(cast_vector_preflight())
            ordinary = device.virtual_program(source)
        with codegen_options(replace(current_codegen_options(), exact_cast_vector_preflight=True)):
            self.assertTrue(cast_vector_preflight())
            selected = device.virtual_program(source)
        self.assertNotIn('col_max', ordinary)
        self.assertIn('col_max', selected)

    def test_zero_probe_distinguishes_every_bf16_encoding(self):
        import torch
        from npu_model.configs.numerics import RtlNumerics as rtl

        raw = torch.arange(65536, dtype=torch.int32)
        values = raw.to(torch.int16).view(torch.bfloat16)
        magnitude = rtl.maximum(values, rtl.mul(values, torch.full_like(values, -1)))
        scaled = rtl.mul(rtl.minimum(magnitude, torch.full_like(values, 2.0 ** -32)),
                         torch.full_like(values, 2.0 ** 32))
        probe = rtl.mul(scaled, rtl.sub(torch.ones_like(values), scaled))
        probe_bits = probe.view(torch.int16).to(torch.int32) & 65535
        magnitude_bits = magnitude.view(torch.int16).to(torch.int32) & 65535
        admitted = (probe_bits == 0) & (magnitude_bits <= 0x4f80)
        original = raw & 32767
        expected = (original == 0) | ((original >= 0x2f80) & (original <= 0x4f80))
        self.assertTrue(torch.equal(admitted, expected))
        self.assertEqual(int(admitted.sum()), 16388)
        # Includes the smallest subnormal: scaling makes every small nonzero
        # input's probe normal, preventing silent acceptance after underflow.
        small = (original > 0) & (original < 0x2f80)
        self.assertGreaterEqual(int(probe_bits[small].min()), 128)


if __name__ == '__main__':
    unittest.main()
