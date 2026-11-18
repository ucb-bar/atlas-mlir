"""Finite lane-domain checks and explicit source-policy admission for normalization."""

import os
import unittest
from decimal import ROUND_HALF_UP, Decimal, localcontext


class BoundedPointwiseMathTest(unittest.TestCase):
    def test_entire_positive_domain_matches_independent_decimal_lut(self):
        from atlas_mlir_native.recip_lane import (
            vrecip_bf16_bits,
            vrecip_bf16_bounded_bits,
        )
        from atlas_mlir_native.sqrt_lane import vsqrt_bf16_bits
        from test_softmax_primitives import _reciprocal

        with localcontext() as context:
            context.prec = 80
            entries = [
                int(((Decimal(128 + fraction) / 128).sqrt() * 65536).to_integral_value(rounding=ROUND_HALF_UP))
                for fraction in range(128)
            ]
            root_two = int((Decimal(2).sqrt() * 65536).to_integral_value(rounding=ROUND_HALF_UP))
        checked = 0
        for code in range(0x2000, 0x5F01):
            self.assertEqual(vrecip_bf16_bounded_bits(code), _reciprocal(code))
            exponent, fraction = divmod(code, 128)
            fixed = entries[fraction] if exponent & 1 else entries[fraction] * root_two // 65536
            expected = ((127 + (exponent - 127) // 2) << 7) | (fixed // 512 % 128)
            self.assertEqual(vsqrt_bf16_bits(code), expected)
            checked += 1
        self.assertEqual(checked, 16129)
        for code in (0, 0x8000, 0x1FFF, 0x5F01, 0xBF80, 0x7F80, 0x7FC0):
            for operation in (vrecip_bf16_bounded_bits, vsqrt_bf16_bits):
                with (
                    self.subTest(code=code, operation=operation),
                    self.assertRaises(ValueError),
                ):
                    operation(code)
        with self.assertRaises(ValueError):
            vrecip_bf16_bits(0x3F00, wide=True)

    def test_source_conversions_preserve_the_positive_input_requirement(self):
        import numpy as np
        from atlas_mlir_native.pointwise_math import input_specs

        numerator, denominator = input_specs("divide_bf16_lut")
        with self.assertRaises(ValueError):
            input_specs("unknown")
        numerator.encode(np.array([0.0, -0.0, -(2.0**63)], dtype=np.float32))
        denominator.encode(np.array([2.0**-63, 1.0, 2.0**63], dtype=np.float32))
        for value in (0.0, -0.0, -1.0, 2.0**-64, 2.0**64, np.inf, np.nan):
            with self.subTest(value=value), self.assertRaises(ValueError):
                denominator.encode(np.array([value], dtype=np.float32))

    @unittest.skipUnless(os.environ.get("ATLAS_RUN_NPU_MODEL") == "1", "select npu_model lane tables")
    def test_selected_model_tables_cover_the_entire_bounded_domain(self):
        import torch
        from atlas_mlir_native.recip_lane import vrecip_bf16_bounded_bits
        from atlas_mlir_native.sqrt_lane import vsqrt_bf16_bits
        from npu_model.configs.numerics import RtlNumerics

        codes = list(range(0x2000, 0x5F01))
        values = torch.tensor(codes, dtype=torch.int16).view(torch.bfloat16)
        for mode, reference in (
            ("rcp", vrecip_bf16_bounded_bits),
            ("sqrt", vsqrt_bf16_bits),
        ):
            actual = RtlNumerics.unary(mode, values).view(torch.int16).tolist()
            self.assertEqual(actual, [reference(code) for code in codes])


if __name__ == "__main__":
    unittest.main()
