"""Independent Fraction-oracle controls for the opt-in vector arithmetic."""

from fractions import Fraction
import math
import unittest

try:
    import numpy as np
except ImportError:
    np = None

from atlas_tile_arithmetic import bf16_round, bf16_value, fp8_value, mxu0_dot, rescale_bf16

if np is not None:
    from atlas_vector_arithmetic import decode_bf16, decode_fp8, ordered_matmul, rescale_bf16_array, round_bf16


@unittest.skipUnless(np is not None, "NumPy is required for the optional vector oracle")
class VectorArithmeticTest(unittest.TestCase):
    def test_all_finite_fp8_codes_and_all_pair_products(self):
        codes = np.array([c for c in range(256) if c not in (0x7F, 0xFF)], dtype=np.uint8)
        decoded = decode_fp8(codes)
        self.assertEqual(decoded.tolist(), [float(fp8_value(int(c))) for c in codes])
        actual = ordered_matmul(codes[:, None], codes[None, :])
        expected = np.array([[bf16_round(fp8_value(int(a)) * fp8_value(int(b))) for b in codes] for a in codes],
            dtype=np.uint16)
        np.testing.assert_array_equal(actual, expected)

    def test_random_ordered_mac_chains(self):
        rng = np.random.default_rng(20261018)
        domain = np.array([c for c in range(256) if c not in (0x7F, 0xFF)], dtype=np.uint8)
        for k in (0, 1, 32, 64, 576):
            a = rng.choice(domain, size=(4, k))
            b = rng.choice(domain, size=(k, 5))
            expected = np.array([[mxu0_dot(x, y) for y in b.T] for x in a], dtype=np.uint16)
            with self.subTest(k=k):
                np.testing.assert_array_equal(ordered_matmul(a, b), expected)

    def test_cancellation_zero_and_long_maximum_barrier(self):
        cases = [([0x38, 0xB8], [0x38, 0x38]), ([0x80, 0x87, 0, 7], [0xFE, 0x7E, 0xB8, 0x38]),
            ([0x7E] * 8192, [0x7E] * 8192), ([0xFE] * 8192, [0x7E] * 8192),
            ([0x7E, 0xFE, 0x08] * 192, [0x7E, 0x7E, 0x08] * 192)]
        for a, b in cases:
            expected = mxu0_dot(a, b)
            actual = ordered_matmul(np.array([a], dtype=np.uint8), np.array(b, dtype=np.uint8)[:, None])
            self.assertEqual(int(actual[0, 0]), expected)
        self.assertEqual(mxu0_dot([0x7E] * 8192, [0x7E] * 8192), 0x4C80)  # 2**26.

    def test_all_finite_bf16_decodes(self):
        codes = np.array([c for c in range(65536) if c & 0x7F80 != 0x7F80], dtype=np.uint16)
        expected = np.array([float(bf16_value(int(c))) for c in codes])
        np.testing.assert_array_equal(decode_bf16(codes), expected)
        # The scalar exact-zero convention canonicalizes both signs.
        expected_codes = np.where((codes & 0x7FFF) == 0, 0, codes)
        np.testing.assert_array_equal(round_bf16(expected), expected_codes)

    def test_bf16_ties_neighbors_subnormals_and_saturation(self):
        rng = np.random.default_rng(1718)
        lows = set(map(int, rng.integers(0, 0x7F7F, size=1000)))
        lows.update([0, 1, 2, 0x7E, 0x7F, 0x80, 0xFE, 0xFF, 0x100, 0x3F7F, 0x3F80, 0x7F7E])
        values = [0.0, -0.0, 2.0**128, -(2.0**128), float.fromhex("0x1.fffffffffffffp1023")]
        for low in lows:
            midpoint = float((bf16_value(low) + bf16_value(low + 1)) / 2)
            for x in (math.nextafter(midpoint, -math.inf), midpoint, math.nextafter(midpoint, math.inf)):
                values.extend((x, -x))
        expected = np.array([bf16_round(Fraction.from_float(x)) for x in values], dtype=np.uint16)
        np.testing.assert_array_equal(round_bf16(values), expected)

    def test_rescale_policy_exactness_and_zero_sign(self):
        rng = np.random.default_rng(1818)
        candidates = set(map(int, rng.integers(0, 65536, size=1500)))
        candidates.update([0, 0x8000, 1, 0x8001, 0x7F, 0x807F, 0x80, 0x8080, 0x3F80, 0xBF80, 0x7F7F, 0xFF7F])
        finite = [c for c in candidates if c & 0x7F80 != 0x7F80]
        for exponent in (-126, -10, -1, 0, 1, 10, 127):
            valid, expected, refused = [], [], []
            for code in finite:
                try:
                    target = rescale_bf16(code, exponent)
                except ValueError:
                    refused.append(code)
                else:
                    valid.append(code)
                    expected.append(target)
            with self.subTest(exponent=exponent):
                np.testing.assert_array_equal(rescale_bf16_array(np.array(valid, dtype=np.uint16), exponent),
                    np.array(expected, dtype=np.uint16))
                if refused:
                    with self.assertRaisesRegex(ValueError, "overflows"):
                        rescale_bf16_array(np.array([refused[0]], dtype=np.uint16), exponent)
        for exponent in (-127, 128):
            with self.assertRaises(ValueError):
                rescale_bf16_array(np.array([0x3F80], dtype=np.uint16), exponent)
        np.testing.assert_array_equal(rescale_bf16_array(np.array([0, 0x8000], dtype=np.uint16), 127), [0, 0x8000])

    def test_physical_tile_layout_and_existing_producer_epoch(self):
        from atlas_tile_arithmetic import output_halves
        from atlas_program import code_sources
        from rtl_core.prepare_program import PRODUCER_NAMES

        self.assertEqual(set(code_sources()), set(PRODUCER_NAMES))
        self.assertNotIn("atlas_vector_arithmetic.py", code_sources())
        rng = np.random.default_rng(1819)
        domain = np.array([c for c in range(256) if c not in (0x7F, 0xFF)], dtype=np.uint8)
        a = rng.choice(domain, size=(32, 32))
        b = rng.choice(domain, size=(32, 32))
        raw = np.array([[mxu0_dot(x, y) for y in b.T] for x in a], dtype=np.uint16)
        vector = ordered_matmul(a, b)
        for exponent in (0, -4):
            scalar = np.array([[rescale_bf16(int(code), exponent) for code in row] for row in raw], dtype=np.uint16)
            actual = rescale_bf16_array(vector, exponent)
            np.testing.assert_array_equal(actual, scalar)
            self.assertEqual(output_halves(scalar), output_halves(actual))

    def test_admission(self):
        for bad in (0x7F, 0xFF):
            with self.assertRaises(ValueError):
                ordered_matmul(np.array([[bad]], dtype=np.uint8), np.array([[0]], dtype=np.uint8))
        with self.assertRaises(ValueError):
            ordered_matmul(np.zeros((1, 2), dtype=np.uint8), np.zeros((1, 2), dtype=np.uint8))
        with self.assertRaises(ValueError):
            ordered_matmul(np.zeros((1, 1), dtype=np.int32), np.zeros((1, 1), dtype=np.uint8))
        for bad in (math.inf, -math.inf, math.nan):
            with self.assertRaises(ValueError):
                round_bf16([bad])
        with self.assertRaises(ValueError):
            rescale_bf16_array(np.array([0x7F80], dtype=np.uint16), 0)


if __name__ == "__main__":
    unittest.main()
