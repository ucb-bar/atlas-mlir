"""K32 partial-domain proof, including normalized short sums.

The finite enumeration covers every BF16 seed magnitude in the inductive
interval and every attainable anchor.  Product sums are overapproximated by
their integer lattice, so it also covers cancellation not represented by a
particular 32-product fixture.  See docs/already-quantized-k32-domain.md.
"""
from fractions import Fraction
from math import gcd
import unittest

import numpy as np

from atlas_mlir_native.already_quantized import valid_codes
from atlas_mlir_native.reference import mxu1_cap448_lane


ADMITTED_CODES = (0, *range(8, 127), *range(136, 255))
MIN_EXPONENT = -26
MAX_EXPONENT = 32


def seed_anchor_cases():
    """All positive BF16 seeds in [2^-26, 2^32], plus reset, and anchors.

    Sign changes only the sign of an aligned seed and the residue distance.
    The upper endpoint 2^32 is included; larger mantissas at that exponent
    are outside the inductive interval.  A nonzero product's exponent is in
    [-12,17]; a zero-only chunk uses the seed exponent alone.
    """
    for anchor in range(-5, 25):
        yield 0, anchor
    for exponent in range(MIN_EXPONENT, MAX_EXPONENT + 1):
        for fraction in range(1 if exponent == MAX_EXPONENT else 128):
            seed = ((exponent + 127) << 7) | fraction
            anchors = {exponent + 7}
            anchors.update(range(max(exponent, -12) + 7,
                                 max(exponent, 17) + 8))
            for anchor in sorted(anchors):
                yield seed, anchor


def minimum_lattice_residue(seed, anchor):
    """Return the least positive overapproximated total, in anchor units.

    q=2^(anchor-31). Products are multiples of g=2^-18 before alignment.
    After truncation their sum is a multiple of max(g,q). The distance of
    the truncated seed to this lattice bounds every possible cancellation.
    """
    exponent = ((seed >> 7) & 255) - 127
    shift = anchor - exponent - 24
    significand = (128 | (seed & 127)) if seed else 0
    aligned = significand >> shift if shift >= 0 else significand << -shift
    product_stride = 1 << max(0, 13 - anchor)
    remainder = aligned % product_stride
    return min(remainder, product_stride - remainder) if remainder else product_stride


def cancellation_chain():
    """Cancel short sums through 2^-18 and zero from reset using admitted bytes and BF16 reseeds only."""
    panels = [
        ((0x50, 0xd0, 0x09, 0x88), (0x50, 0x50, 0x09, 0x0a)),
        ((0x89, 0x08), (0x09, 0x0a)),
        ((0x89, 0x08), (0x09, 0x0a)),
        ((0x30, 0xb0), (0x38, 0x38)),
        ((0x09, 0x88), (0x09, 0x0a)),
    ]
    return [(a + (0,) * (32-len(a)), b + (0,) * (32-len(b))) for a, b in panels]


def bf16_fraction(word):
    if not word:
        return Fraction(0)
    exponent = ((word >> 7) & 255) - 127 - 7
    magnitude = Fraction(128 | (word & 127)) * Fraction(2) ** exponent
    return -magnitude if word & 32768 else magnitude


class QuantizedK32DomainTest(unittest.TestCase):
    def test_product_grid_uses_only_admitted_normal_codes_and_positive_zero(self):
        valid_codes(np.array(ADMITTED_CODES, np.uint8))
        excluded = sorted(set(range(256)) - set(ADMITTED_CODES))
        for code in excluded:
            with self.subTest(code=code), self.assertRaises(ValueError):
                valid_codes(np.array([code], np.uint8))
        # Each normal code is an integer in units 2^-9, including exponent15.
        integers = [0 if not code else (-1 if code & 128 else 1) *
                    (8 | (code & 7)) * (1 << (((code >> 3) & 15)-1))
                    for code in ADMITTED_CODES]
        products = [a*b for a in integers for b in integers]
        self.assertEqual(max(abs(p) for p in products), 448**2 * 2**18)
        self.assertEqual(gcd(*products), 1)
        self.assertEqual(len(products), 239**2)

    def test_ideal_bf16_rne_grid_describes_normalized_cancellation(self):
        # Grid values below 2^-11 have at most eight significant bits, so
        # ordinary BF16 RNE keeps them exact. The selected anchor packing
        # normalizes a nonzero integer of fewer than eight bits the same way.
        for numerator in range(129):
            value = Fraction(numerator, 2**18)
            if value:
                lead = numerator.bit_length()-1
                word = ((lead-18+127) << 7) | ((numerator << (7-lead)) & 127)
                self.assertEqual(bf16_fraction(word), value)
        seed = 0
        words = []
        for a, b in cancellation_chain():
            seed = mxu1_cap448_lane(a, b, seed)
            words.append(seed)
        self.assertEqual(words, [0x3680, 0, 0xb680, 0xb680, 0])
        self.assertEqual(bf16_fraction(words[0]), Fraction(1, 2**18))

    def test_every_seed_anchor_lattice_has_no_nonzero_total_below_2neg26(self):
        count = 0
        minimum = None
        for seed, anchor in seed_anchor_cases():
            residue = minimum_lattice_residue(seed, anchor)
            # Exponent of residue*q, without float rounding or huge arrays.
            exponent = residue.bit_length()-1 + anchor-31
            self.assertGreaterEqual(exponent, MIN_EXPONENT, (hex(seed), anchor))
            minimum = exponent if minimum is None else min(minimum, exponent)
            count += 1
        self.assertEqual(minimum, MIN_EXPONENT)
        self.assertEqual(count, 116895)

    def test_integer_anchor_cannot_wrap_and_bf16_upper_boundary_is_closed(self):
        # Seven headroom bits bound each of 32 products and the incoming
        # seed by 255*2^17 in signed integer-anchor units.
        self.assertLess(33 * 255 * 2**17, 2**31)
        contribution = 32 * 448**2
        self.assertLess(contribution, 2**24)  # Half an upward BF16 ULP at 2^32.
        # anchor<=39 gives q<=256. A normalized lead<7 output is strictly
        # below 128*q, hence far below the upper boundary.
        self.assertLess(128 * 256, 2**32)


if __name__ == '__main__':
    unittest.main()
