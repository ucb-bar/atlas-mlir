"""Precision records used by the refactor Atlas lowering retain selected scales."""
from dataclasses import replace

import numpy as np
import pytest

from merlin.compiler.quantization.contraction.values import FloatBoundarySpec


def test_column_scales_preserve_raw_codes_and_decode_units():
    encode = FloatBoundarySpec('column-test', 'fp8_e4m3', 'encode',
                               column_scale_exponents=((0, 1, 0), (1, 2, 1)))
    source = np.ones((2, 2), dtype=np.float32)
    codes = encode.encode(source)
    assert codes.tolist() == [[0x38, 0x40], [0x38, 0x40]]
    decode = FloatBoundarySpec('restore-test', 'fp8_e4m3', 'decode',
                               column_scale_exponents=((0, 1, 0), (1, 2, -1)))
    np.testing.assert_array_equal(decode.decode(codes), source)
    with pytest.raises(ValueError, match='complete tensor shape'):
        encode.encode(np.ones((2, 3), dtype=np.float32))
    with pytest.raises(ValueError, match='contiguous partition'):
        replace(encode, column_scale_exponents=((0, 1, 0), (2, 3, 1)))
    with pytest.raises(ValueError, match='scalar or column'):
        replace(encode, pre_encode_scale_exponent=1)


def test_selected_underflow_rounding_keeps_normal_threshold_tie():
    before = FloatBoundarySpec('flush-test', 'fp8_e4m3', 'encode', flush_below_normal=True)
    rounded = replace(before, underflow_rounding='round-normal-significand-then-flush')
    threshold = np.float32((2.0**-6) * (1.0 - 2.0**-5))
    values = np.array([np.nextafter(threshold, np.float32(0)), threshold,
                       -threshold, np.float32(2.0**-6)], dtype=np.float32)
    assert before.encode(values).tolist() == [0, 0, 0x80, 8]
    assert rounded.encode(values).tolist() == [0, 8, 0x88, 8]
    with pytest.raises(ValueError, match='flushing encode'):
        replace(rounded, flush_below_normal=False)
