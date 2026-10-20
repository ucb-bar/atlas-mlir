"""Finite PACK equivalence against the actual npu_model converters.

ATLAS_PACK_MODEL_ROOT selects the source tree to import. The exhaustive check
covers every finite BF16 encoding (including signed zeros and subnormals) at
every E8M0 byte. ATLAS_PACK_EXACTNESS_RECEIPT optionally records its evidence.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time
import unittest


def pack_oracle(bits: int, scale: int) -> int:
    """Native PACK scalar, independent of clamps and the MXU converter.

    Integer quotient/remainder implements nearest-even. Exponent zero is
    flushed before rounding. The rounded value is flushed below exponent -6;
    there are no E4M3 subnormals. Scale 255 aliases the saturated shift at 254.
    """
    exponent, fraction = (bits >> 7) & 255, bits & 127
    if exponent == 0 or (exponent == 255 and fraction):
        return 0
    sign = (bits >> 8) & 128
    if exponent == 255:
        return sign | 126
    retained, remainder = divmod(128 + fraction, 16)
    retained += remainder > 8 or (remainder == 8 and retained % 2)
    power = exponent - min(scale, 254)
    if retained == 16:
        power += 1
        retained = 8
    if power < -6:
        return 0
    return sign | min(126, (power + 7) * 8 + retained - 8)


def finite_words() -> list[int]:
    return [bits for bits in range(65536) if bits & 0x7f80 != 0x7f80]


class ScalarOracleTest(unittest.TestCase):
    def test_selected_native_reference_all_encodings_and_admitted_scales(self):
        from atlas_mlir_native.pack_lane import pack_bf16_lane
        for scale in range(122, 129):
            for bits in range(65536):
                actual, expected = pack_bf16_lane(bits, scale), pack_oracle(bits, scale)
                if actual != expected:
                    self.fail(f"native scalar disagreement: bits=0x{bits:04x}, scale={scale}, "
                              f"selected=0x{actual:02x}, independent=0x{expected:02x}")

    def test_directed_native_specials_rounding_and_shift_alias(self):
        for bits, expected in ((0, 0), (0x8000, 0), (1, 0), (0x807f, 0),
                               (0x7fc0, 0), (0xffc0, 0), (0x3f88, 0x38),
                               (0x3f89, 0x39), (0x3f98, 0x3a),
                               (0x3c77, 0), (0x3c78, 8),
                               (0x43f0, 0x7e), (0xc3f0, 0xfe),
                               (0x7f80, 0x7e), (0xff80, 0xfe)):
            self.assertEqual(pack_oracle(bits, 127), expected, hex(bits))
        for bits in range(65536):
            self.assertEqual(pack_oracle(bits, 254), pack_oracle(bits, 255))


@unittest.skipUnless(os.environ.get("ATLAS_PACK_MODEL_ROOT"), "select ATLAS_PACK_MODEL_ROOT")
class ActualModelConverterTest(unittest.TestCase):
    def test_every_finite_bf16_word_at_every_scale(self):
        import torch
        model_root = Path(os.environ["ATLAS_PACK_MODEL_ROOT"]).absolute()
        sys.path.insert(0, str(model_root))
        try:
            numerics = importlib.import_module("npu_model.configs.numerics")
            rtl = importlib.import_module("npu_model.configs.rtl_math")
        finally:
            sys.path.pop(0)
        self.assertEqual(Path(rtl.__file__).absolute().parent.parent.parent, model_root,
                         "cached npu_model import differs from selected source")
        torch.set_num_threads(1)
        words = finite_words()
        self.assertEqual(len(words), 65280)
        tensor = torch.tensor(words, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
        def raw_bytes(value):
            return bytes(value.contiguous().view(torch.uint8).tolist())
        def splat(bits):
            return torch.tensor([bits], dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
        def check(actual, expected, scale, name):
            if actual != expected:
                index = next(i for i, (a, b) in enumerate(zip(actual, expected)) if a != b)
                self.fail(f"{name}: bits=0x{words[index]:04x}, scale={scale}, "
                          f"actual=0x{actual[index]:02x}, native=0x{expected[index]:02x}")
        began = time.monotonic()
        rows = []
        for scale in range(256):
            expected = bytes(pack_oracle(bits, scale) for bits in words)
            # Call both public model converter entry points. The equivalence
            # candidate uses the actual model raw min/max datapath as well.
            native = raw_bytes(numerics.RtlNumerics.to_fp8(tensor, scale))
            check(native, expected, scale, "actual native converter")
            guarded = tensor
            if scale <= 246:
                cap = (scale + 8) << 7 | 96
                guarded = rtl.minmax(guarded, splat(cap | 0x8000), True)
                guarded = rtl.minmax(guarded, splat(cap), False)
            inverse = 254 - min(scale, 254)
            actual = raw_bytes(numerics.RtlNumerics.acc_to_fp8(guarded, inverse))
            check(actual, expected, scale, "actual clamp/MXU converter")
            rows.append(dict(scale=scale, words=len(words),
                             output_sha256=hashlib.sha256(actual).hexdigest()))
        receipt = dict(status="PASS", finite_bf16_encodings=len(words), scales=256,
                       finite_word_scale_pairs=len(words) * 256,
                       model_comparisons=2 * len(words) * 256,
                       elapsed_seconds=time.monotonic() - began,
                       model_root=str(model_root), converters={}, per_scale=rows)
        for module in (numerics, rtl):
            path = Path(module.__file__)
            receipt["converters"][path.name] = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        if os.environ.get("ATLAS_PACK_EXACTNESS_RECEIPT"):
            Path(os.environ["ATLAS_PACK_EXACTNESS_RECEIPT"]).write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == '__main__':
    unittest.main()
