"""Small arithmetic controls, independent of model downloads and long runs."""

from fractions import Fraction
import importlib.util
import json
import os
from pathlib import Path
import random
import shutil
import struct
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import atlas_tile_arithmetic as oracle

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from precision_modes import arithmetic as arithmetic


def bits(value):
    return struct.unpack("<I", struct.pack("<f", float(value)))[0] >> 16


class PrecisionModesCliTest(unittest.TestCase):
    def test_help_without_optional_model_dependencies(self):
        root = Path(__file__).resolve().parents[1]
        entries = ([str(root / "tools/precision_modes/cli.py")],
                   ["-m", "tools.precision_modes"])
        for entry in entries:
            result = subprocess.run([sys.executable, "-S", *entry, "--help"],
                                    cwd=root, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("{tiny,smol,resnet}", result.stdout)


@unittest.skipUnless(HAS_TORCH and shutil.which("c++"),
                     "precision tests require optional Torch and C++")
class PrecisionModesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temporary = tempfile.TemporaryDirectory()
        library = Path(cls.temporary.name) / "ordered_mac.so"
        source = (Path(__file__).resolve().parents[1] /
                  "tools/precision_modes/ordered_mac.cpp")
        subprocess.run(["c++", "-O3", "-shared", "-fPIC", str(source), "-o",
                        str(library)], check=True)
        os.environ["ATLAS_PRECISION_MAC_LIBRARY"] = str(library)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def assert_bits(self, actual, expected):
        self.assertEqual([bits(x) for x in actual.flatten()], expected)

    def reference(self, a, b, ea, eb, chunk=0, low=None):
        """Fraction products, Fraction BF16 stores, then one exact final sum."""
        result = []
        for row in range(a.shape[0]):
            for col in range(b.shape[1]):
                total = Fraction(0)
                for start in range(0, a.shape[1], chunk or a.shape[1]):
                    end = min(start + (chunk or a.shape[1]), a.shape[1])
                    branches = [(b, eb)] if low is None else [(b, eb), low]
                    for weight, exponent in branches:
                        accumulator = 0
                        for k in range(start, end):
                            exact = (Fraction(float(a[row, k])) *
                                     Fraction(float(weight[k, col])))
                            accumulator = oracle.bf16_round(
                                oracle.bf16_value(accumulator) + exact)
                        shape = (a.shape[0], b.shape[1])
                        e = torch.broadcast_to(ea + exponent, shape)[row, col]
                        total += (oracle.bf16_value(accumulator) *
                                  oracle.power2(int(e)))
                result.append(oracle.bf16_round(total))
        return result

    def test_codec_matches_existing_scalar_oracle(self):
        rng = random.Random(947)
        words = [0, 0x8000, 1, 0x7f, 0x80, 0x3f88, 0x3f98, 0x43e0,
                 0x43ef, 0x7f7f] + [rng.randrange(0x7f80) for _ in range(256)]
        front = torch.tensor(words, dtype=torch.int32).to(torch.int16)
        front = front.view(torch.bfloat16)
        for exponent in (-20, 0, 20):
            values = arithmetic.quantize(front, exponent)
            expected = [float(oracle.fp8_value(oracle.pack_bf16(
                w, exponent + 127))) for w in words]
            self.assertEqual(values.tolist(), expected)

    def test_scales_and_default_order_against_fraction(self):
        rng = torch.Generator().manual_seed(733)
        left = torch.randn((2, 65), generator=rng)
        right = torch.randn((65, 3), generator=rng)
        a, ea = arithmetic.pack(left)
        b, eb = arithmetic.pack(right)
        for source, exponent in ((left, ea), (right, eb)):
            words = source.bfloat16().contiguous().view(torch.int16).int()
            expected = oracle.tensor_scale_exponent((words & 65535)
                                                    .flatten().tolist())
            self.assertEqual(int(exponent), expected)
        self.assert_bits(arithmetic.contraction(a, b, ea, eb),
                         self.reference(a, b, ea, eb))
        # Ascending K and a BF16 store per MAC discriminate a BLAS sum.
        a = torch.ones((1, 3), dtype=torch.float64)
        b = torch.tensor([[256.0], [1.0], [1.0]], dtype=torch.float64)
        self.assertEqual(float(arithmetic.ordered_mm(a, b)), 256.0)
        self.assertEqual(float(a @ b), 258.0)

    def test_finer_scales_small_contraction(self):
        left = torch.tensor([[448., 2**-7, 1.], [2**-12, 2**-11, 2**-10]])
        right = torch.tensor([[1., 2**-12], [2., 2**-10], [-1., 2**-11]])
        a, ea = arithmetic.pack(left, "row")
        b, eb = arithmetic.pack(right, "column")
        self.assertNotEqual(int(ea[0]), int(ea[1]))
        self.assertNotEqual(int(eb[0, 0]), int(eb[0, 1]))
        self.assert_bits(arithmetic.contraction(a, b, ea, eb),
                         self.reference(a, b, ea, eb))

    def test_k_split_small_contraction(self):
        a = torch.ones((1, 65), dtype=torch.float64)
        b = torch.ones((65, 1), dtype=torch.float64)
        b[0] = 256
        exponent = torch.tensor(0, dtype=torch.int32)
        actual = arithmetic.contraction(a, b, exponent, exponent, k_chunk=32)
        expected = self.reference(a, b, exponent, exponent, chunk=32)
        self.assert_bits(actual, expected)
        unsplit = arithmetic.contraction(a, b, exponent, exponent)
        self.assertNotEqual(float(actual), float(unsplit))

    def test_two_term_small_contraction(self):
        source = torch.tensor([[1.03, -0.51], [0.17, 2.09], [-0.33, 0.79]])
        (b, eb), low = arithmetic.two_term(source)
        a, ea = arithmetic.pack(torch.tensor([[0.3, -0.8, 1.2]]))
        self.assert_bits(arithmetic.contraction(a, b, ea, eb, low=low),
                         self.reference(a, b, ea, eb, low=low))
        self.assertTrue(bool((low[0] != 0).any()))
        original, original_exponent = arithmetic.pack(source)
        self.assertTrue(torch.equal(original, b))
        self.assertEqual(int(original_exponent), int(eb))

    def test_direct_bf16_avoids_float32_double_round(self):
        values = [1 + 2**-8 + 2**-40, 1 + 3 * 2**-8 - 2**-40]
        values += [-x for x in values]
        source = torch.tensor(values, dtype=torch.float64)
        actual = arithmetic.direct_bf16(source)
        self.assert_bits(actual, [oracle.bf16_round(Fraction(x))
                                 for x in values])
        bridged = source.float().bfloat16().float()
        self.assertFalse(torch.equal(actual, bridged))
        self.assertEqual(bits(arithmetic.direct_bf16(torch.tensor([-0.0]))), 0)

    def test_site_masks_and_opt_in_defaults(self):
        args = SimpleNamespace(precision_mode="default", sites="all",
                               k_chunk=128, finer_scales=False,
                               activation_scale=None, weight_scale=None)
        self.assertEqual(arithmetic.settings(args, "lm_head"),
                         ("tensor", "tensor", 0, False))
        args.precision_mode, args.sites = "two-term", "layer1_0_conv1"
        self.assertTrue(arithmetic.settings(args, "layer1_0_conv1")[-1])
        self.assertFalse(arithmetic.settings(args, "fc")[-1])
        args.precision_mode = "k-split"
        args.sites, args.finer_scales = "attention-head", True
        self.assertEqual(arithmetic.settings(args, "lm_head"),
                         ("row", "column", 128, False))
        site = "model.layers.0.mlp.down_proj"
        self.assertEqual(arithmetic.settings(args, site),
                         ("row", "column", 0, False))

    def test_frozen_default_contraction_cells(self):
        path = (Path(__file__).parent /
                "fixtures/precision_modes/default_cells.json")
        fixtures = json.loads(path.read_text())
        for item in fixtures:
            with self.subTest(model=item["model"]):
                def front(key, shape):
                    words = torch.frombuffer(bytearray.fromhex(item[key]),
                                             dtype=torch.int16)
                    return words.view(torch.bfloat16).reshape(shape).float()
                k = item["k"]
                ea, eb = torch.tensor(item["ea"]), torch.tensor(item["eb"])
                a = arithmetic.quantize(front("a_bf16_le_hex", (1, k)), ea)
                b = arithmetic.quantize(front("b_bf16_le_hex", (k, 1)), eb)
                actual = arithmetic.contraction(a, b, ea, eb)
                self.assertEqual(struct.pack("<f", float(actual)),
                                 bytes.fromhex(item["expected_f32_le_hex"]))


if __name__ == "__main__":
    unittest.main()
