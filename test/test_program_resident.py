"""Legacy residency proofs share the operand and DMA interpreter."""

import hashlib
from pathlib import Path
import re
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch
import zipfile

TOOL_ROOT = Path(__file__).absolute().parents[1] / "tools"
sys.path.insert(0, str(TOOL_ROOT))
import atlas_program as producer
from reuse_proof import prove


def prove_legacy(source, m, k, n, mode="none"):
    scale = re.search(r"^vli.all m4, (\d+)$", source, re.MULTILINE)
    exponent = (int(scale[1]) >> 7) - 127 if scale else None
    return prove(source, m, k, n, exponent, legacy=mode)


def verify(source, m, k, n, output_offset, *, resident_a=False, resident_b=False):
    assert output_offset == m * k + k * n + 256
    mode = "A" if resident_a else "B" if resident_b else "none"
    return prove_legacy(source, m, k, n, mode)["arithmetic"]


def prove_bytes(source, m, k, n, *, pretransposed):
    return prove_legacy(source, m, k, n, "BT" if pretransposed else "B")


def accounting(m, k, n, **options):
    source = producer.assembly(m, k, n, m * k + k * n + 256, **options)
    mode = ("A" if options.get("resident_a") else
            "BT" if options.get("pretranspose_resident_b") else
            "B" if options.get("resident_b") else "none")
    proof = prove_legacy(source, m, k, n, mode)
    return {key: proof[key] for key in ("load_bytes", "store_bytes")}


CASES = [("Smol Q", 192, 960, 960, -12, 5713920), ("Smol K", 192, 960, 320, -12, 2027520),
    ("Smol V", 192, 960, 320, -16, 2027520), ("Smol O", 192, 960, 960, -15, 5713920),
    ("Tiny Q", 128, 2048, 2048, -14, 17039360), ("Tiny K", 128, 2048, 256, -13, 2359296),
    ("Tiny V", 128, 2048, 256, -18, 2359296), ("Tiny O", 128, 2048, 2048, -21, 17039360)]


class ResidentATests(unittest.TestCase):

    def test_all_workloads_match_frozen_producer_assembly(self):
        for role, m, k, n, exponent, mode, digest in FROZEN_ASSEMBLY:
            options = {"resident_a": mode == "A", "resident_b": mode in ("B", "BT"), "pretranspose_resident_b": mode == "BT"}
            with self.subTest(role=role, mode=mode, exponent=exponent):
                if digest is None:
                    with self.assertRaisesRegex(ValueError, "256-KiB"):
                        producer.assembly(m, k, n, m * k + k * n + 256, exponent, **options)
                else:
                    source = producer.assembly(m, k, n, m * k + k * n + 256, exponent, **options)
                    self.assertEqual(hashlib.sha256(source.encode()).hexdigest(), digest)

    def test_invalid_residency_combinations(self):
        cases = [(True, True, False), (True, False, True), (True, True, True), (False, False, True)]
        for resident_a, resident_b, pretranspose in cases:
            options = {"resident_a": resident_a, "resident_b": resident_b, "pretranspose_resident_b": pretranspose}
            with self.subTest(flags=options), self.assertRaises(ValueError):
                producer.assembly(64, 64, 64, 10000, **options)

    def test_default_and_pretranspose_dma_accounting(self):
        self.assertEqual(accounting(64, 64, 64), {"load_bytes": 16384, "store_bytes": 8192})
        self.assertEqual(accounting(64, 64, 64, resident_b=True, pretranspose_resident_b=True),
            {"load_bytes": 12288, "store_bytes": 8192})


    def test_all_smol_and_tiny_original_k_order_and_stores(self):
        for name, m, k, n, exponent, loads in CASES:
            with self.subTest(workload=name):
                offset = m * k + k * n + 256
                baseline = producer.assembly(m, k, n, offset, exponent)
                candidate = producer.assembly(m, k, n, offset, exponent, resident_a=True)
                self.assertEqual(verify(baseline, m, k, n, offset), verify(candidate, m, k, n, offset, resident_a=True))
                self.assertEqual([line for line in baseline.splitlines() if line.startswith("delay ")],
                    [line for line in candidate.splitlines() if line.startswith("delay ")])
                requested = accounting(m, k, n, resident_a=True)
                self.assertEqual(requested, {"load_bytes": loads, "store_bytes": m * n * 2})

    def test_tiny_exact_bank_end_and_smoke_unit_recipe(self):
        for m, k, n in [(64, 64, 64), (128, 2048, 256), (128, 2048, 2048)]:
            offset = m * k + k * n + 256
            source = producer.assembly(m, k, n, offset, resident_a=True)
            verify(source, m, k, n, offset, resident_a=True)
            loads = prove_legacy(source, m, k, n, "A")["loads"]
            a_loads = [item for item in loads if item[1] < m * k]
            self.assertEqual(a_loads[-1], (0x40000 + m * k - 1024, m * k - 1024, 1024))
            if m == 128:
                self.assertEqual(a_loads[-1][0] + a_loads[-1][2], 0x80000)

    def test_capacity_and_mutual_exclusion(self):
        for shape in [(160, 2048, 256), (3136, 576, 64)]:
            with self.assertRaisesRegex(ValueError, "resident A.*256-KiB"):
                producer.assembly(*shape, 10000000, resident_a=True)
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            producer.assembly(64, 64, 64, 10000, resident_a=True, resident_b=True)

    def test_symbolic_proof_rejects_broken_preload_offsets_and_math(self):
        m = k = n = 64
        offset = m * k + k * n + 256
        source = producer.assembly(m, k, n, offset, 0, resident_a=True)
        mutations = [("missing preload", "li x21, 4", "li x21, 3"),
            ("preload destination", "addi x19, x19, 256", "addi x19, x19, 512"),
            ("preload origin", "li x20, 0", "li x20, 1024"), ("A K stride", "addi x4, x4, 256", "addi x4, x4, 1024"),
            ("A row stride", "add x10, x10, x18", "add x10, x10, x16"),
            ("B K stride", "addi x6, x6, 1024", "addi x6, x6, 256"), ("transpose", "vtrpose.xlu m1, m1", "nop"),
            ("continuation", "vmatmul.acc.mxu0", "vmatmul.mxu0"), ("output alias", "add x12, x12, x8", "add x12, x12, x0")]
        for name, old, new in mutations:
            self.assertIn(old, source)
            with self.subTest(mutation=name), self.assertRaises((AssertionError, KeyError)):
                verify(source.replace(old, new), m, k, n, offset, resident_a=True)


class ResidentBTests(unittest.TestCase):
    def test_default_assembly_is_frozen(self):
        # Fixed assembly digests guard the default encoding.
        cases = [((64, 64, 64), "abdcad20dec027880a93a249e46fdbfde9aa28bf9fd1f0bb4f9041e84a85b73c"),
            ((3136, 64, 64), "d3c04207242103608881526c87573a607cd2b46d763100ab759ff5c43fb6cad4"),
            ((3136, 576, 64), "64c045a08bcafcab150f12a4a07e30f14b27aff36676b6a847520c032d0a9106"),
            ((3136, 64, 256), "49dda1920eda26b29de7789df5399e6a4e82e749eda736715d0fb7b1a3c8b6df")]
        for (m, k, n), digest in cases:
            with self.subTest(shape=(m, k, n)):
                source = producer.assembly(m, k, n, m * k + k * n + 256, 0)
                self.assertEqual(hashlib.sha256(source.encode()).hexdigest(), digest)


    def test_all_resnet_shapes_and_bank_boundary(self):
        for m, k, n in [(64, 64, 64), (3136, 64, 64), (3136, 576, 64), (3136, 64, 256), (32, 4096, 64)]:
            with self.subTest(shape=(m, k, n)):
                offset = m * k + k * n + 256
                baseline = producer.assembly(m, k, n, offset, 0)
                candidate = producer.assembly(m, k, n, offset, 0, resident_b=True)
                self.assertEqual(verify(baseline, m, k, n, offset, resident_b=False),
                    verify(candidate, m, k, n, offset, resident_b=True))
                # All conservative compute/rescale/store delays remain exact.
                self.assertEqual([x for x in baseline.splitlines() if x.startswith("delay ")],
                    [x for x in candidate.splitlines() if x.startswith("delay ")])

    def test_capacity_refuses_smol_and_tiny(self):
        for shape in [(192, 960, 960), (128, 2048, 2048)]:
            with self.subTest(shape=shape), self.assertRaisesRegex(ValueError, "256-KiB"):
                producer.assembly(*shape, 10000000, resident_b=True)

    def test_symbolic_proof_rejects_controls(self):
        m = k = n = 64
        offset = m * k + k * n + 256
        source = producer.assembly(m, k, n, offset, 0, resident_b=True)
        controls = [source.replace("vtrpose.xlu m1, m1", "nop"), source.replace("vmatmul.acc.mxu0", "vmatmul.mxu0"),
            source.replace("add x12, x12, x8", "add x12, x12, x0"), source.replace("addi x6, x6, 256", "addi x6, x6, 1024"),
            source.replace("li x21, 4", "li x21, 3")]
        for changed in controls:
            with self.subTest(control=controls.index(changed)), self.assertRaises((AssertionError, KeyError)):
                verify(changed, m, k, n, offset, resident_b=True)

    def test_archive_admission_before_digest(self):
        args = SimpleNamespace(smoke=False, input="/tmp/source.npz", manifest="/tmp/source.json")
        required = [zipfile.ZipInfo(name) for name in ("A.npy", "B.npy", "native_C.npy")]
        archive = MagicMock()
        archive.__enter__.return_value.infolist.return_value = required
        with patch.object(producer.zipfile, "ZipFile", return_value=archive), patch.object(producer, "sha256",
            side_effect=AssertionError("digest before admission")):
            producer.preflight_bundle(args)
            # Prohibited/traversal members exist only as in-memory metadata.
            for name in ("../A.npy",):
                archive.__enter__.return_value.infolist.return_value = required + [zipfile.ZipInfo(name)]
                with self.assertRaises(ValueError):
                    producer.preflight_bundle(args)


class PretransposeResidentBTests(unittest.TestCase):
    def test_existing_resident_b_assembly_bytes(self):
        # Fixed resident-B assembly digests cover unit and scaled outputs.
        cases = [((64, 64, 64), "a61dd3150d9ed2817e064c84c17066a6b08a7d91f29770c160550d7d13d211d9",
                "461125426fa78f6ba1486ed03706e2fd25a1b9ac14f0a0db1680d9dbe7a54fa6",
            ), ((3136, 64, 64), "a4f95c1082148f354155967f6580f9c135681c787aaf9e61b80a2429f6667eae",
                "038a8487211fa4ee0bc7ca1fa148f73c64e0acf76ea97fd75fee3b976f33c41f",
            ), ((3136, 576, 64), "dbf56efdb61fcbc966ead41759db113635a22a04997aa4462e12db29baf8666e",
                "879083244282457ca63af9f2436671660f66e513f1068a684978e8d75b6f08af",
            ), ((3136, 64, 256), "1b127c83b39dc6a8bee8e1963682d8198611c6e182592a3cd9e6e5a953378906",
                "2b0bb5ed9e6a55e77e9e690fd645baf3e488a3e39282af016c2816ea53221c34",
            ), ((32, 4096, 64), "74a7559560ff73de4899f9b205c54d76ba74da716deb0e34f5705ff6957663b4",
                "5834d114c46dc7f3bcf21ef7b2dd8ae6e2349a35d368f4a797caa3b94e90d1e7")]
        for shape, unit, scaled in cases:
            for exponent, digest in [(None, unit), (0, scaled)]:
                with self.subTest(shape=shape, exponent=exponent):
                    m, k, n = shape
                    text = producer.assembly(m, k, n, m * k + k * n + 256, exponent, resident_b=True)
                    self.assertEqual(hashlib.sha256(text.encode()).hexdigest(), digest)


    def test_full_resnet_order_storage_counts_and_bank_boundary(self):
        for name, shape in [("smoke", (64, 64, 64)), ("C1", (3136, 64, 64)), ("C2", (3136, 576, 64)),
            ("C3", (3136, 64, 256)), ("shortcut", (3136, 64, 256)), ("bank boundary", (32, 4096, 64))]:
            with self.subTest(site=name):
                m, k, n = shape
                offset = m * k + k * n + 256
                old = producer.assembly(m, k, n, offset, -14, resident_b=True)
                new = producer.assembly(m, k, n, offset, -14, resident_b=True, pretranspose_resident_b=True)
                prior = prove_bytes(old, *shape, pretransposed=False)
                current = prove_bytes(new, *shape, pretransposed=True)
                for field in ("loads", "stores", "products", "core"):
                    self.assertEqual(current[field], prior[field], field)
                unique, uses = k * n // 1024, m * n // 1024 * (k // 32)
                self.assertEqual((prior["counts"]["vtrpose.xlu"], current["counts"]["vtrpose.xlu"]), (uses, unique))
                self.assertEqual(current["instructions"] - prior["instructions"], 6 * unique - 2 * uses)
                self.assertEqual(current["delays"] - prior["delays"], 134 * unique - 66 * uses)
                static = lambda text: sum(not line.endswith(":") for line in text.splitlines())
                self.assertEqual(static(new) - static(old), 6 - 2 * (k // 32))
                self.assertIn(
                    "dma.wait ch0\nvload m1, 0(x19)\ndelay 34\nvtrpose.xlu m1, m1\ndelay 66\n"
                    "vstore m1, 0(x19)\ndelay 34\naddi x19, x19, 256", new)
                self.assertIn("vload m1, 0(x6)\ndelay 34\nvmatpush.weight.mxu0 w0, m1\ndelay 32", new)
                self.assertEqual(current["b_stores"][-1][0] + 1024, 0x40000 + k * n)
                if name == "bank boundary":
                    self.assertEqual(current["b_stores"][-1][0] // 4, 0x1FF00)
                    self.assertEqual(current["b_stores"][-1][0] + 1024, 0x80000)

    def test_broken_pretranspose_store_pointers_and_continuations_refused(self):
        shape = (64, 64, 64)
        m, k, n = shape
        source = producer.assembly(*shape, m * k + k * n + 256, 0, resident_b=True, pretranspose_resident_b=True)
        changes = {"missing transpose": ("vtrpose.xlu m1, m1", "nop"), "missing store": ("vstore m1, 0(x19)", "nop"),
            "wrong store pointer": ("vstore m1, 0(x19)", "vstore m1, 0(x2)"),
            "wrong preload load pointer": ("vload m1, 0(x19)", "vload m1, 0(x2)"),
            "wrong preload word step": ("addi x19, x19, 256", "addi x19, x19, 1024"),
            "wrong later word step": ("addi x6, x6, 256", "addi x6, x6, 1024"),
            "wrong DRAM step": ("addi x20, x20, 1024", "addi x20, x20, 256"),
            "missing preload tile": ("li x21, 4", "li x21, 3"), "reset continuation": ("vmatmul.acc.mxu0", "vmatmul.mxu0"),
            "aliased output": ("add x12, x12, x8", "add x12, x12, x0"),
            "swapped output halves": ("vstore m3, 0(x9)", "vstore m2, 0(x9)")}
        for name, (old, new) in changes.items():
            self.assertIn(old, source)
            with self.subTest(control=name), self.assertRaises((AssertionError, KeyError)):
                prove_bytes(source.replace(old, new), *shape, pretransposed=True)

    def test_pretranspose_requires_resident_b(self):
        with self.assertRaisesRegex(ValueError, "requires --resident-b"):
            producer.assembly(64, 64, 64, 10000, pretranspose_resident_b=True)


FROZEN_ASSEMBLY = [
    ("ResNet C1", 3136, 64, 64, None, "none", "608889b9feae5a2230c81fe31453c45275370d125410a82ff9b2d01115a66ca9"),
    ("ResNet C1", 3136, 64, 64, -11, "none", "6bf4cb8ad6c46171de44133b02a9c54def09b9e38ab16b7d8a7e7b7647a8239b"),
    ("ResNet C1", 3136, 64, 64, None, "B", "a4f95c1082148f354155967f6580f9c135681c787aaf9e61b80a2429f6667eae"),
    ("ResNet C1", 3136, 64, 64, -11, "B", "97b7331268053d7cc514ff20fb0d2aa5ee923824470733ccae7cd3788b6dc5af"),
    ("ResNet C1", 3136, 64, 64, None, "BT", "292d9736db337d7a2438308b6217fdc770e24af3aab9264b8dd0e3d846b8e1e4"),
    ("ResNet C1", 3136, 64, 64, -11, "BT", "c42ea5ab3e86a6f7f9c78b2cae8ec463d4ac01ccbdac09e052525d7a56119fc6"),
    ("ResNet C1", 3136, 64, 64, None, "A", "2b0d2c86947ccdd13b8812f318a969c99061975ab076550dc8adb0ba3af99dfb"),
    ("ResNet C1", 3136, 64, 64, -11, "A", "cce02e63387a54453b4ce8cec35831fadda101cf5ef9ae513e066f9e4518de71"),
    ("ResNet C2", 3136, 576, 64, None, "none", "0151ff63e36c8a68962a27a183399657e1b6d9d28d122856f1e1707172295bce"),
    ("ResNet C2", 3136, 576, 64, -12, "none", "d488e49fddb932f34f08bcc4401737732b6cb243bdfaebbe624be95b60ec2086"),
    ("ResNet C2", 3136, 576, 64, None, "B", "dbf56efdb61fcbc966ead41759db113635a22a04997aa4462e12db29baf8666e"),
    ("ResNet C2", 3136, 576, 64, -12, "B", "8c2dc98fbaac6c64d8c67451292efacfac92def806e5e27747225d905b256a26"),
    ("ResNet C2", 3136, 576, 64, None, "BT", "1dc0b18aa97e91301dae3472a7112092e3af3a753bc2b2638a71874605598f81"),
    ("ResNet C2", 3136, 576, 64, -12, "BT", "7662d1f19cb7879f944e7fe33233d8d54eb8b9bf8f9886750fa600aac3a429a2"),
    ("ResNet C2", 3136, 576, 64, None, "A", None), ("ResNet C2", 3136, 576, 64, -12, "A", None),
    ("ResNet C3", 3136, 64, 256, None, "none", "6b5617c8a87791dc2248bc28b3e1ca294d9bc46a432034624f8b9f6d575a6f87"),
    ("ResNet C3", 3136, 64, 256, -12, "none", "f09dc3f17d89c2ec604fca7570c2eaba9ee53eaf7aef38e42917c7aa1b8a2a5c"),
    ("ResNet C3", 3136, 64, 256, None, "B", "1b127c83b39dc6a8bee8e1963682d8198611c6e182592a3cd9e6e5a953378906"),
    ("ResNet C3", 3136, 64, 256, -12, "B", "46306f6bdbff9bb4a035fd8bd44c41a021e88175f332eefebe05ae26c33bfe6f"),
    ("ResNet C3", 3136, 64, 256, None, "BT", "11237fb52c83947454d2dd9b8ab0041d12295c5ac88e3fa63b66437edf075152"),
    ("ResNet C3", 3136, 64, 256, -12, "BT", "590c7ba89f062cadbb9689b6fe95f2fd17df03d30ece823ce7cf6e8cb6421a4b"),
    ("ResNet C3", 3136, 64, 256, None, "A", "9935c5535423fd57751a5fd40f6f6b29d50cc2c0cebb9f5bc88905c16d48e1ca"),
    ("ResNet C3", 3136, 64, 256, -12, "A", "69bccff45d7b740de38554ea1822689bd92eeb995d8b1800698b0e3cead8de98"),
    ("ResNet shortcut", 3136, 64, 256, None, "none", "6b5617c8a87791dc2248bc28b3e1ca294d9bc46a432034624f8b9f6d575a6f87"),
    ("ResNet shortcut", 3136, 64, 256, -10, "none", "238b33b6782f64fe2af277082497327d048a159d8f7a3206aa1f235e945871b0"),
    ("ResNet shortcut", 3136, 64, 256, None, "B", "1b127c83b39dc6a8bee8e1963682d8198611c6e182592a3cd9e6e5a953378906"),
    ("ResNet shortcut", 3136, 64, 256, -10, "B", "85977c516a678ad9b537ec731c718637dee34850cff1338372590e747f28f0d1"),
    ("ResNet shortcut", 3136, 64, 256, None, "BT", "11237fb52c83947454d2dd9b8ab0041d12295c5ac88e3fa63b66437edf075152"),
    ("ResNet shortcut", 3136, 64, 256, -10, "BT", "f8c641b1bfd5bbd9d5971e997bd3069d97f2b2c678c23c3427b6ecf70b10908c"),
    ("ResNet shortcut", 3136, 64, 256, None, "A", "9935c5535423fd57751a5fd40f6f6b29d50cc2c0cebb9f5bc88905c16d48e1ca"),
    ("ResNet shortcut", 3136, 64, 256, -10, "A", "be3e06ce236c15de6b92465829d087dbbf9ceb173108c4fe368a3810bd1b8472"),
    ("Smol Q", 192, 960, 960, None, "none", "50820b36c0dd7be7c8397957944f4a484762c2e1b0e8a7f18ff1aa1bcb71968f"),
    ("Smol Q", 192, 960, 960, -12, "none", "5a248d820cb95ed6f067180f8534c6f778f928f66fdf7a51107a78a676d10938"),
    ("Smol Q", 192, 960, 960, None, "B", None), ("Smol Q", 192, 960, 960, -12, "B", None),
    ("Smol Q", 192, 960, 960, None, "BT", None), ("Smol Q", 192, 960, 960, -12, "BT", None),
    ("Smol Q", 192, 960, 960, None, "A", "e6b45170333e7d9a72c7294110cffe45925acfa97ef7289bda2c486a48bcc41c"),
    ("Smol Q", 192, 960, 960, -12, "A", "07f78366fd3f899aebb30238ecfece2e747cfa49a0bf320ba571f94472e5e035"),
    ("Smol K", 192, 960, 320, None, "none", "2b476162f843b1ea878c04e85f4fc5faf5fb90677d9dfd747b88d3af55276324"),
    ("Smol K", 192, 960, 320, -12, "none", "916567dc13668a8ffff5c7b0636598395d53fc8178048edee95cae06b8002b35"),
    ("Smol K", 192, 960, 320, None, "B", None), ("Smol K", 192, 960, 320, -12, "B", None),
    ("Smol K", 192, 960, 320, None, "BT", None), ("Smol K", 192, 960, 320, -12, "BT", None),
    ("Smol K", 192, 960, 320, None, "A", "d795c43c426c31c983f4bcbda78c9ed5a7d3c42247e61f935aca4f6b8b064fd5"),
    ("Smol K", 192, 960, 320, -12, "A", "7e36ca930acd1e04c2e44fcdf9856eb4b849decf71219a40ece6f95e1d4e8a8d"),
    ("Smol V", 192, 960, 320, None, "none", "2b476162f843b1ea878c04e85f4fc5faf5fb90677d9dfd747b88d3af55276324"),
    ("Smol V", 192, 960, 320, -16, "none", "264910ee3666eef90aab5757a31b9980a8cbf50b2b29521d2380dd42b93eec02"),
    ("Smol V", 192, 960, 320, None, "B", None), ("Smol V", 192, 960, 320, -16, "B", None),
    ("Smol V", 192, 960, 320, None, "BT", None), ("Smol V", 192, 960, 320, -16, "BT", None),
    ("Smol V", 192, 960, 320, None, "A", "d795c43c426c31c983f4bcbda78c9ed5a7d3c42247e61f935aca4f6b8b064fd5"),
    ("Smol V", 192, 960, 320, -16, "A", "7225f8d02779beb69257aa7b8e0367ed4bda4769b351dbfa26378398cf91d982"),
    ("Smol O", 192, 960, 960, None, "none", "50820b36c0dd7be7c8397957944f4a484762c2e1b0e8a7f18ff1aa1bcb71968f"),
    ("Smol O", 192, 960, 960, -15, "none", "ecc3f45ad5db18cc085ea4964b79fa480ff2a93f97c8b0ebcf71a450510eee6f"),
    ("Smol O", 192, 960, 960, None, "B", None), ("Smol O", 192, 960, 960, -15, "B", None),
    ("Smol O", 192, 960, 960, None, "BT", None), ("Smol O", 192, 960, 960, -15, "BT", None),
    ("Smol O", 192, 960, 960, None, "A", "e6b45170333e7d9a72c7294110cffe45925acfa97ef7289bda2c486a48bcc41c"),
    ("Smol O", 192, 960, 960, -15, "A", "d330bc5841ea4ea3e39c6a20aa27c4b41137f8ac2bc63232299adbce68ca5dd0"),
    ("Tiny Q", 128, 2048, 2048, None, "none", "5c892ebe141f89b928a7d0ad75370c766c4cf1eea8af4861c587ce625203afde"),
    ("Tiny Q", 128, 2048, 2048, -14, "none", "b9f26f7ea0c1020765abe87196c22905f3c670b9d93f502ae4fcc109c39dbe30"),
    ("Tiny Q", 128, 2048, 2048, None, "B", None), ("Tiny Q", 128, 2048, 2048, -14, "B", None),
    ("Tiny Q", 128, 2048, 2048, None, "BT", None), ("Tiny Q", 128, 2048, 2048, -14, "BT", None),
    ("Tiny Q", 128, 2048, 2048, None, "A", "3b23b1b93b42471bf2b6c7a3ac19f4d6fa74d02abe61f1366d29696bc4652df5"),
    ("Tiny Q", 128, 2048, 2048, -14, "A", "295cf8b87a435c281671fdc066c89289c4eb2ec5452632aaf1e2157c91d848f0"),
    ("Tiny K", 128, 2048, 256, None, "none", "83ce766ac292873699b09f6c3056bec744c01a408e3c30db4543f0da7c08fe3a"),
    ("Tiny K", 128, 2048, 256, -13, "none", "c64fd6b62ed8481bd5c41fb6b36f4a0226b531f22ce085ab0dc1b9c5643a1b0d"),
    ("Tiny K", 128, 2048, 256, None, "B", None), ("Tiny K", 128, 2048, 256, -13, "B", None),
    ("Tiny K", 128, 2048, 256, None, "BT", None), ("Tiny K", 128, 2048, 256, -13, "BT", None),
    ("Tiny K", 128, 2048, 256, None, "A", "41944dae508a60e01d84cfc34994a8838792a0d8865997f817f9c72721d48ea0"),
    ("Tiny K", 128, 2048, 256, -13, "A", "9f14b086b008c0d7a4c74d054fcbe80126f10b610c63d096fd8d1768e8f0c2e5"),
    ("Tiny V", 128, 2048, 256, None, "none", "83ce766ac292873699b09f6c3056bec744c01a408e3c30db4543f0da7c08fe3a"),
    ("Tiny V", 128, 2048, 256, -18, "none", "bafab7490e142717d41418f8a8e540a6b502405985d1a6c03744d1541bb530f1"),
    ("Tiny V", 128, 2048, 256, None, "B", None), ("Tiny V", 128, 2048, 256, -18, "B", None),
    ("Tiny V", 128, 2048, 256, None, "BT", None), ("Tiny V", 128, 2048, 256, -18, "BT", None),
    ("Tiny V", 128, 2048, 256, None, "A", "41944dae508a60e01d84cfc34994a8838792a0d8865997f817f9c72721d48ea0"),
    ("Tiny V", 128, 2048, 256, -18, "A", "e64e1dc63c0600a4467759abbdcd34e7f116406a2934cf94bb7a5a81a68bbbf7"),
    ("Tiny O", 128, 2048, 2048, None, "none", "5c892ebe141f89b928a7d0ad75370c766c4cf1eea8af4861c587ce625203afde"),
    ("Tiny O", 128, 2048, 2048, -21, "none", "14a4fb936a947bb848798476b0b0330081ad3e2f3da2571a5bf68cf8f7914149"),
    ("Tiny O", 128, 2048, 2048, None, "B", None), ("Tiny O", 128, 2048, 2048, -21, "B", None),
    ("Tiny O", 128, 2048, 2048, None, "BT", None), ("Tiny O", 128, 2048, 2048, -21, "BT", None),
    ("Tiny O", 128, 2048, 2048, None, "A", "3b23b1b93b42471bf2b6c7a3ac19f4d6fa74d02abe61f1366d29696bc4652df5"),
    ("Tiny O", 128, 2048, 2048, -21, "A", "f846db1e39bfa3b1108cb283fc88a40384aea921c6326234907c287cf56f31b5")]


if __name__ == "__main__":
    unittest.main()
