"""Source-only full-A residency proofs, reusing the existing symbolic interpreter."""

import hashlib
from pathlib import Path
import sys
import unittest

TOOL_ROOT = Path(__file__).absolute().parents[1] / "tools"
sys.path.insert(0, str(TOOL_ROOT))

import atlas_program as producer
from test_program_resident_b import schedule, verify
from test_program_resident_b_pretranspose import prove_bytes


CASES = [("Smol Q", 192, 960, 960, -12, 5713920), ("Smol K", 192, 960, 320, -12, 2027520),
    ("Smol V", 192, 960, 320, -16, 2027520), ("Smol O", 192, 960, 960, -15, 5713920),
    ("Tiny Q", 128, 2048, 2048, -14, 17039360), ("Tiny K", 128, 2048, 256, -13, 2359296),
    ("Tiny V", 128, 2048, 256, -18, 2359296), ("Tiny O", 128, 2048, 2048, -21, 17039360)]


def accounting(m, k, n, **options):
    """Read actual DMA counts from the symbolic instruction interpreter."""
    source = producer.assembly(m, k, n, m * k + k * n + 256, **options)
    if options.get("pretranspose_resident_b"):
        proof = prove_bytes(source, m, k, n, pretransposed=True)
        stores, loads = proof["stores"], proof["loads"]
    else:
        _, stores, loads, _ = schedule(source)
    return {"load_bytes": sum(item[2] for item in loads), "store_bytes": sum(item[2] for item in stores)}


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

    def test_source_tuple_tracks_current_emitter_and_oracle(self):
        sources = producer.code_sources()
        self.assertEqual(set(sources), {"atlas_program.py", "run_model_tile.py", "atlas_tile_arithmetic.py"})
        for name, digest in sources.items():
            with self.subTest(source=name):
                self.assertEqual(digest, hashlib.sha256((TOOL_ROOT / name).read_bytes()).hexdigest())

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
            loads = schedule(source)[2]
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


# Qualified pure-emitter SHA256 fixtures.
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
