"""Source-only schedule proofs; no numerical model, assembler, or RTL execution."""

from pathlib import Path
import sys

TOOL_ROOT = Path(__file__).absolute().parents[1] / "tools"
sys.path.insert(0, str(TOOL_ROOT))

import hashlib
import re
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch
import zipfile

import atlas_program as producer


def schedule(source):
    """Track every loaded VMEM line back to its original DRAM byte offset."""
    lines = source.splitlines()
    labels = {line[:-1]: i for i, line in enumerate(lines) if line.endswith(":")}
    registers, memory, tensors = {"x0": 0}, {}, {}
    products, stores, loads, arithmetic = [], [], [], []
    weight = None
    pc = 0
    for _ in range(1000000):
        line = lines[pc]
        fields = line.replace(",", "").split()
        op, args = fields[0], fields[1:]
        pc += 1
        if op.endswith(":") or op in ("nop", "delay", "dma.config", "dma.wait"):
            continue
        if op == "li":
            registers[args[0]] = int(args[1], 0)
        elif op in ("add", "addi"):
            registers[args[0]] = registers[args[1]] + (registers[args[2]] if op == "add" else int(args[2]))
        elif op in ("beq", "jal"):
            assert lines[pc] == "nop", "proof requires inert branch delay slots"
            if op == "jal" or registers[args[0]] == registers[args[1]]:
                pc = labels[args[-1]]
        elif op == "dma.load":
            destination, origin, count = (registers[arg] for arg in args[:3])
            destination *= 4
            assert count == 1024 and destination % 32 == origin % 32 == 0
            assert 0 <= destination <= 1536 * 1024 - count
            for offset in range(0, count, 32):
                memory[destination + offset] = origin + offset
            loads.append((destination, origin, count))
        elif op == "vload":
            base = re.fullmatch(r"0\((x\d+)\)", args[1]).group(1)
            address = registers[base] * 4
            tensors[args[0]] = (tuple(memory[address + i] for i in range(0, 1024, 32)), False)
        elif op == "vtrpose.xlu":
            data, transposed = tensors[args[1]]
            tensors[args[0]] = (data, not transposed)
        elif op == "vmatpush.weight.mxu0":
            weight = tensors[args[1]]
        elif op in ("vmatmul.mxu0", "vmatmul.acc.mxu0"):
            assert not tensors[args[1]][1] and weight[1], "transpose invariant"
            products.append((tensors[args[1]][0], weight[0], op == "vmatmul.acc.mxu0"))
        elif op == "dma.store":
            stores.append(tuple(registers[arg] for arg in args[:3]))
        elif op == "ecall":
            return products, stores, loads, arithmetic
        else:
            assert op in ("vmatpop.bf16.acc.mxu0", "vli.all", "vmul.bf16", "vstore"), op
        if op.startswith("v"):
            line = re.sub(r"vload m0, 0\(x\d+\)", "vload m0, A", line)
            arithmetic.append(re.sub(r"vload m1, 0\(x\d+\)", "vload m1, B", line))
    raise AssertionError("symbolic schedule did not terminate")


def verify(source, m, k, n, output_offset, *, resident_b=False, resident_a=False):
    products, stores, loads, arithmetic = schedule(source)
    expected = []
    for row in range(m // 32):
        for column in range(n // 32):
            for inner in range(k // 32):
                a = row * k * 32 + inner * 1024
                b = m * k + column * k * 32 + inner * 1024
                expected.append((tuple(range(a, a + 1024, 32)), tuple(range(b, b + 1024, 32)), inner != 0))
    assert products == expected, "ordered operands or accumulator continuation changed"
    assert stores == [(output_offset + i * 2048, 0xA00, 2048) for i in range(m * n // 1024)], "output alias/layout"
    expected_loads = m * k * (n // 32) + k * n if resident_b else 2 * m * k * (n // 32)
    if resident_a:
        expected_loads = m * k + k * n * (m // 32)
    assert sum(item[2] for item in loads) == expected_loads
    if resident_b:
        b_loads = [item for item in loads if item[1] >= m * k]
        assert b_loads == [(0x40000 + i, m * k + i, 1024) for i in range(0, k * n, 1024)]
        assert all(address == 0x2000 or 0x40000 <= address < 0x80000 for address, _, _ in loads)
    if resident_a:
        a_loads = [item for item in loads if item[1] < m * k]
        assert a_loads == [(0x40000 + i, i, 1024) for i in range(0, m * k, 1024)]
        assert a_loads[-1][0] + 1024 == 0x40000 + m * k <= 0x80000
        assert all(address == 0x2400 or 0x40000 <= address < 0x80000 for address, _, _ in loads)
    return arithmetic


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

    def test_resident_b_assembly_is_frozen(self):
        # Fixed assembly digests guard the resident-B encoding.
        cases = [((64, 64, 64), "461125426fa78f6ba1486ed03706e2fd25a1b9ac14f0a0db1680d9dbe7a54fa6"),
            ((3136, 64, 64), "038a8487211fa4ee0bc7ca1fa148f73c64e0acf76ea97fd75fee3b976f33c41f"),
            ((3136, 576, 64), "879083244282457ca63af9f2436671660f66e513f1068a684978e8d75b6f08af"),
            ((3136, 64, 256), "2b0bb5ed9e6a55e77e9e690fd645baf3e488a3e39282af016c2816ea53221c34"),
            ((32, 4096, 64), "5834d114c46dc7f3bcf21ef7b2dd8ae6e2349a35d368f4a797caa3b94e90d1e7")]
        for (m, k, n), digest in cases:
            with self.subTest(shape=(m, k, n)):
                source = producer.assembly(m, k, n, m * k + k * n + 256, 0, resident_b=True)
                self.assertEqual(hashlib.sha256(source.encode()).hexdigest(), digest)
                _, stores, loads, _ = schedule(source)
                requested = {"load_bytes": sum(item[2] for item in loads), "store_bytes": sum(item[2] for item in stores)}
                self.assertEqual(requested, {"load_bytes": m * k * (n // 32) + k * n, "store_bytes": m * n * 2})

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


if __name__ == "__main__":
    unittest.main()
