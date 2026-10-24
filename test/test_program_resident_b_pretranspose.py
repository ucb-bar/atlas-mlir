"""Symbolic byte movement only: no assembler, numerical model, or RTL execution."""

from collections import Counter
import hashlib
from pathlib import Path
import re
import sys
import unittest

TOOL_ROOT = Path(__file__).absolute().parents[1] / "tools"
sys.path.insert(0, str(TOOL_ROOT))

import atlas_program as producer


def transpose(data):
    # Raw 8-bit lanes: destination row c, lane r receives source row r, lane c.
    return tuple(data[r * 32 + c] for c in range(32) for r in range(32))


def prove_bytes(source, m, k, n, *, pretransposed):
    """Execute scalar flow and actual MREG/VMEM byte permutations on origin labels.

    Distinct integer labels identify every original DRAM byte, including its tile.
    Output tokens stand for unchanged MXU/pop arithmetic; no numeric claim is made.
    VSTORE writes all 32 rows, and later VLOAD independently reconstructs those rows.
    """
    offset = m * k + k * n + 256
    lines = source.splitlines()
    labels = {line[:-1]: i for i, line in enumerate(lines) if line.endswith(":")}
    regs, memory, mregs = {"x0": 0}, {}, {}
    loads, stores, b_stores, products, core = [], [], [], [], []
    counts, instructions, delays = Counter(), 0, 0
    expected_b = {origin: transpose(tuple(range(origin, origin + 1024))) for origin in range(m * k, m * k + k * n, 1024)}
    weight, pc, in_compute = None, 0, False

    def address(arg):
        return 4 * regs[re.fullmatch(r"0\((x\d+)\)", arg).group(1)]

    def read_tile(base):
        assert base % 1024 == 0 and 0 <= base <= 1536 * 1024 - 1024
        return tuple(value for row in range(0, 1024, 32) for value in memory[base + row])

    def write_tile(base, data):
        assert base % 1024 == 0 and len(data) == 1024
        assert 0 <= base <= 1536 * 1024 - 1024
        for row in range(0, 1024, 32):
            memory[base + row] = data[row : row + 32]

    for _ in range(1000000):
        line = lines[pc]
        op, *args = line.replace(",", "").split()
        pc += 1
        if op.endswith(":"):
            in_compute |= op == "rows_loop:"
            continue
        counts[op] += 1
        instructions += 1
        if in_compute and op.startswith("v") and op != "vtrpose.xlu":
            core.append(line)
        if op == "delay":
            delays += int(args[0])
        elif op in ("nop", "dma.wait", "dma.config"):
            pass
        elif op == "li":
            regs[args[0]] = int(args[1], 0)
        elif op in ("add", "addi"):
            regs[args[0]] = regs[args[1]] + (regs[args[2]] if op == "add" else int(args[2]))
        elif op in ("beq", "jal"):
            assert lines[pc] == "nop", "proof requires inert branch delay slots"
            if op == "jal" or regs[args[0]] == regs[args[1]]:
                pc = labels[args[-1]]
        elif op == "dma.load":
            dest, origin, count = (regs[arg] for arg in args[:3])
            dest *= 4  # VMEM uses words; DRAM origin and count use bytes.
            assert count == 1024
            if origin >= m * k:
                assert not in_compute and 0x40000 <= dest <= 0x80000 - 1024
            else:
                assert dest == 0x2000 and 0 <= origin < m * k
            write_tile(dest, tuple(range(origin, origin + count)))
            loads.append((dest, origin, count))
        elif op == "vload":
            mregs[args[0]] = read_tile(address(args[1]))
        elif op == "vtrpose.xlu":
            mregs[args[0]] = transpose(mregs[args[1]])
        elif op == "vstore":
            dest = address(args[1])
            write_tile(dest, mregs[args[0]])
            if args[0] == "m1":
                assert not in_compute and 0x40000 <= dest <= 0x80000 - 1024
                b_stores.append((dest, mregs["m1"]))
        elif op == "vmatpush.weight.mxu0":
            weight = mregs[args[1]]
        elif op in ("vmatmul.mxu0", "vmatmul.acc.mxu0"):
            inner = len(products) % (k // 32)
            output = len(products) // (k // 32)
            row, column = divmod(output, n // 32)
            a = row * k * 32 + inner * 1024
            b = m * k + column * k * 32 + inner * 1024
            assert mregs[args[1]] == tuple(range(a, a + 1024)), "original ordered A bytes"
            assert weight == expected_b[b], "stored/reloaded original transposed B bytes"
            assert (op == "vmatmul.acc.mxu0") == (inner != 0), "full-K reset/continuation"
            products.append((a, b, inner != 0))
        elif op == "vmatpop.bf16.acc.mxu0":
            assert len(products) % (k // 32) == 0 and len(stores) == len(products) // (k // 32) - 1
            # Preserve the two physical BF16 column halves and each half's row order.
            tokens = tuple(-(len(stores) * 2048 + i + 1) for i in range(2048))
            mregs["m2"], mregs["m3"] = tokens[:1024], tokens[1024:]
        elif op in ("vli.all", "vmul.bf16"):
            # Their instructions must match the existing producer exactly (core below).
            pass
        elif op == "dma.store":
            target, src, count = (regs[arg] for arg in args[:3])
            expected = tuple(-(len(stores) * 2048 + i + 1) for i in range(2048))
            assert (target, src, count) == (offset + len(stores) * 2048, 0xA00, 2048), "row-major output pointers"
            assert read_tile(src * 4) + read_tile(src * 4 + 1024) == expected, "physical output halves/rows"
            stores.append((target, src, count))
        elif op == "ecall":
            unique = k * n // 1024
            assert len(products) == m * n // 1024 * (k // 32)
            assert len(stores) == m * n // 1024
            assert loads[:unique] == [(0x40000 + i * 1024, m * k + i * 1024,
                1024) for i in range(unique)], "unique B preload origin/word step"
            assert sum(size for _, _, size in loads) == m * k * (n // 32) + k * n
            expected_stores = [(0x40000 + i * 1024, expected_b[m * k + i * 1024]) for i in range(unique)]
            assert b_stores == (expected_stores if pretransposed else []), "same-tile transpose store/endpoints"
            return dict(loads=loads, stores=stores, products=products, core=core, counts=counts, instructions=instructions,
                delays=delays, b_stores=b_stores)
        else:
            raise AssertionError("unmodeled instruction: " + op)
    raise AssertionError("symbolic program did not terminate")


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

    def test_byte_permutation_and_endpoints(self):
        distinct = tuple(range(1024))
        result = transpose(distinct)
        self.assertEqual((result[0], result[31], result[32], result[1023]), (0, 992, 1, 1023))
        self.assertEqual(sorted(result), list(distinct))
        self.assertEqual(transpose(result), distinct)
        # All 256 raw byte codes survive, with an asymmetric row/column pattern.
        raw = tuple((r * 37 + c * 13) % 256 for r in range(32) for c in range(32))
        self.assertEqual(sorted(transpose(raw)), sorted(raw))
        self.assertEqual(transpose(transpose(raw)), raw)

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


if __name__ == "__main__":
    unittest.main()
