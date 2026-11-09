"""Keep model-compilation handoffs compatible with contract-based timing."""

import json
import unittest

from test_cfg_dma_timing import HALT, WAIT, dma, prefix, upper, vwork
from test_delay_insertion import EMIT, OPT, addi, nop, program, run
from test_instruction_memory import HALT as IMEM_HALT, NOP, stream


def contract_program():
    ops = prefix() + [
        addi(19, 0, 0),
        ("dma_config", "channel = 0 : i32, base_reg = 19 : i32"),
        upper(14, 2), upper(15, 0x90001), addi(16, 0, 32), upper(17, 8),
        ("dma", 'direction = "load", channel = 2 : i32, reg = 14 : i32, '
                'dram = 15 : i32, size = 16 : i32'),
        ("dma_wait", "channel = 2 : i32"),
        ("scalar_load", 'kind = "lw", dst = 5 : i32, base = 17 : i32, offset = 0 : i32'),
        ("delay", "cycles = 4 : i32"), nop(), dma(),
    ] + vwork(5) + [WAIT, HALT]
    return program(ops).replace(
        "module {",
        'module attributes {atlas.dram_contract = {name = "merge-metadata-v1", '
        'words = [{address = 2415923200 : i64, value = 0 : i64}]}} {', 1,
    )


class MergeHandoffsTest(unittest.TestCase):
    def checked(self, tool, source, *options):
        result = run(tool, source, *options)
        self.assertEqual(result.returncode, 0, result.stderr[:2000])
        return result.stdout

    def test_relaxed_branches_export_the_same_words_and_metadata(self):
        for backward in (False, True):
            with self.subTest(backward=backward):
                branch = ("branch", f'kind="bne", lhs=2:i32, rhs=0:i32, '
                          f'offset_bytes={-4096 if backward else 4094}:i32')
                load = ("scalar_load", 'kind="lw", dst=1:i32, base=0:i32, offset=0:i32')
                use = ("alu_imm", 'kind="addi", dst=3:i32, src=1:i32, immediate=1:i32')
                ops = ([NOP, load, use] + [NOP] * 2045 + [branch, NOP, IMEM_HALT]
                       if backward else [branch, NOP, load, use] + [NOP] * 2043 + [IMEM_HALT])
                machine = self.checked(OPT, stream(ops), "--insert-atlas-delays", "--verify-atlas-timing")
                self.assertIn("atlas.branch_trampoline", machine)
                words = [int(word, 16) for word in self.checked(EMIT, machine).splitlines()]
                exported = json.loads(self.checked(EMIT, machine, "--program-json"))
                self.assertEqual([row["word_u32"] for row in exported["instructions"]], words)
                trampolines = [row for row in exported["instructions"]
                               if row["fields"].get("atlas.branch_trampoline")]
                self.assertEqual(len(trampolines), 1)
                self.assertIs(trampolines[0]["fields"]["atlas.branch_trampoline"], True)
                self.checked(EMIT, machine, "--map-json")

    def test_structured_llvm_rechecks_the_retained_dram_contract(self):
        source = contract_program()
        machine = self.checked(OPT, source, "--verify-atlas-timing")
        self.assertIn('atlas.timing_assumptions = ["immutable-dram:merge-metadata-v1"]', machine)
        exported = json.loads(self.checked(EMIT, machine, "--program-json"))
        self.assertEqual(exported["timing_assumptions"], ["immutable-dram:merge-metadata-v1"])
        self.assertIn("merge-metadata-v1", exported["input_dram_contract"])
        structured = self.checked(OPT, machine, "--convert-atlas-to-llvm-calls")
        finalized = self.checked(OPT, structured, "--finalize-atlas-llvm-calls")
        self.assertEqual(finalized, self.checked(OPT, machine, "--convert-atlas-to-llvm"))
        corrupted = structured.replace("value = 0 : i64", "value = 256 : i64")
        self.assertNotEqual(corrupted, structured)
        rejected = run(OPT, corrupted, "--finalize-atlas-llvm-calls")
        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn("conflict", rejected.stderr)
        self.assertEqual(rejected.stdout, "")


if __name__ == "__main__":
    unittest.main()
