"""Source CFG/scalar/edge correspondence survives every emitted handoff."""

from __future__ import annotations

import re
import unittest

from test_virtual_lowering import EXAMPLES, lower, run
from verification_support import (
    TIMED, TIMED_FINAL, UNTIMED, assert_boundaries, finalize_rejects, operations, reorder, rewrite_line,
)

BRANCH = "CFG contract source branch condition, polarity or true target differs"
SCALAR = "CFG contract issued scalar expression differs from source definition"
CFG_BRANCH = lambda line: '"atlas.branch"' in line and "atlas.virtual_cfg_branch" in line


def example(name: str, *, timed: bool = False) -> str:
    return lower((EXAMPLES / f"virtual_bf16_{name}_program.mlir").read_text(), timed=timed)


class CFGContractVerificationTest(unittest.TestCase):
    def check(self, text: str, rejects: str | None = None) -> None:
        assert_boundaries(self, text, TIMED_FINAL[:2] if TIMED in text else UNTIMED, rejects=rejects)

    def test_issued_branch_and_scalar_corruption_fails(self) -> None:
        dynamic, loop, branch = example("dynamic_branch"), example("loop"), example("branch")
        comparison = next(line for line in loop.splitlines() if "atlas.virtual_scalar_result" in line and 'kind = "slt"' in line)
        source_id = re.search(r"atlas.virtual_cfg_source = (\d+) : i32", comparison)[1]
        ops = operations(branch)
        branch_pc = next(n for n, line in enumerate(ops) if '"atlas.branch"' in line)
        target_pc = next(n for n, line in enumerate(ops) if "atlas.virtual_cfg_block = 3 : i32" in line)
        scalar = lambda kind: lambda line: '"atlas.alu_reg"' in line and f'kind = "{kind}"' in line and "atlas.virtual_scalar_result" in line
        mutations = [
            (dynamic, CFG_BRANCH, lambda line: line.replace('kind = "bne"', 'kind = "beq"'), BRANCH),
            (dynamic, CFG_BRANCH, lambda line: re.sub(r"lhs = \d+ : i32", "lhs = 0 : i32", line), BRANCH),
            (loop, CFG_BRANCH, lambda line: line.replace("}> {", '}> {atlas.virtual_cfg_helper = "pack", ', 1),
             "CFG contract PACK helper instruction lacks its source PACK ownership"),
            (loop, CFG_BRANCH, lambda line: line.replace("}> {", f'}}> {{atlas.virtual_cfg_helper = "pack", atlas.virtual_cfg_source = {source_id} : i32, ', 1),
             "CFG contract PACK helper requires an actual source PACK operation"),
            (branch, lambda line: '"atlas.branch"' in line,
             lambda line: re.sub(r"offset_bytes = -?\d+ : i32", f"offset_bytes = {2 * (target_pc - branch_pc)} : i32", line), BRANCH),
            (loop, scalar("add"), lambda line: line.replace('kind = "add"', 'kind = "sub"'), SCALAR),
            (loop, scalar("slt"), lambda line: line.replace('kind = "slt"', 'kind = "sltu"'), SCALAR),
            (loop, lambda line: "atlas.virtual_cfg_branch = 1 : i32" in line,
             lambda line: line.replace("atlas.virtual_cfg_branch = 1 : i32", "atlas.virtual_cfg_branch = 1 : i64"), "CFG contract tags require nonnegative i32 identities"),
        ]
        for index, (machine, where, change, diagnostic) in enumerate(mutations):
            with self.subTest(mutation=index):
                self.check(rewrite_line(machine, where, change), diagnostic)

    def test_simultaneous_scalar_copy_cycle(self) -> None:
        source = (EXAMPLES / "virtual_bf16_swap_loop_program.mlir").read_text()
        for old, new in (("%c1 = arith.constant 1 : i32", "%c1 = arith.constant 1 : i32\n    %c2 = arith.constant 2 : i32"),
                         ("%c0 : !atlas.virtual_state, !atlas.virtual_bf16, !atlas.virtual_bf16, i32)",
                          "%c0, %c1, %c2 : !atlas.virtual_state, !atlas.virtual_bf16, !atlas.virtual_bf16, i32, i32, i32)"),
                         ("%i: i32):", "%i: i32, %u: i32, %v: i32):"),
                         ("%next_i : !atlas.virtual_state, !atlas.virtual_bf16, !atlas.virtual_bf16, i32)",
                          "%next_i, %v, %u : !atlas.virtual_state, !atlas.virtual_bf16, !atlas.virtual_bf16, i32, i32, i32)")):
            source = source.replace(old, new)
        machine = lower(source, timed=False)
        self.check(machine)
        changed = rewrite_line(machine,
                               lambda line: '"atlas.alu_imm"' in line and "atlas.virtual_cfg_edge" in line and 'kind = "addi"' in line
                               and "immediate = 0 : i32" in line and "dst = 0 : i32" not in line,
                               lambda line: line.replace("immediate = 0 : i32", "immediate = 1 : i32"))
        self.check(changed, "CFG contract simultaneous source edge copy lost its incoming origin")

    def test_coordinated_tensor_producer_relabeling_fails(self) -> None:
        source = '''module {
  func.func @sequential() -> !atlas.virtual_state attributes {atlas.input_dram_base = 2415919104 : i64, atlas.output_dram_base = 2415951872 : i64} {
    %s0 = "atlas.virtual_start"() : () -> !atlas.virtual_state
    %s1, %x = "atlas.virtual_input_bf16"(%s0) {index = 0 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_bf16)
    %s2 = "atlas.virtual_output_bf16"(%s1, %x) {index = 0 : i32} : (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state
    %s3, %y = "atlas.virtual_input_bf16"(%s2) {index = 1 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_bf16)
    %s4 = "atlas.virtual_output_bf16"(%s3, %y) {index = 1 : i32} : (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state
    return %s4 : !atlas.virtual_state
  }
}'''
        machine = lower(source, timed=False)
        self.check(machine)
        ops = operations(machine)
        owner = lambda line: int(match[1]) if (match := re.search(r"atlas.virtual_cfg_source = (\d+) : i32", line)) else None
        groups = {n: [line for line in ops if owner(line) == n] for n in (1, 2, 3, 4)}
        results = {n: re.search(r"atlas.virtual_tensor_result = (\d+) : i32", "".join(groups[n]))[1] for n in (1, 3)}
        # Inputs 1 and 3 swap source labels and results, then the outputs swap places to match.
        for n, other in ((1, 3), (3, 1)):
            groups[n] = [re.sub(r"atlas.virtual_tensor_result = \d+", f"atlas.virtual_tensor_result = {results[other]}",
                                re.sub(rf"(atlas.virtual_cfg_(?:source|operation)) = {n} : i32", rf"\g<1> = {other} : i32", line))
                         for line in groups[n]]
        first = next(i for i, line in enumerate(ops) if owner(line) is not None)
        last = max(i for i, line in enumerate(ops) if owner(line) is not None)
        self.check(reorder(machine, ops[:first] + groups[1] + groups[4] + groups[3] + groups[2] + ops[last + 1:]),
                   "CFG contract issued command belongs to another source operation")

    def test_redirect_delay_slot_is_a_local_cfg_prerequisite(self) -> None:
        machine = example("loop")
        branch = next(n for n, line in enumerate(machine.splitlines()) if '"atlas.branch"' in line)
        self.assertIn('kind = "addi"', machine.splitlines()[branch + 1])
        result = run("atlas-opt", rewrite_line(machine, branch + 1, lambda line: line.replace("dst = 0 : i32", "dst = 9 : i32")),
                     "--verify-atlas-generated-schedule")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("NOP", result.stderr)

    def test_structured_llvm_rechecks_source_scalar_corruption(self) -> None:
        structured = run("atlas-opt", example("loop", timed=True), "--convert-atlas-to-llvm-calls")
        self.assertEqual(structured.returncode, 0, structured.stderr)
        self.assertIn("atlas.virtual_cfg_contract", structured.stdout)
        good = run("atlas-opt", structured.stdout, "--finalize-atlas-llvm-calls")
        self.assertEqual(good.returncode, 0, good.stderr)
        changed = rewrite_line(structured.stdout, lambda line: 'atlas.source_op = "atlas.alu_reg"' in line and 'kind = "slt"' in line
                               and "atlas.virtual_scalar_result" in line, lambda line: line.replace('kind = "slt"', 'kind = "sltu"'))
        finalize_rejects(self, changed, SCALAR)


if __name__ == "__main__":
    unittest.main()
