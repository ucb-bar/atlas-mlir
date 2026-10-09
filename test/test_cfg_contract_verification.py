"""Source CFG/scalar/edge correspondence survives every emitted handoff."""
from __future__ import annotations

from pathlib import Path
import re
import unittest

from test_virtual_lowering import ROOT, EXAMPLES, lower, run


class CFGContractVerificationTest(unittest.TestCase):
    def check(self, text: str, *, accepted: bool) -> None:
        boundaries = [("atlas-opt", ("--verify-atlas-generated-schedule",))]
        if 'atlas.timing_state = "timed"' in text:
            boundaries.append(("atlas-emit", ()))
        for tool, flags in boundaries:
            result = run(tool, text, *flags)
            self.assertEqual(result.returncode == 0, accepted, result.stderr)
            if not accepted:
                self.assertIn("CFG contract", result.stderr)

    def changed_line(self, machine: str, predicate, change) -> str:
        lines = machine.splitlines()
        index = next(n for n, line in enumerate(lines) if predicate(line))
        changed = change(lines[index])
        self.assertNotEqual(changed, lines[index])
        lines[index] = changed
        return "\n".join(lines) + "\n"

    def test_existing_branches_loops_and_simultaneous_tensor_copies(self) -> None:
        for name in ("branch", "dynamic_branch", "loop", "swap_loop", "vpu"):
            source = (EXAMPLES / f"virtual_bf16_{name}_program.mlir").read_text()
            for timed in (False, True):
                with self.subTest(name=name, timed=timed):
                    machine = lower(source, timed=timed)
                    self.assertIn("atlas.virtual_cfg_contract", machine)
                    self.check(machine, accepted=True)
            reordered = run("atlas-opt", lower(source, timed=False),
                            "--schedule-atlas-stream")
            self.assertEqual(reordered.returncode, 0, reordered.stderr)
            self.check(reordered.stdout, accepted=True)

    def test_branch_polarity_and_condition_corruption(self) -> None:
        source = (EXAMPLES / "virtual_bf16_dynamic_branch_program.mlir").read_text()
        machine = lower(source, timed=False)
        predicate = lambda line: '"atlas.branch"' in line and "atlas.virtual_cfg_branch" in line
        for transform in (lambda line: line.replace('kind = "bne"', 'kind = "beq"'),
                          lambda line: re.sub(r"lhs = \d+ : i32", "lhs = 0 : i32", line)):
            self.check(self.changed_line(machine, predicate, transform), accepted=False)

    def test_ordinary_branch_cannot_claim_pack_helper_exemption(self) -> None:
        machine = lower((EXAMPLES / "virtual_bf16_loop_program.mlir").read_text(), timed=False)
        predicate = lambda line: '"atlas.branch"' in line and "atlas.virtual_cfg_branch" in line
        forged = self.changed_line(machine, predicate,
                    lambda line: line.replace("}> {", '}> {atlas.virtual_cfg_helper = "pack", ', 1))
        self.check(forged, accepted=False)
        comparison = next(line for line in machine.splitlines()
                          if "atlas.virtual_scalar_result" in line and 'kind = "slt"' in line)
        source_id = re.search(r"atlas.virtual_cfg_source = (\d+) : i32", comparison)[1]
        forged_source = self.changed_line(machine, predicate,
                    lambda line: line.replace("}> {", f'}}> {{atlas.virtual_cfg_helper = "pack", atlas.virtual_cfg_source = {source_id} : i32, ', 1))
        self.check(forged_source, accepted=False)

    def test_valid_machine_target_with_wrong_source_successor(self) -> None:
        machine = lower((EXAMPLES / "virtual_bf16_branch_program.mlir").read_text(), timed=False)
        lines = [line for line in machine.splitlines() if ': (!atlas.state)' in line]
        branch_pc = next(n for n, line in enumerate(lines) if '"atlas.branch"' in line)
        target_pc = next(n for n, line in enumerate(lines) if "atlas.virtual_cfg_block = 3 : i32" in line)
        changed = self.changed_line(machine,
                    lambda line: '"atlas.branch"' in line,
                    lambda line: re.sub(r"offset_bytes = -?\d+ : i32",
                                      f"offset_bytes = {2 * (target_pc - branch_pc)} : i32", line))
        self.check(changed, accepted=False)

    def test_scalar_add_and_comparison_corruption(self) -> None:
        machine = lower((EXAMPLES / "virtual_bf16_loop_program.mlir").read_text(), timed=False)
        for kind, replacement in (("add", "sub"), ("slt", "sltu")):
            changed = self.changed_line(machine,
                    lambda line: '"atlas.alu_reg"' in line and f'kind = "{kind}"' in line
                    and "atlas.virtual_scalar_result" in line,
                    lambda line: line.replace(f'kind = "{kind}"', f'kind = "{replacement}"'))
            self.check(changed, accepted=False)

    def test_simultaneous_scalar_copy_cycle(self) -> None:
        source = (EXAMPLES / "virtual_bf16_swap_loop_program.mlir").read_text()
        source = source.replace("%c1 = arith.constant 1 : i32", "%c1 = arith.constant 1 : i32\n    %c2 = arith.constant 2 : i32")
        source = source.replace("%c0 : !atlas.virtual_state, !atlas.virtual_bf16, !atlas.virtual_bf16, i32)", "%c0, %c1, %c2 : !atlas.virtual_state, !atlas.virtual_bf16, !atlas.virtual_bf16, i32, i32, i32)")
        source = source.replace("%i: i32):", "%i: i32, %u: i32, %v: i32):")
        source = source.replace("%next_i : !atlas.virtual_state, !atlas.virtual_bf16, !atlas.virtual_bf16, i32)", "%next_i, %v, %u : !atlas.virtual_state, !atlas.virtual_bf16, !atlas.virtual_bf16, i32, i32, i32)")
        machine = lower(source, timed=False)
        self.check(machine, accepted=True)
        changed = self.changed_line(machine,
                    lambda line: '"atlas.alu_imm"' in line and "atlas.virtual_cfg_edge" in line
                    and 'kind = "addi"' in line and "immediate = 0 : i32" in line
                    and not "dst = 0 : i32" in line,
                    lambda line: line.replace("immediate = 0 : i32", "immediate = 1 : i32"))
        self.check(changed, accepted=False)

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
        self.check(machine, accepted=True)
        lines = machine.splitlines()
        groups = {n: [] for n in (1,2,3,4)}
        operations = []
        first = last = None
        for n, line in enumerate(lines):
            if ': (!atlas.state)' not in line:
                continue
            operations.append((n,line))
            match = re.search(r"atlas.virtual_cfg_source = (\d+) : i32",line)
            if match:
                group = int(match[1]); groups[group].append(line)
                first = n if first is None else min(first,n); last = n
        result_ids = {n: int(re.search(r"atlas.virtual_tensor_result = (\d+) : i32",
                      next(line for line in groups[n] if "atlas.virtual_tensor_result" in line))[1])
                      for n in (1,3)}
        for n, other in ((1,3),(3,1)):
            for index, line in enumerate(groups[n]):
                for name in ("atlas.virtual_cfg_source","atlas.virtual_cfg_operation"):
                    line = re.sub(rf"{name} = {n} : i32",f"{name} = {other} : i32",line)
                line = re.sub(r"atlas.virtual_tensor_result = \d+ : i32",
                              f"atlas.virtual_tensor_result = {result_ids[other]} : i32",line)
                groups[n][index] = line
        ordered = [line for n,line in operations if n < first]
        ordered += groups[1] + groups[4] + groups[3] + groups[2]
        ordered += [line for n,line in operations if n > last]
        start_line = next(line for line in lines if '"atlas.start"' in line)
        previous = re.search(r"%(\w+) =",start_line)[1]
        rebuilt = []
        for index,line in enumerate(ordered):
            current = f"cfgmutation{index}"
            line = re.sub(r"^\s*%\w+ =",f"  %{current} =",line)
            line = re.sub(r'("atlas\.[^"]+"\()%(\w+)(\))',rf"\g<1>%{previous}\g<3>",line)
            rebuilt.append(line); previous = current
        first_operation = operations[0][0]
        changed = "\n".join(lines[:first_operation] + rebuilt + lines[operations[-1][0]+1:]) + "\n"
        self.check(changed, accepted=False)

    def test_redirect_delay_slot_is_a_local_cfg_prerequisite(self) -> None:
        machine = lower((EXAMPLES / "virtual_bf16_loop_program.mlir").read_text(), timed=False)
        lines = machine.splitlines()
        branch = next(n for n,line in enumerate(lines) if '"atlas.branch"' in line)
        self.assertIn('kind = "addi"',lines[branch+1])
        lines[branch+1] = lines[branch+1].replace("dst = 0 : i32","dst = 9 : i32")
        result = run("atlas-opt","\n".join(lines) + "\n","--verify-atlas-generated-schedule")
        self.assertNotEqual(result.returncode,0)
        self.assertIn("NOP",result.stderr)

    def test_structured_llvm_rechecks_source_scalar_corruption(self) -> None:
        source = (EXAMPLES / "virtual_bf16_loop_program.mlir").read_text()
        result = run("atlas-opt", lower(source), "--convert-atlas-to-llvm-calls")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("atlas.virtual_cfg_contract", result.stdout)
        good = run("atlas-opt", result.stdout, "--finalize-atlas-llvm-calls")
        self.assertEqual(good.returncode, 0, good.stderr)
        changed = self.changed_line(result.stdout,
                    lambda line: 'atlas.source_op = "atlas.alu_reg"' in line
                    and 'kind = "slt"' in line and "atlas.virtual_scalar_result" in line,
                    lambda line: line.replace('kind = "slt"', 'kind = "sltu"'))
        bad = run("atlas-opt", changed, "--finalize-atlas-llvm-calls")
        self.assertNotEqual(bad.returncode, 0)
        self.assertIn("CFG contract", bad.stderr)

    def test_missing_contract_and_wrong_metadata_types_fail(self) -> None:
        machine = lower((EXAMPLES / "virtual_bf16_loop_program.mlir").read_text(), timed=False)
        for before, after in (("atlas.virtual_cfg_contract", "atlas.removed_cfg_contract"),
                              ("atlas.virtual_cfg_branch = 1 : i32", "atlas.virtual_cfg_branch = 1 : i64")):
            with self.subTest(before=before):
                self.assertIn(before, machine)
                self.check(machine.replace(before, after, 1), accepted=False)


if __name__ == "__main__":
    unittest.main()
