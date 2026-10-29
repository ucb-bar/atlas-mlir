"""Source-only frontend timing, operand access and compiler capabilities."""
from pathlib import Path
import os
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import atlas_derive_frontend as frontend
from model_evidence import ModelSourceFixture


@unittest.skipUnless(os.environ.get("ATLAS_DERIVE_MODEL_ROOT"), "set ATLAS_DERIVE_MODEL_ROOT for native source contracts")
class SourceContractTest(ModelSourceFixture, unittest.TestCase):
    source_shapes = frontend._SHAPES

    def derive(self, sources=None, **kwargs):
        return frontend.derive_frontend_sources(sources or self.sources, self.scalar,
                                                fetch_width=kwargs.get("fetch_width", self.cfg.fetch_width))

    def test_delay_immediate_mask_is_extracted_and_propagates(self):
        source = self.mutated("core.py", "self.delay_counter = int(insn.imm) & 0xFFF",
                              "self.delay_counter = int(insn.imm) & 0x7FF")
        expected = self.derive()
        expected["delay"]["immediate_mask"] = 2047
        self.assertEqual(self.derive(source), expected)

    def test_delay_countdown_change_has_precise_unsupported_error(self):
        source = self.mutated("core.py", "self.delay_counter -= 1", "self.delay_counter -= 2")
        with self.assertRaisesRegex(frontend.UnsupportedModelTiming, "only unit countdown reaches zero"):
            self.derive(source)

    def test_ordinary_issue_and_redirect_order_mutations_fail_closed(self):
        for source in (
            self.mutated("core.py", "self.cycle_count += 1", "self.cycle_count += 2"),
            self.mutated("core.py", "target = self._tick_s1()", "target = None"),
            self.mutated("ifu.py", "fetch_address=pc", "fetch_address=self.arch_state.npc"),
            self.mutated("core.py", "state.in_delay_slot = True", "state.in_delay_slot = False"),
        ):
            with self.assertRaisesRegex(frontend.UnsupportedModelTiming, "edge/order semantics"):
                self.derive(source)

    def test_scalar_staging_and_completion_mutations_fail_precisely(self):
        source = self.mutated("exu.py", "uop.execute_delay = 1", "uop.execute_delay = 2")
        with self.assertRaisesRegex(frontend.UnsupportedModelTiming, "deferred completion is one tick"):
            self.derive(source)
        source = self.mutated("exu.py", "uop.insn.exec(self.arch_state)", "self._deferred_exec = uop")
        with self.assertRaisesRegex(frontend.UnsupportedModelTiming, "ScalarExecutionUnit.tick"):
            self.derive(source)

    def test_isa_role_branch_link_and_control_mutations_fail_precisely(self):
        for old, new in (
            ("self.rd, state.xrf[self.rs1] + _sign_extend", "self.rd, state.xrf[self.rs2] + _sign_extend"),
            ("class BEQ(ScalarBranchImm, SBType", "class BEQ(ScalarBranchImm, IType"),
            ("state.write_xrf(self.rd, state.execute_pc + 1)", "state.write_xrf(self.rd, state.execute_pc + 2)"),
            ("state.write_csrf(self.imm, state.read_xrf(self.rs1))", "state.write_csrf(self.imm, 0)"),
        ):
            with self.subTest(mutation=old), self.assertRaisesRegex(frontend.UnsupportedModelTiming, "scalar ISA execution/access semantics"):
                self.derive(self.mutated("isa_definition.py", old, new))

    def test_fetch_width_change_is_explicitly_unsupported(self):
        with self.assertRaisesRegex(frontend.UnsupportedModelTiming, "one S1 issue slot"):
            self.derive(fetch_width=2)

if __name__ == "__main__":
    unittest.main()
