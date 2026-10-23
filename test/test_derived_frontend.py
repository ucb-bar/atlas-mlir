"""Source-only frontend timing, operand access and compiler capabilities."""
from pathlib import Path
import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import atlas_derive_frontend as frontend
import atlas_derive_scratchpad as scratchpad
import atlas_model_timing as timing
from model_evidence import isolate_model_imports


@unittest.skipUnless(os.environ.get("ATLAS_DERIVE_MODEL_ROOT"), "set ATLAS_DERIVE_MODEL_ROOT for native source contracts")
class SourceContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        isolate_model_imports(cls, os.environ["ATLAS_DERIVE_MODEL_ROOT"])
        cls.cfg, _ = timing.configuration()
        from npu_model.isa import EXU
        root = Path(os.environ["ATLAS_DERIVE_MODEL_ROOT"]) / "npu_model"
        cls.sources = {name: scratchpad.read_source(root / "hardware" / name)
                       for name, _, _ in frontend._SHAPES.keys()}
        cls.sources["isa_definition.py"] = scratchpad.read_source(root / "configs/isa_definition.py")
        cls.scalar = {name: item.__name__ for name, item in cls.cfg.isa.operations.items() if item.exu == EXU.SCALAR}

    def derive(self, sources=None, **kwargs):
        return frontend.derive_frontend_sources(sources or self.sources, self.scalar,
                                                fetch_width=kwargs.get("fetch_width", self.cfg.fetch_width))

    def mutated(self, filename, old, new):
        self.assertIn(old, self.sources[filename], "mutation must alter the selected model source")
        return dict(self.sources, **{filename: self.sources[filename].replace(old, new)})

    def test_pinned_frontend_contract_and_operand_metadata(self):
        value = self.derive()
        self.assertEqual(frontend.derive_frontend(self.cfg), value)
        self.assertEqual(value["ordinary_issue_gap"], 1)
        self.assertEqual(value["delay"], dict(base_gap=1, immediate_mask=4095, decrement_per_tick=1))
        self.assertEqual(value["scalar"], dict(read_age=0, write_age=0, completion_delay=1))
        self.assertEqual(value["control"], dict(delay_slots=1, operand_read_age=0, jal_link_pc_offset=1))
        self.assertEqual(value["instructions"]["addi"]["reads"], [dict(resource="xreg", operand="rs1")])
        self.assertEqual(value["instructions"]["seli"]["writes"], [dict(resource="ereg", operand="rd")])
        self.assertEqual(value["instructions"]["beq"]["writes"], [])
        self.assertEqual(value["instructions"]["jal"]["writes"], [dict(resource="xreg", operand="rd")])
        self.assertEqual(value["instructions"]["ecall"]["kind"], "halt")
        self.assertEqual(value["instructions"]["csrrw"]["kind"], "control")

    def test_delay_immediate_mask_is_extracted_and_propagates(self):
        source = self.mutated("core.py", "self.delay_counter = int(insn.imm) & 0xFFF",
                              "self.delay_counter = int(insn.imm) & 0x7FF")
        expected = self.derive()
        expected["delay"]["immediate_mask"] = 2047
        self.assertEqual(self.derive(source), expected)

    def test_delay_countdown_change_has_precise_unsupported_error(self):
        source = self.mutated("core.py", "self.delay_counter -= 1", "self.delay_counter -= 2")
        with self.assertRaisesRegex(frontend.UnsupportedFrontend, "only unit countdown reaches zero"):
            self.derive(source)

    def test_ordinary_issue_and_redirect_order_mutations_fail_closed(self):
        for source in (
            self.mutated("core.py", "self.cycle_count += 1", "self.cycle_count += 2"),
            self.mutated("core.py", "target = self._tick_s1()", "target = None"),
            self.mutated("ifu.py", "fetch_address=pc", "fetch_address=self.arch_state.npc"),
            self.mutated("core.py", "state.in_delay_slot = True", "state.in_delay_slot = False"),
        ):
            with self.assertRaisesRegex(frontend.UnsupportedFrontend, "edge/order semantics"):
                self.derive(source)

    def test_scalar_staging_and_completion_mutations_fail_precisely(self):
        source = self.mutated("exu.py", "uop.execute_delay = 1", "uop.execute_delay = 2")
        with self.assertRaisesRegex(frontend.UnsupportedFrontend, "deferred completion is one tick"):
            self.derive(source)
        source = self.mutated("exu.py", "uop.insn.exec(self.arch_state)", "self._deferred_exec = uop")
        with self.assertRaisesRegex(frontend.UnsupportedFrontend, "ScalarExecutionUnit.tick"):
            self.derive(source)

    def test_isa_role_branch_link_and_control_mutations_fail_precisely(self):
        for old, new in (
            ("self.rd, state.xrf[self.rs1] + _sign_extend", "self.rd, state.xrf[self.rs2] + _sign_extend"),
            ("class BEQ(ScalarBranchImm, SBType", "class BEQ(ScalarBranchImm, IType"),
            ("state.write_xrf(self.rd, state.execute_pc + 1)", "state.write_xrf(self.rd, state.execute_pc + 2)"),
            ("state.write_csrf(self.imm, state.read_xrf(self.rs1))", "state.write_csrf(self.imm, 0)"),
        ):
            with self.subTest(mutation=old), self.assertRaisesRegex(frontend.UnsupportedFrontend, "scalar ISA execution/access semantics"):
                self.derive(self.mutated("isa_definition.py", old, new))

    def test_fetch_width_change_is_explicitly_unsupported(self):
        with self.assertRaisesRegex(frontend.UnsupportedFrontend, "one S1 issue slot"):
            self.derive(fetch_width=2)

    def test_helpers_never_execute_numerics_or_tick(self):
        from npu_model.hardware.core import Core
        from npu_model.hardware.arch_state import ArchState
        with patch.object(Core, "tick", side_effect=AssertionError("model execution")), \
                patch.object(ArchState, "__init__", side_effect=AssertionError("numerical state")):
            self.assertEqual(frontend.derive_frontend(self.cfg), self.derive())


if __name__ == "__main__":
    unittest.main()
