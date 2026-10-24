"""Source-derived PE windows, whole-row projection, and native boundary checks."""
import copy
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import atlas_derive_mxu as mxu
from atlas_model_schema import validate_mxu_weight_timing
from atlas_validate_mxu_weights import validate_weight_timing
from model_evidence import NativeModelFixture


def weight_fixture():
    engines = {}
    for engine, skew in (("mxu0", 1), ("mxu1", 0)):
        engines[engine] = dict(use_window=dict(first_age=1, last_age=32, row_step=skew, column_step=skew),
                               use_rows=[dict(first_age=1 + i * skew, last_age=32 + (i + 31) * skew)
                                         for i in range(32)],
                               reads_before_writes=True, resource_row_axis="output_column")
    return dict(schema="atlas.mxu_weight_timing.v1", engines=engines)


class WeightSchemaTest(unittest.TestCase):
    def test_row_mapping_bounds_and_edge_order_fail_closed(self):
        facts = weight_fixture()
        self.assertEqual(validate_mxu_weight_timing(facts), facts)
        for change in (lambda x: x.update(reads_before_writes=False),
                       lambda x: x.update(resource_row_axis="inner"),
                       lambda x: x["use_rows"][7].update(last_age=69),
                       lambda x: x["use_window"].update(first_age=True),
                       lambda x: x["use_rows"].pop(),
                       lambda x: x["use_window"].update(column_step=4094)):
            changed = copy.deepcopy(facts)
            change(changed["engines"]["mxu0"])
            with self.assertRaisesRegex(ValueError, "MXU weight"):
                validate_mxu_weight_timing(changed)


@unittest.skipUnless(os.environ.get("ATLAS_DERIVE_MODEL_ROOT"), "set ATLAS_DERIVE_MODEL_ROOT for native weight checks")
class NativeWeightTest(NativeModelFixture, unittest.TestCase):
    def test_exact_pe_windows_and_output_column_row_unions(self):
        self.assertEqual(mxu.derive_weight_timing(self.cfg), weight_fixture())
        facts = mxu.derive_weight_timing(self.cfg)
        for i in range(32):
            row = facts["engines"]["mxu0"]["use_rows"][i]
            self.assertEqual(row, dict(first_age=1 + i, last_age=63 + i))
            for j in range(32):
                self.assertLessEqual(row["first_age"], 1 + i + j)
                self.assertGreaterEqual(row["last_age"], 32 + i + j)

    def test_source_only_derivation_constructs_no_state_or_unit(self):
        from npu_model.hardware.arch_state import ArchState
        from npu_model.hardware.mxu import _MatrixExecutionUnit
        with patch.object(ArchState, "__init__", side_effect=AssertionError("numerical state")), \
                patch.object(_MatrixExecutionUnit, "tick", side_effect=AssertionError("native execution")):
            self.assertEqual(mxu.derive_weight_timing(self.cfg), weight_fixture())

    def test_changed_weight_masks_mapping_and_read_order_refuse(self):
        source, checker = mxu._sources(self.cfg)
        mutations = (
            ("1 <= age <= 32", "1 <= age < 32"),
            ("row <= 31", "row < 31"),
            ("age - 1 - _GRID[:, None] - _GRID[None, :]", "age - 1 - _GRID[:, None]"),
            ("used & (age == 1)", "used & (age == 2)"),
            ("read_wb_u8(self.mxu, weight_push.weight)[row]", "read_wb_u8(self.mxu, weight_push.weight)[:, row]"),
            ("self._sample_weights(op, self.cycle - op.issued)", "self._sample_weights(op, self.cycle - op.issued - 1)"),
        )
        for old, new in mutations:
            self.assertIn(old, source)
            with self.subTest(mutation=old), patch.object(mxu, "_sources", return_value=(source.replace(old, new), checker)):
                with self.assertRaises(mxu.UnsupportedMXU):
                    mxu.derive_weight_timing(self.cfg)

    def test_native_first_last_use_and_one_cycle_early_overwrites(self):
        report = validate_weight_timing(self.cfg, mxu.derive_weight_timing(self.cfg))
        self.assertTrue(report["passed"])
        self.assertEqual((report["isolated_runs"], report["boundary_runs"]), (4, 4))
        self.assertEqual((report["consumer_runs"], report["mutable_pe_runs"]), (3, 8))
        boundaries = {check["engine"]: (check["accepted_gap"], check["rejected_gap"])
                      for check in report["checks"] if check["kind"] == "native_overwrite_boundary"}
        self.assertEqual(boundaries, dict(mxu0=(63, 62), mxu1=(32, 31)))
        consumers = {check["engine"]: (check["accepted_gap"], check["rejected_gap"])
                     for check in report["checks"] if check["kind"] == "native_consumer_boundary"}
        self.assertEqual(consumers, dict(mxu0=(1, None), mxu1=(32, 31)))
        mutable = {check["engine"]: (check["accepted_first_age"], check["rejected_first_late_age"],
                                      check["rejected_last_age"], check["accepted_post_last_age"])
                   for check in report["checks"] if check["kind"] == "native_mutable_pe_boundaries"}
        self.assertEqual(mutable, dict(mxu0=(11, 12, 42, 43), mxu1=(1, 2, 32, 33)))

    def test_incorrect_derived_mask_fails_native_gate(self):
        from atlas_model_diagnostics import ModelTimingValidationError
        facts = mxu.derive_weight_timing(self.cfg)
        facts["engines"]["mxu0"]["use_window"]["last_age"] -= 1
        with self.assertRaisesRegex(ModelTimingValidationError, "weight-use mask mismatch"):
            validate_weight_timing(self.cfg, facts)


if __name__ == "__main__":
    unittest.main()
