"""Refusal diagnostics preserve error types and update guidance."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import atlas_model_timing as timing
from atlas_model_diagnostics import UPDATE_GUIDANCE, UnsupportedModelTiming, model_timing_cli


class DiagnosticBoundaryTest(unittest.TestCase):
    def test_cli_adds_guidance_to_refusals(self):
        error = UnsupportedModelTiming("refused")
        self.assertEqual(str(error), "refused")
        with self.assertRaises(SystemExit) as raised:
            model_timing_cli(lambda: self.raise_error(error))()
        self.assertEqual(str(raised.exception), "refused\n" + UPDATE_GUIDANCE)

    @staticmethod
    def raise_error(error):
        raise error

    def test_cli_input_errors_have_no_model_update_guidance(self):
        for error in (ValueError("output directory must be new"), FileNotFoundError("missing input")):
            with self.subTest(error=error), self.assertRaises(SystemExit) as raised:
                model_timing_cli(lambda: self.raise_error(error))()
            self.assertNotIn(UPDATE_GUIDANCE, str(raised.exception))

    def test_unexpected_cli_exceptions_propagate(self):
        for error in (TypeError("bug"), KeyError("bug"), RuntimeError("bug")):
            with self.subTest(error=error), self.assertRaises(type(error)) as raised:
                model_timing_cli(lambda: self.raise_error(error))()
            self.assertIs(raised.exception, error)

    def test_validator_does_not_turn_programming_errors_into_refusals(self):
        for error in (TypeError("observer bug"), KeyError("observer bug")):
            with self.subTest(error=error), \
                    patch.object(timing, "observe_vpu", side_effect=error), \
                    self.assertRaises(type(error)) as raised:
                timing.validate_derived(object(), {"probe": {}}, {}, Path("unused"))
            self.assertIs(raised.exception, error)


if __name__ == "__main__":
    unittest.main()
