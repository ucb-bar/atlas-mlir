"""Native Core fault pairs validate fatal errors without modifying model code."""
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
from atlas_model_diagnostics import ModelTimingValidationError
from atlas_validate_runtime import MAX_CYCLES, native_case, runtime_cases, safe_path, validate_runtime
from model_evidence import isolate_model_imports


class RuntimeValidationBoundsTest(unittest.TestCase):
    def test_cycle_bound_is_explicit_and_finite_before_model_access(self):
        for bound in (0, -1, MAX_CYCLES + 1, 1.0, True):
            with self.subTest(bound=bound), self.assertRaisesRegex(ValueError, "cycle bound"):
                validate_runtime(Path("unused-model"), max_cycles=bound)

    def test_assertions_are_required_before_model_access(self):
        with patch("atlas_validate_runtime.sys.flags") as flags:
            flags.optimize = 1
            with self.assertRaisesRegex(ModelTimingValidationError, "requires Python assertions"):
                validate_runtime(Path("unused-model"))


@unittest.skipUnless(os.environ.get("ATLAS_DERIVE_MODEL_ROOT"),
                     "set ATLAS_DERIVE_MODEL_ROOT for native Core execution")
class NativeRuntimeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = Path(os.environ["ATLAS_DERIVE_MODEL_ROOT"])
        isolate_model_imports(cls, str(cls.model))
        cls.report = validate_runtime(cls.model)
        from atlas_model_timing import configuration
        from atlas_derive_curve import resolve_curve_path
        from npu_model.configs import isa_definition as isa
        cls.cfg, _ = configuration()
        cls.cfg.dma_memory_params = dict(cls.cfg.dma_memory_params,
                                        curves=json.loads(resolve_curve_path(cls.cfg).read_bytes()))
        cls.cases = {case.name: case for case in runtime_cases(isa, cls.cfg)}

    def test_full_core_pairs_have_both_native_success_and_expected_failure(self):
        self.assertEqual(self.report["authority"], "validation_only")
        self.assertEqual(self.report["execution"], "unmodified_native_core")
        self.assertEqual((self.report["positive_cases"], self.report["negative_cases"]), (5, 5))
        self.assertTrue(self.report["passed"])
        coverage = {}
        for row in self.report["cases"]:
            with self.subTest(case=row["name"]):
                self.assertIs(row["ignore_runtime_errors"], False)
                self.assertLessEqual(row["cycles"], MAX_CYCLES)
                coverage.setdefault(row["mechanism"], set()).add(row["expectation"])
                expected = self.cases[row["name"]]
                if row["expectation"] == "positive":
                    self.assertEqual(row["observed"], "ecall_drained")
                else:
                    self.assertEqual(row["observed"], "rejected")
                    self.assertEqual(row["exception_type"], expected.expected_type)
                    self.assertRegex(row["exception_message"], "^" + expected.expected_message + "$")
        self.assertEqual(set(coverage), {"vpu_issue_busy", "lsu_vmem_bank", "mrf_reservation",
                                         "vpu_pair_alignment", "branch_delay_slot"})
        self.assertTrue(all(values == {"positive", "reject"} for values in coverage.values()))

    def test_missing_expected_fault_is_not_a_passing_negative_control(self):
        case = replace(self.cases["runtime_positive"], expected_type="RuntimeError",
                       expected_message="VPU pair-write destination bank must be even")
        with self.assertRaisesRegex(ModelTimingValidationError, "did not propagate expected RuntimeError"):
            native_case(case, self.cfg)

    def test_an_unrelated_native_fault_cannot_satisfy_negative_control(self):
        case = replace(self.cases["issue_busy_negative"], expected_type="BankConflictError")
        with self.assertRaisesRegex(ModelTimingValidationError, "raised unexpected RuntimeError: VPU command"):
            native_case(case, self.cfg)

    def test_bound_exhaustion_is_a_validation_failure(self):
        with self.assertRaisesRegex(ModelTimingValidationError, "did not reach ECALL within 1 cycles"):
            native_case(self.cases["runtime_positive"], self.cfg, max_cycles=1)

    def test_native_scorer_rejects_all_faults_and_accepts_paired_programs(self):
        from atlas_model_scorer import ModelScorer, _model_content, selected_model
        _model_content.cache_clear()
        self.addCleanup(_model_content.cache_clear)
        _, source, data, config = selected_model(self.model)
        scorer = ModelScorer(self.model)
        evidence = dict(source_sha256=source, numerical_data_sha256=data, configuration_sha256=config)
        from npu_model.configs import isa_definition as isa

        def document(case):
            instructions = case.instructions
            if case.scalar_base:
                self.assertEqual(case.scalar_base % 4096, 0)
                instructions = (isa.LUI(rd=1, imm=case.scalar_base >> 12), *instructions)
            return dict(model_evidence=evidence,
                        instructions=[dict(word_u32=insn.to_bytecode()) for insn in instructions])

        for mechanism in ("issue_busy", "bank", "mrf", "runtime", "s1"):
            with self.subTest(mechanism=mechanism):
                self.assertGreater(scorer(document(self.cases[mechanism + "_positive"])), 0)
                case = self.cases[mechanism + "_negative"]
                with self.assertRaises(RuntimeError) as raised:
                    scorer(document(case))
                self.assertEqual(type(raised.exception).__name__, case.expected_type)
                self.assertRegex(str(raised.exception), "^" + case.expected_message + "$")


@unittest.skipUnless(os.environ.get("ATLAS_DERIVE_MODEL_ROOT"),
                     "set ATLAS_DERIVE_MODEL_ROOT for production source refusal checks")
class ProductionRuntimeRefusalTest(unittest.TestCase):
    """Sparse source fixtures prove refusal occurs before importing native code."""

    @classmethod
    def setUpClass(cls):
        model = safe_path(os.environ["ATLAS_DERIVE_MODEL_ROOT"])
        manifest = json.loads((TOOLS / "model-semantics.json").read_text())
        cls.sources = {}
        for relative in manifest["files"]:
            if Path(relative).is_absolute() or ".." in Path(relative).parts:
                raise AssertionError("guard manifest source must be relative")
            cls.sources[relative] = safe_path(model / "npu_model" / relative).read_text()

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = safe_path(directory.name)
        self.model = self.root / "model"
        for relative, source in self.sources.items():
            path = safe_path(self.model / "npu_model" / relative)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(source)
        # Facts hashes numerical data before checking source semantics. The
        # empty directory permits that read but cannot support native imports.
        (self.model / "npu_model/configs/data").mkdir(parents=True)
        self.environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1",
                                OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")

    def run_isolated(self, arguments):
        return subprocess.run([sys.executable, "-B", *arguments], cwd=TOOLS.parent,
                              env=self.environment, text=True, capture_output=True, timeout=60)

    def assert_production_refusal(self, relative, old, new, helper):
        path = self.model / "npu_model" / relative
        source = path.read_text()
        self.assertEqual(source.count(old), 1, "audit mutation must alter exactly one selected source site")
        path.write_text(source.replace(old, new, 1))
        # Both subprocess entry points restrict affinity before importing any
        # model or numerical module. They cannot inherit baseline imports.
        affinity = "import os; os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2]); "
        launch_facts = affinity + (
            "import runpy,sys; sys.argv=sys.argv[1:]; "
            "sys.path.insert(0,str(__import__('pathlib').Path(sys.argv[0]).parent)); "
            "runpy.run_path(sys.argv[0],run_name='__main__')")
        out = self.root / "facts"
        facts = self.run_isolated(["-c", launch_facts, str(TOOLS / "atlas_model_facts.py"),
                                   "--npu-model", str(self.model), "--out", str(out)])
        self.assertNotEqual(facts.returncode, 0, facts.stdout)
        self.assertIn("unsupported semantics: " + relative + ":" + helper, facts.stderr)
        self.assertNotIn("Traceback", facts.stderr)
        self.assertFalse((out / "facts.lock.json").exists())

        launch_scorer = affinity + """
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from atlas_model_scorer import ModelScorer
from atlas_model_semantics import UnsupportedModelSemantics
try:
    ModelScorer(Path(sys.argv[2]))({"instructions": []})
except UnsupportedModelSemantics as error:
    assert not any(name == "npu_model" or name.startswith("npu_model.") for name in sys.modules)
    print(str(error))
    raise SystemExit(3)
raise SystemExit("scorer accepted unsupported semantics")
"""
        scorer = self.run_isolated(["-c", launch_scorer, str(TOOLS), str(self.model)])
        self.assertEqual(scorer.returncode, 3, scorer.stdout + scorer.stderr)
        self.assertIn("unsupported semantics: " + relative + ":" + helper, scorer.stdout)
        self.assertNotIn("Traceback", scorer.stderr)

    def test_audit_runtime_handler_mutation_refuses_facts_and_scoring(self):
        self.assert_production_refusal(
            "hardware/core.py", "if not self.ignore_runtime_errors:\n            return False",
            "if not self.ignore_runtime_errors:\n            return True", "Core._handle_runtime_error")

    def test_audit_scalar_truncation_mutation_refuses_facts_and_scoring(self):
        self.assert_production_refusal(
            "hardware/arch_state.py", "self.xrf[rd] = value\n", "self.xrf[rd] = value & 0xFFFF\n",
            "ArchState.write_xrf")


if __name__ == "__main__":
    unittest.main()
