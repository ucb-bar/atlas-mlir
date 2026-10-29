"""Consumed timing values, deterministic regeneration, and model refusals."""
import ast
from contextlib import contextmanager
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
from atlas_derive_arch import derive_arch
from atlas_derive_costs import check_timing_conditions, derive_costs
from atlas_model_diagnostics import UPDATE_GUIDANCE, UnsupportedModelTiming, model_timing_cli
from atlas_model_timing import compress, validate_derived
from model_evidence import NativeModelFixture, model_evidence_available


class FactsErrors(unittest.TestCase):
    def test_event_sweep(self):
        self.assertEqual(compress([3, 1, 2]), dict(count=3, first_age=1, last_age=3, step=1))
        for ages in ([], [1, 1], [1, 2, 4]):
            with self.subTest(ages=ages), self.assertRaises(ValueError):
                compress(ages)

    def test_register_invariance(self):
        for condition in ("insn.vd == 0", "alias == 0"):
            source = "def planner(insn):\n    alias = insn.vd\n    if " + condition + ":\n        return 1"
            node = ast.parse(source).body[0]
            with self.subTest(condition=condition), self.assertRaisesRegex(
                    UnsupportedModelTiming, "register-dependent timing condition"):
                check_timing_conditions(node, {"vd"})

    def test_cli_errors(self):
        for error in (UnsupportedModelTiming("refused"), ValueError("output directory must be new")):
            def fail():
                raise error
            with self.subTest(error=error), self.assertRaises(SystemExit) as result:
                model_timing_cli(fail)()
            self.assertEqual(str(result.exception), str(error) +
                             ("\n" + UPDATE_GUIDANCE if isinstance(error, UnsupportedModelTiming) else ""))
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(model_evidence_available())
        with patch.dict(os.environ, {"ATLAS_TEST_MODEL_FACTS": ""}, clear=True):
            with self.assertRaisesRegex(FileNotFoundError, "ATLAS_TEST_MODEL_FACTS"):
                model_evidence_available()

@unittest.skipUnless(os.environ.get("ATLAS_DERIVE_MODEL_ROOT"), "set ATLAS_DERIVE_MODEL_ROOT")
class CostProbes(NativeModelFixture, unittest.TestCase):
    def test_native_sweep_refuses_wrong_facts(self):
        from atlas_derive_vpu import derive_vpu
        from atlas_derive_mxu import derive_mxu
        from atlas_derive_memory import derive_memory
        blocks = {}
        for derive in (derive_vpu, derive_mxu, derive_memory):
            blocks.update(derive(self.cfg)[0])
        report, _ = validate_derived(self.cfg, blocks, {}, None)
        self.assertEqual(report["observed_runs"], 49)
        for name, block in blocks.items():
            for field in ("events", "first_free_age"):
                changed = deepcopy(block)
                if field == "events":
                    next(iter(changed["events"].values()))["first_age"] += 1
                else:
                    changed[field] += 1
                with self.subTest(name=name, field=field), self.assertRaisesRegex(
                        UnsupportedModelTiming, "native timing differs"):
                    validate_derived(self.cfg, {name: changed}, {}, None)

    def test_operands_and_capacity(self):
        architecture = derive_arch(self.cfg)
        pairs = derive_costs(self.cfg, architecture)
        self.assertEqual(len(pairs), 89)
        self.assertEqual(pairs["vadd.bf16"][1], dict(rd=6, rs1=8, rs2=10, imm=0))
        self.assertEqual(pairs["vmatpop.bf16.acc.mxu0"][1], dict(rd=2, rs1=0, rs2=1, imm=0))
        wider = derive_costs(self.cfg, dict(architecture, bf16_registers_per_tile=4))
        self.assertEqual(wider["vpack.bf16.fp8"][1]["rs2"], 20)
        for values, reason in ((dict(architecture, mreg_count=4), "cost probes exceed"),
                               (dict(architecture, bf16_registers_per_tile=0), "register span must be positive")):
            with self.subTest(reason=reason), self.assertRaisesRegex(UnsupportedModelTiming, reason):
                derive_costs(self.cfg, values)

    def test_imported_model_path_refuses(self):
        import npu_model
        from atlas_model_facts import regenerate
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(npu_model, "__file__", "/wrong/npu_model/__init__.py"):
            with self.assertRaisesRegex(UnsupportedModelTiming, "imported npu_model differs"):
                regenerate(os.environ["ATLAS_DERIVE_MODEL_ROOT"], Path(directory) / "facts")


@unittest.skipUnless(os.environ.get("ATLAS_DERIVE_MODEL_ROOT"), "set ATLAS_DERIVE_MODEL_ROOT")
class Regeneration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(temporary.cleanup)
        cls.root = Path(temporary.name)
        cls.model = cls.root / "model"
        source = Path(os.environ["ATLAS_DERIVE_MODEL_ROOT"]).absolute()
        if source.is_symlink() or source.resolve() != source:
            raise AssertionError("selected model path contains a symlink")
        for directory, children, names in os.walk(source / "npu_model", followlinks=False):
            parent = Path(directory)
            children[:] = [name for name in children if not cls.blocked(name) and name != "__pycache__"]
            for name in [*children, *[name for name in names if not cls.blocked(name)]]:
                if (parent / name).is_symlink():
                    raise AssertionError("selected model source contains a symlink")
            for name in names:
                if cls.blocked(name):
                    continue
                path = parent / name
                if path.suffix not in {".py", ".json", ".yaml"} and "data" not in path.relative_to(source).parts:
                    continue
                target = cls.model / path.relative_to(source)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
        for command in (("init", "-q"), ("add", "npu_model"),
                        ("-c", "commit.gpgsign=false", "-c", "user.name=Fixture",
                         "-c", "user.email=fixture@example.invalid",
                         "commit", "-qm", "model fixture")):
            subprocess.run(["git", "-C", str(cls.model), *command], check=True, capture_output=True)
        cls.env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1",
                       OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        for index in (1, 2):
            result = cls.regenerate(cls.root / f"facts-{index}")
            if result.returncode:
                raise AssertionError(result.stderr)
        cls.facts = json.loads((cls.root / "facts-1/model-timing.json").read_text())

    @staticmethod
    def blocked(name):
        return any(word in name.lower() for word in ("ham" + "mer", "vl" + "si"))

    @classmethod
    def regenerate(cls, output):
        return subprocess.run([sys.executable, "-B", str(TOOLS / "atlas_model_facts.py"),
                               "--npu-model", str(cls.model), "--out", str(output)],
                              cwd=cls.root, env=cls.env, text=True, capture_output=True)

    @contextmanager
    def mutation(self, relative, old, new):
        path = self.model / "npu_model" / relative
        original = path.read_text()
        self.assertIn(old, original, f"mutation must alter {relative}")
        changed = original.replace(old, new, 1)
        self.assertNotEqual(original, changed)
        path.write_text(changed)
        try:
            yield
        finally:
            path.write_text(original)

    def refuses(self, reason):
        with tempfile.TemporaryDirectory(dir=self.root) as directory:
            result = self.regenerate(Path(directory) / "facts")
        self.assertNotEqual(result.returncode, 0)
        self.assertRegex(result.stderr, reason)
        self.assertIn(UPDATE_GUIDANCE, result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_regeneration_and_consumed_values(self):
        first, second = self.root / "facts-1", self.root / "facts-2"
        self.assertEqual({p.name: p.read_bytes() for p in first.iterdir()},
                         {p.name: p.read_bytes() for p in second.iterdir()})
        facts = self.facts
        blocks = {block["name"]: block for block in facts["op_timing"]}
        self.assertEqual(len(blocks), 49)
        self.assertEqual(blocks["vadd.bf16"]["events"]["read0"],
                         dict(count=64, first_age=0, last_age=63, step=1))
        self.assertEqual(blocks["vadd.bf16"]["events"]["write0"],
                         dict(count=64, first_age=2, last_age=65, step=1))
        self.assertEqual((blocks["vadd.bf16"]["first_free_age"], blocks["vload"]["first_free_age"]), (66, 35))
        self.assertEqual(blocks["vload"]["events"]["vmem_read"],
                         dict(count=32, first_age=1, last_age=32, step=1))
        self.assertEqual([blocks[f"vmatmul.mxu{i}"]["events"]["acc_write"]["first_age"] for i in (0, 1)], [63, 3])
        weights = facts["mxu_weight_timing"]["engines"]
        self.assertEqual((weights["mxu0"]["use_rows"][0], weights["mxu0"]["use_rows"][-1]),
                         (dict(first_age=1, last_age=63), dict(first_age=32, last_age=94)))
        self.assertEqual(weights["mxu1"]["use_rows"], [dict(first_age=1, last_age=32)] * 32)
        dma = facts["dma_timing"]
        maximum = max(dma["min_latency"], *(value for row in dma["curves"] for value in row.values()))
        self.assertEqual((math.ceil(maximum) + dma["issue_interval_cycles"], dma["startup_cycles"]),
                         (47, dict(load=1, store=3)))
        self.assertEqual(dma["retire_to_wait_issue_cycles"], 1)
        self.assertEqual((facts["resources"]["capacity"]["vpu_slots"],
                          facts["resources"]["scalar_lsu"]["load_response_gap"]), (2, 3))
        self.assertEqual(len(facts["architecture"]), 16)
        self.assertEqual([facts["architecture"][key] for key in
                          ("mreg_count", "mreg_bytes", "vmem_bank_count", "scalar_word_bytes")], [64, 1024, 6, 4])
        self.assertEqual(facts["reservation_releases"]["vadd.bf16"]["mreg_write"],
                         dict(first_age=0, release_age=65))
        roles = facts["frontend_timing"]["instructions"]["add"]
        self.assertEqual(roles, dict(reads=[dict(resource="xreg", operand="rs1"),
                                           dict(resource="xreg", operand="rs2")],
                                    writes=[dict(resource="xreg", operand="rd")]))

    def test_every_guarded_source_refuses(self):
        files = json.loads((TOOLS / "model-semantics.json").read_text())["files"]
        for relative in files:
            source = (self.model / "npu_model" / relative).read_text()
            with self.subTest(file=relative), self.mutation(relative, source, source + "\nFACTS_TEST_CHANGE = 1\n"):
                self.refuses("unsupported semantics: " + re.escape(relative) + ":module/class bindings")

    def test_timing_source_and_configuration_refusals(self):
        cases = (
            ("hardware/vpu.py", "ISSUE_SLOTS = 2", "ISSUE_SLOTS = 4", "unsupported VPU source slot capacity"),
            ("hardware/vpu.py", '_GROUPS = (', '_GROUPS = ({"vadd.bf16"},', "VPU timing semantics"),
            ("hardware/mxu.py", "0 <= age < 32", "0 <= age < 31", "unsupported MXU"),
            ("hardware/lsu.py", "self.cycle += 1", "self.cycle += 2", "LSU/XLU control-flow semantics"),
            ("hardware/xlu.py", "TRANSPOSE_CYCLES = 66", "TRANSPOSE_CYCLES = 65", "LSU/XLU control-flow semantics"),
            ("hardware/bank_conflict.py", "bank = reg % 32", "bank = reg % 16", "VPU timing semantics"),
            ("hardware/dma.py", "self.cycle += 1", "self.cycle += 2", "DMA source semantics: dma.py"),
            ("hardware/memory_backend.py", "self._last_ready = 0.0", "self._last_ready = 1.0", "DMA source semantics"),
            ("configs/isa_definition.py", "class VADD_BF16(", "class MISSING_VADD_BF16(",
             "isa_definition.py:(?:MISSING_)?VADD_BF16.exec"),
            ("configs/hardware/default.py", "mrf_depth=32", "mrf_depth=16", "arch_state_config.mrf_depth"),
            ("hardware/config.py", "dma_beat_bytes: int = 32", "dma_beat_bytes: int = 16",
             "capability: dma_beat_bytes"),
            ("hardware/config.py", '"curves": "ee290sim_vcs_probe.json"',
             '"curves": "ee290sim_vcs_probe.json", "latency_mode": "knee"', "dma_memory_params.latency_mode"),
        )
        for relative, old, new, reason in cases:
            with self.subTest(file=relative, mutation=old), self.mutation(relative, old, new):
                self.refuses(reason)

    def test_cli_requires_new_output_directory(self):
        result = self.regenerate(self.root / "facts-1")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--out must be a new directory", result.stderr)
        self.assertNotIn(UPDATE_GUIDANCE, result.stderr)

    def test_curve_data_refuses_unsupported_mode(self):
        with self.mutation("configs/memory_curves/ee290sim_vcs_probe.json",
                           '"latencyMode": "unloaded"', '"latencyMode": "knee"'):
            self.refuses("unsupported capability: dma_memory_params.latency_mode")
