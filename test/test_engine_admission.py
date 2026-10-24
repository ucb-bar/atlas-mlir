"""Source engine predicates, independent native boundaries and payload identity."""
from copy import deepcopy
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import atlas_derive_admission as derive
import atlas_validate_admission as validate
from model_evidence import isolate_model_imports


class RelationalPredicateTest(unittest.TestCase):
    def setUp(self):
        self.admission = dict(vpu=dict(capacity=2, final_write_age={name: 65 for name in
            ("vpu.relu", "vpu.exp", "vpu.exp2", "vpu.sin", "vpu.add")},
            logic_groups=[["vpu.exp", "vpu.exp2"]], double=["vpu.add"]))

    def test_guard_depends_on_full_live_set_and_operation(self):
        guard = lambda issued, name, age: derive.vpu_blocked(self.admission, issued, name, age)
        self.assertFalse(guard([("vpu.relu", 0)], "vpu.exp2", 1))
        self.assertTrue(guard([("vpu.exp", 0)], "vpu.exp2", 1))
        self.assertTrue(guard([("vpu.relu", 0)], "vpu.add", 1))
        self.assertTrue(guard([("vpu.add", 0)], "vpu.sin", 1))
        self.assertTrue(guard([("vpu.relu", 0), ("vpu.exp", 1)], "vpu.sin", 2))
        self.assertFalse(guard([("vpu.relu", 0), ("vpu.exp", 1)], "vpu.sin", 65))

    def test_receipts_do_not_change_semantics_but_guards_do(self):
        semantic = derive.semantic_admission(self.admission)
        changed = deepcopy(self.admission)
        changed.update(validation=dict(passed=True), producer_sources={"test": "receipt"})
        self.assertEqual(derive.semantic_admission(changed), semantic)
        changed["vpu"]["double"] = []
        self.assertNotEqual(derive.semantic_admission(changed), semantic)

    def test_native_live_endpoint_may_follow_the_last_row_write(self):
        blocks = {"vpu.rsum": dict(engine="vpu", first_free_age=40,
            events=dict(write0=dict(last_age=38), write1=dict(last_age=38)))}
        plans = [("vpu.rsum", "vredsum.row.bf16", None, SimpleNamespace(write_last=39))]
        with patch("atlas_derive_vpu._plans", return_value=plans):
            final = derive._vpu_live_ages(None, blocks)
        self.assertEqual(final, {"vpu.rsum": 39})
        admission = dict(vpu=dict(capacity=2, final_write_age=final, double=["vpu.rsum"], logic_groups=[]))
        self.assertTrue(derive.vpu_blocked(admission, [("vpu.rsum", 0)], "vpu.vliOne", 38))
        self.assertFalse(derive.vpu_blocked(admission, [("vpu.rsum", 0)], "vpu.vliOne", 39))
        for changed in (dict(blocks["vpu.rsum"], first_free_age=41),
                        dict(blocks["vpu.rsum"], events=dict(write0=dict(last_age=40)))):
            with patch("atlas_derive_vpu._plans", return_value=plans), self.assertRaises(derive.UnsupportedAdmission):
                derive._vpu_live_ages(None, {"vpu.rsum": changed})


class AdmissionSamplePlanTest(unittest.TestCase):
    def test_coalesced_reads_still_obey_source_slot_capacity_and_release(self):
        # Deliberately different ages distinguish source-derived placement from
        # the old fixed gap 64 and unrelated RELU final-write age 65.
        facts = dict(op_timing=[
            dict(name="vpu.cmax", events=dict(read0=dict(first_age=3, step=2))),
            dict(name="vpu.relu", events=dict(read0=dict(first_age=1, count=64))),
            dict(name="vlsu.vload", events=dict(vmem_read=dict(first_age=1, last_age=32))),
            dict(name="vlsu.vstore", events=dict(vmem_write=dict(first_age=1, last_age=32)))])
        admission = dict(vpu=dict(capacity=1, logic_groups=[], double=[],
            final_write_age={"vpu.cmax": 257, "vpu.relu": 65}),
            xlu=dict(initiation_age=66), scalar_lsu=dict(load_response_gap=3),
            vlsu=dict(initiation_ages={"vlsu.vload": 35}))
        with patch.object(validate, "native_run", side_effect=AssertionError("sample planning executed native model")):
            for capacity in (1, 2, 3):
                admission["vpu"]["capacity"] = capacity
                plan = {sample["id"]: sample for sample in validate.sample_plan(admission, facts)}
                with self.subTest(capacity=capacity):
                    shared = plan["vpu-read-coalescing"]
                    self.assertEqual(shared["issues"], [0, 130])
                    self.assertEqual(shared["accepted"], capacity >= 2)
                    if capacity == 1:
                        release = plan["vpu-read-coalescing-final-slot-reuse"]
                        self.assertEqual(release["assembly"], shared["assembly"])
                        self.assertEqual(release["issues"], [0, 257])
                        self.assertTrue(release["accepted"])
                    else:
                        self.assertNotIn("vpu-read-coalescing-final-slot-reuse", plan)


@unittest.skipUnless(os.environ.get("ATLAS_DERIVE_MODEL_ROOT"), "set ATLAS_DERIVE_MODEL_ROOT for engine source/native contracts")
class NativeEngineAdmissionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = Path(os.environ["ATLAS_DERIVE_MODEL_ROOT"])
        isolate_model_imports(cls, str(cls.model))
        import torch
        torch.set_num_threads(1)
        from atlas_model_timing import configuration, source_digest, numerical_data, encoded, sha256
        from atlas_derive_vpu import derive_vpu
        from atlas_derive_mxu import derive_mxu
        from atlas_derive_memory import derive_memory
        cfg, config = configuration()
        vpu, vr = derive_vpu(cfg)
        mxu, mr = derive_mxu(cfg)
        memory, lr = derive_memory(cfg)
        cls.facts = dict(schema="atlas.model_timing.v1", age_convention="issue_edge_zero",
            model=dict(source_sha256=source_digest(cls.model)[0],
                configuration_sha256=sha256(encoded(config)), numerical_data_sha256=numerical_data(cls.model)[0],
                configuration=config), op_timing=list((vpu | mxu | memory).values()), reservation_releases=vr | mr | lr)
        cls.admission = derive.derive_admission(cls.model, cls.facts)

    def test_source_metadata_is_relational_and_preserves_conservative_boundaries(self):
        admission = self.admission
        self.assertEqual(admission["schema"], "atlas.engine_admission.v1")
        self.assertEqual(admission["scalar_lsu"]["load_response_gap"], 3)
        self.assertTrue(admission["xlu"]["independent_from_vpu"])
        self.assertEqual(admission["policy"]["unknown_vmem"], "aliases_all_banks")
        self.assertTrue(all(name.startswith("vpu.") for name in admission["vpu"]["double"]))
        self.assertTrue(all(name.startswith("vpu.") for group in admission["vpu"]["logic_groups"] for name in group))
        self.assertFalse(admission["validation"]["passed"])

    def test_construction_or_routing_refactor_refuses_before_publication(self):
        original = derive._source
        def changed(root, name):
            text, tree, digest = original(root, name)
            if name == "core.py":
                import ast
                text = text.replace("self.exu_map = {EXU(type(exu).__name__): exu for exu in self.exus}",
                                    "self.exu_map = {}")
                tree = ast.parse(text)
            return text, tree, digest
        with patch.object(derive, "_source", side_effect=changed), self.assertRaisesRegex(ValueError, "construction/routing"):
            derive.derive_admission(self.model, self.facts)

    def test_changed_frozen_guard_refuses_before_native_execution(self):
        changed = deepcopy(self.admission)
        changed["vpu"]["capacity"] = 3
        with patch.object(validate, "native_run") as native, self.assertRaisesRegex(ValueError, "frozen source derivation"):
            validate.validate_admission(self.model, self.facts, changed)
        native.assert_not_called()

    def test_native_boundaries_and_serial_state(self):
        qualified, report = validate.validate_admission(self.model, self.facts, self.admission)
        self.assertTrue(report["passed"])
        self.assertGreater(report["positive_runs"], 0)
        self.assertGreater(report["negative_runs"], 0)
        self.assertEqual(report["positive_runs"], report["numerical_checks"])
        self.assertEqual(derive.semantic_admission(qualified), derive.semantic_admission(self.admission))
