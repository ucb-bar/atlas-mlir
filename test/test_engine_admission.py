"""Source engine predicates, independent native boundaries and payload identity."""
from copy import deepcopy
import os
from pathlib import Path
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
