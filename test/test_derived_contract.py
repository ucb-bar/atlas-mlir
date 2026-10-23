"""Integrity and determinism tests; fixtures here are synthetic test data only."""
import copy
import json
import os
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from atlas_derived_contract import semantic_payload, seal_derived, verify_derived, overlap_semantics
from atlas_model_schema import canonical, sha256


def fixture():
    return dict(test_only=True, model=dict(revision="a" * 40, revision_kind="locked_source",
                source_sha256="b" * 64, numerical_data_sha256="c" * 64,
                configuration={"frequency": 0.75}, configuration_sha256="d" * 64),
                architecture={"mreg_count": 64}, operand_vocabulary={"operations": []}, isa_metadata={"instructions": []},
                geometry=dict(rows=32, columns=32, mreg_count=64), age_convention="issue_edge_zero",
                op_timing=[dict(name="test", first_free_age=2)], reservation_releases={"test": {"unit": {}}},
                vocabulary_sha256="e" * 64, dma_timing=dict(curves=[[0.125, 3.0]],
                validation=dict(passed=True, report_sha256="f" * 64)),
                validation=dict(passed=True, operation_count=1, observed_runs=2, unobserved_runs=2,
                                gate_report_sha256="f" * 64))


def contract_format_fixture():
    # Private format test only: no fixture generated here is exported for CLI use.
    facts = fixture()
    del facts["test_only"]
    return seal_derived(facts)


class DerivedContractTest(unittest.TestCase):
    def test_weight_rows_pe_bounds_and_order_are_bound_to_semantic_digest(self):
        from test_derived_mxu_weights import weight_fixture
        facts = contract_format_fixture()
        facts["mxu_weight_timing"] = weight_fixture()
        facts["overlap_rules_sha256"] = "a" * 64
        with self.assertRaisesRegex(ValueError, "successful MXU weight validation"):
            seal_derived(facts)
        facts["validation"]["mxu_weight_validation"] = dict(passed=True)
        seal_derived(facts)
        self.assertEqual(semantic_payload(facts)["mxu_weight_timing"], facts["mxu_weight_timing"])
        self.assertEqual(semantic_payload(facts)["overlap_rules_sha256"], "a" * 64)
        changed = copy.deepcopy(facts)
        changed["overlap_rules_sha256"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "semantic payload mismatch"):
            verify_derived(changed)
        for change in (lambda x: x["use_rows"][0].update(last_age=62),
                       lambda x: x["use_window"].update(column_step=0),
                       lambda x: x.update(reads_before_writes=False),
                       lambda x: x.update(resource_row_axis="inner")):
            changed = copy.deepcopy(facts)
            change(changed["mxu_weight_timing"]["engines"]["mxu0"])
            with self.assertRaisesRegex(ValueError, "semantic payload mismatch"):
                verify_derived(changed)
        facts["validation"]["mxu_weight_validation"]["passed"] = False
        with self.assertRaisesRegex(ValueError, "successful MXU weight validation"):
            verify_derived(facts)

    def test_test_only_and_raw_observations_cannot_enter_production(self):
        facts = fixture()
        with self.assertRaisesRegex(ValueError, "test-only"):
            seal_derived(facts)
        with self.assertRaisesRegex(ValueError, "test-only"):
            verify_derived(facts)
        del facts["test_only"]
        with self.assertRaisesRegex(ValueError, "raw observations are validation/test-only"):
            verify_derived(facts)

    def test_contract_covers_dma_and_release_values_not_only_validation_hashes(self):
        facts = contract_format_fixture()
        digest = verify_derived(facts)
        for change in (lambda d: d["dma_timing"]["curves"][0].__setitem__(1, 5.0),
                       lambda d: d["op_timing"][0].__setitem__("first_free_age", 3),
                       lambda d: d["reservation_releases"]["test"].__setitem__("unit", {"release_age": 2}),
                       lambda d: d.__setitem__("vocabulary_sha256", "0" * 64),
                       *(lambda d, key=key: d[key].update(changed=True)
                         for key in ("architecture", "operand_vocabulary", "isa_metadata"))):
            altered = copy.deepcopy(facts)
            change(altered)
            self.assertEqual(altered["validation"]["payload_sha256"], digest)
            with self.assertRaisesRegex(ValueError, "semantic payload mismatch"):
                verify_derived(altered)

    def test_receipt_changes_do_not_change_semantic_digest(self):
        facts = contract_format_fixture()
        original = verify_derived(facts)
        facts["model"]["exporter_sha256"] = "1" * 64
        facts["validation"]["gate_report_sha256"] = "2" * 64
        self.assertEqual(verify_derived(facts), original)
        # Elapsed times live in the separate receipt, never in the fact payload.
        for elapsed in (0.1, 999):
            receipt = dict(semantic_payload_sha256=original, derive_seconds=elapsed)
            self.assertEqual(receipt["semantic_payload_sha256"], sha256(canonical(semantic_payload(facts))))
        self.assertEqual(seal_derived(copy.deepcopy(facts)), facts)

    def test_validation_records_are_bound_to_actual_semantics(self):
        for target in ("validation", "dma_timing"):
            facts = contract_format_fixture()
            record = facts["validation"] if target == "validation" else facts[target]["validation"]
            record["payload_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "validation is not bound"):
                verify_derived(facts)
        for target in ("derivation", "validation"):
            facts = contract_format_fixture()
            facts["derivation_contract"][target]["payload_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "record"):
                verify_derived(facts)

    def test_failed_validation_and_incomplete_releases_cannot_be_sealed(self):
        for change in (lambda d: d["validation"].__setitem__("passed", False),
                       lambda d: d["reservation_releases"].clear(),
                       lambda d: d["dma_timing"]["validation"].__setitem__("passed", False)):
            facts = contract_format_fixture()
            change(facts)
            with self.assertRaisesRegex(ValueError, "require"):
                seal_derived(facts)

    def test_explicit_payload_bytes_round_trip_float_values(self):
        facts = contract_format_fixture()
        contract = facts["derivation_contract"]
        self.assertEqual(json.loads(contract["payload_json"]), semantic_payload(facts))
        self.assertEqual(sha256(contract["payload_json"].encode()), verify_derived(facts))


class OverlapSemanticBindingTest(unittest.TestCase):
    def fixture(self):
        return dict(schema="atlas.model_overlap_derived.v1", age_convention="issue_edge_zero",
                    derivation=dict(contract="source-resource-occupancy-v1", source_contract={
                        "capacity": {"vpu_slots": 2}, "producer_sources": [{"sha256": "a" * 64}]}),
                    pairs=[{"first": "vpu.mov", "second": "vpu.mov", "minimum": 1}],
                    composition={"engines": {"vpu": ["vpu.mov"]}},
                    validation={"passed": True, "seconds": 1})

    def digest(self, artifact):
        return sha256(canonical(overlap_semantics(artifact)))

    def test_all_decision_inputs_change_the_digest(self):
        original = self.fixture()
        for change in (lambda d: d.__setitem__("schema", "other"),
                       lambda d: d.__setitem__("age_convention", "other"),
                       lambda d: d["derivation"].__setitem__("contract", "other"),
                       lambda d: d["derivation"]["source_contract"]["capacity"].__setitem__("vpu_slots", 3),
                       lambda d: d["pairs"][0].__setitem__("minimum", 2),
                       lambda d: d["composition"]["engines"]["vpu"].append("vpu.add")):
            changed = copy.deepcopy(original)
            change(changed)
            self.assertNotEqual(self.digest(changed), self.digest(original))

    def test_receipts_and_producer_sources_do_not_change_rules(self):
        original = self.fixture()
        changed = copy.deepcopy(original)
        changed["validation"].update(seconds=999, report_sha256="b" * 64)
        changed["derivation"]["source_contract"]["producer_sources"] = []
        changed["derivation"]["tool_sha256"] = "c" * 64
        self.assertEqual(self.digest(original), self.digest(changed))
        self.assertIn("producer_sources", original["derivation"]["source_contract"])

    def test_overlap_digest_is_bound_to_derived_validation(self):
        facts = contract_format_fixture()
        facts["overlap_rules_sha256"] = self.digest(self.fixture())
        seal_derived(facts)
        original = verify_derived(facts)
        facts["overlap_rules_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "semantic payload mismatch"):
            verify_derived(facts)
        seal_derived(facts)
        self.assertNotEqual(verify_derived(facts), original)


@unittest.skipUnless("ATLAS_REGENERATED_MODEL_FACTS" in os.environ,
                     "set ATLAS_TEST_MODEL_FACTS and ATLAS_REGENERATED_MODEL_FACTS to independent bundles")
class IndependentBundleDeterminismTest(unittest.TestCase):
    def test_independent_regenerations_have_identical_timing_and_overlap(self):
        first = Path(os.environ["ATLAS_TEST_MODEL_FACTS"])
        second = Path(os.environ["ATLAS_REGENERATED_MODEL_FACTS"])
        self.assertNotEqual(first.resolve(), second.resolve())
        for name in ("model-timing.json", "model-overlap.json", "facts.lock.json"):
            self.assertEqual((first / name).read_bytes(), (second / name).read_bytes(), name)
        self.assertEqual(verify_derived(json.loads((first / "model-timing.json").read_text())),
                         verify_derived(json.loads((second / "model-timing.json").read_text())))


if __name__ == "__main__":
    unittest.main()
