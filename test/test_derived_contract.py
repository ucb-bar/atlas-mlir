"""Integrity and determinism tests; fixtures here are synthetic test data only."""
import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from atlas_derived_contract import semantic_payload, seal_derived, verify_derived
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


if __name__ == "__main__":
    unittest.main()
