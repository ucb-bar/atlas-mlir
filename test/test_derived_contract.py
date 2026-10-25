"""Semantic binding and refusals use private synthetic format fixtures."""
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from atlas_derived_contract import seal_derived, verify_derived, overlap_semantics, semantic_payload
from atlas_model_schema import canonical, sha256


def change_at(value, path, replacement):
    changed = deepcopy(value)
    target = changed
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = replacement
    return changed


def contract_format_fixture():
    # Private format fixture only; never exported as a qualified model bundle.
    from test_derived_mxu_weights import weight_fixture
    facts = dict(model=dict(revision="a" * 40, revision_kind="locked_source", source_sha256="b" * 64,
                numerical_data_sha256="c" * 64, configuration={"frequency": 0.75}, configuration_sha256="d" * 64),
                architecture={"mreg_count": 64}, operand_vocabulary={"operations": []}, isa_metadata={"instructions": []},
                geometry=dict(rows=32, columns=32, mreg_count=64), age_convention="issue_edge_zero",
                op_timing=[dict(name="vpu.add", first_free_age=2)],
                reservation_releases={"vpu.add": {"unit": {"release_age": 2}}}, vocabulary_sha256="e" * 64,
                frontend_timing={}, scratchpad_contract={}, replay_timing={}, proposal_costs={},
                overlap_rules_sha256="f" * 64, mxu_weight_timing=weight_fixture(),
                validation=dict(passed=True, operation_count=1, observed_runs=2, unobserved_runs=2,
                                mxu_weight_validation=dict(passed=True), gate_report_sha256="f" * 64))
    for field in ("dma_timing", "engine_admission", "dma_admission"):
        facts[field] = dict(validation=dict(passed=True, report_sha256="f" * 64))
    facts["engine_admission"]["vpu"] = dict(capacity=2)
    facts["dma_admission"]["issue"] = dict(ordinary_gap=1)
    facts["dma_timing"].update(curves=[dict(cycles_per_beat=0.125)], proposal_contract=dict(
        retire_to_wait_issue_cycles=1, per_beat_bound_cycles=47, preconditions={"vmem_grant_denials": 0},
        policy={"intercommand_serialization_cycles": 1}, source_evidence={"ready_base": "request.cycle + latency"}))
    return seal_derived(facts)


def overlap_fixture():
    return dict(schema="atlas.model_overlap_derived.v1", age_convention="issue_edge_zero",
                derivation=dict(contract="source-resource-occupancy-v1", source_contract={
                    "capacity": {"vpu_slots": 2}, "producer_sources": [{"sha256": "a" * 64}]}),
                pairs=[{"first": "vpu.mov", "second": "vpu.mov", "minimum": 1}],
                composition={"engines": {"vpu": ["vpu.mov"]}}, validation={"passed": True, "seconds": 1})


class DerivedContractTest(unittest.TestCase):
    def setUp(self):
        self.facts = contract_format_fixture()
        verify_derived(self.facts)

    def test_decision_inputs_are_bound_to_the_semantic_digest(self):
        paths = [(field,) for field in ("geometry", "age_convention", "vocabulary_sha256", "architecture",
                 "operand_vocabulary", "isa_metadata", "frontend_timing", "scratchpad_contract",
                 "overlap_rules_sha256", "replay_timing", "proposal_costs")]
        paths += [("model", field) for field in ("revision", "revision_kind", "source_sha256",
                  "numerical_data_sha256", "configuration", "configuration_sha256")]
        paths += [("op_timing", 0, "first_free_age"),
                  ("reservation_releases", "vpu.add", "unit", "release_age"),
                  ("dma_timing", "curves", 0, "cycles_per_beat"),
                  ("dma_timing", "proposal_contract", "policy"),
                  ("mxu_weight_timing", "engines", "mxu0", "use_rows", 0, "last_age"),
                  ("engine_admission", "vpu", "capacity"),
                  ("dma_admission", "issue", "ordinary_gap")]
        for path in paths:
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "semantic payload mismatch"):
                verify_derived(change_at(self.facts, path, "tampered"))

    def test_test_only_and_raw_observations_cannot_enter_production(self):
        self.facts["test_only"] = True
        for operation in (seal_derived, verify_derived):
            with self.assertRaisesRegex(ValueError, "test-only"):
                operation(self.facts)
        del self.facts["test_only"]
        del self.facts["derivation_contract"]
        with self.assertRaisesRegex(ValueError, "raw observations are validation/test-only"):
            verify_derived(self.facts)

    def test_validation_records_are_bound_to_actual_semantics(self):
        paths = [("validation",), ("derivation_contract", "derivation"),
                 ("derivation_contract", "validation")]
        paths += [(field, "validation") for field in ("dma_timing", "engine_admission", "dma_admission")]
        for path in paths:
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "validation is not bound|record"):
                verify_derived(change_at(self.facts, (*path, "payload_sha256"), "0" * 64))

    def test_failed_validation_and_incomplete_releases_cannot_be_sealed(self):
        paths = [("validation", "passed"), ("validation", "mxu_weight_validation", "passed")]
        paths += [(field, "validation", "passed") for field in ("dma_timing", "engine_admission", "dma_admission")]
        for path in paths:
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "require successful"):
                seal_derived(change_at(self.facts, path, False))
            with self.assertRaisesRegex(ValueError, "validation"):
                verify_derived(change_at(self.facts, path, False))
        self.facts["reservation_releases"].clear()
        with self.assertRaisesRegex(ValueError, "complete reservation release"):
            seal_derived(self.facts)

    def test_receipts_and_producer_sources_do_not_change_semantics(self):
        digest = verify_derived(self.facts)
        self.facts["model"]["exporter_sha256"] = "1" * 64
        self.facts["validation"]["gate_report_sha256"] = "2" * 64
        for field in ("dma_timing", "engine_admission", "dma_admission"):
            self.facts[field]["validation"]["report_sha256"] = "0" * 64
            self.facts[field]["producer_sources"] = []
        self.assertEqual(verify_derived(self.facts), digest)

    def test_sealing_is_idempotent_and_payload_bytes_preserve_floats(self):
        facts = self.facts
        self.assertEqual(seal_derived(deepcopy(facts)), facts)
        payload = facts["derivation_contract"]["payload_json"]
        self.assertEqual(json.loads(payload), semantic_payload(facts))
        self.assertEqual(json.loads(payload)["model"]["configuration"]["frequency"], 0.75)
        self.assertEqual(sha256(payload.encode()), verify_derived(facts))

    def test_every_dma_proposal_semantic_field_is_bound(self):
        paths = (("retire_to_wait_issue_cycles",), ("per_beat_bound_cycles",),
                 ("preconditions", "vmem_grant_denials"), ("policy", "intercommand_serialization_cycles"),
                 ("source_evidence", "ready_base"))
        original = verify_derived(self.facts)
        for path in paths:
            with self.subTest(path=path):
                changed = change_at(self.facts, ("dma_timing", "proposal_contract", *path), "changed")
                with self.assertRaisesRegex(ValueError, "semantic payload mismatch"):
                    verify_derived(changed)
                self.assertNotEqual(verify_derived(seal_derived(changed)), original)

    def test_overlap_decision_inputs_are_bound_but_receipts_are_not(self):
        original = overlap_fixture()
        digest = lambda value: sha256(canonical(overlap_semantics(value)))
        paths = (("schema",), ("age_convention",), ("derivation", "contract"),
                 ("derivation", "source_contract", "capacity"), ("pairs", 0, "minimum"), ("composition",))
        for path in paths:
            with self.subTest(path=path):
                self.assertNotEqual(digest(change_at(original, path, "changed")), digest(original))
        changed = deepcopy(original)
        changed["validation"].update(seconds=999, report_sha256="b" * 64)
        changed["derivation"]["source_contract"]["producer_sources"] = []
        changed["derivation"]["tool_sha256"] = "c" * 64
        self.assertEqual(digest(original), digest(changed))


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
