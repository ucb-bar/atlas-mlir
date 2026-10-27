"""Selection provenance identities and fail-closed completion snapshots."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from atlas_run_provenance import capture_run_provenance, safe_path, selection_facts_identity
from atlas_derived_contract import seal_derived


def sha(data):
    return hashlib.sha256(data).hexdigest()


class RunProvenanceTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.source = self.write("source.mlir", b"module {}\n")
        self.region = self.write("region.json", b'{ "address": 256, "initial_hex": "010203", "expected_hex": "040506" }\n')
        self.binary = self.write("atlas-opt", b"compiler bytes")
        self.model_source = self.write("model/core.py", b"class Core: pass\n")
        self.numerical = self.write("model/data/table.bin", b"\x00\xff")
        self.adapter = self.write("adapter.py", b"periodic callback implementation\n")

    def write(self, relative, data):
        path = self.directory / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    def capture(self, **kwargs):
        values = dict(regions=(self.region,), files={"atlas-opt": self.binary, "profile-adapter": self.adapter},
                      trees={"model-source": (self.model_source.parent, (".py", ".json", ".yaml")),
                             "numerical": self.numerical.parent},
                      configuration={"numerics": "rtl"}, search_policy="policy-sha256:abc")
        values.update(kwargs)
        return capture_run_provenance(self.source, **values)

    def test_regions_archive_exact_json_initial_and_expected_bytes(self):
        archive = self.directory / "receipt"
        receipt = self.capture(archive_dir=archive).complete()
        semantic = receipt["semantic"]["regions"][0]
        self.assertEqual(semantic["address"], 256)
        self.assertEqual(semantic["initial"]["sha256"], sha(b"\x01\x02\x03"))
        self.assertEqual(semantic["expected"]["sha256"], sha(b"\x04\x05\x06"))
        row = receipt["evidence"]["regions"][0]
        for key, expected in (("json", self.region.read_bytes()), ("initial", b"\x01\x02\x03"),
                              ("expected", b"\x04\x05\x06")):
            self.assertEqual(Path(row[key]["path"]).read_bytes(), expected)
            self.assertEqual(row[key]["sha256"], sha(expected))
        self.assertEqual(receipt["semantic"]["files"]["atlas-opt"]["sha256"], sha(self.binary.read_bytes()))

    def test_inline_region_archive_is_lossless(self):
        receipt = self.capture().complete()
        record = receipt["evidence"]["regions"][0]
        self.assertEqual(bytes.fromhex(record["json"]["hex"]), self.region.read_bytes())
        self.assertEqual(bytes.fromhex(record["initial"]["hex"]), b"\x01\x02\x03")

    def test_identity_excludes_locations_git_and_raw_reports(self):
        first_report = self.write("facts-first.lock.json", b'{"seconds":1}')
        first = self.capture(evidence_files={"facts-lock": first_report}, revisions={"compiler": "a" * 40}).complete()
        source = self.write("relocated/renamed.mlir", self.source.read_bytes())
        region = self.write("relocated/data.json", self.region.read_bytes())
        binary = self.write("relocated/compiler", self.binary.read_bytes())
        adapter = self.write("relocated/profile.py", self.adapter.read_bytes())
        self.write("relocated/model/core.py", self.model_source.read_bytes())
        self.write("relocated/model/data/table.bin", self.numerical.read_bytes())
        second_report = self.write("relocated/facts.lock.json", b'{"seconds":999}')
        second = capture_run_provenance(
            source, regions=(region,), files={"atlas-opt": binary, "profile-adapter": adapter},
            trees={"model-source": (source.parent / "model", (".py", ".json", ".yaml")),
                   "numerical": source.parent / "model/data"}, configuration={"numerics": "rtl"},
            search_policy="policy-sha256:abc", evidence_files={"facts-lock": second_report},
            revisions={"compiler": "b" * 40}).complete()
        self.assertEqual(first["semantic_sha256"], second["semantic_sha256"])
        self.assertNotEqual(first["evidence"], second["evidence"])

    def test_generated_facts_bound_before_use(self):
        capture = self.capture()
        facts = self.write("facts/model-timing.json", b'{"timing":7}')
        report = self.write("facts/validation.json", b'{"seconds":3}')
        capture.bind(files={"model-timing": facts}, evidence_files={"validation": report},
                     semantic={"facts_semantics": {"timing": 7}})
        receipt = capture.complete()
        self.assertEqual(receipt["semantic"]["facts_semantics"], {"timing": 7})
        self.assertIn("validation", receipt["evidence"]["files"])
        report.write_bytes(b'{"seconds":4}')
        with self.assertRaisesRegex(ValueError, "changed during selection"):
            capture.complete()

    def test_each_consumed_byte_source_is_checked_at_completion(self):
        for path in (self.source, self.region, self.binary, self.model_source, self.numerical, self.adapter):
            with self.subTest(path=path.name):
                original = path.read_bytes()
                capture = self.capture()
                path.write_bytes(original + b"mutation")
                with self.assertRaisesRegex(ValueError, "changed during selection"):
                    capture.complete()
                path.write_bytes(original)

    def test_model_roster_addition_and_removal_are_mutations(self):
        capture = self.capture()
        added = self.write("model/new.py", b"new source")
        with self.assertRaisesRegex(ValueError, "tree changed"):
            capture.complete()
        added.unlink()
        capture = self.capture()
        self.model_source.unlink()
        with self.assertRaisesRegex(ValueError, "tree changed"):
            capture.complete()

    def test_cache_bytes_are_not_model_source(self):
        capture = self.capture()
        self.write("model/__pycache__/core.cpython-312.pyc", b"import-generated cache")
        self.write("model/data/__pycache__/table.pyc", b"numerical-tree cache")
        capture.complete()

    def test_semantic_bytes_policy_profile_and_runtime_change_identity(self):
        profiles = {"fifo43": {"configuration": {"accept_fn": {"adapter": "periodic-v1", "period": 11}}}}
        versions = {"python": "3.12.1", "torch": "2.6.0", "numpy": "2.1.0"}
        baseline = self.capture().complete(scoring_configurations=profiles, runtime_versions=versions)["semantic_sha256"]
        changed = self.capture(search_policy={"caller-owned": [1, 2]}).complete(
            scoring_configurations=profiles, runtime_versions=versions)
        self.assertNotEqual(changed["semantic_sha256"], baseline)
        self.assertEqual(changed["semantic"]["search_policy"], {"caller-owned": [1, 2]})
        for key in versions:
            altered = dict(versions, **{key: "different"})
            self.assertNotEqual(self.capture().complete(scoring_configurations=profiles, runtime_versions=altered)["semantic_sha256"], baseline)
        profiles["fifo43"]["configuration"]["accept_fn"]["adapter"] = "periodic-v2"
        self.assertNotEqual(self.capture().complete(scoring_configurations=profiles, runtime_versions=versions)["semantic_sha256"], baseline)
        before = self.capture().complete()["semantic_sha256"]
        self.numerical.write_bytes(b"different numerical bytes")
        self.assertNotEqual(self.capture().complete()["semantic_sha256"], before)

    def test_missing_policy_refuses_without_legacy_fallback(self):
        with self.assertRaisesRegex(ValueError, "versioned effective search policy"):
            self.capture(search_policy=None)

    def test_archived_region_mutation_is_rejected(self):
        capture = self.capture(archive_dir=self.directory / "receipt")
        path = Path(capture.complete()["evidence"]["regions"][0]["initial"]["path"])
        path.write_bytes(b"corrupt archive")
        with self.assertRaisesRegex(ValueError, "archive:"):
            capture.complete()

    def test_parent_paths_rejected_before_filesystem_access(self):
        for path in ("safe/../source.mlir",):
            with self.subTest(path=path), patch.object(Path, "is_symlink", side_effect=AssertionError("filesystem touched")):
                with self.assertRaises(ValueError):
                    safe_path(path)

    def test_linked_input_and_parent_rejected_without_following(self):
        linked = self.directory / "linked.mlir"
        linked.symlink_to(self.source)
        with self.assertRaisesRegex(ValueError, "symlinks"):
            capture_run_provenance(linked, search_policy="policy")
        directory = self.directory / "linked-directory"
        directory.symlink_to(self.model_source.parent, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlinks"):
            self.capture(trees={"model": directory})

    def test_input_replaced_by_symlink_at_completion_is_rejected(self):
        capture = self.capture()
        alternate = self.write("alternate.mlir", self.source.read_bytes())
        self.source.unlink()
        self.source.symlink_to(alternate)
        with self.assertRaisesRegex(ValueError, "unsafe at completion"):
            capture.complete()

    def test_model_tree_symlink_does_not_follow_target(self):
        (self.model_source.parent / "linked.py").symlink_to(self.source)
        with self.assertRaisesRegex(ValueError, "trees must not contain symlinks"):
            self.capture()

    def facts(self, curve_path, curve_bytes=b"selected curve bytes"):
        configuration = {"dma_memory_params": {"curves": curve_path}, "frequency": 0.75}
        facts = dict(model=dict(revision="a" * 40, revision_kind="locked_source",
                     source_sha256="b" * 64, numerical_data_sha256="c" * 64,
                     configuration=configuration, configuration_sha256=sha(json.dumps(
                         configuration, sort_keys=True, separators=(",", ":")).encode())),
                     architecture={}, operand_vocabulary={}, isa_metadata={},
                     geometry={"rows": 32}, age_convention="issue_edge_zero",
                     op_timing=[{"name": "test", "first_free_age": 2}],
                     reservation_releases={"test": {}}, vocabulary_sha256="e" * 64,
                     dma_timing={"curve_sha256": sha(curve_bytes), "validation": {"passed": True}},
                     validation={"passed": True, "operation_count": 1, "observed_runs": 1,
                                 "unobserved_runs": 1})
        return seal_derived(facts)

    def test_complete_identity_remains_stable_across_relocated_fact_reports(self):
        receipts = []
        for index, curve_path in enumerate(("/first/curve.json", "/second/renamed.json")):
            facts = self.facts(curve_path)
            semantic_digest, verified_digest = selection_facts_identity(facts)
            report = self.write(f"report-{index}.json", json.dumps(facts).encode())
            capture = self.capture()
            capture.bind(evidence_files={"facts": report},
                         semantic={"derived_facts_sha256": semantic_digest})
            receipts.append(capture.complete(scoring_configurations={"curve": {
                "sha256": verified_digest,
                "model_identity": {"configuration_sha256": facts["model"]["configuration_sha256"]},
                "configuration": {"dma_memory_params": {"curves": {"sha256": sha(b"selected curve bytes")}}}}}))
        self.assertEqual(receipts[0]["semantic_sha256"], receipts[1]["semantic_sha256"])
        self.assertNotEqual(receipts[0]["evidence"], receipts[1]["evidence"])

    def test_facts_normalization_preserves_curve_and_other_content_identity(self):
        baseline = selection_facts_identity(self.facts("/first/curve.json"))[0]
        changed = self.facts("/second/curve.json")
        changed["model"]["configuration"]["description"] = "/second/curve.json"
        seal_derived(changed)
        self.assertNotEqual(selection_facts_identity(changed)[0], baseline)
        changed = self.facts("/second/curve.json", b"different actual curve bytes")
        self.assertNotEqual(selection_facts_identity(changed)[0], baseline)
        changed = self.facts("/second/curve.json")
        changed["model"]["numerical_data_sha256"] = "f" * 64
        seal_derived(changed)
        self.assertNotEqual(selection_facts_identity(changed)[0], baseline)
        changed["dma_timing"]["curve_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "semantic payload mismatch"):
            selection_facts_identity(changed)


if __name__ == "__main__":
    unittest.main()
