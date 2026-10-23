"""Replay the supplied pinned catalog against freshly regenerated facts bytes."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
VARIABLES = ("ATLAS_REPROJECT_FACTS", "ATLAS_REPROJECT_CATALOG",
             "ATLAS_REPROJECT_QUERIES", "ATLAS_REPROJECT_HOLDOUTS")


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@unittest.skipUnless(any(name in os.environ for name in VARIABLES),
                     "set ATLAS_REPROJECT_FACTS, ATLAS_REPROJECT_CATALOG, "
                     "ATLAS_REPROJECT_QUERIES and ATLAS_REPROJECT_HOLDOUTS for pinned catalog replay")
class PinnedCatalogReprojectionTest(unittest.TestCase):
    def test_fresh_digest_rebinds_base_and_composed_catalog_without_changing_admissions(self):
        paths = {}
        for name in VARIABLES:
            value = os.environ.get(name)
            self.assertTrue(value, f"{name} is required when configuring catalog replay")
            paths[name] = Path(value)
            self.assertTrue(paths[name].is_file(), f"{name} must name an existing file")
        fresh_path = paths["ATLAS_REPROJECT_FACTS"]
        catalog = load(paths["ATLAS_REPROJECT_CATALOG"])["derived"]
        for kind in ("op", "pairs", "composed"):
            self.assertEqual(sha(catalog[kind]), catalog[kind + "_sha256"])
        before, fresh = load(catalog["op"]), load(fresh_path)
        fresh_sha = sha(fresh_path)
        self.assertNotEqual(fresh_sha, catalog["op_sha256"], "supply freshly regenerated facts")
        for key in ("schema", "age_convention", "geometry", "observation", "op_timing",
                    "reservation_releases", "dma_timing", "vocabulary_sha256"):
            self.assertEqual(before[key], fresh[key], key)
        for key in ("revision", "source_sha256", "numerical_data_sha256",
                    "configuration_sha256", "configuration"):
            self.assertEqual(before["model"][key], fresh["model"][key], key)
        lock = load(ROOT / "tools/model-timing.lock.json")
        self.assertEqual(fresh["model"]["revision"], lock["model_revision"])
        self.assertEqual(fresh["model"]["source_sha256"], lock["source_sha256"])
        self.assertEqual(fresh["model"]["configuration_sha256"], lock["configuration_sha256"])
        self.assertTrue(fresh["validation"]["passed"])
        self.assertTrue(fresh["dma_timing"]["validation"]["passed"])
        original_base = load(catalog["pairs"])
        source = original_base["projection"]
        master, validation = Path(source["master_path"]), Path(source["validation_path"])
        self.assertEqual(sha(master), source["master_sha256"])
        self.assertEqual(sha(validation), source["validation_sha256"])
        heldout_receipt = load(Path(catalog["composed"]).parent / "receipt.json")
        queries, holdouts = paths["ATLAS_REPROJECT_QUERIES"], paths["ATLAS_REPROJECT_HOLDOUTS"]
        self.assertEqual(sha(queries), heldout_receipt["queries_sha256"])
        self.assertEqual(sha(holdouts), heldout_receipt["provenance"]["holdouts_sha256"])

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(os.environ.get("ATLAS_REPROJECT_OUTPUT", temporary + "/reprojection"))
            self.assertFalse(output.exists(), "reprojection output must be a new directory")
            output.mkdir(parents=True)
            commands = []

            def run(tool, *args, succeeds=True):
                command = [sys.executable, "-B", str(ROOT / "tools" / tool), *map(str, args)]
                result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                        text=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
                (output / (str(len(commands)) + "-" + tool + ".log")).write_text(result.stdout)
                commands.append(dict(command=command, returncode=result.returncode))
                if succeeds:
                    self.assertEqual(result.returncode, 0, result.stdout[-5000:])
                else:
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("artifact SHA-256 mismatch", result.stdout)

            project_args = ("--master", master, "--validation", validation, "--per-op", fresh_path,
                            "--master-sha256", source["master_sha256"],
                            "--validation-sha256", source["validation_sha256"])
            run("atlas_model_pairs.py", *project_args, "--output", output / "refused", succeeds=False)
            self.assertFalse((output / "refused").exists())
            run("atlas_model_pairs.py", *project_args, "--per-op-sha256", fresh_sha,
                "--output", output / "base")
            run("atlas_model_composition.py", "--master", master, "--per-op", fresh_path,
                "--output-dir", output / "pair-replay")
            run("atlas_model_chain_composition.py", "--master", master, "--per-op", fresh_path,
                "--output", output / "chain-replay")
            run("atlas_model_composed_projection.py", "--per-op", fresh_path,
                "--per-op-sha256", fresh_sha, "--base-pairs", output / "base/model-pairs.json",
                "--pair-audit", output / "pair-replay/pair-summary.json",
                "--chain-audit", output / "chain-replay/summary.json",
                "--queries", queries, "--holdouts", holdouts, "--output", output / "composed")
            base, composed = load(output / "base/model-pairs.json"), load(output / "composed/model-composed.json")
            old_composed = load(catalog["composed"])
            for kind in ("pairs", "chains"):
                self.assertEqual(base[kind], original_base[kind], "base " + kind)
                self.assertEqual(composed[kind], old_composed[kind], "composed " + kind)
            self.assertEqual(base["projection"]["per_op_sha256"], fresh_sha)
            self.assertEqual(load(output / "pair-replay/pair-summary.json")["per_operation_sha256"], fresh_sha)
            self.assertEqual(load(output / "chain-replay/summary.json")["per_op_sha256"], fresh_sha)
            receipt = load(output / "composed/receipt.json")
            self.assertTrue(receipt["passed"])
            self.assertEqual(receipt["provenance"]["per_op_sha256"], fresh_sha)
            self.assertEqual(receipt["provenance"]["master_sha256"], source["master_sha256"])
            for kind in ("op", "pairs", "composed"):
                self.assertEqual(sha(catalog[kind]), catalog[kind + "_sha256"], "input changed")
            (output / "regression-summary.json").write_text(json.dumps(dict(
                passed=True, original_facts_sha256=catalog["op_sha256"], fresh_facts_sha256=fresh_sha,
                model_revision=fresh["model"]["revision"], admissions_unchanged=True,
                base_counts={kind: len(base[kind]) for kind in ("pairs", "chains")},
                composed_counts={kind: len(composed[kind]) for kind in ("pairs", "chains")},
                commands=commands), indent=2) + "\n")


if __name__ == "__main__":
    unittest.main()
