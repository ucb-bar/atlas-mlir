"""Selecting pair facts (merlin.op_pairs.v1) with the RTL timing facts (ATLAS_OP_TIMING, ATLAS_PAIR_FACTS, ATLAS_OOT_BIN_DIR).

`--select-atlas-rtl-evidence` takes `op-pairs=<file> op-pairs-sha256=<sha>` beside the facts; the pair facts must hash
to the digest, describe the same hardware IR, derive from the selected facts and carry only assertion-checked records
naming resolved blocks. The selection is recorded in atlas.rtl_evidence and rechecked wherever the facts are loaded.
"""
import json
from pathlib import Path
import tempfile
import unittest

from test_delay_insertion import EMIT, OPT, program, run
from test_rtl_timing import FACTS, HALT, PAIRS, VERIFY, selection, sha256
from test_rtl_vmul_timing import unary


def pairs_variant(directory, mutate):
    document = json.loads(Path(PAIRS).read_text())
    mutate(document)
    path = Path(directory) / "op-pairs.json"
    path.write_text(json.dumps(document))
    return path


def first_record(change):
    return lambda document: document["op_pairs"][0].update(change)


@unittest.skipUnless(FACTS and PAIRS and OPT.is_file() and EMIT.is_file(),
                     "requires RTL timing facts, pair facts derived from them and built tools")
class PairFactsSelectionTest(unittest.TestCase):
    SOURCE = program([unary("exp", 8, 16), HALT])

    def test_selection_is_recorded(self):
        result = run(OPT, self.SOURCE, selection(pairs=PAIRS))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(f'op_pairs = "{Path(PAIRS).resolve()}"', result.stdout)
        self.assertIn(f'op_pairs_sha256 = "{sha256(PAIRS)}"', result.stdout)
        plain = run(OPT, self.SOURCE, selection())
        self.assertNotIn("op_pairs", plain.stdout)

    def test_selection_reaches_the_resolved_export(self):
        timed = run(OPT, self.SOURCE, selection(pairs=PAIRS), "--insert-atlas-delays", VERIFY)
        self.assertEqual(timed.returncode, 0, timed.stderr)
        exported = run(EMIT, timed.stdout, "--rtl-timing-json")
        self.assertEqual(exported.returncode, 0, exported.stderr)
        document = json.loads(exported.stdout)
        self.assertEqual(document["evidence"]["op_pairs_sha256"], sha256(PAIRS))
        self.assertEqual(document["applicability"]["pair_facts"]["sha256"], sha256(PAIRS))

    def test_invalid_selections_are_rejected(self):
        digest = f"op-timing={Path(FACTS).resolve()} op-timing-sha256={sha256(FACTS)}"
        cases = {
            "hash mismatch": f"--select-atlas-rtl-evidence={digest} op-pairs={PAIRS} op-pairs-sha256={'0' * 64}",
            "selected together": f"--select-atlas-rtl-evidence={digest} op-pairs={PAIRS}",
        }
        with tempfile.TemporaryDirectory() as directory:
            for message, mutate in {
                "different hardware IR": lambda d: d["hw_ir"].update(sha256="0" * 64),
                "different op_timing facts": lambda d: d["op_timing"].update(sha256="0" * 64),
                "not checked against RTL assertions": first_record({"assertions": {"checked": 0, "excluded": 0}}),
                "names no resolved facts block": first_record({"second": "vpu.unknown"}),
                "malformed stable_gap": first_record({"stable_gap": 0}),
                "expected schema merlin.op_pairs.v1": lambda d: d.update(schema="rtl_extract.op_pairs.v1"),
                "malformed capacity entry": lambda d: d["capacity"].append(dict(d["capacity"][0])),
            }.items():
                variant = Path(directory) / str(len(cases))
                variant.mkdir()
                cases[message] = selection(pairs=pairs_variant(variant, mutate))
            for message, option in cases.items():
                with self.subTest(message):
                    result = run(OPT, self.SOURCE, option)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(message, result.stderr)

    def test_recorded_selection_is_rechecked(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "op-pairs.json"
            path.write_bytes(Path(PAIRS).read_bytes())
            selected = run(OPT, self.SOURCE, selection(pairs=path))
            self.assertEqual(selected.returncode, 0, selected.stderr)
            path.write_text(path.read_text() + " ")
            result = run(OPT, selected.stdout, "--insert-atlas-delays")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("selected pair facts hash mismatch", result.stderr)
            incomplete = selected.stdout.replace("op_pairs_sha256", "op_pairs_digest")
            result = run(OPT, incomplete, "--insert-atlas-delays")
            self.assertIn("incomplete RTL pair facts selection", result.stderr)


if __name__ == "__main__":
    unittest.main()
