"""Evidence-checker rejection tests; hardware qualification uses the real replay."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("vls_evidence", ROOT / "tools/check-vls-timing.py")
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)


def comparison_inputs():
    """A controlled valid record for testing mutations of the evidence format."""
    instances, traces = [], {}
    for op, short in (("vload", "L"), ("vstore", "S")):
        accesses = []
        for resource, write, age in (("Vmem", short == "S", 1 if short == "L" else 3),
                                     ("MReg", short == "L", 3 if short == "L" else 1)):
            accesses.append(dict(resource=resource, write=write, first=0, count=32,
                                 age=age, step=1, last_age=age + 31,
                                 anywhere=False, at_completion=False))
        bank_start, bank_end = (1, 32) if short == "L" else (3, 34)
        holds = [dict(unit="VLOAD path" if short == "L" else "VSTORE path",
                      unit_id=2 if short == "L" else 3, index=0, alt=-1,
                      **{"from": 0, "to": 34}),
                 dict(unit="VMEM bank", unit_id=4, index=0, alt=-1,
                      **{"from": bank_start, "to": bank_end})]
        instances.append(dict(operation=op, mreg=0, base_word=0, offset=0,
                              footprint=dict(error="", accesses=accesses, holds=holds,
                                             read_release=0 if short == "L" else 34,
                                             write_release=34, done_age=34)))
        rows = []
        for cycle in range(37):
            row = {"cycle": cycle, "load_busy": int(short == "L" and 1 <= cycle <= 34),
                   "store_busy": int(short == "S" and 1 <= cycle <= 34)}
            for resource, write, age in (("vmem", short == "S", bank_start),
                                         ("mreg", short == "L", 3 if short == "L" else 1)):
                prefix = resource + ("_write" if write else "_read")
                row[prefix] = int(age <= cycle < age + 32)
                if resource == "vmem":
                    row[prefix + "_line"] = 64 + cycle - age
                else:
                    row[prefix + "_reg"], row[prefix + "_row"] = 3, cycle - age
            rows.append(row)
        traces[short] = rows
    return {"instances": instances}, traces


class EvidenceRejectionTest(unittest.TestCase):
    def assert_discrepancy(self, probe, traces):
        try:
            comparisons = CHECK.compare_footprints(probe, traces)
        except CHECK.CheckError:
            return
        fields = ("matches_normalized_stream", "matches_release_and_holds",
                  "matches_required_stream_coverage")
        self.assertTrue(any(record.get(key) is False for record in comparisons for key in fields),
                        "corrupt evidence was accepted")

    def test_control_record_and_missing_or_duplicate_stream(self):
        probe, traces = comparison_inputs()
        result = CHECK.compare_footprints(probe, traces)
        self.assertFalse(any(value is False for row in result for key, value in row.items()
                             if key.startswith("matches_")))
        for duplicate in (False, True):
            bad = copy.deepcopy(probe)
            accesses = bad["instances"][0]["footprint"]["accesses"]
            if duplicate:
                accesses.append(copy.deepcopy(accesses[0]))
            else:
                accesses.pop()
            self.assert_discrepancy(bad, traces)

    def test_wrong_index_unknown_range_and_completion_stream(self):
        for key, value in (("first", 1), ("anywhere", True), ("at_completion", True),
                           ("write", True), ("count", 31), ("step", 2)):
            with self.subTest(key=key):
                probe, traces = comparison_inputs()
                probe["instances"][0]["footprint"]["accesses"][0][key] = value
                self.assert_discrepancy(probe, traces)

    def test_premature_release_and_missing_or_short_reservations(self):
        for change in ("write_release", "path", "bank", "missing_hold"):
            with self.subTest(change=change):
                probe, traces = comparison_inputs()
                footprint = probe["instances"][0]["footprint"]
                if change == "write_release":
                    footprint["write_release"] = 33
                elif change == "missing_hold":
                    footprint["holds"].pop()
                else:
                    footprint["holds"][0 if change == "path" else 1]["to"] -= 1
                self.assert_discrepancy(probe, traces)

    def test_mutated_hardware_observation_and_missing_reference(self):
        probe, traces = comparison_inputs()
        traces["L"][1]["vmem_read_line"] += 1
        self.assert_discrepancy(probe, traces)
        probe, traces = comparison_inputs()
        probe["instances"].pop()
        with self.assertRaises(CHECK.CheckError):
            CHECK.compare_footprints(probe, traces)

    def test_manifest_mismatch_fails_before_tools_or_output(self):
        artifact_root = ROOT / "build/rtl-timing"
        artifact_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=artifact_root) as temporary:
            directory = Path(temporary)
            hardware = directory / "input.mlir"
            hardware.write_bytes(b"not valid MLIR; identity must be checked first\n")
            correct = dict(path=str(hardware), sha256=hashlib.sha256(hardware.read_bytes()).hexdigest(),
                           bytes=hardware.stat().st_size)
            manifest = directory / "manifest.json"
            for field, value in (("sha256", "0" * 64), ("bytes", 0)):
                with self.subTest(field=field):
                    manifest.write_text(json.dumps(dict(schema="atlas.retained_hw_ir.v0", state="verified",
                                                       hardware_ir={**correct, field: value})))
                    output = directory / ("rejected-" + field)
                    # Tool options deliberately absent: rejection must precede
                    # tool lookup, IR parsing and output creation.
                    with self.assertRaisesRegex(CHECK.INDEX.IndexError, "hash/size mismatch"):
                        CHECK.run(SimpleNamespace(manifest=manifest, output=output))
                    self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
