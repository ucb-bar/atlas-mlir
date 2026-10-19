"""Replay accepts only the observed outputs of the selected host program."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from check_model_source_grid_rtl_replay import check


class ModelSourceGridRtlReplayTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.bundle = root / "bundle.json"
        self.result = root / "result.json"
        self.observed = root / "observed"
        self.observed.mkdir()
        self.names = [f"real_host_trial{trial}_n{n_start}"
                      for trial in range(2) for n_start in (0, 32)]
        self.bundle.write_text(json.dumps({
            "schema": "atlas.source_mn_host_data.v1", "engine": "merlin_native",
            "rank": 2, "batch_count": 1, "m_rows": 3, "n_rows": 35,
            "tile_count": 2, "frozen_rhs": {"argument_index": 1},
            "target_identity": "atlas-rtl-selected:mxu0", "program_sha256": "a" * 64,
        }))
        outputs = []
        for index, name in enumerate(self.names):
            data = bytes([index + 1]) * 2048
            (self.observed / f"{name}.bin").write_bytes(data)
            outputs.append({"name": name, "sha256": hashlib.sha256(data).hexdigest()})
        self.report = {
            "schema": "atlas.selected_core_test_run.v1",
            "status": "passed_bounded_core_test", "rtl_revision": "selected",
            "fixtures": [{"name": name, "program_sha256": "a" * 64}
                         for name in self.names],
            "observed_outputs": outputs,
            "cases": [{"name": name, "bf16_cells": 1024} for name in self.names],
        }
        self._write_report()

    def _write_report(self) -> None:
        self.result.write_text(json.dumps(self.report))

    def test_checked_replay(self) -> None:
        record = check(self.bundle, self.result, self.observed)
        self.assertEqual(record["status"], "bound_rtl_outputs")
        self.assertEqual(len(record["outputs"]), 4)

    def test_modified_output_is_refused(self) -> None:
        path = self.observed / f"{self.names[0]}.bin"
        data = bytearray(path.read_bytes())
        data[0] ^= 1
        path.write_bytes(data)
        with self.assertRaisesRegex(ValueError, "changed or mismatched"):
            check(self.bundle, self.result, self.observed)

    def test_changed_program_is_refused(self) -> None:
        self.report["fixtures"][-1]["program_sha256"] = "b" * 64
        self._write_report()
        with self.assertRaisesRegex(ValueError, "changed or mismatched"):
            check(self.bundle, self.result, self.observed)


if __name__ == "__main__":
    unittest.main()
