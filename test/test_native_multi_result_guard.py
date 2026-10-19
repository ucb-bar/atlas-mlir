"""Atlas emission refuses Merlin multi-result candidates it cannot bind."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
MERLIN = os.environ.get("MERLIN_NATIVE_ROOT")
ATLAS_BIN = os.environ.get("ATLAS_OOT_BIN_DIR")


@unittest.skipUnless(MERLIN and ATLAS_BIN, "set MERLIN_NATIVE_ROOT and ATLAS_OOT_BIN_DIR")
class NativeMultiResultGuardTest(unittest.TestCase):
    def setUp(self) -> None:
        for path in (ROOT / "python", Path(MERLIN) / "src"):
            sys.path.insert(0, str(path))
            self.addCleanup(sys.path.remove, str(path))
        temporary_root = Path(os.environ.get("TMPDIR", ROOT / "out"))
        temporary_root.mkdir(parents=True, exist_ok=True)
        temporary = tempfile.TemporaryDirectory(dir=temporary_root)
        self.addCleanup(temporary.cleanup)
        self.work = Path(temporary.name)

    def test_projection_candidate_is_refused_before_emission(self) -> None:
        from atlas_mlir_native.binding import AtlasNativeBinding
        from merlin.semantic_compiler.model import KernelRequest
        from merlin.semantic_compiler.search import SearchLimits
        from merlin.semantic_compiler.snapshot import (
            NativeSnapshot,
            build_native_snapshot,
            packaged_egg_bridge_path,
        )

        snapshot = build_native_snapshot(
            AtlasNativeBinding.profile(), destination=self.work / "snapshot",
            crate=packaged_egg_bridge_path(), cargo_target_dir=self.work / "cargo",
            source_revision="selected-rtl-copy-integration",
        )
        request = KernelRequest.from_record(json.loads(
            (ROOT / "test/examples/native_bf16_relu_request.json").read_text()
        ))
        select = NativeSnapshot.select
        decisions = []

        def forge_projection(self, request, **kwargs):
            accept = kwargs.pop("accept_materialization")
            selected = select(self, request, **kwargs)
            self_test.assertEqual(selected.status, "selected")
            values = list(selected.graph.values)
            # Present the last result as a later projection of the first
            # instruction, as Merlin does for a shared multi-result bundle.
            owner = next(value.id for value in values if value.kind == "instruction")
            values[-1] = replace(values[-1], kind="projection", projection_of=owner)
            forged = replace(selected, graph=replace(selected.graph, values=tuple(values)))
            decisions.append(accept(forged))
            return select(self, request, accept_materialization=accept, **kwargs)

        self_test = self
        destination = self.work / "compiled"
        with patch.object(NativeSnapshot, "select", forge_projection):
            AtlasNativeBinding.compile(
                snapshot, request, fixed_inputs={"x": 3}, fixed_outputs=(35,),
                target_source=ROOT, destination=destination, limits=SearchLimits(),
            )
        self.assertEqual(decisions, [
            (False, "Atlas emission has no multi-result instruction binding"),
        ])
        # The refused candidate published nothing; the real one compiled.
        self.assertTrue((destination / "program.bin").is_file())


if __name__ == "__main__":
    unittest.main()
