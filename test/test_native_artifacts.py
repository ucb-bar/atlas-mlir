"""Native publication works without atlas-opt, LLVM or the selection engine."""

from __future__ import annotations

import hashlib
import json
import struct
import tempfile
import unittest
from pathlib import Path

from atlas_mlir_native.artifacts import publish_program


class NativeArtifactsTest(unittest.TestCase):
    def test_publishes_command_words_and_exact_stage_identities(self):
        import pickle
        from dataclasses import FrozenInstanceError

        from atlas_mlir_native import binding, native

        candidate = binding._MaterializedCandidate(
            [32, 33], "virtual", "machine", {}, [0x73], ("opt", "emit"), 4096, 8192,
        )
        self.assertIs(binding._MaterializedCandidate, native._MaterializedCandidate)
        self.assertEqual(type(candidate).__module__, "atlas_mlir_native.binding")
        self.assertEqual(pickle.loads(pickle.dumps(candidate)), candidate)
        with self.assertRaises(FrozenInstanceError):
            candidate.words = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "constants").mkdir()
            payload = b"\x00\x80\xff\x7f"
            (root / "constants/constant-000.bin").write_bytes(payload)
            constants = [{"path": "constants/constant-000.bin",
                          "sha256": hashlib.sha256(payload).hexdigest(),
                          "address": 4096, "bytes": len(payload)}]
            plan = {"constant_inputs": constants, "output_addresses": [8192]}
            manifest = publish_program(
                root, words=[0x01020304, 0xFFFFFFFF], virtual="virtual\n",
                machine="machine\n", program={"words": [0x01020304, 0xFFFFFFFF]},
                plan=plan, selection={"order": ["x", "y"]},
                metadata={"artifact_kind": "program", "engine": "merlin_native",
                          "qualification_scope": "diagnostic_only",
                          "request_digest": "request", "target_identity": "selected"},
                constant_rows=constants, qualification_machine="mandatory\n",
                qualification_program={"words": [1]},
            )
            self.assertEqual((root / "program.bin").read_bytes(),
                             b"\x04\x03\x02\x01\xff\xff\xff\xff")
            self.assertEqual((root / "execution_plan.json").read_bytes(),
                             (json.dumps(plan, sort_keys=True) + "\n").encode())
            self.assertEqual(json.loads((root / "manifest.json").read_bytes()), manifest)
            self.assertEqual(manifest["qualification_scope"], "diagnostic_only")
            for name, digest in manifest["stage_artifacts"].items():
                self.assertEqual(hashlib.sha256((root / name).read_bytes()).hexdigest(), digest)
            self.assertEqual(manifest["machine_sha256"],
                             manifest["stage_artifacts"]["atlas-machine.mlir"])
            self.assertEqual(manifest["constant_inputs"], constants)

    def test_cannot_override_generated_identity_or_publish_invalid_words(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = dict(words=[1], virtual="", machine="", program={}, plan={}, selection={})
            with self.assertRaisesRegex(ValueError, "computed identities"):
                publish_program(root, **args, metadata={"binary_sha256": "forged"})
            self.assertFalse((root / "manifest.json").exists())
            args["words"] = [2 ** 32]
            with self.assertRaises(struct.error):
                publish_program(root, **args, metadata={})
            self.assertFalse((root / "manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
