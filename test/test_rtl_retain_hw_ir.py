"""Lowering policy regressions; the actual EE290 lowering is an explicit regeneration stage."""
import importlib.util
from pathlib import Path
import tempfile
import unittest


SOURCE = Path(__file__).resolve().parents[1] / "tools" / "retain-ee290-hw-ir.py"
SPEC = importlib.util.spec_from_file_location("retain_hw_ir", SOURCE)
retain = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(retain)


class RetainedHWIRTests(unittest.TestCase):
    def test_debug_matches_recorded_timing_lowering_policy(self):
        command = retain.lowering_command("firtool", Path("input.fir"), Path("debug.anno.json"),
                                          "noAlwaysComb\n", Path("out"), "debug")
        self.assertEqual(command, [
            "firtool", "--format=fir", "--export-module-hierarchy", "--verify-each=true",
            "--warn-on-unprocessed-annotations", "--disable-annotation-classless",
            "--disable-annotation-unknown", "--mlir-timing", "--mlir-print-debuginfo",
            "--lowering-options=noAlwaysComb", "-O=debug", "--preserve-values=named",
            "--annotation-file=debug.anno.json", "--ir-hw", "-o", "out/ee290.debug.hw.mlir", "input.fir"])
        self.assertFalse(any(flag.startswith("--repl-seq-mem") for flag in command))

    def test_default_keeps_sequential_memory_replacement(self):
        command = retain.lowering_command("firtool", Path("input.fir"), Path("hw.anno.json"),
                                          "noAlwaysComb", Path("out"), "legacy")
        self.assertIn("--repl-seq-mem", command)
        self.assertIn("--repl-seq-mem-file=out/mems.conf", command)
        self.assertNotIn("-O=debug", command)
        self.assertNotIn("--preserve-values=named", command)
        self.assertEqual(command[-2:], ["out/ee290.hw.mlir", "input.fir"])

    def test_hierarchy_redirect_keeps_other_annotations(self):
        inline = {"class": "firrtl.transforms.BlackBoxInlineAnno", "name": "Toy.sv", "text": "module Toy; endmodule"}
        sidecar = [{"class": cls, "filename": "/old/file.json"} for cls in retain.HIERARCHY] + [inline]
        prepared = retain.prepare_annotations(sidecar, Path("out"))
        self.assertEqual(prepared[-1], inline)
        self.assertEqual([a["filename"] for a in prepared[:-1]],
                         [str(Path("out") / name) for name in retain.HIERARCHY.values()])

    def test_sha256_hashes_file_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input"
            source.write_bytes(b"abc")
            self.assertEqual(retain.sha256(source),
                             "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")


if __name__ == "__main__":
    unittest.main()
