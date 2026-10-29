"""The pipeline driver keeps every stage's artifact and stops at the first failure."""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from test_virtual_scheduling import wide_return_block
from test_virtual_ssa import BIN, ROOT
from test_virtual_lowering import virtual_chain

ATTENTION = ROOT / "test/examples/virtual_attention_tile.mlir"


def report_of(source: Path, pipeline: str, directory: Path) -> tuple[subprocess.CompletedProcess[str], dict]:
    """Run `pipeline` with an allocation report, returning the run and the report."""
    path = directory / "allocation.json"
    result = subprocess.run([str(BIN / "atlas-opt"), f"--{pipeline}=allocation-report={path}", str(source)],
                            text=True, capture_output=True)
    return result, json.loads(path.read_text())


def drive(source: Path, output: Path, *options: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(ROOT / "tools/lower_atlas_program.py"),
                           str(source), "--output-dir", str(output),
                           "--atlas-bin-dir", str(BIN), *options],
                          text=True, capture_output=True)


class LowerAtlasProgramTest(unittest.TestCase):
    def test_spill_stage_is_retained_before_allocation(self) -> None:
        from test_virtual_spilling import raw_roundtrip

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "copy.mlir"
            source.write_text(raw_roundtrip())
            output = Path(directory) / "compiled"
            result = drive(source, output)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('"atlas.virtual_spill_bf16"', (output / "2-spilled.mlir").read_text())
            allocation = json.loads((output / "3-allocation.json").read_text())
            self.assertTrue(allocation["fits"])
            self.assertTrue(any(row["use"] == "spill" for row in allocation["vmem"]))
            self.assertTrue((output / "5-words.hex").read_text().strip())

    def test_paths_with_spaces_and_quotes_can_be_replayed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "input 'attention'.mlir"
            source.write_text(ATTENTION.read_text())
            done = Path(temp) / 'output "attention"'
            result = drive(source, done)
            self.assertEqual(result.returncode, 0, result.stderr)
            original = (done / "5-words.hex").read_bytes()
            # The logged commands include their working directory and work
            # from an unrelated directory, with no reliance on this driver.
            replay = subprocess.run(["sh", str(done / "commands.txt")],
                                    cwd=temp, text=True, capture_output=True)
            self.assertEqual(replay.returncode, 0, replay.stderr)
            self.assertEqual((done / "5-words.hex").read_bytes(), original)

    def test_missing_tool_is_logged_without_a_traceback(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            done = Path(temp) / "out"
            result = drive(ATTENTION, done, "--atlas-bin-dir", temp)
            self.assertEqual(result.returncode, 1)
            self.assertNotIn("Traceback", result.stderr)
            self.assertIn("atlas-opt", (done / "log.txt").read_text())
            self.assertEqual(len((done / "commands.txt").read_text().splitlines()), 1)
            self.assertFalse((done / "1-virtual.mlir").exists())

    def test_wait_insertion_cannot_overflow_instruction_capacity(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "long.mlir"
            source.write_text(virtual_chain(1050))
            done = Path(temp) / "out"
            result = drive(source, done)
            self.assertEqual(result.returncode, 1, result.stderr)
            report = json.loads((done / "3-allocation.json").read_text())
            self.assertTrue(report["fits"])
            self.assertLess(report["untimed_words"]["words"], 2048)
            self.assertIn("2048-word policy limit", result.stderr)
            self.assertFalse((done / "4-timed.mlir").exists())
            self.assertFalse((done / "5-words.hex").exists())

    def test_legacy_resources_are_included_in_the_report(self) -> None:
        # Boundary tiles use the synchronous helper channels unless
        # boundary DMA conversion runs, which the unscheduled pipeline skips.
        with tempfile.TemporaryDirectory() as temp:
            result, report = report_of(ROOT / "test/examples/virtual_fp8_two_layer_mlp.mlir",
                                       "atlas-compile-unscheduled", Path(temp))
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(report["dma"]["synchronous_channels"], [0, 1])
            self.assertEqual(report["dma"]["peak_in_flight"], 1)
            self.assertTrue(any(unit["weight_slots_used"] == 1 and
                                unit["acc_slots_used"] == 1
                                for unit in report["mxu"]["units"]))

    def test_artifacts_or_the_failing_stage(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            done = Path(temp) / "attention"
            result = drive(ATTENTION, done)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(sorted(path.name for path in done.iterdir()), [
                "1-virtual.mlir", "2-scheduled.mlir", "2-spilled.mlir", "3-allocated.mlir",
                "3-allocation.json", "4-timed.mlir", "5-map.json", "5-program.json", "5-words.hex",
                "commands.txt", "log.txt"])
            self.assertEqual(len((done / "commands.txt").read_text().splitlines()), 8)
            report = json.loads((done / "3-allocation.json").read_text())
            self.assertTrue(report["fits"])
            self.assertEqual(report["failures"], [])
            for kind in ("bf16", "fp8", "scalar"):
                demand = report["registers"][kind]
                self.assertTrue(demand["fits"], kind)
                self.assertLessEqual(demand["peak_live"], demand["used"])
                self.assertLessEqual(demand["used"], demand["capacity"])
            # With FP8 values, BF16 pairs sit above the FP8 registers.
            self.assertEqual(report["registers"]["bf16"]["capacity"], 15)
            # Each value is named as 2-spilled.mlir, the lowering's input, defines it, on its line.
            lines = (done / "2-spilled.mlir").read_text().splitlines()
            for value in report["values"]:
                file, line, _ = value["loc"].split(":")
                self.assertEqual(file, "2-spilled.mlir")
                self.assertRegex(lines[int(line) - 1].split(" = ")[0],
                                 re.escape(value["value"].split("#")[0]) + r"\b")
            self.assertEqual({value["kind"] for value in report["values"]},
                             {"bf16", "fp8", "dma_immediate", "mxu_weight", "mxu_acc", "dma"})
            self.assertEqual(report["dma"]["peak_in_flight"],
                             1 + max(value["staging_window"] for value in report["values"]
                                     if value["kind"] == "dma"))
            self.assertLessEqual(report["untimed_words"]["words"], 2048)

            # 32 BF16 tiles live across one edge exceed 31 register pairs. The
            # spill stage recovers them, and the compact timed schedule fits
            # the instruction limit.
            source = Path(temp) / "wide.mlir"
            source.write_text(wide_return_block(32))
            spilled = Path(temp) / "wide"
            result = drive(source, spilled)
            self.assertIn('"atlas.virtual_save"', (spilled / "2-spilled.mlir").read_text())
            self.assertTrue(json.loads((spilled / "3-allocation.json").read_text())["fits"])
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertLessEqual(len((spilled / "5-words.hex").read_text().split()), 2048)

    def test_allocation_report_names_the_kind_that_does_not_fit(self) -> None:
        # Lowered without spilling, 32 BF16 tiles live across one edge cannot
        # fit 31 register pairs; the report still names that kind, and only it.
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "wide.mlir"
            source.write_text(wide_return_block(32))
            result, report = report_of(source, "lower-atlas-virtual-to-machine", Path(temp))
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("exceeds 31 physical pairs", result.stderr)
            self.assertFalse(report["fits"])
            self.assertEqual(len(report["failures"]), 1)
            self.assertIn("exceeds 31 physical pairs", report["failures"][0])
            bf16 = report["registers"]["bf16"]
            self.assertFalse(bf16["fits"])
            self.assertIsNone(bf16["used"])
            self.assertGreater(bf16["peak_live"], bf16["capacity"])
            self.assertTrue(report["registers"]["fp8"]["fits"])
            self.assertTrue(report["registers"]["scalar"]["fits"])
            self.assertNotIn("untimed_words", report)
            self.assertEqual([value for value in report["values"] if "registers" in value], [])

    def test_a_key_transposed_on_the_xlu_lowers_and_emits(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            done = Path(temp) / "out"
            result = drive(ROOT / "test/examples/virtual_attention_tile_transposed_key.mlir",
                           done)
            self.assertEqual(result.returncode, 0, result.stderr)
            timed = (done / "4-timed.mlir").read_text()
            fields = re.findall(r'"atlas.xlu_transpose"\(%\d+\) <\{dst = (\d+) : i32, src = (\d+) : i32\}>',
                                timed)
            self.assertEqual(len(fields), 1, timed)
            # The transpose does not overwrite its source register in place.
            self.assertNotEqual(*fields[0])
            program = json.loads((done / "5-program.json").read_text())
            words = [op["word_u32"] for op in program["instructions"]
                     if op["operation"] == "atlas.xlu_transpose"]
            dst, src = map(int, fields[0])
            # VTRPOSE_XLU: vs1[18:13], vd[12:7], opcode 0x6b.
            self.assertEqual(words, [(src << 13) | (dst << 7) | 0x6B])
            # Every load issues before the first MXU use, but a loaded tile
            # takes its register where it is first used: at most K and its
            # transpose, or V and the packed P, are live together.
            report = json.loads((done / "3-allocation.json").read_text())
            self.assertEqual(report["registers"]["fp8"]["peak_live"], 2)


if __name__ == "__main__":
    unittest.main()
