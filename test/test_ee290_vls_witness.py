"""VCS witness runner: command construction and result parsing. VCS itself needs a license and is never run here.

With ATLAS_OOT_BIN_DIR and ATLAS_OP_TIMING set, a dry run schedules and emits the copy example with
the real compiler and substitutes stub host compiler and simulator scripts.
"""
import hashlib
import importlib.util
import os
import stat
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("witness", ROOT / "tools/run-ee290-vls-witness.py")
WITNESS = importlib.util.module_from_spec(spec)
spec.loader.exec_module(WITNESS)
PASS_LOG = "".join(f"EE290_VLS_PANEL_PASSED panel={n} output_words=256 preserved_words=1280\n" for n in range(3)) + "EE290_VLS_PASSED panels=3\n"
BIN, FACTS = os.environ.get("ATLAS_OOT_BIN_DIR"), os.environ.get("ATLAS_OP_TIMING")


class WitnessTest(unittest.TestCase):
    def test_commands_select_facts_and_load_the_host_binary(self):
        args = Namespace(atlas_opt=Path("opt"), atlas_emit=Path("emit"), host_cc=Path("cc"), simulator=Path("simv"), program=Path("p.mlir"),
                         facts=Path("/f.json"), schedule="schedule", dma_wait=True, max_cycles=77, sim_arg=["+x"])
        schedule, emit, build, simulate = WITNESS.commands(args, Path("/out"), "abc")
        self.assertEqual(schedule[:3], ["opt", "p.mlir", "--select-atlas-rtl-evidence=op-timing=/f.json op-timing-sha256=abc dma=wait"])
        self.assertIn("--schedule-atlas-stream", schedule)
        self.assertIn("--verify-atlas-rtl-timing", schedule)
        self.assertEqual(emit, ["emit", "/out/final.mlir"])
        self.assertEqual(build[-3:], [str(WITNESS.HOST), "-o", "/out/host.riscv"])
        self.assertEqual(simulate[0], "simv")
        self.assertIn("+loadmem=/out/host.riscv", simulate)
        self.assertIn("+max-cycles=77", simulate)
        self.assertEqual(simulate[-2:], ["+permissive-off", "/out/host.riscv"])
        args.schedule, args.dma_wait = "delay", False
        schedule = WITNESS.commands(args, Path("/out"), "abc")[0]
        self.assertIn("--insert-atlas-delays", schedule)
        self.assertNotIn("dma=wait", schedule[2])

    def test_include_file(self):
        self.assertEqual(WITNESS.include_file(["00000013", "00000073"]),
                         "#define ATLAS_PROGRAM_WORDS 2U\nstatic const uint32_t atlas_program[] = {\n  0x00000013U,\n  0x00000073U,\n};\n")

    def test_result_parsing(self):
        self.assertEqual(WITNESS.judge(0, PASS_LOG), "passed")
        self.assertEqual(WITNESS.judge(1, PASS_LOG), "failed")
        self.assertEqual(WITNESS.judge(0, PASS_LOG.replace("panels=3", "panels=2")), "failed")
        self.assertEqual(WITNESS.judge(0, PASS_LOG + "EE290_VLS_MISMATCH panel=1 word=3\n"), "failed")
        self.assertEqual(WITNESS.judge(0, PASS_LOG + "Error: assertion failed\n"), "failed")
        self.assertEqual(WITNESS.judge(1, "queuing for license\n"), "license_unavailable")

    @unittest.skipUnless(BIN and FACTS, "requires ATLAS_OOT_BIN_DIR and ATLAS_OP_TIMING")
    def test_dry_run_with_stub_host_compiler_and_simulator(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cc, sim = tmp / "cc", tmp / "simv"
            cc.write_text('#!/bin/sh\nwhile [ $# -gt 0 ]; do [ "$1" = -o ] && out=$2; shift; done; : > "$out"\n')
            sim.write_text("#!/bin/sh\nprintf '%s' '" + PASS_LOG + "'\n")
            for path in (cc, sim): path.chmod(path.stat().st_mode | stat.S_IXUSR)
            code = WITNESS.main(["--program", str(ROOT / "test/examples/ee290_vls_copy.mlir"), "--facts", FACTS,
                                 "--atlas-opt", f"{BIN}/atlas-opt", "--atlas-emit", f"{BIN}/atlas-emit",
                                 "--simulator", str(sim), "--host-cc", str(cc), "--output", str(tmp / "out")])
            self.assertEqual(code, 0)
            self.assertIn("ATLAS_PROGRAM_WORDS 10U", (tmp / "out/atlas_program.inc").read_text())
            self.assertIn(hashlib.sha256(Path(FACTS).read_bytes()).hexdigest(), (tmp / "out/final.mlir").read_text())


if __name__ == "__main__":
    unittest.main()
