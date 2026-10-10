"""Retention tool with stub firtool/circt-opt: lowering command, hierarchy redirects and structure checks."""
import json
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

TOOL = Path(__file__).resolve().parents[1] / "tools/retain-ee290-hw-ir.py"
FIRTOOL = """#!/bin/sh
[ "$1" = --version ] && { echo stub-firtool 1; exit 0; }
while [ $# -gt 0 ]; do case "$1" in -o) out=$2;; esac; shift; done
cat > "$out" <<'IR'
%s
IR
"""
HW = "hw.module @AtlasCore() {\n  seq.firreg\n}\nhw.module @ScalarCore() {\n}\nhw.module @DmaEngine() {\n}"
CIRCT_OPT = '#!/bin/sh\n[ "$1" = --version ] && echo stub-circt-opt\nexit 0\n'


class RetainTest(unittest.TestCase):
    def run_tool(self, hw):
        tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        for name, text in (("firtool", FIRTOOL % hw), ("circt-opt", CIRCT_OPT)):
            (tmp / name).write_text(text)
            (tmp / name).chmod((tmp / name).stat().st_mode | stat.S_IXUSR)
        (tmp / "in.fir").write_text("circuit AtlasCore :\n")
        (tmp / "opts.txt").write_text("disallowLocalVariables\n")
        (tmp / "anno.json").write_text(json.dumps([{"class": "sifive.enterprise.firrtl.ModuleHierarchyAnnotation", "filename": "/elsewhere/h.json"}]))
        result = subprocess.run([sys.executable, str(TOOL), "--firrtl", str(tmp / "in.fir"), "--annotations", str(tmp / "anno.json"),
                                 "--lowering-options", str(tmp / "opts.txt"), "--firtool", str(tmp / "firtool"), "--circt-opt", str(tmp / "circt-opt"),
                                 "--output", str(tmp / "out")], capture_output=True, text=True)
        return tmp / "out", result

    def test_verified_manifest_and_hierarchy_redirect(self):
        out, result = self.run_tool(HW)
        self.assertEqual(result.returncode, 0, result.stderr)
        manifest = json.loads((out / "manifest.json").read_text())
        self.assertEqual(manifest["state"], "verified")
        self.assertEqual(manifest["structure"]["sequential_op_occurrences"]["seq.firreg"], 1)
        self.assertEqual(manifest["firtool"]["version"], "stub-firtool 1")
        self.assertEqual([c["stage"] for c in manifest["commands"]], ["lower", "verify"])
        lower = manifest["commands"][0]["argv"]
        self.assertIn("--ir-hw", lower)
        self.assertEqual(json.loads((out / "retained-hw.anno.json").read_text())[0]["filename"], str(out / "top_module_hierarchy.json"))

    def test_missing_atlas_module_is_rejected(self):
        out, result = self.run_tool(HW.replace("@DmaEngine", "@Other"))
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(json.loads((out / "manifest.json").read_text())["state"], "structure_check_failed")


if __name__ == "__main__":
    unittest.main()
