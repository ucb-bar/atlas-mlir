"""Pure admission and resource-wrapper controls; no RTL build or banned objects."""

import importlib.util
import os
from pathlib import Path
import runpy
import stat
import unittest
from unittest.mock import patch


ROOT = Path(__file__).absolute().parent
SPEC = importlib.util.spec_from_file_location("mxu_trace_driver", ROOT / "run.py")
DRIVER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DRIVER)
VECTOR_SPEC = importlib.util.spec_from_file_location("vector_cases", ROOT / "prepare_vector.py")
VECTOR = importlib.util.module_from_spec(VECTOR_SPEC)
VECTOR_SPEC.loader.exec_module(VECTOR)


class Controls(unittest.TestCase):
    def test_pack_stream_and_mxu_halves_are_distinct(self):
        source = [[row * 32 + column for column in range(32)] for row in range(32)]
        packed = VECTOR.initial_source_rows({"kind": "pack", "source_bf16": source})
        stream = [(packed[bank, row] >> (16 * lane)) & 65535
                  for bank in range(2) for row in range(32) for lane in range(16)]
        self.assertEqual(stream, list(range(1024)))
        halves = VECTOR.initial_source_rows({"kind": "rescale", "source_bf16": source})
        wrong_stream = [(halves[bank + 2, row] >> (16 * lane)) & 65535
                        for bank in range(2) for row in range(32) for lane in range(16)]
        self.assertNotEqual(wrong_stream, stream)
        for row in range(32):
            for column in range(32):
                self.assertEqual((halves[2 + column // 16, row] >> (16 * (column % 16))) & 65535, source[row][column])

    def test_lexical_refusal_precedes_filesystem(self):
        with patch.object(DRIVER.os, "lstat") as probe:
            for value in ("/allowed/HAMMER-copy/x", "/allowed/pre-vLsI/x", "/allowed/../x"):
                with self.assertRaises(ValueError):
                    DRIVER.safe_path(value)
            probe.assert_not_called()

    def test_child_symlink_is_rejected_without_target_access(self):
        visited = []
        def probe(path):
            visited.append(str(path))
            mode = stat.S_IFLNK if str(path) == "/allowed/link" else stat.S_IFDIR
            return type("Metadata", (), {"st_mode": mode})()
        with patch.object(DRIVER.os, "lstat", side_effect=probe), patch.object(DRIVER.os, "readlink") as target:
            with self.assertRaises(ValueError):
                DRIVER.safe_path("/allowed/link/leaf")
            target.assert_not_called()
        self.assertEqual(visited, ["/allowed", "/allowed/link"])

    def test_wrapper_bounds_jobs_and_binds_make_tools(self):
        arguments = ["wrapper", "--build", "-j", "0", "-MAKEFLAGS", "-j 2", "file.sv"]
        environment = {"ATLAS_REAL_VERILATOR": "/allowed/verilator_bin",
                       "ATLAS_MAKE_OVERRIDES": "CXX=/allowed/cxx LINK=/allowed/cxx AR=/allowed/ar"}
        with patch.dict(os.environ, environment), patch("sys.argv", arguments), patch.object(os, "execv") as execute:
            runpy.run_path(str(ROOT / "verilator_wrapper.py"), run_name="__main__")
        self.assertEqual(execute.call_args.args,
                         ("/allowed/verilator_bin", ["/allowed/verilator_bin", "--build", "-j", "2",
                          "-MAKEFLAGS", "-j 2 CXX=/allowed/cxx LINK=/allowed/cxx AR=/allowed/ar", "file.sv"]))


if __name__ == "__main__":
    unittest.main()
