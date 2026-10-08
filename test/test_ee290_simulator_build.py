"""Synthetic simulator preparation checks; no VCS or compiler is executed."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shlex
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).absolute().parents[1]
SPEC = importlib.util.spec_from_file_location("ee290_simulator_builder", ROOT / "tools/build-ee290-simulator.py")
B = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(B)


class SimulatorBuildTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="atlas_simulator_test_")
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        for name in ("sources", "lib", "inc", "tool/bin", "cpp/bin"):
            (self.base / name).mkdir(parents=True)
        self.sources = self.base / "sources"
        self.cpp = self.file("sources/dpi.cc", '#include "dpi.h"\n')
        self.header = self.file("sources/dpi.h", "typedef int word;\n")
        self.mem = self.file("sources/top.mems.v", "module mem; endmodule\n")
        self.driver = self.file("sources/TestDriver.v", "module TestDriver; endmodule\n")
        self.package = self.file("sources/pkg.sv", "package p; endpackage\n")
        self.filelist = self.file("sources/sources.f", "\n".join(map(str, (self.driver, self.mem, self.cpp))) + "\n")
        self.vcs = self.file("tool/bin/vcs", "#!/bin/sh\nexit 99\n", executable=True)
        self.cxx = self.file("cpp/bin/g++", "not executed\n", executable=True)
        self.cc = self.file("cpp/bin/gcc", "not executed\n", executable=True)
        for relative in B.VENDOR_FILES:
            self.file("tool/" + relative, "selected vendor bytes\n")
        for name, suffix in (("riscv", ".so"), ("fesvr", ".a"), ("dramsim", ".so")):
            self.file("lib/lib" + name + suffix, "selected library bytes\n")
        self.runtime = self.file("runtime/libstdc++.so.6.0.33", "selected C++ runtime bytes\n")

    def file(self, name, contents, executable=False):
        path = self.base / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents)
        if executable:
            path.chmod(0o755)
        return path

    def recipe(self, extra=()):
        args = ["vcs", "-full64", "-CFLAGS", "-O3 -std=c++17 -I" + str(self.sources),
                "-LDFLAGS", "-L" + str(self.base / "lib") + " -Wl,-rpath," + str(self.base / "lib"),
                "-lriscv", "-lfesvr", "-ldramsim", str(self.package), "-f", str(self.filelist),
                "-sverilog", "-timescale", "1ns/10ps", "-assert", "svaext", "-top", "TestDriver",
                "-debug_pp", "+incdir+" + str(self.sources), "+define+VCS", "+define+FSDB",
                "+define+STOP_COND=!TestDriver.reset", "-j160", *extra,
                "-o", str(self.base / "old_simv"), "-Mdir=" + str(self.base / "old_csrc")]
        return "#!/bin/sh -e\n# saved metadata\n" + shlex.join(args) + " 2>&1\n"

    def test_metadata_parser_rejects_shell_and_hidden_inputs(self):
        parsed = B.saved_recipe(self.recipe())
        self.assertNotIn("-debug_pp", parsed["flags"])
        self.assertEqual(parsed["libraries"], ["riscv", "fesvr", "dramsim"])
        for extra in ((";", "touch", "unexpected"), ("-P", str(self.package)),
                      ("-y", str(self.sources)), ("+define+DEBUG",)):
            with self.subTest(extra=extra), self.assertRaises(B.BuildError):
                B.saved_recipe(self.recipe(extra))
        with self.assertRaises(B.BuildError):
            B.saved_recipe(self.recipe().replace("-O3", "-include injected.h"))
        combined = B.saved_recipe(self.recipe().replace("-timescale 1ns/10ps", "-timescale=1ns/10ps"))
        self.assertIn("-timescale=1ns/10ps", combined["flags"])

    def test_restricted_path_precedes_filesystem_access(self):
        for bad in (self.base / "HaMmEr/../safe", self.base / "VLSI-child/target"):
            with self.assertRaises(B.BuildError):
                B.allowed(bad, missing=True)
        link = self.base / "link"
        link.symlink_to(self.base / "vLsI-hidden/target")
        with self.assertRaises(B.BuildError):
            B.allowed(link, missing=True)

    def test_filelist_requires_identified_literal_sources(self):
        known = {str(path): B.ident(path) for path in (self.filelist, self.driver, self.mem, self.cpp)}
        sources, records = B.filelist_sources([self.filelist], known)
        self.assertEqual(sources, [self.driver, self.mem, self.cpp])
        self.assertEqual(len(records), 1)
        self.filelist.write_text("+incdir+unidentified\n")
        known[str(self.filelist)] = B.ident(self.filelist)
        with self.assertRaises(B.BuildError):
            B.filelist_sources([self.filelist], known)
        self.filelist.write_text(str(self.base / "missing.sv") + "\n")
        known[str(self.filelist)] = B.ident(self.filelist)
        with self.assertRaises((B.BuildError, OSError)):
            B.filelist_sources([self.filelist], known)

    def test_sv_include_closure_and_unsupported_macro(self):
        header = self.file("inc/control.svh", "`define CONTROL 1\n")
        self.driver.write_text('`include "control.svh"\nmodule TestDriver; endmodule\n')
        self.assertEqual(B.sv_headers([self.driver], [self.base / "inc"]), [B.ident(header)])
        for contents in ('`include HEADER_MACRO\n', '`include "missing.svh"\n', '`include "../escape.svh"\n'):
            self.driver.write_text(contents)
            with self.assertRaises(B.BuildError):
                B.sv_headers([self.driver], [self.base / "inc"])

    def test_dependency_records_and_missing_library(self):
        paths = B.dependency_paths("atlas_dependencies: " + str(self.cpp) + " \\\n " + str(self.header) + "\n")
        self.assertEqual(paths, [self.cpp, self.header])
        for text in ("unrecognized: anything\n", "atlas_dependencies:\n", "atlas_dependencies: /missing/header.h\n"):
            with self.assertRaises((B.BuildError, OSError)):
                B.dependency_paths(text)
        self.assertEqual([Path(x["path"]).name for x in B.selected_libraries(B.saved_recipe(self.recipe()))],
                         ["libriscv.so", "libfesvr.a", "libdramsim.so"])
        (self.base / "lib/libfesvr.a").unlink()
        with self.assertRaises(B.BuildError):
            B.selected_libraries(B.saved_recipe(self.recipe()))

    def preparation_args(self, output):
        record = self.file("saved_vcs_record", self.recipe())
        report = {"target_config": "EE290SimConfig", "state": "integrated_execution_passed",
                  "provenance": {"observed_simulator_build_record": B.ident(record),
                                 "simulator_sources": [B.ident(path) for path in (self.filelist, self.driver, self.mem, self.cpp)],
                                 "observed_build_record_direct_sources": [B.ident(self.package)]}}
        witness = self.file("witness.json", json.dumps(report))
        return argparse.Namespace(output=output, witness_report=witness,
                                  expected_witness_sha256=B.ident(witness)["sha256"],
                                  vcs=self.vcs, cxx=self.cxx, cc=None, vcs_version="synthetic-vcs",
                                  cxx_version="synthetic-cxx", jobs=8, timeout_seconds=10,
                                  cxx_runtime=self.runtime,
                                  replacement_inventory=None, expected_replacement_sha256=None)

    def fake_dependencies(self, output, kind, argv, inputs, outputs, environment, timeout_seconds):
        self.assertEqual(kind, "cpp_dependencies")
        self.assertEqual(Path(argv[0]), self.cxx)
        self.assertIn("-M", argv)
        self.assertNotIn("HOME", environment)
        output.mkdir(parents=True)
        source = Path(argv[-1])
        header = source.parent / self.header.name
        (output / "dependencies.d").write_text("atlas_dependencies: " + str(source) + " " + str(header) + "\n")
        (output / "phase.json").write_text('{"state":"phase_completed"}\n')
        return {"state": "phase_completed"}

    def test_prepare_snapshots_then_separate_pinned_run(self):
        output = self.base / "prepared"
        args = self.preparation_args(output)
        with patch.object(B, "new_output", return_value=output), \
                patch.object(B._CAPTURE, "capture_phase", side_effect=self.fake_dependencies) as capture:
            result = B.prepare(args)
        self.assertEqual(result["state"], "prepared", result)
        self.assertEqual(capture.call_count, 2)  # Original + relocated dependency discovery; no VCS.
        plan_path = output / "plan.json"
        plan = json.loads(plan_path.read_text())
        self.assertEqual(plan["argv"][-3:], ["-o", "{output}/simv", "-Mdir={output}/csrc"])
        self.assertNotIn("-debug_pp", plan["argv"])
        self.assertNotIn("+define+DEBUG", plan["argv"])
        self.assertIn("-lstdc++", plan["argv"])
        self.assertEqual(plan["cxx_runtime"], B.ident(self.runtime))
        library_dir = output / "inputs/link-libraries"
        for name in ("libstdc++.so", "libstdc++.so.6", "libstdc++.so.6.0.33"):
            self.assertEqual(B.ident(library_dir / name)["sha256"], B.ident(self.runtime)["sha256"])
            self.assertTrue(any(entry["identity"]["path"] == str(library_dir / name) for entry in plan["inputs"]))
        self.assertTrue(all(flag in plan["argv"] for flag in ("-debug_access+all", "-kdb", "-lca")))
        compiled_list = Path(plan["argv"][plan["argv"].index("-f") + 1])
        self.assertIn(str(output / "inputs"), compiled_list.read_text())
        self.assertTrue(any(entry["role"] == "header" and entry["original"]["path"] == str(self.header)
                            for entry in plan["origins"]))
        args = argparse.Namespace(output=self.base / "compiled", plan=plan_path,
                                  expected_plan_sha256=B.ident(plan_path)["sha256"], timeout_seconds=10)
        def fake_compile(output, kind, argv, inputs, outputs, environment, timeout_seconds):
            self.assertEqual(kind, "vcs_compile")
            self.assertEqual(Path(argv[0]), self.vcs)
            self.assertEqual(environment["LM_PROJECT"], "bwrc_users")
            output.mkdir()
            (output / "phase.json").write_text('{"state":"phase_completed"}\n')
            return {"state": "phase_completed"}
        with patch.object(B, "new_output", return_value=args.output), \
                patch.object(B._CAPTURE, "capture_phase", side_effect=fake_compile) as capture, \
                patch.dict(os.environ, {"LM_PROJECT": "bwrc_users"}, clear=True):
            result = B.run_plan(args)
        self.assertEqual(result["state"], "phase_completed")
        self.assertEqual(capture.call_count, 1)
        # Even an explicitly repinned plan cannot smuggle hidden source options.
        plan["argv"].insert(1, "-P")
        plan_path.write_text(json.dumps(plan))
        args.expected_plan_sha256 = B.ident(plan_path)["sha256"]
        with patch.object(B, "new_output", return_value=self.base / "other"), \
                patch.object(B._CAPTURE, "capture_phase") as capture:
            with self.assertRaises(B.BuildError):
                B.run_plan(args)
            capture.assert_not_called()

    def test_versioned_library_alias_is_retained_as_verified_regular_copy(self):
        link = self.base / "lib/libriscv.so"
        versioned = link.with_name("libriscv.so.0")
        link.rename(versioned)
        link.symlink_to(versioned.name)
        recipe = B.saved_recipe(self.recipe())
        libraries = B.selected_libraries(recipe)
        self.assertEqual(Path(libraries[0]["path"]), versioned)
        output = self.base / "snapshot"
        origins = B.snapshot_libraries(output, recipe["libraries"], libraries)
        directory = output / "inputs/link-libraries"
        for filename in ("libriscv.so", "libriscv.so.0"):
            copied = directory / filename
            self.assertFalse(copied.is_symlink())
            self.assertEqual(B.ident(copied)["sha256"], B.ident(versioned)["sha256"])
            self.assertTrue(any(item["snapshot"]["path"] == str(copied) for item in origins))
        recipe["libdirs"] = [directory]
        resolved = B.selected_libraries(recipe)
        self.assertEqual(Path(resolved[0]["path"]).name, "libriscv.so")

    def test_replacement_is_pinned_and_selected(self):
        replacement = self.file("fresh/top.mems.v", "module mem; wire newer; endmodule\n")
        selected = {str(self.mem): B.ident(self.mem)}
        inventory = {str(self.mem): B.ident(replacement)}
        self.assertEqual(B.replacements(inventory, selected)[str(self.mem)], B.ident(replacement))
        with self.assertRaises(B.BuildError):
            B.replacements({str(self.driver): B.ident(replacement)}, selected)
        inventory[str(self.mem)]["sha256"] = "0" * 64
        with self.assertRaises(B.BuildError):
            B.replacements(inventory, selected)

    def test_vendor_setup_explicit_chain_and_unknown_include(self):
        primary = self.file("tool/bin/synopsys_sim.setup", "OTHERS = $SYNOPSYS_DW_FPGA_LIB_SETUP\nWORK > DEFAULT\nDEFAULT : .\n")
        child = self.file("tool/bin/synopsys_dw_fpga_lib.setup", "DWARE : $SYNOPSYS_SIM/$ARCH/packages/dware\n")
        self.assertEqual(B.vendor_setup(self.base / "tool"), [B.ident(primary), B.ident(child)])
        primary.write_text("OTHERS = unknown.setup\n")
        with self.assertRaises(B.BuildError):
            B.vendor_setup(self.base / "tool")

    def test_preparation_bounds_and_pinned_witness_fail_before_capture(self):
        output = self.base / "never_created"
        args = self.preparation_args(output)
        args.jobs = 9
        with patch.object(B, "new_output", return_value=output), patch.object(B._CAPTURE, "capture_phase") as capture:
            with self.assertRaises(B.BuildError):
                B.prepare(args)
            capture.assert_not_called()
        self.assertFalse(output.exists())
        args.jobs = 8
        args.expected_witness_sha256 = "0" * 64
        with patch.object(B, "new_output", return_value=output), patch.object(B._CAPTURE, "capture_phase") as capture:
            with self.assertRaises(B.BuildError):
                B.prepare(args)
            capture.assert_not_called()
        self.assertFalse(output.exists())

    def test_failed_discovery_retains_gap_without_plan(self):
        output = self.base / "failed"
        args = self.preparation_args(output)
        def failed(output, *unused):
            output.mkdir(parents=True)
            (output / "phase.json").write_text('{"state":"phase_failed"}\n')
            return {"state": "phase_failed"}
        with patch.object(B, "new_output", return_value=output), \
                patch.object(B._CAPTURE, "capture_phase", side_effect=failed):
            result = B.prepare(args)
        self.assertEqual(result["state"], "preparation_failed")
        self.assertFalse((output / "plan.json").exists())
        self.assertTrue((output / "report.json").exists())


if __name__ == "__main__":
    unittest.main()
