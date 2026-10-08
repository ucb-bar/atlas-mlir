"""Synthetic source-overlay tests; no Java compiler or elaborator is run."""

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import zipfile


ROOT = Path(__file__).absolute().parents[1]
SPEC = importlib.util.spec_from_file_location("ee290_source_elaboration", ROOT / "tools/elaborate-ee290-source.py")
SOURCE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SOURCE)


def identity(path):
    data = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return identity(path)


def required_classes(config="chipyard:SelectedConfig"):
    package, name = config.split(":")
    return {"atlas.tile.AtlasCore", "atlas.scalar.ScalarCore", "atlas.lsu.LSU",
            "atlas.config.WithAtlasTile", package + "." + name,
            "chipyard.config.AbstractConfig", "chipyard.iobinders.WithUARTIOCells",
            "chipyard.iobinders.WithDebugIOCells", "chipyard.iobinders.WithSerialTLIOCells",
            "chipyard.iobinders.WithChipIdIOCells"}


class SourceElaborationTest(unittest.TestCase):
    def setUp(self):
        base = SOURCE.CHECK.allowed(ROOT / "build/rtl-timing", missing=True)
        base.mkdir(parents=True, exist_ok=True)
        self.scratch = tempfile.TemporaryDirectory(prefix="source-elaboration-test-", dir=base)
        self.directory = Path(self.scratch.name)
        sources = []
        names = [SOURCE.MAIN_PREFIX + f"atlas/Fixture{i}.scala" for i in range(69)]
        names += [SOURCE.GLUE_PREFIX + f"config/Fixture{i}.scala" for i in range(11)]
        names += sorted(SOURCE.CHIPYARD_SOURCES)
        for index, relative in enumerate(names):
            member = write(self.directory / "original" / relative, f"// selected Scala fixture {index}\n")
            sources.append({"relative_path": relative, "identity": member})
        tools = {role: write(self.directory / "original-tools" / role, role + " fixture\n")
                 for role in SOURCE.TOOLS}
        self.value = {"schema": "atlas.ee290_source_build_inputs.v0", "target_config": "EE290SimConfig",
                      "sources": sources, "tools": tools, "source_config": "chipyard:SelectedConfig",
                      "metadata": {"boundary": "remaining framework is an opaque selected jar"}}
        self.inputs = self.directory / "inputs.json"
        self.save()
        self.calls = []
        self.fallback = False
        self.fail_compile = False

    def tearDown(self):
        self.scratch.cleanup()

    def save(self):
        self.inputs.write_text(json.dumps(self.value, indent=2) + "\n")

    def capture(self, output, kind, argv, inputs, outputs, environment, timeout_seconds, cwd=None):
        self.calls.append((kind, list(argv), list(inputs), dict(environment)))
        output = Path(output)
        output.mkdir()
        if kind == "compile" and not self.fail_compile:
            with zipfile.ZipFile(output / "overlay.jar", "w") as jar:
                for name in required_classes():
                    jar.writestr(name.replace(".", "/") + ".class", b"synthetic class bytes")
        elif kind == "elaborate":
            write(output / (SOURCE.PUBLIC_NAME + ".fir"), "circuit EE290SimConfig\n")
            write(output / (SOURCE.PUBLIC_NAME + ".anno.json"), "[]\n")
            overlay = Path(argv[argv.index("-cp") + 1].split(":")[0])
            framework = Path(argv[argv.index("-cp") + 1].split(":")[1])
            lines = []
            for name in sorted(required_classes()):
                origin = framework if self.fallback and name == "atlas.lsu.LSU" else overlay
                lines.append(f"[0.01s][info][class,load] {name} source: {origin.as_uri()}")
            lines.append(f"[0.01s][info][class,load] chipyard.Generator source: {framework.as_uri()}")
            write(output / "class-load.log", "\n".join(lines) + "\n")
        receipt = {"schema": "atlas.ee290_captured_phase.v0", "kind": kind,
                   "state": "phase_failed" if self.fail_compile and kind == "compile" else "phase_completed",
                   "outputs": [{**row, "files": [{"relative_path": row["relative_path"],
                                                   "identity": identity(output / row["relative_path"])}]}
                               for row in outputs if (output / row["relative_path"]).is_file()]}
        write(output / "phase.json", json.dumps(receipt))
        return receipt

    def run_source(self, name="output", expected=None):
        with mock.patch.object(SOURCE.CAPTURE, "capture_phase", side_effect=self.capture):
            return SOURCE.elaborate(self.inputs, expected or identity(self.inputs)["sha256"], self.directory / name)

    def test_capture_recipe_snapshots_and_class_origins(self):
        with mock.patch.dict("os.environ", {"JAVA_TOOL_OPTIONS": "-javaagent:unselected.jar",
                                           "CLASSPATH": "unselected", "LD_LIBRARY_PATH": "unselected"}):
            report = self.run_source()
        self.assertEqual(report["state"], "source_elaboration_captured", report["failures"])
        self.assertFalse(report["scheduling_qualified"])
        self.assertFalse(report["full_chipyard_source_build"])
        self.assertFalse(report["simulator_build_linked"])
        self.assertEqual(len(report["sources"]), 83)
        self.assertEqual([row[0] for row in self.calls], ["compile", "elaborate"])
        compile_argv, compile_inputs, environment = self.calls[0][1:]
        self.assertIn("-Xplugin-require:chiselplugin", compile_argv)
        self.assertEqual(compile_argv[compile_argv.index("-d") + 1], "{output}/overlay.jar")
        self.assertEqual(sum(row["role"] == "source" for row in compile_inputs), 83)
        for injected in ("JAVA_TOOL_OPTIONS", "_JAVA_OPTIONS", "JDK_JAVA_OPTIONS", "CLASSPATH", "LD_LIBRARY_PATH"):
            self.assertNotIn(injected, environment)
        elab_argv = self.calls[1][1]
        self.assertTrue(elab_argv[elab_argv.index("-cp") + 1].split(":")[0].endswith("compile/overlay.jar"))
        self.assertEqual(elab_argv[elab_argv.index("--legacy-configs") + 1], "chipyard:SelectedConfig")
        self.assertTrue(report["class_origins"]["all_selected_loaded_classes_from_overlay"])
        self.assertEqual(set(report["class_origins"]["required_classes"]), required_classes())
        for source in report["sources"]:
            self.assertEqual(identity(Path(source["snapshot"]["path"])), source["snapshot"])
            self.assertEqual(source["snapshot"]["sha256"], source["identity"]["sha256"])
        self.assertEqual(identity(Path(report["firrtl"]["path"])), report["firrtl"])

    def test_input_mismatch_and_source_coverage_precede_output(self):
        with self.assertRaisesRegex(SOURCE.SourceError, "selected receipt SHA-256 mismatch"):
            self.run_source(expected="0" * 64)
        self.assertFalse((self.directory / "output").exists())
        self.value["sources"].pop()
        self.save()
        with self.assertRaisesRegex(SOURCE.SourceError, "69 Atlas main"):
            self.run_source()
        self.assertFalse(self.calls)
        self.assertFalse((self.directory / "output").exists())

    def test_restricted_source_paths_reject_without_inspection(self):
        self.value["sources"][0]["relative_path"] = "restricted/VLSI-source.scala"
        self.save()
        with self.assertRaisesRegex(SOURCE.SourceError, "allowed relative Scala"):
            self.run_source()
        self.assertFalse((self.directory / "output").exists())
        self.assertFalse(self.calls)

    def test_optional_helpers_are_pinned_and_exposed_on_owned_path(self):
        for role in SOURCE.HELPERS:
            self.value["tools"][role] = write(self.directory / (role + ".tool"), "synthetic executable")
        self.save()
        report = self.run_source()
        self.assertEqual(report["state"], "source_elaboration_captured", report["failures"])
        for role in SOURCE.HELPERS:
            saved = Path(report["tools"][role]["snapshot"]["path"])
            self.assertEqual(saved.name, role)
            self.assertTrue(saved.stat().st_mode & 0o100)
            self.assertEqual(saved.parent, self.directory / "output/helpers")
        self.assertEqual(self.calls[1][3]["HOME"], str(self.directory / "output/home"))
        self.assertEqual(self.calls[1][3]["PATH"], str(self.directory / "output/helpers") + ":/usr/bin:/bin")
        self.assertTrue(SOURCE.HELPERS <= {row["role"] for row in self.calls[1][2]})

    def test_compile_failure_preserves_honest_receipt(self):
        self.fail_compile = True
        report = self.run_source()
        self.assertEqual(report["state"], "source_elaboration_failed")
        self.assertEqual([row[0] for row in self.calls], ["compile"])
        self.assertIn("compile command failed", report["failures"][0])
        self.assertNotIn("source_to_firrtl", report)
        self.assertTrue((self.directory / "output/compile/phase.json").is_file())

    def test_framework_fallback_does_not_claim_source_link(self):
        self.fallback = True
        report = self.run_source()
        self.assertEqual(report["state"], "source_elaboration_failed")
        self.assertIn("instead of source overlay", report["failures"][0])
        self.assertNotIn("source_to_firrtl", report)

    def test_origin_audit_rejects_missing_and_unknown_atlas_classes(self):
        overlay = Path(write(self.directory / "overlay.jar", "overlay")["path"])
        framework = Path(write(self.directory / "framework.jar", "framework")["path"])
        lines = [f"[0.1s][info][class,load] {name} source: {overlay.as_uri()}" for name in required_classes()]
        lines.append(f"[0.1s][info][class,load] chipyard.Generator source: {framework.as_uri()}")
        missing = [line for line in lines if "atlas.scalar.ScalarCore " not in line]
        with self.assertRaisesRegex(SOURCE.SourceError, "missing required"):
            SOURCE.audit_origins("\n".join(missing), overlay, required_classes(), "chipyard:SelectedConfig", framework)
        lines.append(f"[0.1s][info][class,load] atlas.unselected.StaleClass source: {framework.as_uri()}")
        with self.assertRaisesRegex(SOURCE.SourceError, "instead of source overlay"):
            SOURCE.audit_origins("\n".join(lines), overlay, required_classes(), "chipyard:SelectedConfig", framework)

    def test_generated_lambda_requires_verified_overlay_owner(self):
        overlay = Path(write(self.directory / "overlay.jar", "overlay")["path"])
        framework = Path(write(self.directory / "framework.jar", "framework")["path"])
        lines = [f"[0.1s][info][class,load] {name} source: {overlay.as_uri()}" for name in required_classes()]
        lines.append(f"[0.1s][info][class,load] chipyard.Generator source: {framework.as_uri()}")
        hidden = "atlas.scalar.ScalarCore$$Lambda$7/0x00001234"
        lines.append(f"[0.1s][info][class,load] {hidden} source: atlas.scalar.ScalarCore")
        summary = SOURCE.audit_origins("\n".join(lines), overlay, required_classes(), "chipyard:SelectedConfig", framework)
        self.assertEqual(summary["loaded_overlay_generated_lambdas"], [hidden])
        with self.assertRaisesRegex(SOURCE.SourceError, "verified overlay owner"):
            SOURCE.audit_origins("\n".join(lines[:-1] + [lines[-1].replace("source: atlas.scalar.ScalarCore", "source: __JVM_LookupDefineClass__")]),
                                 overlay, required_classes(), "chipyard:SelectedConfig", framework)

    def test_unexpected_file_backed_framework_origin_rejects(self):
        overlay = Path(write(self.directory / "overlay.jar", "overlay")["path"])
        framework = Path(write(self.directory / "framework.jar", "framework")["path"])
        extra = Path(write(self.directory / "extra.jar", "unselected")["path"])
        lines = [f"[0.1s][info][class,load] {name} source: {overlay.as_uri()}" for name in required_classes()]
        lines.append(f"[0.1s][info][class,load] chipyard.Generator source: {framework.as_uri()}")
        lines.append(f"[0.1s][info][class,load] unselected.Injection source: {extra.as_uri()}")
        with self.assertRaisesRegex(SOURCE.SourceError, "unexpected file-backed"):
            SOURCE.audit_origins("\n".join(lines), overlay, required_classes(), "chipyard:SelectedConfig", framework)

    def test_original_source_mutation_rejects(self):
        initial_capture = self.capture

        def changed_capture(*args, **kwargs):
            result = initial_capture(*args, **kwargs)
            if args[1] == "elaborate":
                # Mutate an original selected source, independently of saved snapshots.
                original = Path(self.value["sources"][0]["identity"]["path"])
                original.write_text("changed after compile\n")
            return result

        with mock.patch.object(SOURCE.CAPTURE, "capture_phase", side_effect=changed_capture):
            report = SOURCE.elaborate(self.inputs, identity(self.inputs)["sha256"], self.directory / "changed")
        self.assertEqual(report["state"], "source_elaboration_failed")
        self.assertIn("artifact changed", report["failures"][0])
        self.assertNotIn("source_to_firrtl", report)

    def test_captured_firrtl_mutation_rejects(self):
        initial_capture = self.capture

        def changed_capture(*args, **kwargs):
            result = initial_capture(*args, **kwargs)
            if args[1] == "elaborate":
                (Path(args[0]) / (SOURCE.PUBLIC_NAME + ".fir")).write_text("changed after capture\n")
            return result

        with mock.patch.object(SOURCE.CAPTURE, "capture_phase", side_effect=changed_capture):
            report = SOURCE.elaborate(self.inputs, identity(self.inputs)["sha256"], self.directory / "changed")
        self.assertEqual(report["state"], "source_elaboration_failed")
        self.assertIn("artifact hash/size mismatch", report["failures"][0])
        self.assertNotIn("source_to_firrtl", report)


if __name__ == "__main__":
    unittest.main()
