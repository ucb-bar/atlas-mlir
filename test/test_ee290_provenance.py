"""Small synthetic receipt tests; no compiler, simulator or build is executed."""

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).absolute().parents[1]
SPEC = importlib.util.spec_from_file_location("ee290_provenance", ROOT / "tools/check-ee290-provenance.py")
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)


def identity(path):
    data = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data.encode() if isinstance(data, str) else data)
    return identity(path)


def save(path, value):
    write(path, json.dumps(value, indent=2) + "\n")


class Fixture:
    def __init__(self, directory):
        self.directory = directory
        self.tools = {name: write(directory / "tools" / name, name + " fixture\n")
                      for name in ("firtool", "circt-opt", "atlas-emit", "host-cc", "simulator")}
        self.firrtl = write(directory / "sources/input.fir", "circuit EE290SimConfig\n")
        sidecar = [{"class": "sifive.enterprise.firrtl.ModuleHierarchyAnnotation",
                    "filename": "original-hierarchy.json"}, {"class": "example.annotation", "value": 7}]
        self.annotations = write(directory / "sources/original.anno.json", json.dumps(sidecar))
        self.options = write(directory / "sources/lowering-options.txt", "example-option\n")
        self.retained = directory / "retained"
        snapshots = {}
        for name, member in (("firrtl", self.firrtl), ("annotations", self.annotations),
                             ("lowering_options", self.options)):
            snapshots[name] = write(self.retained / "inputs" / name, Path(member["path"]).read_bytes())
        after = str(self.retained / "top_module_hierarchy.json")
        prepared = copy.deepcopy(sidecar)
        prepared[0]["filename"] = after
        self.hw = write(self.retained / "ee290.hw.mlir", "module { // synthetic selected HW\n}\n")
        self.manifest = {"schema": "atlas.retained_hw_ir.v0", "target_config": "EE290SimConfig",
                         "state": "verified", "original_inputs_unchanged": True,
                         "producer": write(self.retained / "retention-producer.py", "recorded producer\n"),
                         "inputs": {"firrtl": self.firrtl, "annotations": self.annotations,
                                    "lowering_options": self.options}, "snapshots": snapshots,
                         "prepared_annotations": write(self.retained / "inputs/prepared.anno.json", json.dumps(prepared)),
                         "annotation_redirects": [{"class": sidecar[0]["class"],
                                                   "before": sidecar[0]["filename"], "after": after}],
                         "hardware_ir": self.hw,
                         "tools": {"firtool": self.tools["firtool"], "circt_opt": self.tools["circt-opt"]},
                         "commands": []}
        self.manifest["commands"] = [
            {"stage": "lower", "returncode": 0, "cwd": str(self.retained),
             "argv": [self.tools["firtool"]["path"], "--format=fir", "--ir-hw", "--verify-each=true",
                      "--annotation-file=" + self.manifest["prepared_annotations"]["path"],
                      "--lowering-options=example-option", "-o", self.hw["path"], snapshots["firrtl"]["path"]],
             "log": write(self.retained / "lower.log", "lowered\n")},
            {"stage": "verify", "returncode": 0, "cwd": str(self.retained),
             "argv": [self.tools["circt-opt"]["path"], self.hw["path"], "--verify-each", "-o", "/dev/null"],
             "log": write(self.retained / "verify.log", "verified\n")}]
        self.manifest_path = self.retained / "manifest.json"
        save(self.manifest_path, self.manifest)
        self.sv = write(directory / "sources/design.sv", "module EE290SimConfig; endmodule\n")
        self.filelist = write(directory / "sources/sources.f", self.sv["path"] + "\n")
        self.archive = write(directory / "simulator-archives/model.so", "synthetic archive\n")
        self.host = write(directory / "sources/host.c", "int main(void) { return 0; }\n")
        self.build_record = write(directory / "metadata/vcs_rebuild", "saved command text; not a build receipt\n")
        self.reports = []
        self.paths = []
        for number in range(2):
            base = directory / f"arm-{number}"
            original_program = write(directory / "sources" / f"program-{number}.mlir", f"module {{ // arm {number}\n}}\n")
            write(base / "program.mlir", Path(original_program["path"]).read_bytes())
            write(base / "ee290-vls-host.executed.c", Path(self.host["path"]).read_bytes())
            write(base / "run-ee290-vls-witness.executed.py", "recorded witness runner\n")
            emitted = write(base / "emit.log", "00000013\n00000073\n")
            include = write(base / "atlas_program.inc", "#define ATLAS_PROGRAM_WORDS 2U\nstatic const uint32_t atlas_program[] = {\n  0x00000013U,\n  0x00000073U,\n};\n")
            binary = write(base / "host.riscv", f"synthetic host binary {number}\n")
            observations = []
            for i in range(3):
                observations.extend([f"EE290_VLS_OBSERVED panel={i} status=5 marker=1 illegal_pc=0 observer_cycles=100 retired=6 polls=10",
                                     f"EE290_VLS_PANEL_PASSED panel={i} output_words=256 preserved_words=1280"])
            observations.append("EE290_VLS_PASSED panels=3")
            simulation = write(base / "simulation.log", "\n".join(observations) + "\n")
            commands = [
                {"argv": [self.tools["atlas-emit"]["path"], str(base / "program.mlir")],
                 "cwd": str(base), "returncode": 0, "timed_out": False, "log": emitted},
                {"argv": [self.tools["host-cc"]["path"], str(base / "ee290-vls-host.executed.c"), "-o", binary["path"]],
                 "cwd": str(base), "returncode": 0, "timed_out": False, "log": write(base / "host-build.log", "compiled\n")},
                {"argv": [self.tools["simulator"]["path"], "+permissive", "+loadmem=" + binary["path"],
                          "+max-cycles=2000000", "+permissive-off", binary["path"]],
                 "cwd": str(base), "returncode": 0, "timed_out": False, "log": simulation}]
            report = {"schema": "atlas.ee290_vls_witness.v0", "target_config": "EE290SimConfig",
                      "state": "integrated_execution_passed", "integrated_execution_passed": True,
                      "scheduling_qualified": False, "program_word_count": 2,
                      "generated_include": include, "host_binary": binary, "commands": commands,
                      "observations": observations,
                      "provenance": {"manifest": identity(self.manifest_path), "hardware_ir": self.hw,
                                     "inputs": self.manifest["inputs"], "program": original_program,
                                     "host_source": self.host, "atlas_emit": self.tools["atlas-emit"],
                                     "host_cc": self.tools["host-cc"], "simulator": self.tools["simulator"],
                                     "simulator_sources": [self.filelist, self.sv],
                                     "simulator_archive_libraries": [self.archive],
                                     "uninterpreted_source_list_options": [],
                                     "observed_simulator_build_record": self.build_record,
                                     "observed_build_record_direct_sources": [],
                                     "build_record_relationship": "observed command metadata; not a certified build receipt"}}
            path = base / "report.json"
            save(path, report)
            self.reports.append(report)
            self.paths.append(path)
        self.source = write(directory / "sources/Atlas.scala", "class EE290SimConfig\n")
        self.inventory = {"schema": "atlas.ee290_source_inventory.v0", "target_config": "EE290SimConfig",
                          "sources": [{**self.source, "role": "current source observation"}],
                          "metadata": {"description": "current observations, not original build input closure",
                                       "saved_build_record": self.build_record}}
        self.inventory_path = directory / "inventory.json"
        save(self.inventory_path, self.inventory)

    def update_report(self, number):
        save(self.paths[number], self.reports[number])

    def update_manifest(self):
        save(self.manifest_path, self.manifest)
        for number, report in enumerate(self.reports):
            report["provenance"]["manifest"] = identity(self.manifest_path)
            self.update_report(number)

    def run(self, name="checked", *, inventory=True, wrong_hash=False):
        witnesses = [(path, "0" * 64 if wrong_hash and i == 0 else identity(path)["sha256"])
                     for i, path in enumerate(self.paths)]
        return CHECK.check_provenance(self.manifest_path, identity(self.manifest_path)["sha256"],
                                      witnesses, self.directory / name,
                                      self.inventory_path if inventory else None,
                                      identity(self.inventory_path)["sha256"] if inventory else None)


class ProvenanceTest(unittest.TestCase):
    def setUp(self):
        scratch = CHECK.allowed(ROOT / "build/rtl-timing", missing=True)
        scratch.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="ee290-provenance-test-", dir=scratch)
        self.directory = Path(self.temporary.name)
        self.fixture = Fixture(self.directory)

    def tearDown(self):
        self.temporary.cleanup()

    def assert_rejected(self, pattern=None, **kwargs):
        with self.assertRaisesRegex(CHECK.ProvenanceError, pattern or ".+"):
            self.fixture.run(**kwargs)
        self.assertFalse((self.directory / kwargs.get("name", "checked")).exists())

    def test_shared_receipt_links_and_snapshots(self):
        receipt = self.fixture.run()
        self.assertEqual(receipt["state"], "recorded_links_verified")
        self.assertFalse(receipt["scheduling_qualified"])
        self.assertFalse(receipt["build_linkage_complete"])
        self.assertEqual(receipt["edges"]["firrtl_to_retained_hw"]["status"], "verified_recorded_execution")
        self.assertEqual(receipt["edges"]["simulator_numerical_execution"]["status"], "verified_recorded_execution")
        self.assertEqual(len(receipt["arms"]), 2)
        for member in receipt["snapshots"]["witness_reports"]:
            self.assertEqual(identity(Path(member["path"])), member)
        self.assertEqual((self.directory / "checked/retention-manifest.json").read_bytes(), self.fixture.manifest_path.read_bytes())

    def test_wrong_report_hash_and_missing_expected_inventory_hash(self):
        self.assert_rejected("selected receipt SHA-256 mismatch", wrong_hash=True)
        with self.assertRaisesRegex(CHECK.ProvenanceError, "supplied together"):
            CHECK.check_provenance(self.fixture.manifest_path, identity(self.fixture.manifest_path)["sha256"],
                                   [(self.fixture.paths[0], identity(self.fixture.paths[0])["sha256"])],
                                   self.directory / "missing-hash", self.fixture.inventory_path)
        self.assertFalse((self.directory / "missing-hash").exists())

    def test_changed_artifact_identities_reject(self):
        for field in ("firrtl", "hw", "source", "archive"):
            with self.subTest(field=field):
                member = getattr(self.fixture, field)
                path = Path(member["path"])
                original = path.read_bytes()
                path.write_bytes(original + b"changed\n")
                self.assert_rejected("hash/size mismatch")
                path.write_bytes(original)
        simulator = Path(self.fixture.tools["simulator"]["path"])
        simulator.write_bytes(b"changed executable\n")
        self.assert_rejected("hash/size mismatch")

    def test_firrtl_and_hw_crosslinks_cannot_diverge(self):
        for field in ("firrtl", "hardware_ir"):
            with self.subTest(field=field):
                report = self.fixture.reports[1]
                provenance = report["provenance"]
                replacement = write(self.directory / f"alternate-{field}", "different selected bytes\n")
                if field == "firrtl":
                    original = provenance["inputs"][field]
                    provenance["inputs"] = dict(provenance["inputs"])
                    provenance["inputs"][field] = replacement
                else:
                    original = provenance[field]
                    provenance[field] = replacement
                self.fixture.update_report(1)
                self.assert_rejected("crosslink mismatch")
                if field == "firrtl":
                    provenance["inputs"][field] = original
                else:
                    provenance[field] = original
                self.fixture.update_report(1)

    def test_different_simulator_source_and_archive_sets_reject(self):
        report = self.fixture.reports[1]
        provenance = report["provenance"]
        for field in ("simulator_sources", "simulator_archive_libraries"):
            with self.subTest(field=field):
                original = provenance[field]
                provenance[field] = [write(self.directory / (field + ".alternate"), "alternate observed artifact\n")]
                self.fixture.update_report(1)
                self.assert_rejected("divergent witness arms")
                provenance[field] = original
                self.fixture.update_report(1)
        provenance["simulator"] = write(self.directory / "alternate-simulator", "different simulator\n")
        report["commands"][2]["argv"][0] = provenance["simulator"]["path"]
        self.fixture.update_report(1)
        self.assert_rejected("divergent witness arms")

    def test_malformed_identity_and_unsupported_build_receipt(self):
        for member in ({"path": str(self.directory / "missing")},
                       {"sha256": "0" * 64}, {**self.fixture.source, "bytes": True}):
            self.fixture.inventory["metadata"]["supplementary"] = member
            save(self.fixture.inventory_path, self.fixture.inventory)
            self.assert_rejected("malformed artifact identity")
        del self.fixture.inventory["metadata"]["supplementary"]
        self.fixture.inventory["metadata"]["build_receipts"] = []
        save(self.fixture.inventory_path, self.fixture.inventory)
        self.assert_rejected("build receipts are unsupported")

    def test_saved_build_text_and_current_inventory_never_certify(self):
        text = Path(self.fixture.build_record["path"])
        text.write_text("build_linkage_complete=true scheduling_qualified=true\n")
        member = identity(text)
        for number, report in enumerate(self.fixture.reports):
            report["provenance"]["observed_simulator_build_record"] = member
            self.fixture.update_report(number)
        self.fixture.inventory["metadata"].update(saved_build_record=member,
                                                 claims={"build_linkage_complete": True, "scheduling_qualified": True})
        save(self.fixture.inventory_path, self.fixture.inventory)
        receipt = self.fixture.run()
        self.assertFalse(receipt["build_linkage_complete"])
        self.assertFalse(receipt["scheduling_qualified"])
        self.assertEqual(receipt["edges"]["source_to_firrtl"]["status"], "unestablished")
        self.assertEqual(receipt["edges"]["source_or_firrtl_to_simulator_build"]["status"], "unestablished")

    def test_unrelated_annotations_and_lowering_options_reject(self):
        lower = self.fixture.manifest["commands"][0]
        original = list(lower["argv"])
        other = write(self.directory / "different-prepared.anno.json", "[]")
        lower["argv"] = ["--annotation-file=" + other["path"] if arg.startswith("--annotation-file=") else arg
                         for arg in original]
        self.fixture.update_manifest()
        self.assert_rejected("selected prepared annotations")
        lower["argv"] = ["--lowering-options=different" if arg.startswith("--lowering-options=") else arg
                         for arg in original]
        self.fixture.update_manifest()
        self.assert_rejected("options differ from selected snapshot")
        lower["argv"] = original
        prepared = Path(self.fixture.manifest["prepared_annotations"]["path"])
        value = json.loads(prepared.read_text())
        value[1]["value"] = 99
        prepared.write_text(json.dumps(value))
        self.fixture.manifest["prepared_annotations"] = identity(prepared)
        self.fixture.update_manifest()
        self.assert_rejected("prepared annotation transformation")

    def test_executed_snapshots_and_include_location_reject(self):
        program = self.directory / "arm-0/program.mlir"
        prior = program.read_bytes()
        program.write_bytes(b"different executed program\n")
        self.assert_rejected("executed program snapshot")
        program.write_bytes(prior)
        report = self.fixture.reports[0]
        current = Path(report["generated_include"]["path"])
        report["generated_include"] = write(self.directory / "other/atlas_program.inc", current.read_bytes())
        self.fixture.update_report(0)
        self.assert_rejected("not the executed host include")

    def test_false_success_and_short_command_reject(self):
        report = self.fixture.reports[0]
        report["commands"][1]["argv"] = [self.fixture.tools["host-cc"]["path"]]
        self.fixture.update_report(0)
        self.assert_rejected("unsupported host compiler output")

    def test_self_consistent_false_terminal_observations_reject(self):
        report = self.fixture.reports[0]
        path = Path(report["commands"][2]["log"]["path"])
        original = path.read_text()
        for before, after in (("status=5", "status=7"), ("marker=1", "marker=0"),
                              ("illegal_pc=0", "illegal_pc=4"),
                              ("polls=10", "polls=100000"), ("panel=2 status=5", "panel=1 status=5"),
                              ("retired=6", "retired=4294967296"),
                              ("EE290_VLS_PASSED panels=3", "EE290_VLS_PASSED panels=3 suffix")):
            with self.subTest(after=after):
                changed = original.replace(before, after)
                path.write_text(changed)
                report["commands"][2]["log"] = identity(path)
                report["observations"] = [line for line in changed.splitlines() if line.startswith("EE290_VLS_")]
                self.fixture.update_report(0)
                self.assert_rejected("observations|success")

    def test_restricted_paths_symlink_targets_and_loops_before_access(self):
        original_lstat = Path.lstat

        def guarded_lstat(path, *args, **kwargs):
            self.assertFalse(CHECK.restricted(path), "restricted path was inspected")
            return original_lstat(path, *args, **kwargs)

        blocked = self.directory / "hAmMeR-unread" / "input"
        symlink = self.directory / "permitted-link"
        os.symlink("VLsI-unread/input", symlink)
        a, b = self.directory / "cycle-a", self.directory / "cycle-b"
        os.symlink(b.name, a)
        os.symlink(a.name, b)
        with mock.patch.object(Path, "lstat", guarded_lstat), \
                mock.patch.object(Path, "resolve", side_effect=AssertionError("unchecked canonical traversal")):
            for path in (blocked, blocked.parent / ".." / "input", symlink):
                with self.assertRaisesRegex(CHECK.ProvenanceError, "restricted"):
                    CHECK.allowed(path)
            with self.assertRaisesRegex(CHECK.ProvenanceError, "recursive artifact symlink"):
                CHECK.allowed(a)

    def test_duplicate_json_key_and_new_output_required(self):
        data = self.fixture.paths[0].read_text()
        self.fixture.paths[0].write_text(data.replace('"scheduling_qualified": false',
                                                     '"scheduling_qualified": false, "scheduling_qualified": false'))
        self.assert_rejected("duplicate JSON key")
        self.fixture.update_report(0)
        self.fixture.run()
        with self.assertRaisesRegex(CHECK.ProvenanceError, "output already exists"):
            self.fixture.run()


if __name__ == "__main__":
    unittest.main()
