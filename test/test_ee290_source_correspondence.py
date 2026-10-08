import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location("source_correspondence", ROOT / "tools/check-ee290-source-correspondence.py")
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)

class CorrespondenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=ROOT / "build/rtl-timing", prefix="correspondence-test-")
        self.directory = Path(self.tmp.name)
        self.original = [{"class": "firrtl.transforms.DontTouchAnnotation", "target": "~T|A>x"}]
        self.tail = [{"class": "sifive.enterprise.firrtl.MarkDUTAnnotation", "target": "~TestHarness|ChipTop"}]
        self.tail += [{"class": cls, "filename": str(self.directory / name)} for cls,name in M.HIERARCHY.items()]

    def tearDown(self): self.tmp.cleanup()

    def test_annotation_policy_requires_exact_prefix_and_order(self):
        result = M.annotation_policy(self.original, self.original + self.tail)
        self.assertEqual(result["chisel_annotation_count"], 1)
        for changed in [self.original + self.tail + [{}], self.original + list(reversed(self.tail)), self.tail + self.original,
                        [{"class": "modified"}] + self.tail]:
            with self.assertRaises(M.Error): M.annotation_policy(self.original, changed)

    def test_annotation_policy_rejects_semantic_change_and_external_paths(self):
        rows = json.loads(json.dumps(self.original + self.tail))
        rows[1]["target"] = "~TestHarness|OtherTop"
        with self.assertRaisesRegex(M.Error, "MarkDUT"): M.annotation_policy(self.original,rows)
        rows = json.loads(json.dumps(self.original + self.tail))
        rows[2]["filename"] = str(self.directory / "VLSI" / "model_module_hierarchy.json")
        with self.assertRaises(M.Error): M.annotation_policy(self.original,rows)
        with self.assertRaisesRegex(M.Error,"external blackbox"):
            M.audit_annotations([{"class":"firrtl.transforms.BlackBoxPathAnno","name":"external.v"}])
        with self.assertRaisesRegex(M.Error,"unreviewed annotation path"):
            M.audit_annotations([{"class":"unknown","nested":{"path":"external.v"}}])

    def test_embedded_annotations_cannot_hide_output_inputs(self):
        fir = self.directory / "input.fir"
        fir.write_text("FIRRTL version 3.3.0\ncircuit T : %[" + json.dumps(self.tail[1:]) + "]\n")
        with self.assertRaisesRegex(M.Error,"embedded hierarchy"): M.audit_firrtl(fir)
        fir.write_text("circuit T : %[[]]\n")
        M.audit_firrtl(fir)

    def hardware(self, change=False, extra=False):
        lines = ["module {\n", "  hw.module @AtlasCore() {\n"]
        lines += ['    hw.instance "u%d" @M%d() -> () loc(#loc1)\n' % (n,n) for n in range(122)]
        if extra: lines += ['    hw.instance "extra" @Extra() -> ()\n']
        lines += ["    hw.output\n", "  }\n"]
        for n in range(122):
            lines += ["  hw.module @M%d() {\n" % n, "    %%c = hw.constant %d : i1 loc(#loc1)\n" % (1 if change and n == 17 else 0), "    hw.output\n", "  }\n"]
        if extra: lines += ["  hw.module @Extra() {\n", "    hw.output\n", "  }\n"]
        return "".join(lines) + "}\n#loc1 = loc(\"selected.scala\":1:1)\n"

    def test_full_closure_ignores_only_location_aliases(self):
        selected = self.hardware()
        fresh = selected.replace("loc(#loc1)","loc(#loc79)").replace("#loc1 =", "#loc79 =").replace("selected.scala", "snapshot.scala")
        result = M.compare_closures(fresh, selected)
        self.assertEqual(result["module_count"],123)
        self.assertTrue(result["all_modules_match"])
        with self.assertRaisesRegex(M.Error,"M17"): M.compare_closures(self.hardware(change=True), selected)
        with self.assertRaisesRegex(M.Error,"Extra"): M.compare_closures(self.hardware(extra=True), selected)
        short = "module {\n  hw.module @AtlasCore() {\n    hw.output\n  }\n}\n"
        with self.assertRaisesRegex(M.Error,"123 reviewed"): M.compare_closures(short, short)

    def source_fixture(self):
        spec=importlib.util.spec_from_file_location("correspondence_source_fixture",ROOT/"test/test_ee290_source_elaboration.py")
        fixture_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture_module)
        fixture=fixture_module.SourceElaborationTest();fixture.setUp();self.addCleanup(fixture.tearDown)
        capture=fixture.capture
        def complete_capture(output,kind,argv,inputs,outputs,environment,timeout_seconds,cwd=None):
            result=capture(output,kind,argv,inputs,outputs,environment,timeout_seconds,cwd)
            output=Path(output)
            checker=M.CHECK.Checker()
            producer=checker.identity(M.CAPTURE_PATH);dependency=checker.identity(M.CAPTURE._DEPENDENCY)
            snapshots={}
            for key,identity,name in [("producer",producer,"ee290_build_capture.executed.py"),
                                      ("path_checker_dependency",dependency,"check-ee290-provenance.dependency.py")]:
                path=output/name;path.write_bytes(Path(identity["path"]).read_bytes());snapshots[key]=checker.identity(path)
            result.update({"target_config":"EE290SimConfig","declared_outputs":outputs,"scheduling_qualified":False,"inputs_stable":True,"inputs_before":inputs,"inputs_after":inputs,
                           "failures":[],"producer":producer,"path_checker_dependency":dependency,"snapshots":snapshots,
                           "command":{"argv":[arg.replace("{output}",str(output)) for arg in argv],"cwd":str(output),
                                      "environment":environment,"returncode":0,"timed_out":False,"timeout_seconds":timeout_seconds,"elapsed_seconds":0.1,
                                      "stdout":fixture_module.write(output/"stdout.log",""),"stderr":fixture_module.write(output/"stderr.log","")}})
            (output/"phase.json").write_text(json.dumps(result))
            return result
        with mock.patch.object(fixture_module.SOURCE.CAPTURE,"capture_phase",side_effect=complete_capture):
            report=fixture_module.SOURCE.elaborate(fixture.inputs,fixture_module.identity(fixture.inputs)["sha256"],fixture.directory/"output")
        self.assertEqual(report["state"],"source_elaboration_captured",report["failures"])
        return fixture,report,fixture.directory/"output/report.json"

    def test_saved_producer_snapshots_survive_changed_live_source(self):
        fixture,report,path=self.source_fixture()
        changed=fixture.directory/"changed-producer.py";changed.write_text("changed live implementation")
        report["producer"]["path"]=str(changed)
        report["producer_dependencies"]["capture"]["path"]=str(changed)
        path.write_text(json.dumps(report))
        checker=M.CHECK.Checker();selected_path,_,selected=M.select_source_report(checker,path,checker.identity(path)["sha256"])
        result=M.validate_source_receipt(checker,selected_path,selected)
        self.assertEqual(result["status"],"bounded_source_recipe_verified")
        self.assertEqual(result["phase_receipts"]["compile"]["path"],str(path.parent/"compile/phase.json"))

    def test_source_command_mutation_rejects_even_with_rehashed_receipt(self):
        fixture,report,path=self.source_fixture()
        phase_path=path.parent/"elaborate/phase.json";phase=json.loads(phase_path.read_text())
        phase["command"]["argv"] += ["--extra-unreviewed-option"]
        phase_path.write_text(json.dumps(phase))
        report["phases"][1]["receipt"]=M.CHECK.Checker().identity(phase_path)
        with self.assertRaisesRegex(M.Error,"command recipe"):
            M.validate_source_receipt(M.CHECK.Checker(),path,report)

if __name__ == "__main__": unittest.main()
