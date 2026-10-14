"""Regeneration cache, comparison, adapters and artifact inventory."""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'tools'))
from rtl_extract import pairs, regen_cache as cache, regen_model as model, regen_sweeps as sweep
from rtl_extract.regen_cache import StageRunner, safe_path, sha256, stage_key
from rtl_extract.regen_check import compare_artifacts
import test_rtl_extract_pairs as synthetic
import test_rtl_extract_synthetic as circuit
spec = importlib.util.spec_from_file_location('regenerate', REPO / 'tools/regenerate-rtl-timing.py')
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)

class CacheTest(unittest.TestCase):
    def test_content_key(self):
        self.assertEqual(stage_key([], [{'sha256': 'x'}], {'python': '3'}),
                         stage_key([], [{'sha256': 'x'}], {'python': '3'}))
        for inputs, versions, commands in [([{'sha256': 'y'}], {'python': '3'}, []),
                                           ([{'sha256': 'x'}], {'python': '4'}, []),
                                           ([{'sha256': 'x'}], {'python': '3'}, [['new']])]:
            self.assertNotEqual(stage_key([], [{'sha256': 'x'}], {'python': '3'}),
                                stage_key(commands, inputs, versions))

    def test_skip_changed_input_and_corrupt_output(self):
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        source, output = root / 'input', root / 'output'
        source.write_text('first')
        command = [sys.executable, '-c',
                   'from pathlib import Path; import sys; Path(sys.argv[2]).write_bytes(Path(sys.argv[1]).read_bytes())',
                   str(source), str(output)]
        def run():
            return StageRunner(root, root).run('copy', [command], [source], [output], {'python': sys.version})
        self.assertEqual(run()['status'], 'produced')
        self.assertEqual(run()['status'], 'skipped')
        source.write_text('second')
        self.assertEqual(run()['status'], 'produced')
        self.assertEqual(output.read_text(), 'second')
        for mutate in (lambda: output.write_text('corrupt'), output.unlink):
            mutate()
            self.assertEqual(run()['status'], 'produced')
        state = json.loads((root / 'state.json').read_text())['copy']
        self.assertEqual(state['outputs'], [{'path': 'output', 'sha256': sha256(output)}])
        self.assertEqual(state['inputs'][0]['path'], 'input')

    def test_failure_does_not_cache_success(self):
        import subprocess
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        runner = StageRunner(root, root)
        with self.assertRaises(subprocess.CalledProcessError):
            runner.run('broken', [[sys.executable, '-c', 'raise SystemExit(2)']], [],
                       [root / 'missing'], {})
        self.assertEqual(runner.records[0]['status'], 'failed')
        self.assertNotIn('broken', runner.state)

    def test_inventory_outputs_are_validated(self):
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        script = root / 'produce.py'
        script.write_text('from pathlib import Path\nimport json\n'
            'Path("trace").write_text("valid")\n'
            'Path("summary.json").write_text(json.dumps({"outputs":[{"path":"trace"}]}))\n')
        def run():
            return StageRunner(root, root).run('replay', [[sys.executable, script]], [script],
                [root / 'summary.json'], {}, output_inventory=root / 'summary.json')
        self.assertEqual(run()['status'], 'produced')
        self.assertEqual(run()['status'], 'skipped')
        for mutate in ((root / 'trace').unlink, lambda: (root / 'trace').write_text('corrupt'),
                       lambda: (root / 'summary.json').write_text('invalid json')):
            mutate()
            self.assertEqual(run()['status'], 'produced')
    def test_restricted_symlink_is_rejected_before_target_inspection(self):
        with patch.object(Path, 'is_symlink', side_effect=lambda p: p.name == 'alias', autospec=True) as inspect, \
             patch('os.readlink', return_value='/tmp/restricted-' + 'vl' + 'si/child'):
            with self.assertRaisesRegex(ValueError, 'restricted symlink'):
                safe_path('/tmp/alias/file')
        self.assertEqual([call.args[0] for call in inspect.call_args_list], [Path('/tmp'), Path('/tmp/alias')])

    def test_receipt_log_and_temporary_paths_are_checked(self):
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        for name in ('state.json', 'receipt.json.tmp', 'copy.log'):
            runner = StageRunner(root, root)
            def checked(path):
                if Path(path).name == name:
                    raise ValueError('restricted child')
                return safe_path(path)
            with self.subTest(name=name), patch.object(cache, 'safe_path', side_effect=checked):
                with self.assertRaisesRegex(ValueError, 'restricted child'):
                    if name == 'state.json': StageRunner(root, root)
                    elif name.endswith('.tmp'): cache.write_json(root / 'receipt.json', {})
                    else: runner.run('copy', [], [], [root / 'output'], {})

    def test_output_symlink_cannot_escape_build_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'build'; root.mkdir()
            outside = Path(tmp) / 'outside'; outside.mkdir()
            (root / 'escape').symlink_to(outside, target_is_directory=True)
            runner = StageRunner(root, root)
            with self.assertRaisesRegex(ValueError, 'escapes build root'):
                runner.run('unsafe', [], [], [root / 'escape' / 'output'], {})

class RegenCheckTests(unittest.TestCase):
    n = ["compiler-facts.json", "pairs-vpu.json", "sequences-vpu.json", "op-pairs.json",
             "pairs-core_mxu_same.json", "pairs-core_mxu_shared.json", "sequences-core_mxu.json"]
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.g, self.r = [Path(temporary.name) / side for side in ("generated", "reference")]
        for root in (self.g, self.r): self.populate(root)
    def write(self, root, name, value):
        path = root / name; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value)); return hashlib.sha256(path.read_bytes()).hexdigest()
    def read(self, root, name): return json.loads((root / name).read_text())
    def mutate(self, name, change):
        value = self.read(self.g, name); change(value); return self.write(self.g, name, value)
    def hardware_bytes(self, root):
        return ("module {\n" + "\n".join(
            f'firrtl.moduleHierarchyFile = [#hw.output_file<"{root}/{name}_module_hierarchy.json", '
            'excludeFromFileList>]'
            for name in ("model", "top")) + "\nwire = 64\n}\n").encode()
    def populate(self, root, raw=None, recorded_path=None):
        root.mkdir(exist_ok=True); raw = self.hardware_bytes(root) if raw is None else raw
        (root / "debug.mlir").write_bytes(raw)
        hw = {"path": str(recorded_path or root / "debug.mlir"), "sha256": hashlib.sha256(raw).hexdigest()}
        timing = [{"name": name, "events": {"write": {"last_age": 64}}}
                  for name in ("vpu.add", "vpu.relu", "mxu0.push_weight", "mxu1.push_weight")]
        digest = self.write(root, self.n[0], {"schema": "merlin.op_timing.v1", "hw_ir": hw, "op_timing": timing})
        record = {"name": "vpu.add_relu", "gaps": [1, 3], "min_gap": 2, "stable_gap": 2,
                  "results": [{"from": 1, "to": 1, "result": "busy"}, {"from": 2, "to": 3, "result": "accepted"}],
                  "rejected": [{"gap": 1, "invariants": [{"path": "/core", "age": 1}]}],
                  "ops": [{"op": "pw", "inputs": {"instruction": 1}}]}; sources = []
        for index in (1, 2, 4, 5, 6):
            name = self.n[index]; key = "op_sequences" if name.startswith("sequences") else "op_pairs"
            schema = f"rtl_extract.{key}.v1"
            sha = self.write(root, name, {"schema": schema, "hw_ir": hw, key: [copy.deepcopy(record)]})
            if index != 6: sources.append({"path": str(root / name), "sha256": sha, "schema": schema})
        pairs = [{"first": f"{e}.push_weight", "second": f"{e}.push_weight",
              "relation": {"weight_slot": "same"}, "min_gap": 2, "stable_gap": 2} for e in ("vpu", "mxu0", "mxu1")]
        capacity = [{"engine": e, "live": 2, "window": "last_event_age"} for e in ("vpu", "mxu0", "mxu1")]
        self.write(root, self.n[3], {"schema": "merlin.op_pairs.v1", "hw_ir": hw,
            "op_timing": {"path": str(root / self.n[0]), "sha256": digest},
            "sources": sources, "op_pairs": pairs, "capacity": capacity})
    def compare(self, n=None, **kw): return compare_artifacts(self.g, self.r, self.n if n is None else n, **kw)
    def differ(self, name, change, *fields):
        self.populate(self.g); self.mutate(name, change)
        differences = self.compare([name])
        for field in fields: self.assertTrue(any(field in d for d in differences), (field, differences))
    def test_relocation_order_and_compression(self):
        for name, key in ((self.n[0], "hw_ir"), (self.n[3], "op_timing")):
            self.assertNotEqual(self.read(self.g, name)[key]["sha256"], self.read(self.r, name)[key]["sha256"])
        self.assertEqual(self.compare(), [])
        self.mutate(self.n[0], lambda d: d["op_timing"].reverse())
        self.mutate(self.n[1], lambda d: d["op_pairs"][0].update(results=[
            {"from": gap, "to": gap, "result": "busy" if gap == 1 else "accepted"} for gap in (1, 2, 3)]))
        self.assertEqual(self.compare(self.n[:2]), [])
    def test_manifest_and_legacy_paths(self):
        mapping = {f"evidence/{name}": f"system/{name}" for name in self.n}
        for root, directory in ((self.g, "evidence"), (self.r, "system")):
            (root / directory).mkdir()
            for name in self.n: (root / name).rename(root / directory / name)
        self.assertEqual(self.compare(mapping), [])
        differences = self.compare({f"evidence/{self.n[0]}": f"archive/{self.n[0]}"})
        self.assertTrue(any("missing artifact archive/" in d for d in differences))
        for name in self.n: (self.g / "evidence" / name).rename(self.g / name)
        self.assertEqual(self.compare(), [])
        for name in self.n: (self.r / "system" / name).rename(self.r / name)
        self.assertEqual(self.compare({name: f"system/{name}" for name in self.n}), [])
        differences = self.compare({self.n[0]: f"archive/{self.n[0]}"})
        self.assertTrue(any("missing artifact archive/" in d for d in differences))
        for name in self.n: (self.r / name).rename(self.r / "system" / name)
        (self.r / "stale-unchecked").mkdir()
        (self.r / "system" / self.n[1]).rename(self.r / "stale-unchecked" / self.n[1])
        self.assertTrue(any("missing artifact pairs-vpu.json" in d for d in self.compare()))
    def test_hardware_pin_and_override(self):
        for root in (self.g, self.r): self.populate(root, recorded_path=root / "unavailable.mlir")
        self.assertTrue(self.compare())
        options = dict(generated_hw_ir=self.g / "debug.mlir", reference_hw_ir=self.r / "debug.mlir")
        self.assertEqual(self.compare(**options), [])
        (self.g / "wrong.mlir").write_bytes(b"different hardware")
        wrong = {**options, "generated_hw_ir": self.g / "wrong.mlir"}
        self.assertTrue(any("does not match source" in d for d in self.compare([self.n[0]], **wrong)))
        for root in (self.g, self.r):
            doc = self.read(root, self.n[0]); doc["hw_ir"]["sha256"] = "b" * 64
            self.write(root, self.n[0], doc)
        differences = self.compare([self.n[0]], **options)
        self.assertEqual(len(differences), 2)
        self.assertTrue(all("recorded hw_ir sha256" in d and "does not match source" in d for d in differences))
    def test_hardware_normalization_is_narrow(self):
        for extra in (b'other.path = "%s/model_module_hierarchy.json"\n',
                      b'firrtl.moduleHierarchyFile = [#hw.output_file<"%s/other.json", excludeFromFileList>]\n'):
            with self.subTest(extra=extra):
                for root in (self.g, self.r): self.populate(root, self.hardware_bytes(root) + extra%str(root).encode())
                self.assertTrue(any("hw_ir.sha256" in d for d in self.compare([self.n[0]])))
        self.populate(self.r)
        self.populate(self.g, self.hardware_bytes(self.g).replace(b"wire = 64", b"wire = 63"))
        self.assertEqual(len(self.compare([self.n[0]])), 1)
        (self.g / "debug.mlir").unlink()
        self.assertTrue(any("debug.mlir" in d for d in self.compare([self.n[0]])))
    def test_all_record_fields_and_metadata_are_substantive(self):
        cases = [(0, lambda d: d["op_timing"][0]["events"]["write"].update(last_age=63), ("events.write.last_age",)),
        (1, lambda d: d["op_pairs"][0].update(min_gap=1, stable_gap=3), ("min_gap", "stable_gap")),
        (1, lambda d: d["op_pairs"][0]["results"][1].update(result="failed"), ("results.2.result", "results.3.result")),
        (1, lambda d: d["op_pairs"][0]["rejected"][0]["invariants"][0].update(path="/wrong"), ("invariants[0].path",)),
        (3, lambda d: d["op_pairs"][1]["relation"].update(weight_slot="distinct"), ("weight_slot",)),
        (3, lambda d: d["op_pairs"][0].update(stable_gap=3), ("stable_gap",)),
        (3, lambda d: d["capacity"][1].update(live=1, window="next_issue_age"),
         ("capacity.mxu0.live", "capacity.mxu0.window")),
        (6, lambda d: d["op_sequences"][0]["ops"][0]["inputs"].update(instruction=2), ("ops[0].inputs.instruction",)),
        (1, lambda d: d["op_pairs"][0].update(relation={"slot": "same"}, recipe="new"),
         ("relation: absent", "recipe: absent"))]
        for index, change, fields in cases:
            with self.subTest(index=index, fields=fields): self.differ(self.n[index], change, *fields)
    def test_every_dependency_pin_and_schema(self):
        for root in (self.g, self.r):
            doc = self.read(root, self.n[3]); doc["op_timing"]["sha256"] = "f" * 64
            self.write(root, self.n[3], doc)
        differences = self.compare([self.n[3]])
        self.assertEqual(len(differences), 2)
        self.assertTrue(all("does not match source" in d for d in differences))
        for index in range(4):
            with self.subTest(source=index):
                self.differ(self.n[3], lambda d: d["sources"][index].update(sha256="f" * 64), "recorded sha256")
        self.differ(self.n[3], lambda d: d["sources"][0].update(schema="wrong"), "schema does not match source")
        self.populate(self.g); self.populate(self.r)
        digest = self.mutate(self.n[0], lambda d: d["op_timing"][0]["events"]["write"].update(last_age=10))
        self.mutate(self.n[3], lambda d: d["op_timing"].update(sha256=digest))
        self.assertTrue(any("op_timing.sha256" in d for d in self.compare([self.n[3]])))
    def test_missing_duplicate_and_malformed_data_fail(self):
        for index, key in ((0, "op_timing"), (3, "op_pairs")):
            self.differ(self.n[index], lambda d: d[key].append(copy.deepcopy(d[key][0])), "duplicate record")
        self.differ(self.n[0], lambda d: d["op_timing"].pop(1), "vpu.relu: missing")
        for results, message in (([{"from": 1, "to": 2}], "missing results"),
                        ([{"from": 1, "to": 3}, {"from": 2, "to": 3}], "duplicate result"),
                        ([{"from": 1, "to": 4}], "outside gaps")):
            self.differ(self.n[1], lambda d: d["op_pairs"][0].update(results=results), message)
        malformed = [(None, "missing artifact"), ("{", "generated:"),
                ('{"schema": "unknown"}', "unknown timing artifact schema")]
        for raw, message in malformed:
            path = self.g / self.n[0]; path.unlink() if raw is None else path.write_text(raw)
            self.assertTrue(any(message in d for d in self.compare([self.n[0]])))

class ModelAdapterTests(unittest.TestCase):
    def test_makeflags_and_inventory(self):
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        arg = f'CXX={root}/compiler/g++ LINK={root}/compiler/g++ AR={root}/compiler/ar'
        self.assertEqual(model.command_record(['-MAKEFLAGS', arg], root),
                         ['-MAKEFLAGS', 'CXX=compiler/g++ LINK=compiler/g++ AR=compiler/ar'])
        work = root / 'replay'; work.mkdir()
        (work / 'case.txt').write_bytes(b'passed'); (work / 'summary.json').write_text('old')
        self.assertEqual(model.inventory(work, root, [work / 'summary.json']),
                         [{'path': 'replay/case.txt', 'sha256': hashlib.sha256(b'passed').hexdigest()}])
    def test_replay_receipt_and_count(self):
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        work = root / 'replay'
        work.mkdir()
        (work / 'case.txt').write_text('passed')
        args = argparse.Namespace(script=root / 'helper.py', build_root=root, work=work, summary=None, expected_pass=2)
        helper = SimpleNamespace(run=lambda args: [{'case': 'a'}, {'case': 'b'}])
        with patch.object(model, 'load_helper', return_value=helper):
            result = model.replay(args)
            self.assertEqual(result['pass_count'], 2)
            self.assertEqual(json.loads((work / 'summary.json').read_text()), result)
            args.expected_pass = 3
            with self.assertRaisesRegex(ValueError, 'expected 3, got 2'): model.replay(args)
            args.expected_pass = 2; helper.run = lambda args: [{'case': 'a'}, {'case': 'a'}]
            with self.assertRaisesRegex(ValueError, 'duplicate'): model.replay(args)

    def test_actual_model_command(self):
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        rtl = root / 'rtl'
        rtl.mkdir()
        (rtl / 'source.sv').write_text('module X; endmodule')
        args = argparse.Namespace(script=root / 'helper.py', output=root / 'model/VAtlasCore', build_root=root,
                                  rtl=rtl, verilator='verilator', verilator_root=None)
        helper = SimpleNamespace(sh=lambda *args: '')
        def build(args, work):
            helper.sh(['verilator', '--Mdir', str(work / 'obj'), *args.rtl.glob('*.sv')], 30)
            (work / 'obj').mkdir(); binary = work / 'obj/VAtlasCore'; binary.write_bytes(b'binary')
            return binary
        helper.build_model = build
        with patch.object(model, 'load_helper', return_value=helper), patch.object(model.subprocess, 'run', return_value=SimpleNamespace(stdout='Verilator 5\n')):
            result = model.model(args)
        self.assertEqual(result['commands'][0]['argv'], ['verilator', '--Mdir', 'model/obj', 'rtl/source.sv'])
        self.assertEqual(args.output.read_bytes(), b'binary')
        self.assertTrue((root / 'model/build.json').is_file())

class RegenSweepTests(unittest.TestCase):
    def test_partition(self):
        names = list('abcdefg'); chunks = sweep.partition(names, 3)
        self.assertEqual([name for chunk in chunks for name in chunk], names)
        self.assertLessEqual(max(map(len, chunks)) - min(map(len, chunks)), 1)
        self.assertEqual(sweep.partition(names, 20), [[name] for name in names])
        with self.assertRaises(ValueError): sweep.partition(names, 0)

    def test_merge(self):
        self.assertEqual(sweep.merge_records(['a', 'b'], [[{'name': 'b'}], [{'name': 'a'}]]), [{'name': 'a'}, {'name': 'b'}])
        for shards in ([[{'name': 'a'}], [{'name': 'a'}]], [[{'name': 'b'}]], [[{'name': 'a'}, {'name': 'extra'}]]):
            with self.subTest(shards=shards), self.assertRaises(ValueError): sweep.merge_records(['a'], shards)

    def test_recipe_equivalence(self):
        spec = circuit.load(synthetic.SPEC)
        raw = pairs.validate({'engine': 'toy', 'pairs': {
            'guarded': {'first': 'go', 'second': 'go', 'gaps': '1..6'},
            'unguarded': {'first': 'go', 'second': 'go', 'gaps': '1..6', 'guards': []}}}, spec, 'toy')
        document = {'modules': [synthetic.counter()]}
        self.assertEqual(sweep.extract_sharded(document, raw, spec, jobs=4, shards=2), pairs.extract(document, raw, spec))

class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(driver.MANIFEST.read_text())

    def test_inventory_and_selection(self):
        selected = driver.stages(self.config, None)
        mapping = driver.artifacts(self.config, selected)
        self.assertEqual(len(selected), 27)  # exporter expands to configure/build.
        self.assertEqual(len(mapping), 20)
        self.assertEqual(len(driver.stages(self.config, 'sweeps')), 17)
        self.assertEqual(driver.stages(self.config, 'pairs-core_dma,facts'), ['facts', 'pairs-core_dma'])
        for name in ('core_mxu_same', 'core_mxu_shared'):
            self.assertEqual(mapping[f'out/pairs-{name}.json'], f'system/pairs-{name}.json')
        self.assertEqual(mapping['out/sequences-core_mxu.json'], 'system/sequences-core_mxu.json')
        self.assertNotIn('out/sequences-core_mxu.json', self.config['pair_facts']['sequences'])
        self.assertEqual(self.config['pair_facts']['pairs'],
                         ['out/pairs-core_vpu.json', 'out/pairs-core_mxu_same.json', 'out/pairs-core_mxu_shared.json'])
        with self.assertRaisesRegex(ValueError, 'unknown stages'):
            driver.stages(self.config, 'nonexistent')

    def test_extraction_cache_covers_generated_and_file_specs(self):
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        args = driver.argparse.Namespace(python=Path(sys.executable), cmake='cmake', cxx='g++', circt_prefix=root)
        runner = driver.argparse.Namespace(run=Mock())
        with patch.object(driver, 'executable', side_effect=lambda p: Path(p)), \
             patch.object(driver, 'version', return_value='test'), \
             patch.object(driver.subprocess, 'check_output', return_value='test'):
            driver.regenerate(args, self.config, root, runner, ['facts'])
        inputs = runner.run.call_args.args[2]
        self.assertIn(REPO / 'tools/rtl_extract/specs/gen_core.py', inputs)
        self.assertIn(REPO / 'tools/rtl_extract/specs/vpu.yaml', inputs)

    def test_environment_and_output_confinement(self):
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        args = driver.argparse.Namespace(bin=root / 'bin', python=Path(sys.executable))
        driver.emit_environment(args, self.config, root)
        env = dict(shlex.split(line)[1].split('=', 1) for line in (root / 'env.sh').read_text().splitlines())
        self.assertEqual(env['ATLAS_PAIR_FACTS'], str(root / 'out/op-pairs.json'))
        self.assertEqual(env['ATLAS_TIMING_PROBE'], str(args.bin / 'rtl-timing-facts-probe'))
        self.assertEqual(len(env['ATLAS_PAIR_SWEEPS'].split(os.pathsep)), 13)
        self.assertEqual(env['ATLAS_OP_PAIRS'].split(os.pathsep),
                         [str(root / f'out/pairs-mxu{i}.json') for i in (0, 1)])
        for name in ('../outside', str(root.parent / 'outside')):
            with self.assertRaisesRegex(ValueError, 'escapes'):
                driver.output_path(root, name)

if __name__ == '__main__':
    unittest.main()
