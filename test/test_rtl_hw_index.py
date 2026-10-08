"""Structural-index regressions; these do not qualify RTL instruction timing."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SPEC = importlib.util.spec_from_file_location('rtl_hw_index', Path(__file__).resolve().parents[1] / 'tools/index-retained-hw.py')
INDEX = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(INDEX)


DESIGN = '''module {
  hw.module private @Leaf(in %clock : !seq.clock, in %a : i1, out valid : i1) {
    %q = seq.firreg %next clock %clock : i1
    %next = comb.xor %q, %a : i1
    hw.output %next : i1
  }
  hw.module private @Top(in %clock : !seq.clock, in %a : i1, out valid : i1) {
    %v0 = hw.instance "first" sym @inner @Leaf(clock: %clock: !seq.clock, a: %a: i1) -> (valid: i1)
    %v1 = hw.instance "second" @Leaf(clock: %clock: !seq.clock, a: %v0: i1) -> (valid: i1)
    hw.output %v1 : i1
  }
}
'''


class StructuralIndexTest(unittest.TestCase):
    def test_repeated_instances_private_modules_and_bindings(self):
        report = INDEX.build_index(DESIGN, 'Top')
        self.assertEqual(len(report['modules']), 2)
        self.assertEqual([p['path'] for p in report['instance_paths']],
                         [['Top'], ['Top', 'first'], ['Top', 'second']])
        top = next(m for m in report['modules'] if m['name'] == 'Top')
        self.assertEqual(top['instances'][1]['inputs'][1]['ssa'], '%v0')
        self.assertEqual(top['instances'][0]['outputs'][0]['ssa'], '%v0')
        self.assertEqual(top['ports'][-1]['output_locator']['operand_index'], 0)
        self.assertNotIn('text', top['ports'][-1]['output_locator'])
        self.assertIn('hw.output %v1', top['output_locator']['text'])
        self.assertFalse(report['timing_rules_enabled'])

    def test_feedback_stops_at_register_and_instance(self):
        report = INDEX.build_index(DESIGN, 'Top')
        leaf = next(e for e in report['candidate_events'] if e['module'] == 'Leaf')
        self.assertEqual(next(n for n in leaf['cone'] if n['ssa'] == '%q')['boundary'], 'register')
        top = next(e for e in report['candidate_events'] if e['module'] == 'Top')
        self.assertEqual(top['cone'][0]['boundary'], 'instance_output')

    def test_unresolved_reference_and_port_mismatch(self):
        with self.assertRaisesRegex(INDEX.IndexError, 'Unresolved module reference'):
            INDEX.build_index(DESIGN.replace('@Leaf(clock:', '@Absent(clock:'), 'Top')
        with self.assertRaisesRegex(INDEX.IndexError, 'port correspondence'):
            INDEX.build_index(DESIGN.replace('a: %v0:', 'wrong: %v0:'), 'Top')

    def test_zero_result_instance_and_parameterized_opaque_declaration(self):
        design = '''module {
  hw.module.extern @NoOut<WIDTH: ui32>(in %a : i1)
  hw.module private @Top(in %a : i1) {
    hw.instance "side effect" @NoOut<WIDTH: 1>(a: %a: i1) -> ()
    hw.output
  }
}
'''
        report = INDEX.build_index(design, 'Top')
        self.assertEqual(len(report['instance_paths']), 2)
        self.assertEqual(report['instance_paths'][1]['path'], ['Top', 'side effect'])
        top = next(m for m in report['modules'] if m['name'] == 'Top')
        self.assertEqual(top['instances'][0]['outputs'], [])

    def test_indexed_output_and_rejected_incomplete_syntax(self):
        report = INDEX.build_index(DESIGN.replace('valid', 'io_channelBusy_0'), 'Top')
        self.assertEqual(len(report['candidate_events']), 2)
        with self.assertRaisesRegex(INDEX.IndexError, 'Unbalanced|Multiline'):
            INDEX.build_index(DESIGN.replace('%next = comb.xor %q, %a : i1', '%next = comb.xor (%q,\n      %a) : i1'), 'Top')
        with self.assertRaisesRegex(INDEX.IndexError, 'collides with module input'):
            INDEX.build_index(DESIGN.replace('%q = seq.firreg', '%a = seq.firreg'), 'Top')
        with self.assertRaisesRegex(INDEX.IndexError, 'Incomplete typed combinational'):
            INDEX.build_index(DESIGN.replace('%next = comb.xor %q, %a : i1', '%next = comb.xor %q,\n      %a : i1'), 'Top')
        with self.assertRaisesRegex(INDEX.IndexError, 'Unsupported register suffix'):
            INDEX.build_index(DESIGN.replace('clock %clock : i1', 'clock %clock reset sync %a : i1'), 'Top')

    def test_missing_ssa_and_unsupported_operation_boundary(self):
        with self.assertRaisesRegex(INDEX.IndexError, 'Missing event-cone SSA'):
            INDEX.build_index(DESIGN.replace('hw.output %next', 'hw.output %missing'), 'Leaf')
        report = INDEX.build_index(DESIGN.replace('comb.xor', 'unknown.compute'), 'Leaf')
        self.assertEqual(report['candidate_events'][0]['cone'][0]['boundary'], 'unsupported_operation')

    def test_event_hint_ambiguity_and_missing_anchor(self):
        hints = '\n'.join('    %h' + str(i) + ' = comb.and %a, %a {sv.namehint = "' + h + '"} : i1'
                          for i, h in enumerate(INDEX.REQUIRED_HINTS))
        scalar = 'module {\n  hw.module private @ScalarCore(in %a : i1, out valid : i1) {\n' + hints + '\n    hw.output %h0 : i1\n  }\n}\n'
        self.assertEqual(len(INDEX.build_index(scalar, 'ScalarCore')['candidate_events']), 8)
        duplicate = scalar.replace('    hw.output', '    %extra = comb.and %a, %a {sv.namehint = "s1_fire"} : i1\n    hw.output')
        with self.assertRaisesRegex(INDEX.IndexError, 'ambiguous ScalarCore event hint'):
            INDEX.build_index(duplicate, 'ScalarCore')
        with self.assertRaisesRegex(INDEX.IndexError, 'Missing/ambiguous ScalarCore event hint'):
            INDEX.build_index(scalar.replace('is_xlu_launch', 'other_hint'), 'ScalarCore')

    def test_manifest_binding_and_new_output(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            hw = base / 'design.mlir'
            hw.write_text(DESIGN)
            manifest = base / 'manifest.json'
            doc = {'schema': 'atlas.retained_hw_ir.v0', 'state': 'verified', 'hardware_ir': INDEX.identity(hw)}
            manifest.write_text(json.dumps(doc))
            report = INDEX.run(manifest, base / 'report', 'Top')
            self.assertEqual((base / 'report/retention-manifest.json').read_bytes(), manifest.read_bytes())
            self.assertEqual(report['inputs']['hardware_ir']['sha256'], doc['hardware_ir']['sha256'])
            with self.assertRaisesRegex(INDEX.IndexError, 'must be new'):
                INDEX.run(manifest, base / 'report', 'Top')
            hw.write_text(DESIGN + '// changed\n')
            with self.assertRaisesRegex(INDEX.IndexError, 'hash/size mismatch'):
                INDEX.run(manifest, base / 'mismatch', 'Top')
            self.assertFalse((base / 'mismatch').exists())
            doc['state'] = 'started'
            manifest.write_text(json.dumps(doc))
            with self.assertRaisesRegex(INDEX.IndexError, 'verified retained-HW'):
                INDEX.run(manifest, base / 'unverified', 'Top')


if __name__ == '__main__':
    unittest.main()
