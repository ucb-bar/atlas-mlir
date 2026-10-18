"""Actual compiler gates for optional exact splat materialization sharing."""
import os
import importlib.util
import unittest

from atlas_egglog_ir import parse_program
from atlas_egglog_sharing import bounded_splat_sharing

COMPILER = os.environ.get('ATLAS_OPT')


def source(bits=(16256, 16256), metadata=('', '')):
    lines = ['module {', '  func.func @sharing() -> !atlas.virtual_state attributes {atlas.input_dram_base = 2415919104 : i64, atlas.output_dram_base = 2415984640 : i64} {',
             '    %s = "atlas.virtual_start"() : () -> !atlas.virtual_state']
    state = '%s'
    for i, (value, attrs) in enumerate(zip(bits, metadata)):
        lines += [f'    %c{i} = "atlas.virtual_vpu_splat"() {{bits = {value} : i32{attrs}}} : () -> !atlas.virtual_bf16',
                  f'    %o{i} = "atlas.virtual_output_bf16"({state}, %c{i}) {{index = {i} : i32}} : (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state']
        state = f'%o{i}'
    return '\n'.join(lines + [f'    return {state} : !atlas.virtual_state', '  }', '}']) + '\n'


def pressure_source():
    # Native peak15 in the mixed bank; retaining %first across the middle
    # reduction adds a sixteenth BF16 live pair and must be rejected.
    lines = ['module {', '  func.func @pressure() -> !atlas.virtual_state attributes {atlas.input_dram_base = 2415919104 : i64, atlas.output_dram_base = 2415984640 : i64} {',
             '    %s = "atlas.virtual_start"() : () -> !atlas.virtual_state',
             '    %io, %fp = "atlas.virtual_input_fp8"(%s) {index = 0 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_fp8)',
             '    %first = "atlas.virtual_vpu_splat"() {bits = 40000 : i32} : () -> !atlas.virtual_bf16',
             '    %a = "atlas.virtual_output_bf16"(%io, %first) {index = 0 : i32} : (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state']
    for i in range(14):
        lines.append(f'    %v{i} = "atlas.virtual_vpu_splat"() {{bits = {16256+i} : i32}} : () -> !atlas.virtual_bf16')
    value = '%v0'
    for i in range(1, 14):
        lines.append(f'    %r{i} = "atlas.virtual_vpu_binary"({value}, %v{i}) {{kind = "max"}} : (!atlas.virtual_bf16, !atlas.virtual_bf16) -> !atlas.virtual_bf16')
        value = f'%r{i}'
    lines += [f'    %b = "atlas.virtual_output_bf16"(%a, {value}) {{index = 1 : i32}} : (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state',
              '    %last = "atlas.virtual_vpu_splat"() {bits = 40000 : i32} : () -> !atlas.virtual_bf16',
              '    %end = "atlas.virtual_output_bf16"(%b, %last) {index = 2 : i32} : (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state',
              '    return %end : !atlas.virtual_state', '  }', '}']
    return '\n'.join(lines) + '\n'


class DefaultTests(unittest.TestCase):
    def test_default_is_byte_identical_without_parsing_or_compiler(self):
        for text in (source(), 'unsupported\r\nsyntax\n'):
            result = bounded_splat_sharing(text)
            self.assertEqual(result.source, text)
            self.assertEqual(result.evidence['status'], 'disabled')
            self.assertFalse(result.evidence['shares'])

    def test_enabled_requires_actual_allocator_and_positive_bounds(self):
        with self.assertRaisesRegex(ValueError, 'real atlas-opt'):
            bounded_splat_sharing(source(), enabled=True)
        for kw in ({'max_distance': 0}, {'max_constants': 0}, {'max_distance': True}):
            with self.assertRaisesRegex(ValueError, 'positive integers'):
                bounded_splat_sharing(source(), enabled=True, **kw)


@unittest.skipUnless(COMPILER, 'set ATLAS_OPT for actual allocation gates')
class CompilerTests(unittest.TestCase):
    def share(self, text, **options):
        return bounded_splat_sharing(text, enabled=True, atlas_opt=COMPILER, **options)

    def test_exact_raw_values_including_signed_zero_and_nan(self):
        for bits, shares in (((0, 0), 1), ((0, 32768), 0),
                             ((32704, 32704), 1), ((32704, 32705), 0)):
            with self.subTest(bits=bits):
                result = self.share(source(bits))
                self.assertEqual(len(result.evidence['shares']), shares)
                self.assertTrue(result.evidence['baseline_allocation']['accepted'])
                self.assertEqual(result.evidence['status'], 'accepted' if shares else 'no_sharing')
                parse_program(result.source)

    def test_metadata_must_match_and_ssa_inside_metadata_is_opaque(self):
        matching = ', atlas.note = "%c1, %c0"'
        result = self.share(source(metadata=(matching, matching)))
        self.assertEqual(len(result.evidence['shares']), 1)
        self.assertIn('atlas.note = "%c1, %c0"', result.source)
        self.assertIn('"atlas.virtual_output_bf16"(%o0, %c0)', result.source)
        different = self.share(source(metadata=(matching, ', atlas.note = "%c0"')))
        self.assertEqual(different.evidence['status'], 'no_sharing')

    def test_future_use_distance_is_bounded(self):
        text = source().replace('    %o1 =', '    %gap = "atlas.virtual_vpu_splat"() {bits = 99 : i32} : () -> !atlas.virtual_bf16\n    %o1 =')
        refused = self.share(text, max_distance=3)
        accepted = self.share(text, max_distance=4)
        self.assertEqual(refused.evidence['status'], 'no_sharing')
        self.assertEqual(accepted.evidence['status'], 'accepted')
        self.assertEqual(accepted.evidence['shares'][0]['distance'], 4)

    def test_retained_distinct_constant_limit(self):
        text = source((1, 2, 1), ('', '', ''))
        self.assertEqual(self.share(text, max_constants=1).evidence['status'], 'no_sharing')
        self.assertEqual(len(self.share(text, max_constants=2).evidence['shares']), 1)

    def test_forward_reference_and_multiblock_are_unsupported(self):
        malformed = source().replace('(%s, %c0)', '(%s, %c1)', 1)
        multiblock = source().replace('    %c1 =', '  ^other:\n    %c1 =')
        colon_metadata = source(metadata=(', atlas.note = "%c1: note"',) * 2)
        for text in (malformed, multiblock, colon_metadata):
            result = self.share(text)
            self.assertEqual(result.evidence['status'], 'unsupported')
            self.assertEqual(result.source, text)

    def test_actual_allocation_refusal_retains_exact_baseline(self):
        text = pressure_source()
        result = self.share(text, max_distance=100, max_constants=16)
        self.assertEqual(result.evidence['status'], 'allocation_refused')
        self.assertEqual(result.source, text)
        self.assertEqual(len(result.evidence['shares']), 1)
        for label, peak, accepted in (('baseline', 15, True), ('candidate', 16, False)):
            report = result.evidence[label + '_allocation']
            self.assertEqual(report['accepted'], accepted)
            self.assertEqual(report['allocation']['registers']['bf16']['peak_live'], peak)

    def test_compiler_pin_mismatch_is_not_qualified(self):
        with self.assertRaisesRegex(ValueError, 'compiler pin mismatch'):
            self.share(source(), expected_compiler_sha256='0' * 64)

    def test_other_pure_constants_are_not_shared(self):
        text = source((1, 2)).replace('    %c0 =',
            '    %scale0 = "atlas.virtual_scale_constant"() {code = 127 : i32} : () -> !atlas.virtual_scale\n'
            '    %scale1 = "atlas.virtual_scale_constant"() {code = 127 : i32} : () -> !atlas.virtual_scale\n'
            '    %c0 =')
        result = self.share(text)
        self.assertEqual(result.evidence['status'], 'no_sharing')
        self.assertEqual(result.source, text)

    @unittest.skipUnless(importlib.util.find_spec('egglog') and importlib.util.find_spec('merlin'),
                         'select egglog and explicit Merlin src for extraction gate')
    def test_sharing_identical_actually_extracted_terms_keeps_effect_chain(self):
        from atlas_egglog_ir import egglog_roundtrip
        text = source()
        original = parse_program(text)
        decoded, roots = egglog_roundtrip(original)
        self.assertEqual(decoded, text)
        self.assertEqual(roots['%c0'], roots['%c1'])
        result = self.share(decoded)
        rewritten = parse_program(result.source)
        self.assertEqual(len(result.evidence['shares']), 1)
        fixed = lambda p: [(op.op, op.results, op.attrs,
                           tuple(name for name, type in zip(op.operands, op.input_types)
                                 if type != '!atlas.virtual_bf16'))
                          for op in p.operations if not op.pure]
        self.assertEqual(fixed(original), fixed(rewritten))


if __name__ == '__main__':
    unittest.main()
