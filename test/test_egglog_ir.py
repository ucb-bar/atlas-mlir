"""Pure representation, extraction reconstruction and fixed-effect safety."""
from dataclasses import replace
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from atlas_egglog_ir import (
    IllegalExtraction, Node, UnsupportedProgram, decode_program,
    egglog_roundtrip, load_program, parse_program, pure_roots,
)

BF = "!atlas.virtual_bf16"
FP = "!atlas.virtual_fp8"
S = "!atlas.virtual_state"


def source(body: str) -> str:
    return ('module {\n'
            f'  func.func @tile() -> {S} {{\n'
            f'    %s = "atlas.virtual_start"() : () -> {S}\n'
            f'    %t, %x = "atlas.virtual_input_bf16"(%s) {{index = 0 : i32}} : ({S}) -> ({S}, {BF})\n'
            + body + '\n'
            f'    %u = "atlas.virtual_output_bf16"(%t, %y) {{index = 0 : i32}} : ({S}, {BF}) -> {S}\n'
            f'    return %u : {S}\n'
            '  }\n}\n')


MOV = f'    %y = "atlas.virtual_vpu_unary"(%x) {{kind = "mov"}} : ({BF}) -> {BF}'
SPLAT = f'    %y = "atlas.virtual_vpu_splat"() {{bits = 16256 : i32}} : () -> {BF}'


class EgglogIRTests(unittest.TestCase):
    def test_lossless_without_egglog(self):
        text = source(MOV).replace('module {', '// comment\nmodule {')
        program = parse_program(text)
        self.assertEqual(decode_program(program, pure_roots(program)), text)

    def test_properties_and_attributes_roundtrip_without_changing_spelling(self):
        text = source(MOV).replace('{kind = "mov"}', '<{kind = "mov"}>')
        program = parse_program(text)
        self.assertEqual(decode_program(program, pure_roots(program)), text)
        duplicate = text.replace('<{kind = "mov"}>', '<{kind = "mov"}> {kind = "mov"}')
        with self.assertRaises(UnsupportedProgram):
            parse_program(duplicate)

    def test_opaque_proof_metadata_does_not_change_facts_or_spelling(self):
        metadata = '{kind = "mov", atlas.derived_finite = true, atlas.proof_reason = "a, b"}'
        text = source(MOV).replace('{kind = "mov"}', metadata)
        program = parse_program(text)
        self.assertTrue(program.nodes['%y'].attribute('atlas.derived_finite'))
        self.assertEqual(program.nodes['%y'].attribute('atlas.proof_reason'), 'a, b')
        self.assertEqual(decode_program(program, pure_roots(program)), text)

    def test_mutation_decodes_actual_extraction(self):
        program = parse_program(source(SPLAT))
        roots = pure_roots(program)
        roots['%y'] = replace(roots['%y'], attrs=(("bits", "16384 : i32"),))
        result = decode_program(program, roots)
        self.assertNotEqual(result, ''.join(program.lines))
        self.assertEqual(parse_program(result).nodes['%y'].attribute('bits'), 16384)

    def test_elimination_reconnects_fixed_effect_consumer(self):
        program = parse_program(source(MOV))
        result = decode_program(program, {'%y': program.nodes['%x']})
        self.assertNotIn('virtual_vpu_unary', result)
        self.assertIn('virtual_output_bf16"(%t, %x)', result)

    def test_root_alias_to_dominating_pure_operand(self):
        text = source(SPLAT.replace('%y', '%a') + '\n' + MOV.replace('unary"(%x)', 'unary"(%a)'))
        program = parse_program(text)
        roots = pure_roots(program)
        roots['%y'] = program.nodes['%a']
        result = decode_program(program, roots)
        self.assertEqual(result.count('virtual_vpu_splat'), 1)
        self.assertNotIn('virtual_vpu_unary', result)
        self.assertIn('virtual_output_bf16"(%t, %a)', result)

    def test_optional_dead_pure_pruning_removes_transitive_dead_dependencies(self):
        body = SPLAT.replace('%y', '%dead') + '\n' + MOV.replace('%y', '%dead_mov').replace('unary"(%x)', 'unary"(%dead)') + '\n' + MOV
        program = parse_program(source(body))
        roots = pure_roots(program)
        default = decode_program(program, roots)
        self.assertEqual(default, ''.join(program.lines))
        pruned = decode_program(program, roots, prune_dead_pure=True)
        self.assertNotIn('%dead', pruned)
        self.assertNotIn('virtual_vpu_splat', pruned)
        self.assertIn('%y =', pruned)
        self.assertEqual([op.op for op in parse_program(pruned).operations if not op.pure],
                         [op.op for op in program.operations if not op.pure])

    def test_pruning_keeps_stateful_mxu_and_load_operations(self):
        program = load_program(ROOT / 'test/examples/virtual_attention_tile.mlir')
        decoded = parse_program(decode_program(program, pure_roots(program), prune_dead_pure=True))
        self.assertEqual([op for op in decoded.operations if not op.pure][0].op, 'virtual_start')
        self.assertEqual([op.op for op in decoded.operations if not op.pure],
                         [op.op for op in program.operations if not op.pure])

    def test_nested_new_dependency_is_materialized_before_use(self):
        program = parse_program(source(MOV))
        zero = Node('virtual_vpu_splat', (), (('bits', '0 : i32'),), BF)
        maximum = Node('virtual_vpu_binary', (program.nodes['%x'], zero), (('kind', '"max"'),), BF)
        result = decode_program(program, {'%y': maximum})
        self.assertLess(result.index('virtual_vpu_splat'), result.index('virtual_vpu_binary'))
        self.assertEqual(parse_program(result).nodes['%y'], maximum)

    def test_new_splats_are_rematerialized_per_root(self):
        first = MOV.replace('%y', '%a')
        second = MOV
        program = parse_program(source(first + '\n' + second))
        zero = Node('virtual_vpu_splat', (), (('bits', '0 : i32'),), BF)
        clamp = Node('virtual_vpu_binary', (program.nodes['%x'], zero), (('kind', '"max"'),), BF)
        result = decode_program(program, {'%a': clamp, '%y': clamp})
        self.assertEqual(result.count('virtual_vpu_splat'), 2)

    def test_shared_source_nodes_keep_original_ssa_spelling(self):
        text = source(SPLAT.replace('%y', '%a') + '\n' + SPLAT)
        program = parse_program(text)
        self.assertEqual(decode_program(program, pure_roots(program)), text)

    def test_forward_anchor_is_rejected(self):
        text = source(MOV).replace('    %u =',
            f'    %later, %z = "atlas.virtual_input_bf16"(%t) {{index = 1 : i32}} : ({S}) -> ({S}, {BF})\n    %u =')
        text = text.replace('virtual_output_bf16"(%t,', 'virtual_output_bf16"(%later,')
        program = parse_program(text)
        with self.assertRaisesRegex(IllegalExtraction, 'dominate'):
            decode_program(program, {'%y': program.nodes['%z']})

    def test_extraction_rejects_missing_extra_wrong_type_unknown_effect_anchor(self):
        program = parse_program(source(MOV))
        cases = ({}, {'%y': program.nodes['%y'], '%extra': program.nodes['%y']},
                 {'%y': replace(program.nodes['%y'], type=FP)},
                 {'%y': Node('virtual_output_bf16', (), (), BF)},
                 {'%y': Node('anchor', (), (('id', '%forged'),), BF)},
                 {'%y': Node('virtual_vpu_unary', (), (('kind', '"mov"'),), BF)},
                 {'%y': Node('virtual_vpu_splat', (), (('bits', '65536 : i32'),), BF)})
        for extraction in cases:
            with self.subTest(extraction=extraction), self.assertRaises(IllegalExtraction):
                decode_program(program, extraction)

    def test_parser_rejects_duplicate_undefined_wrong_type_stale_state_bad_range(self):
        text = source(MOV)
        cases = (text.replace('%y =', '%x ='), text.replace('unary"(%x)', 'unary"(%missing)'),
                 text.replace('unary"(%x)', 'unary"(%s)'), text.replace('output_bf16"(%t,', 'output_bf16"(%s,'),
                 source(SPLAT.replace('16256', '-1')), text.replace('kind = "mov"', 'kind = "unsupported"'),
                 text.replace('kind = "mov"', 'kind = "mov", kind = "mov"'), text[:-2], text + '}\n')
        for invalid in cases:
            with self.subTest(invalid=invalid), self.assertRaises(UnsupportedProgram):
                parse_program(invalid)

    def test_resource_unit_mismatch_and_double_consumption(self):
        path = ROOT / 'test/examples/virtual_mxu_seeded_fp8.mlir'
        text = path.read_text()
        self.assertTrue(parse_program(text).operations)
        with self.assertRaisesRegex(UnsupportedProgram, 'unit mismatch'):
            parse_program(text.replace('virtual_mxu_acc<0>', 'virtual_mxu_acc<1>'))
        with self.assertRaisesRegex(UnsupportedProgram, 'consumed twice'):
            parse_program(text.replace('readout_fp8"(%s2, %a1,', 'readout_fp8"(%s2, %a0,'))

    def test_no_state_resource_values_in_eclasses(self):
        program = load_program(ROOT / 'test/examples/virtual_attention_tile.mlir')
        self.assertTrue(any('mxu_reset' in op.op for op in program.operations))
        for _, node in program.values:
            self.assertIn(node.type, (BF, FP, '!atlas.virtual_scale'))

    def test_caller_contract_preserved_as_data_only(self):
        text = source(MOV).replace('    %u =',
            f'    %p = "atlas.virtual_pack_fp8"(%y) {{finite_src, scale_code = 127 : i32}} : ({BF}) -> {FP}\n    %u =')
        program = parse_program(text)
        self.assertIsNone(program.nodes['%p'].attribute('finite_src'))
        self.assertEqual(decode_program(program, pure_roots(program)), text)

    def test_egglog_roundtrip_and_mutated_extraction(self):
        try:
            from egglog import EGraph
            from atlas_egglog_schema import from_egglog, to_egglog
        except ImportError:
            self.skipTest('install optional tools/requirements-egglog.txt')
        program = parse_program(source(MOV))
        self.assertEqual(egglog_roundtrip(program)[0], ''.join(program.lines))
        replacement = Node('virtual_vpu_splat', (), (('bits', '16384 : i32'),), BF)
        graph = EGraph()
        expr = to_egglog(replacement)
        graph.register(expr)
        extracted = from_egglog(graph.extract(expr))
        result = decode_program(program, {'%y': extracted})
        self.assertIn('bits = 16384', result)
        self.assertNotIn('virtual_vpu_unary', result)

    def test_egglog_union_extracts_and_decodes_eliminated_operation(self):
        try:
            from egglog import EGraph, union
            from atlas_egglog_schema import from_egglog, to_egglog
        except ImportError:
            self.skipTest('install optional tools/requirements-egglog.txt')
        program = parse_program(source(MOV))
        expr = to_egglog(program.nodes['%y'])
        graph = EGraph()
        graph.register(union(expr).with_(to_egglog(program.nodes['%x'])))
        extracted = from_egglog(graph.extract(expr))
        self.assertEqual(extracted, program.nodes['%x'])
        result = decode_program(program, {'%y': extracted})
        self.assertNotIn('virtual_vpu_unary', result)
        self.assertIn('virtual_output_bf16"(%t, %x)', result)

    def test_malformed_extracted_attribute_json_is_rejected(self):
        try:
            from atlas_egglog_schema import Inputs, Value, from_egglog
        except ImportError:
            self.skipTest('install optional tools/requirements-egglog.txt')
        for attrs in ('{}', '[[]]', '[["bits",42]]', 'not-json'):
            with self.subTest(attrs=attrs), self.assertRaises(IllegalExtraction):
                from_egglog(Value.apply('virtual_vpu_splat', attrs, BF, Inputs.empty()))


if __name__ == '__main__':
    unittest.main()
