"""Adversarial provider/receipt checks; physical execution is a separate gate."""
from dataclasses import replace
from pathlib import Path
import hashlib
import json
import tempfile
import unittest
from atlas_egglog_ir import parse_program, pure_roots
from atlas_egglog_facts import analyze_program
from atlas_egglog_pack import recipe, choices_from_roots, materialize, pack_rules
from select_atlas_candidates import check_admission, check_measurement
from atlas_hand_reference import enumerate_hand, simplify


def fixture(bits=16256, scale=127, adjacent=False, live=False):
    lines = ['module {', '  func.func @test() -> !atlas.virtual_state {',
        '    %s = "atlas.virtual_start"() : () -> !atlas.virtual_state',
        f'    %a = "atlas.virtual_vpu_splat"() {{bits = {bits} : i32}} : () -> !atlas.virtual_bf16']
    if live:
        lines += ['    %s1, %acc0 = "atlas.virtual_mxu_load_acc_bf16"(%s, %a) {unit = 0 : i32} : (!atlas.virtual_state, !atlas.virtual_bf16) -> (!atlas.virtual_state, !atlas.virtual_mxu_acc<0>)',
                  '    %s2, %acc1 = "atlas.virtual_mxu_load_acc_bf16"(%s1, %a) {unit = 0 : i32} : (!atlas.virtual_state, !atlas.virtual_bf16) -> (!atlas.virtual_state, !atlas.virtual_mxu_acc<0>)']
    lines += [f'    %p = "atlas.virtual_pack_fp8"(%a) {{scale_code = {scale} : i32}} : (!atlas.virtual_bf16) -> !atlas.virtual_fp8']
    if adjacent:
        lines += [f'    %q = "atlas.virtual_pack_fp8"(%a) {{scale_code = {scale+1} : i32}} : (!atlas.virtual_bf16) -> !atlas.virtual_fp8']
    lines += [f'    %y = "atlas.virtual_mxu_matmul"(%p, {"%q" if adjacent else "%p"}) {{unit = 1 : i32}} : (!atlas.virtual_fp8, !atlas.virtual_fp8) -> !atlas.virtual_bf16',
              f'    %end = "atlas.virtual_output_bf16"({"%s2" if live else "%s"}, %y) {{index = 0 : i32}} : (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state',
              '    return %end : !atlas.virtual_state', '  }', '}']
    return parse_program('\n'.join(lines)+'\n')


def selected(program, unit=0):
    roots=pure_roots(program)
    for op in program.operations:
        if op.op=='virtual_pack_fp8':roots[op.results[0]]=recipe(program.nodes[op.results[0]],unit)
    return choices_from_roots(program, roots)


class PackMaterializationTest(unittest.TestCase):
    def test_hand_reference_enumerates_full_bounded_assignment_product(self):
        program=fixture(adjacent=True)
        rows=list(enumerate_hand(program))
        self.assertEqual(len(rows),10)
        self.assertEqual(len({row[2] for row in rows[1:]}),9)
        self.assertTrue(all(row[3] is None for row in rows))
        with self.assertRaisesRegex(ValueError,'site budget'):list(enumerate_hand(program,max_sites=1))

    def test_hand_reference_retains_negative_zero_relu(self):
        from atlas_egglog_ir import Node
        program=fixture(bits=0x8000);node=program.nodes['%a']
        relu=Node('virtual_vpu_unary',(node,),(('kind','"relu"'),),node.type)
        self.assertEqual(simplify(relu,program,analyze_program(program)),relu)

    def test_adjacent_conversions_form_one_state_chain(self):
        program=fixture(adjacent=True)
        rewritten=parse_program(materialize(program,selected(program)))
        state='%s'
        for op in rewritten.operations:
            if op.input_types and op.input_types[0]=='!atlas.virtual_state':
                self.assertEqual(op.operands[0],state)
            if op.output_types and op.output_types[0]=='!atlas.virtual_state':state=op.results[0]
        self.assertEqual(sum(op.op=='virtual_mxu_load_acc_bf16' for op in rewritten.operations),2)
        self.assertNotIn('virtual_pack_fp8', ''.join(rewritten.lines))

    def test_all_scale_recipes_have_exact_boundary_shape(self):
        for scale in range(256):
            program=fixture(scale=scale);node=recipe(program.nodes['%p'],0)
            if scale<=246:
                self.assertEqual(node.operands[1].operands[1].attribute('bits'),((scale+8)<<7)|96)
            else:self.assertEqual(node.operands[1],node.operands[0])

    def test_nonfinite_source_cannot_materialize(self):
        for bits in (0x7f80,0xff80,0x7fc0,0xffc1):
            program=fixture(bits=bits)
            with self.assertRaisesRegex(ValueError,'automatically finite'):materialize(program,selected(program))

    def test_forged_unclamped_value_is_refused(self):
        program=fixture(bits=0x7f7f);choice=selected(program)[0]
        with self.assertRaisesRegex(ValueError,'exact equality proof'):
            materialize(program,(replace(choice,clamped=choice.source),))

    def test_full_accumulator_unit_is_refused(self):
        program=fixture(live=True)
        with self.assertRaisesRegex(ValueError,'free accumulator'):materialize(program,selected(program))
        self.assertIn('virtual_mxu_readout_fp8',materialize(program,selected(program,1)))

    def test_changed_scale_and_duplicate_choices_are_refused(self):
        program=fixture();choice=selected(program)[0]
        with self.assertRaisesRegex(ValueError,'producer/scale'):materialize(program,(replace(choice,scale=128),))
        with self.assertRaisesRegex(ValueError,'duplicate'):materialize(program,(choice,choice))

    def test_analysis_must_be_bound_to_current_program(self):
        program=fixture();other=fixture(bits=0x7f80)
        with self.assertRaisesRegex(ValueError,'current automatically'):pack_rules(other,analyze_program(program))


class SelectionReceiptTest(unittest.TestCase):
    def test_metric_and_tier_are_not_interchangeable(self):
        row=dict(status='PASS',program_sha256='a'*64,cycles=[1],correctness=dict(outputs=True,oracle=True,guards=True),metric='cycles',tier='rtl')
        check_measurement(row,'a'*64,metric='cycles',tier='rtl')
        for change in ({'metric':'host_ns'},{'tier':'model'}):
            with self.assertRaises(ValueError):check_measurement(row|change,'a'*64,metric='cycles',tier='rtl')

    def test_unary_data_and_qualification_changes_invalidate_admission(self):
        from unittest.mock import patch
        import atlas_egglog_unary_tables as unary
        from explore_atlas_egglog import implementation_pins
        names=('exp.bin.gz','exp2.bin.gz','rcp.bin.gz','provenance.json','qualification.json')
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for name in names:(root/name).write_bytes((unary.DATA/name).read_bytes())
            with patch.object(unary,'DATA',root):
                pins=implementation_pins()
                self.assertTrue({str(root/name) for name in names} <= pins.keys())
                self.assertIn(str(Path(unary.__file__).resolve()),pins)
                proof=root/'proof.json'
                declaration=dict(schema='atlas.exact-egglog.v1',candidate_bindings={'c':dict(source_sha256='a'*64)},implementation_pins=pins)
                proof.write_text(json.dumps(declaration))
                row=dict(id='c',source_sha256='a'*64,proof=dict(path=str(proof),sha256=hashlib.sha256(proof.read_bytes()).hexdigest()))
                check_admission(row)
                for name in names:
                    path=root/name;original=path.read_bytes();path.write_bytes(original+b'changed')
                    with self.assertRaisesRegex(ValueError,'implementation identity'):check_admission(row)
                    path.write_bytes(original)
                (root/names[0]).unlink()
                with self.assertRaises(FileNotFoundError):implementation_pins()

    def test_candidate_and_provider_identity_are_bound(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);provider=root/'provider.py';provider.write_text('original provider')
            proof=root/'proof.json'
            declaration=dict(schema='atlas.exact-egglog.v1',candidate_bindings={'c':dict(source_sha256='a'*64)},implementation_pins={str(provider):hashlib.sha256(provider.read_bytes()).hexdigest()})
            proof.write_text(json.dumps(declaration))
            row=dict(id='c',source_sha256='a'*64,proof=dict(path=str(proof),sha256=hashlib.sha256(proof.read_bytes()).hexdigest()))
            check_admission(row)
            with self.assertRaisesRegex(ValueError,'bind emitted'):check_admission(row|{'source_sha256':'b'*64})
            provider.write_text('changed provider')
            with self.assertRaisesRegex(ValueError,'implementation identity'):check_admission(row)


if __name__=='__main__':unittest.main()
