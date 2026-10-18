"""Directed and bounded exhaustive soundness tests for Atlas fact transfers."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from atlas_egglog_facts import (Contract, RealFacts, analyze_node, analyze_program, constant,
                               finite_value, from_interval, infer_facts, join,
                               node_digest, ordered, raw_bits)


class FactTests(unittest.TestCase):
    def test_order_bijection_and_constant_classification_exhaustive(self):
        for bits in range(65536):
            self.assertEqual(raw_bits(ordered(bits)), bits)
            f = constant(bits)
            value = finite_value(bits)
            self.assertEqual(f.finite, value is not None)
            self.assertEqual(f.integer_valued, value is not None and value.denominator == 1)
            self.assertEqual(f.nonnegative, value is not None and value >= 0)
            self.assertEqual(f.raw_nonnegative, value is not None and not bits & 0x8000)
            self.assertEqual(f.exact_zero is not None, bits in (0, 0x8000))

    def test_directed_rejections(self):
        finite = constant(0x7F7F)
        zero = constant(0)
        for op, attrs, operands in [
            ('vpu_binary', {'kind':'add'}, (finite,finite)),
            ('vpu_binary', {'kind':'mul'}, (finite,finite)),
            ('vpu_unary', {'kind':'exp'}, (finite,)),
            ('vpu_unary', {'kind':'recip'}, (zero,)),
            ('mxu_matmul', {'unit':0}, (finite,finite)),
            ('mxu_readout_bf16', {}, (finite,)),
            ('anchor', {}, ())]:
            self.assertFalse(infer_facts(op, operands, attrs).finite)
        for bits in (0x7F80,0xFF80,0x7FC0,0xFFC0):
            self.assertFalse(infer_facts('vpu_unary',(constant(bits),),{'kind':'relu'}).finite)
        self.assertFalse(RealFacts().integer_valued)

    def test_clamp_unknown_including_nans(self):
        lower = infer_facts("vpu_binary",(RealFacts(),constant(0)),{"kind":"max"})
        self.assertFalse(lower.finite)
        clamp = infer_facts("vpu_binary",(lower,constant(0x3F80)),{"kind":"min"})
        self.assertTrue(clamp.unit_interval)
        for bits in range(65536):
            result = min(max(ordered(bits),ordered(0)),ordered(0x3F80))
            self.assertTrue(clamp.interval[0] <= result <= clamp.interval[1])

    def test_ranges_zero_join_and_scale(self):
        self.assertEqual(join(constant(0), constant(0x8000)).exact_zero,'either')
        self.assertEqual(constant(0x8000).exact_zero,'negative')
        self.assertEqual(constant(0).exact_zero,'positive')
        self.assertTrue(constant(0x8000).nonnegative)
        self.assertFalse(constant(0x8000).raw_nonnegative)
        self.assertTrue(from_interval(ordered(0),ordered(0x3F80)).unit_interval)
        self.assertFalse(join(constant(0),RealFacts()).finite)
        self.assertEqual(infer_facts('scale_constant',attrs={'code':'255 : i32'}).scale_code,255)
        for scale in range(256):
            f = infer_facts('unpack_fp8',attrs={'scale_code':scale})
            self.assertTrue(f.finite)
            self.assertEqual(f.scale_code,scale)
            self.assertFalse(infer_facts('pack_fp8',(constant(0),),{'scale_code':scale}).finite)
        with self.assertRaises(ValueError):
            RealFacts(integer_valued=True)
        with self.assertRaises(ValueError):
            RealFacts((9,8))

    def test_bounded_raw_interval_transfers_exhaustive(self):
        # Exhaust all pairs in raw windows around -0/+0, finite overflow and
        # both NaN boundaries. These are raw ordered min/max, never IEEE min.
        windows = ((0,20),(0x70,0x90),(0x7FF0,0x8010),(0xFF70,0xFF90),(0xFFE0,0xFFFF))
        for lo,hi in windows:
            for a in range(lo,hi+1):
                for b in range(a,hi+1):
                    left = from_interval(a,b)
                    for c,d in ((lo,hi),(a,a),(b,b)):
                        right = from_interval(c,d)
                        for kind,choose in (('min',min),('max',max)):
                            f = infer_facts('vpu_binary',(left,right),{'kind':kind})
                            self.assertEqual(f.interval,(choose(a,c),choose(b,d)))
                            for x,y in ((a,c),(a,d),(b,c),(b,d)):
                                result = choose(x,y)
                                self.assertTrue(f.interval[0] <= result <= f.interval[1])
                    if left.finite:
                        f = infer_facts('vpu_unary',(left,),{'kind':'relu'})
                        self.assertEqual(f.interval,(max(a,0x8000),max(b,0x8000)))
                        self.assertTrue(f.raw_nonnegative)

    def test_program_resource_facts_and_accumulation_refusal(self):
        from atlas_egglog_ir import parse_program
        for unit in (0,1):
            head = f'''module {{
  func.func @proof() -> !atlas.virtual_state {{
    %s0 = "atlas.virtual_start"() : () -> !atlas.virtual_state
    %s1, %x = "atlas.virtual_input_fp8"(%s0) {{index = 0 : i32}} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %s2, %w = "atlas.virtual_input_fp8"(%s1) {{index = 1 : i32}} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %s3, %weight = "atlas.virtual_mxu_load_weight"(%s2, %w) {{unit = {unit} : i32}} : (!atlas.virtual_state, !atlas.virtual_fp8) -> (!atlas.virtual_state, !atlas.virtual_mxu_weight<{unit}>)
'''
            reset = f'''    %s4, %acc = "atlas.virtual_mxu_reset"(%s3, %x, %weight) : (!atlas.virtual_state, !atlas.virtual_fp8, !atlas.virtual_mxu_weight<{unit}>) -> (!atlas.virtual_state, !atlas.virtual_mxu_acc<{unit}>)
'''
            readout = lambda state,acc: f'''    %s6, %y = "atlas.virtual_mxu_readout_bf16"({state}, {acc}) : (!atlas.virtual_state, !atlas.virtual_mxu_acc<{unit}>) -> (!atlas.virtual_state, !atlas.virtual_bf16)
    return %s6 : !atlas.virtual_state
  }}
}}
'''
            program = parse_program(head+reset+readout('%s4','%acc'))
            result = analyze_program(program)
            self.assertTrue(result.values['%y'].finite)
            self.assertEqual(result.values['%y'].interval,(ordered(0xCC00),ordered(0x4C00)))
            self.assertIn('raw_fp8_32term_reset_abs_lt_2pow25',result.values['%y'].provenance)
            accumulated = f'''    %s5, %acc1 = "atlas.virtual_mxu_accumulate"(%s4, %x, %weight, %acc) : (!atlas.virtual_state, !atlas.virtual_fp8, !atlas.virtual_mxu_weight<{unit}>, !atlas.virtual_mxu_acc<{unit}>) -> (!atlas.virtual_state, !atlas.virtual_mxu_acc<{unit}>)
'''
            result = analyze_program(parse_program(head+reset+accumulated+readout('%s5','%acc1')))
            self.assertFalse(result.values['%y'].finite)
            # A raw FP8 accumulator copy has a different tighter domain.
            load = f'''    %s4, %acc = "atlas.virtual_mxu_load_acc_fp8"(%s3, %x) {{unit = {unit} : i32}} : (!atlas.virtual_state, !atlas.virtual_fp8) -> (!atlas.virtual_state, !atlas.virtual_mxu_acc<{unit}>)
'''
            result = analyze_program(parse_program(head+load+readout('%s4','%acc')))
            self.assertEqual(result.values['%y'].interval,(ordered(0xC3E0),ordered(0x43E0)))
            for bits in (0,0x8000,0x3F80,0x7F7F,0x7FC0):
                literal=f'''    %bf = "atlas.virtual_vpu_splat"() {{bits = {bits} : i32}} : () -> !atlas.virtual_bf16
    %s4, %acc = "atlas.virtual_mxu_load_acc_bf16"(%s3, %bf) {{unit = {unit} : i32}} : (!atlas.virtual_state, !atlas.virtual_bf16) -> (!atlas.virtual_state, !atlas.virtual_mxu_acc<{unit}>)
'''
                result=analyze_program(parse_program(head+literal+readout('%s4','%acc')))
                self.assertEqual(result.values['%y'].interval,constant(bits).interval)

    def test_memo_cannot_reuse_contract_premises(self):
        from atlas_egglog_ir import Node, parse_program
        node = Node('anchor',attrs=(('id','%source'),))
        digest = node_digest(node)
        contracts = {digest:Contract(digest,(ordered(0),ordered(0x3F80)),'finite')}
        memo = {}
        self.assertTrue(analyze_node(node,contracts=contracts,accept_contracts=True,memo=memo).finite)
        self.assertFalse(analyze_node(node,memo=memo).finite)
        # Changing a premise recomputes instead of reading a stale node entry.
        changed = {digest:Contract(digest,(ordered(0x7F80),ordered(0x7F80)),'infinity')}
        self.assertFalse(analyze_node(node,contracts=changed,accept_contracts=True,memo=memo).finite)
        text='''module {
  func.func @context() -> !atlas.virtual_state {
    %s0 = "atlas.virtual_start"() : () -> !atlas.virtual_state
    %s1, %source = "atlas.virtual_input_bf16"(%s0) {index = 0 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_bf16)
    return %s1 : !atlas.virtual_state
  }
}
'''
        program=parse_program(text)
        # Standalone lexical contracts cannot be ingested by a Program.
        with self.assertRaises(ValueError):
            analyze_program(program,contracts=contracts,accept_contracts=True)
        contracts={digest:Contract(digest,(ordered(0),ordered(0x3F80)),"finite",program_sha256=program.sha256)}
        seeds=analyze_program(program,contracts=contracts,accept_contracts=True)
        self.assertTrue(analyze_node(node,contracts=contracts,accept_contracts=True,program_facts=seeds,program=program).finite)
        with self.assertRaises(ValueError):
            analyze_node(node,program_facts=seeds,program=program)
        with self.assertRaises(ValueError):
            analyze_node(node,contracts=changed,accept_contracts=True,program_facts=seeds,program=program)
        other=parse_program(text.replace('@context()', '@different_context()'))
        with self.assertRaises(ValueError):
            analyze_program(other,contracts=contracts,accept_contracts=True)
        self.assertFalse(analyze_program(other,contracts=contracts).values['%source'].finite)
        with self.assertRaises(ValueError):
            analyze_node(node,contracts=contracts,accept_contracts=True,program_facts=seeds,program=other)
        with self.assertRaises(ValueError):
            analyze_node(node,contracts=contracts,accept_contracts=True,program_facts=seeds)

    def test_opt_in_contracts_pinned_to_expression(self):
        from atlas_egglog_ir import Node
        node = Node('anchor',attrs=(('id','%source'),))
        digest = node_digest(node)
        contract = Contract(digest,(ordered(0),ordered(0x3F80)),'probability')
        self.assertFalse(analyze_node(node,contracts={digest:contract}).finite)
        f = analyze_node(node,contracts={digest:contract},accept_contracts=True)
        self.assertTrue(f.unit_interval)
        self.assertIn('caller_contract:probability:',f.provenance[-1])
        bad = Contract('0'*64,contract.interval,'bad')
        with self.assertRaises(ValueError):
            analyze_node(node,contracts={digest:bad},accept_contracts=True)
        literal = Node('virtual_vpu_splat',attrs=(('bits','16256 : i32'),))
        digest = node_digest(literal)
        with self.assertRaises(ValueError):
            analyze_node(literal,contracts={digest:Contract(digest,(ordered(0),ordered(0)),'wrong')},accept_contracts=True)


if __name__ == '__main__':
    unittest.main()
