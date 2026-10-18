"""Actual egglog firing and refusal tests, with consumed extraction roots."""
import sys
from pathlib import Path
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from egglog import EGraph, eq
from atlas_egglog_ir import Node
from atlas_egglog_schema import encode_node, decode_node
from atlas_egglog_rules import EXACT_RULES, bounded_constant_sharing, register_facts


def unary(kind,x):
    return Node('virtual_vpu_unary',(x,),(('kind',f'"{kind}"'),))


def binary(kind,x,y):
    return Node('virtual_vpu_binary',(x,y),(('kind',f'"{kind}"'),))


def splat(bits):
    return Node('virtual_vpu_splat',attrs=(('bits',f'{bits} : i32'),))


class RuleTests(unittest.TestCase):
    def run_rule(self,root,want,*,fires=True):
        graph=EGraph()
        value=encode_node(root)
        graph.register(value)
        register_facts(graph,(root,))
        graph.run(EXACT_RULES.saturate())
        if fires:
            graph.check(eq(value).to(encode_node(want)))
            self.assertEqual(decode_node(graph.extract(value)),want)
        else:
            graph.check_fail(eq(value).to(encode_node(want)))

    def test_copy_and_layout_involution(self):
        x=Node('anchor',attrs=(('id','%unknown'),))
        self.run_rule(unary('mov',x),x)
        x=Node('anchor',attrs=(('id','%fp8'),),type='!atlas.virtual_fp8')
        t=lambda n:Node('virtual_xlu_transpose',(n,),type='!atlas.virtual_fp8')
        self.run_rule(t(t(x)),x)
        self.run_rule(t(x),x,fires=False)

    def test_raw_minmax_idempotence_and_absorption(self):
        # Unknown anchors include arbitrary NaNs and signed-zero encodings.
        x=Node('anchor',attrs=(('id','%x'),))
        y=Node('anchor',attrs=(('id','%y'),))
        for kind,dual in (('min','max'),('max','min')):
            self.run_rule(binary(kind,x,x),x)
            self.run_rule(binary(kind,x,binary(dual,x,y)),x)
            self.run_rule(binary(kind,binary(dual,x,y),x),x)
        self.run_rule(binary('add',x,splat(0)),x,fires=False)

    def test_ordered_bounds_and_special_values(self):
        for a,b in ((0x8000,0),(0xFF80,0x7F80),(0xFFC1,0x7FC1),(0xBF80,0x3F80)):
            self.run_rule(binary('min',splat(a),splat(b)),splat(a))
            self.run_rule(binary('max',splat(a),splat(b)),splat(b))
        x=Node('anchor',attrs=(('id','%x'),))
        self.run_rule(binary('min',x,splat(0x3F80)),x,fires=False)

    def test_relu_guard_and_negative_zero_refusal(self):
        for bits in (0,1,0x3F80,0x7F7F):
            self.run_rule(unary('relu',splat(bits)),splat(bits))
        for bits in (0x8000,0xBF80,0x7F80,0x7FC0,0xFFC0):
            self.run_rule(unary('relu',splat(bits)),splat(bits),fires=False)
        self.run_rule(unary('relu',unary('relu',splat(0xBF80))),unary('relu',splat(0xBF80)))
        x=Node('anchor',attrs=(('id','%unknown'),))
        self.run_rule(unary('relu',x),x,fires=False)

    def test_program_fact_identity_and_contract_admission(self):
        from atlas_egglog_ir import parse_program
        from atlas_egglog_facts import Contract, analyze_program, node_digest, ordered
        a=parse_program('''module {
  func.func @a() -> !atlas.virtual_state {
    %s0 = "atlas.virtual_start"() : () -> !atlas.virtual_state
    %c = "atlas.virtual_vpu_splat"() {bits = 16256 : i32} : () -> !atlas.virtual_bf16
    %s1, %acc = "atlas.virtual_mxu_load_acc_bf16"(%s0, %c) {unit = 0 : i32} : (!atlas.virtual_state, !atlas.virtual_bf16) -> (!atlas.virtual_state, !atlas.virtual_mxu_acc<0>)
    %s2, %y = "atlas.virtual_mxu_readout_bf16"(%s1, %acc) : (!atlas.virtual_state, !atlas.virtual_mxu_acc<0>) -> (!atlas.virtual_state, !atlas.virtual_bf16)
    return %s2 : !atlas.virtual_state
  }
}
''')
        b=parse_program('''module {
  func.func @b() -> !atlas.virtual_state {
    %s0 = "atlas.virtual_start"() : () -> !atlas.virtual_state
    %s1, %y = "atlas.virtual_input_bf16"(%s0) {index = 0 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_bf16)
    return %s1 : !atlas.virtual_state
  }
}
''')
        root=unary('relu',b.nodes['%y'])
        graph=EGraph();graph.register(encode_node(root))
        with self.assertRaises(ValueError):
            register_facts(graph,(root,),program_facts=analyze_program(a),program=b)
        with self.assertRaises(ValueError):
            register_facts(graph,(root,),program_facts=analyze_program(a))
        register_facts(graph,(root,),program_facts=analyze_program(b),program=b)
        graph.run(EXACT_RULES.saturate())
        graph.check_fail(eq(encode_node(root)).to(encode_node(b.nodes['%y'])))
        digest=node_digest(b.nodes['%y'])
        contracts={digest:Contract(digest,(ordered(0),ordered(0x3F80)),'caller',program_sha256=b.sha256)}
        assumed=analyze_program(b,contracts=contracts,accept_contracts=True)
        with self.assertRaises(ValueError):
            register_facts(EGraph(),(root,),program_facts=assumed,program=b)
        graph=EGraph();graph.register(encode_node(root))
        register_facts(graph,(root,),program_facts=assumed,program=b,contracts=contracts,accept_contracts=True)
        graph.run(EXACT_RULES.saturate())
        graph.check(eq(encode_node(root)).to(encode_node(b.nodes['%y'])))
        # Unreachable finite readout %y must not be exported for an unrelated root.
        memo=register_facts(EGraph(),(a.nodes['%c'],),program_facts=analyze_program(a),program=a)
        self.assertNotIn(a.nodes['%y'],memo)

    def test_constant_sharing_pressure_bound(self):
        uses={'a':(0,2,6,8),'b':(1,3,7,9)}
        plan=bounded_constant_sharing(uses,max_span=3,max_live_shared=1)
        for point in range(10):
            self.assertLessEqual(sum(g.uses[0]<=point<g.uses[-1] for g in plan),1)
        self.assertTrue(all(g.uses[-1]-g.uses[0]<=3 for g in plan))
        self.assertTrue(all(len(g.uses)==1 for g in bounded_constant_sharing(uses,max_live_shared=0)))
        self.assertEqual(sorted((g.constant,p) for g in plan for p in g.uses), sorted((n,p) for n,ps in uses.items() for p in ps))
        with self.assertRaises(ValueError):
            bounded_constant_sharing(uses,max_span=-1)


if __name__ == '__main__':
    unittest.main()
