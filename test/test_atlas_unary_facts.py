"""Exact safe-domain unary transfers; raw table evidence, no IEEE premise."""
import struct
import sys
from pathlib import Path
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from atlas_egglog_facts import (RealFacts, constant, from_interval, infer_facts,
                               ordered, raw_bits, analyze_program)
from atlas_egglog_unary_tables import PINS, qualified_image, image, table_bytes


class UnaryFactsTests(unittest.TestCase):
    def test_domains_signedzero_and_narrow_image(self):
        for kind, (interval, _, _) in PINS.items():
            facts = infer_facts('vpu_unary', (from_interval(*interval),), {'kind':kind})
            self.assertTrue(facts.unit_interval)
            self.assertEqual(facts.interval, (0xbd00,0xbf80) if kind=='recip' else (0x8000,0xbf80))
            table=struct.unpack('<65536H',table_bytes(kind))
            for lo,hi in ((interval[0],interval[0]),(interval[1],interval[1]),
                          (interval[0],interval[0]+4),(interval[1]-4,interval[1])):
                values=[ordered(table[raw_bits(k)]) for k in range(lo,hi+1)]
                result=infer_facts('vpu_unary',(from_interval(lo,hi),),{'kind':kind})
                self.assertEqual(result.interval,(min(values),max(values)))
                self.assertTrue(any('unary_table:'+kind in p for p in result.provenance))
            self.assertFalse(infer_facts('vpu_unary',(RealFacts(),),{'kind':kind}).finite)
            for interval_out in ((interval[0]-1,interval[1]),(interval[0],interval[1]+1)):
                self.assertFalse(infer_facts('vpu_unary',(from_interval(*interval_out),),{'kind':kind}).finite)
        for kind in ('exp','exp2'):
            for bits in (0,0x8000):
                self.assertEqual(infer_facts('vpu_unary',(constant(bits),),{'kind':kind}).interval,constant(0x3f80).interval)

    def test_wrapped_finite_and_nonfinite_outside_domains_are_refused(self):
        for kind,bits in (('exp',0xc3a8),('exp',0xc400),('exp2',0xc381),
                          ('exp2',0xc400),('recip',0),('recip',1),
                          ('recip',0x7f7f),('recip',0x7f80),('recip',0x7fc0)):
            self.assertIsNone(qualified_image(kind,constant(bits).interval))
            self.assertFalse(infer_facts('vpu_unary',(constant(bits),),{'kind':kind}).finite)

    def test_nonmonotonic_images_use_interior(self):
        for kind in PINS:
            table=struct.unpack('<65536H',table_bytes(kind))
            lo,hi=PINS[kind][0]
            # Full domain independently enumerated, including any interior extrema.
            values=[ordered(table[raw_bits(k)]) for k in range(lo,hi+1)]
            self.assertEqual(image(kind,(lo,hi)),(min(values),max(values)))

    def test_changed_artifact_refuses_even_with_populated_cache(self):
        self.assertIsNotNone(qualified_image('exp',(0x7fff,0x8000)))
        with patch.object(Path,'read_bytes',return_value=b'wrong table'):
            with self.assertRaisesRegex(ValueError,'identity mismatch'):
                qualified_image('exp',(0x7fff,0x8000))

    def test_unary_cannot_reuse_opted_in_contract_cache(self):
        from atlas_egglog_ir import Node
        from atlas_egglog_facts import Contract, node_digest, analyze_node
        anchor=Node('anchor',(),(('id','%x'),),'!atlas.virtual_bf16')
        unary=Node('virtual_vpu_unary',(anchor,),(('kind','"exp"'),),'!atlas.virtual_bf16')
        digest=node_digest(anchor)
        safe={digest:Contract(digest,PINS['exp'][0],'explicit-local-premise')}
        memo={}
        self.assertTrue(analyze_node(unary,contracts=safe,accept_contracts=True,memo=memo).unit_interval)
        self.assertFalse(analyze_node(unary,memo=memo).finite)
        changed={digest:Contract(digest,constant(0xc3a8).interval,'out-of-domain')}
        self.assertFalse(analyze_node(unary,contracts=changed,accept_contracts=True,memo=memo).finite)

    def test_source_derived_clamp_without_contract(self):
        from atlas_egglog_ir import parse_program
        text='''module {
  func.func @derived() -> !atlas.virtual_state {
    %s0 = "atlas.virtual_start"() : () -> !atlas.virtual_state
    %s1, %x = "atlas.virtual_input_bf16"(%s0) {index = 0 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_bf16)
    %lo = "atlas.virtual_vpu_splat"() {bits = 50087 : i32} : () -> !atlas.virtual_bf16
    %hi = "atlas.virtual_vpu_splat"() {bits = 0 : i32} : () -> !atlas.virtual_bf16
    %a = "atlas.virtual_vpu_binary"(%x, %lo) {kind = "max"} : (!atlas.virtual_bf16, !atlas.virtual_bf16) -> !atlas.virtual_bf16
    %b = "atlas.virtual_vpu_binary"(%a, %hi) {kind = "min"} : (!atlas.virtual_bf16, !atlas.virtual_bf16) -> !atlas.virtual_bf16
    %y = "atlas.virtual_vpu_unary"(%b) {kind = "exp"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16
    %p = "atlas.virtual_pack_fp8"(%y) {scale_code = 127 : i32} : (!atlas.virtual_bf16) -> !atlas.virtual_fp8
    return %s1 : !atlas.virtual_state
  }
}
'''
        f=analyze_program(parse_program(text))
        self.assertFalse(f.values['%x'].finite)
        self.assertTrue(f.values['%y'].unit_interval)
        self.assertFalse(any('caller_contract' in p for p in f.values['%y'].provenance))
        # Same SSA spelling in a new source does not inherit the clamp facts.
        other=text.replace('%y = "atlas.virtual_vpu_unary"(%b)','%y = "atlas.virtual_vpu_unary"(%x)')
        self.assertFalse(analyze_program(parse_program(other)).values['%y'].finite)

if __name__=='__main__':unittest.main()
