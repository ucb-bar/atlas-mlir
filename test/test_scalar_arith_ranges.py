"""Generic scalar shift admission requires a source-proven unsigned range."""
import unittest
from test_virtual_ssa import run

S='!atlas.virtual_state'
T='!atlas.virtual_bf16'

def source(body):
    return f'''module {{
  func.func @scalar(%n: i32) -> {S} attributes {{
    atlas.input_dram_base = 2415919104 : i64,
    atlas.output_dram_base = 2415984640 : i64,
    atlas.control_dram_base = 2416050176 : i64}} {{
    %state = "atlas.virtual_start"() : () -> {S}
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    {body}
    %tile = "atlas.virtual_vpu_splat"() {{bits = 0 : i32}} : () -> {T}
    %done = "atlas.virtual_output_bf16"(%state,%tile) {{index = 0 : i32}} : ({S},{T}) -> {S}
    return %done : {S}
  }}
}}'''

class ScalarArithmeticRangesTest(unittest.TestCase):
    def check(self, body, accepted=True):
        result=run('atlas-opt',source(body),'--verify-atlas-virtual-stream')
        self.assertEqual(result.returncode == 0,accepted,result.stderr)
        if not accepted:self.assertIn('proven unsigned range below32',result.stderr)

    def test_constant_boundaries(self):
        for count in (0,1,31,32,-1):
            for op in ('shli','shrui'):
                with self.subTest(count=count,op=op):
                    self.check(f'%distance = arith.constant {count} : i32\n%shift = arith.{op} %one,%distance : i32',0<=count<32)

    def test_unknown_or_wide_distance_is_refused(self):
        self.check('%shift = arith.shli %one,%n : i32',False)
        self.check('%mask = arith.constant 63 : i32\n%distance = arith.andi %n,%mask : i32\n%shift = arith.shrui %one,%distance : i32',False)

    def test_masked_variable_distance_is_accepted(self):
        self.check('%mask = arith.constant 31 : i32\n%distance = arith.andi %n,%mask : i32\n%shift = arith.shli %one,%distance : i32')

    def test_masked_select_clamp_is_proven(self):
        self.check('''%limit = arith.constant 24 : i32
          %test = arith.cmpi ult,%n,%limit : i32
          %predicate = arith.extui %test : i1 to i32
          %mask = arith.subi %zero,%predicate : i32
          %delta = arith.xori %n,%limit : i32
          %chosen = arith.andi %delta,%mask : i32
          %distance = arith.xori %limit,%chosen : i32
          %shift = arith.shrui %one,%distance : i32''')

    def test_masked_shift_lowers_through_standard_contracts(self):
        text=source('%mask = arith.constant 31 : i32'+'\n%distance = arith.andi %n,%mask : i32\n%shift = arith.shli %one,%distance : i32')
        result=run('atlas-opt',text,'--atlas-compile-schedule','--atlas-compile-unscheduled')
        self.assertEqual(result.returncode,0,result.stderr)
        for name in ('dma','mxu','tile','cfg','source_memory','buffer'):
            self.assertIn('atlas.virtual_'+name+'_contract',result.stdout)

    def test_counted_counter_distance_is_proven(self):
        body='''%limit = arith.constant 32 : i32
          cf.br ^loop(%state,%zero : !atlas.virtual_state,i32)
        ^loop(%token: !atlas.virtual_state,%index: i32):
          %more = arith.cmpi ult,%index,%limit : i32
          cf.cond_br %more,^body(%token : !atlas.virtual_state),^exit(%token : !atlas.virtual_state)
        ^body(%body_state: !atlas.virtual_state):
          %shift = arith.shli %one,%index : i32
          %next = arith.addi %index,%one : i32
          cf.br ^loop(%body_state,%next : !atlas.virtual_state,i32)
        ^exit(%exit_state: !atlas.virtual_state):
          %tile = "atlas.virtual_vpu_splat"() {bits = 0 : i32} : () -> !atlas.virtual_bf16
          %done = "atlas.virtual_output_bf16"(%exit_state,%tile) {index = 0 : i32} : (!atlas.virtual_state,!atlas.virtual_bf16) -> !atlas.virtual_state
          return %done : !atlas.virtual_state'''
        text=source('').split('    %tile =')[0]+body+'\n  }\n}'
        for flags in (('--verify-atlas-virtual-stream',),('--atlas-compile-schedule','--atlas-compile-unscheduled')):
            result=run('atlas-opt',text,*flags)
            self.assertEqual(result.returncode,0,result.stderr)

    def test_captured_mask_widening_rejected(self):
        for op in ('shli','shrui'):
            with self.subTest(op=op):
                text=source(f'%mask = arith.constant 31 : i32\n%distance = arith.andi %n,%mask : i32\n%shift = arith.{op} %n,%distance : i32')
                result=run('atlas-opt',text,'--atlas-compile-schedule','--atlas-compile-unscheduled')
                self.assertEqual(result.returncode,0,result.stderr)
                changed=result.stdout.replace('immediate = 31 : i32','immediate = 63 : i32').replace('constant = 31 : i32','constant = 63 : i32')
                self.assertNotEqual(changed,result.stdout)
                checked=run('atlas-opt',changed,'--verify-atlas-generated-schedule')
                self.assertNotEqual(checked.returncode,0)
                self.assertIn('retained scalar shift distance',checked.stderr)

    def test_captured_constant_overshift_rejected(self):
        for op in ('shli','shrui'):
            for distance in (32,-1):
                with self.subTest(op=op,distance=distance):
                    text=source(f'%distance = arith.constant 31 : i32\n%shift = arith.{op} %n,%distance : i32')
                    result=run('atlas-opt',text,'--atlas-compile-schedule','--atlas-compile-unscheduled')
                    self.assertEqual(result.returncode,0,result.stderr)
                    changed=result.stdout.replace('immediate = 31 : i32',f'immediate = {distance} : i32').replace('constant = 31 : i32',f'constant = {distance} : i32')
                    self.assertNotEqual(changed,result.stdout)
                    checked=run('atlas-opt',changed,'--verify-atlas-generated-schedule')
                    self.assertNotEqual(checked.returncode,0)

    def test_captured_masked_clamp_is_proven_and_widening_rejected(self):
        text=source('''%limit = arith.constant 24 : i32
          %test = arith.cmpi ult,%n,%limit : i32
          %predicate = arith.extui %test : i1 to i32
          %mask = arith.subi %zero,%predicate : i32
          %delta = arith.xori %n,%limit : i32
          %chosen = arith.andi %delta,%mask : i32
          %distance = arith.xori %limit,%chosen : i32
          %shift = arith.shrui %n,%distance : i32''')
        result=run('atlas-opt',text,'--atlas-compile-schedule','--atlas-compile-unscheduled')
        self.assertEqual(result.returncode,0,result.stderr)
        changed=result.stdout.replace('immediate = 24 : i32','immediate = 32 : i32').replace('constant = 24 : i32','constant = 32 : i32')
        self.assertNotEqual(changed,result.stdout)
        checked=run('atlas-opt',changed,'--verify-atlas-generated-schedule')
        self.assertNotEqual(checked.returncode,0)
        self.assertIn('retained scalar shift distance',checked.stderr)

    def test_i1_logical_operations_remain_admitted(self):
        self.check('''%a = arith.cmpi ult,%n,%one : i32
          %b = arith.cmpi eq,%n,%zero : i32
          %and = arith.andi %a,%b : i1
          %or = arith.ori %a,%b : i1
          %xor = arith.xori %a,%b : i1
          %word = arith.extui %xor : i1 to i32''')

if __name__=='__main__':unittest.main()
