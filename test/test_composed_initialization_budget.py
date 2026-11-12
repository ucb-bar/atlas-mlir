"""Default-bounded static initialization checks fail closed on exhaustion."""
import os
import re
from pathlib import Path
import subprocess
import unittest


def source(iterations=100001, *, initialized=True):
    write='''%next = "atlas.virtual_dma_store_bf16"(%body_state, %tile, %out, %bytes)
        : (!atlas.virtual_state, !atlas.virtual_bf16, i32, i32) -> !atlas.virtual_state''' if initialized else '%next = "atlas.virtual_nop"(%token) : (!atlas.virtual_state) -> !atlas.virtual_state'
    # Independent scalar facts deliberately exercise proof work. They do not
    # increase the machine loop's trip count or authorize any extra memory.
    facts='\n'.join(f'%unused{i} = arith.constant {i} : i32' for i in range(12))
    return f'''module {{
      func.func @proof() -> !atlas.virtual_state attributes {{
        atlas.input_dram_base = 2415919104 : i64,
        atlas.output_dram_base = 2415921152 : i64,
        atlas.output_dram_bytes = 4096 : i64}} {{
        %start = "atlas.virtual_start"() : () -> !atlas.virtual_state
        %input = arith.constant -1879048192 : i32
        %out = arith.constant -1879046144 : i32
        %end = arith.constant -1879044096 : i32
        %bytes = arith.constant 2048 : i32
        %loaded, %tile = "atlas.virtual_dma_load_bf16"(%start,%input,%bytes)
          : (!atlas.virtual_state,i32,i32) -> (!atlas.virtual_state,!atlas.virtual_bf16)
        %zero = arith.constant 0 : i32
        cf.br ^loop(%loaded,%zero : !atlas.virtual_state,i32)
      ^loop(%token: !atlas.virtual_state,%counter: i32):
        %count = arith.constant {iterations} : i32
        %more = arith.cmpi ult, %counter, %count : i32
        cf.cond_br %more, ^body(%token : !atlas.virtual_state), ^done(%token : !atlas.virtual_state)
      ^body(%body_state: !atlas.virtual_state):
        {facts}
        {write}
        %one = arith.constant 1 : i32
        %n = arith.addi %counter,%one : i32
        cf.br ^loop(%next,%n : !atlas.virtual_state,i32)
      ^done(%done_state: !atlas.virtual_state):
        %read, %value = "atlas.virtual_dma_load_bf16"(%done_state,%out,%bytes)
          : (!atlas.virtual_state,i32,i32) -> (!atlas.virtual_state,!atlas.virtual_bf16)
        %final = "atlas.virtual_dma_store_bf16"(%read,%value,%end,%bytes)
          : (!atlas.virtual_state,!atlas.virtual_bf16,i32,i32) -> !atlas.virtual_state
        return %final : !atlas.virtual_state
      }}
    }}'''


@unittest.skipUnless(os.environ.get('ATLAS_OOT_BIN_DIR'),'select Atlas tools')
class ComposedInitializationBudgetTest(unittest.TestCase):
    def compile(self,text):
        return subprocess.run([str(Path(os.environ['ATLAS_OOT_BIN_DIR'])/'atlas-opt'),
            '--atlas-compile-unscheduled','-'],input=text,text=True,capture_output=True)

    def test_static_region_within_default_work_budget_compiles(self):
        result=self.compile(source(iterations=1000))
        self.assertEqual(result.returncode,0,result.stderr)

    def test_static_region_above_default_work_budget_is_refused(self):
        result=self.compile(source())
        self.assertNotEqual(result.returncode,0)
        self.assertIn('scalar work limit',result.stderr)

    def test_uninitialized_output_read_is_still_refused(self):
        # Keep this control below every proof budget so it rejects the actual
        # memory error instead of exhausting the work allowance first.
        text=source(iterations=2).replace('%next = "atlas.virtual_dma_store_bf16"(%body_state, %tile, %out, %bytes)',
            '%next = "atlas.virtual_dma_store_bf16"(%body_state, %tile, %end, %bytes)')
        result=self.compile(text)
        self.assertNotEqual(result.returncode,0)
        self.assertIn('before they are initialized',result.stderr)

    def test_runtime_diamond_checks_both_initialization_paths(self):
        text = source(iterations=1).replace('@proof()', '@proof(%choose: i1)').replace(
            'atlas.input_dram_base =', 'atlas.control_dram_base = 2415925248 : i64, atlas.input_dram_base =')
        original = '\n'.join(['%next = "atlas.virtual_dma_store_bf16"(%body_state, %tile, %out, %bytes)',
            '        : (!atlas.virtual_state, !atlas.virtual_bf16, i32, i32) -> !atlas.virtual_state'])
        diamond = '''cf.cond_br %choose, ^left(%body_state : !atlas.virtual_state), ^right(%body_state : !atlas.virtual_state)
      ^left(%left_state: !atlas.virtual_state):
        %left_next = "atlas.virtual_dma_store_bf16"(%left_state, %tile, %out, %bytes)
          : (!atlas.virtual_state, !atlas.virtual_bf16, i32, i32) -> !atlas.virtual_state
        cf.br ^join(%left_next : !atlas.virtual_state)
      ^right(%right_state: !atlas.virtual_state):
        %right_next = "atlas.virtual_dma_store_bf16"(%right_state, %tile, %out, %bytes)
          : (!atlas.virtual_state, !atlas.virtual_bf16, i32, i32) -> !atlas.virtual_state
        cf.br ^join(%right_next : !atlas.virtual_state)
      ^join(%next: !atlas.virtual_state):'''
        text = text.replace(original, diamond)
        result = self.compile(text)
        self.assertEqual(result.returncode, 0, result.stderr)
        for arm in ('left', 'right'):
            with self.subTest(arm=arm):
                changed = text.replace(f'(%{arm}_state, %tile, %out, %bytes)',
                    f'(%{arm}_state, %tile, %end, %bytes)')
                result = self.compile(changed)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('before they are initialized', result.stderr)

    def test_repeated_scalar_state_does_not_prove_nonterminating_loop(self):
        text = source(iterations=2).replace('%n = arith.addi %counter,%one',
                                            '%n = arith.addi %counter,%zero')
        result = self.compile(text)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('scalar-state cycle', result.stderr)

    def test_dead_branch_facts_merge_without_discarding_initialization_paths(self):
        text = source(iterations=512).replace('@proof()', '@proof(%choose: i1)').replace(
            'atlas.input_dram_base =', 'atlas.control_dram_base = 2415925248 : i64, atlas.input_dram_base =')
        original = '\n'.join(['%next = "atlas.virtual_dma_store_bf16"(%body_state, %tile, %out, %bytes)',
            '        : (!atlas.virtual_state, !atlas.virtual_bf16, i32, i32) -> !atlas.virtual_state'])
        # Each branch leaves a different historical scalar value. Neither is
        # live after the join, so their histories must not multiply later paths.
        diamond = '''cf.cond_br %choose, ^left(%body_state : !atlas.virtual_state), ^right(%body_state : !atlas.virtual_state)
      ^left(%left_state: !atlas.virtual_state):
        %left_history = arith.addi %counter, %unused1 : i32
        %left_next = "atlas.virtual_dma_store_bf16"(%left_state, %tile, %out, %bytes)
          : (!atlas.virtual_state, !atlas.virtual_bf16, i32, i32) -> !atlas.virtual_state
        cf.br ^join(%left_next : !atlas.virtual_state)
      ^right(%right_state: !atlas.virtual_state):
        %right_history = arith.addi %counter, %unused2 : i32
        %right_next = "atlas.virtual_dma_store_bf16"(%right_state, %tile, %out, %bytes)
          : (!atlas.virtual_state, !atlas.virtual_bf16, i32, i32) -> !atlas.virtual_state
        cf.br ^join(%right_next : !atlas.virtual_state)
      ^join(%next: !atlas.virtual_state):'''
        text = text.replace(original, diamond)
        result = self.compile(text)
        self.assertEqual(result.returncode, 0, result.stderr)
        # A value used after a join is live even when defined in an outer block.
        # Both paths must initialize the bytes before the very first read.
        text = text.replace('%one = arith.constant 1 : i32', '''
        %checked, %check_tile = "atlas.virtual_dma_load_bf16"(%next, %out, %bytes)
          : (!atlas.virtual_state, i32, i32) -> (!atlas.virtual_state, !atlas.virtual_bf16)
        %one = arith.constant 1 : i32''').replace('cf.br ^loop(%next,%n', 'cf.br ^loop(%checked,%n')
        for arm in ('left', 'right'):
            with self.subTest(arm=arm):
                changed = text.replace(f'(%{arm}_state, %tile, %out, %bytes)',
                    f'(%{arm}_state, %tile, %end, %bytes)')
                result = self.compile(changed)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('before they are initialized', result.stderr)

    def test_captured_source_replay_rejects_nonterminating_scalar_state(self):
        # First publish a terminating loop, then change its increment in both
        # retained SSA and the final owned physical addition. Immediate folding
        # can remove the increment constant's standalone machine instruction.
        compiled = self.compile(source(iterations=2))
        self.assertEqual(compiled.returncode, 0, compiled.stderr)
        machine = compiled.stdout
        additions = re.findall(r'\{[^{}]*def = "arith.addi"[^{}]*\}', machine)
        self.assertEqual(len(additions), 1)
        addition = additions[0]
        result_id = int(re.search(r'\bid = (\d+) : i32', addition).group(1))
        increment = int(re.search(r'operands = array<i32: \d+, (\d+)>', addition).group(1))
        values = [record for record in re.findall(r'\{[^{}]*def = "arith.constant"[^{}]*\}', machine)
                  if re.search(rf'\bid = {increment} : i32\b', record)]
        self.assertEqual(len(values), 1)
        value = values[0]
        self.assertIn('constant = 1 : i32', value)
        operations = [record for record in re.findall(r'\{[^{}]*name = "arith.addi"[^{}]*\}', machine)
                      if re.search(rf'results = array<i32: {result_id}>', record)]
        self.assertEqual(len(operations), 1)
        operation_id = int(re.search(r'\bid = (\d+) : i32', operations[0]).group(1))
        instructions = [line for line in machine.splitlines()
                        if re.search(rf'atlas.virtual_cfg_source = {operation_id} : i32\b', line)
                        and re.search(rf'atlas.virtual_scalar_result = {result_id} : i32\b', line)]
        self.assertEqual(len(instructions), 1)
        instruction = instructions[0]
        self.assertIn('"atlas.alu_imm"', instruction)
        self.assertIn('kind = "addi"', instruction)
        self.assertIn('immediate = 1 : i32', instruction)
        self.assertNotIn('src = 0 : i32', instruction)
        machine = machine.replace(value, value.replace('constant = 1 : i32', 'constant = 0 : i32'), 1)
        machine = machine.replace(instruction, instruction.replace('immediate = 1 : i32', 'immediate = 0 : i32'), 1)
        result = subprocess.run([str(Path(os.environ['ATLAS_OOT_BIN_DIR'])/'atlas-opt'),
             '--verify-atlas-generated-schedule','-'],input=machine,text=True,capture_output=True)
        self.assertNotEqual(result.returncode,0)
        self.assertIn('does not prove output reads initialized',result.stderr)


if __name__=='__main__':unittest.main()
