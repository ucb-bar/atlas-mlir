"""RV32 literal folding preserves arithmetic, allocation and source proofs."""
import re
import unittest

from test_virtual_spilling import BF16, STATE, compile_program, execute_copies, run, wrap


def signed(value):
    value &= 0xffffffff
    return value if value < 2**31 else value-2**32


def selection_source(lines, condition):
    return wrap([
        f'%s0 = "atlas.virtual_start"() : () -> {STATE}',
        f'%s1,%a = "atlas.virtual_input_bf16"(%s0) {{index=0:i32}} : ({STATE}) -> ({STATE},{BF16})',
        f'%s2,%b = "atlas.virtual_input_bf16"(%s1) {{index=1:i32}} : ({STATE}) -> ({STATE},{BF16})',
        *lines,
        f'cf.cond_br {condition},^true(%s2 : {STATE}),^false(%s2 : {STATE})',
        f'^true(%ts: {STATE}):', f'cf.br ^end(%ts,%b : {STATE},{BF16})',
        f'^false(%fs: {STATE}):', f'cf.br ^end(%fs,%a : {STATE},{BF16})',
        f'^end(%es: {STATE},%value: {BF16}):',
        f'%done = "atlas.virtual_output_bf16"(%es,%value) {{index=0:i32}} : ({STATE},{BF16}) -> {STATE}',
        f'return %done : {STATE}',
    ])


def binary_source(operation, value, literal, reverse=False):
    functions = {'addi': lambda a,b: a+b, 'andi': lambda a,b: a&b,
                 'ori': lambda a,b: a|b, 'xori': lambda a,b: a^b,
                 'shli': lambda a,b: a<<b, 'shrui': lambda a,b: a>>b}
    lhs, rhs = (literal & 0xffffffff, value & 0xffffffff) if reverse else (value & 0xffffffff, literal & 0xffffffff)
    expected = functions[operation](lhs,rhs)
    return selection_source([
        f'%input = arith.constant {signed(value)} : i32',
        f'%literal = arith.constant {signed(literal)} : i32',
        f'%result = arith.{operation} '+('%literal,%input' if reverse else '%input,%literal')+' : i32',
        f'%expected = arith.constant {signed(expected)} : i32',
        '%condition = arith.cmpi eq,%result,%expected : i32',
    ], '%condition')


class ScalarImmediateTest(unittest.TestCase):
    def assert_selected(self, program, wanted=True):
        false_tile, true_tile = bytes([0x11])*2048, bytes([0x22])*2048
        memory = execute_copies(program, [(0x90000000,false_tile),(0x90000800,true_tile)])
        self.assertEqual(bytes(memory[0x90020000+i] for i in range(2048)),
                         true_tile if wanted else false_tile)

    def test_binary_signed12_and_shift_boundaries_execute_and_verify(self):
        for operation in ('addi','andi','ori','xori','shli','shrui'):
            for literal in ((0,7,16,31) if operation in ('shli','shrui') else (-2048,-1,0,2047)):
                for reverse in ((False,) if operation in ('shli','shrui') else (False,True)):
                    with self.subTest(operation=operation,literal=literal,reverse=reverse):
                        machine,program = compile_program(binary_source(operation,0x81234567,literal,reverse),spill=False)
                        self.assert_selected(program)
                        kind = {'shli':'slli','shrui':'srli'}.get(operation,operation)
                        self.assertTrue(any(row['operation']=='atlas.alu_imm' and row['fields']['kind']==kind and
                                            row['fields']['immediate']==literal for row in program['instructions']))
                        self.assertRegex(machine, r'constant = '+str(literal)+r' : i32[^{}]*reg = 0 : i32')

    def test_small_comparisons_all_predicates_orders_and_endpoint_values(self):
        comparisons = {'eq':lambda a,b:a==b,'ne':lambda a,b:a!=b,
                       'slt':lambda a,b:signed(a)<signed(b),'sle':lambda a,b:signed(a)<=signed(b),
                       'sgt':lambda a,b:signed(a)>signed(b),'sge':lambda a,b:signed(a)>=signed(b),
                       'ult':lambda a,b:(a&0xffffffff)<(b&0xffffffff),
                       'ule':lambda a,b:(a&0xffffffff)<=(b&0xffffffff),
                       'ugt':lambda a,b:(a&0xffffffff)>(b&0xffffffff),
                       'uge':lambda a,b:(a&0xffffffff)>=(b&0xffffffff)}
        for predicate,reference in comparisons.items():
            for literal in (-50,47):
                for reverse in (False,True):
                    # Keep the varying operand a nonliteral source definition.
                    source = selection_source([
                        '%base = arith.constant 2147483647 : i32',
                        '%delta = arith.constant 1 : i32',
                        '%input = arith.addi %base,%delta : i32',
                        f'%literal = arith.constant {literal} : i32',
                        f'%condition = arith.cmpi {predicate},'+('%literal,%input' if reverse else '%input,%literal')+' : i32',
                    ],'%condition')
                    with self.subTest(predicate=predicate,literal=literal,reverse=reverse):
                        _,program = compile_program(source,spill=False)
                        pair = (literal,0x80000000) if reverse else (0x80000000,literal)
                        self.assert_selected(program,reference(*pair))

    def test_immediate_only_literals_do_not_consume_scalar_registers(self):
        lines = ['%input = arith.constant 305419896 : i32']
        lines += [f'%literal{i} = arith.constant {i+1} : i32' for i in range(24)]
        for i in range(24):
            previous = '%input' if i==0 else f'%sum{i-1}'
            lines.append(f'%sum{i} = arith.addi {previous},%literal{i} : i32')
        lines += ['%expected = arith.constant 305420196 : i32',
                  '%condition = arith.cmpi eq,%sum23,%expected : i32']
        machine,program = compile_program(selection_source(lines,'%condition'),spill=False)
        self.assert_selected(program)
        self.assertEqual(len(re.findall(r'def = "arith.constant"[^{}]*reg = 0 : i32',machine)),24)
        self.assertNotIn('atlas.virtual_save',machine)

    def test_comparison_threshold_increment_preserves_integer_endpoints(self):
        cases = [('sgt',0x7fffffff,False),('sle',0x7fffffff,True),
                 ('ugt',0xffffffff,False),('ule',0xffffffff,True),
                 ('sge',0x80000000,True),('slt',0x80000000,False)]
        for predicate,literal,wanted in cases:
            source = selection_source([
                '%base = arith.constant 2147483647 : i32','%delta = arith.constant 1 : i32',
                '%input = arith.addi %base,%delta : i32',
                f'%literal = arith.constant {signed(literal)} : i32',
                f'%condition = arith.cmpi {predicate},%input,%literal : i32'], '%condition')
            with self.subTest(predicate=predicate,literal=literal):
                _,program = compile_program(source,spill=False)
                self.assert_selected(program,wanted)

    def test_nonencodable_and_mixed_uses_keep_their_literal_producers(self):
        for literal in (-2049,2048,0x7fffffff,0x80000000):
            with self.subTest(literal=literal):
                machine,program = compile_program(binary_source('addi',0x12345678,literal),spill=False)
                self.assert_selected(program)
                self.assertRegex(machine,r'constant = '+str(signed(literal))+r' : i32[^{}]*reg = (?!0 :)[0-9]+ : i32')
        # This literal feeds both an immediate add and a register-only sub.
        source = selection_source([
            '%input = arith.constant 305419896 : i32','%literal = arith.constant 127 : i32',
            '%sum = arith.addi %input,%literal : i32','%result = arith.subi %sum,%literal : i32',
            '%condition = arith.cmpi eq,%result,%input : i32'], '%condition')
        machine,program = compile_program(source,spill=False)
        self.assert_selected(program)
        self.assertRegex(machine,r'constant = 127 : i32[^{}]*reg = (?!0 :)[0-9]+ : i32')
        for literal in (-1,32):
            # The existing virtual shift-domain refusal remains mandatory.
            source = selection_source([
                '%input = arith.constant 305419896 : i32',
                f'%literal = arith.constant {literal} : i32',
                '%result = arith.shli %input,%literal : i32',
                '%condition = arith.cmpi ne,%input,%literal : i32'], '%condition')
            result = run(source,'--verify-atlas-virtual-stream')
            self.assertNotEqual(result.returncode,0)
            self.assertIn('proven unsigned range below32',result.stderr)

    def test_elided_literal_value_operand_and_emitted_immediate_are_checked(self):
        machine,_ = compile_program(binary_source('addi',0x12345678,127),spill=False)
        folded = next(line for line in machine.splitlines() if
                      '"atlas.alu_imm"' in line and 'immediate = 127 : i32' in line)
        altered = [machine.replace('constant = 127 : i32','constant = 126 : i32',1),
                   machine.replace('immediate = 127 : i32','immediate = 126 : i32',1),
                   machine.replace(folded,folded.replace('kind = "addi"','kind = "xori"'),1),
                   machine.replace(folded,re.sub(r'src = \d+ : i32','src = 0 : i32',folded),1)]
        # Also forge the source operator: it cannot use the elided value as a
        # register-only operand just by retaining the literal record.
        altered.append(machine.replace('def = "arith.addi"','def = "arith.subi"',1))
        for index,changed in enumerate(altered):
            with self.subTest(case=index):
                self.assertNotEqual(changed,machine)
                result = run(changed,'--verify-atlas-generated-schedule')
                self.assertNotEqual(result.returncode,0)
                self.assertEqual(result.stdout,'')


if __name__ == '__main__':
    unittest.main()
