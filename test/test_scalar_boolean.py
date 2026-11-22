"""Typed i1 constants and bit operations must control the correct branch."""
import re
import unittest

from test_virtual_spilling import BF16, STATE, compile_program, execute_copies, run, wrap


def source(operation, comparison, constant):
    return wrap([
        f'%s0 = "atlas.virtual_start"() : () -> {STATE}',
        f'%s1,%a = "atlas.virtual_input_bf16"(%s0) {{index=0:i32}} : ({STATE}) -> ({STATE},{BF16})',
        f'%s2,%b = "atlas.virtual_input_bf16"(%s1) {{index=1:i32}} : ({STATE}) -> ({STATE},{BF16})',
        '%one = arith.constant 1 : i32', f'%literal = arith.constant {int(constant)} : i1',
        f'%comparison = arith.cmpi {"eq" if comparison else "ne"},%one,%one : i32',
        f'%condition = arith.{operation} %comparison,%literal : i1',
        f'cf.cond_br %condition,^true(%s2 : {STATE}),^false(%s2 : {STATE})',
        f'^true(%ts: {STATE}):', f'cf.br ^end(%ts,%b : {STATE},{BF16})',
        f'^false(%fs: {STATE}):', f'cf.br ^end(%fs,%a : {STATE},{BF16})',
        f'^end(%es: {STATE},%value: {BF16}):',
        f'%done = "atlas.virtual_output_bf16"(%es,%value) {{index=0:i32}} : ({STATE},{BF16}) -> {STATE}',
        f'return %done : {STATE}',
    ])


class ScalarBooleanTest(unittest.TestCase):
    def test_mixed_comparison_and_literal_truth_tables_select_correct_tile(self):
        false_tile, true_tile = bytes([0x11]) * 2048, bytes([0x22]) * 2048
        operations = {'andi': lambda a, b: a and b, 'ori': lambda a, b: a or b,
                      'xori': lambda a, b: bool(a) != bool(b)}
        for operation, reference in operations.items():
            for comparison in (False, True):
                for constant in (False, True):
                    with self.subTest(operation=operation, comparison=comparison, constant=constant):
                        _, program = compile_program(source(operation, comparison, constant))
                        memory = execute_copies(program, [(0x90000000, false_tile), (0x90000800, true_tile)])
                        wanted = true_tile if reference(comparison, constant) else false_tile
                        self.assertEqual(bytes(memory[0x90020000 + i] for i in range(2048)), wanted)

    def test_true_extui_stores_one(self):
        slot = "!atlas.virtual_spill"
        text = wrap([
            f'%s0 = "atlas.virtual_start"() : () -> {STATE}',
            f'%slot = "atlas.virtual_slot"() {{kind="bf16",words=512:i32,slot_word=-1:i32,purpose="scratch"}} : () -> {slot}',
            f'%input_state,%zero = "atlas.virtual_input_bf16"(%s0) {{index=0:i32}} : ({STATE}) -> ({STATE},{BF16})',
            f'%s1 = "atlas.virtual_save"(%input_state,%slot,%zero) : ({STATE},{slot},{BF16}) -> {STATE}',
            '%truth = arith.constant true',
            '%word = arith.extui %truth : i1 to i32',
            f'%s2 = "atlas.virtual_slot_store"(%s1,%slot,%word) {{kind="sw",byte_offset=0:i32}} : ({STATE},{slot},i32) -> {STATE}',
            f'%s3,%value = "atlas.virtual_restore"(%s2,%slot) : ({STATE},{slot}) -> ({STATE},{BF16})',
            f'%done = "atlas.virtual_output_bf16"(%s3,%value) {{index=0:i32}} : ({STATE},{BF16}) -> {STATE}',
            f'return %done : {STATE}',
        ])
        _, program = compile_program(text)
        memory = execute_copies(program, [(0x90000000, bytes(2048))])
        self.assertEqual(bytes(memory[0x90020000+i] for i in range(4)), bytes([1,0,0,0]))

    def test_noncanonical_i1_constant_certificate_is_rejected(self):
        machine, _ = compile_program(source('xori', True, True))
        altered, count = re.subn(r'(\{[^{}]*constant = )1( : i32[^{}]*type = "i1"[^{}]*\})',
                                r'\g<1>-1\g<2>', machine, count=1)
        self.assertEqual(count, 1)
        result = run(altered, '--verify-atlas-generated-schedule')
        self.assertNotEqual(result.returncode, 0)


if __name__ == '__main__':
    unittest.main()
