"""Shared slice and CFG helpers preserve the legacy compiler's behavior."""

import unittest

from xdsl.dialects import arith, cf, func
from xdsl.dialects.builtin import ModuleOp, i8, i32
from xdsl.interpreter import Interpreter
from xdsl.interpreters.arith import ArithFunctions
from xdsl.interpreters.cf import CfFunctions
from xdsl.interpreters.func import FuncFunctions
from xdsl.ir import Block, Region

from merlin.compiler.schedule.counted_loop import repeat_state_cfg, simplify_state_cfg
from merlin.compiler.transforms.slice_sizes import uniform_slice_elements


def execute(function):
    interpreter = Interpreter(ModuleOp([function]))
    for implementation in (ArithFunctions(), CfFunctions(), FuncFunctions()):
        interpreter.register_implementations(implementation)
    return interpreter.call_op(function)


def straight_line():
    initial = arith.ConstantOp.from_int_and_width(7, i32)
    increment = arith.ConstantOp.from_int_and_width(3, i32)
    update = arith.AddiOp(initial, increment)
    block = Block([initial, increment, update, func.ReturnOp(update)])
    return func.FuncOp("state", ([], [i32]), Region(block)), initial.result


class SharedSupportTest(unittest.TestCase):
    def test_slices_preserve_capacity_alignment_and_minimum_calls(self):
        self.assertEqual(uniform_slice_elements(24, 16, 4), 12)
        self.assertEqual(uniform_slice_elements(25, 16, 4), 16)
        self.assertEqual(uniform_slice_elements(7, 16, 4), 16)
        for elements in range(1, 65):
            for capacity in range(4, 33, 4):
                size = uniform_slice_elements(elements, capacity, 4)
                self.assertEqual(size % 4, 0)
                self.assertLessEqual(size, capacity)
                self.assertEqual((elements + size - 1) // size, (elements + capacity - 1) // capacity)
        for arguments in ((0, 8, 1), (8, 0, 1), (8, 8, 0), (8, 3, 4), (True, 8, 1)):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                uniform_slice_elements(*arguments)

    def test_straight_line_state_and_offset_execution(self):
        function, initial = straight_line()
        induction = repeat_state_cfg(function, iterations=5, steps=(2, 7), index_type=i32, initial_state=initial)
        # Read the exposed offsets inside the repeated transition, ensuring they
        # dominate users and advance once per backedge.
        body = next(block for block in function.body.blocks if any(isinstance(op, arith.AddiOp) for op in block.ops))
        returned = body.last_op
        self.assertIsInstance(returned, cf.BranchOp)
        update = arith.AddiOp(returned.arguments[0], induction.offsets[0])
        body.insert_op_before(update, returned)
        returned.operands = (update.result, *returned.arguments[1:])
        function.verify()
        self.assertEqual(len(induction.offsets), 2)
        self.assertEqual(execute(function), (42,))  # 7 + 5*3 + (0+2+4+6+8)

    def test_existing_cfg_and_local_cse_preserve_state(self):
        initial = arith.ConstantOp.from_int_and_width(7, i32)
        body = Block(arg_types=[i32])
        one = arith.ConstantOp.from_int_and_width(1, i32)
        duplicate = arith.ConstantOp.from_int_and_width(1, i32)
        add = arith.AddiOp(body.args[0], duplicate)
        body.add_ops([one, duplicate, add, func.ReturnOp(add)])
        entry = Block([initial, cf.BranchOp(body, initial)])
        function = func.FuncOp("state", ([], [i32]), Region([entry, body]))
        repeat_state_cfg(function, iterations=4, steps=(), index_type=i32)
        simplify_state_cfg(function)
        function.verify()
        self.assertEqual(execute(function), (11,))
        self.assertEqual(sum(isinstance(op, arith.ConstantOp) for op in body.ops), 1)

    def test_invalid_count_and_overflow_leave_original_cfg_unchanged(self):
        for iterations, steps in ((0, (1,)), (-1, (1,)), (256, (1,)), (True, (1,)), (2, (128,)), (2, (0,))):
            with self.subTest(iterations=iterations, steps=steps):
                function, initial = straight_line()
                original = str(function)
                with self.assertRaises(ValueError):
                    repeat_state_cfg(function, iterations=iterations, steps=steps, index_type=i8, initial_state=initial)
                self.assertEqual(str(function), original)
                function.verify()


if __name__ == "__main__":
    unittest.main()
