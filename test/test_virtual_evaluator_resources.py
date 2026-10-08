"""Resource identities must bind fresh tile contents on each CFG loop visit."""

from __future__ import annotations

import unittest

from test_virtual_evaluator_dma import bf16_bytes
from test_virtual_evaluator_dma_handles import ADDRESS, B, S, constants, input_tile, load, output, ready, store, wait, wrap
from test_virtual_evaluator_mxu_handles import F, TILES, accumulate, diagonal, function, program, read, reset, seed, weight
from test_virtual_evaluator_mxu1 import on_unit
from atlas_virtual_evaluator import MemoryRegion, RuntimeInputs, Tile, VirtualInterfaceError, evaluate, parse_program


class VirtualEvaluatorResourceTest(unittest.TestCase):
    def test_dma_loop_rebinds_static_store_and_load_handles_to_changed_tiles(self) -> None:
        # Iteration one stores A, iteration two stores B at the same address.
        # Both the final bytes and awaited tile must come from the second visit.
        first = Tile("bf16", tuple((row * 2039 + col * 137) & 0xFFFF for row in range(32) for col in range(32)))
        second = Tile("bf16", tuple(bits ^ 0x8155 for bits in first.bits))
        source = wrap(input_tile("bf16"), input_tile("bf16", "io", "io2", "other", 1), *constants(),
                      "%zero = arith.constant 0 : i32", "%one = arith.constant 1 : i32", "%limit = arith.constant 2 : i32",
                      f"cf.br ^loop(%io2, %x, %other, %zero : {S}, {B}, {B}, i32)", f"^loop(%ls: {S}, %a: {B}, %b: {B}, %i: i32):",
                      store("bf16", "ls", "s1", "write", "a"), wait("s1", "s2", "write"), load("bf16", "s2", "s3", "read"), ready("bf16", "s3", "s4", "read"),
                      "%next = arith.addi %i, %one : i32", "%again = arith.cmpi ult, %next, %limit : i32",
                      f"cf.cond_br %again, ^loop(%s4, %b, %a, %next : {S}, {B}, {B}, i32), ^exit(%s4, %loaded : {S}, {B})",
                      f"^exit(%es: {S}, %last: {B}):", output("es", "done", "last"), final="done")
        guard = MemoryRegion(ADDRESS - 13, b"before!before" + b"?" * 2048 + b"after!")
        inputs = RuntimeInputs({0: first, 1: second}, memory=(guard,))
        result = evaluate(parse_program(source), inputs)
        self.assertEqual(dict(result.outputs), {0: second})
        self.assertEqual(result.memory, (MemoryRegion(guard.address, b"before!before" + bf16_bytes(second.bits) + b"after!"),))
        self.assertEqual(inputs.memory, (guard,))
        self.assertEqual(dict(inputs.tiles), {0: first, 1: second})

    def test_mxu1_rejects_a_third_live_weight_or_accumulator(self) -> None:
        weights = (weight("s4", "t0", "w0"), weight("t0", "t1", "w1"), weight("t1", "t2", "w2"),
                   reset("t2", "t3", "a0", "w0"), read("t3", "t4", "y0", "a0"), reset("t4", "t5", "a1", "w1"),
                   read("t5", "t6", "y1", "a1"), reset("t6", "t7", "a2", "w2"), read("t7", "t8", "y2", "a2"))
        accumulators = (seed("s4", "t0", "a0"), seed("t0", "t1", "a1"), seed("t1", "t2", "a2"),
                        read("t2", "t3", "y0", "a0"), read("t3", "t4", "y1", "a1"), read("t4", "t5", "y2", "a2"))
        for kind, operations, final in (("weight", weights, "t8"), ("accumulator", accumulators, "t5")):
            parsed = parse_program(on_unit(program(*operations, final_state=final), 1))
            with self.subTest(kind=kind), self.assertRaisesRegex(VirtualInterfaceError, "at most two live MXU"):
                evaluate(parsed, RuntimeInputs(TILES))

    def test_mxu_loop_rebinds_weight_and_accumulator_versions_from_carried_tiles(self) -> None:
        # Identity activations contract with packed diagonal A, seeded by A.
        # Swapping to B on the next visit changes both weights and seed: 6 -> 4.
        body = "\n".join((input_tile("bf16", "s4", "io5", "other", 4),
                          "%zero = arith.constant 0 : i32", "%one = arith.constant 1 : i32", "%limit = arith.constant 2 : i32",
                          f"cf.br ^loop(%io5, %seed, %other, %zero : {S}, {B}, {B}, i32)", f"^loop(%ls: {S}, %a: {B}, %b: {B}, %i: i32):",
                          f'%packed = "atlas.virtual_pack_fp8"(%a) {{scale_code = 127 : i32}} : ({B}) -> {F}',
                          weight("ls", "ws", "w0", "packed"), seed("ws", "as", "a0", "a"), accumulate("as", "ms", "a1", "a0"), read("ms", "rs", "result", "a1"),
                          "%next = arith.addi %i, %one : i32", "%again = arith.cmpi ult, %next, %limit : i32",
                          f"cf.cond_br %again, ^loop(%rs, %b, %a, %next : {S}, {B}, {B}, i32), ^exit(%rs, %result : {S}, {B})",
                          f"^exit(%es: {S}, %last: {B}):", output("es", "done", "last"), f"func.return %done : {S}"))
        tiles = {**TILES, 4: diagonal("bf16", 0x4000)}
        for unit in (0, 1):
            with self.subTest(unit=unit):
                inputs = RuntimeInputs(tiles)
                result = evaluate(parse_program(on_unit(function(body), unit)), inputs)
                self.assertEqual(dict(result.outputs), {0: diagonal("bf16", 0x4080)})
                self.assertEqual(dict(inputs.tiles), tiles)
                self.assertEqual(result.memory, ())


if __name__ == "__main__":
    unittest.main()
