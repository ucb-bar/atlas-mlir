"""Atomic tile DMA is verified before physical allocation exists.

A virtual load returns its tile and a store completes in place; physical
DMA waits come from insert-atlas-dma-waits after lowering."""

from __future__ import annotations

import unittest

from test_virtual_ssa import BIN, ROOT, run
from test_virtual_mxu_handles import program, load, reset, accumulate, readout


STATE = "!atlas.virtual_state"
EXAMPLE = ROOT / "test/examples/virtual_dma_tiles.mlir"


def tile(fmt: str) -> str:
    return f"!atlas.virtual_{fmt}"


def dma_load(fmt: str, before: str = "io0", after: str = "io2", result: str = "tile", addr: str = "addr", size: str = "size") -> str:
    return (f'    %{after}, %{result} = "atlas.virtual_dma_load_{fmt}"'
            f'(%{before}, %{addr}, %{size}) : ({STATE}, i32, i32) '
            f'-> ({STATE}, {tile(fmt)})')


def dma_store(fmt: str, before: str = "io2", after: str = "io4", value: str = "tile") -> str:
    return (f'    %{after} = "atlas.virtual_dma_store_{fmt}"'
            f'(%{before}, %{value}, %addr, %size) : '
            f'({STATE}, {tile(fmt)}, i32, i32) -> {STATE}')


def constants(fmt: str, address: int = -2147483648, size: int | None = None) -> list[str]:
    if size is None:
        size = 1024 if fmt == "fp8" else 2048
    return [f"    %addr = arith.constant {address} : i32",
            f"    %size = arith.constant {size} : i32"]


def wrap(body: list[str], final: str = "io4", *, flat: bool = False, arguments: str = "") -> str:
    if flat:
        return "\n".join(["module {", *body, "}"])
    return "\n".join([
        "module {", f"  func.func @dma({arguments}) -> {STATE} attributes "
        "{atlas.input_dram_base = 2415919104 : i64, "
        "atlas.output_dram_base = 2415923200 : i64} {", *body,
        f"    return %{final} : {STATE}", "  }", "}",
    ])


def copy(fmt: str = "bf16", *, flat: bool = False, address: int = -2147483648, size: int | None = None) -> str:
    return wrap([
        f'    %io0 = "atlas.virtual_start"() : () -> {STATE}',
        *constants(fmt, address, size),
        dma_load(fmt), dma_store(fmt),
    ], flat=flat)


class VirtualDMATest(unittest.TestCase):
    def setUp(self) -> None:
        self.assertTrue((BIN / "atlas-opt").is_file(), "build atlas-opt first")
        self.assertTrue((BIN / "atlas-emit").is_file(), "build atlas-emit first")

    def accepted(self, source: str) -> str:
        result = run("atlas-opt", source, "--verify-atlas-virtual-stream")
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout

    def rejected(self, source: str, expected: str = "", *, stream: bool = True) -> None:
        options = ("--verify-atlas-virtual-stream",) if stream else ()
        result = run("atlas-opt", source, *options)
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertTrue(result.stderr)
        self.assertEqual(result.stdout, "")
        if expected:
            self.assertIn(expected, result.stderr)

    def test_both_formats_round_trip_in_flat_and_function_streams(self) -> None:
        for fmt in ("fp8", "bf16"):
            for flat in (False, True):
                with self.subTest(fmt=fmt, flat=flat):
                    first = self.accepted(copy(fmt, flat=flat))
                    self.assertEqual(first, self.accepted(first))
                    self.assertIn(f"-> ({STATE}, {tile(fmt)})", first)
                    self.assertIn(f'"atlas.virtual_dma_store_{fmt}"', first)
                    self.assertNotIn("virtual_dma_await", first)
                    self.assertNotIn("virtual_dma_wait", first)
                    self.assertNotIn('"atlas.virtual_output_bf16"', first)
        first = self.accepted(EXAMPLE.read_text())
        self.assertEqual(first, self.accepted(first))

    def test_signed_address_bits_and_top_of_dram_window_are_valid(self) -> None:
        for fmt, size in (("fp8", 1024), ("bf16", 2048)):
            for address in (-2147483648, 2147483648, 4294967296 - size):
                with self.subTest(fmt=fmt, address=address):
                    self.accepted(copy(fmt, address=address))

    def test_addition_proofs_use_unsigned_modulo_i32(self) -> None:
        source = copy().replace(
            "%addr = arith.constant -2147483648 : i32",
            "%a = arith.constant -32 : i32\n"
            "    %b = arith.constant -2147483616 : i32\n"
            "    %half = arith.constant 1024 : i32\n"
            "    %addr = arith.addi %a, %b : i32",
        ).replace("%size = arith.constant 2048 : i32",
                  "%size = arith.addi %half, %half : i32")
        self.accepted(source)
        self.accepted(source.replace(
            "%addr = arith.addi %a, %b : i32",
            "%zero = arith.constant 0 : i32\n"
            "    %sum = arith.addi %a, %b : i32\n"
            "    %addr = arith.addi %sum, %zero : i32"))

    def test_address_proof_rejects_overflow_flags(self) -> None:
        source = copy().replace(
            "%addr = arith.constant -2147483648 : i32",
            "%a = arith.constant 1879048192 : i32\n"
            "    %b = arith.constant 268435456 : i32\n"
            "    %addr = arith.addi %a, %b : i32")
        self.accepted(source)
        for flag in ("nsw", "nuw", "nsw, nuw"):
            with self.subTest(flag=flag):
                self.rejected(source.replace("arith.addi %a, %b : i32",
                    f"arith.addi %a, %b overflow<{flag}> : i32"),
                    "DRAM address and byte length must be proven")

    def test_address_alignment_geometry_and_widened_end_fail(self) -> None:
        for fmt in ("fp8", "bf16"):
            for address in (0, 2147483616, 2147483649, 4294967264):
                with self.subTest(fmt=fmt, address=address):
                    expected = ("DRAM transfer span exceeds the 32-bit address space"
                                if address == 4294967264 else
                                "DRAM address must be a 32-byte-aligned selected-memory address")
                    self.rejected(copy(fmt, address=address), expected)
            for size in (0, 32, 1024 if fmt == "bf16" else 2048, -1):
                with self.subTest(fmt=fmt, size=size):
                    self.rejected(copy(fmt, size=size), "byte length must equal the complete tile size")

    def test_store_geometry_is_checked_independently(self) -> None:
        for fmt in ("fp8", "bf16"):
            source = copy(fmt).replace(
                dma_store(fmt),
                "    %bad_addr = arith.constant 0 : i32\n"
                + dma_store(fmt).replace("%addr, %size", "%bad_addr, %size"))
            self.rejected(source, "DRAM address must be a 32-byte-aligned selected-memory address")
            source = copy(fmt).replace(
                dma_store(fmt),
                "    %bad_size = arith.constant 32 : i32\n"
                + dma_store(fmt).replace("%addr, %size", "%addr, %bad_size"))
            self.rejected(source, "byte length must equal the complete tile size")

    def test_dynamic_values_and_non_additive_proofs_fail(self) -> None:
        for operand in ("addr", "size"):
            source = copy().replace("@dma()", "@dma(%dynamic: i32)")
            constant = ("%addr = arith.constant -2147483648 : i32" if operand == "addr"
                        else "%size = arith.constant 2048 : i32")
            for expression in ("arith.addi %dynamic, %dynamic : i32",
                               "arith.muli %dynamic, %dynamic : i32"):
                with self.subTest(operand=operand, expression=expression):
                    self.rejected(source.replace(constant, f"%{operand} = {expression}"),
                                  "DRAM address and byte length must be proven")
            self.rejected(source.replace(constant, "").replace(f"%{operand}", "%dynamic"),
                          "DRAM address and byte length must be proven")

    def test_only_i1_and_i32_scalar_operations_are_admitted(self) -> None:
        scalar = ("    %one = arith.constant 1 : i32\n"
                  "    %next = arith.addi %one, %one : i32\n"
                  "    %less = arith.cmpi slt, %one, %next : i32\n"
                  "    %true = arith.constant true\n")
        for flat in (False, True):
            source = copy(flat=flat).replace(dma_load("bf16"), scalar + dma_load("bf16"))
            self.accepted(source)
            for instruction in ("%wide = arith.constant 0 : i64",
                                "%float = arith.constant 0.0 : f32",
                                "%product = arith.muli %addr, %size : i32",
                                "%bit = arith.addi %true, %true : i1",
                                "%same = arith.cmpi eq, %true, %true : i1"):
                with self.subTest(flat=flat, instruction=instruction):
                    self.rejected(source.replace(dma_load("bf16"), instruction + "\n" + dma_load("bf16")))
        for scalar_type in ("i1", "i64"):
            source = copy().replace("i32", scalar_type)
            if scalar_type == "i1":
                source = source.replace("arith.constant -2147483648 : i1",
                                        "arith.constant true")
                source = source.replace("arith.constant 2048 : i1", "arith.constant true")
            self.rejected(source, stream=False)

    def test_loads_produce_format_tiles_and_stores_require_matching_tiles(self) -> None:
        for fmt, other in (("fp8", "bf16"), ("bf16", "fp8")):
            self.rejected(copy(fmt).replace(f"-> ({STATE}, {tile(fmt)})", f"-> ({STATE}, {tile(other)})"),
                          stream=False)
            self.rejected(copy(fmt).replace(dma_store(fmt), dma_store(other)), stream=False)

    def test_channel_attributes_cannot_select_physical_resources(self) -> None:
        for operation in (dma_load("bf16"), dma_store("bf16")):
            with self.subTest(operation=operation):
                changed = operation.replace(") :", ") {channel = 0 : i32} :")
                self.rejected(copy().replace(operation, changed),
                              "virtual DMA does not select a physical channel", stream=False)

    def test_completion_operations_are_not_virtual_instructions(self) -> None:
        source = copy()
        for operation in (
            f'    %io3, %again = "atlas.virtual_dma_await_bf16"(%io2, %tile) : ({STATE}, {tile("bf16")}) -> ({STATE}, {tile("bf16")})',
            f'    %io3 = "atlas.virtual_dma_wait"(%io2, %tile) : ({STATE}, {tile("bf16")}) -> {STATE}',
        ):
            with self.subTest(operation=operation):
                self.rejected(source.replace(dma_store("bf16"), operation + "\n" + dma_store("bf16", before="io3")),
                              stream=False)

    def test_dma_has_no_pending_limit_and_keeps_state_order(self) -> None:
        source = copy()
        loads = "\n".join(dma_load("bf16", before=before, after=after, result=result)
                          for before, after, result in (("io2", "l1", "t1"), ("l1", "l2", "t2"), ("l2", "l3", "t3")))
        self.accepted(source.replace(dma_store("bf16"), loads + "\n" + dma_store("bf16", before="l3")))
        # A load after a store of the same bytes reads what the store wrote.
        reload = dma_load("bf16", before="io4", after="io5", result="again")
        self.accepted(source.replace("return %io4", reload + "\n    return %io5"))

    def test_dma_preserves_the_linear_virtual_state_chain(self) -> None:
        source = copy()
        self.rejected(source.replace(dma_store("bf16"), dma_store("bf16", before="io0")),
                      "nonlinear virtual state chain")
        self.rejected(source.replace("return %io4", "return %io2"),
                      "must return the current virtual state")

    def test_boundary_tiles_and_pack_may_follow_dma(self) -> None:
        for operation, after in (
            (f'    %s1, %hidden = "atlas.virtual_input_bf16"(%s0) {{index = 3 : i32}} : ({STATE}) -> ({STATE}, {tile("bf16")})', "s1"),
            (f'    %s1 = "atlas.virtual_output_bf16"(%s0, %seed) {{index = 1 : i32}} : ({STATE}, {tile("bf16")}) -> {STATE}', "s1"),
            (f'    %packed = "atlas.virtual_pack_fp8"(%seed) {{scale_code = 127 : i32}} : ({tile("bf16")}) -> {tile("fp8")}', "s0"),
        ):
            with self.subTest(operation=operation):
                self.accepted(program(*constants("bf16"), dma_load("bf16", before="io2", after="s0"),
                                      operation, final_state=after))

    def test_pure_tile_work_and_mxu_compute_may_follow_a_load(self) -> None:
        pure = (f'    %independent = "atlas.virtual_vpu_unary"(%seed) {{kind = "relu"}} '
                f': ({tile("bf16")}) -> {tile("bf16")}')
        matmul = (f'    %product = "atlas.virtual_mxu_matmul"(%x, %w) {{unit = 1 : i32}} '
                  f': ({tile("fp8")}, {tile("fp8")}) -> {tile("bf16")}')
        self.accepted(program(
            *constants("bf16"), load("io2", "s0", "weight"),
            dma_load("bf16", before="s0", after="s1"), pure, matmul,
            reset("s1", "s2", "acc0"), accumulate("s2", "s3", "acc1", "acc0"),
            readout("s3", "s4", "mxu_tile", "acc1"), final_state="s4"))

    def test_completed_tiles_can_cross_cfg_edges_without_dma_handles(self) -> None:
        source = copy().replace(dma_store("bf16"),
            f"    cf.br ^exit(%io2, %tile : {STATE}, {tile('bf16')})\n"
            f"  ^exit(%ready: {STATE}, %value: {tile('bf16')}):\n"
            + dma_store("bf16", before="ready", value="value"))
        self.accepted(source)

    def test_cfg_branch_loads_join_by_tile(self) -> None:
        source = wrap([
            f'    %io0 = "atlas.virtual_start"() : () -> {STATE}',
            *constants("bf16"),
            f"    cf.cond_br %choose, ^left(%io0 : {STATE}), ^right(%io0 : {STATE})",
            f"  ^left(%left_io: {STATE}):",
            dma_load("bf16", before="left_io", after="left_ready", result="left_tile"),
            f"    cf.br ^join(%left_ready, %left_tile : {STATE}, {tile('bf16')})",
            f"  ^right(%right_io: {STATE}):",
            dma_load("bf16", before="right_io", after="right_ready", result="right_tile"),
            f"    cf.br ^join(%right_ready, %right_tile : {STATE}, {tile('bf16')})",
            f"  ^join(%joined_io: {STATE}, %joined_tile: {tile('bf16')}):",
            dma_store("bf16", before="joined_io", value="joined_tile"),
        ], arguments="%choose: i1")
        first = self.accepted(source)
        self.assertEqual(first, self.accepted(first))
        # The right branch cannot use the left branch's tile.
        self.rejected(source.replace("(%right_ready, %right_tile", "(%right_ready, %left_tile"))

    def test_cfg_store_and_completion_must_be_in_the_unique_return_block(self) -> None:
        source = copy()
        completed_early = source.replace("return %io4 : " + STATE,
            f"cf.br ^exit(%io4 : {STATE})\n  ^exit(%last: {STATE}):\n"
            f"    return %last : {STATE}")
        self.rejected(completed_early, "virtual CFG outputs must be in the return block")
        second_return = source.replace(dma_load("bf16"),
            "    %choose = arith.constant true\n"
            f"    cf.cond_br %choose, ^work(%io0 : {STATE}), ^early(%io0 : {STATE})\n"
            f"  ^early(%early_io: {STATE}):\n    return %early_io : {STATE}\n"
            f"  ^work(%work_io: {STATE}):\n" + dma_load("bf16", before="work_io"))
        self.rejected(second_return, "virtual CFG requires one return block")

    def test_physical_consumers_require_explicit_function_lowering(self) -> None:
        for source in (EXAMPLE.read_text(), copy("fp8"), copy(flat=True)):
            self.accepted(source)
            for tool, options in (
                ("atlas-emit", ()),
                ("atlas-opt", ("--verify-atlas-machine-stream",)),
                ("atlas-opt", ("--convert-atlas-to-llvm",)),
            ):
                with self.subTest(tool=tool, options=options):
                    result = run(tool, source, *options)
                    self.assertNotEqual(result.returncode, 0, result.stdout)
                    self.assertTrue(result.stderr)
                    self.assertEqual(result.stdout, "")
            lowered = run("atlas-opt", source, "--lower-atlas-virtual-to-machine", "--insert-atlas-dma-waits")
            if "func.func" in source:
                self.assertEqual(lowered.returncode, 0, lowered.stderr)
                self.assertNotRegex(lowered.stdout, r'(?m)^\s*%[^\n=]+\s*=\s*"atlas\.virtual_')
            else:
                self.assertNotEqual(lowered.returncode, 0, lowered.stdout)
                self.assertTrue(lowered.stderr)
                self.assertEqual(lowered.stdout, "")


if __name__ == "__main__":
    unittest.main()
