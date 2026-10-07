"""Logical tile DMA events are verified before physical allocation exists."""

from __future__ import annotations

import unittest

from test_virtual_ssa import BIN, ROOT, run
from test_virtual_mxu_handles import program, load, reset, accumulate, readout


STATE = "!atlas.virtual_state"
STORE = "!atlas.virtual_dma_store"
EXAMPLE = ROOT / "test/examples/virtual_dma_tiles.mlir"


def tile(fmt: str) -> str:
    return f"!atlas.virtual_{fmt}"


def event(fmt: str) -> str:
    return f"!atlas.virtual_dma_load_{fmt}"


def dma_load(fmt: str, before: str = "io0", after: str = "io1", handle: str = "pending", addr: str = "addr", size: str = "size") -> str:
    return (f'    %{after}, %{handle} = "atlas.virtual_dma_load_{fmt}"'
            f'(%{before}, %{addr}, %{size}) : ({STATE}, i32, i32) '
            f'-> ({STATE}, {event(fmt)})')


def dma_await(fmt: str, before: str = "io1", after: str = "io2", handle: str = "pending", result: str = "tile") -> str:
    return (f'    %{after}, %{result} = "atlas.virtual_dma_await_{fmt}"'
            f'(%{before}, %{handle}) : ({STATE}, {event(fmt)}) '
            f'-> ({STATE}, {tile(fmt)})')


def dma_store(fmt: str, before: str = "io2", after: str = "io3", value: str = "tile", handle: str = "store") -> str:
    return (f'    %{after}, %{handle} = "atlas.virtual_dma_store_{fmt}"'
            f'(%{before}, %{value}, %addr, %size) : '
            f'({STATE}, {tile(fmt)}, i32, i32) -> ({STATE}, {STORE})')


def dma_wait(before: str = "io3", after: str = "io4", handle: str = "store") -> str:
    return (f'    %{after} = "atlas.virtual_dma_wait"(%{before}, %{handle}) '
            f': ({STATE}, {STORE}) -> {STATE}')


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
        dma_load(fmt), dma_await(fmt), dma_store(fmt), dma_wait(),
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
                    self.assertIn(event(fmt), first)
                    self.assertIn(STORE, first)
                    self.assertIn(f'"atlas.virtual_dma_store_{fmt}"', first)
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

    def test_load_handles_are_format_specific_and_stores_require_matching_tiles(self) -> None:
        for fmt, other in (("fp8", "bf16"), ("bf16", "fp8")):
            self.rejected(copy(fmt).replace(dma_await(fmt), dma_await(other)), stream=False)
            self.rejected(copy(fmt).replace(dma_store(fmt), dma_store(other)), stream=False)
        self.rejected(copy().replace(dma_wait(), dma_wait(handle="pending")), stream=False)

    def test_channel_attributes_cannot_select_physical_resources(self) -> None:
        operations = [dma_load("bf16"), dma_await("bf16"), dma_store("bf16"), dma_wait()]
        for operation in operations:
            with self.subTest(operation=operation):
                changed = operation.replace(") :", ") {channel = 0 : i32} :")
                self.rejected(copy().replace(operation, changed),
                              "virtual DMA does not select a physical channel", stream=False)

    def test_pending_load_and_store_must_finish_before_exit(self) -> None:
        source = copy()
        self.rejected(source.replace("\n" + dma_await("bf16") + "\n" + dma_store("bf16")
                                     + "\n" + dma_wait(), "").replace("return %io4", "return %io1"),
                      "must complete pending DMA before block exit")
        self.rejected(source.replace("\n" + dma_wait(), "").replace("return %io4", "return %io3"),
                      "must complete pending DMA before block exit")
        for flat in (True, False):
            source = copy(flat=flat)
            self.rejected(source.replace("\n" + dma_wait(), "").replace("return %io4", "return %io3"),
                          "must complete pending DMA before block exit")

    def test_one_pending_transfer_is_global_across_formats_and_directions(self) -> None:
        source = copy()
        for fmt in ("fp8", "bf16"):
            inserted = "    %other_size = arith.constant " + str(1024 if fmt == "fp8" else 2048) + " : i32\n"
            inserted += dma_load(fmt, before="io1", after="extra_io", handle="extra", size="other_size")
            changed = source.replace(dma_await("bf16"), inserted + "\n" + dma_await("bf16", before="extra_io"))
            self.rejected(changed, "must complete the pending DMA before another launch")
        changed = source.replace(dma_wait(), dma_load("bf16", before="io3", after="extra_io", handle="extra")
                                 + "\n" + dma_wait(before="extra_io"))
        self.rejected(changed, "must complete the pending DMA before another launch")
        changed = source.replace(dma_wait(), dma_store("bf16", before="io3", after="extra_io", handle="extra")
                                 + "\n" + dma_wait(before="extra_io"))
        self.rejected(changed, "must complete the pending DMA before another launch")

    def test_completion_handles_cannot_be_reused_or_mismatched(self) -> None:
        source = copy()
        self.rejected(source.replace(dma_store("bf16"), dma_await("bf16", before="io2", after="again_io", result="again")
                                     + "\n" + dma_store("bf16", before="again_io")),
                      "must consume the current pending DMA transfer")
        self.rejected(source.replace("return %io4", dma_wait(before="io4", after="again_io") + "\n    return %again_io"),
                      "must consume the current pending DMA transfer")
        changed = source.replace(dma_store("bf16"), dma_load("bf16", before="io2", after="next_io", handle="next")
                                 + "\n" + dma_await("bf16", before="next_io", after="ready_io", result="next_tile")
                                 + "\n" + dma_store("bf16", before="ready_io"))
        self.rejected(changed, "must consume the current pending DMA transfer")

    def test_pending_dma_preserves_the_linear_virtual_state_chain(self) -> None:
        source = copy()
        self.rejected(source.replace(dma_await("bf16"), dma_await("bf16", before="io0")),
                      "nonlinear virtual state chain")
        self.rejected(source.replace(dma_wait(), dma_wait(before="io2")),
                      "nonlinear virtual state chain")
        self.rejected(source.replace("return %io4", "return %io3"),
                      "must return the current virtual state")

    def test_hidden_dma_and_vmem_operations_fail_while_pending(self) -> None:
        for operation in (
            f'    %hidden_io, %hidden = "atlas.virtual_input_bf16"(%s0) {{index = 3 : i32}} : ({STATE}) -> ({STATE}, {tile("bf16")})',
            f'    %hidden_io, %hidden = "atlas.virtual_input_fp8"(%s0) {{index = 3 : i32}} : ({STATE}) -> ({STATE}, {tile("fp8")})',
            f'    %hidden_io = "atlas.virtual_output_bf16"(%s0, %seed) {{index = 1 : i32}} : ({STATE}, {tile("bf16")}) -> {STATE}',
            f'    %packed = "atlas.virtual_pack_fp8"(%seed) {{scale_code = 127 : i32}} : ({tile("bf16")}) -> {tile("fp8")}',
        ):
            with self.subTest(operation=operation):
                before = "s0" if "virtual_pack" in operation else "hidden_io"
                source = program(*constants("bf16"), dma_load("bf16", before="io2", after="s0"),
                                 operation, dma_await("bf16", before=before, after="s1"), final_state="s1")
                self.rejected(source, "must complete pending DMA before implicit memory operations")
                source = program(*constants("bf16"),
                                 dma_store("bf16", before="io2", after="s0", value="seed"),
                                 operation, dma_wait(before=before, after="s1"), final_state="s1")
                self.rejected(source, "must complete pending DMA before implicit memory operations")

    def test_pure_tile_work_and_mxu_compute_are_allowed_while_pending(self) -> None:
        pure = (f'    %independent = "atlas.virtual_vpu_unary"(%seed) {{kind = "relu"}} '
                f': ({tile("bf16")}) -> {tile("bf16")}')
        matmul = (f'    %product = "atlas.virtual_mxu_matmul"(%x, %w) {{unit = 1 : i32}} '
                  f': ({tile("fp8")}, {tile("fp8")}) -> {tile("bf16")}')
        self.accepted(program(
            *constants("bf16"), load("io2", "s0", "weight"),
            dma_load("bf16", before="s0", after="s1"), pure, matmul,
            reset("s1", "s2", "acc0"), accumulate("s2", "s3", "acc1", "acc0"),
            dma_await("bf16", before="s3", after="s4"),
            readout("s4", "s5", "mxu_tile", "acc1"), final_state="s5"))
        self.accepted(copy().replace(dma_wait(),
            f'    %independent = "atlas.virtual_vpu_unary"(%tile) {{kind = "relu"}} '
            f': ({tile("bf16")}) -> {tile("bf16")}\n' + dma_wait()))

    def test_completed_tiles_can_cross_cfg_edges_without_dma_handles(self) -> None:
        source = copy().replace(dma_store("bf16"),
            f"    cf.br ^exit(%io2, %tile : {STATE}, {tile('bf16')})\n"
            f"  ^exit(%ready: {STATE}, %value: {tile('bf16')}):\n"
            + dma_store("bf16", before="ready", value="value"))
        self.accepted(source)

    def test_cfg_branch_loads_complete_locally_and_join_by_tile(self) -> None:
        source = wrap([
            f'    %io0 = "atlas.virtual_start"() : () -> {STATE}',
            *constants("bf16"),
            f"    cf.cond_br %choose, ^left(%io0 : {STATE}), ^right(%io0 : {STATE})",
            f"  ^left(%left_io: {STATE}):",
            dma_load("bf16", before="left_io", after="left_pending", handle="left_event"),
            dma_await("bf16", before="left_pending", after="left_ready", handle="left_event", result="left_tile"),
            f"    cf.br ^join(%left_ready, %left_tile : {STATE}, {tile('bf16')})",
            f"  ^right(%right_io: {STATE}):",
            dma_load("bf16", before="right_io", after="right_pending", handle="right_event"),
            dma_await("bf16", before="right_pending", after="right_ready", handle="right_event", result="right_tile"),
            f"    cf.br ^join(%right_ready, %right_tile : {STATE}, {tile('bf16')})",
            f"  ^join(%joined_io: {STATE}, %joined_tile: {tile('bf16')}):",
            dma_store("bf16", before="joined_io", value="joined_tile"), dma_wait(),
        ], arguments="%choose: i1")
        first = self.accepted(source)
        self.assertEqual(first, self.accepted(first))
        self.rejected(source.replace("(%right_pending, %right_event)",
                                     "(%right_pending, %left_event)"))

    def test_pending_events_cannot_cross_edges_by_argument_or_capture(self) -> None:
        source = copy()
        changed = source.replace(dma_await("bf16"),
            f"    cf.br ^next(%io1, %pending : {STATE}, {event('bf16')})\n"
            f"  ^next(%edge_io: {STATE}, %edge_event: {event('bf16')}):\n"
            + dma_await("bf16", before="edge_io", handle="edge_event"))
        self.rejected(changed, "must complete pending DMA before block exit")
        changed = source.replace(dma_await("bf16"),
            f"    cf.br ^next(%io1 : {STATE})\n"
            f"  ^next(%edge_io: {STATE}):\n" + dma_await("bf16", before="edge_io"))
        self.rejected(changed, "must complete pending DMA before block exit")
        changed = source.replace(dma_wait(),
            f"    cf.br ^next(%io3 : {STATE})\n"
            f"  ^next(%edge_io: {STATE}):\n" + dma_wait(before="edge_io"))
        self.rejected(changed, "must complete pending DMA before block exit")
        self.rejected(source.replace("@dma()", f"@dma(%external: {event('bf16')})"))

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
            lowered = run("atlas-opt", source, "--lower-atlas-virtual-to-machine")
            if "func.func" in source:
                self.assertEqual(lowered.returncode, 0, lowered.stderr)
                self.assertNotIn('"atlas.virtual_', lowered.stdout)
            else:
                self.assertNotEqual(lowered.returncode, 0, lowered.stdout)
                self.assertTrue(lowered.stderr)
                self.assertEqual(lowered.stdout, "")


if __name__ == "__main__":
    unittest.main()
