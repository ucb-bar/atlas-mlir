"""Logical DMA admission and block-local handle ownership through public APIs.

The two-transfer bound follows the pinned target interface. These checks do not
assign channels, staging addresses, or qualify physical completion ordering.
"""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from atlas_virtual_evaluator import MemoryRegion, RuntimeInputs, Scalar, Tile, VirtualInterfaceError, evaluate, parse_program  # noqa: E402


S, B, F = "!atlas.virtual_state", "!atlas.virtual_bf16", "!atlas.virtual_fp8"
STORE = "!atlas.virtual_dma_store"
ADDRESS, DESTINATION = 0x80000000, 0x80002000
INPUT_BASE, OUTPUT_BASE = 0x90000000, 0x90004000
BF16 = Tile("bf16", (0x3F80,) * 1024)
FP8 = Tile("fp8", (0x38,) * 1024)
RAW = b"\x80\x3f" * 1024
ABI = f"atlas.input_dram_base = {INPUT_BASE} : i64, atlas.output_dram_base = {OUTPUT_BASE} : i64"


def type_for(format: str) -> str:
    return B if format == "bf16" else F


def event(format: str) -> str:
    return f"!atlas.virtual_dma_load_{format}"


def constants(format: str = "bf16", address: int = ADDRESS, size: int | None = None, *, addr: str = "addr", length: str = "size") -> tuple[str, str]:
    return (f"%{addr} = arith.constant {address} : i32",
            f"%{length} = arith.constant {size if size is not None else 2048 if format == 'bf16' else 1024} : i32")


def load(format: str, before: str, after: str, handle: str, addr: str = "addr", size: str = "size") -> str:
    return f'%{after}, %{handle} = "atlas.virtual_dma_load_{format}"(%{before}, %{addr}, %{size}) : ({S}, i32, i32) -> ({S}, {event(format)})'


def ready(format: str, before: str, after: str, handle: str, result: str = "loaded") -> str:
    return f'%{after}, %{result} = "atlas.virtual_dma_await_{format}"(%{before}, %{handle}) : ({S}, {event(format)}) -> ({S}, {type_for(format)})'


def store(format: str, before: str, after: str, handle: str, value: str = "x", addr: str = "addr", size: str = "size") -> str:
    return f'%{after}, %{handle} = "atlas.virtual_dma_store_{format}"(%{before}, %{value}, %{addr}, %{size}) : ({S}, {type_for(format)}, i32, i32) -> ({S}, {STORE})'


def wait(before: str, after: str, handle: str) -> str:
    return f'%{after} = "atlas.virtual_dma_wait"(%{before}, %{handle}) : ({S}, {STORE}) -> {S}'


def input_tile(format: str, before: str = "s0", after: str = "io", value: str = "x", index: int = 0) -> str:
    return f'%{after}, %{value} = "atlas.virtual_input_{format}"(%{before}) {{index = {index} : i32}} : ({S}) -> ({S}, {type_for(format)})'


def output(before: str, after: str, value: str = "x") -> str:
    return f'%{after} = "atlas.virtual_output_bf16"(%{before}, %{value}) {{index = 0 : i32}} : ({S}, {B}) -> {S}'


def wrap(*body: str, final: str, flat: bool = False, arguments: str = "", attributes: str = "") -> str:
    start = f'%s0 = "atlas.virtual_start"() : () -> {S}'
    operations = "\n".join((start, *body))
    if flat:
        return f"module {{\n{operations}\n}}"
    attrs = f" attributes {{{attributes}}}" if attributes else ""
    return f"module {{ func.func @dma({arguments}) -> {S}{attrs} {{\n{operations}\nfunc.return %{final} : {S}\n}} }}"


def read_program(format: str = "bf16", address: int = ADDRESS, size: int | None = None, *, flat: bool = False) -> str:
    return wrap(*constants(format, address, size), load(format, "s0", "s1", "h"), ready(format, "s1", "s2", "h"), final="s2", flat=flat)


def regions(address: int = ADDRESS, size: int = 2048) -> tuple[MemoryRegion, ...]:
    return (MemoryRegion(address, RAW[:size]),)


class VirtualEvaluatorDMAHandleTest(unittest.TestCase):
    def accepted(self, source: str, inputs: RuntimeInputs | None = None):
        inputs = inputs if inputs is not None else RuntimeInputs(memory=regions())
        original_tiles, original_memory = dict(inputs.tiles), inputs.memory
        result = evaluate(parse_program(source), inputs)
        self.assertEqual(dict(inputs.tiles), original_tiles)
        self.assertEqual(inputs.memory, original_memory)
        return result

    def rejected(self, source: str, inputs: RuntimeInputs | None = None, error=VirtualInterfaceError) -> None:
        inputs = inputs if inputs is not None else RuntimeInputs(memory=regions())
        with self.assertRaises(error):
            evaluate(parse_program(source), inputs)

    def test_signed_unsigned_addresses_and_exact_top_span_are_admitted(self) -> None:
        for format, size in (("fp8", 1024), ("bf16", 2048)):
            for address in (-2147483648, ADDRESS, (1 << 32) - size):
                for flat in (False, True):
                    with self.subTest(format=format, address=address, flat=flat):
                        memory = regions(address & 0xFFFFFFFF, size)
                        result = self.accepted(read_program(format, address, flat=flat), RuntimeInputs(memory=memory))
                        self.assertEqual(result.memory, memory)

    def test_recursive_unflagged_addition_proves_modulo_i32_address_and_size(self) -> None:
        proof = ("%a = arith.constant -32 : i32", "%b = arith.constant -2147483616 : i32",
                 "%zero = arith.constant 0 : i32", "%half = arith.constant 1024 : i32",
                 "%sum = arith.addi %a, %b : i32", "%addr = arith.addi %sum, %zero : i32",
                 "%size = arith.addi %half, %half : i32")
        source = wrap(*proof, load("bf16", "s0", "s1", "h"), ready("bf16", "s1", "s2", "h"), final="s2")
        self.accepted(source)
        for flag in ("nsw", "nuw", "nsw, nuw"):
            with self.subTest(flag=flag):
                self.rejected(source.replace("arith.addi %a, %b : i32", f"arith.addi %a, %b overflow<{flag}> : i32"))

    def test_address_alignment_exact_geometry_and_widened_end_are_required(self) -> None:
        for format in ("fp8", "bf16"):
            for address in (0, ADDRESS - 32, ADDRESS + 1, (1 << 32) - 32):
                with self.subTest(format=format, address=address):
                    self.rejected(read_program(format, address))
            for size in (0, 32, -1, 2048 if format == "fp8" else 1024):
                with self.subTest(format=format, size=size):
                    self.rejected(read_program(format, size=size))

    def test_store_address_and_length_are_checked_without_a_load(self) -> None:
        for address, size in ((0, 2048), (ADDRESS + 1, 2048), (ADDRESS, 1024), ((1 << 32) - 32, 2048)):
            with self.subTest(address=address, size=size):
                source = wrap(input_tile("bf16"), *constants("bf16", address, size), store("bf16", "io", "s1", "h"), wait("s1", "s2", "h"), final="s2")
                self.rejected(source, RuntimeInputs({0: BF16}, memory=regions()))

    def test_controls_and_block_arguments_are_not_constant_address_proofs(self) -> None:
        for operand in ("addr", "size"):
            body = list(constants())
            body[0 if operand == "addr" else 1] = f"%{operand} = arith.addi %dynamic, %dynamic : i32"
            source = wrap(*body, load("bf16", "s0", "s1", "h"), ready("bf16", "s1", "s2", "h"), final="s2", arguments="%dynamic: i32")
            self.rejected(source, RuntimeInputs(controls=(Scalar(32, 1024),), memory=regions()))
        source = wrap(*constants(), f"cf.br ^next(%s0, %addr : {S}, i32)",
                      f"^next(%entry: {S}, %passed: i32):", load("bf16", "entry", "s1", "h", "passed"), ready("bf16", "s1", "s2", "h"), final="s2")
        self.rejected(source)

    def test_format_specific_await_store_and_wait_mismatches_fail_parsing(self) -> None:
        source = read_program()
        bad = (source.replace(ready("bf16", "s1", "s2", "h"), ready("fp8", "s1", "s2", "h")),
               source.replace(ready("bf16", "s1", "s2", "h"), wait("s1", "s2", "h")),
               wrap(input_tile("bf16"), *constants(), store("bf16", "io", "s1", "h"), ready("bf16", "s1", "s2", "h"), final="s2"),
               wrap(input_tile("bf16"), *constants("fp8"), store("fp8", "io", "s1", "h"), wait("s1", "s2", "h"), final="s2"))
        for text in bad:
            with self.subTest(source=text):
                with self.assertRaises(VirtualInterfaceError):
                    parse_program(text)

    def test_two_pending_read_transfers_complete_in_either_order_with_overlap(self) -> None:
        for order in (("h0", "h1"), ("h1", "h0")):
            source = wrap(*constants(), load("bf16", "s0", "s1", "h0"), load("bf16", "s1", "s2", "h1"),
                          ready("bf16", "s2", "s3", order[0], "first"), ready("bf16", "s3", "s4", order[1], "second"), final="s4")
            with self.subTest(order=order):
                self.assertEqual(self.accepted(source).memory, regions())
        source = wrap(*constants(), "%small = arith.constant 1024 : i32", load("bf16", "s0", "s1", "h0"), load("fp8", "s1", "s2", "h1", size="small"),
                      ready("fp8", "s2", "s3", "h1", "small_tile"), ready("bf16", "s3", "s4", "h0"), final="s4")
        self.accepted(source)

    def test_two_pending_transfers_are_shared_across_load_and_store_directions(self) -> None:
        for order in (0, 1):
            completion = (ready("bf16", "s2", "s3", "read"), wait("s3", "s4", "write")) if order == 0 else (wait("s2", "s3", "write"), ready("bf16", "s3", "s4", "read"))
            source = wrap(input_tile("bf16"), *constants(), *constants(address=DESTINATION, addr="dst", length="store_size"),
                          load("bf16", "io", "s1", "read"), store("bf16", "s1", "s2", "write", addr="dst", size="store_size"), *completion, final="s4")
            memory = (*regions(), MemoryRegion(DESTINATION, bytes(2048)))
            with self.subTest(order=order):
                result = self.accepted(source, RuntimeInputs({0: BF16}, memory=memory))
                self.assertEqual(result.memory, (*regions(), MemoryRegion(DESTINATION, RAW)))

    def test_disjoint_bf16_and_fp8_stores_can_complete_in_either_order(self) -> None:
        for order in (("wide", "narrow"), ("narrow", "wide")):
            source = wrap(input_tile("bf16"), input_tile("fp8", "io", "io2", "small_tile", 1),
                          *constants(), *constants("fp8", DESTINATION, addr="dst", length="small"),
                          store("bf16", "io2", "s1", "wide"), store("fp8", "s1", "s2", "narrow", "small_tile", "dst", "small"),
                          wait("s2", "s3", order[0]), wait("s3", "s4", order[1]), final="s4")
            memory = (MemoryRegion(ADDRESS, bytes(2048)), MemoryRegion(DESTINATION, bytes(1024)))
            with self.subTest(order=order):
                result = self.accepted(source, RuntimeInputs({0: BF16, 1: FP8}, memory=memory))
                self.assertEqual(result.memory, (MemoryRegion(ADDRESS, RAW), MemoryRegion(DESTINATION, bytes(FP8.bits))))

    def test_a_third_pending_handle_is_rejected_and_completion_releases_capacity(self) -> None:
        prefix = (*constants(), load("bf16", "s0", "s1", "h0"), load("bf16", "s1", "s2", "h1"))
        source = wrap(*prefix, load("bf16", "s2", "s3", "h2"), ready("bf16", "s3", "s4", "h0", "v0"),
                      ready("bf16", "s4", "s5", "h1", "v1"), ready("bf16", "s5", "s6", "h2", "v2"), final="s6")
        self.rejected(source)
        source = wrap(*prefix, ready("bf16", "s2", "s3", "h0", "v0"), load("bf16", "s3", "s4", "h2"),
                      ready("bf16", "s4", "s5", "h2", "v2"), ready("bf16", "s5", "s6", "h1", "v1"), final="s6")
        self.accepted(source)

    def test_load_handle_forks_repeated_completions_and_stale_epochs_are_rejected(self) -> None:
        source = wrap(*constants(), load("bf16", "s0", "s1", "old"), ready("bf16", "s1", "s2", "old", "first"),
                      ready("bf16", "s2", "s3", "old", "fork"), final="s3")
        self.rejected(source)
        source = wrap(*constants(), load("bf16", "s0", "s1", "old"), ready("bf16", "s1", "s2", "old", "first"),
                      load("bf16", "s2", "s3", "new"), ready("bf16", "s3", "s4", "old", "stale"), ready("bf16", "s4", "s5", "new"), final="s5")
        self.rejected(source)

    def test_store_wait_is_consumed_once_and_state_cannot_fork(self) -> None:
        source = wrap(input_tile("bf16"), *constants(), store("bf16", "io", "s1", "h"), wait("s1", "s2", "h"), wait("s2", "s3", "h"), final="s3")
        self.rejected(source, RuntimeInputs({0: BF16}, memory=regions()))
        source = read_program().replace(ready("bf16", "s1", "s2", "h"), ready("bf16", "s0", "s2", "h"))
        self.rejected(source)

    def test_pending_handles_cannot_be_dropped_at_return_flat_end_or_branch(self) -> None:
        for flat in (False, True):
            source = wrap(*constants(), load("bf16", "s0", "s1", "h"), final="s1", flat=flat)
            with self.subTest(flat=flat):
                self.rejected(source)
        source = wrap(input_tile("bf16"), *constants(), store("bf16", "io", "s1", "h"), final="s1")
        self.rejected(source, RuntimeInputs({0: BF16}, memory=regions()))
        source = wrap(*constants(), load("bf16", "s0", "s1", "h"), f"cf.br ^exit(%s1 : {S})", f"^exit(%es: {S}):", final="es")
        self.rejected(source)

    def test_dma_handles_cannot_be_captured_or_passed_across_blocks(self) -> None:
        source = wrap(*constants(), load("bf16", "s0", "s1", "h"), f"cf.br ^exit(%s1 : {S})", f"^exit(%es: {S}):", ready("bf16", "es", "s2", "h"), final="s2")
        self.rejected(source)
        source = source.replace(f"cf.br ^exit(%s1 : {S})", f"cf.br ^exit(%s1, %h : {S}, {event('bf16')})").replace(f"^exit(%es: {S}):", f"^exit(%es: {S}, %passed: {event('bf16')}):").replace("%es, %h", "%es, %passed")
        with self.assertRaises(VirtualInterfaceError):
            parse_program(source)

    def test_untaken_paths_still_require_valid_proofs_and_closed_handles(self) -> None:
        for malformed in ("bad_address", "dropped"):
            body = (*constants(address=0 if malformed == "bad_address" else ADDRESS),
                    f"cf.cond_br %choose, ^good(%s0 : {S}), ^bad(%s0 : {S})",
                    f"^good(%gs: {S}):", f"func.return %gs : {S}", f"^bad(%bs: {S}):", load("bf16", "bs", "s1", "h"))
            tail = (ready("bf16", "s1", "s2", "h"),) if malformed == "bad_address" else ()
            source = wrap(*body, *tail, final="s2" if tail else "s1", arguments="%choose: i1")
            with self.subTest(malformed=malformed):
                self.rejected(source, RuntimeInputs(controls=(Scalar(1, 1),)))

    def test_unmapped_memory_on_an_untaken_valid_path_is_not_accessed(self) -> None:
        source = wrap(*constants(), f"cf.cond_br %choose, ^good(%s0 : {S}), ^read(%s0 : {S})",
                      f"^good(%gs: {S}):", f"func.return %gs : {S}", f"^read(%rs: {S}):", load("bf16", "rs", "s1", "h"), ready("bf16", "s1", "s2", "h"), final="s2", arguments="%choose: i1")
        self.accepted(source, RuntimeInputs(controls=(Scalar(1, 1),)))

    def test_loop_visits_create_fresh_dma_handle_epochs(self) -> None:
        source = wrap(*constants(), "%zero = arith.constant 0 : i32", "%one = arith.constant 1 : i32", "%limit = arith.constant 2 : i32",
                      f"cf.br ^loop(%s0, %zero : {S}, i32)", f"^loop(%ls: {S}, %i: i32):",
                      load("bf16", "ls", "s1", "h"), ready("bf16", "s1", "s2", "h"), "%next = arith.addi %i, %one : i32", "%again = arith.cmpi ult, %next, %limit : i32",
                      f"cf.cond_br %again, ^loop(%s2, %next : {S}, i32), ^exit(%s2 : {S})", f"^exit(%es: {S}):", final="es")
        self.assertEqual(self.accepted(source).memory, regions())

    def test_implicit_io_and_pack_are_blocked_until_pending_dma_completes(self) -> None:
        for direction in ("load", "store"):
            launch = load("bf16", "io", "s1", "h") if direction == "load" else store("bf16", "io", "s1", "h")
            for kind in ("input", "output", "pack"):
                hidden = input_tile("bf16", "s1", "hidden", "extra", 1) if kind == "input" else output("s1", "hidden") if kind == "output" else f'%packed = "atlas.virtual_pack_fp8"(%x) {{scale_code = 127 : i32}} : ({B}) -> {F}'
                before = "s1" if kind == "pack" else "hidden"
                completion = ready("bf16", before, "s2", "h") if direction == "load" else wait(before, "s2", "h")
                source = wrap(input_tile("bf16"), *constants(), launch, hidden, completion, final="s2")
                tiles = {0: BF16, 1: BF16} if kind == "input" else {0: BF16}
                with self.subTest(direction=direction, kind=kind):
                    self.rejected(source, RuntimeInputs(tiles, memory=regions()))

    def test_other_vpu_and_mxu_work_are_allowed_while_dma_is_pending(self) -> None:
        relu = f'%positive = "atlas.virtual_vpu_unary"(%x) {{kind = "relu"}} : ({B}) -> {B}'
        add = f'%sum = "atlas.virtual_vpu_binary"(%positive, %x) {{kind = "add"}} : ({B}, {B}) -> {B}'
        matmul = f'%product = "atlas.virtual_mxu_matmul"(%w, %w) {{unit = 1 : i32}} : ({F}, {F}) -> {B}'
        source = wrap(input_tile("bf16"), input_tile("fp8", "io", "io2", "w", 1), *constants(), load("bf16", "io2", "s1", "h"), relu, add, matmul,
                      ready("bf16", "s1", "s2", "h"), output("s2", "done", "sum"), final="done")
        result = self.accepted(source, RuntimeInputs({0: BF16, 1: Tile("fp8", (0,) * 1024)}, memory=regions()))
        self.assertEqual(dict(result.outputs), {0: Tile("bf16", (0x4000,) * 1024)})
        for unit in (0, 1):
            acc = f"!atlas.virtual_mxu_acc<{unit}>"
            seed = f'%m1, %acc = "atlas.virtual_mxu_load_acc_bf16"(%s1, %x) {{unit = {unit} : i32}} : ({S}, {B}) -> ({S}, {acc})'
            read = f'%m2, %copied = "atlas.virtual_mxu_readout_bf16"(%m1, %acc) : ({S}, {acc}) -> ({S}, {B})'
            source = wrap(input_tile("bf16"), *constants(), load("bf16", "io", "s1", "h"), seed, read,
                          ready("bf16", "m2", "s2", "h"), output("s2", "done", "copied"), final="done")
            with self.subTest(unit=unit):
                result = self.accepted(source, RuntimeInputs({0: BF16}, memory=regions()))
                self.assertEqual(dict(result.outputs), {0: BF16})

    def test_executed_overlapping_pending_ranges_involving_a_store_conflict(self) -> None:
        for first, second in (("load", "store"), ("store", "load"), ("store", "store")):
            op0 = load("bf16", "io", "s1", "h0") if first == "load" else store("bf16", "io", "s1", "h0")
            op1 = load("bf16", "s1", "s2", "h1") if second == "load" else store("bf16", "s1", "s2", "h1")
            complete0 = ready("bf16", "s2", "s3", "h0", "v0") if first == "load" else wait("s2", "s3", "h0")
            complete1 = ready("bf16", "s3", "s4", "h1", "v1") if second == "load" else wait("s3", "s4", "h1")
            source = wrap(input_tile("bf16"), *constants(), op0, op1, complete0, complete1, final="s4")
            with self.subTest(first=first, second=second):
                self.rejected(source, RuntimeInputs({0: BF16}, memory=regions()))
        source = wrap(input_tile("bf16"), *constants(), f"cf.cond_br %choose, ^good(%io : {S}), ^bad(%io : {S})", f"^good(%gs: {S}):", f"func.return %gs : {S}",
                      f"^bad(%bs: {S}):", load("bf16", "bs", "s1", "h0"), store("bf16", "s1", "s2", "h1"), ready("bf16", "s2", "s3", "h0"), wait("s3", "s4", "h1"), final="s4", arguments="%choose: i1")
        self.accepted(source, RuntimeInputs({0: BF16}, (Scalar(1, 1),)))
        self.rejected(source, RuntimeInputs({0: BF16}, (Scalar(1, 0),), regions()))

    def test_completed_serial_store_then_load_of_the_same_range_is_allowed(self) -> None:
        source = wrap(input_tile("bf16"), *constants(), store("bf16", "io", "s1", "write"), wait("s1", "s2", "write"),
                      load("bf16", "s2", "s3", "read"), ready("bf16", "s3", "s4", "read"), output("s4", "done", "loaded"), final="done")
        result = self.accepted(source, RuntimeInputs({0: BF16}, memory=(MemoryRegion(ADDRESS, bytes(2048)),)))
        self.assertEqual(dict(result.outputs), {0: BF16})

    def test_completed_mixed_boundary_aliases_share_memory(self) -> None:
        for direction, address in (("store", INPUT_BASE), ("store", OUTPUT_BASE), ("load", OUTPUT_BASE)):
            launch = store("bf16", "io", "s1", "h") if direction == "store" else load("bf16", "io", "s1", "h")
            completion = wait("s1", "s2", "h") if direction == "store" else ready("bf16", "s1", "s2", "h")
            source = wrap(input_tile("bf16"), *constants(address=address), launch, completion, output("s2", "done"), final="done", attributes=ABI)
            with self.subTest(direction=direction, address=address):
                memory = (MemoryRegion(OUTPUT_BASE, RAW),) if direction == "load" else ()
                self.assertEqual(dict(self.accepted(source, RuntimeInputs({0: BF16}, memory=memory)).outputs), {0: BF16})

    def test_explicit_and_implicit_reads_can_alias_and_absent_bases_are_not_invented(self) -> None:
        source = wrap(input_tile("bf16"), *constants(address=INPUT_BASE), load("bf16", "io", "s1", "h"), ready("bf16", "s1", "s2", "h"), output("s2", "done", "loaded"), final="done", attributes=ABI)
        result = self.accepted(source, RuntimeInputs({0: BF16}, memory=regions(INPUT_BASE)))
        self.assertEqual(dict(result.outputs), {0: BF16})
        source = wrap(input_tile("bf16"), *constants(address=INPUT_BASE), store("bf16", "io", "s1", "h"), wait("s1", "s2", "h"), output("s2", "done"), final="done")
        self.accepted(source, RuntimeInputs({0: BF16}, memory=regions(INPUT_BASE)))

    def test_supplied_regions_overlapping_known_implicit_outputs_observe_writes(self) -> None:
        source = wrap(input_tile("bf16"), output("io", "done"), final="done", attributes=ABI)
        memory = (MemoryRegion(OUTPUT_BASE, bytes(2048)),)
        self.assertEqual(self.accepted(source, RuntimeInputs({0: BF16}, memory=memory)).memory, regions(OUTPUT_BASE))
        self.accepted(source, RuntimeInputs({0: BF16}, memory=regions(OUTPUT_BASE + 2048)))

    def test_long_constant_add_chain_is_an_admitted_address_proof(self) -> None:
        proof = ["%origin = arith.constant 2147483648 : i32", "%zero = arith.constant 0 : i32", "%size = arith.constant 2048 : i32"]
        previous = "origin"
        for index in range(1500):
            current = f"address{index}"
            proof.append(f"%{current} = arith.addi %{previous}, %zero : i32")
            previous = current
        source = wrap(*proof, load("bf16", "s0", "s1", "h", previous), ready("bf16", "s1", "s2", "h"), final="s2")
        self.assertEqual(self.accepted(source).memory, regions())

    def test_load_and_store_cannot_alias_a_bound_control_mailbox(self) -> None:
        for direction in ("load", "store"):
            launch = load("bf16", "io", "s1", "h") if direction == "load" else store("bf16", "io", "s1", "h")
            completion = ready("bf16", "s1", "s2", "h") if direction == "load" else wait("s1", "s2", "h")
            source = wrap(input_tile("bf16"), *constants(), launch, completion, final="s2", arguments="%choose: i1", attributes=f"atlas.control_dram_base = {ADDRESS} : i64")
            with self.subTest(direction=direction):
                self.rejected(source, RuntimeInputs({0: BF16}, (Scalar(1, 1),), regions()))
            source = wrap(input_tile("bf16"), *constants(), f"cf.cond_br %choose, ^good(%io : {S}), ^dma(%io : {S})",
                          f"^good(%gs: {S}):", f"func.return %gs : {S}", f"^dma(%ds: {S}):", launch.replace("%io,", "%ds,"), completion,
                          final="s2", arguments="%choose: i1", attributes=f"atlas.control_dram_base = {ADDRESS} : i64")
            with self.subTest(direction=direction, untaken=True):
                self.rejected(source, RuntimeInputs({0: BF16}, (Scalar(1, 1),)))

    def test_declared_boundary_bases_reject_noninteger_negative_and_overflow_values(self) -> None:
        for kind in ("input", "output", "control"):
            for value in ('"invalid"', "-1 : i64", "4294967296 : i64", "4294967295 : i64"):
                attributes = f"atlas.{kind}_dram_base = {value}"
                if kind == "control":
                    source = wrap(final="s0", arguments="%choose: i1", attributes=attributes)
                    inputs = RuntimeInputs(controls=(Scalar(1, 1),))
                else:
                    source = wrap(input_tile("bf16"), output("io", "done"), final="done", attributes=attributes)
                    inputs = RuntimeInputs({0: BF16})
                with self.subTest(kind=kind, value=value):
                    self.rejected(source, inputs)

    def test_known_input_read_alias_requires_matching_initial_memory_bytes(self) -> None:
        for format, tile, payload in (("bf16", BF16, RAW), ("fp8", FP8, bytes(FP8.bits))):
            source = wrap(input_tile(format), *constants(format, INPUT_BASE), load(format, "io", "s1", "h"), ready(format, "s1", "s2", "h"),
                          final="s2", attributes=f"atlas.input_dram_base = {INPUT_BASE} : i64")
            memory = (MemoryRegion(INPUT_BASE, payload),)
            with self.subTest(format=format):
                self.assertEqual(self.accepted(source, RuntimeInputs({0: tile}, memory=memory)).memory, memory)
                inconsistent = bytes([payload[0] ^ 1]) + payload[1:]
                self.rejected(source, RuntimeInputs({0: tile}, memory=(MemoryRegion(INPUT_BASE, inconsistent),)))

    def test_partial_known_input_snapshots_must_be_consistent_without_a_dma_access(self) -> None:
        for format, tile, payload in (("bf16", BF16, RAW), ("fp8", FP8, bytes(FP8.bits))):
            source = wrap(input_tile(format), final="io", attributes=f"atlas.input_dram_base = {INPUT_BASE} : i64")
            memory = (MemoryRegion(INPUT_BASE + 13, payload[13:94]),)
            with self.subTest(format=format):
                self.assertEqual(self.accepted(source, RuntimeInputs({0: tile}, memory=memory)).memory, memory)
                inconsistent = payload[13:93] + bytes([payload[93] ^ 1])
                self.rejected(source, RuntimeInputs({0: tile}, memory=(MemoryRegion(INPUT_BASE + 13, inconsistent),)))

    def test_absent_input_base_keeps_boundary_tiles_separate_from_memory_bytes(self) -> None:
        source = wrap(input_tile("bf16"), output("io", "done"), final="done")
        memory = (MemoryRegion(INPUT_BASE, bytes(2048)),)
        result = self.accepted(source, RuntimeInputs({0: BF16}, memory=memory))
        self.assertEqual(dict(result.outputs), {0: BF16})
        self.assertEqual(result.memory, memory)


if __name__ == "__main__":
    unittest.main()
