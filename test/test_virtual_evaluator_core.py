"""Virtual reference semantics versus literals, compiler schedules, LLVM objects and the selected core.

Each row carries independent literal expectations. LLVM-object agreement needs ATLAS_LLVM_BIN; selected-core
comparison also needs ATLAS_ARC_MODEL, ATLAS_ARC_STATE and ATLAS_MODELIR_ROOT and uses the existing runner.
The physical PACK probe checks converter bits and register/DRAM transport against an independent permutation;
it does not qualify virtual PACK lowering or an FP8 consumer.
"""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path
import sys
import unittest
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from atlas_virtual_evaluator import (  # noqa: E402
    EvaluationResult, MemoryRegion, RuntimeInputs, Scalar, Tile, compare_results, evaluate, evaluate_tile_operation, operation_name,
    parse_program,
)
from test_virtual_dma import copy, dma_await, dma_load, dma_store, dma_wait, wrap  # noqa: E402
from test_virtual_dma_lowering import MARKER  # noqa: E402
from test_virtual_evaluator_arithmetic import ADD_CASES, Stream as MxuStream, inputs as mxu_inputs, sparse  # noqa: E402
from test_virtual_evaluator_memory import BASE, Stream, repeated  # noqa: E402
from test_virtual_evaluator_resources import on_unit  # noqa: E402
from test_virtual_lowering import emitted, lower, object_words, run  # noqa: E402
from test_virtual_mxu_extended import seeded_chain  # noqa: E402
from test_virtual_mxu_handles import BF16, FP8, STATE, accumulate, load, readout, reset  # noqa: E402
from test_virtual_mxu_lowering import instructions  # noqa: E402

EXAMPLES = ROOT / "test/examples"
INPUT_BASE, OUTPUT_BASE, CONTROL_BASE = 0x90000000, 0x90004000, 0x90008000
RELU_OUTPUT_BASE = 0x90001000
# The selected core prunes io_halted; the runner requires an explicit observed i1 manifest state.
HALT_SIGNAL = "scalar/halt_now"
SCHEDULED_PHASE, SCHEDULED_CYCLES = 3, 100000
RANDOM_SEEDS = range(4)
VIRTUAL_OPERATION = r'(?m)^\s*%[^\n=]+\s*=\s*"atlas\.virtual_'
FP8_TO_BF16 = {0x00: 0x0000, 0x30: 0x3F00, 0x38: 0x3F80, 0x40: 0x4000, 0xB8: 0xBF80}
TRIPLE_BF16 = {0x00: 0x0000, 0x30: 0x3FC0, 0x38: 0x4040, 0x40: 0x40C0, 0xB8: 0xC040}
TWICE_BF16 = {0x00: 0x0000, 0x30: 0x3F80, 0x38: 0x4000, 0x40: 0x4080, 0xB8: 0xC000}
EIGHT_FP8 = {0x00: 0x00, 0x30: 0x48, 0x38: 0x50, 0x40: 0x58, 0xB8: 0xD0}
FP8_CODES = (0x00, 0x30, 0x38, 0x40, 0xB8)
READY_CODES = (0xBF80, 0x0000, 0x3F80, 0x4000, 0xBF00, 0x3F00)
READY_SUM = {0xBF80: 0x3F00, 0x0000: 0x3F00, 0x3F80: 0x3FC0, 0x4000: 0x4020, 0xBF00: 0x3F00, 0x3F00: 0x3F80}
# Source-derived unit-scale FP8Pack expectations: even/odd RNE ties, exponent carry, flush around the
# minimum normal, saturation and specials. Rounded 480 is 0x7e here, unlike the MXU converter's 0x7f.
PACK_CASES = (
    (0x0000, 0x00), (0x8000, 0x00), (0x0001, 0x00), (0x007F, 0x00), (0x8001, 0x00), (0x807F, 0x00), (0x7F81, 0x00),
    (0x7FC0, 0x00), (0xFF81, 0x00), (0xFFC0, 0x00), (0x7F80, 0x7E), (0xFF80, 0xFE), (0x3F80, 0x38), (0xBF80, 0xB8),
    (0x3F87, 0x38), (0x3F88, 0x38), (0x3F89, 0x39), (0x3F98, 0x3A), (0xBF98, 0xBA), (0x3FF7, 0x3F), (0x3FF8, 0x40),
    (0xBFF8, 0xC0), (0x3C60, 0x00), (0x3C77, 0x00), (0x3C78, 0x08), (0x3C80, 0x08), (0xBC77, 0x00), (0xBC78, 0x88),
    (0x43E0, 0x7E), (0x43E8, 0x7E), (0x43E9, 0x7E), (0x43F0, 0x7E), (0x7F7F, 0x7E), (0xC3E0, 0xFE), (0xC3E8, 0xFE),
    (0xC3E9, 0xFE), (0xC3F0, 0xFE), (0xFF7F, 0xFE),
)
PACK_SOURCE = EXAMPLES / "vpu_pack_reference.mlir"
PACK_VIRTUAL = '''module {
  %s = "atlas.virtual_start"() : () -> !atlas.virtual_state
  %s1, %a = "atlas.virtual_input_bf16"(%s) {index = 0 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_bf16)
  %packed = "atlas.virtual_pack_fp8"(%a) {scale_code = 127 : i32} : (!atlas.virtual_bf16) -> !atlas.virtual_fp8
}'''
# FP8 values deliberately have no consumer in this lowering smoke check.
FP8_SMOKE = '''module {
  func.func @unused_fp8() -> !atlas.virtual_state attributes {atlas.input_dram_base = 2415919104 : i64, atlas.output_dram_base = 2415935488 : i64} {
    %s = "atlas.virtual_start"() : () -> !atlas.virtual_state
    %s1, %a = "atlas.virtual_input_bf16"(%s) {index = 0 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_bf16)
    %s2, %f = "atlas.virtual_input_fp8"(%s1) {index = 1 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %packed = "atlas.virtual_pack_fp8"(%a) {scale_code = 127 : i32} : (!atlas.virtual_bf16) -> !atlas.virtual_fp8
    %s3 = "atlas.virtual_output_bf16"(%s2, %a) {index = 0 : i32} : (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state
    return %s3 : !atlas.virtual_state
  }
}'''


def tile_bytes(tile: Tile) -> bytes:
    # A BF16 register pair carries the left 16 columns, then the right 16.
    return b"".join(tile.bits[row * 32 + col].to_bytes(2, "little") for half in (0, 16) for row in range(32) for col in range(half, half + 16))


def relu(tile: Tile) -> Tile:
    return Tile("bf16", tuple(0 if bits & 0x8000 else bits for bits in tile.bits))


def compare_result(expected: EvaluationResult, captured, *, output_base: int = OUTPUT_BASE) -> None:
    for index, tile in expected.outputs.items():
        observed = captured(output_base + index * 2048, 2048)
        if len(observed) != 2048:
            raise AssertionError(f"output {index}: expected 2048 bytes, got {len(observed)}")
        for row in range(32):
            for col in range(32):
                offset = (col // 16) * 1024 + (row * 16 + col % 16) * 2
                bits = int.from_bytes(observed[offset:offset + 2], "little")
                wanted = tile.bits[row * 32 + col]
                if bits != wanted:
                    raise AssertionError(f"output {index} tile[{row},{col}]: expected 0x{wanted:04x}, got 0x{bits:04x}")
    for region in expected.memory:
        observed = captured(region.address, len(region.data))
        if len(observed) != len(region.data):
            raise AssertionError(f"memory at 0x{region.address:08x}: expected {len(region.data)} bytes, got {len(observed)}")
        for offset, (wanted, actual) in enumerate(zip(region.data, observed)):
            if wanted != actual:
                raise AssertionError(f"memory at 0x{region.address + offset:08x}: expected 0x{wanted:02x}, got 0x{actual:02x}")


@contextmanager
def selected_core():
    keys = ("ATLAS_ARC_MODEL", "ATLAS_ARC_STATE", "ATLAS_MODELIR_ROOT", "ATLAS_LLVM_BIN")
    missing = [key for key in keys if not os.environ.get(key)]
    if missing:
        message = "selected-core comparison requires " + ", ".join(missing)
        if os.environ.get("ATLAS_REQUIRE_VIRTUAL_CORE") == "1":
            raise RuntimeError(message)
        raise unittest.SkipTest(message)
    model, state, modelir = (Path(os.environ[key]).resolve(strict=True) for key in keys[:3])
    previous = Path.cwd()
    sys.path.insert(0, str(modelir))
    try:
        os.chdir(modelir)
        from mlc.backends import cosim_atlas

        original = cosim_atlas.CosimCore

        class SelectedCore(original):
            def poke(self, name: str, value: int) -> None:
                if name in ("io_dmaTL_d_bits_opcode", "io_dmaTL_d_bits_size"):
                    if name in self._S:
                        raise AssertionError(f"unexpected live ARC input {name}")
                    return
                super().poke(name, value)

        cosim_atlas.CosimCore = SelectedCore
        try:
            yield lambda words, preload, max_cycles: cosim_atlas.run_program(model, state, words, preload=preload, max_cycles=max_cycles,
                                                                       halt_signal=HALT_SIGNAL)
        finally:
            cosim_atlas.CosimCore = original
    finally:
        os.chdir(previous)
        sys.path.remove(str(modelir))


# -- sources and independent literals per family ---------------------------------------------------------------

def relu_case(phase: int):
    bits = tuple(0 if (i + phase) % 19 == 0 else (0x3E00 + i + phase) | (0x8000 if (i // 32 + i + phase) % 2 else 0) for i in range(1024))
    tile = Tile("bf16", bits)
    guards = tuple(MemoryRegion(address, bytes([value]) * 64) for address, value in (
        (INPUT_BASE - 64, 0x5A), (INPUT_BASE + 2048, 0x6B), (RELU_OUTPUT_BASE - 64, 0x7C), (RELU_OUTPUT_BASE + 4096, 0x8D)))
    inputs = RuntimeInputs({0: tile}, memory=(MemoryRegion(INPUT_BASE, tile_bytes(tile)), *guards))
    return inputs, EvaluationResult({0: tile, 1: relu(tile)}, inputs.memory), (64, 128)


def relu_mutated(inputs: RuntimeInputs) -> EvaluationResult:
    # ReLU changed to MOV publishes X twice and preserves the guards.
    return EvaluationResult({0: inputs.tiles[0], 1: inputs.tiles[0]}, inputs.memory)


STATE_TYPE, PAIR = "!atlas.virtual_state", "!atlas.virtual_state, !atlas.virtual_bf16"


def cfg_kernel(arguments: str, body: str) -> str:
    return f'''module {{
  func.func @cfg({arguments}) -> {STATE_TYPE} attributes {{
    atlas.input_dram_base = {INPUT_BASE} : i64,
    atlas.output_dram_base = {OUTPUT_BASE} : i64,
    atlas.control_dram_base = {CONTROL_BASE} : i64}} {{
    %s0 = "atlas.virtual_start"() : () -> {STATE_TYPE}
    %s1, %a0 = "atlas.virtual_input_bf16"(%s0) {{index = 0 : i32}} : ({STATE_TYPE}) -> ({PAIR})
    %s2, %b0 = "atlas.virtual_input_bf16"(%s1) {{index = 1 : i32}} : ({STATE_TYPE}) -> ({PAIR})
    {body}
  }}
}}
'''


def cfg_selection(arguments: str, condition: str) -> str:
    return cfg_kernel(arguments, f'''
    {condition}
    cf.cond_br %choose, ^left(%s2, %a0 : {PAIR}), ^right(%s2, %b0 : {PAIR})
  ^left(%ls: {STATE_TYPE}, %a: !atlas.virtual_bf16):
    cf.br ^join(%ls, %a : {PAIR})
  ^right(%rs: {STATE_TYPE}, %b: !atlas.virtual_bf16):
    cf.br ^join(%rs, %b : {PAIR})
  ^join(%js: {STATE_TYPE}, %selected: !atlas.virtual_bf16):
    %out = "atlas.virtual_output_bf16"(%js, %selected) {{index = 0 : i32}} : ({PAIR}) -> {STATE_TYPE}
    return %out : {STATE_TYPE}''')


def cfg_loop(swap: bool) -> str:
    bf16 = "!atlas.virtual_bf16"
    carried = f"{PAIR}, {bf16}, i32" if swap else f"{PAIR}, i32"
    initial = "%s2, %a0, %b0, %zero" if swap else "%s2, %a0, %zero"
    args = f"%ls: {STATE_TYPE}, %a: {bf16}, " + (f"%b: {bf16}, " if swap else "") + "%i: i32"
    body_args = f"%bs: {STATE_TYPE}, %ba: {bf16}, " + (f"%bb: {bf16}, " if swap else "") + "%bi: i32"
    advance = "" if swap else f'%next = "atlas.virtual_vpu_unary"(%ba) {{kind = "relu"}} : ({bf16}) -> {bf16}'
    backedge = "%bs, %bb, %ba, %j" if swap else "%bs, %next, %j"
    exit_types = f"{PAIR}, {bf16}" if swap else PAIR
    exit_values = "%ls, %a, %b" if swap else "%ls, %a"
    exit_args = f"%es: {STATE_TYPE}, %x: {bf16}" + (f", %y: {bf16}" if swap else "")
    second = f'%out1 = "atlas.virtual_output_bf16"(%out0, %y) {{index = 1 : i32}} : ({PAIR}) -> {STATE_TYPE}' if swap else ""
    return cfg_kernel("%limit: i32", f'''
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    cf.br ^loop({initial} : {carried})
  ^loop({args}):
    %more = arith.cmpi ult, %i, %limit : i32
    cf.cond_br %more, ^body({exit_values}, %i : {carried}), ^exit({exit_values} : {exit_types})
  ^body({body_args}):
    {advance}
    %j = arith.addi %bi, %one : i32
    cf.br ^loop({backedge} : {carried})
  ^exit({exit_args}):
    %out0 = "atlas.virtual_output_bf16"(%es, %x) {{index = 0 : i32}} : ({PAIR}) -> {STATE_TYPE}
    {second}
    return %out{1 if swap else 0} : {STATE_TYPE}''')


def cfg_cases():
    """(name, source, control widths, ((controls, selected output tiles), ...)); tile 2 is ReLU(input 0)."""
    cases = [(f"constant_i1_{bit}", cfg_selection("", f"%choose = arith.constant {bit} : i1"), (), (((), (0 if bit else 1,)),)) for bit in (0, 1)]
    cases.append(("runtime_i1", cfg_selection("%choose: i1", ""), (1,), (((0,), (1,)), ((1,), (0,)))))
    boundaries = ((0x80000000, 0x7FFFFFFF), (0x7FFFFFFF, 0x80000000), (0xFFFFFFFF, 0), (0x80000000, 0x80000000))
    signed = lambda bits: bits if bits < 0x80000000 else bits - 0x100000000
    predicates = {"eq": lambda a, b: a == b, "ne": lambda a, b: a != b,
                  "slt": lambda a, b: signed(a) < signed(b), "sle": lambda a, b: signed(a) <= signed(b),
                  "sgt": lambda a, b: signed(a) > signed(b), "sge": lambda a, b: signed(a) >= signed(b),
                  "ult": lambda a, b: a < b, "ule": lambda a, b: a <= b, "ugt": lambda a, b: a > b, "uge": lambda a, b: a >= b}
    for predicate, reference in predicates.items():
        source = cfg_selection("%lhs: i32, %rhs: i32", f"%choose = arith.cmpi {predicate}, %lhs, %rhs : i32")
        cases.append((f"cmpi_{predicate}", source, (32, 32), tuple((bits, (0 if reference(*bits) else 1,)) for bits in boundaries)))
    wrapping = cfg_selection("%lhs: i32, %rhs: i32", '''%sum = arith.addi %lhs, %rhs : i32
    %zero = arith.constant 0 : i32
    %choose = arith.cmpi eq, %sum, %zero : i32''')
    cases.append(("wrapping_addi", wrapping, (32, 32), (((0xFFFFFFFF, 1), (0,)), ((0x80000000, 0x80000000), (0,)), ((0xFFFFFFFF, 2), (1,)))))
    for swap in (False, True):
        variants = tuple(((count,), ((1, 0) if count % 2 else (0, 1)) if swap else ((2,) if count else (0,))) for count in (0, 1, 2, 3))
        cases.append(("tile_swap" if swap else "relu_loop", cfg_loop(swap), (32,), variants))
    return cases


def cfg_case(widths: tuple[int, ...], controls: tuple[int, ...], indices: tuple[int, ...]):
    def case(phase: int):
        tiles = {index: Tile("bf16", tuple(
            0 if (i + phase) % 19 == 0 else (0x3E80 + index * 0x400 + (i + phase) % 127) | (0x8000 if (i + i // 32 + phase) % 2 else 0)
            for i in range(1024))) for index in (0, 1)}
        mailbox = bytearray(b"\xC3" * 1024)
        for index, bits in enumerate(controls):
            mailbox[index * 4:index * 4 + 4] = bits.to_bytes(4, "little")
        memory = [MemoryRegion(INPUT_BASE + index * 2048, tile_bytes(tile)) for index, tile in tiles.items()]
        memory.append(MemoryRegion(CONTROL_BASE, mailbox))
        memory.extend(MemoryRegion(address, bytes([value]) * 64) for address, value in (
            (INPUT_BASE - 64, 0x51), (INPUT_BASE + 4096, 0x62), (OUTPUT_BASE - 64, 0x73), (OUTPUT_BASE + 4096, 0x84),
            (CONTROL_BASE - 64, 0x95), (CONTROL_BASE + 1024, 0xA6)))
        inputs = RuntimeInputs(tiles, tuple(Scalar(width, bits) for width, bits in zip(widths, controls)), tuple(memory))
        selectable = {**tiles, 2: relu(tiles[0])}
        return inputs, EvaluationResult({index: selectable[selected] for index, selected in enumerate(indices)}, inputs.memory), (None, 64 * len(indices))
    return case


def dma_pending_work() -> str:
    return wrap([
        f'%io0 = "atlas.virtual_start"() : () -> {STATE}',
        "%first_addr = arith.constant -2147483648 : i32",
        "%later_addr = arith.constant -2147481600 : i32",
        "%out_addr = arith.constant -2147475456 : i32",
        "%copy_addr = arith.constant -2147473408 : i32",
        "%size = arith.constant 2048 : i32",
        dma_load("bf16", addr="first_addr", handle="first_event"),
        dma_await("bf16", handle="first_event", result="ready"),
        dma_load("bf16", before="io2", after="io3", addr="later_addr", handle="later_event"),
        '%positive = "atlas.virtual_vpu_unary"(%ready) {kind = "relu"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16',
        dma_await("bf16", before="io3", after="io4", handle="later_event", result="later"),
        '%sum = "atlas.virtual_vpu_binary"(%positive, %later) {kind = "add"} : (!atlas.virtual_bf16, !atlas.virtual_bf16) -> !atlas.virtual_bf16',
        dma_store("bf16", before="io4", after="io5", value="sum", handle="first_store").replace("%addr,", "%out_addr,"),
        '%copied = "atlas.virtual_vpu_unary"(%sum) {kind = "mov"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16',
        dma_wait(before="io5", after="io6", handle="first_store"),
        dma_store("bf16", before="io6", after="io7", value="copied", handle="last_store").replace("%addr,", "%copy_addr,"),
        dma_wait(before="io7", after="io8", handle="last_store"),
    ], final="io8")


def dma_sources() -> dict[str, str]:
    sources = {"bf16_relu": (EXAMPLES / "virtual_dma_tiles.mlir").read_text(),
               "fp8_copy": copy("fp8").replace("%size = arith.constant 1024 : i32", "%size = arith.constant 1024 : i32\n%destination = arith.constant -2147481600 : i32")
                                      .replace('"atlas.virtual_dma_store_fp8"(%io2, %tile, %addr,', '"atlas.virtual_dma_store_fp8"(%io2, %tile, %destination,'),
               "pending_work": dma_pending_work()}
    for fmt, filename in (("fp8", "virtual_dma_mxu.mlir"), ("bf16", "virtual_dma_mxu_bf16.mlir")):
        for unit in (0, 1):
            source = (EXAMPLES / filename).read_text().replace("unit = 0 : i32", f"unit = {unit} : i32")
            for handle in ("weight", "acc"):
                source = source.replace(f"virtual_mxu_{handle}<0>", f"virtual_mxu_{handle}<{unit}>")
            sources[f"mxu_{fmt}_{unit}"] = source
    return sources


def patterned(codes, phase: int, fmt: str = "bf16") -> Tile:
    return Tile(fmt, tuple(codes[(row * 7 + col * 3 + col // 16 + phase) % len(codes)] for row in range(32) for col in range(32)))


def mapped(buffers: dict[int, bytes], writes: dict[int, bytes]):
    """Inputs over `buffers` with guards around each contiguous block, and the memory after `writes`."""
    regions = [MemoryRegion(address, data) for address, data in buffers.items()]
    blocks = []
    for address, data in sorted(buffers.items()):
        end = address + len(data)
        if blocks and blocks[-1][1] == address:
            blocks[-1] = (blocks[-1][0], end)
        else:
            blocks.append((address, end))
    for start, end in blocks:
        regions.extend((MemoryRegion(start - 64, b"\x51" * 64), MemoryRegion(end, b"\x62" * 64)))
    expected = tuple(MemoryRegion(region.address, writes[region.address] + region.data[len(writes[region.address]):])
                     if region.address in writes else region for region in regions)
    return RuntimeInputs(memory=tuple(regions)), expected


def dma_case(name: str):
    def case(phase: int):
        if name == "bf16_relu":
            tile = patterned((0xBF80, 0x3F80, 0x8000, 0x0001, 0x8001, 0x7FC1, 0xFFC1, 0x4000), phase)
            inputs, expected = mapped({0x80000800: tile_bytes(tile), 0x80001000: b"\xA5" * 2048}, {0x80001000: tile_bytes(relu(tile))})
            traffic = (64, 64)
        elif name == "fp8_copy":
            # Raw DMA transports every FP8 encoding, including signed zeros/NaNs.
            tile = Tile("fp8", tuple((index + phase) % 256 for index in range(1024)))
            inputs, expected = mapped({0x80000000: bytes(tile.bits) + b"\xC3" * 1024, 0x80000800: b"\xA5" * 1024 + b"\xD4" * 1024}, {0x80000800: bytes(tile.bits)})
            traffic = (32, 32)
        elif name == "pending_work":
            ready = patterned(READY_CODES, phase)
            result = tile_bytes(Tile("bf16", tuple(READY_SUM[bits] for bits in ready.bits)))
            inputs, expected = mapped({0x80000000: tile_bytes(ready), 0x80000800: tile_bytes(Tile("bf16", (0x3F00,) * 1024)),
                                       0x80002000: b"\xA5" * 2048, 0x80002800: b"\xB6" * 2048}, {0x80002000: result, 0x80002800: result})
            traffic = (128, 128)
        else:
            fmt = name.split("_")[1]
            x, shift = patterned(FP8_CODES, phase, "fp8"), phase % 7
            weight = bytes(0x38 if k == (col + shift) % 32 else 0 for col in range(32) for k in range(32))
            codes = tuple(x.bits[row * 32 + (col + shift) % 32] for row in range(32) for col in range(32))
            # Reset+continue gives 2X. Readout code 129 multiplies by four, giving 8X in FP8; BF16 keeps 2X.
            payload = bytes(EIGHT_FP8[code] for code in codes) if fmt == "fp8" else tile_bytes(Tile("bf16", tuple(TWICE_BF16[code] for code in codes)))
            initial = b"\xA5" * len(payload) + (b"\xD4" * 1024 if fmt == "fp8" else b"")
            inputs, expected = mapped({0x90000000: bytes(x.bits), 0x90000400: weight, 0x90001000: initial}, {0x90001000: payload})
            traffic = (64, len(payload) // 32)
        return inputs, EvaluationResult({}, expected), traffic
    return case


def mxu_program(body: str, *, extra_input: bool = False) -> str:
    extra = (f'%io3, %next_x = "atlas.virtual_input_fp8"(%io2) {{index = 2 : i32}} : ({STATE}) -> ({STATE}, {FP8})' if extra_input else "")
    return f'''module {{
      func.func @mxu0() -> {STATE} attributes {{atlas.input_dram_base = 2415919104 : i64, atlas.output_dram_base = 2415935488 : i64}} {{
        %io0 = "atlas.virtual_start"() : () -> {STATE}
        %io1, %x = "atlas.virtual_input_fp8"(%io0) {{index = 0 : i32}} : ({STATE}) -> ({STATE}, {FP8})
        %io2, %w = "atlas.virtual_input_fp8"(%io1) {{index = 1 : i32}} : ({STATE}) -> ({STATE}, {FP8})
        {extra}
        {body}
      }}
    }}'''


MXU_SOURCES = {
    "legacy_pack": mxu_program(f'''
    %first = "atlas.virtual_mxu_matmul"(%x, %w) {{unit = 0 : i32}} : ({FP8}, {FP8}) -> {BF16}
    %positive = "atlas.virtual_vpu_unary"(%first) {{kind = "relu"}} : ({BF16}) -> {BF16}
    %packed = "atlas.virtual_pack_fp8"(%positive) {{scale_code = 127 : i32}} : ({BF16}) -> {FP8}
    %second = "atlas.virtual_mxu_matmul"(%packed, %w) {{unit = 0 : i32}} : ({FP8}, {FP8}) -> {BF16}
    %out0 = "atlas.virtual_output_bf16"(%io2, %first) {{index = 0 : i32}} : ({STATE}, {BF16}) -> {STATE}
    %out1 = "atlas.virtual_output_bf16"(%out0, %second) {{index = 1 : i32}} : ({STATE}, {BF16}) -> {STATE}
    return %out1 : {STATE}
'''),
    "continuation": mxu_program("\n".join((
        load("io3", "s0", "weight"), reset("s0", "s1", "a0"), accumulate("s1", "s2", "a1", "a0").replace("%x,", "%next_x,"),
        readout("s2", "s3", "y", "a1"), f'%out = "atlas.virtual_output_bf16"(%s3, %y) {{index = 0 : i32}} : ({STATE}, {BF16}) -> {STATE}',
        f"return %out : {STATE}")), extra_input=True),
    "seed_bf16": seeded_chain(0, "bf16", 127), "seed_fp8": seeded_chain(0, "fp8", 127),
}


def guarded(tiles: dict[int, Tile], output_count: int) -> RuntimeInputs:
    regions = [MemoryRegion(INPUT_BASE + index * 2048, tile_bytes(tile) if tile.format == "bf16" else bytes(tile.bits) + bytes([0xC1 + index]) * 1024)
               for index, tile in tiles.items()]
    regions.extend(MemoryRegion(address, bytes([value]) * 64) for address, value in (
        (INPUT_BASE - 64, 0x51), (INPUT_BASE + (max(tiles) + 1) * 2048, 0x62), (OUTPUT_BASE - 64, 0x73), (OUTPUT_BASE + output_count * 2048, 0x84)))
    return RuntimeInputs(tiles, memory=tuple(regions))


def panel(phase: int) -> Tile:
    codes = tuple(FP8_TO_BF16)
    return Tile("fp8", tuple(codes[(row * 7 + col * 3 + col // 16 + phase) % len(codes)] for row in range(32) for col in range(32)))


def identity_weight(shift: int = 0) -> Tile:
    return Tile("fp8", tuple(0x38 if k == (column + shift) % 32 else 0 for column in range(32) for k in range(32)))


def mxu_case(name: str, unit: int):
    def case(phase: int):
        if name == "legacy_pack":
            x, shift = panel(phase), phase % 7
            first = Tile("bf16", tuple(FP8_TO_BF16[x.bits[row * 32 + (col + shift) % 32]] for row in range(32) for col in range(32)))
            final = Tile("bf16", tuple(0 if (code := x.bits[row * 32 + (col + 2 * shift) % 32]) == 0xB8 else FP8_TO_BF16[code] for row in range(32) for col in range(32)))
            inputs, outputs, traffic = guarded({0: x, 1: identity_weight(shift)}, 2), {0: first, 1: final}, (64, 128)
        elif name == "continuation":
            # MXU0 rounds each FMA: +1, +half-ULP, -half-ULP becomes 0x3f7f. MXU1 rounds once per operation, keeping 0x3f80.
            k0, columns = (phase % 10) * 3, (3 + phase, 20 + phase)
            first, next_x, weight, expected = ([0] * 1024 for _ in range(4))
            for row in range(32):
                negative = (row * 5 + phase) % 3 == 0
                first[row * 32 + k0] = 0xB8 if negative else 0x38
                next_x[row * 32 + k0 + 1] = 0x98 if negative else 0x18
                next_x[row * 32 + k0 + 2] = 0x18 if negative else 0x98
                for col in columns:
                    expected[row * 32 + col] = (0x3F7F if unit == 0 else 0x3F80) | (0x8000 if negative else 0)
            for col in columns:
                weight[col * 32 + k0:col * 32 + k0 + 3] = (0x38, 0x18, 0x18)
            inputs = guarded({0: Tile("fp8", first), 1: Tile("fp8", weight), 2: Tile("fp8", next_x)}, 1)
            outputs, traffic = {0: Tile("bf16", expected)}, (96, 64)
        else:
            # Identity products give 2X before FP8 readout/reseed and exactly 3X afterwards.
            x = panel(phase)
            inputs = guarded({0: x, 1: identity_weight(), 2: Tile("bf16", tuple(FP8_TO_BF16[code] for code in x.bits))}, 1)
            outputs, traffic = {0: Tile("bf16", tuple(TRIPLE_BF16[code] for code in x.bits))}, (128, 64)
        return inputs, EvaluationResult(outputs, inputs.memory), traffic
    return case


def mxu_structure(name: str):
    def structure(test, machine: str) -> None:
        if name == "legacy_pack":
            test.assertEqual(machine.count('"atlas.mxu_matmul"'), 2)
            test.assertIn('"atlas.vpu_pack"', machine)
        elif name == "continuation":
            test.assertIn("accumulate = false", machine)
            test.assertIn("accumulate = true", machine)
        else:
            for text in (f'kind = "acc_{name.removeprefix("seed_")}"', 'kind = "acc_fp8"', 'format = "fp8"', 'format = "bf16"'):
                test.assertIn(text, machine)
    return structure


def vpu_case(phase: int):
    # Indices break identical left/right patterns; literals are selected FP32 RNE addition then BF16 truncation.
    indices = tuple((row * 5 + col * 3 + col // 16 + phase) % len(ADD_CASES) for row in range(32) for col in range(32))
    tiles = {operand: Tile("bf16", tuple(ADD_CASES[index][operand] for index in indices)) for operand in (0, 1)}
    summed = Tile("bf16", tuple(ADD_CASES[index][2] for index in indices))
    memory = tuple(MemoryRegion(INPUT_BASE + operand * 2048, tile_bytes(tile)) for operand, tile in tiles.items()) + guards(4096, 8192)
    inputs = RuntimeInputs(tiles, memory=memory)
    # The selected ReLU keeps raw positive encodings and gives +0 for every sign-set encoding.
    return inputs, EvaluationResult({0: tiles[0], 1: tiles[0], 2: summed, 3: relu(summed)}, memory), (128, 256)


def guards(input_size: int, output_size: int) -> tuple[MemoryRegion, ...]:
    return tuple(MemoryRegion(address, bytes([value]) * 64) for address, value in (
        (INPUT_BASE - 64, 0x51), (INPUT_BASE + input_size, 0x62), (OUTPUT_BASE - 64, 0x73), (OUTPUT_BASE + output_size, 0x84)))


def dma_issue_completion(test, machine: str) -> None:
    entries = instructions(machine)
    launches = [index for index, entry in enumerate(entries) if entry["operation"] == "atlas.dma" and MARKER in entry["fields"]]
    for launch, kind in ((launches[1], "relu"), (launches[2], "mov")):
        marker = entries[launch]["fields"][MARKER]
        completion = next(index for index in range(launch + 1, len(entries))
                          if entries[index]["operation"] == "atlas.dma_wait" and entries[index]["fields"].get(MARKER) == marker)
        test.assertTrue(any(entry["operation"] == "atlas.vpu_unary" and entry["fields"]["kind"] == kind for entry in entries[launch + 1:completion]))
        test.assertEqual(entries[launch]["fields"]["channel"], entries[completion]["fields"]["channel"])


def vpu_structure(test, machine: str) -> None:
    for kind in ("mov", "add", "relu"):
        test.assertIn(f'kind = "{kind}"', machine)


DUAL = (EXAMPLES / "virtual_mxu_accumulation.mlir").read_text().replace("atlas.output_dram_base = 2415923200", "atlas.output_dram_base = 2415935488")
DUAL = DUAL.replace("%io2, %w =", "%io2, %w0 =").replace("(%io2, %w)", "(%inputs_ready, %w0)").replace("(%io3, %w)", "(%io3, %w1)")
DUAL = DUAL.replace('%io3, %weight0 =', '%inputs_ready, %w1 = "atlas.virtual_input_fp8"(%io2) {index = 2 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_fp8)\n    %io3, %weight0 =')
TWO_DMA = wrap([
    f'%io0 = "atlas.virtual_start"() : () -> {STATE}',
    "%a_addr = arith.constant -2147483648 : i32",
    "%b_addr = arith.constant -2147481600 : i32",
    "%addr = arith.constant -2147475456 : i32",
    "%size = arith.constant 2048 : i32",
    dma_load("bf16", addr="a_addr", handle="a_event"),
    dma_load("bf16", before="io1", after="io2", addr="b_addr", handle="b_event"),
    dma_await("bf16", before="io2", after="io3", handle="b_event", result="b"),
    dma_await("bf16", before="io3", after="io4", handle="a_event", result="a"),
    '%positive = "atlas.virtual_vpu_unary"(%a) {kind = "relu"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16',
    '%sum = "atlas.virtual_vpu_binary"(%positive, %b) {kind = "add"} : (!atlas.virtual_bf16, !atlas.virtual_bf16) -> !atlas.virtual_bf16',
    dma_store("bf16", before="io4", after="io5", value="sum"),
    dma_wait(before="io5", after="io6"),
], final="io6")


def two_chains_case(phase: int):
    x = panel(phase)
    inputs = guarded({0: x, 1: identity_weight(), 2: identity_weight(3)}, 2)
    first = Tile("bf16", tuple(TWICE_BF16[code] for code in x.bits))
    second = Tile("bf16", tuple(TWICE_BF16[x.bits[row * 32 + (col + 3) % 32]] for row in range(32) for col in range(32)))
    return inputs, EvaluationResult({0: first, 1: second}, inputs.memory), (96, 128)


def two_dma_case(phase: int):
    ready = patterned(READY_CODES, phase)
    payload = tile_bytes(Tile("bf16", tuple(READY_SUM[bits] for bits in ready.bits)))
    inputs, memory = mapped({0x80000000: tile_bytes(ready), 0x80000800: tile_bytes(Tile("bf16", (0x3F00,) * 1024)), 0x80002000: b"\xA5" * 2048},
                            {0x80002000: payload})
    return inputs, EvaluationResult({}, memory), (128, 64)


# -- the table ---------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Row:
    name: str
    source: str
    case: Callable  # phase -> (inputs, literal EvaluationResult, (reads, writes) or None; None entries are unchecked)
    core_phases: tuple[int, ...] = ()
    output_base: int = OUTPUT_BASE
    output_bytes: int = 0  # 0xA5 padding preloaded at output_base
    max_cycles: int = 20000
    max_steps: int = 10000
    widths: tuple[int, ...] | None = None
    scheduled: bool = False  # default and seeded compiler schedules preserve its reference and structure
    core_scheduled: bool = False  # the selected core also runs the original and first two schedules at SCHEDULED_PHASE
    mutation: tuple[str, str, Callable] | None = None  # (old, new, expected result from inputs) for a one-word change
    structure: Callable | None = None

    @property
    def phases(self) -> tuple[int, ...]:
        return tuple(sorted(set(self.core_phases) | ({0, SCHEDULED_PHASE} if self.scheduled else set())))


def rows() -> tuple[Row, ...]:
    table = [Row("shared_relu", (EXAMPLES / "virtual_bf16_shared_relu.mlir").read_text(), relu_case, (0, 257), RELU_OUTPUT_BASE, 4096,
                 mutation=('kind = "relu"', 'kind = "mov"', relu_mutated))]
    for name, source, widths, variants in cfg_cases():
        for controls, indices in variants:
            table.append(Row(f"cfg_{name}_{controls}", source, cfg_case(widths, controls, indices), (0, 113), output_bytes=4096, max_steps=256,
                             widths=widths, scheduled=name in ("runtime_i1", "tile_swap", "relu_loop"), core_scheduled=(name, controls) == ("tile_swap", (3,))))
    for name, source in dma_sources().items():
        table.append(Row(f"dma_{name}", source, dma_case(name), (0, 3), max_cycles=100000, scheduled=name in ("pending_work", "mxu_fp8_0", "mxu_bf16_1"),
                         core_scheduled=name == "pending_work", structure=dma_issue_completion if name == "pending_work" else None))
    for unit in (0, 1):
        for name, source in MXU_SOURCES.items():
            table.append(Row(f"mxu{unit}_{name}", on_unit(source, unit), mxu_case(name, unit), (0, 3), output_bytes=(2 if name == "legacy_pack" else 1) * 2048,
                             max_cycles=100000, scheduled=name in ("legacy_pack", "seed_bf16" if unit == 0 else "seed_fp8"),
                             core_scheduled=(unit, name) == (0, "legacy_pack"), structure=mxu_structure(name)))
    table.append(Row("vpu", (EXAMPLES / "virtual_bf16_vpu_program.mlir").read_text(), vpu_case, (0, 5), output_bytes=8192, scheduled=True,
                     core_scheduled=True, structure=vpu_structure))
    for same_unit in (False, True):
        source = DUAL
        if same_unit:
            source = source.replace("unit = 1 : i32", "unit = 0 : i32").replace("virtual_mxu_weight<1>", "virtual_mxu_weight<0>").replace("virtual_mxu_acc<1>", "virtual_mxu_acc<0>")
        table.append(Row(f"two_chains_{'same_unit' if same_unit else 'both_units'}", source, two_chains_case, (0, 3), output_bytes=4096, max_cycles=100000,
                         scheduled=True, core_scheduled=True))
    table.append(Row("two_dma_reverse_await", TWO_DMA, two_dma_case, (0, 3), max_cycles=100000, scheduled=True, core_scheduled=True))
    return tuple(table)


ROWS = rows()


@lru_cache(maxsize=None)
def lowered(source: str) -> str:
    return lower(source)


@lru_cache(maxsize=None)
def schedules(source: str) -> tuple[tuple[str, str], ...]:
    variants = []
    for seed in (None, *RANDOM_SEEDS):
        option = "--schedule-atlas-virtual" + (f"=random-seed={seed}" if seed is not None else "")
        result = run("atlas-opt", source, option, "--verify-atlas-virtual-stream")
        if result.returncode:
            raise AssertionError(result.stderr)
        variants.append(("default" if seed is None else f"seed{seed}", result.stdout))
    return tuple(variants)


def order(program) -> tuple:
    return tuple((operation_name(op), tuple(sorted((name, str(value)) for name, value in (*op.attributes.items(), *op.properties.items()) if name != "op_name__")))
                 for op in program.operations)


def shape(program) -> tuple:
    positions = {block: index for index, block in enumerate(program.blocks)}
    return tuple((tuple(str(arg.type) for arg in block.args), operation_name(tuple(block.ops)[-1]),
                  tuple(positions[successor] for successor in tuple(block.ops)[-1].successors)) for block in program.blocks)


def pack_input(phase: int) -> tuple[Tile, Tile]:
    indices = tuple((row * 5 + col * 3 + col // 16 + phase) % len(PACK_CASES) for row in range(32) for col in range(32))
    return Tile("bf16", tuple(PACK_CASES[index][0] for index in indices)), Tile("fp8", tuple(PACK_CASES[index][1] for index in indices))


def physical_pack_bytes(logical: Tile) -> bytes:
    # VectorFSM streams all rows of the left BF16 register, then the right; FP8Pack pairs successive 16-lane
    # rows. Derived from those source rules, not the compiler's production relayout or allocator.
    return bytes(logical.bits[((p // 16) % 32) * 32 + (p // 512) * 16 + p % 16] for p in range(1024))


def pack_operation():
    return next(op for op in parse_program(PACK_VIRTUAL).operations if operation_name(op) == "atlas.virtual_pack_fp8")


class VirtualEvaluatorCoreTest(unittest.TestCase):
    def test_rows_and_schedules_match_independent_literals(self) -> None:
        for row in ROWS:
            program = parse_program(row.source)
            if row.widths is not None:
                self.assertEqual(program.control_widths, row.widths)
            for phase in row.phases:
                with self.subTest(row=row.name, phase=phase):
                    inputs, literal, _ = row.case(phase)
                    tiles, memory = dict(inputs.tiles), inputs.memory
                    result = evaluate(program, inputs, max_steps=row.max_steps)
                    self.assertEqual(dict(result.outputs), dict(literal.outputs))
                    self.assertEqual(result.memory, literal.memory)
                    self.assertEqual((dict(inputs.tiles), inputs.memory), (tiles, memory))
                    for variant, scheduled in schedules(row.source) if row.scheduled and phase in (0, SCHEDULED_PHASE) else ():
                        with self.subTest(schedule=variant):
                            compare_results(result, evaluate(parse_program(scheduled), inputs, max_steps=row.max_steps))
        op = pack_operation()
        for phase in (0, 11):
            with self.subTest(pack_phase=phase):
                source, literal = pack_input(phase)
                packed = evaluate_tile_operation(op, (source,))
                self.assertEqual(packed, literal)
                self.assertNotEqual(physical_pack_bytes(packed)[:32], bytes(packed.bits[:32]))

    def test_rows_lower_with_expected_structure_and_schedules_reorder(self) -> None:
        for row in ROWS:
            with self.subTest(row=row.name):
                machine = lowered(row.source)
                self.assertNotRegex(machine, VIRTUAL_OPERATION)
                words = emitted(machine)
                self.assertTrue(words)
                if row.structure:
                    row.structure(self, machine)
                if row.mutation:
                    old, new, _ = row.mutation
                    self.assertEqual(sum(old in line and '"atlas.vpu_unary"' in line for line in machine.splitlines()), 1)
                    changed = emitted(machine.replace(old, new))
                    self.assertEqual(len(changed), len(words))
                    self.assertEqual(sum(left != right for left, right in zip(words, changed)), 1)
        smoke = lowered(FP8_SMOKE)
        self.assertIn('"atlas.vpu_pack"', smoke)
        self.assertTrue(emitted(smoke))
        probe = PACK_SOURCE.read_text()
        self.assertNotIn('"atlas.mxu_', probe)
        self.assertNotIn('direction = "fp8_to_bf16"', probe)
        self.assertTrue(emitted(probe))
        changed = diverse = False
        for source in dict.fromkeys(row.source for row in ROWS if row.scheduled):
            original = parse_program(source)
            orders = set()
            for variant, scheduled in schedules(source):
                with self.subTest(schedule=variant, source=original.function.sym_name.data):
                    parsed = parse_program(scheduled)
                    self.assertEqual(shape(parsed), shape(original))
                    self.assertEqual(Counter(order(parsed)), Counter(order(original)))
                    changed |= order(parsed) != order(original)
                    orders.add(order(parsed))
            diverse |= len(orders) > 1
        self.assertTrue(changed, "all compiler schedules retained source order")
        self.assertTrue(diverse, "random seeds produced no distinct operation orders")

    def test_llvm_objects_preserve_emitted_words(self) -> None:
        texts = {}
        for row in ROWS:
            if row.core_phases:
                texts[lowered(row.source)] = row.name
                if row.mutation:
                    texts[lowered(row.source).replace(*row.mutation[:2])] = row.name + "_mutated"
            for variant, scheduled in schedules(row.source)[:2] if row.scheduled else ():
                texts[lower(scheduled)] = f"{row.name}_{variant}"
        texts[PACK_SOURCE.read_text()] = "physical_pack"
        texts[lowered(FP8_SMOKE)] = "unused_fp8"
        for machine, name in texts.items():
            with self.subTest(machine=name):
                self.assertEqual(object_words(machine), emitted(machine))

    def test_selected_core_matches_rows_schedules_and_detects_mutation(self) -> None:
        with selected_core() as execute:
            for row in ROWS:
                phases = set(row.core_phases) | ({SCHEDULED_PHASE} if row.core_scheduled else set())
                for phase in sorted(phases) if row.core_phases else ():
                    inputs, literal, traffic = row.case(phase)
                    expected = evaluate(parse_program(row.source), inputs, max_steps=row.max_steps)
                    compare_results(literal, expected)
                    preload = [(region.address, region.data) for region in inputs.memory]
                    if row.output_bytes:
                        preload.append((row.output_base, b"\xA5" * row.output_bytes))
                    scheduled = row.core_scheduled and phase == SCHEDULED_PHASE
                    variants = (("original", row.source), *schedules(row.source)[:2]) if scheduled else (("original", row.source),)
                    for variant, source in variants:
                        with self.subTest(row=row.name, phase=phase, schedule=variant):
                            machine = lowered(source)
                            words = object_words(machine)
                            self.assertEqual(words, emitted(machine))
                            result = execute(words, preload, SCHEDULED_CYCLES if scheduled else row.max_cycles)
                            self.assertTrue(result.halted, "selected core did not halt")
                            if variant == "original":
                                for wanted, observed in zip(traffic, (result.reads, result.writes)):
                                    if wanted is not None:
                                        self.assertEqual(observed, wanted)
                            compare_result(expected, result.slave.captured, output_base=row.output_base)
                            if row.mutation and not scheduled:
                                old, new, mutated = row.mutation
                                corrupted = execute(object_words(machine.replace(old, new)), preload, row.max_cycles)
                                self.assertTrue(corrupted.halted, "mutated program did not halt")
                                self.assertEqual((corrupted.reads, corrupted.writes), traffic)
                                with self.assertRaisesRegex(AssertionError, "output 1 tile"):
                                    compare_result(expected, corrupted.slave.captured, output_base=row.output_base)
                                compare_result(mutated(inputs), corrupted.slave.captured, output_base=row.output_base)

    def test_selected_core_unit_scale_physical_pack_converter_and_transport(self) -> None:
        # This hand-authored machine probe is deliberately separate from virtual PACK lowering.
        with selected_core() as execute:
            words = object_words(PACK_SOURCE.read_text())
            op = pack_operation()
            for phase in (0, 11):
                with self.subTest(phase=phase):
                    source, literal = pack_input(phase)
                    packed = evaluate_tile_operation(op, (source,))
                    self.assertEqual(packed, literal)
                    preserved = (MemoryRegion(INPUT_BASE, tile_bytes(source)), MemoryRegion(OUTPUT_BASE + 1024, b"\xC3" * 1024), *guards(2048, 2048))
                    result = execute(words, [(region.address, region.data) for region in preserved] + [(OUTPUT_BASE, b"\xA5" * 1024)], 20000)
                    self.assertTrue(result.halted, "physical PACK probe did not halt")
                    self.assertEqual((result.reads, result.writes), (64, 32))
                    self.assertEqual(result.slave.captured(OUTPUT_BASE, 1024), physical_pack_bytes(packed))
                    for region in preserved:
                        self.assertEqual(result.slave.captured(region.address, len(region.data)), region.data, f"preserved memory changed at 0x{region.address:08x}")

    def test_captured_comparison_reports_tile_coordinates_and_preservation_failures(self) -> None:
        inputs, expected, _ = relu_case(0)
        buffers = {region.address: region.data for region in inputs.memory}
        buffers.update({RELU_OUTPUT_BASE + index * 2048: tile_bytes(tile) for index, tile in expected.outputs.items()})
        captured = lambda address, size: buffers[address][:size]
        compare_result(expected, captured, output_base=RELU_OUTPUT_BASE)
        original = buffers[RELU_OUTPUT_BASE + 2048]
        # Row 2, column 19 is in the second physical register half.
        changed = bytearray(original)
        changed[1024 + (2 * 16 + 3) * 2] ^= 1
        buffers[RELU_OUTPUT_BASE + 2048] = bytes(changed)
        with self.assertRaisesRegex(AssertionError, r"output 1 tile\[2,19\]"):
            compare_result(expected, captured, output_base=RELU_OUTPUT_BASE)
        buffers[RELU_OUTPUT_BASE + 2048] = original
        for region in inputs.memory:
            buffers[region.address] = b"\x00" * len(region.data)
            with self.assertRaisesRegex(AssertionError, "memory at 0x"):
                compare_result(expected, captured, output_base=RELU_OUTPUT_BASE)
            buffers[region.address] = region.data


class VirtualEvaluatorComparisonTest(unittest.TestCase):
    def diagnostic(self, expected, actual, *fragments):
        with self.assertRaises(AssertionError) as failure:
            compare_results(expected, actual)
        for fragment in fragments:
            self.assertIn(fragment, str(failure.exception))

    def test_equal_results_ignore_output_and_region_order_and_adjacent_partitions(self):
        tile = repeated((0x0000, 0x8000, 0x7FC1, 0xFFC1))
        gap_after = MemoryRegion(BASE + 32, b"gap-after")
        whole = MemoryRegion(BASE, b"abcdefgh")
        first, second = MemoryRegion(BASE, b"abc"), MemoryRegion(BASE + 3, b"defgh")
        expected = EvaluationResult({7: tile, 2: repeated((0x3F80,))}, (gap_after, whole))
        actual = EvaluationResult({2: repeated((0x3F80,)), 7: tile}, (second, gap_after, first))
        compare_results(expected, actual)
        compare_results(actual, expected)

    def test_independent_evaluator_events_can_be_issued_and_published_in_different_orders(self):
        tiles = {0: repeated((0x3F80, 0xBF80)), 1: repeated((0x7FC1, 0x8000))}
        regions = (MemoryRegion(BASE, b"?" * 2048), MemoryRegion(BASE + 0x2000, b"!" * 2048))

        def execute(reverse):
            stream = Stream()
            values = (stream.input(index=0), stream.input(index=1))
            order = (1, 0) if reverse else (0, 1)
            pending = {index: stream.store(values[index], BASE + index * 0x2000) for index in order}
            for index in reversed(order):
                stream.wait(pending[index])
            for index in order:
                stream.output(values[index], index)
            return evaluate(stream.program(), RuntimeInputs(tiles, memory=regions[::-1] if reverse else regions))

        compare_results(execute(False), execute(True))

    def test_output_indices_must_match_even_when_tiles_are_equal(self):
        tile = repeated((0,))
        expected = EvaluationResult({2: tile, 7: tile})
        for actual in (EvaluationResult({2: tile}), EvaluationResult({2: tile, 7: tile, 9: tile})):
            with self.subTest(indices=tuple(actual.outputs)):
                self.diagnostic(expected, actual, "output indices", "expected [2, 7]")

    def test_raw_signed_zero_and_nan_payload_differences_report_logical_coordinates(self):
        for row, col, wanted, observed in ((2, 19, 0x8000, 0x0000), (9, 5, 0x7FC1, 0x7FC2)):
            with self.subTest(row=row, col=col):
                original = [0] * 1024
                original[row * 32 + col] = wanted
                changed = list(original)
                changed[row * 32 + col] = observed
                expected = EvaluationResult({7: Tile("bf16", tuple(original))})
                actual = EvaluationResult({7: Tile("bf16", tuple(changed))})
                self.diagnostic(expected, actual, f"output 7 tile[{row},{col}]", f"expected 0x{wanted:04x}", f"got 0x{observed:04x}")

    def test_first_memory_byte_difference_uses_address_order_across_partitions(self):
        later, changed_later = MemoryRegion(BASE + 32, b"later"), MemoryRegion(BASE + 32, b"Later")
        whole = MemoryRegion(BASE, bytes(range(8)))
        first = MemoryRegion(BASE, bytes(range(4)))
        second = MemoryRegion(BASE + 4, b"\x04\xfe\x06\x07")
        expected = EvaluationResult({}, (later, whole))
        actual = EvaluationResult({}, (changed_later, second, first))
        self.diagnostic(expected, actual, "memory at 0x80000005", "expected 0x05", "got 0xfe")

    def test_mapping_holes_and_extra_zero_bytes_are_not_equal_to_mapped_zero_bytes(self):
        full = EvaluationResult({}, (MemoryRegion(BASE, b"\x00" * 8),))
        hole = EvaluationResult({}, (MemoryRegion(BASE + 4, b"\x00" * 4), MemoryRegion(BASE, b"\x00" * 3)))
        self.diagnostic(full, hole, "memory mapping at 0x80000003", "missing from actual")
        self.diagnostic(hole, full, "memory mapping at 0x80000003", "missing from expected")
        extra = EvaluationResult({}, (MemoryRegion(BASE, b"\x00" * 9),))
        self.diagnostic(full, extra, "memory mapping at 0x80000008", "missing from expected")
        self.diagnostic(full, EvaluationResult({}), "memory mapping at 0x80000000", "missing from actual")

    def test_comparison_detects_a_relu_to_mov_semantic_mutation(self):
        original = sparse({(2, 19): 0xBF80}, "bf16")

        def execute(kind):
            stream = Stream()
            value = stream.input()
            result = stream.pure("vpu_unary", value, "bf16", f'{{kind = "{kind}"}}')
            stream.output(result, 7)
            return evaluate(stream.program(), RuntimeInputs({0: original}))

        self.diagnostic(execute("relu"), execute("mov"), "output 7 tile[2,19]", "expected 0x0000", "got 0xbf80")

    def test_comparison_detects_mxu_reset_changed_to_seeded_accumulation(self):
        one = sparse({(0, 0): 0x38})
        seed = sparse({(0, 0): 0x3F80}, "bf16")
        runtime = mxu_inputs(one, one, seed)

        def execute(use_seed):
            stream = MxuStream()
            stream.weight()
            if use_seed:
                stream.seed()
                stream.accumulate("acc", "result_acc")
            else:
                stream.reset("result_acc")
            stream.readout("result_acc", "result")
            stream.output("result")
            return evaluate(stream.program(), runtime)

        expected, actual = execute(False), execute(True)
        self.diagnostic(expected, actual, "output 0 tile[0,0]", "expected 0x3f80", "got 0x4000")

    def test_comparison_detects_dma_loading_a_corrupted_source_into_the_destination(self):
        original = b"\x5a" * 1024
        corrupt = original[:19] + b"\x5b" + original[20:]
        original_source = MemoryRegion(BASE + 0x4000, original)
        corrupt_source = MemoryRegion(BASE + 0x6000, corrupt)
        destination = MemoryRegion(BASE, b"?" * 1024)
        runtime = RuntimeInputs(memory=(corrupt_source, destination, original_source))

        def execute(address):
            stream = Stream()
            ready = stream.await_(stream.load(address, "fp8"), "fp8")
            stream.wait(stream.store(ready, BASE, "fp8"))
            return evaluate(stream.program(), runtime)

        self.diagnostic(execute(BASE + 0x4000), execute(BASE + 0x6000), "memory at 0x80000013", "expected 0x5a", "got 0x5b")
        self.assertEqual(runtime.memory[1].data, b"?" * 1024)



if __name__ == "__main__":
    unittest.main()
