"""Replay admitted command effects, DMA ordering and native timing constraints.

Compiler-side admission/replay retains its diagnostic scope. Independent reference
and evaluator authority remain separate; no hardware execution is established.
"""

from __future__ import annotations

from collections import Counter

from merlin.compiler.semantic.target_binding import NativeCompilationError

from .policy import BF16_BINARY_MODES, BF16_ROW_REDUCTION_MODES, BF16_UNARY_MODES


def _tile_effects(program: dict) -> list[tuple]:
    """Replay scalar addresses and observable operations from structured words.

    This check reads the emitted stream, independently of virtual placement and
    of the OOT schedule verifier. It admits only the diagnostic tile families.
    """
    allowed = {
        "atlas.dma_config", "atlas.alu_imm", "atlas.upper", "atlas.dma",
        "atlas.dma_wait", "atlas.vload", "atlas.vstore", "atlas.delay",
        "atlas.xlu_transpose", "atlas.vpu_unary", "atlas.vpu_binary", "atlas.vpu_reduce",
        "atlas.mxu_push", "atlas.mxu_matmul",
        "atlas.mxu_pop", "atlas.csr", "atlas.trap", "atlas.vpu_pack",
        "atlas.scalar_load", "atlas.scalar_store", "atlas.branch",
    }
    registers = {0: 0}
    effects: list[tuple] = []
    rows = program.get("instructions")
    if not isinstance(rows, list):
        raise NativeCompilationError("compile_error", "Atlas structured stream has no instructions")
    for index, row in enumerate(rows):
        op = row.get("operation")
        fields = row.get("fields")
        if op not in allowed or not isinstance(fields, dict):
            raise NativeCompilationError("unqualified_target", "tile stream contains an unmodeled instruction")
        if op == "atlas.dma_config":
            if index != 0 or fields.get("base_reg") != 0 or fields.get("channel") != 0:
                raise NativeCompilationError("unqualified_target", "movement DMA base differs from zero")
            if row.get("word_u32") != 0x7F:
                raise NativeCompilationError("unqualified_target", "movement DMA_CONFIG word differs from selected RTL")
            effects.append(("config",))
        elif op == "atlas.upper":
            if fields.get("kind") != "lui" or fields.get("dst") == 0:
                raise NativeCompilationError("unqualified_target", "movement stream has an unmodeled upper operation")
            registers[fields["dst"]] = (fields["immediate"] << 12) & 0xffffffff
        elif op == "atlas.alu_imm":
            if fields.get("kind") != "addi" or fields.get("src") not in registers or (
                fields.get("dst") == 0 and (fields.get("src"), fields.get("immediate")) != (0, 0)
            ):
                raise NativeCompilationError("unqualified_target", "movement stream has an unmodeled scalar address")
            if fields["dst"] != 0:
                registers[fields["dst"]] = (registers[fields["src"]] + fields["immediate"]) & 0xffffffff
        elif op == "atlas.dma":
            if any(fields.get(key) not in registers for key in ("dram", "reg", "size")):
                raise NativeCompilationError("unqualified_target", "movement DMA reads an unknown scalar address")
            effects.append(("dma", fields.get("direction"),
                            registers[fields["dram"]], registers[fields["reg"]],
                            registers[fields["size"]], fields.get("channel")))
        elif op == "atlas.dma_wait":
            channel = fields.get("channel")
            if type(channel) is not int or not 0 <= channel < 8 or (
                row.get("word_u32") != ((1 << 25) | (channel << 12) | 0x7F)
            ):
                raise NativeCompilationError("unqualified_target", "movement DMA_WAIT word differs from selected RTL")
            effects.append(("wait", fields.get("channel")))
        elif op == "atlas.vload":
            if fields.get("format") != "raw" or fields.get("offset") != 0 or (
                fields.get("base") not in registers
            ):
                raise NativeCompilationError("unqualified_target", "movement load changes bits or scratch address")
            effects.append(("vload", fields["dst"], registers[fields["base"]]))
        elif op == "atlas.vstore":
            if fields.get("format") != "raw" or fields.get("offset") != 0 or (
                fields.get("base") not in registers
            ):
                raise NativeCompilationError("unqualified_target", "movement store changes bits or scratch address")
            effects.append(("vstore", fields["src"], registers[fields["base"]]))
        elif op == "atlas.xlu_transpose":
            effects.append(("transpose", fields.get("src"), fields.get("dst")))
        elif op == "atlas.vpu_unary":
            if fields.get("kind") not in {"relu", *BF16_UNARY_MODES.values()}:
                raise NativeCompilationError("unqualified_target", "BF16 tile has an unadmitted VPU mode")
            effects.append((fields["kind"], fields.get("src"), fields.get("dst")))
        elif op == "atlas.vpu_reduce":
            if fields.get("kind") not in BF16_ROW_REDUCTION_MODES.values():
                raise NativeCompilationError("unqualified_target", "BF16 tile has an unadmitted reduction mode")
            effects.append((fields["kind"], fields.get("src"), fields.get("dst")))
        elif op == "atlas.vpu_binary":
            if fields.get("kind") not in BF16_BINARY_MODES.values():
                raise NativeCompilationError("unqualified_target", "BF16 tile has an unadmitted binary VPU mode")
            effects.append((fields["kind"], fields.get("lhs"), fields.get("rhs"), fields.get("dst")))
        elif op == "atlas.vpu_pack":
            effects.append(("pack", fields.get("direction"), fields.get("src"),
                            fields.get("dst"), fields.get("scale_reg")))
        elif op == "atlas.scalar_load":
            if fields.get("kind") == "seli":
                if fields.get("base") != 0 or not 0 <= fields.get("offset", -1) <= 255:
                    raise NativeCompilationError("unqualified_target", "pack scale load is invalid")
                effects.append(("seli", fields.get("dst"), fields.get("offset")))
            elif fields.get("kind") == "lw":
                if fields.get("base") not in registers or type(fields.get("offset")) is not int:
                    raise NativeCompilationError("unqualified_target", "pack loop reads an unknown address")
                effects.append(("lw", fields.get("dst"),
                                registers[fields["base"]] + fields["offset"]))
            else:
                raise NativeCompilationError("unqualified_target", "pack stream has an unadmitted scalar load")
        elif op == "atlas.scalar_store":
            if fields.get("kind") != "sw" or fields.get("base") not in registers or (
                type(fields.get("offset")) is not int
            ):
                raise NativeCompilationError("unqualified_target", "pack loop has an unknown store address")
            effects.append(("sw", fields.get("src"),
                            registers[fields["base"]] + fields["offset"]))
        elif op == "atlas.branch":
            if fields.get("kind") != "blt" or any(
                register not in registers for register in (fields.get("lhs"), fields.get("rhs"))
            ):
                raise NativeCompilationError("unqualified_target", "pack loop has an unknown branch")
            effects.append(("blt", fields.get("lhs"), fields.get("rhs"),
                            registers[fields["lhs"]], registers[fields["rhs"]],
                            fields.get("offset_bytes")))
        elif op == "atlas.mxu_push":
            effects.append(("mxu_push", fields.get("unit"), fields.get("slot"),
                            fields.get("kind"), fields.get("src")))
        elif op == "atlas.mxu_matmul":
            effects.append(("mxu_matmul", fields.get("unit"), fields.get("acc_slot"),
                            fields.get("weight_slot"), fields.get("accumulate"),
                            fields.get("src")))
        elif op == "atlas.mxu_pop":
            effects.append(("mxu_pop", fields.get("unit"), fields.get("slot"),
                            fields.get("format"), fields.get("scale_reg"),
                            fields.get("dst")))
        elif op == "atlas.csr":
            if (fields.get("kind"), fields.get("address"), fields.get("dst")) != (
                "rrw", 0xC10, 0
            ) or registers.get(fields.get("source")) != 1:
                raise NativeCompilationError("unqualified_target", "movement completion marker changed")
            effects.append(("complete",))
        elif op == "atlas.trap":
            if index != len(rows) - 1 or fields.get("kind") != "ecall":
                raise NativeCompilationError("unqualified_target", "movement stream has a nonterminal trap")
            effects.append(("trap",))
    return effects


def _check_atomic_dma_schedule(observed: list[tuple], expected: list[tuple], *, context: str) -> None:
    """Replay a physical DMA schedule against logical transfers and scratch contents."""
    def transfer(effect: tuple) -> tuple:
        return effect[1], effect[2], effect[4]

    def non_dma(effect: tuple) -> tuple:
        return effect[:2] if effect[0] in {"vload", "vstore"} else effect

    observed_dma = Counter(map(transfer, (e for e in observed if e[0] == "dma")))
    expected_dma = Counter(map(transfer, (e for e in expected if e[0] == "dma")))
    observed_order = [non_dma(e) for e in observed if e[0] not in {"dma", "wait"}]
    expected_order = [non_dma(e) for e in expected if e[0] not in {"dma", "wait"}]
    if observed_dma != expected_dma:
        raise NativeCompilationError(
            "unqualified_target", f"{context} DMA transfers differ: "
            f"missing={list((expected_dma - observed_dma).elements())!r}, "
            f"extra={list((observed_dma - expected_dma).elements())!r}"
        )
    if observed_order != expected_order:
        index = next(i for i in range(max(len(observed_order), len(expected_order)))
                     if observed_order[i:i + 1] != expected_order[i:i + 1])
        raise NativeCompilationError(
            "unqualified_target", f"{context} non-DMA effect {index} differs: "
            f"expected={expected_order[index:index + 1]!r}, observed={observed_order[index:index + 1]!r}"
        )

    def overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
        return a[0] < b[1] and b[0] < a[1]

    def chunks(word: int, size: int) -> tuple[int, ...]:
        if type(word) is not int or type(size) is not int or size not in {1024, 2048} or word % 256:
            raise NativeCompilationError("unqualified_target", f"{context} has misaligned scratch access")
        return tuple(word * 4 + offset for offset in range(0, size, 1024))

    # The expected stream assigns a source to every VLOAD and a register to
    # every output DMA. The allocator may choose different scratch words and
    # channels, but it cannot change these data identities.
    expected_loads: list[int] = []
    expected_stores: dict[tuple[int, int], list[tuple[int, ...]]] = {}
    for index, effect in enumerate(expected):
        if effect[0] == "vload":
            source = next((old for old in reversed(expected[:index])
                           if old[0:2] == ("dma", "load")
                           and old[3] <= effect[2] < old[3] + old[4] // 4), None)
            if source is None:
                raise NativeCompilationError("unqualified_target", f"{context} has an unbound expected input")
            expected_loads.append(source[2] + (effect[2] - source[3]) * 4)
        elif effect[0:2] == ("dma", "store"):
            registers = []
            for word in chunks(effect[3], effect[4]):
                source = next((old for old in reversed(expected[:index])
                               if old[0] == "vstore" and old[2] * 4 == word), None)
                if source is None:
                    raise NativeCompilationError("unqualified_target", f"{context} has an unbound expected output")
                registers.append(source[1])
            expected_stores.setdefault((effect[2], effect[4]), []).append(tuple(registers))

    pending: dict[int, list[tuple[str, int, tuple[int, ...]]]] = {}
    scratch: dict[int, tuple[str, int]] = {}
    load_index = 0
    for effect in observed:
        kind = effect[0]
        if kind == "dma":
            direction, dram, word, size, channel = effect[1:]
            if (direction not in {"load", "store"} or type(dram) is not int
                    or type(channel) is not int
                    or not 0 <= channel < 8):
                raise NativeCompilationError("unqualified_target", f"{context} has invalid DMA")
            words = chunks(word, size)
            span = (words[0], words[-1] + 1024)
            if any(overlaps(span, active) for items in pending.values()
                   for _, _, active_words in items
                   for active in ((active_words[0], active_words[-1] + 1024),)):
                raise NativeCompilationError("unqualified_target", f"{context} reuses scratch before DMA completion")
            if direction == "store":
                planned = expected_stores.get((dram, size), [])
                if not planned or tuple(scratch.get(addr) for addr in words) != tuple(
                    ("register", reg) for reg in planned.pop(0)
                ):
                    raise NativeCompilationError("unqualified_target", f"{context} DMA reads wrong output data")
            pending.setdefault(channel, []).append((direction, dram, words))
        elif kind == "wait":
            channel = effect[1]
            if type(channel) is not int or not 0 <= channel < 8:
                raise NativeCompilationError("unqualified_target", f"{context} has invalid DMA wait")
            for direction, dram, words in pending.pop(channel, []):
                if direction == "load":
                    for offset, addr in enumerate(words):
                        scratch[addr] = ("dram", dram + offset * 1024)
        elif kind in {"vload", "vstore"}:
            word = effect[2]
            addr = chunks(word, 1024)[0]
            span = (addr, addr + 1024)
            if any(overlaps(span, active) for items in pending.values()
                   for _, _, active_words in items
                   for active in ((active_words[0], active_words[-1] + 1024),)):
                raise NativeCompilationError("unqualified_target", f"{context} accesses in-flight DMA scratch")
            if kind == "vload":
                if load_index >= len(expected_loads) or scratch.get(addr) != ("dram", expected_loads[load_index]):
                    raise NativeCompilationError("unqualified_target", f"{context} VLOAD reads wrong input data")
                load_index += 1
            else:
                scratch[addr] = ("register", effect[1])
        elif kind in {"complete", "trap"} and pending:
            raise NativeCompilationError("unqualified_target", f"{context} ends with DMA in flight")
    if pending or load_index != len(expected_loads) or any(expected_stores.values()):
        raise NativeCompilationError("unqualified_target", f"{context} leaves an incomplete transfer")


def check_copy_stream(program: dict, *, input_address: int,
                      output_address: int, pair: int) -> None:
    effects = _tile_effects(program)
    expected = [
        ("config",),
        ("dma", "load", input_address, 0, 1024, 0),
        ("wait", 0), ("vload", pair, 0),
        ("dma", "load", input_address + 1024, 256, 1024, 0),
        ("wait", 0), ("vload", pair + 1, 256),
        ("vstore", pair, 0x10000),
        ("dma", "store", output_address, 0x10000, 1024, 1),
        ("wait", 1),
        ("vstore", pair + 1, 0x10100),
        ("dma", "store", output_address + 1024, 0x10100, 1024, 1),
        ("wait", 1), ("complete",), ("trap",),
    ]
    if effects != expected:
        raise NativeCompilationError(
            "unqualified_target", "emitted copy addresses, scratch words, order, waits, or pair differ from selection"
        )


def check_copy_twice_stream(program: dict, *, input_address: int,
                            output_addresses: tuple[int, int], pair: int) -> None:
    effects = _tile_effects(program)
    expected = [
        ("config",),
        ("dma", "load", input_address, 0, 1024, 0),
        ("wait", 0), ("vload", pair, 0),
        ("dma", "load", input_address + 1024, 256, 1024, 0),
        ("wait", 0), ("vload", pair + 1, 256),
    ]
    for index, output_address in enumerate(output_addresses):
        scratch = 0x10000 + index * 0x200
        expected.extend((
            ("vstore", pair, scratch),
            ("dma", "store", output_address, scratch, 1024, 1),
            ("wait", 1),
            ("vstore", pair + 1, scratch + 256),
            ("dma", "store", output_address + 1024, scratch + 256, 1024, 1),
            ("wait", 1),
        ))
    expected.extend((("complete",), ("trap",)))
    if effects != expected:
        raise NativeCompilationError(
            "unqualified_target", "emitted two-output copy differs in sharing, addresses, order, or waits"
        )


def _check_mxu_dag_timing(program: dict) -> None:
    """Check conservative data availability against the emitted issue stream."""
    cycle = 0
    ready_at: dict[int, int] = {}
    weight_ready: dict[tuple[int, int], int] = {}
    accumulator_ready: dict[tuple[int, int], int] = {}
    pending_stores: list[int] = []
    load_path_ready = 0
    for row in program["instructions"]:
        op, fields = row["operation"], row["fields"]
        if op == "atlas.vload":
            if cycle < load_path_ready:
                raise NativeCompilationError("unqualified_target", "VLOAD path is still active")
            # npu_model/hardware/lsu.py: 32 MRF rows + 3 issue cycles;
            # _accept runs before the final-row tick, so +34 is still busy.
            load_path_ready = cycle + (32 + 3)
            ready_at[fields["dst"]] = cycle + 34
        elif op == "atlas.mxu_push":
            halves = (0, 1) if fields["kind"] == "acc_bf16" else (0,)
            if any(ready_at.get(fields["src"] + half, 1 << 60) > cycle for half in halves):
                raise NativeCompilationError("unqualified_target", "MXU push source is not available")
            key = (fields["unit"], fields["slot"])
            if fields["kind"] == "weight_fp8":
                weight_ready[key] = cycle + 1
            else:
                # A push writes row0 one cycle after issue; a POP may read it
                # the following cycle while later rows continue streaming.
                accumulator_ready[key] = cycle + 2
        elif op == "atlas.mxu_matmul":
            key = (fields["unit"], fields["weight_slot"])
            if (ready_at.get(fields["src"], 1 << 60) > cycle
                    or weight_ready.get(key, 1 << 60) > cycle):
                raise NativeCompilationError("unqualified_target", "MXU operand is not available")
            accumulator_ready[(fields["unit"], fields["acc_slot"])] = cycle + 64
        elif op == "atlas.mxu_pop":
            if accumulator_ready.get((fields["unit"], fields["slot"]), 1 << 60) > cycle:
                raise NativeCompilationError("unqualified_target", "MXU readout is premature")
            ready_at[fields["dst"]] = cycle + 32
            if fields["format"] == "bf16":
                ready_at[fields["dst"] + 1] = cycle + 32
        elif op == "atlas.vpu_unary":
            pair = fields["src"]
            if any(ready_at.get(pair + half, 1 << 60) > cycle for half in (0, 1)):
                raise NativeCompilationError("unqualified_target", "VPU unary reads an unavailable pair")
            ready_at[fields["dst"]] = cycle + 66
            ready_at[fields["dst"] + 1] = cycle + 66
        elif op == "atlas.vpu_reduce":
            pair = fields["src"]
            if any(ready_at.get(pair + half, 1 << 60) > cycle for half in (0, 1)):
                raise NativeCompilationError("unqualified_target", "VPU row reduction reads an unavailable pair")
            completion = 39 if fields["kind"] == "row_sum" else 34
            ready_at[fields["dst"]] = cycle + completion
            ready_at[fields["dst"] + 1] = cycle + completion
        elif op == "atlas.vpu_binary":
            if any(ready_at.get(pair + half, 1 << 60) > cycle
                   for pair in (fields["lhs"], fields["rhs"]) for half in (0, 1)):
                raise NativeCompilationError("unqualified_target", "VPU binary reads an unavailable pair")
            ready_at[fields["dst"]] = cycle + 66
            ready_at[fields["dst"] + 1] = cycle + 66
        elif op == "atlas.vpu_pack":
            unpack = fields["direction"] == "fp8_to_bf16"
            halves = (0,) if unpack else (0, 1)
            if any(ready_at.get(fields["src"] + half, 1 << 60) > cycle for half in halves):
                raise NativeCompilationError("unqualified_target", "pack conversion reads unavailable data")
            ready_at[fields["dst"]] = cycle + (66 if unpack else 65)
            if unpack:
                ready_at[fields["dst"] + 1] = cycle + 66
        elif op == "atlas.xlu_transpose":
            if ready_at.get(fields["src"], 1 << 60) > cycle:
                raise NativeCompilationError("unqualified_target", "XLU reads an unavailable FP8 register")
            ready_at[fields["dst"]] = cycle + 65
        elif op == "atlas.vstore":
            if ready_at.get(fields["src"], 1 << 60) > cycle:
                raise NativeCompilationError("unqualified_target", "VPU store reads an unavailable register")
            pending_stores.append(cycle)
        elif op == "atlas.dma" and fields["direction"] == "store":
            if not pending_stores or cycle - pending_stores.pop(0) < 34:
                raise NativeCompilationError("unqualified_target", "DMA reads scratch before VSTORE completes")
        elif op == "atlas.delay":
            count = fields.get("cycles")
            if type(count) is not int or not 0 <= count <= 65535:
                raise NativeCompilationError("unqualified_target", "MXU/VPU DAG delay is invalid")
            cycle += count
        cycle += 1
    if pending_stores:
        raise NativeCompilationError("unqualified_target", "VPU scratch store has no DMA completion")
