"""Fixtures for generated (resource-contract-v4) artifacts.

Lowering is the only producer of generated artifacts, and every consumer requires the
whole envelope: the marker, a timing state, the DMA/MXU/tile contracts, a CFG contract
that owns each instruction and a source memory contract. Prefer mutating lowered output (`insert_after`, `remove_line`).
Where one checker needs a hand-built stream, `artifact` wraps it in a timed envelope and
derives the rest so that only the checker under test can reject the fixture:

- DMA contract records come from tile records that carry an explicit transfer id.
- Tile records are derived from the stream when the fixture authors none; addresses are
  read from LUI/ADDI constants and unknown operands become zero placeholders.
- A prologue loads every tensor register read before it is written, stages a PACK-style
  VSTORE endpoint for VLOADs outside a completed DMA load, and writes the VMEM halves a
  DMA store reads that no earlier stream VSTORE wrote.
- CFG scaffolding: one source block per fixture block (see `structured` for labels), one
  synthetic source operation per issued command, one single-register tensor value per
  VLOAD/POP destination, and an `arith.constant` condition for conditional terminators.
  DMA launches are owned by source operations named after a one-launch virtual DMA.
- Source memory effects mirror the checker's derivation from the tile and CFG records.

The scaffolding is not an oracle for CFG correspondence (test_cfg_contract_verification.py
covers that on lowered artifacts). Checks run in the order schedule, DMA memory, DMA, MXU,
tile, CFG, source memory, timing, so a negative fixture may leave the scaffolding inconsistent as long as
its test asserts the intended diagnostic. Registers written by VPU/XLU/VLI/PACK are only
killed, never redefined; a fixture that reads such a result needs real CFG records.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re


STATE = "!atlas.state"
MARKER = 'atlas.generated_from_virtual = "resource-contract-v4"'
# Classification diagnostics (lib/AtlasGeneratedArtifact.cpp).
UNSUPPORTED = "unsupported Atlas virtual-to-machine artifact marker"
UNMARKED = "generated resource metadata requires an Atlas virtual-to-machine artifact marked"
INCOMPLETE = "resource-contract-v4 artifact requires"
CLASSIFICATION = (UNSUPPORTED, UNMARKED, INCOMPLETE)
TIMED = 'atlas.timing_state = "timed", atlas.timing_provider = "npu-model-rtl-match-v1"'
NOP = ("alu_imm", 'kind = "addi", dst = 0 : i32, src = 0 : i32, immediate = 0 : i32')
DELAY = ("delay", 'cycles = 256 : i32, atlas.delay_reason = "tensor_completion"')
TRAP = ("trap", 'kind = "ecall"')
TILE_TAG = "atlas.virtual_tile_command"
MXU_TAG = "atlas.virtual_mxu_command"
TRANSFER_TAG = "atlas.virtual_dma_transfer"
# The source memory contract counts each source operation's DMA launches by its name.
DMA_SOURCE = {"load": "atlas.virtual_dma_load_fp8", "store": "atlas.virtual_dma_store_fp8"}
LABEL, BRANCH, JUMP = "@label", "@branch", "@jump"
FIELD_RE = re.compile(r'([\w.]+) = (?:(-?\d+) : i32|"([^"]*)"|(true|false))')

# Scratch resources reserved for the prologue; fixture bodies must not rely on them.
SIZE_REG, BASE_REG, DRAM_REG = 29, 30, 31
STAGE_REG = 62  # the tensor register staged VSTOREs write from
PROLOGUE_LUI = 80  # VMEM word 0x50000 (byte 0x140000, bank 5); one 16 KiB slot per register
PROLOGUE_DRAM_LUI = 0x90800
CONDITION_REG = 1
MASK = 0xffffffff

Op = tuple[str, str]


@dataclass
class Block:
    """A source block of fixture operations with one terminator.

    `branch` names (true, false) successor blocks reached through source edges and `goto`
    an unconditional successor. A block without terminator returns. Raw redirects without
    source edges are ordinary operations with explicit offsets.
    """

    ops: list = field(default_factory=list)
    branch: tuple[int, int] | None = None
    goto: int | None = None


def fields(text: str) -> dict:
    return {name: int(integer) if integer else string if string is not None else boolean == "true"
            for name, integer, string, boolean in FIELD_RE.findall(text)}


def integer_tag(f: dict, name: str) -> int | None:
    value = f.get(name)
    return value if type(value) is int else None


def attribute(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return f'"{value}"'
    if isinstance(value, (tuple, list)):
        return "array<i32" + (": " + ", ".join(str(v) for v in value) if value else "") + ">"
    return f"{value} : i32"


def record_text(entries) -> str:
    return "[" + ", ".join("{" + ", ".join(f"{key} = {attribute(value)}" for key, value in entry.items()) + "}"
                           for entry in entries) + "]"


def tensor_reads(name: str, f: dict) -> list[int]:
    if name == "vstore":
        return [f["src"]]
    if name == "mxu_push":
        return [f["src"], f["src"] + 1] if f["kind"] == "acc_bf16" else [f["src"]]
    if name == "mxu_matmul":
        return [f["src"]]
    if name in ("vpu_unary", "vpu_reduce"):
        return [f["src"], f["src"] + 1]
    if name == "vpu_binary":
        return [f["lhs"], f["lhs"] + 1, f["rhs"], f["rhs"] + 1]
    if name == "xlu_transpose":
        return [f["src"]]
    if name == "vpu_pack":
        return [f["src"], f["src"] + 1] if f["direction"] == "bf16_to_fp8" else [f["src"]]
    return []


def tensor_writes(name: str, f: dict) -> list[int]:
    if name == "vload":
        return [f["dst"]]
    if name == "mxu_pop":
        return [f["dst"], f["dst"] + 1] if f["format"] == "bf16" else [f["dst"]]
    if name in ("vpu_unary", "vpu_binary", "vpu_reduce"):
        return [f["dst"], f["dst"] + 1]
    if name == "xlu_transpose":
        return [f["dst"]]
    if name == "vli":
        return [f["dst"], f["dst"] + 1] if f["mode"] in ("all", "row") else [f["dst"]]
    if name == "vpu_pack":
        return [f["dst"], f["dst"] + 1] if f["direction"] == "fp8_to_bf16" else [f["dst"]]
    return []


@dataclass
class _Emitted:
    name: str
    text: str
    block: int
    tags: list = field(default_factory=list)
    target: int | None = None  # block index for redirects
    tile: int | None = None
    condition: int | None = None  # the branch-condition value this ALU op defines


def tile_id(op: _Emitted) -> int | None:
    """The tile record an operation issues: its authored tag, else a derived one."""
    return integer_tag(fields(op.text), TILE_TAG) if TILE_TAG in op.text else op.tile


class _Scalars:
    """Constants materialized by LUI/ADDI/ADD, as the checkers recompute them."""

    def __init__(self) -> None:
        self.known = {0: 0}

    def step(self, name: str, f: dict) -> None:
        if name == "upper":
            if f["kind"] == "lui":
                self.known[f["dst"]] = (f["immediate"] << 12) & MASK
            else:
                self.known.pop(f["dst"], None)
        elif name == "alu_imm":
            if f["kind"] == "addi" and f["src"] in self.known:
                self.known[f["dst"]] = (self.known[f["src"]] + f["immediate"]) & MASK
            else:
                self.known.pop(f["dst"], None)
        elif name == "alu_reg":
            if f["kind"] == "add" and f["lhs"] in self.known and f["rhs"] in self.known:
                self.known[f["dst"]] = (self.known[f["lhs"]] + self.known[f["rhs"]]) & MASK
            else:
                self.known.pop(f["dst"], None)
        elif name in ("scalar_load", "csr") and f.get("dst", 0):
            self.known.pop(f["dst"], None)
        self.known[0] = 0

    def get(self, reg: int) -> int | None:
        return self.known.get(reg)


class _Builder:
    def __init__(self, blocks: list[Block], tile, mxu) -> None:
        self.blocks = blocks
        self.authored = tile is not None
        self.records = [dict(r) for r in tile] if tile is not None else []
        self.mxu = list(mxu)
        self.values: list[dict] = []
        self.sources: list[dict] = []
        self.block_records: list[dict] = []
        self.edges: list[dict] = []
        self.emitted: list[_Emitted] = []
        self.staged: list[dict] = []  # prologue VSTOREs that later DMA stores claim

    # -- prologue -----------------------------------------------------------------
    def plan_prologue(self) -> tuple[list[int], list[int], list[int]]:
        """Registers needing an origin before any read, VLOAD destinations needing a
        PACK-style VSTORE predecessor because no completed DMA load contains them, and
        VMEM halves a DMA store reads that no earlier stream VSTORE writes."""
        defined: set[int] = set()
        origins: list[int] = []
        stores: list[int] = []
        staged: list[int] = []
        stored: set[int] = set()
        scalars = _Scalars()
        completed: list[dict] = []
        pending: dict[int, dict] = {}
        for block in self.blocks:
            for name, text in block.ops:
                f = fields(text)
                for reg in tensor_reads(name, f):
                    if reg not in defined and reg not in origins:
                        origins.append(reg)
                if name == "vload" and not self.authored:
                    address = (((scalars.get(f["base"]) or 0) + f["offset"] * 32) & MASK) * 4 & MASK
                    contained = any(r["vmem_byte"] <= address and address + 1024 <= r["vmem_byte"] + r["bytes"]
                                    for r in completed)
                    if not contained and f["dst"] not in stores:
                        stores.append(f["dst"])
                if name == "vstore" and not self.authored:
                    stored.add((((scalars.get(f["base"]) or 0) + f["offset"] * 32) & MASK) * 4 & MASK)
                if name == "dma" and not self.authored:
                    pending[f["channel"]] = {"kind": f["direction"], "vmem_byte": (scalars.get(f["reg"]) or 0) * 4 & MASK,
                                             "bytes": scalars.get(f["size"]) or 0}
                    launch = pending[f["channel"]]
                    if launch["kind"] == "store" and launch["bytes"] in (1024, 2048):
                        for address in range(launch["vmem_byte"], launch["vmem_byte"] + launch["bytes"], 1024):
                            if address in stored:
                                stored.discard(address)
                            else:
                                staged.append(address)
                if name == "dma_wait" and not self.authored and f["channel"] in pending:
                    launch = pending.pop(f["channel"])
                    if launch["kind"] == "load":
                        completed.append(launch)
                defined.update(tensor_writes(name, f))
                scalars.step(name, f)
        return origins, stores, staged

    def prologue(self) -> list[tuple[str, str, int | None]]:
        """(name, fields, tile id) triples loading every needed register from VMEM and
        writing the VMEM halves that DMA stores read."""
        origins, stores, staged = self.plan_prologue()
        registers = origins + [r for r in stores if r not in origins]
        if staged and STAGE_REG not in registers:
            registers.append(STAGE_REG)
        ops: list[tuple[str, str, int | None]] = [("dma_config", "channel = 0 : i32, base_reg = 0 : i32", None)]
        if not registers:
            return ops
        ops += [("alu_imm", f'kind = "addi", dst = {SIZE_REG} : i32, src = 0 : i32, immediate = 1024 : i32', None),
                ("upper", f'kind = "lui", dst = {DRAM_REG} : i32, immediate = {PROLOGUE_DRAM_LUI} : i32', None)]
        for slot, reg in enumerate(registers):
            vmem = ((PROLOGUE_LUI + slot) << 12) * 4
            launch, wait, load = (len(self.records) + n for n in range(3))
            self.records += [dict(id=launch, kind="dma_load", reg=-1, vmem_byte=vmem, dram_byte=PROLOGUE_DRAM_LUI << 12,
                                  bytes=1024, channel=0, transfer=-1, after=()),
                             dict(id=wait, kind="dma_wait", reg=-1, vmem_byte=0, dram_byte=0, bytes=0, channel=0,
                                  transfer=-1, after=(launch,)),
                             dict(id=load, kind="vload", reg=reg, vmem_byte=vmem, dram_byte=0, bytes=1024, channel=-1,
                                  transfer=-1, after=(wait,))]
            ops += [("upper", f'kind = "lui", dst = {BASE_REG} : i32, immediate = {PROLOGUE_LUI + slot} : i32', None),
                    ("dma", f'direction = "load", channel = 0 : i32, reg = {BASE_REG} : i32, dram = {DRAM_REG} : i32, size = {SIZE_REG} : i32', launch),
                    ("dma_wait", "channel = 0 : i32", wait),
                    ("vload", f'dst = {reg} : i32, base = {BASE_REG} : i32, offset = 0 : i32, format = "raw"', load),
                    (*DELAY, None)]
            if reg in stores:
                store = len(self.records)
                self.records.append(dict(id=store, kind="vstore", reg=reg, vmem_byte=vmem, dram_byte=0, bytes=1024,
                                         channel=-1, transfer=-1, after=()))
                ops += [("vstore", f'src = {reg} : i32, base = {BASE_REG} : i32, offset = 0 : i32, format = "raw"', store),
                        (*DELAY, None)]
        for address in staged:
            word = address // 4
            low = ((word + 2048) & 4095) - 2048
            store = len(self.records)
            self.staged.append(dict(id=store, kind="vstore", reg=STAGE_REG, vmem_byte=address, dram_byte=0, bytes=1024,
                                    channel=-1, transfer=-1, after=()))
            self.records.append(self.staged[-1])
            ops += [("upper", f'kind = "lui", dst = {BASE_REG} : i32, immediate = {((word - low) >> 12) & 0xFFFFF} : i32', None),
                    ("alu_imm", f'kind = "addi", dst = {BASE_REG} : i32, src = {BASE_REG} : i32, immediate = {low} : i32', None),
                    ("vstore", f'src = {STAGE_REG} : i32, base = {BASE_REG} : i32, offset = 0 : i32, format = "raw"', store),
                    (*DELAY, None)]
        return ops

    # -- layout -------------------------------------------------------------------
    def layout(self) -> None:
        prologue = self.prologue()
        for index, block in enumerate(self.blocks):
            if index == 0:
                for name, text, tile in prologue:
                    self.emitted.append(_Emitted(name, text, 0, tile=tile))
            for name, text in block.ops:
                self.emitted.append(_Emitted(name, text, index))
            self.terminate(index, block)

    def terminate(self, index: int, block: Block) -> None:
        def jal(target: int, edge: int) -> None:
            tags = [f"atlas.virtual_cfg_edge = {edge} : i32"]
            self.emitted.append(_Emitted("jump", 'kind = "jal", dst = 0 : i32, base = 0 : i32', index, list(tags), target))
            self.emitted.append(_Emitted(*NOP, index, list(tags)))

        def edge(target: int) -> int:
            self.edges.append(dict(id=len(self.edges), **{"from": index}, to=target, incoming=()))
            return self.edges[-1]["id"]

        if block.branch is not None:
            condition = self.add_value(index, CONDITION_REG, "i1", "arith.constant", constant=1)
            self.emitted.append(_Emitted("alu_imm", f'kind = "addi", dst = {CONDITION_REG} : i32, src = 0 : i32, immediate = 1 : i32', index,
                                         [f"atlas.virtual_scalar_result = {condition} : i32"], condition=condition))
            true, false = block.branch
            true_edge, false_edge = edge(true), edge(false)
            # target -1 resolves to the true-edge jump after the false-edge jump pair.
            self.emitted.append(_Emitted("branch", f'kind = "bne", lhs = {CONDITION_REG} : i32, rhs = 0 : i32', index,
                                         [f"atlas.virtual_cfg_branch = {index} : i32"], target=-1))
            self.emitted.append(_Emitted(*NOP, index))
            jal(false, false_edge)
            jal(true, true_edge)
            self.block_records.append(dict(condition=condition, edges=(true_edge, false_edge)))
        elif block.goto is not None:
            jal(block.goto, edge(block.goto))
            self.block_records.append(dict(condition=-1, edges=(self.edges[-1]["id"],)))
        else:
            if not block.ops or block.ops[-1][0] != "trap":
                if block.ops and block.ops[-1][0] == "delay":
                    self.emitted.append(_Emitted(*NOP, index))  # a halt cannot directly follow DELAY
                self.emitted.append(_Emitted(*TRAP, index))
            self.block_records.append(dict(condition=-1, edges=()))

    # -- contracts ------------------------------------------------------------------
    def add_value(self, block: int, reg: int, type_: str, def_: str, **extra) -> int:
        self.values.append(dict(id=len(self.values), block=block, reg=reg, type=type_, def_=def_, operands=(), **extra))
        return self.values[-1]["id"]

    def add_source(self, block: int, name: str, operands=(), results=(), tile=(), mxu=()) -> dict:
        self.sources.append(dict(id=len(self.sources), block=block, name=name, operands=tuple(operands),
                                 results=tuple(results), tile_commands=tuple(tile), mxu_commands=tuple(mxu)))
        return self.sources[-1]

    def derive_tile_records(self) -> None:
        scalars = _Scalars()
        pending: dict[int, int] = {}
        completed: list[tuple[dict, int]] = []
        unclaimed: list[dict] = list(self.staged)
        last_store: dict[int, int] = {}
        for op in self.emitted:
            f = fields(op.text)
            if op.tile is not None or op.name not in ("dma", "dma_wait", "vload", "vstore"):
                if op.tile is not None and op.name == "vstore":
                    last_store[f["src"]] = op.tile  # a prologue endpoint
                scalars.step(op.name, f)
                continue
            identity = len(self.records)
            op.tile = identity
            if op.name == "dma":
                transfer = f.get(TRANSFER_TAG, -1)
                vmem, size = (scalars.get(f["reg"]) or 0) * 4 & MASK, scalars.get(f["size"]) or 0
                after = []
                if f["direction"] == "store":
                    for store in list(unclaimed):
                        address = store["vmem_byte"]
                        if vmem <= address and address + 1024 <= vmem + size and all(self.records[a]["vmem_byte"] != address for a in after):
                            store["transfer"] = transfer
                            after.append(store["id"])
                            unclaimed.remove(store)
                self.records.append(dict(id=identity, kind=f"dma_{f['direction']}", reg=-1, vmem_byte=vmem,
                                         dram_byte=scalars.get(f["dram"]) or 0, bytes=size, channel=f["channel"],
                                         transfer=transfer, after=tuple(after)))
                pending[f["channel"]] = identity
            elif op.name == "dma_wait":
                launch = pending.pop(f["channel"], None)
                self.records.append(dict(id=identity, kind="dma_wait", reg=-1, vmem_byte=0, dram_byte=0, bytes=0,
                                         channel=f["channel"], transfer=f.get(TRANSFER_TAG, -1),
                                         after=(launch,) if launch is not None else ()))
                if launch is not None and self.records[launch]["kind"] == "dma_load":
                    completed.append((self.records[launch], identity))
            else:
                address = (((scalars.get(f["base"]) or 0) + f["offset"] * 32) & MASK) * 4 & MASK
                reg = f["dst" if op.name == "vload" else "src"]
                record = dict(id=identity, kind=op.name, reg=reg, vmem_byte=address, dram_byte=0, bytes=1024,
                              channel=-1, transfer=-1, after=())
                if op.name == "vload":
                    containing = [(launch, wait) for launch, wait in completed
                                  if launch["vmem_byte"] <= address and address + 1024 <= launch["vmem_byte"] + launch["bytes"]]
                    if containing:
                        launch, wait = containing[-1]
                        record.update(transfer=launch["transfer"], after=(wait,))
                    elif reg in last_store:
                        record["after"] = (last_store[reg],)
                else:
                    unclaimed.append(record)
                    last_store[reg] = identity
                self.records.append(record)
            scalars.step(op.name, f)

    def dma_records(self) -> list[dict]:
        launches = {tile_id(op): fields(op.text) for op in self.emitted if op.name == "dma" and tile_id(op) is not None}
        result: dict[int, dict] = {}
        for r in self.records:
            if r["kind"] not in ("dma_load", "dma_store") or r["transfer"] < 0 or r["transfer"] in result or r["id"] not in launches:
                continue
            f = launches[r["id"]]
            result[r["transfer"]] = dict(id=r["transfer"], channel=r["channel"], direction=r["kind"][4:],
                                         staging_word=r["vmem_byte"] // 4, dram_byte=r["dram_byte"], size_bytes=r["bytes"],
                                         staging_reg=f["reg"], dram_reg=f["dram"], size_reg=f["size"])
        return [result[key] for key in sorted(result)]

    def source_memory_records(self) -> list[dict]:
        """deriveEffects (lib/AtlasSourceMemoryEffectContract.cpp): an unowned launch is the mailbox."""
        owners = {command: source for source in self.sources for command in source["tile_commands"]}
        effects: list[dict] = []
        for r in self.records:
            if r["kind"] not in ("dma_load", "dma_store"):
                continue
            owner = owners.get(r["id"])
            waits = [w["id"] for w in self.records if w["kind"] == "dma_wait" and tuple(w["after"]) == (r["id"],)]
            effect = dict(id=len(effects), source=owner["id"] if owner else -1, block=owner["block"] if owner else -1,
                          launch=r["id"], completion=waits[0] if waits else -1, dram_byte=r["dram_byte"] & MASK,
                          bytes=r["bytes"], write=r["kind"] == "dma_store")
            effect["predecessors"] = tuple(
                e["id"] for e in effects
                if (e["block"] == effect["block"] or e["source"] == -1) and (e["write"] or effect["write"])
                and e["dram_byte"] < effect["dram_byte"] + effect["bytes"] and effect["dram_byte"] < e["dram_byte"] + e["bytes"])
            effects.append(effect)
        return effects

    def scaffold(self) -> None:
        writes: dict[int, int] = {}
        for op in self.emitted:
            for reg in tensor_writes(op.name, fields(op.text)):
                writes[reg] = writes.get(reg, 0) + 1
        mailbox = sorted((op for op in self.emitted if op.name == "scalar_load" and op.block == 0
                          and fields(op.text).get("kind") == "lw" and tile_id(op) is not None), key=tile_id)
        arguments = {id(op): self.add_value(0, fields(op.text)["dst"], "i32", "argument") for op in mailbox}
        origin: dict[int, int] = {}
        owned: list[tuple[_Emitted, dict]] = []
        for op in self.emitted:
            f = fields(op.text)
            op.tags.insert(0, f"atlas.virtual_cfg_block = {op.block} : i32")
            if op.name == "delay":
                continue
            if id(op) in arguments:
                value = arguments[id(op)]
                op.tags += [f"atlas.virtual_scalar_argument = {value} : i32", f"atlas.virtual_scalar_result = {value} : i32"]
                continue
            # Source operations follow block order, as lowering numbers them.
            if op.condition is not None:
                owned.append((op, self.add_source(op.block, "arith.constant", results=[op.condition])))
                continue
            # Malformed tags stay in the stream for their checker but bind no source record.
            tile, mxu = tile_id(op), integer_tag(f, MXU_TAG)
            if tile is not None and mailbox and tile < 2 + len(mailbox):
                continue
            operands = []
            for reg in tensor_reads(op.name, f):
                if reg in origin and origin[reg] not in operands:
                    operands.append(origin[reg])
            results = []
            written = tensor_writes(op.name, f)
            if op.name in ("vload", "mxu_pop"):
                type_ = "bf16" if len(written) == 2 else "fp8"
                value = self.add_value(op.block, written[0], type_, f"fixture.{op.name}")
                results.append(value)
                op.tags.append(f"atlas.virtual_tensor_result = {value} : i32")
                for reg in written:
                    origin[reg] = value
            else:
                for reg in written:
                    origin.pop(reg, None)
            if tile is not None or mxu is not None or results:
                name = DMA_SOURCE[f["direction"]] if op.name == "dma" else f"fixture.{op.name}"
                owned.append((op, self.add_source(op.block, name, operands, results,
                                                  [tile] if tile is not None else [], [mxu] if mxu is not None else [])))
        # Lowering numbers tile commands in source order, which the source memory contract requires;
        # authored records and the appended prologue may be issued out of id order, so number by tile id.
        for block in range(len(self.blocks)):
            slots = [n for n, s in enumerate(self.sources) if s["block"] == block and s["tile_commands"]]
            for n, source in zip(slots, sorted((self.sources[n] for n in slots), key=lambda s: s["tile_commands"])):
                self.sources[n] = source
        for n, source in enumerate(self.sources):
            source["id"] = n
        for op, source in owned:
            op.tags += [f"atlas.virtual_cfg_source = {source['id']} : i32", f"atlas.virtual_cfg_operation = {source['id']} : i32"]
        for index, record in enumerate(self.block_records):
            stable = [v["id"] for v in self.values if v["type"] in ("bf16", "fp8") and v["block"] < index
                      and all(writes.get(v["reg"] + half, 0) == 1 for half in range(2 if v["type"] == "bf16" else 1))]
            record.update(id=index, args=tuple(arguments.values()) if index == 0 else (), live_in=tuple(stable),
                          operations=tuple(s["id"] for s in self.sources if s["block"] == index))

    # -- text -----------------------------------------------------------------------
    def resolve_targets(self) -> None:
        starts = {}
        for position, op in enumerate(self.emitted):
            starts.setdefault(op.block, position)
        for position, op in enumerate(self.emitted):
            if op.target is None:
                continue
            if op.name == "branch" and op.target < 0:
                target = position + 4  # past the delay slot and the false edge's jump pair
            else:
                target = starts[op.target]
            key = "offset_bytes" if op.name == "branch" else "offset"
            op.text += f", {key} = {2 * (target - position)} : i32"

    def text(self) -> str:
        self.layout()
        if not self.authored:
            self.derive_tile_records()
        for op in self.emitted:
            if op.tile is not None and TILE_TAG not in op.text:
                op.text += f", {TILE_TAG} = {op.tile} : i32"
        self.scaffold()
        self.resolve_targets()
        lines = [f'%s0 = "atlas.start"() : () -> {STATE}']
        for index, op in enumerate(self.emitted):
            attributes = ", ".join([op.text, *op.tags])
            lines.append(f'%s{index + 1} = "atlas.{op.name}"(%s{index}) {{{attributes}}} : ({STATE}) -> {STATE}')
        values = [{**{k: v for k, v in value.items() if k != "def_"}, "def": value["def_"]} for value in self.values]
        blocks = [dict(id=b["id"], condition=b["condition"], args=b["args"], live_in=b["live_in"],
                       operations=b["operations"], edges=b["edges"]) for b in self.block_records]
        cfg = (f"{{values = {record_text(values)}, blocks = {record_text(blocks)}, "
               f"edges = {record_text(self.edges)}, operations = {record_text(self.sources)}}}")
        attributes = (f"{MARKER}, {TIMED}, atlas.virtual_dma_contract = {record_text(self.dma_records())}, "
                      f"atlas.virtual_mxu_contract = {record_text(self.mxu)}, atlas.virtual_tile_contract = {record_text(self.records)}, "
                      f"atlas.virtual_cfg_contract = {cfg}, "
                      f"atlas.virtual_source_memory_contract = {{effects = {record_text(self.source_memory_records())}}}")
        return f"module attributes {{{attributes}}} {{\n" + "\n".join(lines) + "\n}\n"


def _result(line: str) -> str:
    return re.match(r'\s*(%[\w.]+) = ', line)[1]


def _rethread(lines: list[str], index: int, old: str, new: str) -> None:
    if index < len(lines) and f"({old})" in lines[index]:
        lines[index] = lines[index].replace(f"({old})", f"({new})", 1)


def insert_after(machine: str, index: int, operations) -> str:
    """Insert (name, fields) operations after line `index` of a printed artifact, owned by
    that line's source block."""
    lines = machine.splitlines()
    previous = _result(lines[index])
    block = re.search(r'atlas\.virtual_cfg_block = (\d+) : i32', lines[index])[1]
    inserted = []
    for offset, (name, text) in enumerate(operations):
        result = f"%inserted{index}_{offset}"
        inserted.append(f'  {result} = "atlas.{name}"({previous}) {{{text}, atlas.virtual_cfg_block = {block} : i32}} : ({STATE}) -> {STATE}')
        previous = result
    _rethread(lines, index + 1, _result(lines[index]), previous)
    lines[index + 1:index + 1] = inserted
    return "\n".join(lines) + "\n"


def remove_line(machine: str, index: int) -> str:
    """Remove the operation on line `index` of a printed artifact, rethreading its state."""
    lines = machine.splitlines()
    operand = re.search(r'"atlas\.\w+"\((%[\w.]+)\)', lines[index])[1]
    _rethread(lines, index + 1, _result(lines[index]), operand)
    del lines[index]
    return "\n".join(lines) + "\n"


def label(name: str) -> Op:
    return LABEL, name


def branch_to(name: str) -> Op:
    return BRANCH, name


def jump_to(name: str) -> Op:
    return JUMP, name


def structured(stream) -> list[Block]:
    """Source blocks for a flat stream with label/branch_to/jump_to pseudo-operations.

    Labels start blocks; an open block falls through to the next. A conditional redirect
    takes its label and falls through to the next block. The NOP delay slot written after
    a pseudo-redirect is dropped because blocks emit their own. A leading label gets an
    empty entry block so that the entry has no predecessors.
    """
    items = list(stream)
    blocks, labels, redirects = [Block()], {}, []
    closed = False
    index = 0
    while index < len(items):
        name, fields = items[index]
        if name == LABEL:
            if closed or blocks[-1].ops or index == 0:
                if not closed:
                    blocks[-1].goto = len(blocks)
                blocks.append(Block())
                closed = False
            labels[fields] = len(blocks) - 1
        else:
            if closed:
                blocks.append(Block())
                closed = False
            if name in (BRANCH, JUMP):
                redirects.append((len(blocks) - 1, name, fields))
                closed = True
                if index + 1 < len(items) and items[index + 1] == NOP:
                    index += 1
            else:
                blocks[-1].ops.append((name, fields))
                closed = name == "trap"
        index += 1
    for block, name, target in redirects:
        if name == JUMP:
            blocks[block].goto = labels[target]
        else:
            if block + 1 == len(blocks):
                blocks.append(Block())
            blocks[block].branch = (labels[target], block + 1)
    return blocks


def artifact(stream, *, tile=None, mxu=()) -> str:
    """A complete generated artifact around `stream`: a flat list of (name, fields)
    operations, a flat list with label pseudo-operations (see `structured`), or a list
    of Blocks laid out in order. A flat list without labels forms one returning block.
    `tile` supplies hand-authored tile records bound to the stream through its own
    `atlas.virtual_tile_command` tags; without it, records are derived from the stream.
    `mxu` supplies hand-authored MXU records for the stream's `atlas.virtual_mxu_command` tags.
    """
    if stream and isinstance(stream[0], Block):
        blocks = list(stream)
    elif any(name in (LABEL, BRANCH, JUMP) for name, _ in stream):
        blocks = structured(stream)
    else:
        blocks = [Block(list(stream))]
    return _Builder(blocks, tile, mxu).text()
