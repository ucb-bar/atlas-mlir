"""Hand-built generated artifacts for streams no lowering produces; prefer mutating lowered output.

`artifact` wraps a stream in a timed envelope and derives every contract so that only the checker under test
can reject it. A prologue loads each tensor register read before it is written, DMA-loads each VMEM half a VLOAD
reads outside a completed DMA load, and writes each half a DMA store reads that no stream VSTORE wrote. The CFG
scaffolding is not a CFG oracle: checks run in the order schedule, DMA memory, DMA, MXU, tile, CFG, source memory,
buffer, timing, so a negative fixture may leave later scaffolding inconsistent.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from verification_support import DELAY, MARKER, MASK, NOP, STATE, TIMED, PROVIDER, TRANSFER, TRAP, constant, fields

TILE_TAG = "atlas.virtual_tile_command"
MXU_TAG = "atlas.virtual_mxu_command"
# The source memory contract counts each source operation's DMA launches by its name.
DMA_SOURCE = {"load": "atlas.virtual_dma_load_fp8", "store": "atlas.virtual_dma_store_fp8"}
LABEL, BRANCH, JUMP = "@label", "@branch", "@jump"
# Scratch resources reserved for the prologue; fixture bodies must not rely on them.
SIZE_REG, BASE_REG, DRAM_REG, STAGE_REG = 29, 30, 31, 62
VMEM_BYTES = 1536 * 1024
PROLOGUE_LUI = 80  # VMEM word 0x50000 (byte 0x140000, bank 5); one 16 KiB slot per register
PROLOGUE_DRAM_LUI = 0x90800
CONDITION_REG = 1


@dataclass
class Block:
    """Operations with a terminator: `branch` (true, false) or `goto` successors through source edges, else return."""

    ops: list = field(default_factory=list)
    branch: tuple[int, int] | None = None
    goto: int | None = None


@dataclass
class _Emitted:
    name: str
    text: str
    block: int
    tags: list = field(default_factory=list)
    target: int | None = None  # block index for redirects
    tile: int | None = None
    condition: int | None = None  # the branch-condition value this ALU op defines


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
    return [f["src"]] if name in ("vstore", "mxu_push", "mxu_matmul") else []


def tensor_writes(name: str, f: dict) -> list[int]:
    if name == "mxu_pop":
        return [f["dst"], f["dst"] + 1] if f["format"] == "bf16" else [f["dst"]]
    return [f["dst"]] if name == "vload" else []


class _Scalars:
    """Constants materialized by LUI/ADDI, as the checkers recompute them."""

    def __init__(self) -> None:
        self.known = {0: 0}

    def step(self, name: str, f: dict) -> None:
        dst = f.get("dst")
        if name == "upper" and f["kind"] == "lui":
            self.known[dst] = (f["immediate"] << 12) & MASK
        elif name == "alu_imm" and f["kind"] == "addi" and f["src"] in self.known:
            self.known[dst] = (self.known[f["src"]] + f["immediate"]) & MASK
        elif name in ("upper", "alu_imm", "alu_reg", "scalar_load", "csr") and dst:
            self.known.pop(dst, None)
        self.known[0] = 0

    def get(self, reg: int) -> int:
        return self.known.get(reg) or 0

    def vector_byte(self, f: dict) -> int:
        return ((self.get(f["base"]) + f["offset"] * 32) & MASK) * 4 & MASK


class _Builder:
    def __init__(self, blocks: list[Block], mxu) -> None:
        self.blocks = blocks
        self.records: list[dict] = []
        self.mxu = list(mxu)
        self.values: list[dict] = []
        self.sources: list[dict] = []
        self.block_records: list[dict] = []
        self.edges: list[dict] = []
        self.emitted: list[_Emitted] = []

    def plan_prologue(self) -> tuple[list[int], list[int], list[int]]:
        """Registers read before written, VMEM halves VLOADs read outside completed DMA loads, and halves DMA stores
        read that no earlier VSTORE writes."""
        defined: set[int] = set()
        origins, loads, staged = [], [], []
        stored: set[int] = set()
        scalars = _Scalars()
        completed: list[dict] = []
        pending: dict[int, dict] = {}
        for block in self.blocks:
            for name, text in block.ops:
                f = fields(text)
                origins += [reg for reg in tensor_reads(name, f) if reg not in defined and reg not in origins]
                if name == "vload":
                    address = scalars.vector_byte(f)
                    contained = any(r["vmem_byte"] <= address and address + 1024 <= r["vmem_byte"] + r["bytes"] for r in completed)
                    # Halves no DMA load can stage are rejected before the buffer check.
                    if not contained and address % 1024 == 0 and address + 1024 <= VMEM_BYTES and address not in loads:
                        loads.append(address)
                elif name == "vstore":
                    stored.add(scalars.vector_byte(f))
                elif name == "dma":
                    launch = pending[f["channel"]] = {"kind": f["direction"], "vmem_byte": scalars.get(f["reg"]) * 4 & MASK,
                                                      "bytes": scalars.get(f["size"])}
                    if launch["kind"] == "store" and launch["bytes"] in (1024, 2048):
                        for address in range(launch["vmem_byte"], launch["vmem_byte"] + launch["bytes"], 1024):
                            if address in stored:
                                stored.discard(address)
                            elif address not in staged:
                                staged.append(address)
                elif name == "dma_wait" and f["channel"] in pending:
                    launch = pending.pop(f["channel"])
                    if launch["kind"] == "load":
                        completed.append(launch)
                defined.update(tensor_writes(name, f))
                scalars.step(name, f)
        return origins, loads, staged

    def prologue(self) -> list[tuple[str, str]]:
        origins, loads, staged = self.plan_prologue()
        registers = origins + ([STAGE_REG] if staged and STAGE_REG not in origins else [])
        ops = [("dma_config", "channel = 0 : i32, base_reg = 0 : i32")]
        if not registers and not loads:
            return ops
        ops += [("alu_imm", f'kind = "addi", dst = {SIZE_REG} : i32, src = 0 : i32, immediate = 1024 : i32'),
                ("upper", f'kind = "lui", dst = {DRAM_REG} : i32, immediate = {PROLOGUE_DRAM_LUI} : i32')]
        load = [("dma", f'direction = "load", channel = 0 : i32, reg = {BASE_REG} : i32, dram = {DRAM_REG} : i32, size = {SIZE_REG} : i32'),
                ("dma_wait", "channel = 0 : i32")]
        for slot, reg in enumerate(registers):
            ops += [("upper", f'kind = "lui", dst = {BASE_REG} : i32, immediate = {PROLOGUE_LUI + slot} : i32'), *load,
                    ("vload", f'dst = {reg} : i32, base = {BASE_REG} : i32, offset = 0 : i32, format = "raw"'), DELAY]
        for address in staged:
            ops += [*constant(BASE_REG, address // 4), ("vstore", f'src = {STAGE_REG} : i32, base = {BASE_REG} : i32, offset = 0 : i32, format = "raw"'), DELAY]
        for address in loads:
            ops += [*constant(BASE_REG, address // 4), *load]
        return ops

    def layout(self) -> None:
        self.emitted = [_Emitted(name, text, 0) for name, text in self.prologue()]
        for index, block in enumerate(self.blocks):
            self.emitted += [_Emitted(name, text, index) for name, text in block.ops]
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
        completed: list[tuple[dict, int]] = []  # DMA loads with their WAIT
        unclaimed: list[dict] = []  # VSTOREs no DMA store has captured
        claimed: list[dict] = []
        for op in self.emitted:
            f = fields(op.text)
            if op.name in ("dma", "dma_wait", "vload", "vstore"):
                op.tile = identity = len(self.records)
                if op.name == "dma":
                    transfer = f.get(TRANSFER, -1)
                    vmem, size = scalars.get(f["reg"]) * 4 & MASK, scalars.get(f["size"])
                    after = []
                    if f["direction"] == "store":
                        # Stores of one transfer identity share a staged half.
                        for store in [*unclaimed, *(c for c in claimed if c["transfer"] == transfer)]:
                            address = store["vmem_byte"]
                            if vmem <= address and address + 1024 <= vmem + size and all(self.records[a]["vmem_byte"] != address for a in after):
                                store["transfer"] = transfer
                                after.append(store["id"])
                                if store in unclaimed:
                                    unclaimed.remove(store)
                                    claimed.append(store)
                    self.records.append(dict(id=identity, kind=f"dma_{f['direction']}", reg=-1, vmem_byte=vmem,
                                             dram_byte=scalars.get(f["dram"]), bytes=size, channel=f["channel"],
                                             transfer=transfer, after=tuple(after)))
                    pending[f["channel"]] = identity
                elif op.name == "dma_wait":
                    launch = pending.pop(f["channel"], None)
                    self.records.append(dict(id=identity, kind="dma_wait", reg=-1, vmem_byte=0, dram_byte=0, bytes=0,
                                             channel=f["channel"], transfer=f.get(TRANSFER, -1),
                                             after=(launch,) if launch is not None else ()))
                    if launch is not None and self.records[launch]["kind"] == "dma_load":
                        completed.append((self.records[launch], identity))
                else:
                    address = scalars.vector_byte(f)
                    record = dict(id=identity, kind=op.name, reg=f["dst" if op.name == "vload" else "src"], vmem_byte=address,
                                  dram_byte=0, bytes=1024, channel=-1, transfer=-1, after=())
                    if op.name == "vstore":
                        unclaimed.append(record)
                    containing = [(launch, wait) for launch, wait in completed
                                  if launch["vmem_byte"] <= address and address + 1024 <= launch["vmem_byte"] + launch["bytes"]]
                    if op.name == "vload" and containing:
                        record.update(transfer=containing[-1][0]["transfer"], after=(containing[-1][1],))
                    self.records.append(record)
            scalars.step(op.name, f)

    def dma_records(self) -> list[dict]:
        launches = {op.tile: fields(op.text) for op in self.emitted if op.name == "dma" and op.tile is not None}
        result: dict[int, dict] = {}
        for r in self.records:
            if r["kind"] in ("dma_load", "dma_store") and r["transfer"] >= 0 and r["transfer"] not in result and r["id"] in launches:
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

    def buffer_reads(self) -> list[dict]:
        """derive (lib/AtlasBufferContractVerification.cpp): the writer words each reader observes."""
        reads: list[dict] = []
        for r in self.records:
            spans = []
            if r["kind"] == "dma_store":
                spans = [(store, self.records[store]) for store in r["after"]]
            elif r["kind"] == "vload" and len(r["after"]) == 1:
                writer = r["after"][0]
                if self.records[writer]["kind"] == "dma_wait" and len(self.records[writer]["after"]) == 1:
                    writer = self.records[writer]["after"][0]
                spans = [(writer, r)]
            reads += [dict(command=r["id"], writer=writer, vmem_byte=span["vmem_byte"], bytes=span["bytes"],
                           word=(span["vmem_byte"] - self.records[writer]["vmem_byte"]) // 4, layout="copy") for writer, span in spans]
        return reads

    def scaffold(self) -> None:
        writes: dict[int, int] = {}
        for op in self.emitted:
            for reg in tensor_writes(op.name, fields(op.text)):
                writes[reg] = writes.get(reg, 0) + 1
        origin: dict[int, int] = {}
        for op in self.emitted:
            f = fields(op.text)
            op.tags.insert(0, f"atlas.virtual_cfg_block = {op.block} : i32")
            if op.name == "delay":
                continue
            if op.condition is not None:
                source = self.add_source(op.block, "arith.constant", results=[op.condition])
            else:
                # Malformed tags stay in the stream for their checker but bind no source record.
                mxu = f.get(MXU_TAG) if type(f.get(MXU_TAG)) is int else None
                operands = list(dict.fromkeys(origin[reg] for reg in tensor_reads(op.name, f) if reg in origin))
                written = tensor_writes(op.name, f)
                results = []
                if written:
                    value = self.add_value(op.block, written[0], "bf16" if len(written) == 2 else "fp8", f"fixture.{op.name}")
                    results.append(value)
                    op.tags.append(f"atlas.virtual_tensor_result = {value} : i32")
                    origin.update((reg, value) for reg in written)
                if op.tile is None and mxu is None and not results:
                    continue
                name = DMA_SOURCE[f["direction"]] if op.name == "dma" else f"fixture.{op.name}"
                source = self.add_source(op.block, name, operands, results, [] if op.tile is None else [op.tile], [] if mxu is None else [mxu])
            op.tags += [f"atlas.virtual_cfg_source = {source['id']} : i32", f"atlas.virtual_cfg_operation = {source['id']} : i32"]
        for index, record in enumerate(self.block_records):
            stable = [v["id"] for v in self.values if v["type"] in ("bf16", "fp8") and v["block"] < index
                      and all(writes.get(v["reg"] + half, 0) == 1 for half in range(2 if v["type"] == "bf16" else 1))]
            record.update(id=index, args=(), live_in=tuple(stable), operations=tuple(s["id"] for s in self.sources if s["block"] == index))

    def resolve_targets(self) -> None:
        starts = {}
        for position, op in enumerate(self.emitted):
            starts.setdefault(op.block, position)
        for position, op in enumerate(self.emitted):
            if op.target is not None:
                # The conditional branch skips its delay slot and the false edge's jump pair.
                target = position + 4 if op.name == "branch" and op.target < 0 else starts[op.target]
                op.text += f", {'offset_bytes' if op.name == 'branch' else 'offset'} = {2 * (target - position)} : i32"

    def text(self) -> str:
        self.layout()
        self.derive_tile_records()
        for op in self.emitted:
            if op.tile is not None:
                op.text += f", {TILE_TAG} = {op.tile} : i32"
        self.scaffold()
        self.resolve_targets()
        lines = [f'%s0 = "atlas.start"() : () -> {STATE}']
        for index, op in enumerate(self.emitted):
            lines.append(f'%s{index + 1} = "atlas.{op.name}"(%s{index}) {{{", ".join([op.text, *op.tags])}}} : ({STATE}) -> {STATE}')
        values = [{**{k: v for k, v in value.items() if k != "def_"}, "def": value["def_"]} for value in self.values]
        blocks = [{key: b[key] for key in ("id", "condition", "args", "live_in", "operations", "edges")} for b in self.block_records]
        cfg = (f"{{values = {record_text(values)}, blocks = {record_text(blocks)}, "
               f"edges = {record_text(self.edges)}, operations = {record_text(self.sources)}}}")
        attributes = (f"{MARKER}, {TIMED}, {PROVIDER}, atlas.virtual_dma_contract = {record_text(self.dma_records())}, "
                      f"atlas.virtual_mxu_contract = {record_text(self.mxu)}, atlas.virtual_tile_contract = {record_text(self.records)}, "
                      f"atlas.virtual_cfg_contract = {cfg}, "
                      f"atlas.virtual_source_memory_contract = {{effects = {record_text(self.source_memory_records())}}}, "
                      f"atlas.virtual_buffer_contract = {{packs = [], reads = {record_text(self.buffer_reads())}}}")
        return f"module attributes {{{attributes}}} {{\n" + "\n".join(lines) + "\n}\n"


def label(name: str) -> tuple[str, str]:
    return LABEL, name


def branch_to(name: str) -> tuple[str, str]:
    return BRANCH, name


def jump_to(name: str) -> tuple[str, str]:
    return JUMP, name


def structured(stream) -> list[Block]:
    """Source blocks for a stream with label pseudo-operations; a conditional redirect falls through to the next block.
    A pseudo-redirect's NOP delay slot is dropped, and a leading label gets an empty entry block without predecessors."""
    items = list(stream)
    blocks, labels, redirects = [Block()], {}, []
    closed = False
    index = 0
    while index < len(items):
        name, text = items[index]
        if name == LABEL:
            if closed or blocks[-1].ops or index == 0:
                if not closed:
                    blocks[-1].goto = len(blocks)
                blocks.append(Block())
                closed = False
            labels[text] = len(blocks) - 1
        else:
            if closed:
                blocks.append(Block())
                closed = False
            if name in (BRANCH, JUMP):
                redirects.append((len(blocks) - 1, name, text))
                closed = True
                if index + 1 < len(items) and items[index + 1] == NOP:
                    index += 1
            else:
                blocks[-1].ops.append((name, text))
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


def artifact(stream, *, mxu=()) -> str:
    """A complete generated artifact around a flat (name, fields) stream, optionally with label pseudo-operations.
    `mxu` supplies hand-authored MXU records for the stream's `atlas.virtual_mxu_command` tags."""
    blocks = structured(stream) if any(name in (LABEL, BRANCH, JUMP) for name, _ in stream) else [Block(list(stream))]
    return _Builder(blocks, mxu).text()
