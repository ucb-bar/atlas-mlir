#!/usr/bin/env python3
"""Simulate a lowered Atlas program in npu_model and report unit utilization.

Takes a lower_atlas_program.py artifact directory (5-program.json and
4-timed.mlir). Prints total cycles, each execution unit's busy cycles and
utilization, and whether the program halted on ECALL without runtime errors.
With --profile, also attributes every cycle to the virtual operation whose
instruction holds the in-order S1 stage (issuing, or stalled on a DELAY or
DMA.WAIT), with how many of those cycles the DMA engine was busy.

Each emitted word is decoded by trying every npu_model instruction class with
its opcode and keeping the one whose encoding reproduces the word exactly. Three
npu_model behaviors differ from the Atlas RTL that atlas-emit and the lowering
follow, and are adapted here:
  - VR operands are six bits each at [18:13] and [24:19]; npu_model's encoder
    masks them to seven and five bits (docs/source-discrepancies.md).
  - dma.config has funct7=0; npu_model gives it funct7=1, the dma.wait word
    (docs/source-discrepancies.md).
  - DMA VMEM addresses count 4-byte words (docs/dialect-reference.md,
    --insert-atlas-delays); npu_model main counts bytes. The DMA commands and
    the VMEM bank check are patched to words, and VLOAD/VSTORE bank ranges use
    the byte address npu_model's own exec computes. Transfer time depends only
    on the byte count, so timing is unchanged; without this, two staging
    windows 512 words apart report a false bank conflict.
DRAM holds zeros; npu_model timing does not depend on values.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import re
import sys
from collections import OrderedDict
from pathlib import Path

import npu_model.hardware  # noqa: F401  (before the ISA: circular import)
import npu_model.configs.hardware as hw_configs
import npu_model.configs.isa_definition as isa_definition
import npu_model.hardware.bank_conflict as bank_conflict
import npu_model.hardware.dma as dma_unit
from npu_model.isa import (IsaSpec, RType, IType, SType, SBType, UType, UJType,
                           VLSType, VRType, VIType, CSRType)
from npu_model.isa_patterns import ScalarComputeReg, TensorBaseOffset
from npu_model.isa_types import Bundled, ScalarReg
from npu_model.logging import LoggerConfig
from npu_model.simulation import Simulation
from npu_model.software.program import InstantiableProgram


def word_addressed_dma() -> None:
    original = bank_conflict.vmem_accesses

    def vmem_accesses(insn, state):
        if isinstance(insn, TensorBaseOffset):
            return bank_conflict._vmem_range_to_banks(
                isa_definition._vmem_line_address(state, insn.rs1, insn.imm),
                state.cfg.mrf_depth * state.cfg.mrf_width)
        if isinstance(insn, ScalarComputeReg) and insn.mnemonic.startswith(("dma.load", "dma.store")):
            vmem = insn.rd if insn.mnemonic.startswith("dma.load") else insn.rs1
            return bank_conflict._vmem_range_to_banks(4 * state.read_xrf(vmem),
                                                      state.read_xrf(insn.rs2))
        return original(insn, state)

    def load(self, state):
        data = state.read_dram(state.read_xrf(self.rs1), state.read_xrf(self.rs2))
        state.write_vmem(4 * state.read_xrf(self.rd), 0, data)

    def store(self, state):
        data = state.read_vmem(4 * state.read_xrf(self.rs1), 0, state.read_xrf(self.rs2))
        state.write_dram(state.read_xrf(self.rd), data)

    bank_conflict.vmem_accesses = dma_unit.vmem_accesses = vmem_accesses
    isa_definition._DMA_LOAD_CHN.exec = load
    isa_definition._DMA_STORE_CHN.exec = store


def fields(word: int, cls: type) -> dict[str, int]:
    bits = lambda lo, n: (word >> lo) & ((1 << n) - 1)
    if issubclass(cls, RType):
        return {"rd": bits(7, 5), "rs1": bits(15, 5), "rs2": bits(20, 5)}
    if issubclass(cls, (IType, CSRType)):
        return {"rd": bits(7, 5), "rs1": bits(15, 5), "imm": bits(20, 12)}
    if issubclass(cls, SType):
        return {"rs1": bits(15, 5), "rs2": bits(20, 5), "imm": (bits(25, 7) << 5) | bits(7, 5)}
    if issubclass(cls, SBType):
        imm = (bits(31, 1) << 12) | (bits(7, 1) << 11) | (bits(25, 6) << 5) | (bits(8, 4) << 1)
        return {"rs1": bits(15, 5), "rs2": bits(20, 5), "imm": imm}
    if issubclass(cls, UType):
        return {"rd": bits(7, 5), "imm": bits(12, 20)}
    if issubclass(cls, UJType):
        imm = (bits(31, 1) << 20) | (bits(12, 8) << 12) | (bits(20, 1) << 11) | (bits(21, 10) << 1)
        return {"rd": bits(7, 5), "imm": imm}
    if issubclass(cls, VLSType):
        return {"vd": bits(7, 6), "rs1": bits(15, 5), "imm": bits(20, 12)}
    if issubclass(cls, VRType):
        return {"vd": bits(7, 6), "vs1": bits(13, 6), "es1": bits(13, 6), "vs2": bits(19, 6)}
    if issubclass(cls, VIType):
        return {"vd": bits(7, 6), "imm": bits(16, 16)}
    raise ValueError(f"unknown format for {cls.__name__}")


IMMEDIATE_BITS = ((SBType, 13), (UJType, 21), (UType, 20), (VIType, 16))


def candidates(cls: type, raw: dict[str, int]):
    """`cls` built from `raw`, with the immediate unsigned and then signed."""
    width = next((bits for kind, bits in IMMEDIATE_BITS if issubclass(cls, kind)), 12)
    for signed in (False, True):
        kwargs = {}
        try:
            for param in cls.params:
                for named in ((param.reg, param.imm) if isinstance(param, Bundled) else (param,)):
                    value = raw[named.repr]
                    if signed and named.repr == "imm" and value & (1 << (width - 1)):
                        value -= 1 << width
                    kwargs[named.repr] = named.inner(value)
            yield cls(**kwargs)
        except (TypeError, ValueError, ExceptionGroup):
            continue


def rtl_word(instruction) -> int:
    if not isinstance(instruction, VRType):
        return instruction.to_bytecode() & 0xFFFFFFFF
    operand = lambda *names: next((getattr(instruction, n) for n in names if hasattr(instruction, n)), 0)
    return ((instruction.funct7 << 25) | (operand("vs2") << 19) | (operand("vs1", "es1") << 13) |
            (operand("vd") << 7) | instruction.opcode)


def decode(word: int):
    opcode = word & 0x7F
    if opcode == 0x7F:  # RTL: dma.config has funct7=0, dma.wait funct7=1
        channel, funct7 = (word >> 12) & 7, word >> 25
        if funct7 == 0 and word & 0x01F00F80 == 0:
            return IsaSpec.operations[f"dma.config.ch{channel}"](rs1=ScalarReg((word >> 15) & 0x1F))
        if funct7 == 1 and word & 0x01FF8F80 == 0:
            return IsaSpec.operations[f"dma.wait.ch{channel}"]()
        raise ValueError(f"unexpected DMA control word {word:08x}")
    for cls in IsaSpec.operations.values():
        if cls.opcode == opcode:
            for instruction in candidates(cls, fields(word, cls)):
                if rtl_word(instruction) == word:
                    return instruction
    raise ValueError(f"no npu_model instruction encodes {word:08x}")


def owners(program: list[dict], timed: str) -> list[str]:
    """Each word's virtual operation, as `id:name`, from the CFG contract."""
    contract = timed[timed.index("atlas.virtual_cfg_contract"):]
    names = {int(i): name.removeprefix("atlas.virtual_") for i, name in re.findall(
        r'id = (\d+) : i32, mxu_commands = array<i32[^>]*>, name = "([^"]+)"', contract)}
    labels = []
    for word in program:
        f = word["fields"]
        source = f.get("atlas.virtual_cfg_source", f.get("atlas.virtual_cfg_operation"))
        label = (f"{source}:{names.get(source, '?')}" if source is not None
                 else word["operation"].removeprefix("atlas."))
        if f.get("atlas.virtual_cfg_helper"):
            label += f" ({f['atlas.virtual_cfg_helper']} helper)"
        labels.append(label)
    return labels


def simulate(directory: Path, hardware: str, max_cycles: int) -> dict:
    word_addressed_dma()
    program = json.loads((directory / "5-program.json").read_text())["instructions"]
    labels = owners(program, (directory / "4-timed.mlir").read_text())
    sim = Simulation(hardware_config=getattr(hw_configs, hardware)(),
                     logger_config=LoggerConfig(filename="/dev/null"),
                     program=InstantiableProgram([decode(w["word_u32"]) for w in program]),
                     verbose=False, record_timeline=True)
    cycles = []
    tick = sim.core.tick

    def recording_tick():
        tick()
        cycles.append(dict(sim.core.last_cycle))

    sim.core.tick = recording_tick
    try:
        with contextlib.redirect_stdout(io.StringIO()):  # DMA completion prints
            sim.run(max_cycles=max_cycles)
        stats = sim.get_stats()
        state = sim.core.arch_state
        halted, reason = state.halted, getattr(state, "halt_reason", None)
        timeline = sim.timeline
    finally:
        sim.close()
    profile: OrderedDict[str, dict] = OrderedDict()
    for cycle, busy in zip(cycles, timeline):
        pc = cycle["s1_pc"]
        label = labels[pc] if pc is not None and pc < len(labels) else "(drain)"
        row = profile.setdefault(label, dict.fromkeys(
            ("cycles", "issue", "delay", "dma.wait", "other", "dma_busy"), 0))
        kind = "issue" if cycle["s1_fire"] else cycle["stall"] or "other"
        row["cycles"] += 1
        row[kind if kind in row else "other"] += 1
        row["dma_busy"] += busy["DMA0"]
    return {"cycles": stats.cycles, "instructions": stats.total_instructions,
            "halted": halted, "halt_reason": reason, "runtime_errors": stats.runtime_errors,
            "units": {name: vars(s) for name, s in stats.exu_stats.items()},
            "profile": profile}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("artifacts", type=Path, help="a lower_atlas_program.py output directory")
    parser.add_argument("--hardware", default="FullDramHardwareConfig",
                        help="npu_model hardware config (the default covers the 0x9000_0000 DRAM ABI)")
    parser.add_argument("--max-cycles", type=int, default=200000)
    parser.add_argument("--profile", action="store_true",
                        help="print cycles per virtual operation holding the frontend")
    parser.add_argument("--json", type=Path, help="write the statistics and profile here")
    args = parser.parse_args()

    result = simulate(args.artifacts, args.hardware, args.max_cycles)
    dma = result["units"]["DMA0"]["busy_cycles"]
    print(f"{result['cycles']} cycles, DMA busy {dma} ({dma / result['cycles']:.1%}), "
          f"halted={result['halted']} ({result['halt_reason']}), "
          f"runtime_errors={result['runtime_errors']}")
    for name, unit in result["units"].items():
        print(f"  {name:<8} busy {unit['busy_cycles']:>6} ({unit['utilization']:.1%}), "
              f"{unit['instructions']} instructions")
    if args.profile:
        print(f"\n{'virtual op holding S1':<40} {'cycles':>6} {'issue':>6} {'delay':>6} "
              f"{'wait':>6} {'other':>6} {'DMA busy':>8}")
        for label, row in result["profile"].items():
            print(f"{label:<40} {row['cycles']:>6} {row['issue']:>6} {row['delay']:>6} "
                  f"{row['dma.wait']:>6} {row['other']:>6} {row['dma_busy']:>8}")
    if args.json:
        args.json.write_text(json.dumps(result, indent=1) + "\n")
    ok = result["halted"] and result["halt_reason"] == "ecall" and not result["runtime_errors"]
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
