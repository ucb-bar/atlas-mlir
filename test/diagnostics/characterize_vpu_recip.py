"""Characterize every BF16 VRECIP input code on one selected AtlasCore.

This evaluator-only diagnostic uses an integer transcription of the selected
RcpLUT/LUTParams computation. It does not provide a production implementation
or make npu_model's torch.reciprocal an oracle for the selected hardware.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
ASSEMBLY = ROOT / "test/examples/vpu_recip_pair.S"
PANEL_CODES = 1024
TOTAL_CODES = 1 << 16
PANELS = TOTAL_CODES // PANEL_CODES
INPUT_BASE = 0x90000000
OUTPUT_BASE = 0x90000800
GUARD_BASE = 0x90001000


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_identity(rtl: Path, arithmetic: Path) -> dict:
    revision = subprocess.run(
        ["git", "-C", str(rtl), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    gitlink = subprocess.run(
        ["git", "-C", str(rtl), "ls-tree", "HEAD", "dependencies/sp26-fp-units"],
        check=True, capture_output=True, text=True,
    ).stdout.strip().split()
    if revision != "0079c0541111197741a231c002e3843fa6f545b2":
        raise ValueError("selected RTL revision changed")
    if (len(gitlink) != 4 or gitlink[:2] != ["160000", "commit"]
            or gitlink[2] != "9a0cc09c41ab918a3548580185f39cca8d559e0d"):
        raise ValueError("selected arithmetic gitlink changed")
    paths = {
        "rtl_rcp": rtl / "src/main/scala/atlas/vector/laneBoxes/Rcp.scala",
        "lut": arithmetic / "src/main/scala/sp26-fp-units/vpuLUTs/RcpLUT.scala",
        "lut_params": arithmetic / "src/main/scala/sp26-fp-units/vpuLUTs/LUTParams.scala",
        "fp_type": arithmetic / "src/main/scala/sp26-fp-units/common.scala",
    }
    hashes = {name: _sha(path.resolve(strict=True)) for name, path in paths.items()}
    if hashes["rtl_rcp"] != "6520e13247f3256aa2b1b56a0ee5627ac2eb88296b5dea284cdb8dc265464faf":
        raise ValueError("selected RTL Rcp.scala bytes changed")
    return {"rtl_revision": revision, "arithmetic_gitlink": gitlink[2],
            "source_sha256": hashes,
        "arithmetic_checkout_note": "source bytes recorded; supplied arithmetic checkout revision not independently established"}


def selected_rtl_recip_bits(raw: int) -> int:
    """Source-derived RcpLUT integer model; BF16 passes m=1, n=16.

    The 128-entry LUT rounds positive rational values as Scala math.round.
    The conversion in LUTParams truncates to seven fraction bits and wraps the
    signed exponent to eight bits before its special-case comparison.
    """
    if not 0 <= raw <= 0xFFFF:
        raise ValueError("BF16 encoding must fit 16 bits")
    sign = raw & 0x8000
    exponent = (raw >> 7) & 0xFF
    mantissa = raw & 0x7F
    denominator = 128 + mantissa
    numerator = 128 << 16
    scaled = (2 * numerator + denominator) // (2 * denominator)
    highest = scaled.bit_length() - 1
    result_exponent = (254 + highest - 16 - exponent) & 0xFF
    result_fraction = (scaled >> max(highest - 7, 0)) & 0x7F
    if exponent == 0xFF:
        return sign
    if exponent == 0 or result_exponent == 0xFF:
        return sign | 0x7F80
    return sign | (result_exponent << 7) | result_fraction


def _assembler_words(assembler_path: Path) -> tuple[int, ...]:
    spec = importlib.util.spec_from_file_location("atlas_selected_recip_assembler", assembler_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("selected Atlas assembler cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    words = tuple(module.assemble(ASSEMBLY.read_text()))
    if len(words) != 36 or words[17] != module.VRECIP_BF16(4, 0):
        raise AssertionError("transcribed VRECIP stream changed")
    return words


def _drive(model: Path, state: Path, modelir: Path, words: tuple[int, ...],
           first_panel: int, panel_count: int) -> tuple[list[dict], list[dict]]:
    original_cwd = Path.cwd()
    sys.path.insert(0, str(modelir))
    try:
        os.chdir(modelir)
        from mlc.backends import cosim_atlas

        original_core = cosim_atlas.CosimCore

        class SelectedCore(original_core):
            def peek(self, name: str) -> int:
                return super().peek("scalar/halt_now" if name == "io_halted" else name)

            def poke(self, name: str, value: int) -> None:
                if name in ("io_dmaTL_d_bits_opcode", "io_dmaTL_d_bits_size"):
                    if name in self._S:
                        raise AssertionError(f"unexpected live ARC input {name}")
                    return
                super().poke(name, value)

        cosim_atlas.CosimCore = SelectedCore
        try:
            def run_panels() -> tuple[list[dict], list[dict]]:
                core = SelectedCore(model, state)
                core.reset()
                imem = cosim_atlas.TileLinkAdapter(core, "imemTL")
                csr = cosim_atlas.TileLinkAdapter(core, "csrTL")
                slave = cosim_atlas.TileLinkSlave(
                    core, "dmaTL", size_bytes=cosim_atlas._DRAM_WINDOW, beat_bytes=32,
                )
                for index, word in enumerate(words):
                    if not imem.put(cosim_atlas._IMEM_BASE + 4 * index, word, size=2):
                        raise AssertionError("IMEM write failed")
                rows: list[dict] = []
                differences: list[dict] = []
                for panel in range(first_panel, first_panel + panel_count):
                    values = range(panel * PANEL_CODES, (panel + 1) * PANEL_CODES)
                    source = struct.pack("<1024H", *values)
                    expected = struct.pack(
                        "<1024H", *(selected_rtl_recip_bits(code) for code in values)
                    )
                    guard = b"\x5a" * 32
                    slave.preload(INPUT_BASE, source[:1024])
                    slave.preload(INPUT_BASE + 1024, source[1024:])
                    slave.preload(OUTPUT_BASE, b"\xa5" * 2048)
                    slave.preload(GUARD_BASE, guard)
                    reads_before, writes_before = slave.reads, slave.writes
                    if not csr.put(cosim_atlas._START_CSR, 1, size=2):
                        raise AssertionError("CSR start failed")
                    started = False
                    for cycle in range(8000):
                        slave.step()
                        halted = core.peek("io_halted") == 1
                        if not halted:
                            started = True
                        core.tick()
                        if started and halted:
                            break
                    else:
                        raise AssertionError(f"panel {panel} did not halt")
                    if core.peek("csrfile/reg_halt_reason") != 2 or slave._q:
                        raise AssertionError(f"panel {panel} did not drain and halt by ECALL")
                    observed = slave.captured(OUTPUT_BASE, 2048)
                    if slave.captured(INPUT_BASE, 2048) != source:
                        raise AssertionError(f"panel {panel} changed an input")
                    if slave.captured(GUARD_BASE, 32) != guard:
                        raise AssertionError(f"panel {panel} changed the guard")
                    if (slave.reads - reads_before, slave.writes - writes_before) != (64, 64):
                        raise AssertionError(f"panel {panel} DMA beat count changed")
                    mismatches = 0
                    for offset, (actual, wanted) in enumerate(zip(
                        struct.unpack("<1024H", observed),
                        struct.unpack("<1024H", expected), strict=True,
                    )):
                        if actual != wanted:
                            mismatches += 1
                            if len(differences) < 32:
                                differences.append({
                                    "input": f"{panel * PANEL_CODES + offset:04x}",
                                    "observed": f"{actual:04x}",
                                    "expected": f"{wanted:04x}",
                                })
                    rows.append({
                        "panel": panel,
                        "cycles": cycle,
                        "checked_codes": PANEL_CODES,
                        "mismatches": mismatches,
                        "dma_reads": slave.reads - reads_before,
                        "dma_writes": slave.writes - writes_before,
                        "output_sha256": hashlib.sha256(observed).hexdigest(),
                    })
                return rows, differences

            return cosim_atlas.large_stack_call(run_panels)
        finally:
            cosim_atlas.CosimCore = original_core
    finally:
        os.chdir(original_cwd)
        sys.path.remove(str(modelir))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arc-model", required=True, type=Path)
    parser.add_argument("--arc-state", required=True, type=Path)
    parser.add_argument("--modelir", required=True, type=Path)
    parser.add_argument("--assembler", required=True, type=Path)
    parser.add_argument("--rtl-root", required=True, type=Path)
    parser.add_argument("--arithmetic-root", required=True, type=Path)
    parser.add_argument("--first-panel", type=int, default=0)
    parser.add_argument("--panel-count", type=int, default=PANELS)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if not 0 <= args.first_panel < PANELS or not 1 <= args.panel_count <= PANELS - args.first_panel:
        parser.error("panel range must lie within the 64 BF16 code panels")
    output = args.out.resolve()
    if output.exists():
        raise FileExistsError("reciprocal characterization requires a fresh output root")
    model = args.arc_model.resolve(strict=True)
    state = args.arc_state.resolve(strict=True)
    modelir = args.modelir.resolve(strict=True)
    assembler = args.assembler.resolve(strict=True)
    source = _source_identity(args.rtl_root.resolve(strict=True),
                              args.arithmetic_root.resolve(strict=True))
    words = _assembler_words(assembler)
    started = time.monotonic()
    rows, differences = _drive(model, state, modelir, words,
                               args.first_panel, args.panel_count)
    duration = time.monotonic() - started
    output.mkdir(parents=True)
    report = {
        "schema": "atlas.handwritten_vpu_recip_all_codes.v1",
        "scope": "selected standalone AtlasCore BF16 VRECIP raw codes; source-derived LUT checker",
        "status": "pass" if not any(row["mismatches"] for row in rows) else "fail",
        "complete_domain": args.first_panel == 0 and args.panel_count == PANELS,
        "first_panel": args.first_panel,
        "panels": len(rows),
        "checked_codes": sum(row["checked_codes"] for row in rows),
        "mismatches": sum(row["mismatches"] for row in rows),
        "first_differences": differences,
        "seconds": duration,
        "core_instances": 1,
        "core_resets": 1,
        **source,
        "arc_model_sha256": _sha(model),
        "arc_state_sha256": _sha(state),
        "assembler_sha256": _sha(assembler),
        "assembly_sha256": _sha(ASSEMBLY),
        "diagnostic_sha256": _sha(Path(__file__)),
        "program_words_sha256": hashlib.sha256(struct.pack("<36I", *words)).hexdigest(),
        "source_model_parameters": {"lut_entries": 128, "fixed_fraction_bits": 16,
                                    "fixed_integer_bits": 1, "output_fraction_bits": 7},
        "panels_result": rows,
    }
    (output / "receipt.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in (
        "status", "complete_domain", "panels", "checked_codes", "mismatches", "seconds",
    )}))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
