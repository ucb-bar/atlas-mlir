"""atlasc build: model -> capture -> policy -> importer -> program.bin -> host ELF."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import host, policy as policies, toolchain
from .capture import Capture, capture, load

MARKER = ".atlasc"


@dataclass
class BuildResult:
    out: Path
    program: Path
    words: int
    elf: Path | None
    report: Path
    notes: list[str] = field(default_factory=list)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fresh(out: Path, force: bool) -> None:
    """The tools refuse stale output, so each build gets an empty directory.

    --force only replaces a directory an earlier atlasc build made.
    """
    if out.exists() and any(out.iterdir()):
        if not force:
            raise FileExistsError(f"{out} is not empty; pass --force to replace an earlier build")
        if not (out / MARKER).is_file():
            raise FileExistsError(f"{out} was not made by atlasc; refusing to replace it")
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / MARKER).write_text("atlasc build output\n")


def _load_inputs(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    from safetensors.numpy import load_file
    return load_file(str(path))


def build(source: Path | None = None, *, out: Path, model: Any = None,
          inputs: tuple[Any, ...] | None = None, captured: Capture | None = None,
          example_inputs: Path | None = None, policy: str | Path = "fp8",
          force: bool = False, host_cc: str | None = None, host_elf: bool = True) -> BuildResult:
    """Compile one model. Give a loader .py or .pt2 path, a model and inputs, or a capture."""
    out = Path(out)
    bins = toolchain.bin_dir()
    _fresh(out, force)
    if captured is None:
        if model is None:
            if source is None:
                raise ValueError("give a model source, a model and inputs, or a captured Linalg")
            model, inputs = load(Path(source))
        captured = capture(model, tuple(inputs), out / "capture")
        example_inputs = captured.inputs

    linalg = captured.linalg.read_text()
    policy_path = out / "policy.json"
    policy_path.write_text(json.dumps(policies.resolve(policy, linalg), indent=2) + "\n")
    compiled = out / "compiled"
    toolchain.run_tool("compile_atlas_linalg_tile.py", "--linalg", captured.linalg,
                       "--argument-manifest", captured.arguments, "--weights", captured.weights,
                       "--policy", policy_path, "--atlas-bin-dir", bins,
                       "--output-dir", compiled, step="compile")
    program = compiled / "program.bin"
    words = program.stat().st_size // 4

    cc = toolchain.host_cc(host_cc) if host_elf else None
    packaged = host.package(compiled, out, _load_inputs(example_inputs), cc)
    notes = []
    if not host_elf:
        packaged["elf_skipped"] = "--no-host-elf"
    if packaged.get("elf_skipped"):
        notes.append(f"host ELF skipped: {packaged['elf_skipped']}")

    manifest = json.loads((compiled / "import-manifest.json").read_text())
    report = {
        "schema": "atlas_mlir.build_report.v1",
        "status": "compiled",
        "source": str(source) if source else None,
        "selected_rtl_revision": manifest["selected_rtl_revision"],
        "policy": str(policy),
        "mxu_units": manifest["mxu_units"],
        "program": {"path": str(program), "words": words, "sha256": _sha256(program)},
        "inputs": [{"name": slot["source_key"], "dram_address": hex(slot["dram_address"]),
                    "role": slot["physical_role"]}
                   for slot in manifest["input_slots"] if slot["source_kind"] == "input"],
        "outputs": [{"index": row["index"], "dram_address": hex(row["dram_address"]),
                     "layout": row["physical_layout"]} for row in manifest["outputs"]],
        "host": packaged,
        "toolchain": {"bin_dir": str(bins), "host_cc": cc},
        "notes": notes,
    }
    report_path = out / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    elf = Path(packaged["elf"]) if packaged["elf"] else None
    return BuildResult(out, program, words, elf, report_path, notes)


def compile(model: Any, inputs: tuple[Any, ...], out: str | Path, *,
            policy: str | Path = "fp8", force: bool = False,
            host_cc: str | None = None) -> BuildResult:
    """Python entry point: compile a torch.nn.Module (or ExportedProgram) for Atlas."""
    return build(out=Path(out), model=model, inputs=tuple(inputs), policy=policy,
                 force=force, host_cc=host_cc)
