#!/usr/bin/env python3
"""Choose a fully verified Atlas program using cold cycles from the selected npu_model.

The default is retained as candidate zero, ties retain the earlier candidate,
and builtin compiles exactly once without importing the model. The bounded
search only exposes existing scheduling and free DMA-window choices.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import threading
import time
from typing import Callable


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Candidate:
    lookahead: int = 0
    random_seed: int | None = None
    store_windows: tuple[int, ...] = ()
    wait_placement: str = "default"

    def __post_init__(self) -> None:
        if self.lookahead not in (0, 1, 2, 4, 8):
            raise ValueError("lookahead must be 0, 1, 2, 4, or 8")
        if self.random_seed is not None and (self.random_seed < 0 or self.lookahead):
            raise ValueError("random-seed must be nonnegative and excludes lookahead")
        if any(rank < 0 or rank > 0xFFFFFFFF for rank in self.store_windows):
            raise ValueError("store-window ranks must be unsigned 32-bit integers")
        if self.wait_placement not in ("default", "source"):
            raise ValueError("wait-placement must be default or source")


def _option_path(path: Path) -> str:
    return '"' + str(path.resolve()).replace("\\", "\\\\").replace('"', '\\"') + '"'


def evidence_option(op_timing: Path | None = None, op_timing_sha256: str | None = None,
                    pairs: Path | None = None, pairs_sha256: str | None = None,
                    composed: Path | None = None, composed_sha256: str | None = None) -> list[str]:
    if not op_timing:
        if op_timing_sha256 or pairs or pairs_sha256 or composed or composed_sha256:
            raise ValueError("model evidence requires --op-timing and its digest")
        return []  # The source may already carry selected model evidence.
    if not op_timing_sha256 or (bool(pairs) != bool(pairs_sha256)) or (bool(composed) != bool(composed_sha256)):
        raise ValueError("each model evidence file requires its explicit SHA-256 digest")
    if composed and not pairs:
        raise ValueError("composed model evidence requires its pinned pairs")
    options = f"op-timing={_option_path(op_timing)} op-timing-sha256={op_timing_sha256} dma=wait"
    if pairs:
        options += f" pairs={_option_path(pairs)} pairs-sha256={pairs_sha256} chains=true"
    if composed:
        options += f" composed={_option_path(composed)} composed-sha256={composed_sha256}"
    return ["--select-atlas-model-evidence=" + options]


def candidate_options(candidate: Candidate, provider: str, spill: bool) -> list[str]:
    options = [f"provider={provider}", f"spill={str(spill).lower()}", "allocation-report=allocation.json"]
    if candidate.lookahead:
        options.append(f"lookahead={candidate.lookahead}")
    if candidate.random_seed is not None:
        options.append(f"random-seed={candidate.random_seed}")
    if candidate.store_windows:
        options.append("store-windows='" + ",".join(map(str, candidate.store_windows)) + "'")
    if candidate.wait_placement != "default":
        options.append("wait-placement=" + candidate.wait_placement)
    return ["--atlas-compile-candidate=" + " ".join(options)]


def select_program(source: Path, output_dir: Path, atlas_bin_dir: Path, *, provider: str = "model",
                   profile: str = "fifo43", model: Path | None = None, model_regions: tuple[Path, ...] = (),
                   spill: bool = True, spill_mxu: bool = True, boundary: str = "boundary",
                   guarded_pack: bool = False,
                   op_timing: Path | None = None, op_timing_sha256: str | None = None,
                   pairs: Path | None = None, pairs_sha256: str | None = None,
                   composed: Path | None = None, composed_sha256: str | None = None,
                   max_candidates: int = 24, random_seeds: tuple[int, ...] = tuple(range(8)), workers: int = 12,
                   store_window_candidates: tuple[tuple[int, ...], ...] = (),
                   scorer: Callable | None = None) -> dict:
    """Return a receipt; selected artifacts are also copied to output_dir.

    All candidates run allocation, timing and generated-schedule verification
    before scoring. Failed alternatives remain in the receipt and cannot win;
    a failed default or mismatched model identity fails the compilation.
    max_candidates is a deterministic bound, including rejected candidates.
    """
    if provider not in ("builtin", "model") or profile not in ("fifo43", "curve"):
        raise ValueError("unsupported provider or memory profile")
    if max_candidates < 1 or workers < 1 or boundary not in ("boundary", "none"):
        raise ValueError("candidate and worker counts must be positive and boundary must be boundary or none")
    source, output, binaries = Path(source).resolve(strict=True), Path(output_dir).resolve(), Path(atlas_bin_dir).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("output directory must be empty to prevent stale artifacts")
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    commands, logs = [], []
    log_lock, score_lock = threading.Lock(), threading.Lock()
    receipt = {"schema": "atlas.model_selection.v1", "provider": provider, "profile": profile,
               "guarded_pack": guarded_pack,
               "max_candidates": max_candidates, "workers": workers,
               "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
               "candidates": []}

    def command(argv: list[str], directory: Path, stdout: Path | None = None) -> str:
        rendered = f"cd {shlex.quote(str(directory))} && " + shlex.join(argv)
        if stdout:
            rendered += f" > {shlex.quote(str(stdout))}"
        with log_lock:
            commands.append(rendered)
            (output / "commands.txt").write_text("\n".join(commands) + "\n")
        try:
            result = subprocess.run(argv, cwd=directory, capture_output=True, text=True,
                                    env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
        except OSError as error:
            result = subprocess.CompletedProcess(argv, 1, "", str(error))
        with log_lock:
            logs.append(f"$ {rendered}\n{result.stderr}exit {result.returncode}\n")
            (output / "log.txt").write_text("\n".join(logs))
        if result.returncode:
            raise ValueError(result.stderr.strip() or f"{argv[0]} exited {result.returncode}")
        if stdout:
            stdout.write_text(result.stdout)
        return result.stdout

    options = evidence_option(op_timing, op_timing_sha256, pairs, pairs_sha256,
                              composed, composed_sha256) if provider == "model" else []
    options.append("--atlas-compile-prepare=" +
                   f"spill-mxu-state={str(spill_mxu).lower()} boundary={boundary}")
    if guarded_pack:
        options[-1] += " guarded-pack=true"
    prepared = output / "1-virtual.mlir"
    command([str(binaries / "atlas-opt"), str(source), *options,
             "--mlir-print-op-on-diagnostic=false", "-o", str(prepared)], output)
    standard_scorer = scorer is None
    if standard_scorer and provider == "model":
        if __package__:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
        from atlas_model_scorer import ModelScorer
        if model is None:
            raise ValueError("model scoring requires --npu-model")
        scorer = ModelScorer(model, model_regions)
    seen: set[Candidate] = set()
    score_cache: dict[tuple[int, ...], int] = {}
    best: dict | None = None

    def compile_candidate(candidate: Candidate, row: dict) -> None:
        directory = output / row["directory"]
        before = time.monotonic()
        try:
            command([str(binaries / "atlas-opt"), str(prepared), *candidate_options(candidate, provider, spill),
                     "--mlir-print-op-on-diagnostic=false", "-o", "final.mlir"], directory)
            program = json.loads(command([str(binaries / "atlas-emit"), "final.mlir", "--program-json"],
                                         directory, directory / "program.json"))
            words = tuple(instruction["word_u32"] for instruction in program["instructions"])
            (directory / "program.hex").write_text("".join(f"{word:08x}\n" for word in words))
            if provider == "model":
                with score_lock:
                    if words not in score_cache:
                        score_cache[words] = scorer(program, profile=profile)
                    row["cycles"] = score_cache[words]
            row.update(status="PASS", words=len(words), program_sha256=hashlib.sha256(
                b"".join(word.to_bytes(4, "little") for word in words)).hexdigest())
        except Exception as error:
            row.update(status="FAIL", error=str(error))
            if row["index"] == 0:
                raise ValueError("default candidate failed: " + str(error)) from error
        finally:
            row["elapsed_seconds"] = time.monotonic() - before
    def evaluate_batch(specifications: list[tuple[Candidate, str]]) -> None:
        nonlocal best
        pending = []
        for candidate, category in specifications:
            if candidate in seen or len(receipt["candidates"]) >= (max_candidates if provider == "model" else 1):
                continue
            seen.add(candidate)
            index = len(receipt["candidates"])
            directory = output / "candidates" / f"{index:03d}-{category}"
            directory.mkdir(parents=True)
            row = {"index": index, "class": category, "options": asdict(candidate),
                   "directory": str(directory.relative_to(output))}
            receipt["candidates"].append(row)
            pending.append((candidate, row))
        with ThreadPoolExecutor(max_workers=min(workers, max(1, len(pending)))) as executor:
            futures = [executor.submit(compile_candidate, candidate, row) for candidate, row in pending]
            for future in futures:
                future.result()
        best = min((row for row in receipt["candidates"] if row["status"] == "PASS"),
                   key=lambda row: (row.get("cycles", 0), row["index"]))

    evaluate_batch([(Candidate(), "default")])
    if provider == "model":
        evaluate_batch([(Candidate(lookahead=lookahead), "lookahead") for lookahead in (1, 2, 4)] +
                       [(Candidate(random_seed=seed), "random-seed") for seed in random_seeds])
        def winner() -> Candidate:
            options = dict(best["options"])
            options["store_windows"] = tuple(options["store_windows"])
            return Candidate(**options)

        program = json.loads((output / best["directory"] / "program.json").read_text())
        store_count = sum(instruction.get("operation") == "atlas.dma" and
                          instruction.get("fields", {}).get("direction") == "store"
                          for instruction in program["instructions"])
        # All legal per-store free ranks are expressible; allocation maps ranks
        # modulo that store's actual free set and never claims a held window.
        window_base = winner()
        placements = [(replace(window_base, wait_placement="source"), "wait-placement")]
        placements += [(replace(window_base, store_windows=tuple(ranks)), "explicit-store-windows")
                       for ranks in store_window_candidates]
        placements += [(replace(window_base, store_windows=(rank,) * store_count), "uniform-store-windows")
                       for rank in range(1, 8) if store_count]
        evaluate_batch(placements)
    selected = output / best["directory"]
    for name in ("final.mlir", "allocation.json", "program.json", "program.hex"):
        shutil.copyfile(selected / name, output / name)
    receipt.update(selected_index=best["index"], selected_class=best["class"],
                   default_cycles=receipt["candidates"][0].get("cycles"),
                   selected_cycles=best.get("cycles"),
                   elapsed_seconds=time.monotonic() - started)
    (output / "selection.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--provider", choices=("builtin", "model"), default="builtin")
    parser.add_argument("--profile", choices=("fifo43", "curve"), default="fifo43")
    parser.add_argument("--npu-model", type=Path, help="selected npu_model source tree; must match provider evidence")
    parser.add_argument("--model-region", type=Path, action="append", default=[],
                        help="DRAM region JSON with address, initial_hex and optional expected_hex")
    parser.add_argument("--op-timing", type=Path)
    parser.add_argument("--op-timing-sha256")
    parser.add_argument("--pairs", type=Path)
    parser.add_argument("--pairs-sha256")
    parser.add_argument("--composed", type=Path)
    parser.add_argument("--composed-sha256")
    parser.add_argument("--max-candidates", type=int, default=24)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--store-windows", action="append", default=[], metavar="RANKS",
                        help="additional candidate's comma-separated free ranks, one per DMA store")


def model_arguments(args: argparse.Namespace) -> dict:
    return {"provider": args.provider, "profile": args.profile, "model": args.npu_model,
            "model_regions": tuple(args.model_region), "op_timing": args.op_timing,
            "op_timing_sha256": args.op_timing_sha256, "pairs": args.pairs,
            "pairs_sha256": args.pairs_sha256, "composed": args.composed,
            "composed_sha256": args.composed_sha256, "max_candidates": args.max_candidates,
            "workers": args.workers,
            "store_window_candidates": tuple(tuple(map(int, value.split(','))) for value in args.store_windows)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--atlas-bin-dir", type=Path,
                        default=Path(os.environ.get("ATLAS_OOT_BIN_DIR", ROOT / "build/bin")))
    parser.add_argument("--boundary", choices=("boundary", "none"), default="boundary")
    parser.add_argument("--no-spill", action="store_true")
    parser.add_argument("--guarded-pack", action="store_true")
    add_model_arguments(parser)
    parser.set_defaults(provider="model")
    args = parser.parse_args()
    try:
        receipt = select_program(args.source, args.output_dir, args.atlas_bin_dir,
                                 boundary=args.boundary, spill=not args.no_spill,
                                 guarded_pack=args.guarded_pack, **model_arguments(args))
    except (OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 1
    print(json.dumps({key: receipt[key] for key in ("selected_index", "selected_class", "default_cycles",
                                                  "selected_cycles", "elapsed_seconds")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
