"""Compile and replay the public E3 shape seeds through the installed native CLI.

This authoring test supplies source shapes and fixed addresses only. Runtime
tensor values and the independent expected outputs enter the later evaluator.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from merlin.semantic_compiler.search import SearchLimits
from merlin.semantic_compiler.snapshot import open_native_snapshot
from merlin.semantic_compiler.target_binding import verify_native_publication

from atlas_native_support.mxu0 import Mxu0Contract
from atlas_native_support.mxu0_emit import verify_mxu0_artifact
from atlas_native_support.mxu0_program_set import verify_mxu0_program_set

from generate_native_shape_seeds import SHAPES, make_case


def encoded(value: object) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def compile_one(
    cli: Path, snapshot: Path, target_source: Path, root: Path,
    shape: tuple[int, int, int], contract: Mxu0Contract,
    limits: SearchLimits | None = None,
) -> dict:
    name = "x".join(map(str, shape))
    case_root = root / name
    case_root.mkdir(parents=True)
    request, abi = make_case(shape, contract)
    request_path, abi_path = case_root / "request.json", case_root / "abi.json"
    request_path.write_text(encoded(request.record()))
    abi_path.write_text(encoded(abi))
    command = [
        str(cli), "native-compile", "--engine", "merlin_native", "--support", "atlas_tensor",
        "--snapshot", str(snapshot), "--request", str(request_path), "--abi", str(abi_path),
        "--target-source", str(target_source), "--mode", "strict-native",
        "--out", str(case_root / "compiled"), "--status-file", str(case_root / "status.json"),
    ]
    if limits is not None:
        limit_path = case_root / "search-limits.json"
        limit_path.write_text(encoded(limits.record()))
        command.extend(("--search-limits", str(limit_path)))
    result = subprocess.run(command, text=True, capture_output=True, timeout=90, check=False)
    (case_root / "compile.stdout").write_text(result.stdout)
    (case_root / "compile.stderr").write_text(result.stderr)
    if result.returncode:
        raise RuntimeError(f"{name}: native compile failed with exit {result.returncode}: {result.stderr}")
    status = json.loads((case_root / "status.json").read_text())
    artifact = case_root / "compiled"
    manifest = json.loads((artifact / "manifest.json").read_text())
    if status.get("status") != "emitted" or status.get("engine") != "merlin_native":
        raise ValueError(f"{name}: native CLI did not report emitted status")
    verify_native_publication(
        artifact, manifest, engine="merlin_native", request_digest=request.digest(),
        target_identity=contract.identity,
    )
    loaded = open_native_snapshot(snapshot)
    if manifest.get("artifact_kind") == "program_set":
        verify_mxu0_program_set(
            loaded, artifact, rtl_root=target_source,
            expected_request_digest=request.digest(),
        )
        programs = len(manifest["segments"])
    else:
        verify_mxu0_artifact(
            loaded, artifact, rtl_root=target_source,
            expected_request_digest=request.digest(),
        )
        programs = 1
    return {
        "shape": list(shape), "status": status["status"],
        "artifact_kind": manifest.get("artifact_kind", "program"),
        "program_count": programs, "request_digest": request.digest(),
        "compiled": str(artifact),
        "partition_trigger": manifest.get("partition_trigger"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--target-source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("output path must be fresh")
    cli = Path(sys.executable).with_name("merlin-targetgen")
    if not cli.is_file():
        parser.error("installed merlin-targetgen command is unavailable")
    snapshot, target_source = (
        path.resolve(strict=True) for path in (args.snapshot, args.target_source)
    )
    loaded = open_native_snapshot(snapshot)
    contract = Mxu0Contract.load()
    if loaded.profile.target_identity != contract.identity:
        parser.error("snapshot and selected Atlas contract differ")
    args.out.mkdir(parents=True)
    rows = [compile_one(cli, snapshot, target_source, args.out, shape, contract)
            for shape in SHAPES]
    # The tighter node cap must reject the combined graph and resume through
    # independently selected complete ordered-K output chains.
    limited = SearchLimits(candidate_nodes=12)
    forced = compile_one(
        cli, snapshot, target_source, args.out / "forced-fallback",
        (31, 32, 33), contract, limited,
    )
    if (forced["artifact_kind"] != "program_set"
            or forced["partition_trigger"] != "combined_resource_limit"
            or forced["program_count"] != 2):
        raise ValueError("tight-budget case did not exercise search-feedback partition")
    receipt = {
        "schema": "atlas.native_shape_seed_compile.v1",
        "engine": "merlin_native", "selected_rtl_revision": contract.record["rtl_revision"],
        "seed_status": "pass", "seed_count": len(rows), "cases": rows,
        "forced_fallback": forced,
        "scope": "typed native compile and target verifier replay; hardware execution separate",
    }
    (args.out / "receipt.json").write_text(encoded(receipt))
    print(json.dumps({"seeds": len(rows), "forced_fallback": "pass",
                      "receipt": str(args.out / "receipt.json")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
