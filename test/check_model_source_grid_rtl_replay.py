"""Bind a checked source-grid host bundle to observed AtlasCore tile outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


def check(bundle_path: Path, result_path: Path, observed: Path) -> dict:
    bundle = json.loads(bundle_path.read_text())
    result = json.loads(result_path.read_text())
    revision = result.get("rtl_revision")
    target = bundle.get("target_identity")
    if (bundle.get("schema") != "atlas.source_mn_host_data.v1"
            or bundle.get("engine") != "merlin_native"
            or bundle.get("rank") != 2 or bundle.get("batch_count") != 1
            or not isinstance(bundle.get("frozen_rhs"), dict)
            or bundle["frozen_rhs"].get("argument_index") != 1
            or result.get("schema") != "atlas.selected_core_test_run.v1"
            or result.get("status") != "passed_bounded_core_test"
            or type(revision) is not str or type(target) is not str
            or revision not in target
            or observed.resolve() != (result_path.parent / "observed").resolve()):
        raise ValueError("host bundle and selected-core result have incompatible identities")
    m, n, tile_count = (bundle.get(key) for key in ("m_rows", "n_rows", "tile_count"))
    if (type(m) is not int or not 1 <= m <= 32 or type(n) is not int or n <= 32
            or type(tile_count) is not int or tile_count != (n + 31) // 32):
        raise ValueError("replay requires the checked rank-two one-M-tile grid")
    names = [f"real_host_trial{trial}_n{n_start}"
             for trial in range(2) for n_start in range(0, n, 32)]
    fixtures = {item["name"]: item for item in result.get("fixtures", [])}
    outputs = {item["name"]: item for item in result.get("observed_outputs", [])}
    cases = {item["name"]: item for item in result.get("cases", [])}
    if (len(fixtures) != len(result.get("fixtures", []))
            or len(outputs) != len(result.get("observed_outputs", []))
            or set(fixtures) != set(names) or set(outputs) != set(names)
            or any(cases.get(name, {}).get("bf16_cells") != 1024 for name in names)):
        raise ValueError("selected-core result does not cover both host invocations")
    program_sha = bundle.get("program_sha256")
    if type(program_sha) is not str or re.fullmatch(r"[0-9a-f]{64}", program_sha) is None:
        raise ValueError("host bundle has no checked program identity")
    checked = []
    for name in names:
        path = observed / f"{name}.bin"
        raw = path.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        if (len(raw) != 2048 or fixtures[name].get("program_sha256") != program_sha
                or outputs[name].get("sha256") != sha):
            raise ValueError(f"changed or mismatched selected-core output: {name}")
        checked.append({"name": name, "sha256": sha})
    return {"status": "bound_rtl_outputs", "program_sha256": program_sha,
            "outputs": checked}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--rtl-result", required=True, type=Path)
    parser.add_argument("--observed", required=True, type=Path)
    args = parser.parse_args()
    try:
        print(json.dumps(check(args.bundle, args.rtl_result, args.observed), sort_keys=True))
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, json.dumps({"status": "replay_refused", "reason": str(error)}) + "\n")


if __name__ == "__main__":
    main()
