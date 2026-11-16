"""Consume Merlin's conservative scheduler for fixed-only physical streams.

Timing comes from the selected OOT provider before scheduling. Merlin constructs
and materializes the plan; the OOT verifies, encodes and independently exports
the final stream. DMA and control flow are explicitly outside this adapter.
"""

import argparse
import json
import struct
import subprocess
from hashlib import sha256
from pathlib import Path

from merlin.compile.scheduling.memory_effects import EncodedAccess
from merlin.compile.scheduling.static_schedule import (
    materialize_timing_plan,
    schedule_fixed_block,
    verify_timed_emission,
)
from merlin.compile.scheduling.timed_schedule import (
    ResourceUse,
    TimedAccess,
    TimedEffects,
)
from merlin.sched.ir import Call, Const, Kernel
from merlin.xdsl_dialects._common import text

from .shared_dense import RTL_REVISION


def render(rows):
    lines = ["module {", '  %s0 = "atlas.start"() : () -> !atlas.state']
    for i, row in enumerate(rows, 1):
        fields = []
        for name, value in row["fields"].items():
            if type(value) is str:
                encoded = json.dumps(value)
            elif type(value) is bool:
                encoded = str(value).lower()
            elif type(value) is int:
                encoded = f"{value} : i32"
            else:
                raise ValueError("unsupported machine field")
            fields.append(f"{name} = {encoded}")
        lines.append(
            f'  %s{i} = "{row["operation"]}"(%s{i - 1}) {{{", ".join(fields)}}} : (!atlas.state) -> !atlas.state'
        )
    return "\n".join((*lines, "}")) + "\n"


def compile_fixed(source, *, destination, atlas_bin_dir):
    destination, bins = Path(destination), Path(atlas_bin_dir)
    destination.mkdir(parents=True, exist_ok=False)
    path = destination / "source.mlir"
    path.write_text(source)

    def export(flag, source_path, untimed=False):
        command = [
            str(bins / "atlas-emit"),
            *(["--allow-untimed"] if untimed else []),
            flag,
            str(source_path),
        ]
        return json.loads(
            subprocess.run(command, check=True, capture_output=True, text=True).stdout
        )

    program = export("--program-json", path, True)
    evidence = export("--fixed-effects-json", path, True)
    if (
        evidence["schema"] != "atlas.fixed_effects.v1"
        or evidence["selected_rtl_revision"] != RTL_REVISION
        or program["selected_rtl_revision"] != RTL_REVISION
        or [r["word_u32"] for r in program["instructions"]]
        != [r["word_u32"] for r in evidence["instructions"]]
    ):
        raise ValueError("fixed evidence differs from actual source words")
    # Existing delay instructions carry no source computation; schedule afresh.
    rows, effects = [], {}
    calls = []
    for row, timing in zip(
        program["instructions"], evidence["instructions"], strict=True
    ):
        if timing["delay"]:
            continue
        call = Call(
            "word", (("index", Const(len(rows))), ("bits", Const(row["word_u32"])))
        )
        rows.append(row)
        calls.append(call)
        effects[call] = timing
    kernel = Kernel("fixed_stream", (), tuple(calls))

    def project(call):
        if call.instr == "idle":
            return TimedEffects("idle", call.arg("cycles").value)
        age = effects[call]["complete"]
        # Intentionally conservative projection: hold the complete machine until
        # every access/reservation in the trusted provider has expired.
        return TimedEffects(
            "serialized-machine",
            age,
            (TimedAccess(EncodedAccess("machine-state", 0, 1, True), 0, age),),
            (ResourceUse("machine", 0, age),),
        )

    def advance(call):
        return (
            call.arg("cycles").value
            if call.instr == "idle"
            else effects[call]["issue_gap"]
        )

    def delay(cycles):
        result = []
        while cycles:
            amount = min(
                cycles, 4096
            )  # DELAY's target-owned 12-bit immediate plus issue.
            result.append(Call("idle", (("cycles", Const(amount)),)))
            cycles -= amount
        return tuple(result)

    options = {
        "project": project,
        "capacities": {"machine": 1},
        "issue_gap": advance,
        "model_identity": evidence["provider"],
        "clock": "drained-block-relative",
    }
    plan = schedule_fixed_block(kernel, mode="serial", **options)
    emission = materialize_timing_plan(kernel, plan, delay=delay, **options)
    actual_rows = [
        (
            {
                "operation": "atlas.delay",
                "fields": {"cycles": c.arg("cycles").value - 1},
            }
            if c.instr == "idle"
            else rows[c.arg("index").value]
        )
        for c in emission.kernel.body
    ]
    (destination / "schedule.mlir").write_text(text(plan, generic=True))
    (destination / "schedule.mk").write_text(kernel.text())
    (destination / "emitted.mk").write_text(emission.kernel.text())
    (destination / "fixed-effects.json").write_text(
        json.dumps(evidence, indent=2) + "\n"
    )
    pending = destination / "scheduled.mlir"
    pending.write_text(render(actual_rows))
    final = destination / "machine.mlir"
    # Verification only: no target scheduling pass may repair the shared plan.
    subprocess.run(
        [
            str(bins / "atlas-opt"),
            str(pending),
            "--verify-atlas-timing",
            "--verify-atlas-machine-stream",
            "-o",
            str(final),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    emitted = export("--program-json", final)
    encoded = subprocess.run(
        [str(bins / "atlas-emit"), str(final)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    binary = b"".join(struct.pack("<I", int(v, 16)) for v in encoded.split())
    final_rows = emitted["instructions"]
    if len(final_rows) != len(emission.kernel.body):
        raise ValueError("final instruction count differs")
    for call, row in zip(emission.kernel.body, final_rows, strict=True):
        if call.instr == "idle":
            if row["operation"] != "atlas.delay" or row["fields"] != {
                "cycles": advance(call) - 1
            }:
                raise ValueError("final delay differs")
        elif row["word_u32"] != call.arg("bits").value:
            raise ValueError("final source instruction differs")
    if binary != b"".join(struct.pack("<I", r["word_u32"]) for r in final_rows):
        raise ValueError("binary differs from verified instruction stream")
    verify_timed_emission(kernel, plan, emission, **options)
    (destination / "program.json").write_text(json.dumps(emitted, indent=2) + "\n")
    (destination / "program.bin").write_bytes(binary)
    manifest = {
        "source_sha256": sha256(source.encode()).hexdigest(),
        "binary_sha256": sha256(binary).hexdigest(),
        "selected_rtl_revision": RTL_REVISION,
        "timing_provider": evidence["provider"],
        "scheduler": "merlin-static-source-order-serial",
        "words": len(final_rows),
        "scope": evidence["scope"],
        "tools": {
            n: sha256((bins / n).read_bytes()).hexdigest()
            for n in ("atlas-opt", "atlas-emit")
        },
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--atlas-bin-dir", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            compile_fixed(
                args.source.read_text(),
                destination=args.destination,
                atlas_bin_dir=args.atlas_bin_dir,
            )
        )
    )
