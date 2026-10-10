#!/usr/bin/env python3
"""Run a scheduled VLS program on the integrated EE290SimConfig VCS simulator (issue #11 witness).

atlas-opt selects the `merlin.op_timing.v1` facts and schedules the delay-free program, atlas-emit encodes
it, a RISC-V host (test/ee290-vls-host.c) loads it over the bus, and `simv +loadmem` reports three data panels.
Observer cycle counts include host polling and are not kernel timings.
"""
import argparse
import hashlib
import re
import subprocess
import sys
from pathlib import Path

HOST = Path(__file__).resolve().parents[1] / "test/ee290-vls-host.c"
FAILURE = re.compile(r"EE290_VLS_FAILED|EE290_VLS_MISMATCH|assertion failed|\$fatal|fatal:|error:|error-", re.I)


def commands(args, out, digest):
    """The four commands in order: schedule, emit, host build, simulate (the emitted words go in atlas_program.inc)."""
    select = f"--select-atlas-rtl-evidence=op-timing={args.facts.resolve()} op-timing-sha256={digest}" + (" dma=wait" if args.dma_wait else "")
    pass_name = "--insert-atlas-delays" if args.schedule == "delay" else "--schedule-atlas-stream"
    host = out / "host.riscv"
    return [
        [str(args.atlas_opt), str(args.program), select, pass_name, "--verify-atlas-rtl-timing", "-o", str(out / "final.mlir")],
        [str(args.atlas_emit), str(out / "final.mlir")],
        [str(args.host_cc), "-std=gnu99", "-O2", "-Wall", "-Wextra", "-Werror", "-fno-common", "-fno-builtin-printf", "-march=rv64imafd", "-mabi=lp64d",
         "-mcmodel=medany", "-specs=htif_nano.specs", "-static", "-T", "htif.ld", f"-I{out}", str(HOST), "-o", str(host)],
        [str(args.simulator), "+permissive", f"+max-cycles={args.max_cycles}", f"+loadmem={host}", "+ntb_random_seed=1", *args.sim_arg, "+permissive-off", str(host)],
    ]


def include_file(words):
    return f"#define ATLAS_PROGRAM_WORDS {len(words)}U\nstatic const uint32_t atlas_program[] = {{\n" + "".join(f"  0x{w}U,\n" for w in words) + "};\n"


def judge(returncode, log):
    """Return 'passed', 'license_unavailable' or 'failed' from the simulator exit status and log."""
    if (returncode == 0 and log.count("EE290_VLS_PANEL_PASSED panel=") == 3 and "EE290_VLS_PASSED panels=3" in log and not FAILURE.search(log)):
        return "passed"
    return "license_unavailable" if re.search(r"queuing for license|license checkout failed|server node is down", log, re.I) else "failed"


def run(argv, log, timeout):
    result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    log.write_text(result.stdout + result.stderr)
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("program", "facts", "atlas-opt", "atlas-emit", "simulator", "host-cc", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--schedule", choices=("delay", "schedule"), default="schedule", help="atlas-opt consumer (default: schedule)")
    p.add_argument("--dma-wait", action="store_true", help="select dma=wait (not needed for the VLS-only example)")
    p.add_argument("--max-cycles", type=int, default=2000000)
    p.add_argument("--timeout-seconds", type=float, default=300)
    p.add_argument("--sim-arg", action="append", default=[])
    args = p.parse_args(argv)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    opt, emit, build, simulate = commands(args, out, hashlib.sha256(args.facts.read_bytes()).hexdigest())
    for step, command in (("schedule", opt), ("emit", emit)):
        result = run(command, out / f"{step}.log", args.timeout_seconds)
        if result.returncode:
            print(f"EE290 witness: {step} failed; see {out / (step + '.log')}", file=sys.stderr)
            return 2
    words = (out / "emit.log").read_text().split()
    if not words or len(words) > 32768 or any(not re.fullmatch(r"[0-9a-fA-F]{8}", w) for w in words):
        print("EE290 witness: atlas-emit output is not an instruction word stream", file=sys.stderr)
        return 2
    (out / "atlas_program.inc").write_text(include_file(words))
    if run(build, out / "host-build.log", args.timeout_seconds).returncode:
        print(f"EE290 witness: host build failed; see {out / 'host-build.log'}", file=sys.stderr)
        return 2
    result = run(simulate, out / "simulation.log", args.timeout_seconds)
    state = judge(result.returncode, (out / "simulation.log").read_text(errors="replace"))
    print(f"{state}: {out / 'simulation.log'}")
    return 0 if state == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
