"""Compile objects, or measure existing objects/ELFs. Never links an ELF."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

from ..soc import carrier_clock
from .sections import allowed, measure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sizing = sub.add_parser("measure")
    sizing.add_argument("inputs", nargs="+")
    sizing.add_argument("-o", required=True)
    compile_ = sub.add_parser("compile")
    compile_.add_argument("sources", nargs="+")
    compile_.add_argument("--cc", required=True)
    compile_.add_argument("-I", dest="includes", action="append", default=[])
    compile_.add_argument("--clock-mhz", type=int,
                          help="default: the clock of the declared on-chip carriers")
    compile_.add_argument("--assembler-root")
    compile_.add_argument("-o", required=True)
    args = parser.parse_args()
    if args.command == "measure":
        receipt = {"measurements": [measure(p) for p in args.inputs]}
        allowed(args.o).write_text(json.dumps(receipt, indent=2) + "\n")
        return
    if args.clock_mhz is None:
        clock_hz = carrier_clock("onchip")[0]
        if clock_hz % 1000000:
            parser.error(f"the on-chip carriers' {clock_hz} Hz is not whole MHz; pass --clock-mhz")
        args.clock_mhz = clock_hz // 1000000
    if not 1 <= args.clock_mhz <= 1000:
        parser.error("clock must be an integer in [1, 1000] MHz")
    out = allowed(args.o)
    out.mkdir(parents=True, exist_ok=False)
    cc = allowed(args.cc)
    includes = ["-I" + str(allowed(p)) for p in args.includes]
    flags = ["-O3", "-DNDEBUG", "-O1", "-Wall", "-Wextra",
             "-Wno-error=coverage-mismatch", "-march=rv64imafd", "-mabi=lp64d",
             "-mcmodel=medany", "-specs=nosys.specs", "-Os",
             "-ffunction-sections", "-fdata-sections", "-fno-fast-math",
             "-ffp-contract=off", "-frounding-math",
             f"-DSYS_CLK_FREQ={args.clock_mhz * 1000000}",
             f"-DMTIME_FREQ={args.clock_mhz * 1000}"]
    if args.assembler_root:
        flags.append("-Wa,-I," + str(allowed(args.assembler_root)))
    commands, objects, sources = [], [], []
    for index, source in enumerate(args.sources):
        source = allowed(source)
        sources.append({"path": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest()})
        obj = out / f"{index:03d}-{source.name}.o"
        command = ["nice", "-n", str(max(0, 10 - os.getpriority(os.PRIO_PROCESS, 0))),
                   str(cc), *flags, *includes, "-c", str(source), "-o", str(obj)]
        with (out / f"{index:03d}.log").open("w") as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT,
                           env={**os.environ, "OMP_NUM_THREADS": "1"})
        commands.append(command)
        objects.append(measure(obj))
    (out / "sections.json").write_text(json.dumps(
        {"commands": commands, "sources": sources, "clock_mhz": args.clock_mhz,
         "measurements": objects}, indent=2) + "\n")


if __name__ == "__main__":
    main()
