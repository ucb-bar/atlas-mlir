#!/usr/bin/env python3
"""Build selected AtlasCore sources and run the pinned generated-tile RTL test."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

PIN = "0079c0541111197741a231c002e3843fa6f545b2"
PROGRAM_SHA256 = "848c70a4aa373e89a50c3a2984f753e9b470e6a21d7f5c17b34fec45badda95c"
MXU1_PROGRAM_SHA256 = "9df497b39c02f9e6a153b9480231886408f074bf7829cd8effb712752d43622d"
K64_PROGRAM_SHA256 = "d365b77beb3bded04fca491eed614dad8105b5706768ec9463905c84ef0f32d8"
CASE_NAMES = ("zeros", "unit_cell", "ones", "ordered_ties", "tail_a",
              "tail_b", "tail_c", "tail_d", "tail_e", "tail_f",
              "mxu1_zeros", "mxu1_unit_cell", "mxu1_ones", "mxu1_rows",
              "k64_first", "k64_second", "k64_both",
              "k64_boundary_ties", "k64_ones")
ROOT = Path(__file__).resolve().parents[2]
TEST_SOURCE = ROOT / "test/rtl/CompiledProgramAtlasCoreDiagnosticTest.scala"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def classpath_file(path: Path) -> list[Path]:
    entries = [Path(value) for value in path.read_text().strip().split(os.pathsep)]
    if not entries or not all(entry.is_file() for entry in entries):
        raise ValueError(f"classpath contains a missing file: {path}")
    return entries


def one_jar(entries: list[Path], prefix: str) -> Path:
    found = [entry for entry in entries if entry.name.startswith(prefix)]
    if len(found) != 1:
        raise ValueError(f"expected one {prefix} jar, found {len(found)}")
    return found[0]


def fixture_identity(directory: Path) -> tuple[str, int, int, int, str]:
    manifest = directory / "manifest.properties"
    if not manifest.is_file():
        raise ValueError(f"missing fixture manifest: {manifest}")
    properties = dict(line.split("=", 1) for line in manifest.read_text().splitlines()
                      if line and not line.startswith("#") and "=" in line)
    if properties.get("schema") != "atlas.rtl_tile_fixture.v1":
        raise ValueError("unsupported RTL fixture schema")
    name = properties.get("name", "")
    if re.fullmatch(r"[a-z][a-z0-9_]*", name) is None or name in CASE_NAMES:
        raise ValueError("fixture name is invalid or duplicates a built-in case")
    count = int(properties.get("input_count", "0"))
    if count < 1 or count > 16:
        raise ValueError("fixture input count must be between 1 and 16")
    files = [manifest, directory / "program.bin", directory / "expected.bin"]
    files += [directory / f"input-{index}.bin" for index in range(count)]
    if not all(path.is_file() for path in files):
        raise ValueError("fixture is missing a program, input, or expected-output file")
    if digest(directory / "program.bin") != properties.get("program_sha256"):
        raise ValueError("fixture program SHA-256 does not match its manifest")
    input_lengths = [(directory / f"input-{index}.bin").stat().st_size
                     for index in range(count)]
    expected_bytes = (directory / "expected.bin").stat().st_size
    output_count = int(properties.get("output_count", "1"))
    if (not 1 <= output_count <= 16 or expected_bytes != 2048 * output_count or any(
        length not in (1024, 2048) for length in input_lengths
    )):
        raise ValueError("fixture input/output byte count is invalid")
    addresses = [int(properties[f"input.{index}.address"], 0)
                 for index in range(count)]
    output_address = int(properties["output.address"], 0)
    if (any(address <= 0 or address % 32 for address in addresses)
            or output_address <= 0 or output_address % 32):
        raise ValueError("fixture addresses must be positive and 32-byte aligned")
    ranges = [(address, address + length)
              for address, length in zip(addresses, input_lengths, strict=True)]
    ranges.append((output_address, output_address + expected_bytes))
    if any(left[0] < right[1] and right[0] < left[1]
           for index, left in enumerate(ranges)
           for right in ranges[index + 1:]):
        raise ValueError("fixture input and output ranges overlap")
    combined = hashlib.sha256()
    for path in files:
        combined.update(path.name.encode() + b"\0" + bytes.fromhex(digest(path)))
    return name, count, sum(input_lengths) // 32, expected_bytes // 2, combined.hexdigest()


def run_logged(command: list[str], *, log: Path, env: dict[str, str]) -> float:
    start = time.monotonic()
    with log.open("w") as output:
        result = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT,
                                env=env, check=False)
    if result.returncode:
        raise RuntimeError(f"command exited {result.returncode}; see {log.name}")
    return round(time.monotonic() - start, 3)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rtl-root", type=Path, required=True)
    parser.add_argument("--dependency-classpath-file", type=Path, required=True)
    parser.add_argument("--scalatest-classpath-file", type=Path, required=True)
    parser.add_argument("--scala-compiler", type=Path, required=True)
    parser.add_argument("--chisel-plugin", type=Path, required=True)
    parser.add_argument("--verilator", type=Path, required=True)
    parser.add_argument("--program", type=Path, required=True)
    parser.add_argument("--mxu1-program", type=Path, required=True)
    parser.add_argument("--k64-program", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, action="append", default=[])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args(argv)

    if args.out.exists():
        parser.error(f"output directory must be fresh: {args.out}")
    if not 1 <= args.jobs <= 16:
        parser.error("--jobs must be between 1 and 16")
    for path in (args.rtl_root, args.scala_compiler, args.chisel_plugin,
                 args.verilator, args.program, args.mxu1_program,
                 args.k64_program, TEST_SOURCE):
        if not path.exists():
            parser.error(f"missing input: {path}")
    revision = subprocess.run(
        ["git", "-C", str(args.rtl_root), "rev-parse", "HEAD"],
        text=True, capture_output=True, check=True).stdout.strip()
    if revision != PIN:
        parser.error(f"selected RTL revision must be {PIN}; got {revision}")
    changed_sources = subprocess.run(
        ["git", "-C", str(args.rtl_root), "diff", "--name-only", "HEAD", "--",
         "src/main/scala", "dependencies/fpex", "dependencies/sp26-fp-units"],
        text=True, capture_output=True, check=True).stdout.strip()
    if changed_sources:
        parser.error(f"selected RTL sources have tracked changes: {changed_sources}")
    if digest(args.program) != PROGRAM_SHA256:
        parser.error("program SHA-256 does not match the pinned generated tile")
    if digest(args.mxu1_program) != MXU1_PROGRAM_SHA256:
        parser.error("MXU1 program SHA-256 does not match the pinned generated tile")
    if digest(args.k64_program) != K64_PROGRAM_SHA256:
        parser.error("K64 program SHA-256 does not match the pinned generated tile")
    try:
        fixtures = [fixture_identity(path) for path in args.fixture]
    except (KeyError, OSError, ValueError) as error:
        parser.error(str(error))
    if len({item[0] for item in fixtures}) != len(fixtures):
        parser.error("fixture names must be unique")

    dependencies = classpath_file(args.dependency_classpath_file)
    scalatest = classpath_file(args.scalatest_classpath_file)
    scala_lib = one_jar(dependencies, "scala-library-2.13.16")
    scala_reflect = one_jar(dependencies, "scala-reflect-2.13.16")
    roots = (
        args.rtl_root / "src/main/scala",
        args.rtl_root / "dependencies/fpex/src/main/scala",
        args.rtl_root / "dependencies/sp26-fp-units/src/main/scala",
    )
    sources = sorted(path for root in roots for path in root.rglob("*.scala"))
    if len(sources) != 131:
        parser.error(f"expected 131 selected Scala sources; got {len(sources)}")

    out = args.out.resolve()
    out.mkdir(parents=True)
    for name in ("selected-classes", "test-classes", "tmp", "bin", "observed"):
        (out / name).mkdir()
    wrapper = out / "bin/verilator"
    wrapper.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        "args=()\nseen_jobs=false\nreplace_next=false\n"
        "for arg in \"$@\"; do\n"
        "  if $replace_next; then arg=" + str(args.jobs) + "; replace_next=false; fi\n"
        "  if [[ \"$arg\" == -j ]]; then seen_jobs=true; replace_next=true; fi\n"
        "  args+=(\"$arg\")\ndone\n"
        "if ! $seen_jobs; then args+=(-j " + str(args.jobs) + "); fi\n"
        "exec " + shlex.quote(str(args.verilator.resolve())) + " \"${args[@]}\"\n"
    )
    wrapper.chmod(0o755)
    env = os.environ.copy()
    env["TMPDIR"] = str(out / "tmp")
    env["PATH"] = str(out / "bin") + os.pathsep + env.get("PATH", "")
    env["ATLAS_DMA_DECODE_PROGRAM"] = str(args.program.resolve())
    env["ATLAS_MXU1_PROGRAM"] = str(args.mxu1_program.resolve())
    env["ATLAS_K64_PROGRAM"] = str(args.k64_program.resolve())
    if args.fixture:
        env["ATLAS_EXTRA_FIXTURES"] = os.pathsep.join(
            str(path.resolve()) for path in args.fixture)
    env["ATLAS_CORE_TEST_WORKSPACE"] = str(out / "rtl-workspace")
    env["ATLAS_RTL_OUTPUT_DIR"] = str(out / "observed")

    compiler_cp = os.pathsep.join(map(str, (args.scala_compiler, scala_lib,
                                            scala_reflect)))
    selected_cp = os.pathsep.join(map(str, dependencies))
    test_cp = os.pathsep.join((str(out / "selected-classes"), selected_cp,
                               *map(str, scalatest)))
    report: dict[str, object] = {
        "schema": "atlas.selected_core_test_run.v1",
        "status": "failed",
        "rtl_revision": revision,
        "source_count": len(sources),
        "program_sha256": PROGRAM_SHA256,
        "mxu1_program_sha256": MXU1_PROGRAM_SHA256,
        "k64_program_sha256": K64_PROGRAM_SHA256,
        "test_sha256": digest(TEST_SOURCE),
        "chisel_plugin_sha256": digest(args.chisel_plugin),
        "dependency_classpath_sha256": digest(args.dependency_classpath_file),
        "scalatest_classpath_sha256": digest(args.scalatest_classpath_file),
        "jobs": args.jobs,
        "scope": "nineteen built-in AtlasCore launches using MXU0, MXU1 and K64 images plus independently prepared tile fixtures; test-side TileLink DRAM with bounded stalls",
    }
    if fixtures:
        report["fixtures"] = [{"name": name, "input_count": count,
                               "read_beats": read_beats, "bf16_cells": cells,
                               "sha256": sha,
                               "program_sha256": digest(directory / "program.bin")}
                              for (name, count, read_beats, cells, sha), directory
                              in zip(fixtures, args.fixture, strict=True)]
    try:
        report["selected_build_seconds"] = run_logged([
            "java", "-Xmx8g", "-cp", compiler_cp, "scala.tools.nsc.Main",
            "-classpath", selected_cp,
            "-Xplugin:" + str(args.chisel_plugin),
            "-d", str(out / "selected-classes"), *map(str, sources),
        ], log=out / "selected-scalac.log", env=env)
        report["test_build_seconds"] = run_logged([
            "java", "-Xmx4g", "-cp", compiler_cp, "scala.tools.nsc.Main",
            "-classpath", test_cp, "-d", str(out / "test-classes"),
            str(TEST_SOURCE),
        ], log=out / "test-scalac.log", env=env)
        report["rtl_test_seconds"] = run_logged([
            "java", "-Xmx8g", "-XX:ActiveProcessorCount=" + str(args.jobs),
            "-Djava.io.tmpdir=" + str(out / "tmp"),
            "-cp", str(out / "test-classes") + os.pathsep + test_cp,
            "org.scalatest.tools.Runner", "-o", "-s",
            "atlas.tile.CompiledProgramAtlasCoreDiagnosticTest",
        ], log=out / "rtl-test.log", env=env)
        log = re.sub(r"\x1b\[[0-9;]*m", "", (out / "rtl-test.log").read_text())
        cases = re.findall(
            r"Selected AtlasCore ([a-z0-9_]+): (\d+) DMA beats, (\d+) BF16 cells, (\d+) cycles",
            log,
        )
        expected_names = CASE_NAMES + tuple(item[0] for item in fixtures)
        fixture_beats = {name: read_beats + cells // 16
                         for name, _, read_beats, cells, _ in fixtures}
        fixture_cells = {name: cells for name, _, _, cells, _ in fixtures}
        if tuple(name for name, _, _, _ in cases) != expected_names or \
                any(int(beats) != fixture_beats.get(
                    name, 192 if name.startswith("k64_") else 128)
                    or int(cells) != fixture_cells.get(name, 1024)
                    for name, beats, cells, _ in cases) or \
                "Selected AtlasCore tail_grid: 6 launches, 2145 BF16 logical cells" not in log or \
                "Tests: succeeded 1, failed 0" not in log:
            raise RuntimeError("RTL run did not report all expected passing launches")
        report["cases"] = [{"name": name, "dma_beats": int(beats),
                            "bf16_cells": int(cells), "cycles": int(cycles)}
                           for name, beats, cells, cycles in cases]
        report["observed_outputs"] = [
            {"name": name, "sha256": digest(out / "observed" / f"{name}.bin")}
            for name, _, _, _, _ in fixtures
        ]
        report["rtl_test_log_sha256"] = digest(out / "rtl-test.log")
        report["status"] = "passed_bounded_core_test"
    except (OSError, RuntimeError) as error:
        report["error"] = str(error)
    (out / "result.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"], "result": str(out / "result.json")}))
    return 0 if report["status"] == "passed_bounded_core_test" else 1


if __name__ == "__main__":
    sys.exit(main())
