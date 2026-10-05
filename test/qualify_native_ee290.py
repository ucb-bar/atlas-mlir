"""Evaluator-only EE290 Verilator check of an already compiled Atlas program.

The generated host ELF embeds public test inputs and expected outputs. It is an
evaluation harness, not the deployable host runtime or a compiler input.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import resource
import struct
import subprocess
from pathlib import Path

from qualify_native_support import CASES, digest, inputs_and_expected, load_artifact


def _words_at(address: int, data: bytes) -> tuple[list[int], list[int]]:
    if address % 4 or len(data) % 4:
        raise ValueError("EE290 diagnostic requires word-aligned buffers")
    return ([address + offset for offset in range(0, len(data), 4)],
            list(struct.unpack("<" + "I" * (len(data) // 4), data)))


def _append_words(addresses: list[int], values: list[int], address: int, data: bytes) -> None:
    more_addresses, more_values = _words_at(address, data)
    addresses.extend(more_addresses)
    values.extend(more_values)


def _allow_generated_initialization_stack() -> None:
    # This large EE290 Verilator image nests generated initializers deeply.
    # Its default 8 MiB process stack segfaults before reading any ELF.
    _, hard = resource.getrlimit(resource.RLIMIT_STACK)
    resource.setrlimit(resource.RLIMIT_STACK, (hard, hard))


def _git_revision(path: Path) -> str:
    run = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                         capture_output=True, text=True, check=True)
    dirty = subprocess.run(["git", "-C", str(path), "status", "--porcelain",
                            "--untracked-files=no"], capture_output=True,
                           text=True, check=True)
    if dirty.stdout.strip():
        raise ValueError(f"tracked source changed after simulator build: {path}")
    return run.stdout.strip()


def _host_source(assembler: Path, words: tuple[int, ...], plan: dict,
                 sources: dict[str, bytes], expected: bytes, case: str, phase: int) -> str:
    spec = importlib.util.spec_from_file_location("atlas_reference_host_emitter", assembler)
    if spec is None or spec.loader is None:
        raise ValueError("cannot load the pinned Atlas bare-metal host emitter")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    preload_addresses: list[int] = []
    preload_values: list[int] = []
    check_addresses: list[int] = []
    check_values: list[int] = []
    regions: list[tuple[int, int]] = []
    for item in plan["inputs"]:
        data = sources[item["source"]]
        if len(data) != item["byte_length"]:
            raise ValueError("input does not match the compiled ABI")
        address = item["byte_address"]
        regions.append((address, address + len(data)))
        _append_words(preload_addresses, preload_values, address, data)
        _append_words(check_addresses, check_values, address, data)

    output = plan["outputs"][0]
    address = output["byte_address"]
    if len(expected) != output["byte_length"]:
        raise ValueError("expected output does not match the compiled ABI")
    regions.append((address, address + len(expected)))
    _append_words(preload_addresses, preload_values, address, b"\xA5" * len(expected))
    _append_words(check_addresses, check_values, address, expected)

    guard_address = 0x90004000
    guard = b"\x5A" * 32
    regions.append((guard_address, guard_address + len(guard)))
    if any(left[0] < right[1] and right[0] < left[1]
           for index, left in enumerate(regions) for right in regions[index + 1:]):
        raise ValueError("compiled ABI and guard regions overlap")
    _append_words(preload_addresses, preload_values, guard_address, guard)
    _append_words(check_addresses, check_values, guard_address, guard)
    return module.emit_c_file(words, test_name=f"merlin_native_{case}_phase{phase}",
                              golden=(preload_addresses, preload_values,
                                      check_addresses, check_values))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--phase", type=int, default=0)
    parser.add_argument("--program", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--ee290-source", type=Path, required=True)
    parser.add_argument("--expected-ee290-revision", required=True)
    parser.add_argument("--host-emitter", type=Path, required=True)
    parser.add_argument("--expected-host-emitter-sha256", required=True)
    parser.add_argument("--cc", type=Path, required=True)
    parser.add_argument("--simulator", type=Path, required=True)
    parser.add_argument("--expected-simulator-sha256", required=True)
    parser.add_argument("--dramsim-ini-dir", type=Path, required=True)
    parser.add_argument("--max-cycles", type=int, default=10_000_000)
    parser.add_argument("--wall-timeout", type=int, default=600)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("output root must be fresh")
    if args.max_cycles <= 0 or args.wall_timeout <= 0:
        parser.error("cycle and wall bounds must be positive")
    artifact = args.program.resolve(strict=True)
    ee290_source = args.ee290_source.resolve(strict=True)
    assembler = args.host_emitter.resolve(strict=True)
    cc = args.cc.resolve(strict=True)
    simulator = args.simulator.resolve(strict=True)
    ini_dir = args.dramsim_ini_dir.resolve(strict=True)
    for ini_name in ("DDR3_micron_64M_8B_x4_sg15.ini", "system.ini"):
        if not (ini_dir / ini_name).is_file():
            parser.error(f"DRAMSim configuration missing: {ini_dir / ini_name}")
    setup_revision = _git_revision(ee290_source)
    if setup_revision != args.expected_ee290_revision:
        parser.error("EE290 setup revision differs from selected source variant")
    atlas_revision = _git_revision(ee290_source / "generators/atlas-npu")
    if atlas_revision != args.source_revision:
        parser.error("Atlas submodule revision differs from compiled target")
    submodule_revisions = {
        name: _git_revision(ee290_source / "generators" / name)
        for name in ("rocket-chip", "shuttle", "tacit")
    }
    if digest(simulator) != args.expected_simulator_sha256:
        parser.error("simulator bytes differ from selected build")
    if digest(assembler) != args.expected_host_emitter_sha256:
        parser.error("host emitter bytes differ from selected source")
    words, plan = load_artifact(args.case, artifact, args.source_revision)
    sources, expected = inputs_and_expected(args.case, args.phase)
    source = _host_source(assembler, words, plan, sources, expected,
                          args.case, args.phase)

    args.out.mkdir(parents=True)
    host_c = args.out / "host_eval.c"
    elf = args.out / "host_eval.riscv"
    stdout = args.out / "simulator.stdout.log"
    stderr = args.out / "simulator.stderr.log"
    host_c.write_text(source)
    compiler = [str(cc), "-std=gnu99", "-O2", "-Wall", "-Wextra",
                "-fno-common", "-fno-builtin-printf", "-march=rv64imafd",
                "-mabi=lp64d", "-mcmodel=medany", "-static",
                "-specs=htif_nano.specs", "-T", "htif.ld", str(host_c),
                "-o", str(elf)]
    compile_run = subprocess.run(compiler, text=True, capture_output=True,
                                 timeout=args.wall_timeout, check=False)
    (args.out / "compiler.stdout.log").write_text(compile_run.stdout)
    (args.out / "compiler.stderr.log").write_text(compile_run.stderr)
    if compile_run.returncode:
        raise RuntimeError(f"host compiler failed ({compile_run.returncode})")

    command = [str(simulator), "+permissive", "+dramsim",
               f"+dramsim_ini_dir={ini_dir}", f"+max-cycles={args.max_cycles}",
               f"+loadmem={elf}", "+permissive-off", str(elf)]
    timed_out = False
    try:
        run = subprocess.run(command, text=True, capture_output=True,
                             timeout=args.wall_timeout, check=False,
                             preexec_fn=_allow_generated_initialization_stack)
        stdout.write_text(run.stdout)
        stderr.write_text(run.stderr)
        returncode = run.returncode
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        stdout.write_bytes(exc.stdout or b"")
        stderr.write_bytes(exc.stderr or b"")
        returncode = None
    passed = not timed_out and returncode == 0 and "*** PASSED ***" in stdout.read_text()
    receipt = {
        "schema": "atlas.native_ee290_bounded_diagnostic.v1",
        "status": "passed" if passed else "failed",
        "scope": "diagnostic EE290 source variant; evaluator-embedded public tensors; no deployed host runtime",
        "case": args.case, "phase": args.phase,
        "selected_atlas_source_revision": args.source_revision,
        "ee290_setup_revision": setup_revision,
        "ee290_submodule_revisions": submodule_revisions,
        "checker_sha256": digest(Path(__file__)),
        "program_sha256": plan["program"]["sha256"],
        "compiler_manifest_sha256": digest(artifact / "manifest.json"),
        "host_emitter_sha256": digest(assembler),
        "host_cc_sha256": digest(cc),
        "host_source_sha256": digest(host_c), "host_elf_sha256": digest(elf),
        "simulator_sha256": digest(simulator),
        "input_sha256": {name: hashlib.sha256(data).hexdigest()
                         for name, data in sources.items()},
        "expected_sha256": hashlib.sha256(expected).hexdigest(),
        "cycle_limit": args.max_cycles, "wall_timeout_s": args.wall_timeout,
        "simulator_stack_soft_limit": "raised_to_hard_limit",
        "simulator_exit_code": returncode, "timed_out": timed_out,
        "simulator_stdout_sha256": digest(stdout),
        "simulator_stderr_sha256": digest(stderr),
    }
    (args.out / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": receipt["status"], "receipt": str(args.out / "receipt.json")},
                     sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
