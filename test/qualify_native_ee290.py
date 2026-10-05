"""Evaluator-only EE290 Verilator check of an already compiled Atlas program.

The historical host mode embeds public expected outputs. The input-only driver
mode keeps expected outputs in this evaluator. Neither is a deployed runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
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


def _c_array(values: list[int], name: str, *, wide: bool = False) -> str:
    kind = "uint64_t" if wide else "uint32_t"
    suffix = "ULL" if wide else "U"
    digits = 16 if wide else 8
    body = ",\n".join(f"  0x{value:0{digits}x}{suffix}" for value in values)
    return f"static const {kind} {name}[] = {{\n{body}\n}};\n"


def _driver_host_source(words: tuple[int, ...], plan: dict,
                        sources: dict[str, bytes]) -> str:
    """Render an input-only host; expected output stays in this evaluator process."""
    pre_addresses: list[int] = []
    pre_values: list[int] = []
    observed_addresses: list[int] = []
    regions: list[tuple[int, int]] = []
    for item in plan["inputs"]:
        data = sources[item["source"]]
        if len(data) != item["byte_length"]:
            raise ValueError("input does not match the compiled ABI")
        address = item["byte_address"]
        regions.append((address, address + len(data)))
        _append_words(pre_addresses, pre_values, address, data)
        addresses, _ = _words_at(address, data)
        observed_addresses.extend(addresses)
    input_word_count = len(observed_addresses)
    if len(plan["outputs"]) != 1:
        raise ValueError("bounded EE290 host requires one output")
    output = plan["outputs"][0]
    address = output["byte_address"]
    length = output["byte_length"]
    regions.append((address, address + length))
    _append_words(pre_addresses, pre_values, address, b"\xa5" * length)
    observed_addresses.extend(_words_at(address, b"\0" * length)[0])
    output_word_count = length // 4
    guard_address = 0x90004000
    guard = b"\x5a" * 32
    regions.append((guard_address, guard_address + len(guard)))
    _append_words(pre_addresses, pre_values, guard_address, guard)
    addresses, _ = _words_at(guard_address, guard)
    observed_addresses.extend(addresses)
    if any(left[0] < right[1] and right[0] < left[1]
           for index, left in enumerate(regions) for right in regions[index + 1:]):
        raise ValueError("compiled ABI and guard regions overlap")
    if len(set(observed_addresses)) != len(observed_addresses):
        raise ValueError("duplicate observed host address")
    text = ("#include <inttypes.h>\n#include <stdint.h>\n#include <stdio.h>\n"
            "#include \"atlas_host.h\"\n"
            + _c_array(list(words), "atlas_program")
            + _c_array(pre_addresses, "preload_addresses", wide=True)
            + _c_array(pre_values, "preload_words")
            + _c_array(observed_addresses, "observed_addresses", wide=True)
            + "static void emit_digest(unsigned group, size_t begin, size_t end) {\n"
            + "  uint64_t first = 0xcbf29ce484222325ULL;\n"
            + "  uint64_t second = 0x9e3779b97f4a7c15ULL;\n"
            + "  for (size_t i = begin; i < end; ++i) {\n"
            + "    uint64_t address = observed_addresses[i];\n"
            + "    uint32_t value = *(volatile uint32_t *)(uintptr_t)address;\n"
            + "    uint64_t mixed = address ^ ((uint64_t)value << 32) ^ value;\n"
            + "    first = (first ^ mixed) * 0x100000001b3ULL;\n"
            + "    second = (second + mixed) * 0x9ddfea08eb382d69ULL;\n"
            + "    second ^= second >> 32;\n"
            + "  }\n"
            + "  printf(\"ATLAS_DIGEST %u %u %016\" PRIx64 \" %016\" PRIx64 \"\\n\",\n"
            + "         group, (unsigned)(end - begin), first, second);\n"
            + "}\n"
            + "int main(void) {\n"
            + f"  for (size_t i = 0; i < {len(pre_addresses)}u; ++i) {{\n"
            + "    *(volatile uint32_t *)(uintptr_t)preload_addresses[i] = preload_words[i];\n"
            + "  }\n  __asm__ volatile(\"fence\" ::: \"memory\");\n"
            + "  atlas_ee290_result result;\n"
            + f"  atlas_ee290_status status = atlas_ee290_run(atlas_program, {len(words)}u, "
              "200000000u, &result);\n"
            + "  printf(\"ATLAS_RESULT %u %u %u %u %u %u %u\\n\", (unsigned)status,\n"
            + "         result.completion_marker, result.cycles, result.instructions,\n"
            + "         result.status, result.illegal_pc, result.polls);\n"
            + "  if (status != ATLAS_EE290_OK) return (int)status;\n"
            + f"  emit_digest(0u, 0u, {input_word_count}u);\n"
            + f"  emit_digest(1u, {input_word_count}u, "
              f"{input_word_count + output_word_count}u);\n"
            + f"  emit_digest(2u, {input_word_count + output_word_count}u, "
              f"{len(observed_addresses)}u);\n"
            + "  return 0;\n}\n")
    return text


def _digest_words(addresses: list[int], values: list[int]) -> tuple[str, str]:
    mask = (1 << 64) - 1
    first = 0xCBF29CE484222325
    second = 0x9E3779B97F4A7C15
    for address, value in zip(addresses, values, strict=True):
        mixed = address ^ (value << 32) ^ value
        first = ((first ^ mixed) * 0x100000001B3) & mask
        second = ((second + mixed) * 0x9DDFEA08EB382D69) & mask
        second ^= second >> 32
    return f"{first:016x}", f"{second:016x}"


def _check_driver_output(stdout: str, plan: dict, sources: dict[str, bytes],
                         expected: bytes) -> dict:
    header = re.findall(r"^ATLAS_RESULT (\d+) (\d+) (\d+) (\d+) (\d+) (\d+) (\d+)$",
                        stdout, re.MULTILINE)
    if len(header) != 1:
        return {"passed": False, "reason": "missing or duplicate Atlas result"}
    status, marker, cycles, instructions, hw_status, illegal_pc, polls = map(int, header[0])
    output = plan["outputs"][0]
    if len(expected) != output["byte_length"]:
        raise ValueError("expected output does not match compiled ABI")
    groups: list[tuple[list[int], list[int]]] = []
    input_addresses: list[int] = []
    input_values: list[int] = []
    for item in plan["inputs"]:
        addresses, values = _words_at(item["byte_address"], sources[item["source"]])
        input_addresses.extend(addresses)
        input_values.extend(values)
    groups.append((input_addresses, input_values))
    groups.append(_words_at(output["byte_address"], expected))
    groups.append(_words_at(0x90004000, b"\x5a" * 32))
    observed_rows = re.findall(r"^ATLAS_DIGEST ([0-2]) (\d+) ([0-9a-f]{16}) ([0-9a-f]{16})$",
                               stdout, re.MULTILINE)
    observed = {int(group): (int(count), first, second)
                for group, count, first, second in observed_rows}
    matches = (len(observed_rows) == 3 and set(observed) == {0, 1, 2} and
               all(observed[index] == (len(addresses), *_digest_words(addresses, values))
                   for index, (addresses, values) in enumerate(groups)))
    passed = status == 0 and marker == 1 and matches
    return {"passed": passed, "checked_words": sum(len(row[0]) for row in groups),
            "observed_digest_groups": len(observed_rows), "digest_matches": matches,
            "validation": "two_64_bit_digests_per_region_bounded_validation",
            "driver_status": status, "completion_marker": marker,
            "cycles": cycles, "instructions": instructions,
            "hardware_status": hw_status, "illegal_pc": illegal_pc,
            "polls": polls}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--phase", type=int, default=0)
    parser.add_argument("--program", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--ee290-source", type=Path, required=True)
    parser.add_argument("--expected-ee290-revision", required=True)
    parser.add_argument("--host-emitter", type=Path)
    parser.add_argument("--expected-host-emitter-sha256")
    parser.add_argument("--host-driver", type=Path)
    parser.add_argument("--expected-host-driver-sha256")
    parser.add_argument("--expected-host-driver-header-sha256")
    parser.add_argument("--host-link-script", type=Path)
    parser.add_argument("--host-specs", type=Path)
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
    if args.host_driver is None and (args.host_emitter is None or
                                     args.expected_host_emitter_sha256 is None):
        parser.error("reference host mode requires pinned --host-emitter")
    if args.host_driver is not None and (args.expected_host_driver_sha256 is None or
                                         args.expected_host_driver_header_sha256 is None):
        parser.error("driver host mode requires pinned C source and header")
    if args.host_driver is not None and (args.host_link_script is None or args.host_specs is None):
        parser.error("driver host mode requires selected link script and specs")
    assembler = args.host_emitter.resolve(strict=True) if args.host_emitter else None
    driver = args.host_driver.resolve(strict=True) if args.host_driver else None
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
    if assembler is not None and digest(assembler) != args.expected_host_emitter_sha256:
        parser.error("host emitter bytes differ from selected source")
    if driver is not None and digest(driver) != args.expected_host_driver_sha256:
        parser.error("host driver bytes differ from selected source")
    if driver is not None and digest(driver.with_suffix(".h")) != args.expected_host_driver_header_sha256:
        parser.error("host driver header differs from selected source")
    words, plan = load_artifact(args.case, artifact, args.source_revision)
    sources, expected = inputs_and_expected(args.case, args.phase)
    if driver is None:
        source = _host_source(assembler, words, plan, sources, expected,
                              args.case, args.phase)
    else:
        source = _driver_host_source(words, plan, sources)

    args.out.mkdir(parents=True)
    host_c = args.out / "host_eval.c"
    elf = args.out / "host_eval.riscv"
    stdout = args.out / "simulator.stdout.log"
    stderr = args.out / "simulator.stderr.log"
    host_c.write_text(source)
    if driver is None:
        compiler = [str(cc), "-std=gnu99", "-O2", "-Wall", "-Wextra",
                    "-fno-common", "-fno-builtin-printf", "-march=rv64imafd",
                    "-mabi=lp64d", "-mcmodel=medany", "-static",
                    "-specs=htif_nano.specs", "-T", "htif.ld", str(host_c),
                    "-o", str(elf)]
        link_script = specs = None
        binding_path = None
        support_version = None
    else:
        try:
            from importlib.metadata import version

            from atlas_native_support.host_build import ee290_baremetal_recipe
        except ImportError as exc:
            parser.error(f"install the selected Atlas OOT host binding: {exc}")
        binding_path = Path(ee290_baremetal_recipe.__code__.co_filename).resolve(strict=True)
        support_version = version("atlas-native-support")
        link_script = args.host_link_script.resolve(strict=True)
        specs = args.host_specs.resolve(strict=True)
        recipe = ee290_baremetal_recipe(compiler=cc, link_script=link_script,
                                        specs=specs, driver_source=driver)
        compiler = recipe.command(sources=(host_c,), output=elf)
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
    driver_check = (_check_driver_output(stdout.read_text(), plan, sources, expected)
                    if driver is not None else None)
    passed = (not timed_out and returncode == 0 and
              (driver_check["passed"] if driver_check is not None
               else "*** PASSED ***" in stdout.read_text()))
    receipt = {
        "schema": "atlas.native_ee290_bounded_diagnostic.v1",
        "status": "passed" if passed else "failed",
        "scope": ("diagnostic EE290 source variant; input-only host and external digest check"
                  if driver is not None else
                  "diagnostic EE290 source variant; evaluator-embedded public tensors"),
        "case": args.case, "phase": args.phase,
        "selected_atlas_source_revision": args.source_revision,
        "ee290_setup_revision": setup_revision,
        "ee290_submodule_revisions": submodule_revisions,
        "checker_sha256": digest(Path(__file__)),
        "program_sha256": plan["program"]["sha256"],
        "compiler_manifest_sha256": digest(artifact / "manifest.json"),
        "host_mode": "input_only_driver" if driver is not None else "upstream_evaluator",
        "goldens_embedded_in_host_elf": driver is None,
        "host_emitter_sha256": digest(assembler) if assembler is not None else None,
        "host_driver_sha256": digest(driver) if driver is not None else None,
        "host_driver_header_sha256": digest(driver.with_suffix(".h")) if driver is not None else None,
        "host_build_binding_sha256": digest(binding_path) if binding_path is not None else None,
        "host_support_package_version": support_version,
        "driver_check": driver_check,
        "host_cc_sha256": digest(cc),
        "host_link_script_sha256": digest(link_script) if link_script is not None else None,
        "host_specs_sha256": digest(specs) if specs is not None else None,
        "shared_build_recipe": "merlin.HarnessBuildRecipe" if driver is not None else None,
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
