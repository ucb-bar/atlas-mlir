#!/usr/bin/env python3
"""Prepare, then explicitly run, a captured EE290SimConfig VCS compilation.

This deliberately accepts the selected local VCS recipe, not arbitrary shell or
build-system commands. Preparation runs only C++ dependency discovery. Licensed
compilation requires --run and the independently reviewed plan's SHA256.
"""
from __future__ import annotations

import argparse
from collections import deque
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import stat
import subprocess
import sys


def bootstrap(path):
    """Guard the shared checker before importing it, including symlink targets."""
    path = Path(path)
    denied = lambda p: any(x in part.lower() for part in p.parts
                           for x in ("ham" + "mer", "vl" + "si"))
    if denied(path):
        raise ValueError("restricted dependency path")
    if not path.is_absolute():
        path = Path.cwd() / path
    pending, current, links = deque(path.parts[1:]), Path(path.anchor), set()
    while pending:
        part = pending.popleft()
        if part == ".":
            continue
        if part == "..":
            current = current.parent
            continue
        candidate = current / part
        if denied(candidate):
            raise ValueError("restricted dependency path")
        if stat.S_ISLNK(candidate.lstat().st_mode):
            if candidate in links or len(links) >= 64:
                raise ValueError("recursive dependency symlink")
            links.add(candidate)
            target = Path(os.readlink(candidate))
            if denied(target):
                raise ValueError("restricted dependency symlink target")
            if target.is_absolute():
                current = Path(target.anchor)
                pending.extendleft(reversed(target.parts[1:]))
            else:
                pending.extendleft(reversed(target.parts))
        else:
            current = candidate
    return current


_CAPTURE_PATH = bootstrap(Path(__file__).parent / "ee290_build_capture.py")
_SPEC = importlib.util.spec_from_file_location("atlas_simulator_capture", _CAPTURE_PATH)
_CAPTURE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_CAPTURE)
_CHECK = _CAPTURE._CHECK
BuildError = _CHECK.ProvenanceError
SOURCE_SUFFIXES = {".sv", ".v", ".cc", ".cpp"}
CPP_SUFFIXES = {".cc", ".cpp"}
SIMPLE_FLAGS = {"-full64", "-notice", "-line", "-quiet", "-q", "+rad",
                "+vcs+lic+wait", "+vc+list", "-sverilog", "+v2k", "-debug_pp"}
SELECTED_FLAGS = {"+lint=all,noVCDE,noONGS,noUI", "-error=PCWM-L", "-error=noZMMCM",
                  "+systemverilogext+.sv+.svi+.svh+.svt", "+libext+.sv", "+libext+.v",
                  "+verilog2001ext+.v95+.vt+.vp"}
SELECTED_DEFINES = {"+define+CLOCK_PERIOD=1.0", "+define+RESET_DELAY=777.7",
                    "+define+PRINTF_COND=TestDriver.printf_cond", "+define+STOP_COND=!TestDriver.reset",
                    "+define+MODEL=TestHarness", "+define+RANDOMIZE_MEM_INIT",
                    "+define+RANDOMIZE_REG_INIT", "+define+RANDOMIZE_GARBAGE_ASSIGN",
                    "+define+RANDOMIZE_INVALID_ASSIGN", "+define+VCS", "+define+FSDB"}
LINK_NAMES = {"riscv", "fesvr", "dramsim"}
VENDOR_FILES = ("linux64/bin/vcs1", "linux64/lib/libvirsim.so",
                "linux64/lib/liberrorinf.so", "linux64/lib/libsnpsmalloc.so",
                "linux64/lib/libvfs.so", "linux64/lib/libvcsnew.so",
                "linux64/lib/libsimprofile.so", "linux64/lib/libuclinative.so",
                "linux64/lib/libvcsucli.so", "linux64/lib/vcs_tls.o",
                "linux64/lib/vcs_save_restore_new.o", "include/svdpi.h",
                "include/vpi_user.h")


def allowed(path, missing=False):
    result = _CHECK.allowed(path, missing=missing)
    # VCS's vendor wrapper constructs shell commands internally. Do not pass
    # workspace names requiring shell quoting to that wrapper.
    if not re.fullmatch(r"[A-Za-z0-9_./:+@%=-]+", str(result)):
        raise BuildError("selected VCS path requires unsupported shell quoting")
    return result


def ident(path):
    return _CHECK.Checker().identity(allowed(path))


def verify(member):
    if not isinstance(member, dict):
        raise BuildError("file identity is required")
    if not isinstance(member.get("path"), str) or not Path(member["path"]).is_absolute():
        raise BuildError("absolute file identity is required")
    allowed(member["path"])
    return _CHECK.Checker().verify(member, Path.cwd())


def pinned_json(path, expected):
    member = ident(path)
    if not re.fullmatch(r"[0-9a-f]{64}", expected or "") or member["sha256"] != expected:
        raise BuildError("selected JSON SHA256 mismatch")
    return json.loads(Path(member["path"]).read_text()), member


def new_output(path):
    output = allowed(path, missing=True)
    root = allowed(Path(__file__).parents[1])
    base = allowed(root / "build/rtl-timing", missing=True)
    if output == base or not output.is_relative_to(base) or output.exists():
        raise BuildError("output must be a new directory under ignored build/rtl-timing")
    result = subprocess.run(["git", "-C", str(root), "check-ignore", "--no-index",
                             "--quiet", "--", str(output)], capture_output=True)
    if result.returncode:
        raise BuildError("output is not ignored by this repository")
    return output


def saved_recipe(text, prepared=False):
    """Interpret one metadata command; never execute shell text."""
    lines = [line.strip() for line in text.splitlines()
             if line.strip() and not line.lstrip().startswith("#")]
    if len(lines) != 1:
        raise BuildError("saved VCS record must contain exactly one command")
    args = shlex.split(lines[0])
    if args and args[-1] == "2>&1":
        args.pop()
    if not args or Path(args.pop(0)).name != "vcs":
        raise BuildError("selected command is not VCS")
    result = {"flags": [], "direct": [], "defines": [], "incdirs": [],
              "filelists": [], "libraries": [], "cflags": [], "ldflags": []}
    seen = set()
    while args:
        value = args.pop(0)
        if value in ("-CFLAGS", "-LDFLAGS", "-f", "-timescale", "-assert", "-top", "-o") or \
                (prepared and value in ("-cc", "-cpp", "-ld")):
            if not args:
                raise BuildError("missing VCS option argument")
            argument = args.pop(0)
            if value != "-f" and value in seen:
                raise BuildError("duplicate selected VCS option")
            seen.add(value)
            if value in ("-CFLAGS", "-LDFLAGS"):
                result["cflags" if value == "-CFLAGS" else "ldflags"] = shlex.split(argument)
            elif value == "-f":
                result["filelists"].append(allowed(argument))
            elif value == "-o":
                result["output"] = allowed(argument, missing=True)
            elif value in ("-cc", "-cpp", "-ld"):
                result[value[1:]] = allowed(argument)
            else:
                expected = {"-timescale": "1ns/10ps", "-assert": "svaext", "-top": "TestDriver"}
                if argument != expected[value]:
                    raise BuildError("unsupported selected VCS elaboration option")
                result["flags"].extend([value, argument])
        elif value.startswith("-Mdir="):
            if "-Mdir" in seen:
                raise BuildError("duplicate VCS output directory")
            seen.add("-Mdir")
            result["mdir"] = allowed(value.split("=", 1)[1], missing=True)
        elif re.fullmatch(r"-j[0-9]+", value):
            if "jobs" in result or (prepared and not 1 <= int(value[2:]) <= 8):
                raise BuildError("duplicate or unbounded selected job count")
            result["jobs"] = int(value[2:])
        elif value.startswith("+incdir+"):
            result["incdirs"].extend(allowed(x) for x in value[8:].split("+") if x)
        elif value.startswith("+define+"):
            name = value[8:].split("=", 1)[0]
            if name in ("DEBUG", "SYNTHESIS", "GATE_LEVEL", "TESTBENCH_IN_UVM") or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
                raise BuildError("unsupported selected preprocessor mode")
            if not re.fullmatch(r"[A-Za-z0-9_+.!/=:-]+", value):
                raise BuildError("unsafe selected preprocessor argument")
            if value not in SELECTED_DEFINES or value in result["defines"]:
                raise BuildError("unsupported or duplicate selected define")
            result["defines"].append(value)
        elif value.startswith("-l") and value[2:] in (LINK_NAMES | ({"stdc++"} if prepared else set())):
            result["libraries"].append(value[2:])
        elif prepared and value in ("-debug_access+all", "-kdb", "-lca"):
            result["flags"].append(value)
        elif value == "-timescale=1ns/10ps":
            if "-timescale" in seen:
                raise BuildError("duplicate selected timescale")
            seen.add("-timescale")
            result["flags"].append(value)
        elif value in SIMPLE_FLAGS or value in SELECTED_FLAGS:
            if any(x in value for x in (";", "`", "$", "&", "|", "\n")):
                raise BuildError("unsafe saved argument")
            if value != "-debug_pp":
                result["flags"].append(value)
        elif Path(value).is_absolute() and Path(value).suffix in SOURCE_SUFFIXES:
            result["direct"].append(allowed(value))
        else:
            raise BuildError("unsupported saved VCS argument: " + value)
    if not {"-CFLAGS", "-LDFLAGS", "-f", "-top", "-o", "-Mdir"}.issubset(seen) or "-full64" not in result["flags"]:
        raise BuildError("incomplete selected VCS recipe")
    result["cincs"] = []
    for flag in result["cflags"]:
        if flag.startswith("-I") and len(flag) > 2:
            result["cincs"].append(allowed(flag[2:]))
        elif flag not in ("-O3", "-std=c++17"):
            raise BuildError("unsupported C++ compiler option")
    result["libdirs"] = []
    for flag in result["ldflags"]:
        if flag.startswith("-L") and len(flag) > 2:
            result["libdirs"].append(allowed(flag[2:]))
        elif flag.startswith("-Wl,-rpath,"):
            allowed(flag[len("-Wl,-rpath,"):])
        else:
            raise BuildError("unsupported selected linker option")
    if set(result["libraries"]) != LINK_NAMES | ({"stdc++"} if prepared else set()):
        raise BuildError("selected simulator library set is incomplete")
    if prepared and ("jobs" not in result or not {"cc", "cpp", "ld"}.issubset(result)):
        raise BuildError("prepared recipe must select bounded jobs and compiler/linker tools")
    return result


def filelist_sources(paths, known):
    result, records, active = [], [], set()

    def visit(path):
        path = allowed(path)
        if path in active:
            raise BuildError("recursive simulator filelist")
        if str(path) not in known:
            raise BuildError("unidentified simulator filelist")
        verify(known[str(path)])
        records.append(known[str(path)])
        active.add(path)
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith(("#", "//")):
                continue
            if line.startswith("-f "):
                nested = Path(line[3:].strip())
                visit(nested if nested.is_absolute() else path.parent / nested)
                continue
            if line.startswith(("+", "-")):
                raise BuildError("filelist metadata/options are unsupported")
            member = Path(line)
            member = allowed(member if member.is_absolute() else path.parent / member)
            if member.suffix not in SOURCE_SUFFIXES or str(member) not in known:
                raise BuildError("unidentified or unsupported filelist source")
            verify(known[str(member)])
            result.append(member)
        active.remove(path)

    for path in paths:
        visit(path)
    return result, records


def replacements(data, selected):
    """Accept a path→identity mapping, or entries[{original,replacement}]."""
    entries = data.get("replacements", data.get("entries", data)) if isinstance(data, dict) else data
    if isinstance(entries, dict):
        entries = [{"original": key, "replacement": value} for key, value in entries.items()]
    if not isinstance(entries, list):
        raise BuildError("replacement inventory entries are required")
    result = {}
    for entry in entries:
        original = entry.get("original")
        if isinstance(original, dict):
            original = verify(original)["path"]
        original = str(allowed(original))
        if original not in selected or original in result:
            raise BuildError("replacement does not name one selected source")
        value = verify(entry.get("replacement"))
        if Path(value["path"]).suffix not in SOURCE_SUFFIXES:
            raise BuildError("replacement is not a source file")
        result[original] = value
    return result


def sv_headers(sources, directories):
    """Resolve literal SV includes conservatively, including inactive branches."""
    result, active = {}, set()

    def visit(path):
        path = allowed(path)
        if path in active:
            raise BuildError("recursive SV include")
        active.add(path)
        text = re.sub(r"/\*.*?\*/|//[^\n]*", "", path.read_text(), flags=re.S)
        for line in text.splitlines():
            match = re.match(r'\s*`include\s+(.+?)\s*$', line)
            if not match:
                continue
            literal = re.fullmatch(r'"([^"\n]+)"', match[1])
            if not literal:
                raise BuildError("nonliteral SV include is unsupported")
            include = Path(literal[1])
            if include.is_absolute() or ".." in include.parts:
                raise BuildError("escaping SV include is unsupported")
            candidates = [allowed(base / include, missing=True) for base in [path.parent, *directories]]
            found = next((candidate for candidate in candidates if candidate.is_file()), None)
            if found is None:
                raise BuildError("unresolved SV include: " + literal[1])
            if str(found) not in result:
                result[str(found)] = ident(found)
                visit(found)
        active.remove(path)

    for source in sources:
        if source.suffix in (".sv", ".v"):
            visit(source)
    return list(result.values())


def dependency_paths(text):
    text = text.replace("\\\n", " ")
    if not text.startswith("atlas_dependencies:") or "\n" in text.rstrip("\n"):
        raise BuildError("unsupported C++ dependency output")
    tokens = shlex.split(text.partition(":")[2])
    if not tokens:
        raise BuildError("empty C++ dependency output")
    return [allowed(token) for token in tokens]


def selected_libraries(recipe):
    result = []
    for name in recipe["libraries"]:
        found = None
        for directory in recipe["libdirs"]:
            for suffix in (".so", ".a"):
                candidate = allowed(directory / ("lib" + name + suffix), missing=True)
                if candidate.is_file():
                    found = candidate
                    break
            if found:
                break
        if found is None:
            raise BuildError("unresolved selected linker library: " + name)
        result.append(ident(found))
    return result


def vendor_setup(vcs_home):
    """Bind the selected default setup chain without broad library traversal.

    This Verilog-only recipe accepts vendor package mappings and the default
    current-directory workspace. Third-party LMC_HOME mappings stay inactive
    because that environment variable is never passed to the VCS child.
    """
    primary = allowed(vcs_home / "bin/synopsys_sim.setup", missing=True)
    if not primary.exists():
        return []
    result, active = [], set()
    variables = {"SYNOPSYS_SIM": str(vcs_home), "ARCH": "linux64",
                 "SYNOPSYS_DW_FPGA_LIB_SETUP": str(vcs_home / "bin/synopsys_dw_fpga_lib.setup")}

    def visit(path):
        path = allowed(path)
        if path in active:
            raise BuildError("recursive VCS setup include")
        active.add(path)
        result.append(ident(path))
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("--"):
                continue
            match = re.fullmatch(r"([A-Za-z0-9_]+)\s*([=:>])\s*(.*?)\s*", line)
            if not match:
                raise BuildError("unsupported VCS installation setup syntax")
            name, operator, value = match.groups()
            if _CHECK.restricted(Path(value)):
                raise BuildError("restricted VCS installation setup reference")
            if name == "OTHERS":
                if operator != "=" or value != "$SYNOPSYS_DW_FPGA_LIB_SETUP":
                    raise BuildError("unsupported VCS setup include selection")
                visit(Path(variables["SYNOPSYS_DW_FPGA_LIB_SETUP"]))
            elif operator == ":":
                if value == ".":
                    continue
                if value in ("$LMC_HOME/synopsys/smartmodel", "$LMC_HOME/synopsys/flexmodel"):
                    continue  # The isolated child has no LMC_HOME.
                if not value.startswith("$SYNOPSYS_SIM/$ARCH/packages/"):
                    raise BuildError("unsupported external VCS library mapping")
                expanded = value.replace("$SYNOPSYS_SIM", variables["SYNOPSYS_SIM"]).replace("$ARCH", "linux64")
                allowed(expanded, missing=True)
            elif operator == ">":
                if (name, value) != ("WORK", "DEFAULT"):
                    raise BuildError("unsupported VCS workspace alias")
            elif "$" in value or "/" in value or not re.fullmatch(r"[A-Za-z0-9_. -]+", value):
                raise BuildError("unsupported VCS installation setup value")
        active.remove(path)

    visit(primary)
    return result


def mirrored(output, path):
    path = allowed(path)
    return output / "inputs/files" / Path(*path.parts[1:])


def snapshot(output, member):
    verified = verify(member)
    target = mirrored(output, verified["path"])
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(verified["path"], target)
    actual = ident(target)
    if (actual["sha256"], actual["bytes"]) != (verified["sha256"], verified["bytes"]):
        raise BuildError("snapshot identity mismatch")
    return actual


def snapshot_libraries(output, names, members):
    """Retain linker lookup names even when the selected file is a symlink.

    Regular byte-identical copies avoid output symlinks. Preserve the resolved
    basename beside libNAME.so as well, for the usual versioned SONAME lookup.
    """
    directory = output / "inputs/link-libraries"
    directory.mkdir(parents=True, exist_ok=True)
    result = []
    for name, member in zip(names, members):
        source = Path(verify(member)["path"])
        lookup = "lib" + name + (".a" if source.suffix == ".a" else ".so")
        filenames = (source.name, lookup, "libstdc++.so.6") if name == "stdc++" else (source.name, lookup)
        for filename in dict.fromkeys(filenames):
            destination = directory / filename
            if not destination.exists():
                shutil.copyfile(source, destination)
            actual = ident(destination)
            if (actual["sha256"], actual["bytes"]) != (member["sha256"], member["bytes"]):
                raise BuildError("linker library copy identity mismatch")
            result.append({"role": "link_library" if filename == source.name else "link_library_lookup",
                           "original": member, "snapshot": actual,
                           "link_name": name, "lookup_filename": lookup})
    return result


def compiler_environment(cxx, output):
    # No caller-provided compiler injection variables or shell setup scripts.
    return {"PATH": str(cxx.parent) + ":/usr/bin:/bin", "LC_ALL": "C", "LANG": "C"}


def prepare(args):
    producer_before, capture_before = ident(__file__), ident(_CAPTURE_PATH)
    output = new_output(args.output)
    witness, witness_identity = pinned_json(args.witness_report, args.expected_witness_sha256)
    if witness.get("target_config") != "EE290SimConfig" or witness.get("state") != "integrated_execution_passed":
        raise BuildError("a passing selected integrated witness receipt is required")
    provenance = witness["provenance"]
    record = verify(provenance["observed_simulator_build_record"])
    recipe = saved_recipe(Path(record["path"]).read_text())
    known = {verify(member)["path"]: member for member in provenance["simulator_sources"]}
    sources, list_records = filelist_sources(recipe["filelists"], known)
    directs = {verify(member)["path"]: member for member in provenance["observed_build_record_direct_sources"]}
    if set(map(str, recipe["direct"])) != set(directs):
        raise BuildError("saved direct sources do not match witnessed identities")
    sources = [*recipe["direct"], *sources]
    selected = {str(source): ident(source) for source in sources}
    replace, replacement_identity = {}, None
    if args.replacement_inventory:
        data, replacement_identity = pinned_json(args.replacement_inventory, args.expected_replacement_sha256)
        replace = replacements(data, selected)
    elif args.expected_replacement_sha256:
        raise BuildError("replacement inventory is missing")
    effective = [Path(replace.get(str(source), selected[str(source)])["path"]) for source in sources]
    vcs, cxx = allowed(args.vcs), allowed(args.cxx)
    cc = allowed(args.cc) if args.cc else allowed(cxx.parent / "gcc")
    if vcs.name != "vcs" or any(not os.access(path, os.X_OK) for path in (vcs, cxx, cc)):
        raise BuildError("explicit executable VCS and C++ tools are required")
    if not args.vcs_version.strip() or not args.cxx_version.strip():
        raise BuildError("caller-declared selected tool versions are required")
    if not 1 <= args.jobs <= 8 or not math.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0:
        raise BuildError("jobs must be 1..8 and timeout must be finite and positive")
    vcs_home = allowed(vcs.parent.parent)
    vendor = [ident(vcs_home / relative) for relative in VENDOR_FILES]
    vendor.extend(vendor_setup(vcs_home))
    libraries = selected_libraries(recipe)
    runtime = ident(args.cxx_runtime)
    if not re.fullmatch(r"libstdc\+\+\.so(?:\.[0-9]+)*", Path(runtime["path"]).name):
        raise BuildError("--cxx-runtime must select the explicit libstdc++ shared library")
    libraries.append(runtime)
    library_names = [*recipe["libraries"], "stdc++"]
    headers = sv_headers(effective, recipe["incdirs"])
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema": "atlas.ee290_simulator_preparation.v0", "target_config": "EE290SimConfig",
              "state": "preparing", "scheduling_qualified": False, "witness_report": witness_identity,
              "historical_build_record": record, "replacement_inventory": replacement_identity,
              "failures": [], "dependency_phases": []}
    try:
        cpp_headers = {}
        for number, source in enumerate(effective):
            if source.suffix not in CPP_SUFFIXES:
                continue
            flags = ["-w", "-pipe", "-fPIC", "-std=c++17", "-O3", *["-I" + str(directory) for directory in recipe["cincs"]],
                     "-I" + str(vcs_home / "include")]
            phase_dir = output / "dependencies" / str(number)
            phase = _CAPTURE.capture_phase(
                phase_dir, "cpp_dependencies", [str(cxx), *flags, "-M", "-MT", "atlas_dependencies",
                "-MF", "{output}/dependencies.d", str(source)],
                [{"role": "tool", "identity": ident(cxx)}, {"role": "source", "identity": ident(source)}],
                [{"role": "dependencies", "relative_path": "dependencies.d", "kind": "file"}],
                compiler_environment(cxx, output), args.timeout_seconds)
            report["dependency_phases"].append(ident(phase_dir / "phase.json"))
            if phase["state"] != "phase_completed":
                raise BuildError("C++ dependency discovery failed; see retained phase")
            for member in dependency_paths((phase_dir / "dependencies.d").read_text()):
                if member != source:
                    cpp_headers[str(member)] = ident(member)
        headers.extend(cpp_headers.values())
        snapshots, origins = {}, []
        for role, members in (("source", [ident(p) for p in effective]), ("header", headers)):
            for member in members:
                copy = snapshot(output, member)
                snapshots[member["path"]] = copy
                origins.append({"role": role, "original": member, "snapshot": copy})
        origins.extend(snapshot_libraries(output, library_names, libraries))
        # Make both quoted includes and explicit search directories refer to
        # the same mirrored source/header trees. Compiler-default system
        # headers remain selected, checked external inputs to the VCS phase.
        cpp_flags = [flag for flag in recipe["cflags"] if not flag.startswith("-I")]
        cpp_flags += ["-I" + str(mirrored(output, path)) for path in recipe["cincs"]]
        for directory in {*recipe["cincs"], *recipe["incdirs"]}:
            mirrored(output, directory).mkdir(parents=True, exist_ok=True)
        staged_sources = [Path(snapshots[str(source)]["path"]) for source in effective]
        staged_known = {entry["snapshot"]["path"] for entry in origins}
        staged_known.update(cpp_headers)
        for member in sv_headers(staged_sources, [mirrored(output, directory) for directory in recipe["incdirs"]]):
            if member["path"] not in staged_known:
                raise BuildError("staged SV include escapes the selected closure")
        # Recheck the relocated source/header recipe before asking for a
        # licensed compile. This catches alias names lost during canonical
        # copying and new dependencies introduced by include-path rewriting.
        dependency_inputs = [{"role": "tool", "identity": ident(cxx)}]
        dependency_inputs += [{"role": "header", "identity": entry["snapshot"]}
                              for entry in origins if entry["role"] == "header"]
        dependency_inputs += [{"role": "compiler_default_header", "identity": member}
                              for member in cpp_headers.values()]
        for number, source in enumerate(staged_sources):
            if source.suffix not in CPP_SUFFIXES:
                continue
            phase_dir = output / "dependencies" / ("staged-" + str(number))
            phase = _CAPTURE.capture_phase(
                phase_dir, "cpp_dependencies", [str(cxx), "-w", "-pipe", "-fPIC", *cpp_flags,
                "-I" + str(vcs_home / "include"), "-M", "-MT", "atlas_dependencies",
                "-MF", "{output}/dependencies.d", str(source)],
                [*dependency_inputs, {"role": "source", "identity": ident(source)}],
                [{"role": "dependencies", "relative_path": "dependencies.d", "kind": "file"}],
                compiler_environment(cxx, output), args.timeout_seconds)
            report["dependency_phases"].append(ident(phase_dir / "phase.json"))
            if phase["state"] != "phase_completed":
                raise BuildError("staged C++ dependency discovery failed; see retained phase")
            if any(str(member) not in staged_known for member in dependency_paths((phase_dir / "dependencies.d").read_text())):
                raise BuildError("staged C++ dependency escapes the selected closure")
        libdirs = [str(output / "inputs/link-libraries")]
        ldflags = ["-L" + path for path in libdirs] + ["-Wl,-rpath," + path for path in libdirs]
        filelist = output / "inputs/sim_sources.f"
        filelist.write_text("".join(snapshots[str(source)]["path"] + "\n" for source in effective[len(recipe["direct"]):]))
        argv = [str(vcs), *recipe["flags"], "-CFLAGS", shlex.join(cpp_flags),
                "-LDFLAGS", shlex.join(ldflags), *["-l" + name for name in library_names],
                *[snapshots[str(source)]["path"] for source in effective[:len(recipe["direct"])]],
                "-f", str(filelist), *recipe["defines"],
                *["+incdir+" + str(mirrored(output, directory)) for directory in recipe["incdirs"]],
                "-debug_access+all", "-kdb", "-lca", "-j" + str(args.jobs),
                "-cc", str(cc), "-cpp", str(cxx), "-ld", str(cxx),
                "-o", "{output}/simv", "-Mdir={output}/csrc"]
        inputs = [{"role": "tool", "identity": ident(vcs)},
                  {"role": "cxx_tool", "identity": ident(cxx)},
                  {"role": "cc_tool", "identity": ident(cc)},
                  {"role": "prepared_filelist", "identity": ident(filelist)}]
        inputs += [{"role": entry["role"], "identity": entry["snapshot"]} for entry in origins]
        inputs += [{"role": "compiler_default_header", "identity": member} for member in cpp_headers.values()]
        inputs += [{"role": "vendor_dependency", "identity": member} for member in vendor]
        environment = compiler_environment(cxx, output)
        environment.update({"VCS_HOME": str(vcs_home), "VCS_64": "1"})
        plan = {"schema": "atlas.ee290_simulator_plan.v0", "target_config": "EE290SimConfig",
                "state": "prepared", "scheduling_qualified": False, "witness_report": witness_identity,
                "historical_build_record": record, "historical_filelists": list_records,
                "replacement_inventory": replacement_identity,
                "replacements": [{"original": selected[key], "replacement": value} for key, value in replace.items()],
                "producer": producer_before, "capture_library": capture_before,
                "tool_versions": {"vcs": {"declaration": args.vcs_version, "identity": ident(vcs)},
                                  "cxx": {"declaration": args.cxx_version, "identity": ident(cxx)}},
                "cxx_runtime": runtime,
                "origins": origins, "dependency_phases": report["dependency_phases"],
                "dependency_closure": {"selected_sources": len(effective), "headers": len(headers),
                                       "project_link_libraries": len(libraries), "vendor_files": len(vendor),
                                       "relocated_dependency_checks_passed": True},
                "argv": argv, "inputs": inputs, "environment": environment,
                "outputs": [{"role": "simulator", "relative_path": "simv", "kind": "file"},
                            {"role": "simulator_archive", "relative_path": "simv.daidir", "kind": "tree"},
                            {"role": "generated_build", "relative_path": "csrc", "kind": "tree"}],
                "observability": {"flags": ["-debug_access+all", "-kdb", "-lca"],
                                  "DEBUG": False, "internal_trace_recorded": False},
                "limitations": ["The selected VCS wrapper and vendor compiler are trusted tool boundaries; all internal subprocess and system runtime dependencies are not reconstructed.",
                                "Compiler-default system headers are hashed external inputs, not a hermetic relocated toolchain.",
                                "Caller-declared tool versions are recorded with executable hashes, not independently verified version strings.",
                                "A new successful build and rerun can bind captured generated sources to that new executable; historical binary linkage and original Chisel derivation remain separate.",
                                "Debug access flags permit a later trace attempt; they do not establish signal visibility or timing evidence."]}
        for entry in inputs:
            verify(entry["identity"])
        if ident(__file__) != producer_before or ident(_CAPTURE_PATH) != capture_before:
            raise BuildError("builder/capture implementation changed during preparation; use a new output")
        plan_path = output / "plan.json"
        plan_path.write_text(json.dumps(plan, indent=2) + "\n")
        report.update({"state": "prepared", "plan": ident(plan_path)})
    except (OSError, ValueError, BuildError) as error:
        report["state"] = "preparation_failed"
        report["failures"].append(str(error))
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def run_plan(args):
    output = new_output(args.output)
    plan, plan_identity = pinned_json(args.plan, args.expected_plan_sha256)
    if plan.get("schema") != "atlas.ee290_simulator_plan.v0" or plan.get("state") != "prepared" or plan.get("target_config") != "EE290SimConfig":
        raise BuildError("selected prepared simulator plan is required")
    verify(plan["producer"])
    verify(plan["capture_library"])
    if plan["producer"]["sha256"] != ident(__file__)["sha256"]:
        raise BuildError("selected builder implementation changed")
    # The independently pinned plan is still not an arbitrary command runner.
    argv = plan["argv"]
    if not isinstance(argv, list) or any(not isinstance(value, str) for value in argv) or \
            argv[-3:] != ["-o", "{output}/simv", "-Mdir={output}/csrc"]:
        raise BuildError("plan does not use fresh isolated outputs")
    if Path(argv[0]).name != "vcs" or any(flag not in argv for flag in ("-debug_access+all", "-kdb", "-lca")):
        raise BuildError("unsupported selected plan command")
    parsed = saved_recipe(shlex.join([value.replace("{output}", str(output)) for value in argv]), prepared=True)
    if parsed["output"] != output / "simv" or parsed["mdir"] != output / "csrc":
        raise BuildError("selected plan output mismatch")
    known = {verify(entry["identity"])["path"]: entry["identity"] for entry in plan["inputs"]}
    compiled, unused = filelist_sources(parsed["filelists"], known)
    compiled = [*parsed["direct"], *compiled]
    for path in compiled:
        if str(path) not in known:
            raise BuildError("unidentified plan source")
    for member in [*sv_headers(compiled, parsed["incdirs"]), *selected_libraries(parsed)]:
        if member["path"] not in known:
            raise BuildError("unidentified plan include/library dependency")
    for name in ("cc", "cpp", "ld"):
        if str(parsed.get(name)) not in known:
            raise BuildError("unidentified plan compiler/linker")
    environment = dict(plan["environment"])
    if set(environment) != {"PATH", "LC_ALL", "LANG", "VCS_HOME", "VCS_64"} or \
            environment["VCS_HOME"] != str(allowed(argv[0]).parent.parent) or environment["VCS_64"] != "1":
        raise BuildError("unsupported plan environment")
    expected_path = str(parsed["cpp"].parent) + ":/usr/bin:/bin"
    if environment["PATH"] != expected_path:
        raise BuildError("unsupported compiler tool search path")
    environment["TMPDIR"] = str(output)
    environment["SNPS_VCS_TMPDIR"] = str(output)
    # License configuration is an explicit runtime input, never baked into
    # portable source or preparation. Capture receipt retains the supplied value.
    for key in ("SNPSLMD_LICENSE_FILE", "LM_LICENSE_FILE"):
        if os.environ.get(key):
            for value in os.environ[key].split(":"):
                if re.fullmatch(r"[0-9]+@[A-Za-z0-9_.-]+", value):
                    continue
                if not value.startswith("/"):
                    raise BuildError("unsupported license file/service selection")
                allowed(value)
            environment[key] = os.environ[key]
    if os.environ.get("LM_PROJECT"):
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", os.environ["LM_PROJECT"]):
            raise BuildError("unsupported license project selection")
        environment["LM_PROJECT"] = os.environ["LM_PROJECT"]
    for key in ("LD_LIBRARY_PATH",):
        if os.environ.get(key):
            for value in os.environ[key].split(":"):
                if not value:
                    raise BuildError("relative loader search paths are unsupported")
                allowed(value)
            environment[key] = os.environ[key]
    inputs = [*plan["inputs"], {"role": "selected_plan", "identity": plan_identity}]
    phase = _CAPTURE.capture_phase(output, "vcs_compile", argv, inputs,
                                 plan["outputs"], environment, args.timeout_seconds)
    return {"state": phase["state"], "phase": ident(output / "phase.json"),
            "scheduling_qualified": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run", action="store_true", help="explicit licensed compilation of a pinned plan")
    parser.add_argument("--prepare-only", action="store_true", help="explicitly select the default unlicensed preparation")
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--expected-plan-sha256")
    parser.add_argument("--witness-report", type=Path)
    parser.add_argument("--expected-witness-sha256")
    parser.add_argument("--vcs", type=Path)
    parser.add_argument("--vcs-version")
    parser.add_argument("--cxx", type=Path)
    parser.add_argument("--cc", type=Path, help="C compiler; defaults to gcc beside --cxx")
    parser.add_argument("--cxx-version")
    parser.add_argument("--cxx-runtime", type=Path,
                        help="explicit libstdc++ shared library for both linking and runtime loading")
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--timeout-seconds", type=float, default=600)
    parser.add_argument("--replacement-inventory", type=Path)
    parser.add_argument("--expected-replacement-sha256")
    args = parser.parse_args()
    if args.run and args.prepare_only:
        parser.error("--run and --prepare-only are mutually exclusive")
    if args.run:
        if not args.plan or not args.expected_plan_sha256:
            parser.error("--run requires --plan and --expected-plan-sha256")
        result = run_plan(args)
    else:
        if any(getattr(args, name) is None for name in ("witness_report", "expected_witness_sha256", "vcs", "vcs_version", "cxx", "cxx_version", "cxx_runtime")):
            parser.error("preparation requires pinned witness, explicit VCS/C++ tools/runtime and version declarations")
        if args.plan or args.expected_plan_sha256:
            parser.error("a plan is accepted only with --run")
        result = prepare(args)
    print(json.dumps(result, indent=2))
    return 0 if result["state"] in ("prepared", "phase_completed") else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError, BuildError) as error:
        print("EE290 simulator build: " + str(error), file=sys.stderr)
        sys.exit(2)
