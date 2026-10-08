"""Capture execution facts for one explicit EE290SimConfig build phase.

This library does not interpret successful commands as build-linkage proof.
Callers provide and validate the phase recipe and complete input selection for
their claimed scope. Hidden tool inputs and system runtime remain outside this
recorder's completeness claims. No shell command or saved build text is run.
"""

from collections import deque
from datetime import datetime, timezone
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import time


def _bootstrap_path(path):
    """Guard the dependency itself before importing its shared path checker."""
    path = Path(path)
    denied = lambda value: any("hammer" in part.lower() or "vlsi" in part.lower()
                               for part in value.parts)
    if denied(path):
        raise ValueError("restricted capture dependency path")
    if not path.is_absolute():
        path = Path.cwd() / path
    if denied(path):
        raise ValueError("restricted capture dependency path")
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
            raise ValueError("restricted capture dependency path")
        if stat.S_ISLNK(candidate.lstat().st_mode):
            if candidate in links or len(links) >= 64:
                raise ValueError("recursive capture dependency symlink")
            links.add(candidate)
            target = Path(os.readlink(candidate))
            if denied(target):
                raise ValueError("restricted capture dependency symlink target")
            if target.is_absolute():
                current = Path(target.anchor)
                pending.extendleft(reversed(target.parts[1:]))
            else:
                pending.extendleft(reversed(target.parts))
        else:
            current = candidate
    return current


_DEPENDENCY = _bootstrap_path(Path(__file__).parent / "check-ee290-provenance.py")
_SPEC = importlib.util.spec_from_file_location("atlas_ee290_provenance_dependency", _DEPENDENCY)
_CHECK = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_CHECK)
CaptureError = _CHECK.ProvenanceError


def _output_spec(output, declarations):
    if not isinstance(declarations, list) or not declarations:
        raise CaptureError("at least one declared output is required")
    result, paths = [], set()
    reserved = {"phase.json", "command.stdout.log", "command.stderr.log",
                "ee290_build_capture.executed.py", "check-ee290-provenance.dependency.py"}
    for item in declarations:
        if not isinstance(item, dict) or not isinstance(item.get("role"), str) or not item["role"] or \
                item.get("kind") not in ("file", "tree") or not isinstance(item.get("relative_path"), str):
            raise CaptureError("malformed declared output")
        path = Path(item["relative_path"])
        if path.is_absolute() or not path.parts or ".." in path.parts or \
                path.as_posix() in reserved or _CHECK.restricted(path):
            raise CaptureError("declared output must be an allowed relative owned path")
        destination = _CHECK.allowed(output / path, missing=True)
        if not destination.is_relative_to(output) or destination == output or destination in paths:
            raise CaptureError("duplicate or escaping declared output")
        if destination.exists():
            raise CaptureError("declared output already exists")
        paths.add(destination)
        result.append({"role": item["role"], "kind": item["kind"], "relative_path": path.as_posix()})
    return result


def _outputs(output, declarations):
    checker, result = _CHECK.Checker(), []

    def file(path):
        resolved = _CHECK.allowed(path)
        if not resolved.is_relative_to(output):
            raise CaptureError("output symlink escapes the fresh phase directory")
        member = checker.identity(resolved)
        return {"relative_path": path.relative_to(output).as_posix(), "identity": member}

    def tree(path, active):
        resolved = _CHECK.allowed(path)
        if not resolved.is_relative_to(output) or resolved in active:
            raise CaptureError("escaping or recursive output tree")
        entries = []
        with os.scandir(resolved) as children:
            for child in children:
                # Validate names before stat, opening or following a target.
                if _CHECK.restricted(Path(child.name)):
                    raise CaptureError("restricted generated output name")
                entries.append(child.name)
        files = []
        for name in sorted(entries):
            child = path / name
            target = _CHECK.allowed(child)
            if not target.is_relative_to(output):
                raise CaptureError("output symlink escapes the fresh phase directory")
            mode = target.stat().st_mode
            if stat.S_ISDIR(mode):
                files.extend(tree(child, active | {resolved}))
            elif stat.S_ISREG(mode):
                files.append(file(child))
            else:
                raise CaptureError("unsupported generated output file kind")
        return files

    for declaration in declarations:
        path = output / declaration["relative_path"]
        resolved = _CHECK.allowed(path)
        if declaration["kind"] == "file":
            files = [file(path)]
        elif not stat.S_ISDIR(resolved.stat().st_mode):
            raise CaptureError("declared tree output is not a directory")
        else:
            files = tree(path, set())
            if not files:
                raise CaptureError("declared tree output is empty")
        result.append({**declaration, "files": files})
    return result


def capture_phase(output_dir, kind, argv, inputs, outputs, environment,
                  timeout_seconds, cwd=None):
    """Run a selected command and return its retained execution receipt.

    Preflight errors raise CaptureError without creating output_dir. Command,
    timeout, input-stability and expected-output failures produce phase_failed
    receipts. Exactly one input with role 'tool' must identify argv[0]. Only
    the caller-supplied environment is passed to the child. {output} expands
    in argv/cwd. Inputs are file identities; trees are exported as file lists.
    """
    if not isinstance(kind, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", kind):
        raise CaptureError("invalid named phase kind")
    if not isinstance(argv, list) or not argv or any(not isinstance(value, str) or not value for value in argv):
        raise CaptureError("nonempty non-shell argv is required")
    if type(timeout_seconds) not in (int, float) or not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise CaptureError("a finite positive timeout is required")
    if not isinstance(environment, dict) or any(not isinstance(key, str) or not isinstance(value, str) or
                                               "=" in key or "\0" in key + value for key, value in environment.items()):
        raise CaptureError("environment must contain explicit string key/value pairs")
    if any(environment.get(key) for key in ("JAVA_TOOL_OPTIONS", "_JAVA_OPTIONS", "JDK_JAVA_OPTIONS", "CLASSPATH")):
        raise CaptureError("uncontrolled Java option/classpath injection is unsupported")
    output = _CHECK.allowed(output_dir, missing=True)
    root = _CHECK.allowed(__file__).parents[1]
    artifact_root = _CHECK.allowed(root / "build/rtl-timing", missing=True)
    if output == artifact_root or not output.is_relative_to(artifact_root):
        raise CaptureError("phase output must be beneath ignored build/rtl-timing")
    if output.exists():
        raise CaptureError("phase output directory already exists")
    ignored = subprocess.run(["git", "-C", str(root), "check-ignore", "--no-index", "--quiet", "--", str(output)],
                             capture_output=True, check=False)
    if ignored.returncode:
        raise CaptureError("phase output must be ignored by the repository")
    expanded = [value.replace("{output}", str(output)) for value in argv]
    cwd = Path(str(cwd).replace("{output}", str(output))) if cwd is not None else output
    working = _CHECK.allowed(cwd, missing=cwd == output)
    if working != output and not stat.S_ISDIR(working.stat().st_mode):
        raise CaptureError("phase cwd is not a directory")
    declared = _output_spec(output, outputs)
    if not isinstance(inputs, list) or not inputs:
        raise CaptureError("explicit input identities are required")
    checker, before, tool = _CHECK.Checker(), [], None
    for entry in inputs:
        if not isinstance(entry, dict) or not isinstance(entry.get("role"), str) or not entry["role"]:
            raise CaptureError("malformed selected input")
        actual = checker.verify(entry.get("identity"), working)
        before.append({"role": entry["role"], "identity": actual})
        if entry["role"] == "tool":
            if tool is not None:
                raise CaptureError("exactly one selected tool input is required")
            tool = actual
    if tool is None:
        raise CaptureError("exactly one selected tool input is required")
    executable = Path(expanded[0])
    if not executable.is_absolute():
        raise CaptureError("argv[0] must be an explicitly identified absolute executable")
    executable = _CHECK.allowed(executable)
    if str(executable) != tool["path"] or not os.access(executable, os.X_OK):
        raise CaptureError("argv[0] does not match the selected executable tool")
    if executable.name.lower() in ("make", "gmake", "sbt"):
        raise CaptureError("this capture workflow does not invoke make or sbt")
    expanded[0] = str(executable)
    producer = checker.identity(_CHECK.allowed(__file__))
    dependency = checker.identity(_DEPENDENCY)
    producer_bytes = Path(producer["path"]).read_bytes()
    dependency_bytes = Path(dependency["path"]).read_bytes()
    checker.recheck()
    output.mkdir(parents=True, exist_ok=False)
    (output / "ee290_build_capture.executed.py").write_bytes(producer_bytes)
    (output / "check-ee290-provenance.dependency.py").write_bytes(dependency_bytes)
    stdout, stderr = output / "command.stdout.log", output / "command.stderr.log"
    started, timed_out, code, launch_error = time.monotonic(), False, None, None
    child, interrupted, descendants = None, None, False
    with stdout.open("wb") as out, stderr.open("wb") as err:
        try:
            child = subprocess.Popen(expanded, cwd=working, env=dict(environment),
                                     stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                                     shell=False, start_new_session=True)
            try:
                code = child.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
            except BaseException as error:
                interrupted = error
        except OSError as error:
            launch_error = str(error)
        finally:
            if child is not None:
                # Reaping the leader does not prove its inherited-log writers
                # exited. Stop the whole session before inspecting outputs.
                try:
                    os.killpg(child.pid, 0)
                except ProcessLookupError:
                    pass
                else:
                    descendants = code is not None and not timed_out and interrupted is None
                    try:
                        os.killpg(child.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                # Preserve an already observed leader exit code. poll() may
                # itself reap a killed leader before wait() is needed.
                polled = child.poll()
                if polled is None:
                    polled = child.wait()
                if code is None:
                    code = polled
    after, failures = [], []
    for entry in inputs:
        try:
            path = Path(entry["identity"]["path"])
            actual = _CHECK.Checker().identity(path if path.is_absolute() else working / path)
            after.append({"role": entry["role"], "identity": actual})
        except (OSError, CaptureError) as error:
            after.append({"role": entry["role"], "error": str(error)})
    stable = before == after
    if launch_error:
        failures.append("command launch failed: " + launch_error)
    if interrupted is not None:
        failures.append("command interrupted: " + type(interrupted).__name__)
    if descendants:
        failures.append("command left process-group descendants after leader exit")
    if code != 0 or timed_out:
        failures.append("command failed or timed out")
    if not stable:
        failures.append("selected inputs changed during execution")
    recorded_outputs = []
    try:
        recorded_outputs = _outputs(output, declared)
    except (OSError, CaptureError) as error:
        failures.append("declared outputs unavailable: " + str(error))
    final = _CHECK.Checker()
    snapshots = {"producer": final.identity(output / "ee290_build_capture.executed.py"),
                 "path_checker_dependency": final.identity(output / "check-ee290-provenance.dependency.py")}
    if snapshots["producer"]["sha256"] != producer["sha256"] or \
            snapshots["path_checker_dependency"]["sha256"] != dependency["sha256"]:
        failures.append("capture implementation snapshots changed during execution")
    receipt = {"schema": "atlas.ee290_captured_phase.v0", "target_config": "EE290SimConfig", "kind": kind,
               "state": "phase_failed" if failures else "phase_completed", "scheduling_qualified": False,
               "created_utc": datetime.now(timezone.utc).isoformat(), "producer": producer,
               "path_checker_dependency": dependency, "snapshots": snapshots,
               "command": {"argv": expanded, "cwd": str(working), "environment": dict(environment),
                           "returncode": code, "timed_out": timed_out, "timeout_seconds": timeout_seconds,
                           "elapsed_seconds": time.monotonic() - started,
                           "stdout": final.identity(stdout), "stderr": final.identity(stderr)},
               "inputs_before": before, "inputs_after": after, "inputs_stable": stable,
               "declared_outputs": declared, "outputs": recorded_outputs, "failures": failures,
               "limitations": ["A completed phase records a successful selected command; it does not certify its recipe or build lineage.",
                               "Input-selection completeness, hidden tool inputs, subprocess tool closure and system runtime require separate scoped review.",
                               "A dedicated recipe checker must bind phase semantics and parent/output relationships before promoting any build-link edge."]}
    (output / "phase.json").write_text(json.dumps(receipt, indent=2) + "\n")
    if interrupted is not None:
        raise interrupted
    return receipt
