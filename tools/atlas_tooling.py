"""File identities, JSON receipts and bounded subprocesses for Atlas tools."""

import hashlib
import json
import stat
import subprocess
import time
import zipfile
from atlas_path_guard import checked


def preflight_archive(path):
    path = checked(path)
    if path.suffix.lower() in (".npz", ".jar", ".pth"):
        with zipfile.ZipFile(path) as archive:
            names = []
            for member in archive.infolist():
                name = member.filename
                if (not name or name.startswith(("/", "\\")) or "\\" in name or ":" in name
                    or any(part in ("", ".", "..") for part in name.rstrip("/").split("/"))
                    or stat.S_ISLNK(member.external_attr >> 16)):
                    raise ValueError("Unadmitted archive member path")
                names.append(name)
            if len(names) != len(set(names)):
                raise ValueError("Duplicate archive members")
    return path


def sha(data):
    return hashlib.sha256(data).hexdigest()


def sha256(path):
    return sha(preflight_archive(path).read_bytes())


def write_json(path, value):
    checked(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def run_command(command, **kwargs):
    kwargs.setdefault("timeout", 1500)
    return subprocess.run(command, **kwargs)


def bounded(command, cwd, env, log_path, support=None):
    started = time.monotonic()
    with checked(log_path).open("w") as log:
        try:
            code = run_command(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        except subprocess.TimeoutExpired:
            code = 124
    usage = {"command": command, "returncode": code, "wall_seconds": time.monotonic() - started, "wall_limit_seconds": 1500}
    (support.write_json if support is not None else write_json)(log_path.with_suffix(".resources.json"), usage)
    return usage


class Admission:
    def __init__(self):
        self.inputs = {}

    def read(self, path, digest=None):
        path = checked(path)
        raw = path.read_bytes()
        actual = sha(raw)
        if digest is not None and digest != actual:
            raise ValueError("input digest differs: " + str(path))
        if str(path) in self.inputs and self.inputs[str(path)] != actual:
            raise ValueError("input changed during admission")
        self.inputs[str(path)] = actual
        return raw

    def recheck(self):
        return all(sha256(path) == digest for path, digest in self.inputs.items())
