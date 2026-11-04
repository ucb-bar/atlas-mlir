"""Run outside the checkout, without ATLAS_OOT_BIN_DIR or PYTHONPATH."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile

from atlas_mlir import toolchain


def run(*args):
    result = subprocess.run(list(map(str, args)), text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(f"{args}: {result.stdout}\n{result.stderr}")


def check_program(out):
    report = json.loads((out / "report.json").read_text())
    data = (out / "compiled/program.bin").read_bytes()
    assert data and len(data) % 4 == 0
    assert report["program"]["words"] == len(data) // 4
    assert struct.unpack_from("<I", data, len(data) - 4)[0] == 0x73
    assert (out / "host/atlas_program.h").is_file()
    assert (out / "host/atlas_captured_data.h").is_file()
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--require-host-elf", action="store_true")
    args = parser.parse_args()
    assert not os.environ.get("ATLAS_OOT_BIN_DIR")
    assert not os.environ.get("PYTHONPATH")
    bins = toolchain.bin_dir()
    assert "site-packages" in str(bins), bins
    assert "site-packages" in str(toolchain.data_root())
    entry = [
        e
        for e in importlib.metadata.entry_points(group="merlin.native_target_compilers")
        if e.name == "atlas"
    ]
    assert len(entry) == 1
    for name in ("atlas-opt", "atlas-emit"):
        run(name, "--help")
    run("atlasc", "doctor", *(["--capture"] if args.capture else []))
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        run("atlasc", "demo", "-o", root / "demo")
        check_program(root / "demo")
        if args.capture:
            import torch

            # A fresh user model: no access to repository fixtures.
            torch.manual_seed(0)
            model = torch.nn.Sequential(
                torch.nn.Linear(32, 32), torch.nn.ReLU(), torch.nn.Linear(32, 32)
            ).eval()
            inputs = (torch.ones(32, 32),)
            pt2 = root / "model.pt2"
            torch.export.save(torch.export.export(model, inputs), pt2)
            run("atlasc", "build", pt2, "--no-host-elf", "-o", root / "pt2")
            check_program(root / "pt2")
            loader = root / "model.py"
            loader.write_text(
                "import torch\ndef get_model_and_inputs():\n"
                "    torch.manual_seed(0)\n"
                "    return torch.nn.Sequential(torch.nn.Linear(32,32), "
                "torch.nn.ReLU(), torch.nn.Linear(32,32)).eval(), "
                "(torch.ones(32,32),)\n"
            )
            if args.require_host_elf:
                assert shutil.which("clang") and shutil.which("ld.lld"), (
                    "host ELF check needs clang and lld"
                )
            flags = (
                ["--host-cc", "clang"]
                if shutil.which("clang") and shutil.which("ld.lld")
                else ["--no-host-elf"]
            )
            run(
                sys.executable,
                "-m",
                "atlas_mlir",
                "build",
                loader,
                *flags,
                "-o",
                root / "loader",
            )
            assert check_program(root / "loader") == check_program(root / "pt2")
            if "--host-cc" in flags:
                elf = (root / "loader/host/atlas-host.elf").read_bytes()
                assert elf[:6] == b"\x7fELF\x02\x01"
                assert struct.unpack_from("<H", elf, 18)[0] == 243
    print("Installed wheel smoke test passed")


if __name__ == "__main__":
    main()
