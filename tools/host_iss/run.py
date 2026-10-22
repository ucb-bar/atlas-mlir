"""Run an unchanged RV64 graph ELF in generic QEMU with a diagnostic device.

Portable adaptation of the validated diagnostic bridge sources.
This adaptation itself has not executed a whole ELF. QEMU is all-stop while
npu_model ticks; npu_model is idle while QEMU runs. No SoC/FPGA binding.
"""

import argparse
import importlib
import json
import os
from pathlib import Path
import signal
import struct
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).absolute().parent.parent))

from atlas_tooling import write_json, sha as digest
from atlas_host_layout import HEADER as LAYOUT_HEADER, HEADER_SHA256
from elf_image import Image, mmio_sites
from gdb_remote import Remote
from model_device import Device, checked
from bundle import Bundle, sha, verify_completion


def interrupted(signum, frame):
    raise InterruptedError("host ISA replay terminated by signal " + str(signum))


def resnet_completion(remote, image, device, out):
    mailbox = remote.read(image.symbols["atlas_workload_mailbox"]["address"], 928)
    (out / "mailbox.bin").write_bytes(mailbox)
    fields = struct.unpack_from("<10I", mailbox)
    tohost = int.from_bytes(remote.read(image.symbols["tohost"]["address"], 8), "little")
    logits = remote.read(image.symbols["logits"]["address"], 4000)
    (out / "logits.bin").write_bytes(logits)
    expected = image.bytes_at(image.symbols["graph_expected"]["address"], 4000)
    verify_completion(mailbox, tohost, logits, expected, device.completed)
    return dict(mailbox_words=fields, tohost=tohost, logits_words=1000,
        logits_sha256=digest(logits), all_logits_exact=True)


def run(args):
    out = checked(Path(args.out).absolute())
    out.mkdir(exist_ok=False)
    receipt = dict(status="FAILED", command=[sys.executable, *sys.argv],
        scope="generic RV64 ISA execution plus serialized npu_model MMIO device; no SoC/FPGA/platform qualification",
        portable_adaptation=True, full_soc_executed=False, host_elf_executed=False, verbose_rsp_trace_enabled=False,
        kernels=[])
    signal.signal(signal.SIGTERM, interrupted)
    process = remote = device = None
    deadline = time.monotonic() + args.wall_seconds
    pins = {}
    try:
        workload = getattr(args, "workload", "resnet")
        contract = None
        if workload == "resnet":
            bundle = Bundle(Path(args.bundle_root).absolute())
        else:
            from block_bundle import BlockBundle
            bundle = BlockBundle(Path(args.bundle_root).absolute(), workload)
            contract = importlib.import_module(workload + "_contract")
        receipt["workload"] = workload
        elf = checked(Path(args.elf).absolute())
        qemu = checked(Path(args.qemu).absolute())
        assert sha(elf) == bundle.sums["host/host.elf"], "ELF differs from supplied bundle"
        image = Image(elf.read_bytes())
        sites = mmio_sites(image)
        if contract is None:
            stop_sites = {pc: length for name in ("atlas_workload_main", "atlas_block_trap_report")
                for pc, length, word in image.instructions(name) if length == 4 and word == 0x10500073}
        else:
            contract.validate_bundle(bundle, image)
            stop_sites = contract.stop_sites(image)
        assert stop_sites, "completion WFI instruction missing"
        helpers = ["run.py", "bundle.py", "model_device.py", "elf_image.py", "gdb_remote.py"]
        if contract is not None:
            helpers.append("block_bundle.py")
            path = checked(Path(contract.__file__).absolute())
            pins[str(path)] = sha(path)
        for name in helpers:
            path = checked(Path(__file__).absolute().parent / name)
            pins[str(path)] = sha(path)
        for path in (elf, qemu):
            pins[str(path)] = sha(path)
        pins[str(LAYOUT_HEADER)] = HEADER_SHA256
        for module in ("atlas_tooling", "atlas_path_guard", "atlas_host_layout", "run_model_tile", "atlas_tile_arithmetic"):
            path = checked(Path(sys.modules[module].__file__).absolute())
            pins[str(path)] = sha(path)
        model_root = checked(Path(args.model_root).absolute())
        sock = checked(out / "qemu.sock")
        command = [str(qemu), "-machine", "virt", "-m", "8G", "-smp", "1", "-accel", "tcg,thread=single", "-bios", "none",
            "-nodefaults", "-nographic", "-monitor", "none", "-serial", "none", "-S", "-gdb",
            "unix:qemu.sock,server=on,wait=off", "-kernel", str(elf)]
        receipt.update(qemu_command=command, inputs_sha256=pins, bundle_sha256sums_sha256=bundle.checksums_sha256,
            mmio_sites=sites, completion_breakpoints=sorted(stop_sites))
        with (out / "qemu.log").open("wb") as qlog:
            process = subprocess.Popen(command, cwd=out, stdout=qlog, stderr=subprocess.STDOUT, start_new_session=True)
            startup_deadline = min(deadline, time.monotonic() + 30)
            while not sock.exists():
                if process.poll() is not None:
                    raise RuntimeError("QEMU exited before debugger socket")
                if time.monotonic() >= startup_deadline:
                    raise TimeoutError("QEMU startup deadline")
                time.sleep(0.05)
            original_cwd = Path.cwd()
            os.chdir(out)
            remote = Remote("qemu.sock", trace=None, timeout=min(args.wall_seconds, 1800), deadline=deadline)
            assert remote.request("?")[:1] in (b"S", b"T")
            remote.set_reg(32, image.entry)
            for pc, site in sites.items():
                assert remote.read(pc, site["length"]) == image.bytes_at(pc, site["length"])
                remote.breakpoint(pc, site["length"])
            for pc, length in stop_sites.items():
                remote.breakpoint(pc, length)
            cycle_cap = getattr(args, "kernel_cycle_cap", None) or (
                contract.CYCLE_CAP if contract is not None else 20000000)
            device = Device(remote, out, deadline, bundle, model_root, cycle_cap=cycle_cap)
            receipt["kernel_cycle_cap"] = cycle_cap
            receipt["memory_backend"] = device.memory_backend
            accesses = 0
            while time.monotonic() < deadline:
                receipt["host_elf_execution_attempted"] = True
                pc, stop = remote.resume()
                receipt["host_elf_executed"] = True
                if pc in stop_sites:
                    break
                assert pc in sites, ("unexpected debugger stop", hex(pc), stop)
                access = sites[pc]
                address = remote.reg(access["base"]) + access["offset"]
                if access["store"]:
                    device.write32(address, remote.reg(access["reg"]) & 0xFFFFFFFF)
                else:
                    value = device.read32(address)
                    if access["signed"] and value & 0x80000000:
                        value |= 0xFFFFFFFF00000000
                    remote.set_reg(access["reg"], value)
                remote.set_reg(32, pc + access["length"])
                accesses += 1
                assert accesses < 100000, "MMIO access cap"
            else:
                raise TimeoutError("host graph wall cap")
            completion = (resnet_completion(remote, image, device, out) if contract is None
                else contract.verify_completion(remote, image, device, bundle, out))
            bundle.verify_bookends()
            device.verify_bookends()
            assert all(sha(Path(path)) == expected_hash for path, expected_hash in pins.items())
            assert time.monotonic() < deadline, "absolute completion deadline"
            receipt.update(status="PASS_RV64_HOST_WITH_MODEL_DEVICE", kernels=device.completed, mmio_accesses=accesses,
                one_device_reset=True, model_binding=device.binding, device_payloads_sha256=device.pins,
                source_bookends_equal=True, **completion)
    except Exception as error:
        receipt["error"] = dict(type=type(error).__name__, message=str(error))
        if device:
            receipt["kernels"] = device.completed
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)

        def terminate_qemu():
            if not process or process.poll() is not None:
                return
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)

        cleanup = []
        if device:
            cleanup.append(device.close)
        if remote:
            cleanup.append(remote.close)
        if "original_cwd" in locals():
            cleanup.append(lambda: os.chdir(original_cwd))
        cleanup.append(terminate_qemu)
        for close in cleanup:
            try:
                close()
            except Exception as error:
                detail = dict(type=type(error).__name__, message=str(error))
                receipt.setdefault("cleanup_errors", []).append(detail)
                receipt.setdefault("error", detail)
                receipt["status"] = "FAILED"
        receipt["wall_remaining_seconds"] = deadline - time.monotonic()
        write_json(out / "receipt.json", receipt)
    print(json.dumps({k: receipt[k] for k in ("status", "error") if k in receipt}), flush=True)
    return 0 if receipt["status"].startswith("PASS_") else 1


if __name__ == "__main__":
    if not __debug__:
        raise SystemExit("Python -O is unsupported: strict runtime checks require assertions")
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--wall-seconds", type=int, default=21600)
    parser.add_argument("--workload", choices=("resnet", "tiny", "smol"), default="resnet")
    parser.add_argument("--kernel-cycle-cap", type=int)
    for name in ("elf", "bundle-root", "model-root", "qemu"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    assert 0 < args.wall_seconds <= 86400
    assert args.kernel_cycle_cap is None or args.kernel_cycle_cap > 0
    raise SystemExit(run(args))
