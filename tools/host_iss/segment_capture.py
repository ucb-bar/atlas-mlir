"""Capture ResNet segment boundary state from the full RV64 host in QEMU, or check one segment ELF in QEMU.

capture: runs the unchanged full host ELF and stops at every atlas_graph_pack entry (boundary k = start of segment
k) and at completion (boundary 54). Each boundary saves, with QEMU pmemsave, the tensors entering segment k, every
tensor written by segment k-1, the whole DMA arena, the device context and, at the end, the logits.
check: runs one segment ELF, intercepting its HTIF console and exit, and reports its SEG_ lines.

The device is a replay of the npu_model results: at START it requires the host-staged arena to equal the frozen
initial image and IMEM to equal the frozen program plus runtime epilogue, then writes the frozen output tile image.
Every payload is pinned to the npu_model epoch3 receipt, whose model outputs equalled these bytes for all 54 launches.
QEMU stops while the device acts. No SoC timing claim.
"""

import argparse
import json
import os
import re
import struct
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).absolute().parent))
sys.path.insert(0, str(Path(__file__).absolute().parents[1]))
sys.path.insert(0, str(Path(__file__).absolute().parents[1] / "chipyard"))

from elf_image import Image, mmio_sites
from gdb_remote import Remote, frame
from resnet_segments import admitted_path, bound_host_file, context_fields, host_identity, plan, read
from atlas_tooling import sha, write_json
from bundle import EPILOGUE
from atlas_host_layout import (ATLAS_IMEM_BASE as IMEM, ATLAS_CSR_BASE as CSR, ATLAS_DRAM_BASE as ARENA,
    ATLAS_CSR_CYCLES as CSR_CYCLES, ATLAS_CSR_RETIRED as CSR_RETIRED, ATLAS_CSR_STATUS as CSR_STATUS,
    ATLAS_CSR_PC as CSR_PC, ATLAS_CSR_DBG0 as CSR_DBG0, ATLAS_CSR_DBG1 as CSR_DBG1,
    ATLAS_CSR_EXEC_CONTROL as CSR_START, ATLAS_IMEM_BYTES)


class Qemu:
    def __init__(self, qemu, elf, out, icount, wall_seconds):
        self.out = out
        self.deadline = time.monotonic() + wall_seconds
        self.remote = None
        sock = out / "qemu.sock"
        command = [str(qemu), "-machine", "virt", "-m", "8G", "-smp", "1", *(["-icount", "shift=0"] if icount else []),
                   "-bios", "none", "-nodefaults", "-nographic", "-monitor", "none", "-serial", "none", "-S", "-gdb",
                   "unix:qemu.sock,server=on,wait=off", "-kernel", str(elf)]
        with (out / "qemu.log").open("wb") as log:
            self.process = subprocess.Popen(command, cwd=out, stdout=log, stderr=subprocess.STDOUT)
        self.command = command
        try:
            deadline = min(self.deadline, time.monotonic() + 30)
            while not sock.exists():
                if self.process.poll() is not None or time.monotonic() >= deadline:
                    raise RuntimeError("QEMU did not open its debugger socket")
                time.sleep(0.05)
            previous_cwd = Path.cwd()
            try:
                os.chdir(out)
                self.remote = Remote("qemu.sock", timeout=min(wall_seconds, 1800), deadline=self.deadline)
            finally:
                os.chdir(previous_cwd)
            assert self.remote.request("?")[:1] in (b"S", b"T")
        except BaseException:
            self.close()
            raise

    def monitor(self, command):
        self.remote.send(frame(b"qRcmd," + command.encode().hex().encode()))
        assert self.remote.byte() == b"+"
        output = b""
        while True:
            reply = self.remote.receive()
            if reply == b"OK":
                return output
            assert reply[:1] == b"O", reply
            output += bytes.fromhex(reply[1:].decode())

    def save(self, address, size, path):
        """Physical memory to a file via the QEMU monitor; much faster than RSP reads."""
        path.unlink(missing_ok=True)
        # A leading '/' would continue the size expression; QEMU runs in self.out.
        output = self.monitor(f"pmemsave {address:#x} {size} {json.dumps(os.path.relpath(path, self.out))}")
        assert not output, output
        raw = read(path)
        assert len(raw) == size
        return raw

    def close(self):
        if self.remote:
            self.remote.close()
        if self.process.poll() is None:
            self.process.kill()
        self.process.wait(timeout=10)


class ReplayDevice:
    """Atlas CSR/IMEM behaviour sufficient for the raw runtime, with frozen npu_model results."""

    def __init__(self, qemu, payloads, scratch):
        self.qemu, self.payloads, self.scratch = qemu, payloads, scratch
        self.imem, self.csr = {}, {CSR_STATUS: 1, CSR_CYCLES: 0, CSR_RETIRED: 0, CSR_PC: 0, CSR_DBG0: 0, CSR_DBG1: 0}
        self.started = []

    def read32(self, address):
        if IMEM <= address < IMEM + ATLAS_IMEM_BYTES:
            return self.imem.get(address - IMEM, 0)
        return self.csr[address - CSR]

    def write32(self, address, value):
        if IMEM <= address < IMEM + ATLAS_IMEM_BYTES:
            self.imem[address - IMEM] = value
        elif address - CSR == CSR_START:
            assert value == 1
            self.start()
        else:
            assert address - CSR in (CSR_DBG0, CSR_DBG1, CSR_CYCLES, CSR_RETIRED), hex(address)
            self.csr[address - CSR] = value

    def start(self):
        k = self.payloads["first"] + len(self.started)
        assert k < 54
        if self.payloads.get("single"):
            assert not self.started, "segment attempted more than one launch"
        p = self.payloads["kernels"][k]
        program = p["program"][:-4] + EPILOGUE
        words = struct.unpack(f"<{len(program) // 4}I", program)
        assert sorted(self.imem) == [4 * i for i in range(len(words))], "IMEM extent differs from frozen program"
        assert tuple(self.imem[4 * i] for i in range(len(words))) == words, "host IMEM differs from frozen program"
        assert self.csr[CSR_DBG0] == 0 and self.csr[CSR_DBG1] == k + 1, "sequence marker differs"
        staged = self.qemu.save(ARENA, len(p["initial"]), self.scratch / "staged.bin")
        assert staged == p["initial"], f"host-staged arena differs from frozen input of launch {k}"
        self.qemu.remote.write(ARENA + p["output_offset"], p["expected"])
        self.csr.update({CSR_DBG0: k + 1, CSR_STATUS: 5, CSR_CYCLES: self.csr[CSR_CYCLES] + p["cycles"],
                         CSR_RETIRED: self.csr[CSR_RETIRED] + len(words)})
        self.imem.clear()
        self.started.append(dict(index=k, staged_sha256=sha(staged), program_sha256=sha(program)))


def model_directories(model_root):
    entries = [(int(match[1]), admitted_path(path / "model")) for path in model_root.iterdir()
               if not re.search(r"hammer|vlsi", path.name, re.I)
               and (match := re.fullmatch(r"kernel-(\d+)(?:-[A-Za-z0-9_.-]+)?", path.name))]
    entries.sort()
    if [index for index, _ in entries] != list(range(54)):
        raise ValueError("model kernel indices must be exactly 0..53")
    directories = [directory for _, directory in entries]
    return directories


def payloads(model_root, epoch, host):
    """Frozen per-launch model payloads, each pinned to the npu_model epoch3 receipt."""
    receipt = json.loads(read(epoch))
    host_build, digests = host_identity(host)
    if host_build["elf_sha256"] not in receipt.get("inputs_sha256", {}).values():
        raise ValueError("epoch receipt does not bind the supplied host ELF")
    assert receipt["status"] == "PASS_RV64_HOST_WITH_MODEL_DEVICE" and len(receipt["kernels"]) == 54
    assets = bound_host_file(host / "assets.c", digests).decode()
    rows = assets.split("graph_kernels[54]={\n", 1)[1].split("\n};")[0].splitlines()
    kernels = []
    for k, (row, model) in enumerate(zip(rows, model_directories(model_root), strict=True)):
        pinned = receipt["kernels"][k]
        raw = {n: read(model / f) for n, f in (("program", "program.bin"), ("initial", "initial-dram.bin"),
                                                         ("expected", "expected-bf16-tiles.bin"))}
        fields = dict(re.findall(r"\.(\w+)=(-?\d+)", row))
        assert len(raw["program"]) >= 4 and len(raw["program"]) % 4 == 0 and raw["program"][-4:] == struct.pack("<I", 0x73)
        assert pinned["index"] == k and pinned["exact"] and pinned["drained"]
        assert sha(raw["program"][:-4] + EPILOGUE) == pinned["program_sha256"]
        assert sha(raw["initial"]) == pinned["host_input_sha256"]
        assert sha(raw["expected"]) == pinned["expected_output_sha256"] == pinned["actual_output_sha256"]
        assert int(fields["output_offset"]) + len(raw["expected"]) + 256 == int(fields["dram_bytes"]) == len(raw["initial"])
        kernels.append(dict(raw, output_offset=int(fields["output_offset"]), dram_bytes=int(fields["dram_bytes"]),
                            cycles=pinned["cycles"], model=str(model)))
    return kernels


class ModelPayloads:
    """Bundle-style reads of kernels/NN/model/* from the frozen network-model directories."""

    def __init__(self, model_root):
        self.dirs = model_directories(model_root)

    def read(self, name):
        _, index, _, member = name.split("/")
        return read(self.dirs[int(index)] / member)


def model_device(qemu, args, out):
    """The live npu_model Core (as in run.py) instead of the replay; it checks the same inputs and outputs."""
    from model_device import Device

    device = Device(qemu.remote, out, qemu.deadline, ModelPayloads(args.model_root), args.npu_model)
    device.started = device.completed
    return device


def serve(qemu, image, device, stops, on_stop):
    """Run until a stop handler returns True; bridge MMIO accesses to the device."""
    sites = mmio_sites(image)
    for pc, site in sites.items():
        qemu.remote.breakpoint(pc, site["length"])
    for pc in stops:
        qemu.remote.breakpoint(pc, 2)
    while time.monotonic() < qemu.deadline:
        pc, _ = qemu.remote.resume()
        if pc in sites:
            access = sites[pc]
            address = qemu.remote.reg(access["base"]) + access["offset"]
            if access["store"]:
                device.write32(address, qemu.remote.reg(access["reg"]) & 0xFFFFFFFF)
            else:
                value = device.read32(address)
                if access["signed"] and value & 0x80000000:
                    value |= 0xFFFFFFFF00000000
                qemu.remote.set_reg(access["reg"], value)
            qemu.remote.set_reg(32, pc + access["length"])
        elif pc in stops:
            if on_stop(pc):
                return
            if qemu.remote.reg(32) == pc:
                # Resuming on an inserted breakpoint would stop again at once: step over it first.
                assert qemu.remote.request(f"z0,{pc:x},2") == b"OK"
                assert qemu.remote.request("s")[:3] in (b"S05", b"T05")
                qemu.remote.breakpoint(pc, 2)
        else:
            raise AssertionError(f"unexpected stop at {pc:#x}")
    raise TimeoutError("host segment wall deadline")


def capture(args):
    out = args.out
    out.mkdir(parents=True)
    p = plan(args.host / "graph/graph.c")
    kernels = payloads(args.model_root, args.epoch, args.host)
    image = Image(read(args.elf))
    sym = image.symbols
    if sha(read(args.elf)) != sha(read(args.host / "host.elf")):
        raise ValueError("capture ELF differs from full-host build")
    qemu = Qemu(args.qemu, args.elf, out, False, args.wall_seconds)
    device = None
    qemu.remote.set_reg(32, image.entry)
    pack = sym["atlas_graph_pack"]["address"]
    finish = {pc for name in ("atlas_workload_main", "atlas_block_trap_report") for pc, n, w in image.instructions(name)
              if n == 4 and w == 0x10500073}
    boundaries = []

    def buffer(name):
        name = {"output": "logits"}.get(name, name)
        return sym[name]["address"], p["sizes"].get(name, 4000)

    def on_stop(pc):
        b = len(boundaries)
        directory = out / f"boundary-{b:02d}"
        directory.mkdir()
        names = list(p["segments"][b]["live_in"]) if b < 54 else []
        names += [n for n in (p["segments"][b - 1]["written"] if b else []) if n not in names]
        regions = {}
        for name in names:
            address, size = buffer(name)
            regions[name] = sha(qemu.save(address, size, directory / f"{name}.bin"))
        arena = max(k["dram_bytes"] for k in kernels)
        regions["arena"] = sha(qemu.save(ARENA, arena, directory / "arena.bin"))
        ctx = qemu.save(sym["atlas_resnet_state"]["address"], 976, directory / "context.bin")
        fields = context_fields(ctx)
        assert fields["completed"] == b and not fields["failed"] and fields["sequence"][:b] == list(range(1, b + 1))
        boundaries.append(dict(boundary=b, pc=pc, regions=regions, context=fields, launches_started=len(device.started)))
        print(json.dumps(dict(boundary=b, regions=len(regions), launches=len(device.started))), flush=True)
        if b < 54:
            assert pc == pack and len(device.started) == b
            return False
        assert pc in finish
        mailbox = qemu.save(sym["atlas_workload_mailbox"]["address"], 928, directory / "mailbox.bin")
        tohost = int.from_bytes(qemu.save(sym["tohost"]["address"], 8, directory / "tohost.bin"), "little")
        logits = read(directory / "output.bin")
        expected = image.bytes_at(sym["graph_expected"]["address"], 4000)
        assert tohost == 1 and struct.unpack_from("<5I", mailbox)[2:5] == (0, 0, 54) and logits == expected
        boundaries[-1].update(tohost=tohost, logits_sha256=sha(logits), all_logits_exact=True)
        return True

    started = time.monotonic()
    try:
        device = (model_device(qemu, args, out) if args.npu_model
                  else ReplayDevice(qemu, dict(first=0, kernels=kernels), out))
        serve(qemu, image, device, {pack, *finish}, on_stop)
        if args.npu_model:
            device.verify_bookends()
    finally:
        try:
            if device and args.npu_model:
                device.close()
        finally:
            qemu.close()
    for b, entry in enumerate(boundaries):
        if b:
            prev = boundaries[b - 1]["regions"]
            for name, digest in entry["regions"].items():
                if name in prev and name.startswith("tensor_"):
                    assert digest == prev[name], f"{name} changed between boundaries {b - 1} and {b}"
    scope = "npu_model Core device" if args.npu_model else "frozen npu_model replay device"
    receipt = dict(status="PASS_CAPTURE", scope=f"QEMU RV64 full host with {scope}; state only",
                   elf=str(args.elf), elf_sha256=sha(read(args.elf)), qemu=qemu.command, epoch=str(args.epoch),
                   epoch_sha256=sha(read(args.epoch)), wall_seconds=time.monotonic() - started,
                   kernels=[dict(index=i, model=k["model"], output_offset=k["output_offset"],
                                 dram_bytes=k["dram_bytes"], initial_sha256=sha(k["initial"]),
                                 expected_sha256=sha(k["expected"])) for i, k in enumerate(kernels)],
                   launches=device.started, boundaries=boundaries)
    write_json(out / "capture.json", receipt)
    print(json.dumps(dict(status=receipt["status"], boundaries=len(boundaries), wall=receipt["wall_seconds"])))


def check(args):
    """Run one segment ELF in QEMU with the replay device; its own checks decide PASS/FAIL."""
    out = args.out
    out.mkdir(parents=True)
    kernels = payloads(args.model_root, args.epoch, args.host)
    image = Image(read(args.elf))
    k = struct.unpack_from("<I", image.bytes_at(image.symbols["atlas_segment"]["address"], 4))[0]
    if not 0 <= k < 54:
        raise ValueError("segment index outside 0..53")
    qemu = Qemu(args.qemu, args.elf, out, True, args.wall_seconds)
    device = ReplayDevice(qemu, dict(first=k, single=True, kernels=kernels), out)
    qemu.remote.set_reg(32, image.entry)
    write, exit_ = image.symbols["atlas_segment_write"]["address"], image.symbols["atlas_segment_exit"]["address"]
    console, status = [], {}

    def on_stop(pc):
        if pc == write:
            s, n = qemu.remote.reg(10), qemu.remote.reg(11)
            console.append(qemu.remote.read(s, n).decode())
            qemu.remote.set_reg(32, qemu.remote.reg(1))
            return False
        status["exit"] = qemu.remote.reg(10)
        return True

    started = time.monotonic()
    try:
        serve(qemu, image, device, {write, exit_}, on_stop)
    finally:
        qemu.close()
    log = "".join(console)
    (out / "console.log").write_text(log)
    finals = [line for line in log.splitlines() if line.startswith(("SEG_PASS ", "SEG_FAIL ", "SEG_TRAP "))]
    passed = (status.get("exit") == 0 and len(finals) == 1 and finals[0].startswith(f"SEG_PASS segment={k} ")
              and " state_ok=1 mismatches=0 " in finals[0] and len(device.started) == 1)
    receipt = dict(status="PASS_QEMU_SEGMENT" if passed else "FAIL_QEMU_SEGMENT", segment=k, exit=status.get("exit"),
                   elf=str(args.elf), elf_sha256=sha(read(args.elf)), launches=device.started,
                   wall_seconds=time.monotonic() - started, console=log.splitlines())
    write_json(out / "receipt.json", receipt)
    print(json.dumps({key: receipt[key] for key in ("status", "segment", "exit", "wall_seconds")}))
    return 0 if passed else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("capture", "check"))
    for name in ("elf", "host", "model-root", "epoch", "qemu", "out"):
        parser.add_argument("--" + name, type=admitted_path, required=True)
    parser.add_argument("--npu-model", type=admitted_path,
                        help="capture: pinned npu_model checkout; run the live model device")
    parser.add_argument("--wall-seconds", type=int, default=21600)
    args = parser.parse_args()
    if args.wall_seconds <= 0:
        parser.error("--wall-seconds must be positive")
    if args.command == "check" and args.npu_model:
        parser.error("--npu-model applies only to capture")
    for name in ("elf", "epoch", "qemu"):
        if not getattr(args, name).is_file():
            parser.error("--" + name + " must be a file")
    for name in ("host", "model_root"):
        if not getattr(args, name).is_dir():
            parser.error("--" + name.replace("_", "-") + " must be a directory")
    if args.command == "capture":
        capture(args)
    else:
        raise SystemExit(check(args))


if __name__ == "__main__":
    if not __debug__:
        raise SystemExit("Python -O is unsupported")
    main()
