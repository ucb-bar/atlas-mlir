"""Execute one admitted full Tiny layer/head segment in a fresh RV64 guest."""
import argparse
import datetime
import json
import os
from pathlib import Path
import resource
import signal
import struct
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).absolute().parents[1]))
from atlas_path_guard import checked
from atlas_tooling import sha256
import atlas_host_layout as hal
from atlas_program import code_sources
from elf_image import Image, mmio_sites
from full_bundle import FullBundle
from gdb_remote import Remote
from model_device import Device
from process_cleanup import reap_qemu
from result_lines import verify as verify_uart
from segment_uart import Transport
from run_model_tile import configure_model, model_binding
import tiny_segment_contract as contract


def execute(args):
    out = checked(args.out)
    out.mkdir(parents=True, exist_ok=False)
    started, deadline = time.monotonic(), time.monotonic() + args.wall_seconds
    receipt = dict(schema="atlas.segment_execution.v1", status="FAILED", segmented=True, continuous=False,
        fresh_device=False, host_elf_executed=False, source_bookends_equal=False, hardware_execution=False,
        segment_index=args.segment_index, command=[sys.executable, *sys.argv], kernels=[], captured_replies=False)
    remote = process = device = uart = None
    pins, previous_cwd = {}, Path.cwd()
    def progress(stage):
        state = dict(stage=stage, status=receipt["status"] if stage == "finished" else "RUNNING", runner_pid=os.getpid(),
            launch_range=receipt.get("launch_range"), fresh_device=receipt.get("fresh_device", False),
            qemu_pid=process.pid if process else None, segment_index=args.segment_index,
            updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            elapsed_seconds=time.monotonic()-started, completed_launches=len(device.completed) if device else 0,
            completed_device_cycles=sum(k["cycles"] for k in device.completed) if device else 0,
            runner_max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        temporary = out / "run-state.tmp"
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(out / "run-state.json")
    progress("static-admission")
    def interrupted(signum, frame):
        raise InterruptedError("segment interrupted by signal " + str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    try:
        bundle_root = checked(args.bundle_root)
        elf, qemu, model_root = (checked(p) for p in (bundle_root / "host/host.elf", args.qemu, args.model_root))
        Transport.pin(pins)
        for name, digest in code_sources(traffic_reuse=True).items():
            path = checked(Path(__file__).absolute().parents[1] / name)
            pins[str(path)] = digest
        for module in (contract, sys.modules["process_cleanup"], sys.modules["chipyard.arena"], sys.modules["segment_boundary"], sys.modules["model_device"],
                       sys.modules["elf_image"], sys.modules["gdb_remote"], sys.modules["tiny_full_contract"],
                       sys.modules["full_bundle"], sys.modules["bundle"], sys.modules["result_lines"],
                       sys.modules["run_model_tile"], sys.modules["atlas_host_layout"], sys.modules["atlas_path_guard"],
                       sys.modules["atlas_tooling"]):
            path = checked(Path(module.__file__).absolute())
            pins[str(path)] = sha256(path)
        for path in (checked(__file__), elf, qemu, checked(hal.HEADER)):
            pins[str(path)] = sha256(path)
        bundle = FullBundle(bundle_root, "tiny-full")
        image = Image(bundle.read("host/host.elf"))
        # Assembler/profile and the entire immutable chain are admitted before Core creation.
        uart = Transport(bundle.build["uart"], getattr(args, "sifive_uart_bridge", False), image, out)
        before = model_binding(model_root)
        _, _, assembler, _ = configure_model(model_root)
        binding = model_binding(model_root)
        assert binding["tracked_source_diff_exit"] == 0 and before["tracked_source_diff_exit"] == 0
        assert binding["revision"] == before["revision"] and binding["curve_sha256"] == before["curve_sha256"]
        profile = dict(revision=binding["revision"], backend="curve", curve_sha256=binding["curve_sha256"])
        chain, data = contract.validate_bundle(bundle, image, assembler, profile)
        assert all(sha256(checked(path)) == h for path, h in pins.items()), "admission helper changed"
        assert model_binding(model_root) == binding, "admission model source changed"
        assert 0 <= args.segment_index < len(chain.boundaries) - 1
        launch_range = tuple(b["launch_cursor"] for b in chain.boundaries[args.segment_index:args.segment_index+2])
        count = launch_range[1] - launch_range[0]
        ready, stops, success = contract.runtime_sites(image)
        sites = mmio_sites(image)
        qemu_low, qemu_high = 0x80000000, 0x80000000 + (8 << 30)
        assert all(qemu_low <= s["address"] < s["address"] + s["memory_bytes"] <= qemu_high
                   for s in image.segments), "ELF exceeds actual QEMU virt8GiB DRAM"
        assert qemu_low <= chain.placement["arena_base"] <= qemu_high - max(
            item["total_dram_bytes"] for item in data["kernels"]), "arena exceeds actual QEMU DRAM"
        if args.corrupt_region:
            assert args.corrupt_region in ("hidden_0", "hidden_1")
            assert 0 <= args.corrupt_offset < chain.layout[args.corrupt_region]["bytes"]
        receipt.update(launch_range=list(launch_range), elf_sha256=bundle.sums["host/host.elf"],
            contract_sha256=bundle.sums["contract.json"], model_profile=profile,
            chain_sha256sums_sha256=chain.checksums_sha256, full_static_admission=True,
            live_regions=chain.data["live_regions"], model_binding=binding)
        serial = uart.prepare()
        command = [str(qemu), "-machine", "virt", "-m", "8G", "-smp", "1", "-accel", "tcg,thread=single",
            "-bios", "none", "-nodefaults", "-nographic", "-monitor", "none", "-serial", "file:"+serial,
            "-S", "-gdb", "unix:qemu.sock,server=on,wait=off", "-kernel", str(elf)]
        receipt["qemu_command"] = command
        progress("static-admitted")
        with (out / "qemu.log").open("wb") as qlog:
            process = subprocess.Popen(command, cwd=out, stdout=qlog, stderr=subprocess.STDOUT, start_new_session=True)
        progress("qemu-started")
        startup = min(deadline, time.monotonic() + 30)
        while not (out / "qemu.sock").exists():
            if process.poll() is not None or time.monotonic() >= startup:
                raise RuntimeError("QEMU segment startup failed")
            time.sleep(0.05)
        os.chdir(out)
        remote = Remote("qemu.sock", deadline=deadline, timeout=min(args.wall_seconds, 1800))
        assert remote.request("?")[:1] in (b"S", b"T")
        remote.set_reg(32, image.entry)
        for pc, length in ready.items():
            remote.breakpoint(pc, length)
        for pc, length in stops.items():
            remote.breakpoint(pc, length)
        for pc, access in sites.items():
            assert remote.read(pc, access["length"]) == image.bytes_at(pc, access["length"])
            remote.breakpoint(pc, access["length"])
        uart.register(remote, image, set(ready) | set(stops) | set(sites))
        while True:
            pc, _ = remote.resume()
            if pc in ready:
                break
            assert uart.handle(remote, pc), "unexpected guest stop before initialized segment entry"
        receipt["host_elf_executed"] = True
        assert pc in ready, "guest did not reach normal initialized segment entry"
        assert remote.reg(2) % 16 == 0 and image.symbols["__stack_bottom"]["address"] <= remote.reg(2) < image.symbols["__stack_top"]["address"]
        progress("restoring-complete-live-state")
        chain.restore(args.segment_index, remote.write)
        # Confirm complete restoration before applying the explicit negative control.
        for name, member in chain.boundaries[args.segment_index]["regions"].items():
            region = chain.layout[name]
            assert remote.read(region["address"], region["bytes"]) == chain.read(member), "segment restore differs"
        progress("live-state-restored")
        if args.corrupt_region:
            address = chain.layout[args.corrupt_region]["address"] + args.corrupt_offset
            original = remote.read(address, 1)
            remote.write(address, bytes([original[0] ^ 1]))
            receipt["negative_control"] = dict(region=args.corrupt_region, byte_offset=args.corrupt_offset, xor=1)
        remote.write(image.symbols["tiny_full_segment_request"]["address"], struct.pack("<Q", args.segment_index))
        device = Device(remote, out, deadline, bundle, model_root, cycle_cap=contract.full.CYCLE_CAP,
                        launch_range=launch_range)
        receipt["fresh_device"] = True
        receipt["memory_backend"] = device.memory_backend
        binding = device.binding
        receipt["model_binding"] = binding
        progress("device-ready")
        remote.set_reg(32, pc + ready[pc])
        access_cap = 100000 + sum(len(words)//4 + 64 for _, words, _ in device.programs.values())
        accesses = 0
        while time.monotonic() < deadline:
            pc, stop = remote.resume()
            if pc in stops:
                break
            if uart.handle(remote, pc):
                continue
            assert pc in sites, ("unexpected segment guest stop", pc, stop)
            access = sites[pc]
            address = remote.reg(access["base"]) + access["offset"]
            if access["store"]:
                value = remote.reg(access["reg"]) & 0xffffffff
                starting = address == hal.ATLAS_CSR_BASE + hal.ATLAS_CSR_EXEC_CONTROL and value == 1
                if starting:
                    progress("executing-global-kernel-" + str(launch_range[0] + len(device.completed)))
                device.write32(address, value)
                if starting:
                    progress("kernel-completed")
            else:
                value = device.read32(address)
                if access["signed"] and value & 0x80000000:
                    value |= 0xffffffff00000000
                remote.set_reg(access["reg"], value)
            remote.set_reg(32, pc + access["length"])
            accesses += 1
            assert accesses <= access_cap, "segment MMIO access cap"
        else:
            raise TimeoutError("segment absolute wall cap")
        receipt.update(kernels=device.completed, mmio_accesses=accesses)
        assert pc in success, "guest trapped during segment"
        end_control = struct.unpack("<8Q", chain.read(chain.boundaries[args.segment_index+1]["regions"]["control"]))
        mailbox = struct.unpack("<16Q", remote.read(image.symbols["tiny_full_mailbox"]["address"], 128))
        assert mailbox == (contract.full.MAGIC, 2, launch_range[1], launch_range[1]-1, 0, 0,
            int(args.segment_index // 23 > 0), args.segment_index+1, end_control[2], end_control[4],
            0, 0, 0, 0, 0, 0), "segment actual mailbox completion"
        assert remote.read(image.symbols["tohost"]["address"], 8) == struct.pack("<Q", 1)
        assert remote.read(image.symbols["fromhost"]["address"], 8) == bytes(8)
        assert len(device.completed) == device.launch_count == count
        assert [k["index"] for k in device.completed] == list(range(*launch_range))
        for summary in device.completed:
            item = dict(data["kernels"][summary["index"]])
            item.update(program=bundle.read(item["original"]), expected_sha256=bundle.sums[item["expected"]])
            contract.verify_kernel(summary, item, summary["index"])
        # Mutable cursors/state are checked in addition to every serialized semantic byte.
        state = struct.unpack("<5QII", remote.read(image.symbols["tiny_full_state_global"]["address"], 48))
        initial = struct.unpack("<5QII", image.bytes_at(image.symbols["tiny_full_state_global"]["address"], 48))
        end_control = struct.unpack("<8Q", chain.read(chain.boundaries[args.segment_index+1]["regions"]["control"]))
        assert state[:4] == initial[:4] and state[4:] == end_control[2:5]
        context = struct.unpack("<9Q", remote.read(image.symbols["tiny_full_projection_state"]["address"], 72))
        original = struct.unpack("<9Q", image.bytes_at(image.symbols["tiny_full_projection_state"]["address"], 72))
        assert context[2] == launch_range[1] and context[7] == end_control[6]
        assert all(context[i] == original[i] for i in (0, 1, 3, 4, 5, 6, 8))
        host = remote.read(image.symbols["tiny_full_host"]["address"], 112)
        assert struct.unpack_from("<QQ", host, 72) == (0, 0)
        assert struct.unpack_from("<6I", host, 88)[1:3] == (0, 0)
        assert struct.unpack_from("<I", host, 108)[0] == launch_range[1]
        compared = chain.compare(args.segment_index, remote.read)
        receipt["boundaries"] = {row["name"]: row["sha256"] for row in compared}
        receipt["boundary_comparisons"] = compared
        uart_raw = (out / "uart.log").read_bytes()
        receipt["uart"] = verify_uart(uart_raw, "tiny-segment", count, progress_launches=[count])
        bundle.verify_bookends()
        chain.verify_bookends()
        device.verify_bookends()
        assert all(sha256(checked(path)) == h for path, h in pins.items())
        assert model_binding(model_root) == binding, "model source bookends differ"
        assert time.monotonic() < deadline
        receipt.update(status="PASS_SEGMENT_WITH_MODEL_DEVICE", source_bookends_equal=True, inputs_sha256=pins,
            measured_kernel_cycles=sum(k["cycles"] for k in device.completed))
    except Exception as error:
        receipt["error"] = dict(type=type(error).__name__, message=str(error))
        if device:
            receipt["kernels"] = device.completed
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        receipt["runner_max_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if device and hasattr(device, "core"):
            receipt["actual_core_cycles"] = device.core.cycle_count
        receipt.setdefault("measured_kernel_cycles", sum(k["cycles"] for k in receipt["kernels"]))
        progress("finished")
        if process:
            try:
                status = Path(f"/proc/{process.pid}/status").read_text()
                receipt["qemu_peak_rss_kib"] = int(next(l.split()[1] for l in status.splitlines() if l.startswith("VmHWM:")))
            except (FileNotFoundError, StopIteration):
                pass
        if process:
            try:
                receipt["qemu_child_reaped"] = reap_qemu(process)
            except Exception as error:
                receipt.setdefault("cleanup_errors", []).append(str(error))
                receipt["status"] = "FAILED"
        for obj in (remote, device):
            if obj:
                try:
                    obj.close()
                except Exception as error:
                    receipt.setdefault("cleanup_errors", []).append(str(error))
                    receipt["status"] = "FAILED"
        os.chdir(previous_cwd)
        if uart:
            receipt["uart_transport"] = uart.summary()
            try:
                uart.close()
            except Exception as error:
                receipt.setdefault("cleanup_errors", []).append(str(error))
                receipt["status"] = "FAILED"
        receipt["wall_seconds"] = time.monotonic() - started
        (out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({k: receipt[k] for k in ("status", "error") if k in receipt}), flush=True)
    return 0 if receipt["status"].startswith("PASS_") else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle-root", "model-root", "qemu", "out"):
        parser.add_argument("--" + name, type=checked, required=True)
    parser.add_argument("--segment-index", type=int, required=True, choices=range(69))
    parser.add_argument("--wall-seconds", type=int, required=True)
    parser.add_argument("--sifive-uart-bridge", action="store_true")
    parser.add_argument("--corrupt-region", choices=("hidden_0", "hidden_1"))
    parser.add_argument("--corrupt-offset", type=int, default=1)
    args = parser.parse_args()
    if not 0 < args.wall_seconds <= 43200:
        parser.error("explicit segment wall cap must be1..43200seconds")
    raise SystemExit(execute(args))


if __name__ == "__main__":
    if not __debug__:
        raise SystemExit("Python -O is unsupported")
    main()
