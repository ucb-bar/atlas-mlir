"""Focused full-model image admission and live staging boundary controls."""

import contextlib
import hashlib
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import Mock

TOOLS = Path(__file__).absolute().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(TOOLS / "host_iss"))
from full_bundle import FullBundle
from model_device import Device
from run import arguments, mmio_access_cap


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


class FullBundleTest(unittest.TestCase):
    def fixture(self, mutate=None):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        activation, weights, expected = b"a" * 1024, b"w" * 1024, b"e" * 2048
        initial = activation + weights + b"\xa5" * 256 + b"\xcd" * 2048 + b"\x5a" * 256
        descriptor = dict(source="assets/program.S", original="assets/program.bin",
                          activation="assets/a.bin", weights="assets/w.bin", expected="assets/out.bin",
                          physical_mkn=[32, 32, 32], initial_sha256=sha(initial),
                          output_offset=2304, total_dram_bytes=4608, dram_base=0x100000000,
                          requested_dma=dict(load_bytes=2048, store_bytes=2048))
        provenance = dict(schema="atlas.full_bundle.v1", workload="tiny-full", kernel_count=2,
                          kernels=[descriptor, dict(descriptor)], contract="contract.json",
                          contract_sha256=sha(b"{}"))
        if mutate is not None:
            mutate(provenance)
        files = {"host/host.elf": b"bound elf", "host/build.json": b"{}", "contract.json": b"{}",
                 "provenance.json": json.dumps(provenance).encode(),
                 "assets/program.S": b"ecall\n", "assets/program.bin": struct.pack("<I", 0x73),
                 "assets/a.bin": activation, "assets/w.bin": weights, "assets/out.bin": expected}
        for name, raw in files.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        (root / "SHA256SUMS").write_text("".join(sha(raw) + "  " + name + "\n"
                                                    for name, raw in files.items()))
        return root, initial

    def test_shared_weights_reconstruct_each_exact_image(self):
        root, initial = self.fixture()
        bundle = FullBundle(root, "tiny-full")
        self.assertEqual(bundle.initial_dram(0), initial)
        self.assertEqual(bundle.initial_dram(1), initial)
        paths = bundle.kernel_paths(0)
        paths["physical_mkn"][0] = 64
        paths["requested_dma"]["store_bytes"] = 0
        self.assertEqual(bundle.kernel_paths(0)["physical_mkn"], [32, 32, 32])
        self.assertEqual(bundle.kernel_paths(0)["requested_dma"]["store_bytes"], 2048)
        self.assertEqual(len([name for name in bundle.sums if name.endswith("w.bin")]), 1)

    def test_changed_shared_weight_and_image_hash_rejected(self):
        root, _ = self.fixture()
        bundle = FullBundle(root, "tiny-full")
        (root / "assets/w.bin").write_bytes(b"x" * 1024)
        with self.assertRaisesRegex(ValueError, "changed bundle payload"):
            bundle.initial_dram(1)
        root, _ = self.fixture(lambda value: value["kernels"][0].update(initial_sha256="0" * 64))
        with self.assertRaisesRegex(ValueError, "initial image"):
            FullBundle(root, "tiny-full")

    def test_geometry_and_counts_rejected_before_image_allocation(self):
        mutations = [lambda p: p.update(kernel_count=20001),
                     lambda p: p.update(kernel_count=True),
                     lambda p: p.update(contract_sha256="0" * 64),
                     lambda p: p["kernels"][0].update(physical_mkn=[True, 32, 32]),
                     lambda p: p["kernels"][0].update(physical_mkn=[32, 31, 32]),
                     lambda p: p["kernels"][0].update(total_dram_bytes=64 * 1024 * 1024 + 32),
                     lambda p: p["kernels"][0].update(output_offset=2048),
                     lambda p: p["kernels"][0].update(weights="unlisted.bin"),
                     lambda p: p["kernels"][0].update(requested_dma=dict(load_bytes=1, store_bytes=1))]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                root, _ = self.fixture(mutate)
                with self.assertRaises(ValueError):
                    FullBundle(root, "tiny-full")

    def test_device_compares_real_guest_staging_to_reconstructed_image(self):
        root, initial = self.fixture()
        bundle = FullBundle(root, "tiny-full")
        device = Device.__new__(Device)
        device.bundle, device.completed, device.kernel_count, device.pins = bundle, [], 2, {}
        device.core = Mock()
        device.core.exus = []
        device.core.arch_state.halted = True
        device.core.arch_state.check_flag.return_value = False
        device.core.arch_state.read_csrf.side_effect = lambda address: int(address == 0xC11)
        device.programs = [(None, bytes(4), bundle.kernel_paths(0))]
        device.imem, device.written = bytearray(4), {0}
        device.remote = Mock()
        device.remote.read.return_value = b"x" + initial[1:]
        # This must fail before ticking any device or substituting captured data.
        with self.assertRaisesRegex(AssertionError, "host staging differs"):
            device.start()
        device.remote.read.assert_called_once_with(0x100000000, len(initial))
        device.core.tick.assert_not_called()
        self.assertEqual(device.completed, [])


class FullBudgetTest(unittest.TestCase):
    def cli(self, *extra):
        return ["--out", "out", "--elf", "elf", "--bundle-root", "bundle",
                "--model-root", "model", "--qemu", "qemu", *extra]

    def test_full_budget_explicit_and_bounded_old_budget_unchanged(self):
        self.assertEqual(arguments(self.cli()).wall_seconds, 21600)
        self.assertEqual(arguments(self.cli("--workload", "tiny-full", "--wall-seconds", "1209600")).wall_seconds,
                         1209600)
        for extra in [("--workload", "tiny-full"),
                      ("--workload", "tiny", "--wall-seconds", "86401"),
                      ("--workload", "smol-full", "--wall-seconds", "2592001"),
                      ("--wall-seconds", "0"), ("--kernel-cycle-cap", "0")]:
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                arguments(self.cli(*extra))

    def test_full_mmio_budget_uses_actual_admitted_program_sizes(self):
        device = Mock()
        device.programs = [(None, bytes(20000), {})] * 840
        self.assertEqual(mmio_access_cap(device, "tiny-full"), 100000 + 840 * 5064)
        self.assertEqual(mmio_access_cap(device, "tiny"), 100000)


if __name__ == "__main__":
    unittest.main()
