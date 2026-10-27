"""Full Smol admission binds selected code to captured data and bounded DDR."""

import copy
from pathlib import Path
import struct
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

TOOLS = Path(__file__).absolute().parents[1] / "tools"
sys.path[:0] = [str(TOOLS), str(TOOLS / "host_iss")]
import smol_full_contract as contract
from atlas_program import assembly, dma_load_bytes
from chipyard.arena import relocate, relocate_assembly


class FullSmolTest(unittest.TestCase):
    def fixture(self, reuse=False, arena=0x100000000):
        m, k, n = 32, 32, 960
        base = 0x100000000
        captured = dict(index=0, physical_mkn=[m, k, n], output_offset=m*k+k*n+256,
                        total_dram_bytes=m*k+k*n+2*m*n+512, output_rescale_exponent=-15,
                        source="baseline.S", original="baseline.bin", activation="activation.bin",
                        weights="weights.bin", expected="expected.bin", dram_base=base,
                        requested_dma=dict(load_bytes=m*k+k*n, store_bytes=2*m*n))
        source = "li x30, 1\ndma.config x30, ch0\ndma.load x1, x2, x3, ch0\necall\n"
        raw = struct.pack("<4I", 0x00100f13, 30 << 15 | 0x7f, 3 << 20 | 2 << 15 | 1 << 7 | 0x7b, 0x73)
        selected = dict(captured, source="selected.S", original="selected.bin", dram_base=arena,
                        program_policy="traffic-reuse-v1" if reuse else "baseline-v1")
        if reuse:
            source = assembly(m, k, n, captured["output_offset"], -15, traffic_reuse=True)
            selected["requested_dma"] = dict(load_bytes=dma_load_bytes(m, k, n, traffic_reuse=True), store_bytes=2*m*n)
        payloads = {"baseline.S": b"li x30, 1\ndma.config x30, ch0\ndma.load x1, x2, x3, ch0\necall\n",
                    "baseline.bin": raw,
                    "selected.S": relocate_assembly(source, base, arena).encode(),
                    "selected.bin": struct.pack("<2I", 0x13, 0x73) if reuse else relocate(raw, base, arena)}
        physical = dict(arena_base=arena, dram_base=0x80000000, dram_bytes=64 << 30)
        policy = "qualified-reuse-v1" if reuse else "captured-baseline-v1"
        bundle = SimpleNamespace(provenance=dict(placement=physical, program_policy=policy), read=payloads.__getitem__)
        data = dict(placement=physical, program_policy=policy, kernels=[selected])
        return bundle, data, [captured], payloads

    def test_captured_baseline_and_relocated_code_are_bound(self):
        for arena in (0x100000000, 0x90000000):
            bundle, data, captured, _ = self.fixture(arena=arena)
            contract.selected_kernels(bundle, data, captured)

    def test_public_reuse_code_and_dma_are_bound(self):
        for arena in (0x100000000, 0x90000000):
            bundle, data, captured, payloads = self.fixture(reuse=True, arena=arena)
            program = SimpleNamespace(assemble=lambda: [0x13, 0x73])
            contract.selected_kernels(bundle, data, captured, lambda _: program)
            contract.selected_kernels(bundle, data, captured)
            payloads["selected.S"] += b"nop\n"
            with self.assertRaisesRegex(ValueError, "reuse source"):
                contract.selected_kernels(bundle, data, captured)

    def test_wrong_reuse_binary_and_missing_assembler_fail_static_admission(self):
        bundle, data, captured, payloads = self.fixture(reuse=True)
        with self.assertRaisesRegex(ValueError, "pinned assembly"):
            contract.selected_kernels(bundle, data, captured)
        payloads["selected.bin"] = payloads["baseline.bin"]
        program = SimpleNamespace(assemble=lambda: [0x13, 0x73])
        with self.assertRaisesRegex(ValueError, "reuse binary"):
            contract.selected_kernels(bundle, data, captured, lambda _: program)

    def test_immutable_pointers_must_be_readonly_and_fully_file_backed(self):
        image = Mock()
        image.segments = [dict(address=0x80000000, file_bytes=64, memory_bytes=128, flags=4)]
        contract.readonly_span(image, 0x80000020, 32)
        with self.assertRaisesRegex(ValueError, "read-only"):
            contract.readonly_span(image, 0x80000020, 33)
        image.segments[0]["flags"] = 6
        with self.assertRaisesRegex(ValueError, "read-only"):
            contract.readonly_span(image, 0x80000000, 1)

    def test_changed_inputs_layout_policy_and_dma_fail_before_execution(self):
        bundle, data, captured, _ = self.fixture(reuse=True)
        for change in (dict(activation="replaced.bin"), dict(dram_base=True),
                       dict(output_offset=data["kernels"][0]["output_offset"]+64),
                       dict(program_policy="unqualified"), dict(requested_dma=dict(load_bytes=32, store_bytes=61440))):
            altered = copy.deepcopy(data)
            altered["kernels"][0].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                contract.selected_kernels(bundle, altered, captured)
        data["placement"]["dram_bytes"] = 1 << 31
        with self.assertRaises(ValueError):
            contract.selected_kernels(bundle, data, captured)

    def test_finite_reference_rejects_bf16_nan_and_inf(self):
        self.assertTrue(contract.finite(struct.pack("<2H", 0x3f80, 0), "uint16"))
        for word in (0x7f80, 0x7fc0, 0xff80):
            self.assertFalse(contract.finite(struct.pack("<H", word), "uint16"))
        self.assertFalse(contract.finite(struct.pack("<f", float("nan")), "float32"))


if __name__ == "__main__":
    unittest.main()
