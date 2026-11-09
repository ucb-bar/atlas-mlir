"""Every raw PACK scale byte, including emitted programs on RTL model numerics.

These are mandatory portable model checks, without an opt-in execution flag.
Scale 255 aliases 254 in the VPU datapath; it is not a NaN scale here.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile
import time
import unittest

from test_guarded_pack_exactness import pack_oracle
from test_pack_roundtrip import source
from test_virtual_lowering import BIN, ROOT, emitted, lower, run

_EVIDENCE: dict[str, object] = {}


def record_evidence(section: str, evidence: dict[str, object]) -> None:
    _EVIDENCE[section] = evidence
    if receipt := os.environ.get("ATLAS_PACK_SCALE_RECEIPT"):
        Path(receipt).write_text(json.dumps(_EVIDENCE, indent=2) + "\n")


def panel_words(scale: int) -> list[int]:
    """Logical row-major BF16 lanes, with scale-relative boundary coverage."""
    cases = [0, 0x8000, 1, 0x807f, 0x007f, 0x8001,
             0x7f80, 0xff80, 0x7fc0, 0xffc0, 0x7f81, 0xffff,
             0x0080, 0x8080, 0x7f7f, 0xff7f, 0x3f80, 0xbf80]
    # Below/at the flush boundary, ties to even on each parity, significand
    # carry, and rounding to 480 (which must clamp to finite E4M3 0x7e).
    for power in (-8, -7, -6, -1, 0, 1, 7, 8, 9):
        exponent = min(scale, 254) + power
        if 1 <= exponent <= 254:
            for fraction in (0, 1, 7, 8, 9, 23, 24, 96, 111, 112, 119, 120, 127):
                bits = (exponent << 7) | fraction
                cases.extend((bits, bits | 0x8000))
    # Include every directed case and mix different values across all rows,
    # columns, and the two physical BF16 half-tile registers.
    return [cases[lane] if lane < len(cases) else cases[(lane * 37 + lane // 32) % len(cases)]
            for lane in range(1024)]


def bf16_strips(words: list[int]) -> bytes:
    return b"".join(struct.pack("<H", words[row * 32 + column])
                    for half in (0, 16) for row in range(32)
                    for column in range(half, half + 16))


class PackScaleDomainTest(unittest.TestCase):
    def test_all_bf16_encodings_at_all_256_raw_scale_bytes(self) -> None:
        import torch
        from atlas_mlir_native.pack_lane import pack_bf16_lane
        from npu_model.configs.numerics import RtlNumerics

        torch.set_num_threads(1)
        words = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
        began = time.monotonic()
        rows = []
        for scale in range(256):
            with self.subTest(scale=scale):
                expected = bytes(pack_oracle(bits, scale) for bits in range(65536))
                actual = bytes(RtlNumerics.to_fp8(words, scale).view(torch.uint8).tolist())
                selected = bytes(pack_bf16_lane(bits, scale) for bits in range(65536))
                for name, observed in (("actual model VPU converter", actual),
                                       ("selected scalar reference", selected)):
                    if observed != expected:
                        lane = next(i for i, (a, b) in enumerate(zip(observed, expected)) if a != b)
                        self.fail(f"{name}: scale={scale}, bf16=0x{lane:04x}, "
                                  f"actual=0x{observed[lane]:02x}, expected=0x{expected[lane]:02x}")
                rows.append({"scale": scale, "words": 65536,
                             "output_sha256": hashlib.sha256(actual).hexdigest()})
        self.assertEqual(len(rows), 256)
        record_evidence("converters", {"status": "PASS", "bf16_encodings": 65536,
            "scales": 256, "word_scale_pairs": 65536 * 256,
            "model_comparisons": 65536 * 256, "scalar_reference_comparisons": 65536 * 256,
            "elapsed_seconds": time.monotonic() - began, "per_scale": rows})

    def test_emitted_scalar_and_default_programs_execute_every_raw_scale(self) -> None:
        import torch
        from npu_model.configs import isa_definition as isa
        from npu_model.configs.hardware.default import DefaultHardwareConfig
        from npu_model.configs.numerics import RtlNumerics
        from npu_model.logging import LoggerConfig
        from npu_model.simulation import Simulation
        from npu_model.software.program import InstantiableProgram
        from tools.check_program_decode import decode_binary
        from tools.model_instructions import model_instruction

        self.assertTrue((BIN / "atlas-opt").is_file(), "build atlas-opt first")
        self.assertTrue((BIN / "atlas-emit").is_file(), "build atlas-emit first")
        torch.set_num_threads(1)
        inventory = json.loads((ROOT / "docs/selected-variant-inventory.json").read_text())
        output = 0x10000
        initial = bytearray((offset * 17 + 43) % 256 for offset in range(output + 1024 + 64))
        config = DefaultHardwareConfig()
        config.arch_state_config = dataclasses.replace(config.arch_state_config, numerics="rtl")
        began = time.monotonic()
        rows = []
        with tempfile.TemporaryDirectory(prefix="atlas-pack-scale-domain-") as temporary:
            # Reuse one trace path, keeping disk use bounded across 512 runs.
            trace = str(Path(temporary) / "execution.json")
            for scale in range(256):
                words = panel_words(scale)
                initial[:2048] = bf16_strips(words)
                expected = bytearray(initial)
                expected[output:output + 1024] = bytes(pack_oracle(bits, scale) for bits in words)
                for route in ("scalar", "default"):
                    with self.subTest(scale=scale, route=route):
                        if route == "scalar":
                            machine = lower(source(scale))
                            self.assertIn('"atlas.scalar_store"', machine)
                            self.assertIn('"atlas.branch"', machine)
                            self.assertEqual(machine.count('"atlas.vpu_pack"'), 1)
                        else:
                            compiled = run("atlas-opt", source(scale), "--atlas-compile")
                            self.assertEqual(compiled.returncode, 0, compiled.stderr)
                            machine = compiled.stdout
                            self.assertNotIn('"atlas.scalar_store"', machine)
                            self.assertEqual(machine.count('"atlas.vpu_pack"'), 2)
                        binary = b"".join(word.to_bytes(4, "little") for word in emitted(machine))
                        decoded = decode_binary(inventory, binary)
                        program = InstantiableProgram([
                            model_instruction(row, inventory, isa) for row in decoded["instructions"]])
                        program.dram_base = 0x90000000
                        program.memory_regions = [(0, torch.tensor(list(initial), dtype=torch.uint8))]
                        simulation = Simulation(config, LoggerConfig(filename=trace), program, verbose=False)
                        try:
                            state = simulation.core.arch_state
                            self.assertIsInstance(state.math, RtlNumerics)
                            state.vmem.fill_(0xcd)
                            simulation.run(max_cycles=100000)
                            self.assertTrue(state.halted, f"scale={scale}, route={route}")
                            self.assertEqual(simulation.get_stats().runtime_errors, 0)
                            # Whole initialized DRAM comparison binds row/column
                            # layout and proves input and output guards survive.
                            observed = bytes(state.read_dram(program.dram_base, len(initial)).tolist())
                            if observed != bytes(expected):
                                offset = next(i for i, (a, b) in enumerate(zip(observed, expected)) if a != b)
                                self.fail(f"DRAM mismatch at 0x{program.dram_base + offset:08x}: "
                                          f"actual=0x{observed[offset]:02x}, expected=0x{expected[offset]:02x}")
                            rows.append({"scale": scale, "route": route,
                                "cycles": simulation.get_stats().cycles,
                                "program_words": len(binary) // 4,
                                "program_sha256": hashlib.sha256(binary).hexdigest(),
                                "output_sha256": hashlib.sha256(observed[output:output + 1024]).hexdigest(),
                                "checked_dram_bytes": len(initial)})
                        finally:
                            simulation.close()
        self.assertEqual(len(rows), 512)
        record_evidence("programs", {"status": "PASS", "scales": 256, "programs": 512,
            "lanes_per_program": 1024, "model": "npu_model", "numerics": "rtl",
            "elapsed_seconds": time.monotonic() - began, "per_scale_route": rows})

    def test_out_of_byte_range_scales_are_rejected(self) -> None:
        from atlas_mlir_native.pack_lane import pack_bf16_lane

        for scale in (-1, 256):
            for explicit in (False, True):
                with self.subTest(scale=scale, explicit=explicit):
                    checked = run("atlas-opt", source(scale, explicit=explicit), "--atlas-compile")
                    self.assertNotEqual(checked.returncode, 0)
                    self.assertTrue(checked.stderr)
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                pack_bf16_lane(0x3f80, scale)


if __name__ == "__main__":
    unittest.main()
