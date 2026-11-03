"""Device 3x3 stride-two FP8 max pool: geometry admission and exact original-model execution."""

import os
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).absolute().parents[1] / "tools"))
from atlas_program_checks import check_matrix_registers
from atlas_resnet_pool import MaxPool

# Zero and every nonnegative normal FP8 E4M3 byte: the inputs a fused ReLU FP8 producer emits.
NONNEGATIVE_NORMAL = np.concatenate([[0], np.arange(8, 127)]).astype(np.uint8)


def byte_maximum(source, height, width, channels):
    padded = np.zeros((height + 2, width + 2, channels), np.uint8)
    padded[1:-1, 1:-1] = np.frombuffer(source, np.uint8).reshape(height, width, channels)
    result = np.zeros((height // 2, width // 2, channels), np.uint8)
    for ky in range(3):
        for kx in range(3):
            result = np.maximum(result, padded[ky:ky + height:2, kx:kx + width:2])
    return result.tobytes()


class MaxPoolGeometryTests(unittest.TestCase):
    def test_resnet_stem_layout(self):
        pool = MaxPool(source_offset=201216, constants_offset=1004032, output_offset=256, channels=64, height=112, width=112)
        pool.validate(1008128)
        self.assertEqual((pool.groups, pool.pairs_per_group, pool.position_lines), (7, 8, 2))
        self.assertEqual((pool.dma_load_bytes, pool.dma_store_bytes), (802816 + 4096, 200704))
        check_matrix_registers(pool.assembly(0x90000000))

    def test_shift_matrices_are_one_hot_rows(self):
        tiles = np.frombuffer(MaxPool(0, 1 << 20, 2 << 20, 64, 16, 16).constants(), np.uint8).reshape(4, 32, 32)
        rows = np.arange(32)
        self.assertTrue(np.array_equal(np.argmax(tiles[0], 1), rows))
        self.assertTrue(np.array_equal(np.argmax(tiles[1, :30], 1), rows[2:]) and not tiles[1, 30:].any())
        self.assertTrue(np.array_equal(np.argmax(tiles[2, 2:], 1), rows[:30]) and not tiles[2, :2].any())
        self.assertTrue(np.array_equal(np.argmax(tiles[3, :2], 1), rows[30:]) and not tiles[3, 2:].any())
        self.assertEqual(int((tiles != 0).sum()), 32 + 30 + 30 + 2)

    def test_rejected_geometry(self):
        for fields in (dict(channels=48, height=16, width=16), dict(channels=64, height=15, width=16),
                       dict(channels=64, height=16, width=24), dict(channels=64, height=4, width=16)):
            with self.subTest(**fields), self.assertRaises(ValueError):
                MaxPool(source_offset=0, constants_offset=1 << 20, output_offset=2 << 20, **fields).validate()
        with self.assertRaisesRegex(ValueError, "overlap"):
            MaxPool(source_offset=0, constants_offset=1024, output_offset=32768, channels=32, height=16, width=32).validate()


@unittest.skipUnless(os.environ.get("ATLAS_NPU_MODEL_ROOT"), "original model fixture required")
class MaxPoolExecutionTests(unittest.TestCase):
    def test_equals_byte_maximum(self):
        sys.path.insert(0, os.environ["ATLAS_NPU_MODEL_ROOT"])
        from atlas_rtl_reference import RtlReference
        reference = RtlReference(0x90000000)
        rng = np.random.default_rng(7)
        try:
            for height, width, channels in ((16, 16, 64), (16, 32, 32), (8, 32, 128), (16, 16, 256)):
                source = rng.choice(NONNEGATIVE_NORMAL, height * width * channels).tobytes()
                pool = MaxPool(0, len(source), len(source) + 4096 + 256, channels, height, width)
                arena = source + pool.constants() + b"\xa5" * 256 + b"\xcd" * pool.output_bytes + b"\x5a" * 256
                requested = dict(load_bytes=pool.dma_load_bytes, store_bytes=pool.dma_store_bytes)
                output, _ = reference.run(pool.assembly(0x90000000), arena, pool.output_offset, pool.output_bytes, requested)
                with self.subTest(height=height, width=width, channels=channels):
                    self.assertEqual(output, byte_maximum(source, height, width, channels))
        finally:
            reference.close()

    def test_timed_core_checks_alignment_banks_and_hazards(self):
        sys.path.insert(0, os.environ["ATLAS_NPU_MODEL_ROOT"])
        import io
        import torch
        from dataclasses import replace
        from npu_model.configs.hardware.default import DefaultHardwareConfig
        from npu_model.hardware import Core
        from npu_model.logging.logger import Logger, LoggerConfig
        from npu_model.util.converter import input_to_program
        source = np.random.default_rng(8).choice(NONNEGATIVE_NORMAL, 16 * 16 * 64).tobytes()
        pool = MaxPool(0, len(source), len(source) + 4096 + 256, 64, 16, 16)
        arena = source + pool.constants() + b"\xa5" * 256 + b"\xcd" * pool.output_bytes + b"\x5a" * 256
        config = DefaultHardwareConfig()
        config.arch_state_config = replace(config.arch_state_config, numerics="rtl", randomize_init=True, init_seed=42)
        core = Core(config, Logger(LoggerConfig(filename="/dev/null")))
        core.arch_state.write_dram(0x90000000, torch.frombuffer(bytearray(arena), dtype=torch.uint8).clone())
        core.load_program(input_to_program(io.StringIO(pool.assembly(0x90000000))))
        core.arch_state.halted = False
        while not core.is_finished() and core.cycle_count < 1_000_000:
            core.tick()
        self.assertEqual(core.arch_state.halt_reason, "ecall")
        output = core.arch_state.read_dram(0x90000000 + pool.output_offset, pool.output_bytes).numpy().tobytes()
        self.assertEqual(output, byte_maximum(source, 16, 16, 64))
        core.close()

    def test_unit_scale_round_trip_of_every_admitted_byte(self):
        sys.path.insert(0, os.environ["ATLAS_NPU_MODEL_ROOT"])
        import torch
        from npu_model.configs.numerics import RtlNumerics
        raw = torch.from_numpy(NONNEGATIVE_NORMAL.copy()).view(torch.float8_e4m3fn)
        decoded = RtlNumerics.from_fp8(raw, 127)
        self.assertTrue(torch.equal(RtlNumerics.to_fp8(decoded, 127).view(torch.uint8), raw.view(torch.uint8)))
        bits = decoded.view(torch.int16).to(torch.int32)
        self.assertTrue(bool((bits[1:] > bits[:-1]).all()), "decoded order must follow byte order")


if __name__ == "__main__":
    unittest.main()
