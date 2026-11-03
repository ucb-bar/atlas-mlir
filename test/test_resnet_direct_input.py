"""Direct-input bounds include the scalar sentinel and padded gather addresses."""

import os
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).absolute().parents[1] / "tools"))
from atlas_resnet_input import DirectInput, kernel_major_weights


def patches(source, channels, height, width, stride, rows):
    """Host im2col in (ky, kx, channel) K order, as 32x32 tiles in (row tile, K tile) order."""
    image = np.frombuffer(source, np.uint8)[:height * width * channels].reshape(height, width, channels)
    out_h, out_w = (height - 1) // stride + 1, (width - 1) // stride + 1
    matrix = np.zeros((rows, 9 * channels), np.uint8)
    for row in range(out_h * out_w):
        y, x = divmod(row, out_w)
        for tap in range(9):
            iy, ix = y * stride + tap // 3 - 1, x * stride + tap % 3 - 1
            if 0 <= iy < height and 0 <= ix < width:
                matrix[row, tap * channels:(tap + 1) * channels] = image[iy, ix]
    return matrix.reshape(rows // 32, 32, -1, 32).transpose(0, 2, 1, 3).tobytes()


class DirectInputBoundsTests(unittest.TestCase):
    def test_unrepresentable_exclusive_source_end_is_rejected(self):
        for offset in (2**32 - 1024, 2**32 - 992):
            source = DirectInput(offset=offset, channels=32, height=1, width=32, zero_offset=32)
            with self.subTest(offset=offset), self.assertRaisesRegex(ValueError, "source exclusive end"):
                source.validate(32, 32, output_offset=2304, output_bytes=2048)

    def test_nearest_aligned_representable_exclusive_end_is_accepted(self):
        source = DirectInput(offset=2**32 - 1056, channels=32, height=1, width=32, zero_offset=32)
        source.validate(32, 32, output_offset=2304, output_bytes=2048)
        self.assertIn(f"li x29, {2**32 - 32}", source.gather(10, 11, "boundary"))

    def test_padded_stride_two_uses_logical_end(self):
        source = DirectInput(offset=2**32 - 3328, channels=32, height=3, width=5, stride=2, zero_offset=32)
        self.assertEqual(source.output_rows, 6)
        self.assertEqual(source.source_bytes, 1024)
        source.validate(32, 32, output_offset=2304, output_bytes=2048)
        self.assertIn(f"li x29, {source.offset + 480}", source.gather(10, 11, "padded"))

    def test_padded_stride_two_address_overflow_is_rejected(self):
        source = DirectInput(offset=2**32 - 3264, channels=32, height=3, width=5, stride=2, zero_offset=32)
        with self.assertRaisesRegex(ValueError, "padded spatial-row address arithmetic"):
            source.validate(32, 32, output_offset=2304, output_bytes=2048)


class PatchInputTests(unittest.TestCase):
    def test_kxk_geometry_and_kernel_major_weights(self):
        source = DirectInput(offset=4096, channels=64, height=56, width=56, stride=1, zero_offset=0, kernel=3, padding=1)
        source.validate(3136, 576, output_offset=2**20, output_bytes=200704)
        with self.assertRaisesRegex(ValueError, "same padding"):
            DirectInput(offset=4096, channels=64, height=56, width=56, zero_offset=0, kernel=3).validate(3136, 576)
        with self.assertRaisesRegex(ValueError, "gather_block"):
            source.gather(1, 4, "tile")
        rng = np.random.default_rng(3)
        channels, n = 64, 64
        weights = rng.integers(0, 256, 9 * channels * n, dtype=np.uint8).tobytes()
        tiles = np.frombuffer(weights, np.uint8).reshape(n // 32, 9 * channels // 32, 32, 32)
        permuted = np.frombuffer(kernel_major_weights(weights, channels, 3, n), np.uint8).reshape(tiles.shape)
        for k_new in (0, 5, 63, 64, 300, 575):
            tap, channel = divmod(k_new, channels)
            k_old = channel * 9 + tap
            for column in (0, 31, 32, 63):
                self.assertEqual(permuted[column // 32, k_new // 32, k_new % 32, column % 32],
                                 tiles[column // 32, k_old // 32, k_old % 32, column % 32])

    @unittest.skipUnless(os.environ.get("ATLAS_NPU_MODEL_ROOT"), "original model fixture required")
    def test_gather_block_equals_host_patches(self):
        sys.path.insert(0, os.environ["ATLAS_NPU_MODEL_ROOT"])
        from atlas_rtl_reference import RtlReference
        base = 0x90000000
        reference = RtlReference(base)
        rng = np.random.default_rng(4)
        try:
            # Stride one with padded output rows, stride two, and a bank jump after four tiles.
            for channels, height, width, stride, row_tiles in ((32, 6, 6, 1, 2), (64, 9, 10, 2, 1)):
                rows = row_tiles * 32
                tiles = row_tiles * 9 * channels // 32
                spatial = (height * width + 31) // 32 * 32
                source = rng.integers(1, 256, spatial * channels, dtype=np.uint8).tobytes()
                # Absolute DMA addresses stand in for relocate_assembly's displacement.
                direct = DirectInput(offset=base + 64, channels=channels, height=height, width=width, stride=stride,
                                     zero_offset=base, kernel=3, padding=1)
                direct.validate(rows, 9 * channels)
                output = 64 + len(source)
                lines = direct.gather_block(0, row_tiles, 1024, 8192, "probe")
                lines += ["li x1, 1024", "li x2, %d" % (base + output), "li x3, 1024", "li x4, %d" % tiles, "store:",
                          "dma.store x2, x1, x3, ch0", "dma.wait ch0", "addi x1, x1, 256", "add x2, x2, x3",
                          "li x5, 2048", "bne x1, x5, same", "nop", "li x1, 8192", "same:", "addi x4, x4, -1",
                          "bne x4, x0, store", "nop", "ecall"]
                arena = bytes(64) + source + b"\xcd" * (tiles * 1024)
                requested = dict(load_bytes=tiles * 1024, store_bytes=tiles * 1024)
                actual, _ = reference.run("\n".join(lines) + "\n", arena, output, tiles * 1024, requested)
                with self.subTest(channels=channels, stride=stride):
                    self.assertEqual(actual, patches(source, channels, height, width, stride, rows))
        finally:
            reference.close()


if __name__ == "__main__":
    unittest.main()
