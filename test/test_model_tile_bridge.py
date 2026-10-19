"""Checks for the source-bound MLIR C-interface tile binding."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from atlas_model_tile_bridge import _source_file, render_wrapper


class ModelTileBridgeTest(unittest.TestCase):
    def test_source_window_pads_k_and_crops_result(self) -> None:
        compiler = shutil.which("cc")
        if compiler is None:
            self.skipTest("a C compiler is required for the source-window test")
        temporary_root = Path(os.environ.get("TMPDIR", ROOT / "out"))
        temporary_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root) as temporary:
            directory = Path(temporary)
            source = directory / "window.c"
            source.write_text(r'''
#include "atlas_model_tile.h"
#include <assert.h>
#include <limits.h>
#include <stdlib.h>
static unsigned writes, packs;
static unsigned starts, status_reads;
static int pack_lhs(const float *source, uint8_t *destination) {
  assert(source[29] == 1 && source[30] == 2 && source[31] == 3);
  assert(source[32 + 29] == 4 && source[32 + 30] == 5 && source[32 + 31] == 6);
  assert(source[28] == 0 && source[2 * 32 + 31] == 0);
  for (unsigned i = 0; i < 1024; ++i) destination[i] = 0;
  ++packs;
  return 0;
}
static int pack_rhs(const float *source, uint8_t *destination) {
  assert(source[29] == 10 && source[30] == 20 && source[31] == 30);
  assert(source[32 + 29] == 40 && source[32 + 30] == 50 && source[32 + 31] == 60);
  assert(source[28] == 0 && source[2 * 32 + 31] == 0);
  for (unsigned i = 0; i < 1024; ++i) destination[i] = 0;
  ++packs;
  return 0;
}
static int unpack(const volatile uint8_t *source, float *destination) {
  (void)source;
  for (unsigned row = 0; row < 32; ++row)
    for (unsigned col = 0; col < 32; ++col)
      destination[row * 32 + col] = (float)(row * 100 + col);
  return 0;
}
static uint32_t read32(void *ctx, uintptr_t address) {
  (void)ctx; (void)address;
  if (starts && status_reads++ == 0) return 1;
  return 5;
}
static void write32(void *ctx, uintptr_t address, uint32_t word) {
  (void)ctx; (void)word;
  if (address == 0x40018u) { ++starts; status_reads = 0; }
}
static int write_bytes(void *ctx, uintptr_t address, const uint8_t *data, size_t bytes) {
  (void)ctx; (void)address; (void)data; assert(bytes == 1024); ++writes; return 0;
}
static int read_bytes(void *ctx, uintptr_t address, uint8_t *data, size_t bytes) {
  (void)ctx; (void)address; assert(bytes == 2048);
  for (size_t i = 0; i < bytes; ++i) data[i] = 0;
  return 0;
}
int main(void) {
  atlas_host host;
  atlas_host_map map = {0x20000u, 0x40000u, 4u};
  atlas_host_mmio mmio = {read32, write32, 0};
  atlas_host_memory memory = {1, 0, 0};
  assert(atlas_host_init(&host, map, mmio, memory) == ATLAS_HOST_OK);
  uint32_t program[] = {0x00000073u};
  _Alignas(float) uint8_t scratch[ATLAS_MODEL_TILE_SCRATCH_BYTES];
  atlas_model_tile tile = {0};
  tile.host = &host;
  tile.io = (atlas_tile_memory_io){write_bytes, read_bytes, 0};
  tile.program_words = program;
  tile.program_word_count = 1;
  tile.input_addresses[0] = 0x90000000u;
  tile.input_addresses[1] = 0x90000400u;
  tile.output_address = 0x90000800u;
  tile.constant_operand = 2;
  tile.transpose_input[1] = 1;
  tile.pack[0] = pack_lhs;
  tile.pack[1] = pack_rhs;
  tile.unpack = unpack;
  tile.max_polls = 2;
  tile.scratch = scratch;
  tile.scratch_bytes = sizeof scratch;
  float lhs_data[10] = {0, 1, 2, 3, 0, 0, 4, 5, 6, 0};
  float rhs_data[14] = {0, 0, 10, 40, 0, 0, 20, 50, 0, 0, 30, 60, 0, 0};
  atlas_f32_memref2 lhs = {lhs_data, lhs_data, 1, {2, 3}, {5, 1}};
  atlas_f32_memref2 rhs = {rhs_data, rhs_data, 2, {3, 2}, {4, 1}};
  atlas_f32_memref2 result = {0};
  atlas_host_result device = ATLAS_HOST_INVALID;
  assert(atlas_model_tile_run_window(&tile, &lhs, &rhs, &result, 2, 2, 3,
                                     &device) == ATLAS_MODEL_TILE_OK);
  assert(device == ATLAS_HOST_OK && writes == 2 && packs == 2);
  assert(result.sizes[0] == 2 && result.sizes[1] == 2);
  assert(result.aligned[0] == 0 && result.aligned[1] == 1);
  assert(result.aligned[2] == 100 && result.aligned[3] == 101);
  free(result.allocated);
  lhs.strides[0] = INT64_MAX;
  result.allocated = (float *)0x1;
  assert(atlas_model_tile_run_window(&tile, &lhs, &rhs, &result, 2, 2, 3,
                                     &device) == ATLAS_MODEL_TILE_INVALID);
  assert(result.allocated == (float *)0x1 && writes == 2);
  return 0;
}
''')
            binary = directory / "window"
            subprocess.run(
                [compiler, "-O2", "-Wall", "-Wextra", "-Werror", "-I", str(ROOT / "runtime"),
                 str(source), str(ROOT / "runtime/atlas_model_tile.c"),
                 str(ROOT / "runtime/atlas_host.c"), "-o", str(binary)],
                check=True, capture_output=True, text=True,
            )
            completed = subprocess.run([str(binary)], check=False, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_checked_tile_reports_input_domain_before_device_launch(self) -> None:
        compiler = shutil.which("cc")
        if compiler is None:
            self.skipTest("a C compiler is required for the wrapper execution test")
        temporary_root = Path(os.environ.get("TMPDIR", ROOT / "out"))
        temporary_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root) as temporary:
            directory = Path(temporary)
            header, source, _ = render_wrapper("domain_probe", 2, (0, 1))
            suffix = hashlib.sha256(b"domain_probe").hexdigest()[:12]
            (directory / "atlas_model_tile_wrapper.h").write_text(header)
            (directory / "atlas_model_tile_wrapper.c").write_text(source)
            (directory / "atlas_program.h").write_text(
                '#include "atlas_host.h"\n#include <stdint.h>\n'
                '#define ATLAS_PROGRAM_BITSTREAM_BOUND 0\n'
                '#define ATLAS_PROGRAM_WORD_COUNT 1u\n'
                'static const uint32_t atlas_program_words[] = {0x00000073u};\n'
                'static const atlas_host_map atlas_program_map = {0x20000u, 0x40000u, 4u};\n'
            )
            (directory / "atlas_captured_data.h").write_text(
                '#include "atlas_model_tile.h"\n'
                '#define ATLAS_CAPTURED_INPUT_ADDRESS_0 0x90000000u\n'
                '#define ATLAS_CAPTURED_INPUT_ADDRESS_1 0x90000400u\n'
                '#define ATLAS_CAPTURED_OUTPUT_ADDRESS 0x90000800u\n'
                'static inline int atlas_pack_runtime_input_0_f32(const float *src, uint8_t *dst) '
                '{ (void)src; (void)dst; return 2; }\n'
                'static inline int atlas_pack_runtime_input_1_f32(const float *src, uint8_t *dst) '
                '{ (void)src; (void)dst; return 0; }\n'
                'static inline int atlas_unpack_output_bf16_f32(const volatile uint8_t *src, float *dst) '
                '{ (void)src; (void)dst; return 0; }\n'
            )
            (directory / "main.c").write_text(
                '#include "atlas_model_tile_wrapper.h"\n#include <assert.h>\n'
                '#include <stdlib.h>\n'
                'static unsigned writes;\n'
                'void domain_callback(atlas_f32_memref2 *, const atlas_f32_memref2 *, '
                'const atlas_f32_memref2 *) __asm__("_mlir_ciface_domain_probe");\n'
                '_Noreturn void atlas_model_tile_fail(atlas_model_tile_status status, '
                'atlas_host_result device) {\n'
                '  assert(status == ATLAS_MODEL_TILE_INPUT_DOMAIN);\n'
                '  assert(device == ATLAS_HOST_INVALID && writes == 0);\n'
                '  _Exit(42);\n}\n'
                'static uint32_t read32(void *ctx, uintptr_t address) '
                '{ (void)ctx; (void)address; return 5; }\n'
                'static void write32(void *ctx, uintptr_t address, uint32_t word) '
                '{ (void)ctx; (void)address; (void)word; ++writes; }\n'
                'static int write_bytes(void *ctx, uintptr_t address, const uint8_t *src, size_t bytes) '
                '{ (void)ctx; (void)address; (void)src; (void)bytes; ++writes; return 0; }\n'
                'static int read_bytes(void *ctx, uintptr_t address, uint8_t *dst, size_t bytes) '
                '{ (void)ctx; (void)address; (void)dst; (void)bytes; ++writes; return 0; }\n'
                'int main(void) {\n'
                '  atlas_host host;\n'
                '  atlas_host_mmio mmio = {read32, write32, 0};\n'
                '  atlas_host_memory memory = {1, 0, 0};\n'
                f'  assert(atlas_host_init(&host, atlas_tile_map_{suffix}(), mmio, memory) == ATLAS_HOST_OK);\n'
                '  atlas_tile_memory_io io = {write_bytes, read_bytes, 0};\n'
                f'  assert(atlas_bind_tile_{suffix}(&host, io, 10) == 0);\n'
                '  float data[1024] = {0};\n'
                '  atlas_f32_memref2 input = {data, data, 0, {32, 32}, {32, 1}};\n'
                '  atlas_f32_memref2 result = {0};\n'
                '  atlas_host_result device = ATLAS_HOST_OK;\n'
                f'  atlas_model_tile_status status = atlas_tile_run_checked_{suffix}('
                '&result, &input, &input, &device);\n'
                '  assert(status == ATLAS_MODEL_TILE_INPUT_DOMAIN);\n'
                '  assert(result.allocated == 0 && writes == 0 && !host.active);\n'
                '  domain_callback(&result, &input, &input);\n'
                '  return 99;\n}\n'
            )
            binary = directory / "checked-tile"
            subprocess.run(
                [compiler, "-O2", "-Wall", "-Wextra", "-Werror",
                 "-DATLAS_DIAGNOSTIC_ALLOW_UNBOUND", "-DATLAS_DIAGNOSTIC_ASSUME_COHERENT",
                 "-DATLAS_DIAGNOSTIC_PRECISION_POLICY", "-I", str(directory),
                 "-I", str(ROOT / "runtime"), str(directory / "main.c"),
                 str(directory / "atlas_model_tile_wrapper.c"),
                 str(ROOT / "runtime/atlas_model_tile.c"), str(ROOT / "runtime/atlas_host.c"),
                 "-o", str(binary)], check=True, capture_output=True, text=True,
            )
            execution = subprocess.run([str(binary)], check=False,
                                       capture_output=True, text=True)
            self.assertEqual(execution.returncode, 42, execution.stderr)

    def test_exact_external_symbol_and_source_layout(self) -> None:
        header, source, binder = render_wrapper("forward$kernel_1__rmatmul_0", 1, (0, 1))
        self.assertIn(f"int {binder}(atlas_host *host", header)
        self.assertIn(
            f"int {binder}(atlas_host *host, atlas_tile_memory_io io, size_t max_polls)", source
        )
        self.assertIn('__asm__("_mlir_ciface_forward$kernel_1__rmatmul_0")', source)
        self.assertIn("tile.transpose_input[0] = 0;", source)
        self.assertIn("tile.transpose_input[1] = 1;", source)
        self.assertIn("tile.constant_operand = 1;", source)
        self.assertIn("tile.pack[0] = atlas_pack_runtime_input_0_f32;", source)
        self.assertIn("tile.pack[1] = atlas_pack_runtime_input_1_f32;", source)
        self.assertIn("(tile.host && tile.host->active)", source)
        self.assertNotEqual(render_wrapper("forward$kernel_3", 1, (0, 1))[2], binder)
        _, two_runtime, _ = render_wrapper("forward$kernel_3", 2, (0, 1))
        self.assertIn("tile.constant_operand = 2;", two_runtime)
        self.assertIn("ATLAS_CAPTURED_INPUT_ADDRESS_1", two_runtime)
        self.assertNotIn("atlas_captured_constants[0]", two_runtime)

    def test_rejects_unsafe_symbol_and_unknown_layout(self) -> None:
        with self.assertRaisesRegex(ValueError, "linker name"):
            render_wrapper('kernel";system(0)', 1, (0, 1))
        with self.assertRaisesRegex(ValueError, "operand"):
            render_wrapper("kernel", 3, (0, 1))
        with self.assertRaisesRegex(ValueError, "layout"):
            render_wrapper("kernel", 1, (0, 2))

    def test_artifacts_must_stay_inside_the_probe(self) -> None:
        temporary_root = Path(os.environ.get("TMPDIR", ROOT / "out"))
        temporary_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root) as temporary:
            root = Path(temporary)
            inside = root / "bridge.mlir"
            inside.write_text("module {}\n")
            self.assertEqual(_source_file(root, inside.name), inside)
            with self.assertRaisesRegex(ValueError, "escapes"):
                _source_file(root, "../outside.mlir")
            link = root / "link.mlir"
            link.symlink_to(inside)
            with self.assertRaisesRegex(ValueError, "escapes"):
                _source_file(root, link.name)


if __name__ == "__main__":
    unittest.main()
