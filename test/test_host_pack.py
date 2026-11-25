"""Host packing shortcuts keep every check and every packed byte: the word-wide FP8 code check and row-run packing."""
from __future__ import annotations

import ctypes
from math import prod
import os
import re
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from atlas_mlir_native.binding import BF16_ZERO_BINARY_ATTRS, TARGET_IDENTITY
from atlas_mlir_native.quantized_loop_c import MATMUL_DISPATCH, MATMUL_TYPES
from atlas_mlir_native.shared_dispatch_c import CHECKS, REGION_TYPES
from merlin.compiler.semantic.model import IndexMap, KernelRequest, SemanticNode, TensorType, pointwise_identity_maps
from merlin.compiler.transforms.pointwise import PointwiseTiling, assemble_pointwise_output, materialize_pointwise_input

ROOT = Path(__file__).resolve().parents[1]


def admitted(code):
    """The FP8 code rule, stated independently: zero, or magnitude 8 to 126 with either sign."""
    return code == 0 or 8 <= (code & 127) <= 126


def compile_library(directory, text):
    (directory / "probe.c").write_text(text)
    subprocess.run([shutil.which("cc"), "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "-Wno-unused-function",
                    "-shared", "-fPIC", f"-I{ROOT / 'runtime'}", "probe.c", "-o", "probe.so"],
                   cwd=directory, check=True, capture_output=True, text=True)
    return ctypes.CDLL(str(directory / "probe.so"))


class FP8CodeCheckTest(unittest.TestCase):
    def test_every_code_in_every_lane_and_alignment(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as directory:
            library = compile_library(Path(directory), '#include "atlas_host.h"\n' + REGION_TYPES + CHECKS + MATMUL_TYPES +
                                      MATMUL_DISPATCH + "int probe(const uint8_t *v,size_t n) { return atlas_matmul_codes_admitted(v,n); }\n")
            library.probe.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
            buffer = (ctypes.c_uint8 * 64)()
            base = ctypes.addressof(buffer)
            fill = [0x08, 0x7E, 0x88, 0xFE, 0x00, 0x38, 0xB8, 0x40]
            for i in range(64):
                buffer[i] = fill[i % 8]
            for start in range(8):
                for length in (1, 7, 8, 9, 16, 23, 40):
                    self.assertEqual(library.probe(base + start, length), 1)
                    for at in range(length):
                        for code in range(256):
                            buffer[start + at] = code
                            self.assertEqual(library.probe(base + start, length), int(admitted(code)),
                                             (start, length, at, code))
                        buffer[start + at] = fill[(start + at) % 8]
            # Codes outside the checked span never count.
            for i in range(64):
                buffer[i] = 0x7F
            self.assertEqual(library.probe(base + 1, 0), 1)


MATMUL_SHIM = """
#define BASE 0x100000u
static uint8_t device[1u<<20];
atlas_host_result atlas_host_ensure_loaded(atlas_host *h,const uint32_t *w,size_t n) { (void)h; (void)w; (void)n; return ATLAS_HOST_OK; }
atlas_host_result atlas_host_start(atlas_host *h,const atlas_host_buffer *b,size_t n) { (void)h; (void)b; (void)n; return ATLAS_HOST_OK; }
atlas_host_result atlas_host_wait(atlas_host *h,size_t polls) { (void)h; (void)polls; return ATLAS_HOST_OK; }
static int put(void *c,uintptr_t a,const uint8_t *v,size_t n) { (void)c; memcpy(device+(a-BASE),v,n); return 0; }
static int get(void *c,uintptr_t a,uint8_t *v,size_t n) { (void)c; memcpy(v,device+(a-BASE),n); return 0; }
static const uint32_t words[1]={0};
static const atlas_matmul_plan plans[]={PLANS};
uint8_t *memory(void) { return device; }
int run(int i,const void *a,const void *b,void *out,uint8_t *raw,uint8_t *converted) {
  static atlas_host host; atlas_region_context region={&host,{put,get,0},raw,1u}; atlas_matmul_source_context ctx={&region,converted};
  const void *inputs[]={a,b}; void *outputs[]={out};
  return atlas_matmul_dispatch(&plans[i],&ctx,inputs,outputs);
}
"""


class MatmulPackTest(unittest.TestCase):
    # m, n, k, batches, rhs batches, LHS stored by columns, RHS stored by columns, f32 result, streamed
    CASES = [(35, 37, 65, 2, 2, 0, 0, 0, 0), (35, 37, 65, 2, 1, 1, 0, 1, 0), (33, 40, 70, 1, 1, 0, 1, 0, 1),
             (1, 2, 100, 1, 1, 0, 0, 1, 1), (64, 16, 32, 1, 1, 1, 1, 0, 1)]
    LHS, RHS, OUT, STATUS = 0x100000, 0x140000, 0x180000, 0x1F0000

    def plan(self, m, n, k, batches, rhs_batches, lhs_columns, rhs_columns, f32, streamed):
        mc, nc, kc = (m + 31) // 32, (n + 31) // 32, (k + 31) // 32
        lhs, rhs, out = batches * mc * kc * 1024, rhs_batches * nc * kc * 1024, batches * mc * nc * 2048
        output_bytes = batches * m * n * (4 if f32 else 2)
        raw, output_offset, status_offset = (2048, 0, 0) if streamed else (lhs + rhs + out + 4, lhs + rhs, lhs + rhs + out)
        return (f"{{words,1u,{raw}u,{output_bytes}u,{output_bytes}u,2u,{{{batches * m * k}u,{rhs_batches * n * k}u}},0u,0u,1u,"
                f"{lhs_columns}u,{rhs_columns}u,{f32}u,{batches * m * k}u,{rhs_batches * n * k}u,{batches}u,{rhs_batches}u,"
                f"{m}u,{n}u,{k}u,{mc}u,{nc}u,{kc}u,{lhs}u,{output_offset}u,{{{{0}},{{0}}}},0,0u,0u,0u,"
                f"2u,{{{{{self.LHS}u,0u,{lhs}u,0u}},{{{self.RHS}u,{lhs}u,{rhs}u,0u}}}},"
                f"{self.STATUS}u,4u,{status_offset}u,{self.OUT}u,{output_offset}u,{out}u,{streamed}u}}"), raw, output_bytes

    def test_device_bytes_and_results_equal_the_tile_layout(self):
        import numpy as np

        rng = np.random.default_rng(7)
        codes = np.array([v for v in range(256) if admitted(v) and v], np.uint8)
        plans = [self.plan(*case) for case in self.CASES]
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as directory:
            library = compile_library(Path(directory), '#include "atlas_host.h"\n#include <string.h>\n' + REGION_TYPES + CHECKS +
                                      MATMUL_TYPES + MATMUL_DISPATCH + MATMUL_SHIM.replace("PLANS", ",".join(p[0] for p in plans)))
            library.memory.restype = ctypes.POINTER(ctypes.c_uint8 * (1 << 20))
            device = np.ctypeslib.as_array(library.memory().contents)
            for index, ((m, n, k, batches, rhs_batches, lhs_columns, rhs_columns, f32, streamed), (_, raw, output_bytes)) in \
                    enumerate(zip(self.CASES, plans)):
                with self.subTest(index):
                    logical_a = rng.choice(codes, (batches, m, k))
                    logical_b = rng.choice(codes, (rhs_batches, n, k))
                    stored = [np.ascontiguousarray(v.transpose(0, 2, 1) if columns else v)
                              for v, columns in ((logical_a, lhs_columns), (logical_b, rhs_columns))]
                    # Misaligned copies exercise the unaligned head and tail of the code check and the row copies.
                    holders = [np.zeros(v.size + 3, np.uint8) for v in stored]
                    for holder, v in zip(holders, stored):
                        holder[3:] = v.ravel()
                    mc, nc = (m + 31) // 32, (n + 31) // 32
                    device[:] = 0xA5
                    tiles = rng.integers(0, 256, batches * mc * nc * 2048, dtype=np.uint8)
                    device[self.OUT - self.LHS:][:tiles.size] = tiles
                    device[self.STATUS - self.LHS:][:4] = 0
                    out = np.full(output_bytes, 0x33, np.uint8)
                    raw_buffer, converted = np.zeros(raw, np.uint8), np.zeros(output_bytes, np.uint8)
                    pointers = [h.ctypes.data + 3 for h in holders] + [v.ctypes.data for v in (out, raw_buffer, converted)]
                    self.assertEqual(library.run(index, *(ctypes.c_void_p(p) for p in pointers)), 0)
                    for logical, address, rows in ((logical_a, self.LHS, m), (logical_b, self.RHS, n)):
                        count, row_tiles, k_tiles = logical.shape[0], (rows + 31) // 32, (k + 31) // 32
                        padded = np.zeros((count, row_tiles * 32, k_tiles * 32), np.uint8)
                        padded[:, :rows, :k] = logical
                        expected = padded.reshape(count, row_tiles, 32, k_tiles, 32).transpose(0, 1, 3, 2, 4).ravel()
                        start = address - self.LHS
                        np.testing.assert_array_equal(device[start:start + expected.size], expected)
                    words = tiles.view("<u2").reshape(batches, mc, nc, 2, 32, 16)
                    result = words.transpose(0, 1, 4, 2, 3, 5).reshape(batches, mc * 32, nc * 32)[:, :m, :n]
                    expected = (result.astype("<u4") << 16) if f32 else result.astype("<u2")
                    np.testing.assert_array_equal(out, np.ascontiguousarray(expected).view(np.uint8).ravel())
                    # A refused code anywhere fails the call before any device write and leaves the output unchanged.
                    for holder in holders:
                        for at in (3, 3 + holder.size // 2, holder.size - 1):
                            original, holder[at] = holder[at], 0x7F
                            device[:] = 0xA5
                            out[:] = 0x33
                            self.assertEqual(library.run(index, *(ctypes.c_void_p(p) for p in pointers)), 2)
                            self.assertTrue((device[:self.STATUS - self.LHS] == 0xA5).all() and (out == 0x33).all())
                            holder[at] = original


def pointwise_request(shape, source_shape, mapping):
    type_ = TensorType(shape, "bf16", "raw-bit-pattern")
    attrs = BF16_ZERO_BINARY_ATTRS["mul_bf16_zero"]
    a = SemanticNode("a", "input", (), type_, (attrs[0],), "input")
    b = SemanticNode("b", "input", (), TensorType(source_shape, "bf16", "raw-bit-pattern"), (attrs[0],), "input")
    identity = pointwise_identity_maps(len(shape), 1)[0]
    output = SemanticNode("result", "mul_bf16_zero", ("a", "b"), type_, attrs, index_maps=(identity, mapping, identity))
    return KernelRequest((a, b, output), (output.id,), ("external",), (("a", "external"), ("b", "external")),
                         TARGET_IDENTITY, "diagnostic", "pack-test")


class PointwisePackTest(unittest.TestCase):
    CASES = {
        # name: (logical shape, second operand shape, its map, whether rows are one source run)
        "dense": ((65, 70), (65, 70), IndexMap(2, ((1, 0), (0, 1)), (0, 0)), True),
        "row_broadcast": ((65, 70), (70,), IndexMap(2, ((0, 1),), (0,)), True),
        "offset_window": ((40, 33), (50, 40), IndexMap(2, ((1, 0), (0, 1)), (3, 5)), True),
        "transpose": ((40, 33), (33, 40), IndexMap(2, ((0, 1), (1, 0)), (0, 0)), False),
        "rank4_dense": ((1, 3, 7, 9), (1, 3, 7, 9), IndexMap(4, ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)),
                                                             (0, 0, 0, 0)), True),
        "rank3_channel_broadcast": ((3, 7, 9), (3, 1, 1), IndexMap(3, ((1, 0, 0), (0, 0, 0), (0, 0, 0)), (0, 0, 0)), False),
        # A singleton logical dimension's coordinate is always 0, so its coefficient never reaches the emitted constants.
        "singleton_inactive_coefficient": ((1, 4), (4,), IndexMap(2, ((7, 1),), (0,)), True),
        # Weighted by its source stride these coefficients are 2**64, outside the C domain, but they are inactive.
        "weighted_row_outside_c_domain": ((1, 4), (2, 4), IndexMap(2, ((2**62, 0), (0, 1)), (0, 0)), True),
        "weighted_constant_outside_c_domain": ((1, 2, 4), (2, 4), IndexMap(3, ((2**62, 1, 0), (0, 0, 1)), (0, 0)), True),
    }

    def test_packed_and_assembled_bytes_equal_the_reference(self):
        from merlin.compiler.lower.c.pointwise import render_pointwise_c

        for name, (shape, source_shape, mapping, runs) in self.CASES.items():
            with self.subTest(name), tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as directory:
                plan = PointwiseTiling(pointwise_request(shape, source_shape, mapping), (32, 32), 2, b"\x00\x00",
                                       operand_pad_elements=(b"\x00\x00", b"\x5a\xa5"))
                text = render_pointwise_c(plan, "probe")
                self.assertEqual("one source run" in text.split("probe_pack_1")[1].split("probe_assemble")[0], runs)
                self.assertTrue(all(int(literal) <= 2**63 - 1 for literal in re.findall(r"\b[0-9]+(?=u|LL|ULL)", text)), name)
                library = compile_library(Path(directory), "#include <string.h>\n" + text + """
void pack(int operand,uint8_t *out,const uint8_t *source) { if (operand) probe_pack_1(out,source); else probe_pack_0(out,source); }
void assemble(uint8_t *out,const uint8_t *tiles) { probe_assemble(out,tiles); }
""")
                for operand, node in enumerate(plan.source.nodes[:-1]):
                    payload = bytes((i * 37 + 11 + operand) % 251 + 1 for i in range(prod(node.type.shape) * 2))
                    expected = materialize_pointwise_input(plan, operand, payload)
                    out = ctypes.create_string_buffer(b"\xee" * len(expected), len(expected))
                    library.pack(operand, out, payload)
                    self.assertEqual(out.raw, expected, (name, operand))
                tiles = bytes((i * 13 + 7) % 256 for i in range(plan.packed_bytes))
                out = ctypes.create_string_buffer(b"\xee" * (prod(plan.shape) * 2), prod(plan.shape) * 2)
                library.assemble(out, tiles)
                self.assertEqual(out.raw, assemble_pointwise_output(plan, tiles), name)


    def test_coefficients_outside_the_c_domain_refuse_on_every_path(self):
        from merlin.compiler.lower.c.pointwise import render_pointwise_c

        # The plan admits it (the first coordinate is always 0), but the index expression cannot be stated in C.
        plan = PointwiseTiling(pointwise_request((1, 4), (4,), IndexMap(2, ((2**100, 1),), (0,))), (32, 32), 2, b"\x00\x00")
        with self.assertRaisesRegex(ValueError, "signed C index domain"):
            render_pointwise_c(plan, "probe")


if __name__ == "__main__":
    unittest.main()
