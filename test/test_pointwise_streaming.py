"""Check bounded pointwise plans and generated C block placement and tails."""
from __future__ import annotations

import ctypes
import os
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import unittest

from atlas_mlir_native.binding import (
    BF16_SIGMOID_DOMAIN, BF16_ZERO_BINARY_ATTRS, DRAM_BASE, TARGET_IDENTITY,
)
from atlas_mlir_native.pointwise import plan_abi, prepare_execution_plan
from atlas_mlir_native.sigmoid import SILU_ATTRS
from merlin.semantic_compiler.codegen import CodegenOptions, codegen_options
from merlin.semantic_compiler.model import IndexMap, KernelRequest, SemanticNode, TensorType, pointwise_identity_maps

ROOT = Path(__file__).resolve().parents[1]


def request(shape, *, unary=False, broadcast=False):
    type_ = TensorType(shape, "bf16", "raw-bit-pattern")
    attrs = SILU_ATTRS if unary else BF16_ZERO_BINARY_ATTRS["mul_bf16_zero"]
    a = SemanticNode("a", "input", (), type_, (BF16_SIGMOID_DOMAIN,) if unary else (attrs[0],), "input")
    b = SemanticNode("b", "input", (), TensorType((shape[-1],) if broadcast else shape, "bf16", "raw-bit-pattern"),
                     (attrs[0],), "input")
    identity = pointwise_identity_maps(len(shape), 1)[0]
    inputs = (a,) if unary else (a, b)
    maps = ((identity, identity) if unary else
            (identity, IndexMap(2, ((0, 1),), (0,)) if broadcast else identity, identity))
    output = SemanticNode("result", "silu_bf16_lut" if unary else "mul_bf16_zero", tuple(n.id for n in inputs),
                          type_, attrs, index_maps=maps)
    return KernelRequest((*inputs, output), (output.id,), ("external",),
                         tuple((node.id, "external") for node in inputs), TARGET_IDENTITY, "diagnostic", "streaming-test")


class PointwiseStreamingTest(unittest.TestCase):
    def test_default_window_full_shapes_and_resident_control(self):
        for shape, unary, launches, device_bytes in (
            ((1, 5632), True, 4, 180224), ((1, 5632), False, 8, 135168),
            ((128, 5632), True, 11, 262144), ((128, 5632), False, 22, 196608),
        ):
            with self.subTest(shape=shape, unary=unary):
                source = request(shape, unary=unary)
                full, device, streaming = prepare_execution_plan(source)
                self.assertEqual(streaming["launch_count"], launches)
                self.assertEqual(sum((*device.input_packed_bytes, device.packed_bytes)), device_bytes)
                self.assertEqual(full.body_request().digest(), device.body_request().digest())
                self.assertEqual(streaming["block_tiles"] * launches, streaming["total_tiles"])
                self.assertEqual(full.source.digest(), source.digest())
                abi = plan_abi(source)
                self.assertEqual(len(abi["fixed_inputs"]), 1 if unary else 2)
        full, device, streaming = prepare_execution_plan(request((33, 35)))
        self.assertIsNone(streaming)
        self.assertIs(full, device)

    def test_one_tile_frame_refuses_too_small_window(self):
        from merlin.semantic_compiler.target_binding import NativeCompilationError
        with codegen_options(CodegenOptions(io_memory_bytes=4096)):
            with self.assertRaisesRegex(NativeCompilationError, "one complete tile frame"):
                prepare_execution_plan(request((33, 65)))

    @unittest.skipUnless(shutil.which("cc"), "C compiler required")
    def test_generated_dispatch_dense_and_broadcast_tails(self):
        for broadcast in (False, True):
            with self.subTest(broadcast=broadcast), codegen_options(CodegenOptions(io_memory_bytes=18432)):
                self._check_c_dispatch(broadcast)

    def _check_c_dispatch(self, broadcast):
        from merlin.compile.c_conversions import render_bf16_codes_c
        from merlin.compile.c_pointwise import render_pointwise_c
        from atlas_mlir_native.pointwise import input_spec, BF16_POINTWISE_POLICY
        from atlas_mlir_native.tile_io_c import render_tile_io_package

        source = request((65, 65), broadcast=broadcast)
        full, device, streaming = prepare_execution_plan(source)
        self.assertIsNotNone(streaming)
        abi = plan_abi(source)
        addresses = [DRAM_BASE + abi["fixed_inputs"][node.id] * 2048 for node in source.nodes[:-1]]
        output = DRAM_BASE + abi["fixed_outputs"][0] * 2048
        execution = {"input_addresses": addresses, "output_addresses": [output]}
        helpers = render_pointwise_c(full, "pointwise") + "\n" + "\n".join(
            render_bf16_codes_c(input_spec(policy=BF16_POINTWISE_POLICY), node.type.shape[0] if len(node.type.shape) == 1
                               else node.type.shape[0] * node.type.shape[1], "pointwise_encode_" + str(i))
            for i, node in enumerate(source.nodes[:-1]))
        _, files = render_tile_io_package(source, b"\x01\x00\x00\x00", execution, "pointwise", target_source=ROOT,
                                         runtime_linkage="embedded", packed_sizes=(*full.input_packed_bytes, full.packed_bytes),
                                         tile_bytes=2048, helpers=helpers, source_dtype="bf16", streaming=streaming)
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as directory:
            root = Path(directory)
            for name, data in files.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
            (root / "client.c").write_text('''#include "pointwise.h"
int call(atlas_host_read32 rd,atlas_host_write32 wr,atlas_tile_write put,atlas_tile_read get,
         const uint8_t *const *inputs,uint8_t *output) {
  atlas_host host; pointwise_context raw; pointwise_source_context source;
  unsigned char scratch[POINTWISE_WORKSPACE_BYTES],conversion[POINTWISE_SOURCE_WORKSPACE_BYTES];
  if (atlas_host_init(&host,(atlas_host_map){0x20000,0x40000,131072},
      (atlas_host_mmio){rd,wr,0},(atlas_host_memory){1,0,0}) ||
      pointwise_bind(&raw,&host,(atlas_tile_memory_io){put,get,0},scratch,sizeof(scratch),8) ||
      pointwise_source_bind(&source,&raw,conversion,sizeof(conversion))) return -1;
  return pointwise_source_run(&source,inputs,output,0);
}''')
            subprocess.run([shutil.which("cc"), "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "-shared", "-fPIC",
                            "pointwise.c", "client.c", "-o", "client.so"], cwd=root, check=True, capture_output=True)
            client = ctypes.CDLL(str(root / "client.so"))
            self._call(client, source, full, device, streaming, addresses, output, broadcast)

    def _call(self, client, source, full, device, streaming, addresses, output, broadcast):
        memory = bytearray([0xA7] * 18432)
        words, errors, launches = {}, [], []
        polls = None
        rd_type = ctypes.CFUNCTYPE(ctypes.c_uint32, ctypes.c_void_p, ctypes.c_size_t)
        wr_type = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32)
        mem_type = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t)
        values = [tuple(0x3F80 + 0x80 * ((r + c) % 2) for r in range(65) for c in range(65)),
                  tuple(0x3F80 + 0x80 * (c % 2) for r in range(1 if broadcast else 65) for c in range(65))]

        def read32(_, address):
            nonlocal polls
            if address == 0x40004:
                return len(words)
            if address != 0x40008:
                errors.append("unexpected CSR read")
                return 3
            if polls is None:
                return 5
            polls += 1
            return 0 if polls == 1 else 5

        def physical_index(r, c):
            return ((c // 16) * 512 + r * 16 + c % 16) * 2

        def write32(_, address, word):
            nonlocal polls
            if 0x20000 <= address < 0x40000:
                words[(address - 0x20000) // 4] = word
                return
            try:
                self.assertEqual((address, word), (0x40018, 1))
                self.assertEqual(words, {0: 1})
                before = bytes(memory)
                for tile in range(streaming["block_tiles"]):
                    tile_index = len(launches) * streaming["block_tiles"] + tile
                    mr, nc = divmod(tile_index, full.tile_counts[1])
                    for r in range(32):
                        for c in range(32):
                            logical_r, logical_c = mr * 32 + r, nc * 32 + c
                            at = tile * 2048 + physical_index(r, c)
                            inputs = [int.from_bytes(memory[a - DRAM_BASE + at:a - DRAM_BASE + at + 2], "little") for a in addresses]
                            if logical_r < 65 and logical_c < 65:
                                expected = [values[0][logical_r * 65 + logical_c],
                                            values[1][logical_c if broadcast else logical_r * 65 + logical_c]]
                                self.assertEqual(inputs, expected)
                            floats = [struct.unpack("<f", (code << 16).to_bytes(4, "little"))[0] for code in inputs]
                            product = int.from_bytes(struct.pack("<f", floats[0] * floats[1]), "little") >> 16
                            where = output - DRAM_BASE + at
                            memory[where:where + 2] = product.to_bytes(2, "little")
                out_start = output - DRAM_BASE
                self.assertEqual(memory[:out_start], before[:out_start])
                self.assertEqual(memory[out_start + device.packed_bytes:], before[out_start + device.packed_bytes:])
                launches.append(1)
                polls = 0
            except Exception as error:
                errors.append(repr(error))

        def transfer(address, pointer, count, write):
            at = address - DRAM_BASE
            if not 0 <= at <= len(memory) - count:
                errors.append("out of window transfer")
                return 1
            if write:
                memory[at:at + count] = ctypes.string_at(pointer, count)
            else:
                ctypes.memmove(pointer, bytes(memory[at:at + count]), count)
            return 0

        callbacks = (rd_type(read32), wr_type(write32), mem_type(lambda _, a, p, n: transfer(a, p, n, True)),
                     mem_type(lambda _, a, p, n: transfer(a, p, n, False)))
        client.call.argtypes = [rd_type, wr_type, mem_type, mem_type, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
        encoded = [ctypes.create_string_buffer(b"".join(code.to_bytes(2, "little") for code in value)) for value in values]
        inputs = (ctypes.c_void_p * 2)(*(ctypes.addressof(value) for value in encoded))
        output_buffer = ctypes.create_string_buffer(bytes([0xA7]) * (65 * 65 * 2 + 32))
        self.assertEqual(client.call(*callbacks, inputs, ctypes.addressof(output_buffer) + 16), 0)
        self.assertEqual(errors, [])
        self.assertEqual(len(launches), streaming["launch_count"])
        expected = b"".join((0x3F80 + 0x80 * (((r + c) % 2) + (c % 2))).to_bytes(2, "little")
                            for r in range(65) for c in range(65))
        self.assertEqual(output_buffer.raw[16:-17], expected)
        self.assertEqual(output_buffer.raw[:16], bytes([0xA7]) * 16)
        self.assertEqual(output_buffer.raw[-17:-1], bytes([0xA7]) * 16)


if __name__ == "__main__":
    unittest.main()
