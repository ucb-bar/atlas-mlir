"""Exercise generated RMSNorm host staging without emulating its arithmetic."""
from __future__ import annotations

import ctypes
from dataclasses import replace
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from atlas_mlir_native.binding import AtlasNativeBinding, DRAM_BASE
from atlas_mlir_native.pointwise import input_spec
from atlas_mlir_native.rmsnorm import request_for
from atlas_mlir_native.rmsnorm_c import _render_package, prepare_c_package
from atlas_mlir_native.row_loop import prepare_plan
from merlin.semantic_compiler.codegen import CodegenOptions, codegen_options
from merlin.semantic_compiler.model import IndexMap, KernelRequest, SemanticNode, TensorType, pointwise_identity_maps

ROOT = Path(__file__).resolve().parents[1]


def precision_for(shape):
    request = request_for(shape)
    dtype = TensorType(shape, "f32", "raw-bit-pattern")
    data = SemanticNode("hidden", "input", (), dtype, effect="input")
    gamma = SemanticNode("weight", "input", (), TensorType((shape[1],), "f32", "raw-bit-pattern"), effect="input")
    identity = pointwise_identity_maps(2, 1)[0]
    output = SemanticNode("source_result", "f32_mul", (data.id, gamma.id), dtype,
                          (("numeric_semantics", "ieee754-f32-mul-rne-v1"),),
                          index_maps=(identity, IndexMap(2, ((0, 1),), (0,)), identity))
    source = KernelRequest((data, gamma, output), (output.id,), ("external",),
                           ((data.id, "external"), (gamma.id, "external")), request.target_identity, "diagnostic")
    request = replace(request, source_identity=source.digest())
    return SimpleNamespace(source=source, request=request, input_ids=(data.id, gamma.id),
                           operand_specs=(input_spec(), input_spec()), record=lambda: {"test_precision": source.digest()})


@unittest.skipUnless(shutil.which("cc"), "native C compiler required")
class RMSNormCTest(unittest.TestCase):
    def test_checked_publication_is_required(self):
        precision = precision_for((33, 64))
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as directory:
            compiled = Path(directory)
            with patch("atlas_mlir_native.rmsnorm_c.verify_row_loop", side_effect=ValueError("changed publication")):
                with self.assertRaisesRegex(ValueError, "changed publication"):
                    prepare_c_package(precision, compiled, "norm", target_source=ROOT)
            precision.request = replace(precision.request, source_identity="unbound")
            with self.assertRaisesRegex(ValueError, "not bound"):
                prepare_c_package(precision, compiled, "norm", target_source=ROOT)

    def test_native_staging_tails_and_atomic_failure(self):
        for shape in ((1, 32), (32, 64), (65, 64)):
            with self.subTest(shape=shape), codegen_options(CodegenOptions(io_memory_bytes=8 * 1024 * 1024)):
                self.check_package(shape)

    def check_package(self, shape):
        precision = precision_for(shape)
        plan = prepare_plan(precision.request)
        record, files = _render_package(precision, plan, b"\x01\x00\x00\x00", "norm", target_source=ROOT,
                                        runtime_linkage="separate")
        self.assertEqual(record["input_node_ids"], ["hidden", "weight"])
        self.assertEqual(record["source_dispatch_function"], "norm_source_dispatch")
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as directory:
            root = Path(directory)
            for member, data in files.items():
                path = root / member
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
            (root / "client.c").write_text('''#include "norm.h"
int call(atlas_host_read32 rd,atlas_host_write32 wr,atlas_tile_write put,atlas_tile_read get,
         const float *const *inputs,float *output) {
  atlas_host host; norm_context raw; norm_source_context source;
  unsigned char scratch[NORM_WORKSPACE_BYTES],conversion[NORM_SOURCE_WORKSPACE_BYTES];
  if (atlas_host_init(&host,(atlas_host_map){0x20000,0x40000,131072},
      (atlas_host_mmio){rd,wr,0},(atlas_host_memory){1,0,0}) ||
      norm_bind(&raw,&host,(atlas_tile_memory_io){put,get,0},scratch,sizeof(scratch),8) ||
      norm_source_bind(&source,&raw,conversion,sizeof(conversion))) return -1;
  void *outputs[]={output};
  return norm_source_dispatch(&source,(const void *const *)inputs,outputs);
}''')
            result = subprocess.run([shutil.which("cc"), "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "-shared", "-fPIC",
                                     "norm.c", "runtime/atlas_host.c", "client.c", "-o", "client.so"],
                                    cwd=root, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            client = ctypes.CDLL(str(root / "client.so"))
            self.call_package(client, precision, plan)

    def call_package(self, client, precision, plan):
        m, n = plan.tiling.shape
        body = plan.tiling.regions[0].request
        abi = AtlasNativeBinding.plan_boundary_abi(body)
        columns = n // 32
        memory = bytearray([0xA7]) * (plan.end - DRAM_BASE)
        errors, words, writes, launches, imem_writes = [], {}, [], [], []
        polls = None
        fail_read = False
        rd_type = ctypes.CFUNCTYPE(ctypes.c_uint32, ctypes.c_void_p, ctypes.c_size_t)
        wr_type = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32)
        mem_type = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t)

        def pair(r, c):
            return (c // 16) * 1024 + (r * 16 + c % 16) * 2

        codes = [0x3F80 + ((r * 5 + c) % 16) for r in range(m) for c in range(n)]
        gammas = [0x3F80 + c % 8 for c in range(n)]
        values = [ctypes.create_string_buffer(b"".join((x << 16).to_bytes(4, "little") for x in vector))
                  for vector in (codes, gammas)]
        inputs = (ctypes.c_void_p * 2)(*(ctypes.addressof(value) for value in values))
        output = ctypes.create_string_buffer(bytes([0xD3]) * (m * n * 4 + 32))

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

        def write32(_, address, word):
            nonlocal polls
            if 0x20000 <= address < 0x40000:
                words[(address - 0x20000) // 4] = word
                imem_writes.append((address, word))
                return
            try:
                self.assertEqual((address, word), (0x40018, 1))
                for group, row in enumerate(plan.tiling.regions):
                    for key, slot in abi["fixed_inputs"].items():
                        at = plan.address(DRAM_BASE + slot * 2048, group) - DRAM_BASE
                        actual = memory[at:at + 2048]
                        expected = bytearray(2048)
                        if key.startswith("data_") or key.startswith("gamma_"):
                            column = int(key.rsplit("_", 1)[1])
                            for r in range(32):
                                for c in range(32):
                                    code = (gammas[column * 32 + c] if key.startswith("gamma_") else
                                            codes[(row.row_start + r) * n + column * 32 + c] if r < row.valid_rows else 0)
                                    offset = pair(r, c)
                                    expected[offset:offset + 2] = code.to_bytes(2, "little")
                        else:
                            raw = bytes.fromhex(next(item.data_hex for item in body.constants if item.node_id == key))
                            for r in range(32):
                                for c in range(32):
                                    offset = pair(r, c)
                                    expected[offset:offset + 2] = raw[(r * 32 + c) * 2:(r * 32 + c + 1) * 2]
                        self.assertEqual(actual, expected, key)
                    writable = plan.writable_base - DRAM_BASE + group * plan.writable_bytes
                    self.assertEqual(memory[writable:writable + plan.writable_bytes], bytes(plan.writable_bytes))
                    # The fake device copies data tiles. Arithmetic correctness
                    # belongs to the separately authenticated native program.
                    for column, slot in enumerate(abi["fixed_outputs"]):
                        dst = plan.address(DRAM_BASE + slot * 2048, group) - DRAM_BASE
                        src = plan.address(DRAM_BASE + abi["fixed_inputs"][f"data_{column}"] * 2048, group) - DRAM_BASE
                        memory[dst:dst + 2048] = memory[src:src + 2048]
                launches.append(1)
                polls = 0
            except Exception as error:
                errors.append(repr(error))

        def transfer(address, pointer, count, write):
            at = address - DRAM_BASE
            if not 0 <= at <= len(memory) - count:
                errors.append("out of selected window")
                return 1
            if write:
                writes.append(address)
                memory[at:at + count] = ctypes.string_at(pointer, count)
            elif fail_read and address == plan.address(DRAM_BASE + abi["fixed_outputs"][-1] * 2048, plan.groups - 1):
                return 1
            else:
                ctypes.memmove(pointer, bytes(memory[at:at + count]), count)
            return 0

        callbacks = (rd_type(read32), wr_type(write32),
                     mem_type(lambda _, a, p, count: transfer(a, p, count, True)),
                     mem_type(lambda _, a, p, count: transfer(a, p, count, False)))
        client.call.argtypes = [rd_type, wr_type, mem_type, mem_type, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
        self.assertEqual(client.call(*callbacks, inputs, ctypes.addressof(output) + 16), 0)
        self.assertEqual(errors, [])
        self.assertEqual(len(launches), 1)
        expected = b"".join((code << 16).to_bytes(4, "little") for code in codes)
        self.assertEqual(output.raw[16:16 + len(expected)], expected)
        self.assertEqual(output.raw[:16] + output.raw[16 + len(expected):-1], bytes([0xD3]) * 32)
        # Invalid source codes, late gamma errors, and aliases must refuse
        # before staging or image loading, with caller output unchanged.
        for bad, operand in ((0x7F800000, 0), (0x00800000, 0), (0x7FC00000, 1)):
            offset = (len(codes) - 1 if operand == 0 else n - 1) * 4
            saved = values[operand].raw[offset:offset + 4]
            ctypes.memmove(ctypes.addressof(values[operand]) + offset, bad.to_bytes(4, "little"), 4)
            before = (len(writes), len(imem_writes), len(launches), output.raw)
            self.assertEqual(client.call(*callbacks, inputs, ctypes.addressof(output) + 16), 2)
            self.assertEqual((len(writes), len(imem_writes), len(launches), output.raw), before)
            ctypes.memmove(ctypes.addressof(values[operand]) + offset, saved, 4)
        before = (len(writes), len(launches))
        self.assertEqual(client.call(*callbacks, inputs, inputs[0]), 1)
        self.assertEqual((len(writes), len(launches)), before)
        saved_input = inputs[1]
        inputs[1] = inputs[0]
        self.assertEqual(client.call(*callbacks, inputs, ctypes.addressof(output) + 16), 1)
        self.assertEqual((len(writes), len(launches)), before)
        inputs[1] = saved_input
        fail_read = True
        polls = None
        snapshot = output.raw
        self.assertEqual(client.call(*callbacks, inputs, ctypes.addressof(output) + 16), 3)
        self.assertEqual(output.raw, snapshot)
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
