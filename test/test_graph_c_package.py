"""Export real compiler artifacts; run the C API with fake platform callbacks."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from atlas_graph_model_bridge import main, prepare, prepare_c_package


def callback_test(plan, inputs, outputs, constants, one_code=0x38):
    """Exercise host orchestration only. These callbacks do not execute Atlas."""
    addresses = ", ".join(str(n) + "u" for n in plan["input_addresses"])
    output_addresses = ", ".join(str(n) + "u" for n in plan["output_addresses"])
    return f"""
#include "model.h"
#include <assert.h>
#include <string.h>
static const uintptr_t addresses[] = {{{addresses}}};
static const uintptr_t out_addresses[] = {{{output_addresses}}};
static unsigned running, polls, never_halt, starts, writes, before, after;
static unsigned expected_code;
static float input_data[{inputs}][1024], output_data[{outputs}][1026];
static const float *inputs[{inputs}];
static float *outputs[{outputs}];
static uint32_t read32(void *ctx, uintptr_t address) {{
  (void)ctx;
  if (address == 0x40004u) return starts;
  assert(address == 0x40008u);
  if (!running) return 5u;
  if (never_halt || !polls++) return 0u;
  running = 0;
  return 5u;
}}
static void write32(void *ctx, uintptr_t address, uint32_t word) {{
  (void)ctx;
  if (address >= 0x20000u && address < 0x40000u) {{
    size_t count;
    const uint32_t *program = model_program(&count);
    size_t at = (address - 0x20000u) / 4u;
    assert(at < count && program[at] == word);
    return;
  }}
  assert(address == 0x40018u && word == 1u);
  assert(writes == {inputs + constants}u && before == {inputs + constants + outputs}u);
  ++starts;
  polls = 0; running = 1;
}}
static int cache(void *ctx, const atlas_host_buffer *buffer, int pre) {{
  (void)ctx;
  assert(buffer->size_bytes && !(buffer->address % 32u));
  if (pre) ++before;
  else {{ assert(!running && buffer->access == ATLAS_HOST_DEVICE_WRITE); ++after; }}
  return 0;
}}
static int write_memory(void *ctx, uintptr_t address, const uint8_t *data, size_t bytes) {{
  (void)ctx;
  assert(!running && writes < {inputs + constants}u && address == addresses[writes]);
  if (writes < {inputs}u) {{
    assert(bytes == 1024);
    for (size_t i = 0; i < bytes; ++i) assert(data[i] == expected_code);
  }}
  ++writes;
  return 0;
}}
static int read_memory(void *ctx, uintptr_t address, uint8_t *data, size_t bytes) {{
  (void)ctx;
  assert(!running && after == {outputs}u && bytes == 2048u);
  size_t output = 0;
  while (output < {outputs}u && out_addresses[output] != address) ++output;
  assert(output < {outputs}u);
  for (size_t i = 0; i < 1024u; ++i) {{
    size_t half = i / 512, row = i % 512 / 16, column = i % 16 + half * 16;
    float value = (float)(output * 64 + row * 2 + column + 1);
    uint32_t bits;
    memcpy(&bits, &value, 4);
    data[2*i] = (uint8_t)(bits >> 16);
    data[2*i+1] = (uint8_t)(bits >> 24);
  }}
  return 0;
}}
int main(void) {{
  atlas_host host;
  atlas_host_map map = {{0x20000u, 0x40000u, 131072u}};
  atlas_host_mmio mmio = {{read32, write32, 0}};
  atlas_host_memory memory = {{0, cache, 0}};
  atlas_tile_memory_io io = {{write_memory, read_memory, 0}};
  assert(atlas_host_init(&host, map, mmio, memory) == ATLAS_HOST_OK);
  assert(model_bind(&host, io, 3u) == 0);
  for (size_t i = 0; i < {inputs}u; ++i) inputs[i] = input_data[i];
  for (size_t i = 0; i < {outputs}u; ++i) outputs[i] = &output_data[i][1];
  atlas_host_result device;
  for (unsigned trial = 0; trial < 2; ++trial) {{
    for (size_t i = 0; i < {inputs}u; ++i)
      for (size_t j = 0; j < 1024u; ++j) input_data[i][j] = (float)trial;
    for (size_t i = 0; i < {outputs}u; ++i)
      for (size_t j = 0; j < 1026u; ++j) output_data[i][j] = -999.0f;
    expected_code = trial ? {one_code}u : 0u;
    writes = before = after = 0;
    assert(model_run(inputs, outputs, &device) == ATLAS_MODEL_TILE_OK);
    assert(device == ATLAS_HOST_OK && !host.active && starts == trial + 1);
    for (size_t i = 0; i < {outputs}u; ++i) {{
      assert(output_data[i][0] == -999.0f && output_data[i][1025] == -999.0f);
      for (size_t r = 0; r < 32; ++r)
        for (size_t c = 0; c < 32; ++c)
          assert(outputs[i][r*32+c] == (float)(i * 64 + r * 2 + c + 1));
    }}
  }}
  inputs[0] = 0;
  assert(model_run(inputs, outputs, &device) == ATLAS_MODEL_TILE_INVALID && starts == 2);
  inputs[0] = input_data[0];
  if ({outputs}u > 1u) {{
    outputs[1] = outputs[0];
    assert(model_run(inputs, outputs, &device) == ATLAS_MODEL_TILE_INVALID && starts == 2);
    outputs[1] = &output_data[1][1];
  }}
  for (size_t i = 0; i < {outputs}u; ++i)
    for (size_t j = 0; j < 1024u; ++j) outputs[i][j] = -123.0f;
  writes = before = after = 0; never_halt = 1;
  assert(model_run(inputs, outputs, &device) == ATLAS_MODEL_TILE_DEVICE_ERROR);
  assert(device == ATLAS_HOST_TIMEOUT && host.active && starts == 3);
  for (size_t i = 0; i < {outputs}u; ++i)
    for (size_t j = 0; j < 1024u; ++j) assert(outputs[i][j] == -123.0f);
  assert(model_run(inputs, outputs, &device) == ATLAS_MODEL_TILE_DEVICE_ERROR);
  assert(device == ATLAS_HOST_BUSY && starts == 3);
  never_halt = 0;
  assert(atlas_host_wait(&host, 3) == ATLAS_HOST_OK && !host.active);
  return 0;
}}
"""


@unittest.skipUnless(
    all(
        os.environ.get(name)
        for name in ("MERLIN_NATIVE_ROOT", "ATLAS_GRAPH_SOURCE", "ATLAS_GRAPH_ARTIFACT", "ATLAS_GRAPH_HOST_BRIDGE")
    ),
    "select the source, compiled graph and typed host bridge",
)
class GraphCPackageTest(unittest.TestCase):
    def setUp(self):
        sys.path.insert(0, str(Path(os.environ["MERLIN_NATIVE_ROOT"]) / "src"))
        self.addCleanup(lambda: sys.path.remove(str(Path(os.environ["MERLIN_NATIVE_ROOT"]) / "src")))
        self.compiled = Path(os.environ["ATLAS_GRAPH_ARTIFACT"])
        self.record, self.generated = prepare(
            Path(os.environ["ATLAS_GRAPH_SOURCE"]), Path(os.environ["ATLAS_GRAPH_HOST_BRIDGE"]), self.compiled
        )

    def one_code(self):
        import numpy as np

        from merlin.semantic_compiler.precision_values import FloatBoundarySpec

        precision = json.loads((self.compiled / "precision-plan.json").read_text())
        fields = precision["boundary_conversions"][0]["conversion_spec"]
        spec = FloatBoundarySpec(
            **{**fields, "admitted_code_ranges": tuple(tuple(pair) for pair in fields["admitted_code_ranges"])}
        )
        return int(spec.encode(np.array([1.0], dtype=np.float32))[0])

    @unittest.skipUnless(os.environ.get("ATLAS_GRAPH_COMPILE_INPUTS"), "select staged model parameters")
    def test_model_entry_embeds_parameters_and_accepts_only_runtime_inputs(self):
        import numpy as np

        from merlin.frontends.compile_inputs import (
            load_compile_inputs,
            load_lifted_argument_bytes,
        )
        from merlin.semantic_compiler.precision_values import FloatBoundarySpec

        stage = Path(os.environ["ATLAS_GRAPH_COMPILE_INPUTS"])
        selected = load_compile_inputs(stage)
        lifted = load_lifted_argument_bytes(stage, selected)
        record, files = prepare_c_package(self.record, self.generated, self.compiled, "model", compile_inputs=stage)
        self.assertEqual(len(record["model_interface"]["runtime_inputs"]), 1)
        fields = record["precision_plan"]["boundary_conversions"][0]["conversion_spec"]
        spec = FloatBoundarySpec(
            **{**fields, "admitted_code_ranges": tuple(tuple(pair) for pair in fields["admitted_code_ranges"])}
        )
        one_code = int(spec.encode(np.array([1.0], dtype=np.float32))[0])
        expected = {i: spec.encode(np.frombuffer(raw, dtype="<f4")).tobytes() for i, raw in lifted.items()}
        runtime_index = next(i for i, row in enumerate(selected.argument_bindings) if row["kind"] == "input")
        constant_index = record["input_count"]
        for i, row in enumerate(json.loads((self.compiled / "manifest.json").read_text())["constant_inputs"]):
            expected[constant_index + i] = files[row["path"]]
        total_inputs = constant_index + record["constant_count"]
        arrays = "\n".join(
            f"static const unsigned char parameter_{index}[] = {{{','.join(map(str, raw))}}};"
            for index, raw in expected.items()
        )
        checks = "\n".join(
            f"if (writes == {index}u) assert(bytes == sizeof(parameter_{index}) && "
            f"!memcmp(data, parameter_{index}, bytes));"
            for index in expected
        )
        addresses = ",".join(str(value) + "u" for value in record["device_layout"]["input_addresses"])
        source = f"""
#include "model.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>
{arrays}
static const uintptr_t addresses[] = {{{addresses}}};
static unsigned writes, starts, trial, running, polls;
static uint32_t read32(void *ctx, uintptr_t address) {{
  (void)ctx; (void)address;
  if (running && !polls++) return 0;
  running = 0; return 5;
}}
static void write32(void *ctx, uintptr_t address, uint32_t value) {{
  (void)ctx; (void)value;
  if (address == 0x40018u) {{ ++starts; running = 1; polls = 0; }}
}}
static int cache(void *ctx, const atlas_host_buffer *buffer, int before) {{
  (void)ctx; (void)buffer; (void)before; return 0;
}}
static int write_memory(void *ctx, uintptr_t address, const uint8_t *data, size_t bytes) {{
  (void)ctx; assert(writes < {total_inputs}u && address == addresses[writes]);
  {checks}
  if (writes == {runtime_index}u) {{
    assert(bytes == 1024); for (size_t i = 0; i < bytes; ++i) assert(data[i] == (trial ? {one_code}u : 0u));
  }}
  ++writes; return 0;
}}
static int read_memory(void *ctx, uintptr_t address, uint8_t *data, size_t bytes) {{
  (void)ctx; (void)address; memset(data, 0, bytes); return 0;
}}
int main(void) {{
  atlas_host host;
  atlas_host_map map = {{0x20000u, 0x40000u, 131072u}};
  atlas_host_mmio mmio = {{read32, write32, 0}};
  atlas_host_memory memory = {{0, cache, 0}};
  atlas_tile_memory_io io = {{write_memory, read_memory, 0}};
  assert(atlas_host_init(&host, map, mmio, memory) == ATLAS_HOST_OK);
  assert(model_bind(&host, io, 3) == 0 && MODEL_MODEL_INPUT_COUNT == 1);
  float input[1024], output[MODEL_OUTPUT_COUNT][1024];
  const void *inputs[] = {{input}};
  void *outputs[MODEL_OUTPUT_COUNT];
  for (size_t i = 0; i < MODEL_OUTPUT_COUNT; ++i) outputs[i] = output[i];
  for (trial = 0; trial < 2; ++trial) {{
    for (size_t i = 0; i < 1024; ++i) input[i] = (float)trial;
    writes = 0;
    int status = model_model_dispatch(0, inputs, outputs);
    if (status) fprintf(stderr, "model dispatch: status=%d writes=%u starts=%u\\n", status, writes, starts);
    assert(status == 0);
    assert(writes == {total_inputs}u && starts == trial + 1);
  }}
  assert(model_model_run(0, (float *const *)outputs, 0) == ATLAS_MODEL_TILE_INVALID);
  return 0;
}}
"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for member in ("model.c", "model.h"):
                (root / member).write_bytes(files[member])
            (root / "check.c").write_text(source)
            subprocess.run(
                [
                    "cc",
                    "-std=c11",
                    "-O2",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    "-DATLAS_DIAGNOSTIC_GRAPH_PRECISION=1",
                    "model.c",
                    "check.c",
                    "-o",
                    "check",
                ],
                cwd=root,
                check=True,
                capture_output=True,
            )
            run = subprocess.run([str(root / "check")], check=False, capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)

    def test_pair_runs_without_checkout_headers_and_preserves_host_protocol(self):
        compiler = shutil.which("cc")
        if not compiler:
            self.skipTest("C compiler unavailable")
        record, files = prepare_c_package(self.record, self.generated, self.compiled, "model")
        self.assertEqual({p for p in files if p.endswith((".c", ".h"))}, {"model.c", "model.h"})
        self.assertEqual(files["model.bin"], (self.compiled / "program.bin").read_bytes())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for member in ("model.c", "model.h"):
                (root / member).write_bytes(files[member])
            (root / "check.c").write_text(
                callback_test(
                    record["device_layout"],
                    record["input_count"],
                    record["output_count"],
                    record["constant_count"],
                    self.one_code(),
                )
            )
            subprocess.run(
                [
                    compiler,
                    "-std=c11",
                    "-O2",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    "-DATLAS_DIAGNOSTIC_GRAPH_PRECISION=1",
                    "model.c",
                    "check.c",
                    "-o",
                    "check",
                ],
                cwd=root,
                check=True,
                capture_output=True,
            )
            subprocess.run([str(root / "check")], check=True, capture_output=True)

    def test_export_compiles_to_a_freestanding_rv64_object(self):
        llvm = os.environ.get("ATLAS_LLVM_BIN")
        if not llvm:
            self.skipTest("select the LLVM toolchain")
        _, files = prepare_c_package(self.record, self.generated, self.compiled, "model")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for member in ("model.c", "model.h"):
                (root / member).write_bytes(files[member])
            subprocess.run(
                [
                    str(Path(llvm) / "clang"),
                    "--target=riscv64-unknown-elf",
                    "-march=rv64gcv",
                    "-mabi=lp64d",
                    "-mcmodel=medany",
                    "-std=c11",
                    "-O2",
                    "-ffreestanding",
                    "-fno-builtin",
                    "-DATLAS_FREESTANDING=1",
                    "-DATLAS_DIAGNOSTIC_GRAPH_PRECISION=1",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    "-Wno-gcc-install-dir-libstdcxx",
                    "-c",
                    "model.c",
                    "-o",
                    "model.o",
                ],
                cwd=root,
                check=True,
                capture_output=True,
            )
            elf = (root / "model.o").read_bytes()
            self.assertEqual(elf[:6], b"\x7fELF\x02\x01")
            self.assertEqual(struct.unpack_from("<HH", elf, 16), (1, 243))
            self.assertEqual(struct.unpack_from("<I", elf, 48)[0] & 6, 4)  # double-float ABI

    def test_package_runs_between_compiled_host_operations(self):
        from merlin.compile.c_sources import c_source_pair
        from merlin.compile.dispatch_c import DispatchCall, emit_program_c
        from merlin.runtime.program import build_program
        from merlin.xdsl_dialects.lowering.arena_plan import ArenaRequirements
        from merlin.xdsl_dialects.lowering.dispatch_program import (
            Buffer,
            DispatchProgram,
            Node,
        )
        from merlin.xdsl_dialects.lowering.global_plan_emission import dispatch_digest

        compiler = shutil.which("cc")
        if not compiler:
            self.skipTest("C compiler unavailable")
        record, files = prepare_c_package(self.record, self.generated, self.compiled, "model")
        ni, no = record["input_count"], record["output_count"]
        inputs, outputs = [f"x{i}" for i in range(ni)], [f"y{i}" for i in range(no)]
        buffers = {key: Buffer(key, [32, 32], "f32", "arg", i) for i, key in enumerate(inputs)}
        buffers.update({key: Buffer(key, [32, 32], "f32", "intermediate") for key in [*outputs, "pre", "device"]})
        graph = DispatchProgram(
            "forward",
            list(range(ni)),
            buffers,
            [
                Node("dispatch", "pre", [inputs[0]], ["pre"]),
                Node("dispatch", "device", ["pre", *inputs[1:]], ["device", *outputs[1:]]),
                Node("dispatch", "post", ["device"], [outputs[0]]),
            ],
            outputs,
        )
        program = build_program(
            graph,
            memory_requirements=ArenaRequirements(
                8192, (0, 1, 2), (0, 1, 2), alignment=4, dispatch_digest=dispatch_digest(graph)
            ),
        )
        description, driver = emit_program_c(
            program,
            "application",
            {
                0: DispatchCall("host_pre", "cpu"),
                1: DispatchCall(record["dispatch_function"], "device"),
                2: DispatchCall("host_post", "cpu"),
            },
        )
        self.assertEqual(description["inputs"], inputs)
        self.assertEqual(description["outputs"], outputs)
        one_code = self.one_code()
        callbacks = callback_test(record["device_layout"], ni, no, record["constant_count"], one_code).split(
            "\nint main(void)", 1
        )[0]
        callbacks = callbacks.replace('#include "model.h"', '#include "combined.h"')
        harness = (
            callbacks
            + f"""
int host_pre(void *ctx, const void *const *in, void *const *out) {{
  (void)ctx; memcpy(out[0], in[0], 4096); return 0;
}}
int host_post(void *ctx, const void *const *in, void *const *out) {{
  (void)ctx;
  for (unsigned i = 0; i < 1024; ++i) ((float *)out[0])[i] = ((const float *)in[0])[i] + 1.0f;
  return 0;
}}
int main(void) {{
  atlas_host host;
  atlas_host_map map = {{0x20000u, 0x40000u, 131072u}};
  atlas_host_mmio mmio = {{read32, write32, 0}};
  atlas_host_memory memory = {{0, cache, 0}};
  atlas_tile_memory_io io = {{write_memory, read_memory, 0}};
  assert(atlas_host_init(&host, map, mmio, memory) == ATLAS_HOST_OK);
  assert(model_bind(&host, io, 3u) == 0);
  _Alignas(4) unsigned char arena[8192];
  application_state state = {{0}};
  application_buffer buffers[{ni + no}];
  for (unsigned i = 0; i < {ni}; ++i) {{
    inputs[i] = input_data[i]; buffers[i] = (application_buffer){{input_data[i], 4096}};
  }}
  for (unsigned i = 0; i < {no}; ++i) {{
    outputs[i] = &output_data[i][1]; buffers[{ni}+i] = (application_buffer){{outputs[i], 4096}};
  }}
  for (unsigned trial = 0; trial < 2; ++trial) {{
    for (unsigned i = 0; i < {ni}; ++i)
      for (unsigned j = 0; j < 1024; ++j) input_data[i][j] = (float)trial;
    for (unsigned i = 0; i < {no}; ++i)
      for (unsigned j = 0; j < 1026; ++j) output_data[i][j] = -999.0f;
    expected_code = trial ? {one_code}u : 0u; writes = before = after = 0;
    assert(application_run(&state, arena, sizeof(arena), buffers, {ni + no}, 0) == APPLICATION_OK);
    assert(!state.active && state.completed == 3 && starts == trial + 1);
    for (unsigned i = 0; i < {no}; ++i) {{
      assert(output_data[i][0] == -999.0f && output_data[i][1025] == -999.0f);
      for (unsigned r = 0; r < 32; ++r)
        for (unsigned c = 0; c < 32; ++c)
          assert(outputs[i][r*32+c] == (float)(i*64+r*2+c+1+(i == 0)));
    }}
  }}
  return 0;
}}
"""
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pair = c_source_pair(
                "combined",
                public_headers={
                    "model.h": files["model.h"].decode(),
                    "application.h": driver["application.h"],
                },
                private_headers={},
                sources={
                    "model.c": files["model.c"].decode(),
                    "application.c": driver["application.c"],
                },
            )
            for member, text in pair.items():
                (root / member).write_text(text)
            (root / "check.c").write_text(harness)
            subprocess.run(
                [
                    compiler,
                    "-std=c11",
                    "-O2",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    "-DATLAS_DIAGNOSTIC_GRAPH_PRECISION=1",
                    "combined.c",
                    "check.c",
                    "-o",
                    "check",
                ],
                cwd=root,
                check=True,
                capture_output=True,
            )
            subprocess.run([str(root / "check")], check=True, capture_output=True)

    def test_changed_compilation_is_refused_after_wrapper_preparation(self):
        members = ["program.bin", "execution_plan.json", "precision-plan.json"]
        members += [
            p
            for p in prepare_c_package(self.record, self.generated, self.compiled, "model")[1]
            if p.startswith("constants/")
        ]
        for member in members:
            with self.subTest(member=member), tempfile.TemporaryDirectory() as directory:
                copy = Path(directory) / "compiled"
                shutil.copytree(self.compiled, copy)
                data = bytearray((copy / member).read_bytes())
                data[0] ^= 1
                (copy / member).write_bytes(data)
                with self.assertRaisesRegex(ValueError, "changed during C export"):
                    prepare_c_package(self.record, self.generated, copy, "model")

    def test_export_cannot_overwrite_native_compilation(self):
        with tempfile.TemporaryDirectory() as directory:
            copy = Path(directory) / "compiled"
            shutil.copytree(self.compiled, copy)
            before = {p.relative_to(copy): p.read_bytes() for p in copy.rglob("*") if p.is_file()}
            self.assertEqual(
                main(
                    [
                        "--linalg",
                        os.environ["ATLAS_GRAPH_SOURCE"],
                        "--host-bridge",
                        os.environ["ATLAS_GRAPH_HOST_BRIDGE"],
                        "--compiled",
                        str(copy),
                        "--c-package-name",
                        "model",
                        "--out",
                        str(copy),
                    ]
                ),
                2,
            )
            self.assertEqual(before, {p.relative_to(copy): p.read_bytes() for p in copy.rglob("*") if p.is_file()})

    def test_cli_exports_a_fresh_package_with_matching_member_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "export"
            argv = [
                "--linalg",
                os.environ["ATLAS_GRAPH_SOURCE"],
                "--host-bridge",
                os.environ["ATLAS_GRAPH_HOST_BRIDGE"],
                "--compiled",
                str(self.compiled),
                "--c-package-name",
                "model",
                "--out",
                str(root),
            ]
            self.assertEqual(main(argv), 0)
            manifest = json.loads((root / "manifest.json").read_bytes())
            self.assertFalse(manifest["elf_included"])
            self.assertFalse(manifest["executed"])
            for member, digest in manifest["files"].items():
                self.assertEqual(hashlib.sha256((root / member).read_bytes()).hexdigest(), digest)
            expected = [f"{word:08x}" for (word,) in struct.iter_unpack("<I", (root / "model.bin").read_bytes())]
            self.assertEqual((root / "model.words.txt").read_text().splitlines(), expected)
            for item in manifest["device_layout"]["constant_inputs"]:
                self.assertEqual(hashlib.sha256((root / item["path"]).read_bytes()).hexdigest(), item["sha256"])
            before = (root / "manifest.json").read_bytes()
            self.assertEqual(main(argv), 2)
            self.assertEqual((root / "manifest.json").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
