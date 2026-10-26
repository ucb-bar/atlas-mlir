"""Run generated packed-loop C with explicit platform callbacks and npu_model."""

import ctypes
import dataclasses
import hashlib
import importlib.metadata
import json
import os
import shutil
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path

from test_looped_mnk import (
    ROOT,
    build_snapshot,
    chunked_case_data,
    compile_case,
    seeded_case_data,
    seeded_source_request,
    source_request,
)


@unittest.skipUnless(
    os.environ.get("MERLIN_NATIVE_ROOT") and os.environ.get("ATLAS_OOT_BIN_DIR"),
    "set MERLIN_NATIVE_ROOT and ATLAS_OOT_BIN_DIR",
)
class LoopedCPackageTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from atlas_mlir_native.looped_c import prepare_c_package

        cls.temporary = tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR"))
        cls.root = Path(cls.temporary.name)
        cls.snapshot = build_snapshot(cls.root)
        cls.cases = []
        for shape in ((64, 64, 64), (33, 65, 31), (1, 7, 65), (257, 513, 32), (65, 1025, 33)):
            target = cls.root / ("_".join(map(str, shape)))
            _, _, _, _, execution = compile_case(cls.snapshot, shape, target)
            record, files = prepare_c_package(target / "source-request.json", target, "layer", target_source=ROOT)
            package = target / "c-package"
            package.mkdir()
            for name, data in files.items():
                (package / name).write_bytes(data)
            cls.cases.append((shape, target, package, record, execution))
        for shape, relu in (((32, 32, 32), False), ((33, 35, 65), False), ((1, 7, 65), True)):
            target = cls.root / ("seeded_" + "_".join(map(str, shape)))
            _, _, _, _, execution = compile_case(cls.snapshot, shape, target, source=seeded_source_request(shape, relu))
            record, files = prepare_c_package(target / "source-request.json", target, "layer", target_source=ROOT)
            package = target / "c-package"
            package.mkdir()
            for name, data in files.items():
                (package / name).write_bytes(data)
            cls.cases.append((shape, target, package, record, execution))
        from atlas_mlir_native.binding import AtlasNativeBinding

        from merlin.semantic_compiler.search import SearchLimits

        for shape, relu in (((1, 2, 4097), True), ((1, 2, 8193), False)):
            target = cls.root / ("chunked_" + "_".join(map(str, shape)))
            request = source_request(*shape, relu=relu)
            abi = AtlasNativeBinding.plan_boundary_abi(request)
            manifest = AtlasNativeBinding.compile(
                cls.snapshot,
                request,
                fixed_inputs=abi["fixed_inputs"],
                fixed_outputs=tuple(abi["fixed_outputs"]),
                target_source=ROOT,
                destination=target,
                limits=SearchLimits(),
            )
            record, files = prepare_c_package(target / "source-request.json", target, "layer", target_source=ROOT)
            package = target / "c-package"
            package.mkdir()
            for name, data in files.items():
                (package / name).parent.mkdir(parents=True, exist_ok=True)
                (package / name).write_bytes(data)
            first = target / manifest["segments"][0]["path"]
            execution = json.loads((first / "execution_plan.json").read_text())
            cls.cases.append((shape, target, package, record, execution))
        captured = os.environ.get("ATLAS_LOOPED_CAPTURE")
        if captured:
            target = Path(captured) / "compiled"
            record, files = prepare_c_package(
                target / "source-request.json",
                target,
                "layer",
                target_source=ROOT,
                compile_inputs=Path(captured) / "compile-inputs",
            )
            execution = json.loads((target / "execution_plan.json").read_text())
            shape = (*execution["shape"], execution["reduction_extent"])
            package = cls.root / "captured-c-package"
            package.mkdir()
            for name, data in files.items():
                (package / name).write_bytes(data)
            cls.cases.append((shape, target, package, record, execution))

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_cross_compile_matches_the_application_float_abi(self):
        tools = os.environ.get("ATLAS_LLVM_BIN")
        if not tools:
            self.skipTest("select the LLVM toolchain")
        for _, _, package, _, _ in self.cases:
            subprocess.run(
                [
                    str(Path(tools) / "clang"),
                    "--target=riscv64-unknown-elf",
                    "-march=rv64gcv",
                    "-mabi=lp64d",
                    "-mcmodel=medany",
                    "-std=c11",
                    "-O2",
                    "-ffreestanding",
                    "-fno-builtin",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    "-Wno-gcc-install-dir-libstdcxx",
                    "-c",
                    "layer.c",
                    "-o",
                    "layer.o",
                ],
                cwd=package,
                check=True,
                capture_output=True,
            )
            data = (package / "layer.o").read_bytes()
            self.assertEqual(data[:6], b"\x7fELF\x02\x01")
            self.assertEqual(struct.unpack_from("<HH", data, 16), (1, 243))
            self.assertEqual(struct.unpack_from("<I", data, 48)[0] & 6, 4)

    def test_binding_rejects_context_host_overlap_without_mutation(self):
        compiler = shutil.which("cc")
        if not compiler:
            self.skipTest("C compiler unavailable")
        for shape, _, package, record, _ in self.cases:
            if record["schema"] != "atlas.looped_c_package.v1":
                continue
            with self.subTest(shape=shape, seeded="LAYER_SEED_BYTES" in (package / "layer.h").read_text()):
                source = package / "alias-bind.c"
                source.write_text(r'''#include "layer.h"
#include <assert.h>
#include <string.h>
static uint32_t read32(void *p, uintptr_t a) { (void)p; (void)a; return 5; }
static void write32(void *p, uintptr_t a, uint32_t v) { (void)p; (void)a; (void)v; }
static int put(void *p, uintptr_t a, const uint8_t *d, size_t n) {
  (void)p; (void)a; (void)d; (void)n; return 0;
}
static int get(void *p, uintptr_t a, uint8_t *d, size_t n) {
  (void)p; (void)a; (void)d; (void)n; return 0;
}
int main(void) {
  union { atlas_host host; layer_context context;
          unsigned char bytes[sizeof(atlas_host)+sizeof(layer_context)]; } shared;
  unsigned char before[sizeof(shared)];
  static unsigned char workspace[LAYER_WORKSPACE_BYTES];
  atlas_host_map map = {0x20000u,0x40000u,131072u};
  atlas_host_mmio mmio = {read32,write32,0};
  atlas_host_memory memory = {.coherent=1};
  atlas_tile_memory_io io = {put,get,0};
  for (size_t offset=0;offset<=_Alignof(layer_context);offset+=_Alignof(layer_context)) {
    memset(&shared,0xa5,sizeof(shared));
    assert(atlas_host_init(&shared.host,map,mmio,memory)==ATLAS_HOST_OK);
    memcpy(before,&shared,sizeof(shared));
    layer_context *context=(layer_context *)(void *)(shared.bytes+offset);
    assert(layer_bind(context,&shared.host,io,workspace,sizeof(workspace),100)==LAYER_INVALID);
    assert(memcmp(before,&shared,sizeof(shared))==0);
  }
  layer_context disjoint;
  assert(layer_bind(&disjoint,&shared.host,io,workspace,sizeof(workspace),100)==LAYER_OK);
  return 0;
}
''')
                executable = package / "alias-bind"
                subprocess.run(
                    [compiler, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
                     str(source), str(package / "layer.c"), "-o", str(executable)],
                    check=True, capture_output=True,
                )
                subprocess.run([str(executable)], check=True, capture_output=True)

    def test_cli_rejects_mutated_or_stale_native_members(self):
        from atlas_mlir_native.looped_c import prepare_c_package

        _, compiled, _, _, _ = self.cases[0]
        for member in ("program.bin", "execution_plan.json", "looped-mnk-tiling.json", "atlas-virtual.mlir"):
            with self.subTest(member=member), tempfile.TemporaryDirectory(dir=self.root) as directory:
                copy = Path(directory) / "compiled"
                shutil.copytree(compiled, copy, ignore=shutil.ignore_patterns("c-package"))
                payload = bytearray((copy / member).read_bytes())
                payload[0] ^= 1
                (copy / member).write_bytes(payload)
                with self.assertRaises((ValueError, UnicodeError)):
                    prepare_c_package(compiled / "source-request.json", copy, "layer", target_source=ROOT)
        # Rehashing a false ABI must not make it agree with the source IR.
        with tempfile.TemporaryDirectory(dir=self.root) as directory:
            copy = Path(directory) / "compiled"
            shutil.copytree(compiled, copy, ignore=shutil.ignore_patterns("c-package"))
            plan = json.loads((copy / "execution_plan.json").read_text())
            plan["input_addresses"][0] += 0x20000
            payload = (json.dumps(plan, sort_keys=True, indent=2) + "\n").encode()
            (copy / "execution_plan.json").write_bytes(payload)
            manifest = json.loads((copy / "manifest.json").read_text())
            digest = hashlib.sha256(payload).hexdigest()
            manifest["execution_plan_sha256"] = digest
            manifest["stage_artifacts"]["execution_plan.json"] = digest
            (copy / "manifest.json").write_text(json.dumps(manifest, sort_keys=True) + "\n")
            with self.assertRaisesRegex(ValueError, "checked source lowering"):
                prepare_c_package(compiled / "source-request.json", copy, "layer", target_source=ROOT)
        exported = self.root / "cli-package"
        result = subprocess.run(
            [
                os.sys.executable,
                str(ROOT / "tools/atlas_graph_model_bridge.py"),
                "--request",
                str(compiled / "source-request.json"),
                "--compiled",
                str(compiled),
                "--c-package-name",
                "layer",
                "--out",
                str(exported),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        manifest = (exported / "manifest.json").read_bytes()
        repeated = subprocess.run(result.args, capture_output=True, text=True, check=False)
        self.assertNotEqual(repeated.returncode, 0)
        self.assertEqual(manifest, (exported / "manifest.json").read_bytes())

    def test_chunk_decomposition_and_storage_are_replayed_after_rehashing(self):
        from atlas_mlir_native.looped_c import prepare_c_package

        _, compiled, _, _, _ = next(case for case in self.cases if case[3]["schema"] == "atlas.chunked_c_package.v1")
        for fault in ("plan", "overlap", "role"):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory(dir=self.root) as directory:
                copy = Path(directory) / "compiled"
                shutil.copytree(compiled, copy, ignore=shutil.ignore_patterns("c-package"))
                manifest = json.loads((copy / "manifest.json").read_text())
                if fault == "role":
                    manifest["segments"][0]["role"] = "carry"
                else:
                    path = "chunked-mnk-tiling.json" if fault == "plan" else "chunk-bindings.json"
                    data = json.loads((copy / path).read_text())
                    if fault == "plan":
                        data["k_block_tiles"] += 1
                    else:
                        data["frames"]["seed"] = data["frames"]["out"]
                    payload = (json.dumps(data, sort_keys=True) + "\n").encode()
                    (copy / path).write_bytes(payload)
                    manifest["stage_artifacts"][path] = hashlib.sha256(payload).hexdigest()
                (copy / "manifest.json").write_text(json.dumps(manifest, sort_keys=True) + "\n")
                with self.assertRaises(ValueError):
                    prepare_c_package(compiled / "source-request.json", copy, "layer", target_source=ROOT)

    def test_streamed_panel_mapping_is_replayed_after_rehashing(self):
        from atlas_mlir_native.looped_c import prepare_c_package

        _, compiled, _, record, _ = self.cases[3]
        self.assertIn("streaming", record)
        with tempfile.TemporaryDirectory(dir=self.root) as directory:
            copied = Path(directory) / "compiled"
            shutil.copytree(compiled, copied)
            execution = json.loads((copied / "execution_plan.json").read_text())
            execution["streaming"]["launch_counts"][1] += 1
            payload = (json.dumps(execution, sort_keys=True, indent=2) + "\n").encode()
            (copied / "execution_plan.json").write_bytes(payload)
            manifest = json.loads((copied / "manifest.json").read_text())
            digest = hashlib.sha256(payload).hexdigest()
            manifest["execution_plan_sha256"] = digest
            manifest["stage_artifacts"]["execution_plan.json"] = digest
            (copied / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
            with self.assertRaisesRegex(ValueError, "checked source lowering"):
                prepare_c_package(copied / "source-request.json", copied, "changed", target_source=ROOT)

    @unittest.skipUnless(os.environ.get("ATLAS_LOOPED_CAPTURE"), "select a fresh source-bound capture")
    def test_captured_parameter_binding_rejects_changed_weights(self):
        from atlas_mlir_native.looped_c import prepare_c_package

        captured = Path(os.environ["ATLAS_LOOPED_CAPTURE"])
        _, compiled, _, record, _ = self.cases[-1]
        self.assertEqual([row["source_role"] for row in record["source_region"]["runtime_inputs"]], ["lhs"])
        self.assertEqual([row["role"] for row in record["source_region"]["parameters"]], ["rhs"])
        with tempfile.TemporaryDirectory(dir=self.root) as directory:
            copied = Path(directory) / "compile-inputs"
            shutil.copytree(captured / "compile-inputs", copied)
            weights = copied / "weights.safetensors"
            raw = bytearray(weights.read_bytes())
            raw[-1] ^= 1
            weights.write_bytes(raw)
            with self.assertRaisesRegex(ValueError, "differs"):
                prepare_c_package(
                    compiled / "source-request.json", compiled, "layer", target_source=ROOT, compile_inputs=copied
                )

    @unittest.skipUnless(os.environ.get("ATLAS_RUN_NPU_MODEL") == "1", "set ATLAS_RUN_NPU_MODEL=1")
    def test_c_pack_launch_readback_and_repeated_inputs_execute_on_model(self):
        import torch
        from npu_model.configs import isa_definition as isa
        from npu_model.configs.hardware.default import DefaultHardwareConfig
        from npu_model.logging import LoggerConfig
        from npu_model.simulation import Simulation
        from npu_model.software.program import InstantiableProgram
        from test_spill_model_execution import MODEL_REVISION, model_instruction
        from tools.check_program_decode import decode_binary

        origin = json.loads(importlib.metadata.distribution("npu-model").read_text("direct_url.json") or "{}")
        self.assertTrue(
            MODEL_REVISION in origin.get("url", "") or origin.get("vcs_info", {}).get("commit_id") == MODEL_REVISION
        )
        compiler = shutil.which("cc")
        if not compiler:
            self.skipTest("C compiler unavailable")
        inventory = json.loads((ROOT / "docs/selected-variant-inventory.json").read_text())
        torch.set_num_threads(1)
        READ = ctypes.CFUNCTYPE(ctypes.c_uint32, ctypes.c_void_p, ctypes.c_size_t)
        WRITE = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32)
        PUT = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t)
        GET = PUT
        for shape, compiled, package, record, execution in self.cases:
            with self.subTest(shape=shape):
                from atlas_mlir_native.looped_c import prepare_c_package

                _, files = prepare_c_package(
                    compiled / "source-request.json",
                    compiled,
                    "layer",
                    target_source=ROOT,
                    compile_inputs=Path(os.environ["ATLAS_LOOPED_CAPTURE"]) / "compile-inputs"
                    if "source_region" in record
                    else None,
                    runtime_linkage="separate",
                )
                package = package / "linked"
                package.mkdir()
                for filename, data in files.items():
                    (package / filename).parent.mkdir(parents=True, exist_ok=True)
                    (package / filename).write_bytes(data)
                (package / "shim.c").write_text("""#include "layer.h"
static atlas_host host;
static layer_context context;
static uint8_t scratch[LAYER_WORKSPACE_BYTES];
#ifdef LAYER_SEED_BYTES
static uint8_t seed_memory[LAYER_SEED_BYTES+32];
void set_seed(const void *value) {
  __builtin_memset(seed_memory,0x9d,sizeof(seed_memory));
  __builtin_memcpy(seed_memory+16,value,LAYER_SEED_BYTES);
}
const void *get_seed(void) { return seed_memory; }
#endif
#ifdef LAYER_SOURCE_INPUT_COUNT
static layer_source_context source_context;
static uint8_t source_scratch[LAYER_SOURCE_WORKSPACE_BYTES];
#endif
static unsigned before, after, reject_cache;
static int cache(void *ctx, const atlas_host_buffer *b, int pre) {
  (void)ctx; if (!b->size_bytes) return 1;
  if (pre) { ++before; if (reject_cache) return 1; } else ++after;
  return 0;
}
int bind(atlas_host_read32 rd, atlas_host_write32 wr, atlas_tile_write put, atlas_tile_read get) {
  atlas_host_map map = {0x20000u,0x40000u,131072u};
  atlas_host_mmio mmio = {rd,wr,0};
  atlas_host_memory memory = {0,cache,0};
  if (atlas_host_init(&host,map,mmio,memory)) return -1;
  int result = layer_bind(&context,&host,(atlas_tile_memory_io){put,get,0},scratch,sizeof(scratch),20);
#ifdef LAYER_SOURCE_INPUT_COUNT
  if (result) return result;
  return layer_source_bind(&source_context,&context,source_scratch,sizeof(source_scratch));
#else
  return result;
#endif
}
int call(const void *a, const void *w, void *out, int *device) {
#ifdef LAYER_SEED_BYTES
  return layer_run(&context,a,w,seed_memory+16,out,(atlas_host_result *)device);
#else
  return layer_run(&context,a,w,out,(atlas_host_result *)device);
#endif
}
int dispatch_call(const void *a, const void *w, void *out) {
#ifdef LAYER_SEED_BYTES
  const void *inputs[3] = {a,w,seed_memory+16}; void *outputs[1] = {out};
#else
  const void *inputs[2] = {a,w}; void *outputs[1] = {out};
#endif
  return layer_dispatch(&context,inputs,outputs);
}
#ifdef LAYER_REGION_INPUT_COUNT
_Static_assert(LAYER_REGION_INPUT_COUNT == 1, "this capture has one runtime region operand");
int region_call(const void *a, void *out, int *device) {
  const uint8_t *inputs[1] = {a};
  return layer_region_run(&context,inputs,out,(atlas_host_result *)device);
}
int region_dispatch_call(const void *a, void *out) {
  const void *inputs[1] = {a}; void *outputs[1] = {out};
  return layer_region_dispatch(&context,inputs,outputs);
}
int source_call(const float *a, float *out, int *device) {
  const float *inputs[1] = {a};
  return layer_source_run(&source_context,inputs,out,(atlas_host_result *)device);
}
int source_dispatch_call(const float *a, float *out) {
  const void *inputs[1] = {a}; void *outputs[1] = {out};
  return layer_source_dispatch(&source_context,inputs,outputs);
}
const void *source_workspace(void) { return source_scratch; }
#endif
unsigned count(unsigned which) { return which==0 ? before : which==1 ? after : (unsigned)host.active; }
void cache_error(unsigned on) { reject_cache=on; }
int finish(void) { return atlas_host_wait(&host,20); }
""")
                library_path = package / ("layer-" + "_".join(map(str, shape)) + ".so")
                subprocess.run(
                    [
                        compiler,
                        "-std=c11",
                        "-O2",
                        "-Wall",
                        "-Wextra",
                        "-Werror",
                        "-shared",
                        "-fPIC",
                        "layer.c",
                        "runtime/atlas_host.c",
                        "shim.c",
                        "-o",
                        str(library_path),
                    ],
                    cwd=package,
                    check=True,
                    capture_output=True,
                )
                lib = ctypes.CDLL(str(library_path))
                lib.bind.argtypes = [READ, WRITE, PUT, GET]
                lib.call.argtypes = [ctypes.c_void_p] * 4
                lib.dispatch_call.argtypes = [ctypes.c_void_p] * 3
                seeded = len(record["host_inputs"]) == 3
                if seeded:
                    lib.set_seed.argtypes = [ctypes.c_void_p]
                    lib.get_seed.restype = ctypes.c_void_p
                if "source_region" in record:
                    lib.region_call.argtypes = [ctypes.c_void_p] * 3
                    lib.region_dispatch_call.argtypes = [ctypes.c_void_p] * 2
                    lib.source_call.argtypes = [ctypes.c_void_p] * 3
                    lib.source_dispatch_call.argtypes = [ctypes.c_void_p] * 2
                    lib.source_workspace.restype = ctypes.c_void_p
                lib.count.argtypes = [ctypes.c_uint]
                lib.count.restype = ctypes.c_uint
                lib.cache_error.argtypes = [ctypes.c_uint]
                dram = bytearray([0xA5]) * (256 * 1024)
                words = {}
                errors = []
                cycles = []
                control = {
                    "running": False,
                    "polls": 0,
                    "timeout": False,
                    "read_error": False,
                    "fail_read_at": None,
                    "write_error": False,
                    "starts": 0,
                    "writes": 0,
                    "cache_before": 0,
                }
                native_manifest = json.loads((compiled / "manifest.json").read_text())
                if native_manifest.get("artifact_kind") == "program_set":
                    images = {
                        (compiled / row["path"] / "program.bin").read_bytes(): json.loads(
                            (compiled / row["path"] / "execution_plan.json").read_text()
                        )
                        for row in native_manifest["segments"]
                    }
                else:
                    images = {(compiled / "program.bin").read_bytes(): execution}
                output_address = execution["output_addresses"][0] - 0x90000000
                output_size = execution["output_bytes"]

                def read32(_, address, control=control, errors=errors):
                    if address == 0x40004:
                        return control["starts"]
                    if address != 0x40008:
                        errors.append("wrong CSR read")
                        return 3
                    if not control["running"]:
                        return 5
                    control["polls"] += 1
                    if control["timeout"] or control["polls"] == 1:
                        return 0
                    control["running"] = False
                    return 5 if not errors else 3

                def write32(
                    _,
                    address,
                    word,
                    words=words,
                    images=images,
                    control=control,
                    errors=errors,
                    dram=dram,
                    output_address=output_address,
                    output_size=output_size,
                    package=package,
                    cycles=cycles,
                ):
                    if 0x20000 <= address < 0x40000:
                        if address == 0x20000:
                            words.clear()
                        words[(address - 0x20000) // 4] = word
                        return
                    if address != 0x40018 or word != 1:
                        errors.append("wrong launch")
                        return
                    control.update(running=True, polls=0, starts=control["starts"] + 1)
                    try:
                        actual_binary = b"".join(words[i].to_bytes(4, "little") for i in range(len(words)))
                        assert actual_binary in images
                        control["cache_before"] += len(images[actual_binary]["input_addresses"]) + 1
                        decoded = decode_binary(inventory, actual_binary)
                        program = InstantiableProgram(
                            [model_instruction(row, inventory, isa) for row in decoded["instructions"]]
                        )
                        program.dram_base = 0x90000000
                        initial = bytes(dram)
                        program.memory_regions = [(0, torch.tensor(list(initial), dtype=torch.uint8))]
                        config = DefaultHardwareConfig()
                        config.arch_state_config = dataclasses.replace(
                            config.arch_state_config, numerics="rtl", randomize_init=True, init_seed=217
                        )
                        simulation = Simulation(
                            config,
                            LoggerConfig(filename=str(package / f"model-{control['starts']}.json")),
                            program,
                            verbose=False,
                        )
                        state = simulation.core.arch_state
                        try:
                            state.vmem.fill_(0xCD)
                            simulation.run(max_cycles=500000)
                            assert state.halted and simulation.get_stats().runtime_errors == 0
                            result = bytes(state.read_dram(program.dram_base, len(dram)).tolist())
                            assert result[:output_address] == initial[:output_address]
                            assert result[output_address + output_size :] == initial[output_address + output_size :]
                            dram[:] = result
                            cycles.append(simulation.get_stats().cycles)
                        finally:
                            state.close()
                    except Exception as error:  # noqa: BLE001 -- ctypes drops exceptions; check errors after each call.
                        errors.append(repr(error))

                def put(_, address, data, size, control=control, errors=errors, dram=dram):
                    if control["write_error"]:
                        return 1
                    if control["running"]:
                        errors.append("input reused before completion")
                        return 1
                    control["writes"] += 1
                    at = address - 0x90000000
                    dram[at : at + size] = ctypes.string_at(data, size)
                    return 0

                def get(_, address, data, size, control=control, errors=errors, dram=dram):
                    if control["read_error"] or control["starts"] == control["fail_read_at"]:
                        return 1
                    if control["running"]:
                        errors.append("premature readback")
                        return 1
                    at = address - 0x90000000
                    ctypes.memmove(data, bytes(dram[at : at + size]), size)
                    return 0

                callbacks = (READ(read32), WRITE(write32), PUT(put), GET(get))
                self.assertEqual(lib.bind(*callbacks), 0)
                m, n, k = shape
                row_stride = k if "source_region" in record else 1
                w = bytes((0xB8 if c % 2 else 0x38) if i % 11 == c % 11 else 0 for c in range(n) for i in range(k))
                weights = ctypes.create_string_buffer(w)
                out = ctypes.create_string_buffer(m * n * 2)
                for shift in (0, 1):
                    a = bytes((0x38, 0x40, 0x44)[(r * row_stride + i + shift) % 3] for r in range(m) for i in range(k))
                    source = ctypes.create_string_buffer(a)
                    expected = b"".join(
                        (
                            struct.unpack(
                                "<I",
                                struct.pack(
                                    "<f",
                                    sum(1 + (r * row_stride + i + shift) % 3 for i in range(k) if i % 11 == c % 11)
                                    * (-1 if c % 2 else 1),
                                ),
                            )[0]
                            >> 16
                        ).to_bytes(2, "little")
                        for r in range(m)
                        for c in range(n)
                    )
                    if seeded:
                        a, w, seed_bytes, expected = seeded_case_data(shape, shift, relu="epilogue" in execution)
                        source = ctypes.create_string_buffer(a)
                        weights = ctypes.create_string_buffer(w)
                        lib.set_seed(ctypes.create_string_buffer(seed_bytes))
                    elif record["schema"] == "atlas.chunked_c_package.v1":
                        a, w, expected = chunked_case_data(
                            shape, shift, relu="final" in {item["role"] for item in record["programs"]}
                        )
                        source = ctypes.create_string_buffer(a)
                        weights = ctypes.create_string_buffer(w)
                    device = ctypes.c_int(-1)
                    if "source_region" in record:
                        floats = (ctypes.c_float * (m * k))(
                            *(1 + (r * row_stride + i + shift) % 3 for r in range(m) for i in range(k))
                        )
                        float_output = (ctypes.c_float * (m * n))()
                        float_input_before = ctypes.string_at(floats, m * k * 4)
                        status = (
                            lib.source_call(floats, float_output, ctypes.byref(device))
                            if not shift
                            else lib.source_dispatch_call(floats, float_output)
                        )
                        expected_floats = b"".join(b"\0\0" + expected[i : i + 2] for i in range(0, len(expected), 2))
                        self.assertEqual(ctypes.string_at(float_output, m * n * 4), expected_floats)
                        self.assertEqual(ctypes.string_at(floats, m * k * 4), float_input_before)
                    else:
                        status = (
                            lib.call(source, weights, out, ctypes.byref(device))
                            if not shift
                            else lib.dispatch_call(source, weights, out)
                        )
                    self.assertEqual(errors, [])
                    self.assertEqual(status, 0)
                    if not shift:
                        self.assertEqual(device.value, 0)
                    if "source_region" not in record:
                        self.assertEqual(out.raw, expected)
                    self.assertEqual(source.raw[:-1], a)
                    if seeded:
                        self.assertEqual(
                            ctypes.string_at(lib.get_seed(), len(seed_bytes) + 32),
                            b"\x9d" * 16 + seed_bytes + b"\x9d" * 16,
                        )
                if "source_region" in record:
                    previous_output = ctypes.string_at(float_output, m * n * 4)
                    previous_writes = control["writes"]
                    floats[-1] = float("nan")
                    self.assertEqual(lib.source_call(floats, float_output, ctypes.byref(device)), 2)
                    self.assertEqual(ctypes.string_at(float_output, m * n * 4), previous_output)
                    self.assertEqual(control["writes"], previous_writes)
                    floats[-1] = 1 + ((m - 1) * row_stride + k - 1 + 1) % 3
                    input_before_alias = ctypes.string_at(floats, m * k * 4)
                    self.assertEqual(lib.source_call(floats, floats, ctypes.byref(device)), 1)
                    self.assertEqual(ctypes.string_at(floats, m * k * 4), input_before_alias)
                ctypes.memset(out, 0x6B, m * n * 2)
                before = out.raw
                starts = control["starts"]
                if seeded:
                    writes = control["writes"]
                    for code in (0x7F80, 0x7FC0, 0x8000, 0x0001):
                        lib.set_seed(ctypes.create_string_buffer(code.to_bytes(2, "little") + seed_bytes[2:]))
                        self.assertEqual(lib.call(source, weights, out, None), 2)
                        self.assertEqual(control["writes"], writes)
                        self.assertEqual(out.raw, before)
                    lib.set_seed(ctypes.create_string_buffer(seed_bytes))
                for code in (0x01, 0x80, 0x78, 0xFF):
                    invalid = ctypes.create_string_buffer(bytes([code]) + a[1:])
                    self.assertEqual(lib.call(invalid, weights, out, None), 2)
                    self.assertEqual(control["starts"], starts)
                    self.assertEqual(out.raw, before)
                invalid_weights = ctypes.create_string_buffer(w[:-1] + b"\x01")
                self.assertEqual(lib.call(source, invalid_weights, out, None), 2)
                self.assertEqual(control["starts"], starts)
                self.assertEqual(out.raw, before)
                self.assertEqual(lib.call(source, weights, source, None), 1)
                self.assertEqual(control["starts"], starts)
                control["write_error"] = True
                self.assertEqual(lib.call(source, weights, out, None), 3)
                self.assertEqual(control["starts"], starts)
                self.assertEqual(out.raw, before)
                control["write_error"] = False
                control["read_error"] = True
                self.assertEqual(lib.call(source, weights, out, None), 3)
                self.assertEqual(out.raw, before)
                control["read_error"] = False
                if "streaming" in record:
                    control["fail_read_at"] = control["starts"] + 2
                    self.assertEqual(lib.call(source, weights, out, None), 3)
                    self.assertEqual(out.raw, before)
                    self.assertEqual(lib.count(2), 0)
                    control["fail_read_at"] = None
                control["timeout"] = True
                if "source_region" in record:
                    source_before = ctypes.string_at(float_output, m * n * 4)
                    self.assertEqual(lib.source_call(floats, float_output, ctypes.byref(device)), 4)
                    self.assertEqual(ctypes.string_at(float_output, m * n * 4), source_before)
                else:
                    self.assertEqual(lib.call(source, weights, out, None), 4)
                self.assertEqual(lib.count(2), 1)
                self.assertEqual(out.raw, before)
                writes = control["writes"]
                self.assertEqual(lib.call(source, weights, out, None), 4)
                self.assertEqual(control["writes"], writes)
                if "source_region" in record:
                    scratch_bytes = record["source_region"]["source_interface"]["workspace_bytes"]
                    source_scratch_before = ctypes.string_at(lib.source_workspace(), scratch_bytes)
                    self.assertEqual(lib.source_call(floats, float_output, ctypes.byref(device)), 4)
                    self.assertEqual(ctypes.string_at(lib.source_workspace(), scratch_bytes), source_scratch_before)
                    self.assertEqual(ctypes.string_at(float_output, m * n * 4), source_before)
                control["timeout"] = False
                self.assertEqual(lib.finish(), 0)
                lib.cache_error(1)
                starts = control["starts"]
                self.assertEqual(lib.call(source, weights, out, None), 4)
                self.assertEqual(control["starts"], starts)
                self.assertEqual(out.raw, before)
                lib.cache_error(0)
                self.assertEqual(lib.call(source, weights, out, None), 0)
                self.assertEqual(out.raw, expected)
                self.assertEqual(errors, [])
                self.assertEqual(lib.count(0), control["cache_before"] + 1)
                self.assertEqual(lib.count(1), control["starts"])
                print(json.dumps({"shape": shape, "model_launches": control["starts"], "cycles": cycles}), flush=True)


if __name__ == "__main__":
    unittest.main()
