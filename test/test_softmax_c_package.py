"""Run the generated C interface against the real model's instruction executor."""

import ctypes
import dataclasses
import importlib.metadata
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from test_looped_mnk import ROOT, build_snapshot
from test_tiled_softmax import build_case, reference, scores_for


@unittest.skipUnless(
    os.environ.get("MERLIN_NATIVE_ROOT") and os.environ.get("ATLAS_OOT_BIN_DIR"),
    "select Merlin and Atlas tools",
)
class SoftmaxCPackageTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from atlas_mlir_native.softmax_c import prepare_c_package

        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        snapshot = build_snapshot(cls.root / "snapshot")
        cls.snapshot = snapshot
        cls.cases = []
        for shape in ((1, 65), (33, 96), (1, 113)):
            root = cls.root / f"case-{shape[0]}-{shape[1]}"
            root.mkdir()
            plan, compiled, _ = build_case(snapshot, root, shape, "split", candidate_nodes=24)
            record, files = prepare_c_package(root / "source-request.json", root, "layer", target_source=ROOT)
            package = root / "c-package"
            package.mkdir()
            for filename, data in files.items():
                (package / filename).parent.mkdir(parents=True, exist_ok=True)
                (package / filename).write_bytes(data)
            cls.cases.append((root, plan, compiled, record, package))
        from atlas_mlir_native.row_loop import compile_row_loop, prepare_plan
        from test_tiled_softmax import source_request

        from merlin.semantic_compiler.search import SearchLimits

        root = cls.root / "case-row-loop"
        request = source_request((33, 163))
        compile_row_loop(snapshot, request, target_source=ROOT, destination=root, limits=SearchLimits())
        plan = prepare_plan(request).tiling
        record, files = prepare_c_package(root / "source-request.json", root, "layer", target_source=ROOT)
        package = root / "c-package"
        package.mkdir()
        for filename, data in files.items():
            (package / filename).parent.mkdir(parents=True, exist_ok=True)
            (package / filename).write_bytes(data)
        cls.cases.append((root, plan, [root], record, package))
        source_artifact = os.environ.get("ATLAS_SOURCE_SOFTMAX_ARTIFACT")
        if source_artifact:
            from atlas_mlir_native.binding import AtlasNativeBinding
            from atlas_mlir_native.row_softmax import prepare_row_softmax
            from atlas_mlir_native.softmax_c_source import prepare_source_c_package

            from merlin.frontends.compile_inputs import load_compile_inputs

            root = cls.root / "source-case"
            root.mkdir()
            source_artifact = Path(source_artifact)
            shutil.copytree(source_artifact / "compile-inputs", root / "compile-inputs")
            staged = load_compile_inputs(root / "compile-inputs")
            precision = AtlasNativeBinding.prepare_precision_graph(
                staged.program,
                entry=staged.entry,
                policy=staged.precision_policy,
                original_source_sha256=staged.original_source_sha256,
            )
            plan = prepare_row_softmax(precision.request)
            (root / "source-request.json").write_text(json.dumps(precision.request.record()))
            compiled = []
            for index in range(len(plan.regions)):
                path = root / f"compiled-{index}"
                shutil.copytree(source_artifact / f"compiled/compiled-{index}", path)
                compiled.append(path)
            record, files = prepare_source_c_package(
                root / "compile-inputs/program.mlir",
                root / "compile-inputs/precision-policy.json",
                root,
                "layer",
                entry=None,
                target_source=ROOT,
                compile_inputs=root / "compile-inputs",
            )
            package = root / "c-package"
            package.mkdir()
            for filename, data in files.items():
                (package / filename).parent.mkdir(parents=True, exist_ok=True)
                (package / filename).write_bytes(data)
            cls.cases.append((root, plan, compiled, record, package))

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_cli_publication_identity_and_rv64_object(self):
        from atlas_mlir_native.softmax_c import prepare_c_package

        clang = shutil.which("clang")
        for root, _, compiled, record, package in self.cases:
            if not clang:
                self.fail("Clang is required for the RV64 lp64d object check")
            subprocess.run(
                [
                    clang,
                    "--target=riscv64-unknown-elf",
                    "-march=rv64gcv",
                    "-mabi=lp64d",
                    "-ffreestanding",
                    "-std=c11",
                    "-O2",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    "-c",
                    "layer.c",
                    "-o",
                    "layer.rv64.o",
                ],
                cwd=package,
                check=True,
                capture_output=True,
            )
            elf = (package / "layer.rv64.o").read_bytes()
            self.assertEqual(elf[:6], b"\x7fELF\x02\x01")
            self.assertEqual(int.from_bytes(elf[48:52], "little") & 6, 4)  # double-float ABI
            destination = self.root / (root.name + "-cli-c-package")
            command = [
                os.sys.executable,
                str(ROOT / "tools/atlas_graph_model_bridge.py"),
                "--request",
                str(root / "source-request.json"),
                "--compiled",
                str(root),
                "--c-package-name",
                "layer",
                "--out",
                str(destination),
            ]
            if "source_interface" in record:
                command = [
                    os.sys.executable,
                    str(ROOT / "tools/atlas_graph_model_bridge.py"),
                    "--linalg",
                    str(root / "compile-inputs/program.mlir"),
                    "--compile-inputs",
                    str(root / "compile-inputs"),
                    "--diagnostic-graph-policy",
                    str(root / "compile-inputs/precision-policy.json"),
                    "--compiled",
                    str(root),
                    "--c-package-name",
                    "layer",
                    "--out",
                    str(destination),
                ]
            result = subprocess.run(command, capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(
                (destination / "layer.c").read_bytes(),
                (package / "layer.c").read_bytes(),
            )
            self.assertNotEqual(subprocess.run(command, capture_output=True, check=False).returncode, 0)
            separate = self.root / (root.name + "-separate-runtime")
            separate_command = command.copy()
            separate_command[separate_command.index("--out") + 1] = str(separate)
            separate_command += ["--runtime-linkage", "separate"]
            result = subprocess.run(separate_command, capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads((separate / "manifest.json").read_text())["runtime_linkage"], "separate")
            for filename in ("layer.c", "runtime/atlas_host.c"):
                subprocess.run(
                    [
                        clang,
                        "--target=riscv64-unknown-elf",
                        "-march=rv64gcv",
                        "-mabi=lp64d",
                        "-ffreestanding",
                        "-std=c11",
                        "-O2",
                        "-Wall",
                        "-Wextra",
                        "-Werror",
                        "-c",
                        filename,
                        "-o",
                        filename + ".o",
                    ],
                    cwd=separate,
                    check=True,
                    capture_output=True,
                )
            linker = shutil.which("ld.lld")
            self.assertIsNotNone(linker, "LLD is required for relocatable region linking")
            subprocess.run(
                [linker, "-r", "layer.c.o", "runtime/atlas_host.c.o", "-o", "linked.rv64.o"],
                cwd=separate,
                check=True,
                capture_output=True,
            )
            binary = compiled[0] / "program.bin"
            old = binary.read_bytes()
            try:
                binary.write_bytes(bytes([old[0] ^ 1]) + old[1:])
                with self.assertRaisesRegex(ValueError, "binary differs|stage artifact changed"):
                    prepare_c_package(root / "source-request.json", root, "layer", target_source=ROOT)
            finally:
                binary.write_bytes(old)

    @unittest.skipUnless(os.environ.get("ATLAS_SOURCE_SOFTMAX_ARTIFACT"), "select a captured softmax compilation")
    def test_staged_source_cli_replays_the_precision_and_row_group(self):
        root = self.root / "source-case"
        common = [
            "--engine",
            "merlin_native",
            "--support",
            "atlas",
            "--snapshot",
            str(self.snapshot.root),
            "--compile-inputs",
            str(root / "compile-inputs"),
            "--diagnostic-graph-policy",
            str(root / "compile-inputs/precision-policy.json"),
            "--row-softmax-tile-index",
            "0",
            "--mode",
            "diagnostic",
        ]
        destination = self.root / "source-cli"
        limits = Path(os.environ["ATLAS_SOURCE_SOFTMAX_ARTIFACT"]) / "search-limits.json"
        command = [
            os.sys.executable,
            "-m",
            "merlin.targetgen.cli",
            "native-compile",
            *common,
            "--target-source",
            str(ROOT),
            "--abi",
            "auto",
            "--search-limits",
            str(limits),
            "--out",
            str(destination),
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        verify = [
            os.sys.executable,
            "-m",
            "merlin.targetgen.cli",
            "native-verify",
            *common,
            "--artifact",
            str(destination),
            "--out",
            str(self.root / "source-verify.json"),
        ]
        result = subprocess.run(verify, capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        verify[verify.index("--row-softmax-tile-index") + 1] = "1"
        self.assertNotEqual(subprocess.run(verify, capture_output=True, check=False).returncode, 0)

    @unittest.skipUnless(os.environ.get("ATLAS_RUN_NPU_MODEL") == "1", "enable the selected model")
    def test_c_calls_all_row_groups_without_partial_outputs_or_early_reuse(self):
        import torch
        from atlas_mlir_native.layout import unpack_bf16_pair_halves
        from npu_model.configs import isa_definition as isa
        from npu_model.configs.hardware.default import DefaultHardwareConfig
        from npu_model.logging import LoggerConfig
        from npu_model.simulation import Simulation
        from npu_model.software.program import InstantiableProgram
        from test_spill_model_execution import MODEL_REVISION, model_instruction
        from tools.check_program_decode import decode_binary

        from merlin.semantic_compiler.tiled_softmax import assemble_row_softmax_outputs

        origin = json.loads(importlib.metadata.distribution("npu-model").read_text("direct_url.json") or "{}")
        self.assertTrue(
            MODEL_REVISION in origin.get("url", "") or origin.get("vcs_info", {}).get("commit_id") == MODEL_REVISION
        )
        compiler = shutil.which("cc")
        self.assertIsNotNone(compiler, "host C compiler required")
        torch.set_num_threads(1)
        inventory = json.loads((ROOT / "docs/selected-variant-inventory.json").read_text())
        read_type = ctypes.CFUNCTYPE(ctypes.c_uint32, ctypes.c_void_p, ctypes.c_size_t)
        write_type = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32)
        memory_type = ctypes.CFUNCTYPE(
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_size_t,
        )
        executions = 0
        for case_root, plan, compiled, record, package in self.cases:
            # Link distinct compiled regions with one runtime. Keep the standalone
            # package intact for its separate CLI reproducibility checks.
            from atlas_mlir_native.softmax_c import prepare_c_package
            from atlas_mlir_native.softmax_c_source import prepare_source_c_package

            package = package / "linked"
            package.mkdir()
            if "source_interface" in record:
                _, members = prepare_source_c_package(
                    case_root / "compile-inputs/program.mlir",
                    case_root / "compile-inputs/precision-policy.json",
                    case_root,
                    "layer",
                    entry=None,
                    target_source=ROOT,
                    compile_inputs=case_root / "compile-inputs",
                    runtime_linkage="separate",
                )
            else:
                _, members = prepare_c_package(
                    case_root / "source-request.json",
                    case_root,
                    "layer",
                    target_source=ROOT,
                    runtime_linkage="separate",
                )
            for filename, data in members.items():
                (package / filename).parent.mkdir(parents=True, exist_ok=True)
                (package / filename).write_bytes(data)
            composed = plan.shape == (1, 65)
            if composed:
                self._write_composed_call(case_root, plan, package)
            (package / "shim.c").write_text("""#include "layer.h"
static atlas_host host;
static layer_context ctx;
static uint8_t scratch[LAYER_WORKSPACE_BYTES];
#ifdef LAYER_SOURCE_INPUT_COUNT
static layer_source_context source_ctx;
static uint8_t source_scratch[LAYER_SOURCE_WORKSPACE_BYTES];
#endif
int bind(atlas_host_read32 rd, atlas_host_write32 wr, atlas_tile_write put, atlas_tile_read get) {
  if (atlas_host_init(&host,(atlas_host_map){0x20000,0x40000,131072},
      (atlas_host_mmio){rd,wr,0},(atlas_host_memory){1,0,0})) return -1;
  int status=layer_bind(&ctx,&host,(atlas_tile_memory_io){put,get,0},scratch,sizeof(scratch),4);
#ifdef LAYER_SOURCE_INPUT_COUNT
  if (!status) status=layer_source_bind(&source_ctx,&ctx,source_scratch,sizeof(source_scratch));
#endif
  return status;
}
int call(const void *in, void *out, int *device) {
  return layer_run(&ctx,in,out,(atlas_host_result *)device);
}
int dispatch(const void *in, void *out) {
  const void *inputs[1]={in}; void *outputs[1]={out};
  return layer_dispatch(&ctx,inputs,outputs);
}
int finish(void) { return atlas_host_wait(&host,4); }
unsigned active(void) { return host.active; }
const void *workspace(void) { return scratch; }
#ifdef LAYER_SOURCE_INPUT_COUNT
int source_call(const float *in, float *out, int *device) {
  const float *inputs[1]={in};
  return layer_source_run(&source_ctx,inputs,out,(atlas_host_result *)device);
}
int source_dispatch(const float *in, float *out) {
  const void *inputs[1]={in}; void *outputs[1]={out};
  return layer_source_dispatch(&source_ctx,inputs,outputs);
}
const void *source_workspace(void) { return source_scratch; }
#endif
""")
            sources = ["layer.c", "runtime/atlas_host.c", "shim.c"]
            if composed:
                sources += ["other.c", "application.c", "application_binding.c"]
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
                    *sources,
                    "-o",
                    "layer.so",
                ],
                cwd=package,
                check=True,
                capture_output=True,
            )
            lib = ctypes.CDLL(str(package / "layer.so"))
            lib.bind.argtypes = [read_type, write_type, memory_type, memory_type]
            lib.call.argtypes = [ctypes.c_void_p] * 3
            lib.dispatch.argtypes = [ctypes.c_void_p] * 2
            lib.workspace.restype = ctypes.c_void_p
            if composed:
                lib.bind_application.argtypes = [read_type, write_type, memory_type, memory_type]
                lib.call_application.argtypes = [ctypes.c_void_p] * 3
            if "source_interface" in record:
                lib.source_call.argtypes = [ctypes.c_void_p] * 3
                lib.source_dispatch.argtypes = [ctypes.c_void_p] * 2
                lib.source_workspace.restype = ctypes.c_void_p
            dram, words, errors = bytearray([0xA5]) * (256 * 1024), {}, []
            images = {
                (root / "program.bin").read_bytes(): json.loads((root / "execution_plan.json").read_text())
                for root in compiled
            }
            control = {
                "running": False,
                "polls": 0,
                "timeout": False,
                "reads": 0,
                "fail_read": 0,
                "starts": 0,
                "writes": 0,
            }

            def rd(_, address, errors=errors, control=control):
                if address == 0x40004:
                    return control["starts"]
                if address != 0x40008:
                    errors.append("wrong CSR read")
                if not control["running"]:
                    return 5
                control["polls"] += 1
                if control["timeout"] or control["polls"] == 1:
                    return 0
                control["running"] = False
                return 5 if not errors else 3

            def wr(
                _,
                address,
                word,
                words=words,
                control=control,
                errors=errors,
                images=images,
                dram=dram,
                package=package,
            ):
                nonlocal executions
                if 0x20000 <= address < 0x40000:
                    if address == 0x20000:
                        words.clear()
                    words[(address - 0x20000) // 4] = word
                    return
                control.update(running=True, polls=0, starts=control["starts"] + 1)
                try:
                    self.assertEqual((address, word), (0x40018, 1))
                    binary = b"".join(words[i].to_bytes(4, "little") for i in range(len(words)))
                    execution = images[binary]
                    program = InstantiableProgram(
                        [
                            model_instruction(row, inventory, isa)
                            for row in decode_binary(inventory, binary)["instructions"]
                        ]
                    )
                    program.dram_base = 0x90000000
                    initial = bytes(dram)
                    program.memory_regions = [(0, torch.tensor(list(initial), dtype=torch.uint8))]
                    config = DefaultHardwareConfig()
                    config.arch_state_config = dataclasses.replace(
                        config.arch_state_config,
                        numerics="rtl",
                        randomize_init=True,
                        init_seed=217,
                    )
                    sim = Simulation(
                        config,
                        LoggerConfig(filename=str(package / f"model-{control['starts']}.json")),
                        program,
                        verbose=False,
                    )
                    state = sim.core.arch_state
                    try:
                        state.vmem.fill_(0xCD)
                        sim.run(max_cycles=500000)
                        self.assertTrue(state.halted)
                        self.assertEqual(sim.get_stats().runtime_errors, 0)
                        result = bytes(state.read_dram(program.dram_base, len(dram)).tolist())
                        guards = bytearray(result)
                        for output in execution.get("output_addresses", []):
                            at = output - program.dram_base
                            guards[at : at + 2048] = initial[at : at + 2048]
                        staging = execution.get("staging_workspace")
                        if execution.get("schema") == "atlas.row_loop_execution_plan.v1":
                            staging = {
                                "address": execution["output_base"],
                                "bytes": execution["groups"] * execution["output_stride_bytes"],
                            }
                        if staging is not None:
                            at = staging["address"] - program.dram_base
                            end = at + staging["bytes"]
                            self.assertTrue(0 <= at < end <= len(initial))
                            self.assertEqual(at % 2048, 0)
                            self.assertEqual(end % 2048, 0)
                            for source in execution.get("input_addresses", []):
                                start = source - program.dram_base
                                self.assertTrue(start + 2048 <= at or end <= start)
                                self.assertEqual(result[start : start + 2048], initial[start : start + 2048])
                            guards[at:end] = initial[at:end]
                        self.assertEqual(bytes(guards), initial)
                        dram[:] = result
                        executions += 1
                    finally:
                        state.close()
                except Exception as error:  # noqa: BLE001 -- ctypes cannot propagate callback exceptions.
                    errors.append(repr(error))

            def put(_, address, data, size, control=control, errors=errors, dram=dram):
                if control["running"]:
                    errors.append("input reused before completion")
                    return 1
                control["writes"] += 1
                at = address - 0x90000000
                dram[at : at + size] = ctypes.string_at(data, size)
                return 0

            def get(_, address, data, size, control=control, errors=errors, dram=dram):
                control["reads"] += 1
                if control["running"]:
                    errors.append("read before completion")
                    return 1
                if control["reads"] == control["fail_read"]:
                    return 1
                at = address - 0x90000000
                ctypes.memmove(data, bytes(dram[at : at + size]), size)
                return 0

            callbacks = (
                read_type(rd),
                write_type(wr),
                memory_type(put),
                memory_type(get),
            )
            self.assertEqual(lib.bind(*callbacks), 0)
            if composed:
                self.assertEqual(lib.bind_application(*callbacks), 0)
            size = plan.shape[0] * plan.shape[1] * 2
            out, device = (
                ctypes.create_string_buffer(b"\xa7" * size, size),
                ctypes.c_int(-1),
            )
            for panel in ("split", "span"):
                scores = scores_for(plan.shape, panel)
                source = ctypes.create_string_buffer(scores.astype("<u2").tobytes(), size)
                expected = assemble_row_softmax_outputs(
                    plan,
                    tuple(tuple(unpack_bf16_pair_halves(tile) for tile in group) for group in reference(plan, scores)),
                )
                starts = control["starts"]
                status = lib.call(source, out, ctypes.byref(device)) if panel == "split" else lib.dispatch(source, out)
                self.assertEqual((status, errors), (0, []))
                self.assertEqual(control["starts"] - starts, record.get("launches_per_invocation", len(plan.regions)))
                self.assertEqual(out.raw, expected)
                self.assertEqual(source.raw, scores.astype("<u2").tobytes())
                if composed:
                    other_out = ctypes.create_string_buffer(b"\xa7" * size, size)
                    starts = control["starts"]
                    self.assertEqual(lib.call_application(source, out, other_out), 0)
                    self.assertEqual((out.raw, other_out.raw, errors), (expected, expected, []))
                    self.assertEqual(control["starts"] - starts, 2 * len(plan.regions))
                if "source_interface" in record:
                    import numpy as np
                    from atlas_mlir_native.softmax_precision import input_spec

                    floats = (scores.astype("<u4") << 16).view("<f4")
                    perturbed = floats.view("<u4").copy()
                    perturbed[(perturbed & 0x7FFFFFFF) != 0] |= 0x7000
                    floats = perturbed.view("<f4")
                    codes = input_spec().encode(floats)
                    actual_source = ctypes.create_string_buffer(floats.tobytes(), size * 2)
                    float_output = ctypes.create_string_buffer(b"\xa7" * (size * 2), size * 2)
                    wanted = assemble_row_softmax_outputs(
                        plan,
                        tuple(
                            tuple(unpack_bf16_pair_halves(tile) for tile in group) for group in reference(plan, codes)
                        ),
                    )
                    wanted_float = (np.frombuffer(wanted, dtype="<u2").astype("<u4") << 16).tobytes()
                    status = (
                        lib.source_call(actual_source, float_output, ctypes.byref(device))
                        if panel == "split"
                        else lib.source_dispatch(actual_source, float_output)
                    )
                    self.assertEqual((status, errors, float_output.raw), (0, [], wanted_float))
                    self.assertEqual(actual_source.raw, floats.tobytes())
                    for bad_value in (np.nan, np.inf, 1e-37, 64.0):
                        bad_values = floats.copy()
                        bad_values[-1, -1] = bad_value
                        starts, writes = control["starts"], control["writes"]
                        self.assertEqual(
                            lib.source_call(
                                ctypes.create_string_buffer(bad_values.tobytes()), float_output, ctypes.byref(device)
                            ),
                            2,
                        )
                        self.assertEqual(
                            (control["starts"], control["writes"], float_output.raw), (starts, writes, wanted_float)
                        )
            unchanged = out.raw
            starts, writes = control["starts"], control["writes"]
            bad = ctypes.create_string_buffer(b"\x01\x00" + source.raw[2:], size)
            self.assertEqual(lib.call(bad, out, ctypes.byref(device)), 2)
            self.assertEqual(lib.call(source, source, ctypes.byref(device)), 1)
            self.assertEqual(lib.call(lib.workspace(), out, ctypes.byref(device)), 1)
            self.assertEqual(
                (control["starts"], control["writes"], out.raw),
                (starts, writes, unchanged),
            )
            control.update(timeout=True)
            self.assertEqual(lib.call(source, out, ctypes.byref(device)), 4)
            self.assertEqual(device.value, 4)  # ATLAS_HOST_TIMEOUT
            self.assertEqual(lib.active(), 1)
            if "source_interface" in record:
                scratch_bytes = record["source_interface"]["workspace_bytes"]
                scratch_before = ctypes.string_at(lib.source_workspace(), scratch_bytes)
                self.assertEqual(lib.source_call(actual_source, float_output, ctypes.byref(device)), 4)
                self.assertEqual(ctypes.string_at(lib.source_workspace(), scratch_bytes), scratch_before)
            starts, writes = control["starts"], control["writes"]
            self.assertEqual(lib.call(source, out, ctypes.byref(device)), 4)
            self.assertEqual(
                (control["starts"], control["writes"], out.raw),
                (starts, writes, unchanged),
            )
            control["timeout"] = False
            self.assertEqual(lib.finish(), 0)
            if len(plan.regions) > 1:
                scores = scores_for(plan.shape, "uniform")
                source = ctypes.create_string_buffer(scores.astype("<u2").tobytes(), size)
                expected = assemble_row_softmax_outputs(
                    plan,
                    tuple(tuple(unpack_bf16_pair_halves(tile) for tile in group) for group in reference(plan, scores)),
                )
                self.assertNotEqual(expected[: plan.shape[1] * 2], unchanged[: plan.shape[1] * 2])
                control.update(reads=0, fail_read=plan.column_tiles + 1)
                self.assertEqual(lib.call(source, out, ctypes.byref(device)), 3)
                self.assertEqual(out.raw, unchanged)  # First row group was private staging only.
                control["fail_read"] = 0
                self.assertEqual(lib.call(source, out, ctypes.byref(device)), 0)
                self.assertEqual(out.raw, expected)
            self.assertEqual(errors, [])
        print(json.dumps({"softmax_c_model_executions": executions, "rv64_execution": False}))

    def _write_composed_call(self, root, plan, package):
        """Compile two source regions and connect the normal application C ABI."""
        from atlas_mlir_native.binding import AtlasNativeBinding
        from atlas_mlir_native.row_softmax import prepare_row_softmax
        from atlas_mlir_native.softmax_c import prepare_c_package

        from merlin.compile.dispatch_c import DispatchCall, emit_program_c
        from merlin.compile.semantic_regions import outline_semantic_regions
        from merlin.runtime.program import build_program
        from merlin.semantic_compiler.search import SearchLimits
        from merlin.xdsl_dialects.lowering.arena_plan import ArenaRequirements
        from merlin.xdsl_dialects.lowering.global_plan import (
            BufferRepresentation,
            CycleInterval,
            GlobalPlan,
            RegionAlternative,
            ValueRepresentation,
        )
        from merlin.xdsl_dialects.lowering.global_plan_emission import dispatch_digest

        # Both consumers read one source tensor; all their results remain visible.
        second = dataclasses.replace(plan.source.nodes[1], id="second_probabilities")
        source = dataclasses.replace(
            plan.source,
            nodes=(*plan.source.nodes, second),
            outputs=(*plan.source.outputs, second.id),
            output_storages=("external",) * 2,
        )
        rep = ValueRepresentation("memory", "dense", "bf16")
        alternatives = tuple(
            RegionAlternative(
                f"row_{i}",
                (i,),
                symbol + "_dispatch",
                "device",
                CycleInterval.unknown("uncalibrated"),
                (BufferRepresentation(source.nodes[0].id, rep),),
                (BufferRepresentation(result, rep),),
            )
            for i, (symbol, result) in enumerate(zip(("layer", "other"), source.outputs, strict=True))
        )
        selected = GlobalPlan(alternatives, (), CycleInterval.unknown("uncalibrated"))
        outlined = outline_semantic_regions(source, entry="forward", plan=selected)
        for index, (key, region) in enumerate(outlined.regions):
            destination = root / key
            destination.mkdir()
            (destination / "source-request.json").write_text(json.dumps(region.record()))
            for row, tile in enumerate(prepare_row_softmax(region).regions):
                abi = AtlasNativeBinding.plan_boundary_abi(tile.request)
                AtlasNativeBinding.compile(
                    self.snapshot,
                    tile.request,
                    fixed_inputs=abi["fixed_inputs"],
                    fixed_outputs=tuple(abi["fixed_outputs"]),
                    target_source=ROOT,
                    destination=destination / f"compiled-{row}",
                    limits=SearchLimits(candidate_nodes=128),
                )
            _, files = prepare_c_package(
                destination / "source-request.json",
                destination,
                "layer" if index == 0 else "other",
                target_source=ROOT,
                runtime_linkage="separate",
            )
            for filename, data in files.items():
                if filename == "README.md":
                    continue
                path = package / filename
                if filename.startswith("runtime/") and path.exists():
                    self.assertEqual(path.read_bytes(), data)
                else:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(data)
        program = build_program(
            outlined.logical,
            global_plan=selected,
            global_plan_emitter=outlined,
            memory_requirements=lambda dispatch: ArenaRequirements(
                1, (0, 1), (0, 1), alignment=1, dispatch_digest=dispatch_digest(dispatch)
            ),
        )
        report, files = emit_program_c(
            program,
            "application",
            {i: DispatchCall(row.implementation, "device", context_index=i) for i, row in enumerate(alternatives)},
        )
        (package / "application.json").write_text(json.dumps(report, indent=2))
        for filename, text in files.items():
            (package / filename).write_text(text)
        (package / "application_binding.c").write_text("""#include "layer.h"
#include "other.h"
#include "application.h"
static atlas_host host;
static layer_context first;
static other_context second;
static unsigned char scratch1[LAYER_WORKSPACE_BYTES], scratch2[OTHER_WORKSPACE_BYTES];
static application_state state;
int bind_application(atlas_host_read32 rd, atlas_host_write32 wr, atlas_tile_write put, atlas_tile_read get) {
  if (atlas_host_init(&host,(atlas_host_map){0x20000,0x40000,131072},
      (atlas_host_mmio){rd,wr,0},(atlas_host_memory){1,0,0})) return -1;
  if (layer_bind(&first,&host,(atlas_tile_memory_io){put,get,0},scratch1,sizeof(scratch1),4)) return -2;
  return other_bind(&second,&host,(atlas_tile_memory_io){put,get,0},scratch2,sizeof(scratch2),4);
}
int call_application(void *input, void *out1, void *out2) {
  void *items[]={&first,&second}; application_contexts contexts={items,2};
  application_buffer buffers[]={{input,130},{out1,130},{out2,130}};
  return application_run(&state,0,0,buffers,3,&contexts);
}
""")


if __name__ == "__main__":
    unittest.main()
