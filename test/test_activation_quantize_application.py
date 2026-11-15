"""Exact source grouping, slicing, publication, and C export for quantizers."""

from dataclasses import replace
import ctypes
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from merlin.semantic_compiler.codegen import codegen_options, current_codegen_options
from atlas_mlir_native.activation_quantize_application import compile_source, prepare_c_package, prepare_plan
from test_already_quantized import quantizer_source
from test_activation_quantize_device import dynamic_source, host_reference

ROOT = Path(__file__).resolve().parents[1]
BIN = Path(os.environ.get("ATLAS_OOT_BIN_DIR", ROOT / "build/bin"))


def scheduled_reference(source, raw):
    plan = prepare_plan(source)
    codes, exponents = bytearray(plan.count), bytearray(int(np.prod(plan.exponents.type.shape)) * 4)
    for launch in plan.launches:
        first = launch["input_offset_bytes"]
        code, exp = host_reference(plan.bodies[launch["body_index"]], raw[first:first + launch["input_bytes"]])
        at = launch["code_offset_bytes"]
        codes[at:at + launch["code_bytes"]] = code
        at = launch["exponent_offset_bytes"]
        exponents[at:at + launch["exponent_bytes"]] = exp
    return bytes(codes), bytes(exponents)


class QuantizerApplicationScheduleTest(unittest.TestCase):
    def test_captured_schedule_record_and_replay_ignore_later_recipe_selection(self):
        sources = (quantizer_source(count=524320),
                   dynamic_source(shape=(1, 64, 64), reduce="head"))
        for pipeline in ("mandatory", "optimized"):
            for static_sizing in (False, True):
                with self.subTest(pipeline=pipeline, static_sizing=static_sizing), \
                        codegen_options(replace(current_codegen_options(), io_memory_bytes=67108864)):
                    for source in sources:
                        with codegen_options(replace(current_codegen_options(), native_pipeline=pipeline,
                                                     static_quantizer_sizing=static_sizing)):
                            plan = prepare_plan(source)
                            captured = plan.record()
                        other = "optimized" if pipeline == "mandatory" else "mandatory"
                        with codegen_options(replace(current_codegen_options(), native_pipeline=other,
                                                     static_quantizer_sizing=not static_sizing)):
                            self.assertEqual(plan.record(), captured)
                            replay = prepare_plan(source,
                                _pipeline=captured.get("lowering_pipeline", "mandatory"),
                                _static_sizing="static_instruction_envelope" in captured,
                                _static_capacity=captured["static_whole_body_max_elements"])
                            self.assertEqual(replay.record(), captured)

    @patch("atlas_mlir_native.activation_quantize_device.prove_plan")
    def test_dynamic_replay_proves_with_captured_pipeline(self, proof):
        source = dynamic_source(shape=(2, 64, 64), reduce="head")
        for captured_pipeline in ("mandatory", "optimized"):
            other = "optimized" if captured_pipeline == "mandatory" else "mandatory"
            with self.subTest(pipeline=captured_pipeline), patch.dict(os.environ, ATLAS_NATIVE_PIPELINE=other):
                proof.reset_mock()
                plan = prepare_plan(source, _pipeline=captured_pipeline)
                self.assertTrue(proof.called)
                self.assertEqual(plan.pipeline, captured_pipeline)
                self.assertTrue(all(call.kwargs["_pipeline"] == captured_pipeline
                                    for call in proof.call_args_list))

    def test_static_full_and_tail_chunks_keep_every_byte(self):
        source = quantizer_source(count=4103, fixed=-9)
        plan = prepare_plan(source)
        self.assertEqual([item["code_bytes"] for item in plan.launches], [2048, 2048, 7])
        self.assertEqual(len(plan.bodies), 2)
        raw = np.resize(np.array([0, 0x8000, 1, 127, 0x3f88, 0x3ff8, 0x7f7f, 0x7f80, 0xff80, 0x7fc1], dtype="<u2"), 4103).tobytes()
        self.assertEqual(scheduled_reference(source, raw), host_reference(source, raw))

    @patch("atlas_mlir_native.activation_quantize_device.prove_plan")
    def test_dynamic_whole_groups_all_reductions_rules_and_tails(self, _proof):
        for reduction in ("tensor", "head", "token", "block", "head_block"):
            for rule in ("ref", "nc"):
                with self.subTest(reduction=reduction, rule=rule):
                    source = dynamic_source(shape=(2, 5, 7), reduce=reduction, rows=3, rule=rule)
                    raw = np.resize(np.array([0, 0x8000, 1, 127, 0x3f88, 0x3ff8, 0x7f7f, 0x7f80, 0xff80, 0x7fc1], dtype="<u2"), 70).tobytes()
                    self.assertEqual(scheduled_reference(source, raw), host_reference(source, raw))
        plan = prepare_plan(dynamic_source(shape=(2, 5, 7), reduce="head_block", rows=3))
        self.assertEqual([item["code_bytes"] for item in plan.launches], [21, 14, 21, 14])
        self.assertEqual([item["exponent_offset_bytes"] for item in plan.launches], [0, 12, 20, 32])

    @patch("atlas_mlir_native.activation_quantize_device.prove_plan")
    def test_f32_and_reversed_source_output_order(self, _proof):
        source = dynamic_source(shape=(2, 3, 7), reduce="head", dtype="bf16")
        source = replace(source, outputs=tuple(reversed(source.outputs)))
        raw = np.resize(np.array([0, 0x80000000, 0x3f888000, 0x7f7fffff, 0x7f800000, 0x7f800001], dtype="<u4"), 42).tobytes()
        self.assertEqual(scheduled_reference(source, raw), host_reference(source, raw))
        self.assertEqual(prepare_plan(source).source.outputs, source.outputs)

    @unittest.skipUnless((BIN / "atlas-opt").is_file(), "select built Atlas tools")
    def test_large_application_geometry_reuses_bodies(self):

        with codegen_options(replace(current_codegen_options(), io_memory_bytes=4325376)):
            static = prepare_plan(quantizer_source(count=64 * 5632))
        self.assertEqual(len(static.launches), 1)
        self.assertEqual(len(static.bodies), 1)
        for pipeline, packs in (("mandatory", [8] * 4), ("optimized", [8] * 4)):
            with self.subTest(pipeline=pipeline), patch.dict(os.environ, ATLAS_NATIVE_PIPELINE=pipeline):
                dynamic = prepare_plan(dynamic_source(shape=(1, 32, 64, 64), reduce="head"))
                self.assertEqual(len(dynamic.launches), len(packs))
                self.assertEqual(len(dynamic.bodies), len(set(packs)))
                self.assertEqual([item["code_bytes"] for item in dynamic.launches], [p * 4096 for p in packs])
                self.assertEqual([item["exponent_bytes"] for item in dynamic.launches], [p * 4 for p in packs])

    @patch("atlas_mlir_native.activation_quantize_device.prove_plan")
    def test_irregular_head_tile_counts_do_not_use_full_tile_batch_budget(self, _proof):
        for length in (1056, 2016, 4064):
            plan = prepare_plan(dynamic_source(shape=(10, 1, length), reduce="head"))
            self.assertEqual(len(plan.launches), 10)
            self.assertTrue(all(item["code_bytes"] == length for item in plan.launches))
        for length in (64, 96, 320, 512, 992, 1024, 3072, 4096):
            plan = prepare_plan(dynamic_source(shape=(10, 1, length), reduce="head"))
            self.assertEqual(len(plan.launches), 2)

    def test_largest_static_prefix_extent_keeps_measured_whole_body(self):

        with codegen_options(replace(current_codegen_options(), io_memory_bytes=4325376)):
            plan = prepare_plan(quantizer_source(count=524256))
        self.assertEqual(len(plan.launches), 1)
        self.assertEqual(plan.launches[0]["code_bytes"], 524256)

    @patch("atlas_mlir_native.activation_quantize_device.prove_plan")
    def test_packed_heads_preserve_independent_exponents_and_remainder(self, _proof):
        source = dynamic_source(shape=(10, 16, 64), reduce="head", rule="nc")
        plan = prepare_plan(source)
        self.assertEqual([item["exponent_bytes"] for item in plan.launches], [32, 8])
        values = np.resize(np.array([0, 1, 127, 128, 0x3f88, 0x3ff8, 0x7f7f, 0x7f80, 0xff80, 0x7fc1], dtype="<u2"), 10240)
        self.assertEqual(scheduled_reference(source, values.tobytes()), host_reference(source, values.tobytes()))

    def test_small_selected_io_retains_static_chunk_fallback(self):

        with codegen_options(replace(current_codegen_options(), io_memory_bytes=8192)):
            plan = prepare_plan(quantizer_source(count=65536))
        self.assertEqual(len(plan.launches), 32)
        self.assertTrue(all(item['code_bytes'] == 2048 for item in plan.launches))

    def test_oversized_dynamic_group_refused_without_splitting_amax(self):
        for source in (dynamic_source(shape=(2, 5000), reduce="token"),
                       dynamic_source(shape=(2, 1, 4129), reduce="head"),
                       dynamic_source(shape=(2, 33, 1024), reduce="head"),
                       dynamic_source(shape=(2, 17, 1024), reduce="tensor")):
            with self.assertRaisesRegex(ValueError, "indivisible dynamic reduction group"):
                prepare_plan(source)

    @unittest.skipUnless((BIN / "atlas-opt").is_file(), "select built Atlas tools")
    def test_decode_heads_preserve_complete_reduction_groups(self):
        source = dynamic_source(shape=(1, 32, 65, 64), reduce="head", rule="ref", rows=32)
        plan = prepare_plan(source)
        self.assertEqual(len(plan.launches), 32)
        self.assertEqual(len(plan.bodies), 1)
        self.assertEqual(plan.bodies[0].nodes[0].type.shape, (1, 65, 64))
        for head, launch in enumerate(plan.launches):
            self.assertEqual(launch["input_offset_bytes"], head * 8320)
            self.assertEqual(launch["input_bytes"], 8320)
            self.assertEqual(launch["code_offset_bytes"], head * 4160)
            self.assertEqual(launch["code_bytes"], 4160)
            self.assertEqual(launch["exponent_offset_bytes"], head * 4)
            self.assertEqual(launch["exponent_bytes"], 4)

    @unittest.skipUnless((BIN / "atlas-opt").is_file(), "select built Atlas tools")
    def test_window496_heads_keep_one_complete_group_per_launch(self):
        source = dynamic_source(shape=(1, 32, 496, 64), reduce="head")
        plan = prepare_plan(source)
        self.assertEqual(len(plan.launches), 32)
        self.assertEqual(len(plan.bodies), 1)
        self.assertEqual(plan.bodies[0].nodes[0].type.shape, (1, 496, 64))
        for head, launch in enumerate(plan.launches):
            self.assertEqual(launch["input_offset_bytes"], head * 63488)
            self.assertEqual(launch["input_bytes"], 63488)
            self.assertEqual(launch["code_offset_bytes"], head * 31744)
            self.assertEqual(launch["code_bytes"], 31744)
            self.assertEqual(launch["exponent_offset_bytes"], head * 4)
            self.assertEqual(launch["exponent_bytes"], 4)

    @unittest.skipUnless((BIN / "atlas-opt").is_file(), "select built Atlas tools")
    def test_large_head_batching_scales_the_complete_group_budget(self):
        from merlin.semantic_compiler.codegen import CodegenOptions, codegen_options

        for length, packed in ((8192, 4), (16384, 2), (24576, 1), (31744, 1)):
            with self.subTest(length=length):
                source = dynamic_source(shape=(8, 1, length), reduce="head")
                default = prepare_plan(source)
                self.assertTrue(all(item["code_bytes"] <= packed * length for item in default.launches))
                self.assertEqual(sum(item["code_bytes"] for item in default.launches), 8 * length)
                with codegen_options(CodegenOptions(execution_instruction_limit=1100000)):
                    plan = prepare_plan(source)
                self.assertEqual(len(plan.launches), 8 // packed)
                self.assertTrue(all(item["code_bytes"] == packed * length for item in plan.launches))

    @unittest.skipUnless((BIN / "atlas-opt").is_file(), "select built Atlas tools")
    @patch.dict(os.environ, ATLAS_NATIVE_PIPELINE="optimized")
    def test_machine_bound_rejects_large_aligned_groups_during_planning(self):
        from atlas_mlir_native.application import AtlasApplicationAdapter
        from merlin.semantic_compiler.codegen import CodegenOptions, codegen_options

        for length in (32736,):
            with self.subTest(length=length):
                source = dynamic_source(shape=(1, 1, length), reduce="head", rule="ref")
                with self.assertRaisesRegex(ValueError, "indivisible dynamic reduction group.*instruction bound"):
                    prepare_plan(source)
                self.assertFalse(AtlasApplicationAdapter._is_vector_quantizer(source, source.outputs))
                with codegen_options(CodegenOptions(execution_instruction_limit=128000000)):
                    self.assertEqual(len(prepare_plan(source).launches), 1)

    @unittest.skipUnless((BIN / "atlas-opt").is_file(), "select built Atlas tools")
    def test_full_tile_head32768_admits_under_default_bound(self):
        from atlas_mlir_native.activation_quantize_device import prepare_plan as body_plan, prove_plan
        source = dynamic_source(shape=(1, 1, 32768), reduce="head", rule="ref")
        for pipeline in ("mandatory", "optimized"):
            with self.subTest(pipeline=pipeline), patch.dict(os.environ, ATLAS_NATIVE_PIPELINE=pipeline):
                plan = prepare_plan(source)
                self.assertEqual(len(plan.launches), 1)
                bound = prove_plan(body_plan(plan.bodies[0]))[3]
                self.assertTrue(bound["admitted"])
                self.assertLessEqual(bound["instruction_bound"], 1000000)

    @unittest.skipUnless((BIN / "atlas-opt").is_file(), "select built Atlas tools")
    def test_large_irregular_groups_use_actual_bound_and_selected_limit(self):
        from types import SimpleNamespace

        for length in (4128, 27680, 29760):
            with self.subTest(length=length):
                source = dynamic_source(shape=(1, 1, length), reduce="head", rule="ref")
                plan = prepare_plan(source)
                self.assertEqual(len(plan.launches), 1)
                from atlas_mlir_native.activation_quantize_device import prepare_plan as body_plan, prove_plan
                bound = prove_plan(body_plan(plan.bodies[0]))[3]["instruction_bound"]
                with self.assertRaisesRegex(ValueError, "indivisible dynamic reduction group"):
                    prepare_plan(source, limits=SimpleNamespace(functional_instruction_limit=bound - 1))


@unittest.skipUnless((BIN / "atlas-opt").is_file(), "select built Atlas tools")
class QuantizerApplicationCompilationTest(unittest.TestCase):
    def test_multiple_bodies_default_proof_c_export_and_source_seal(self):
        with codegen_options(replace(current_codegen_options(), native_pipeline="optimized", static_quantizer_sizing=True)):
            from atlas_mlir_native.activation_quantize_application import _static_body_capacity
            from atlas_mlir_native.limits import functional_instruction_limit
            source = quantizer_source(count=_static_body_capacity(functional_instruction_limit()) + 32)
            with tempfile.TemporaryDirectory(prefix="atlas-quantizer-schedule-") as directory:
                root = Path(directory)
                compiled = root / "compiled"
                manifest = compile_source(None, source, target_source=ROOT, destination=compiled, limits=None)
                self.assertEqual(manifest["artifact_kind"], "program_set")
                self.assertEqual(len(manifest["segments"]), 2)
                with codegen_options(replace(current_codegen_options(), native_pipeline="mandatory")):
                    region = prepare_c_package(source, compiled, "quant", target_source=ROOT)
                self.assertEqual(region.outputs, source.outputs)
                self.assertEqual(region.description["launches_per_invocation"], 2)
                self.assertEqual(region.description["device_launch_schedule"]["total_launches"], 2)
                self.assertNotIn("launches_per_call", region.description)
                self.assertEqual(region.programs, ("segments/quant/segments/body_000", "segments/quant/segments/body_001"))
                # An application flattens these image leaves. Its existing branch
                # verifier must accept all authenticated scheduling metadata.
                from hashlib import sha256
                from merlin.semantic_compiler.target_binding import verify_native_publication
                parent = root / "application"
                branch = parent / "segments/quant"
                branch.parent.mkdir(parents=True)
                shutil.copytree(compiled, branch)
                parent_rows = [{**row, "path": "segments/quant/" + row["path"]} for row in manifest["segments"]]
                artifacts = {"segments/quant/" + filename: sha256((compiled / filename).read_bytes()).hexdigest()
                             for filename in ("source-request.json", "quantizer-schedule.json", "quantizer-manifest.json")}
                parent_manifest = dict(engine="merlin_native", artifact_kind="program_set", request_digest=source.digest(),
                                       target_identity=source.target_identity, segments=parent_rows, stage_artifacts=artifacts)
                (parent / "manifest.json").write_text(json.dumps(parent_manifest))
                verify_native_publication(parent, parent_manifest, engine="merlin_native", request_digest=source.digest(), target_identity=source.target_identity)
                package = root / "c"
                for filename, data in region.files.items():
                    path = package / filename
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(data)
                if shutil.which("cc"):
                    subprocess.run(["cc", "-std=c11", "-Wall", "-Wextra", "-Werror", "-fsyntax-only", "quant.c"], cwd=package, check=True, capture_output=True)
                # Rehashing both the source and mutable manifest must not let a
                # different source/body slice mapping through export replay.
                schedule = compiled / "quantizer-schedule.json"
                record = json.loads(schedule.read_text())
                record["launches"][0]["code_offset_bytes"] = 1
                schedule.write_text(json.dumps(record))
                manifest["stage_artifacts"]["quantizer-schedule.json"] = sha256(schedule.read_bytes()).hexdigest()
                (compiled / "quantizer-manifest.json").write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError, "slice mapping"):
                    prepare_c_package(source, compiled, "changed", target_source=ROOT)

    def test_dynamic_head4096_under_default_proof_and_c_export(self):
        source = dynamic_source(shape=(1, 64, 64), reduce="head", dtype="bf16")
        with tempfile.TemporaryDirectory(prefix="atlas-quantizer-head4096-") as directory:
            compiled = Path(directory) / "compiled"
            with codegen_options(replace(current_codegen_options(), native_pipeline="mandatory")):
                compile_source(None, source, target_source=ROOT, destination=compiled, limits=None)
            with codegen_options(replace(current_codegen_options(), native_pipeline="optimized")):
                region = prepare_c_package(source, compiled, "head4096", target_source=ROOT)
            self.assertEqual(region.description["launches_per_invocation"], 1)

    def test_dynamic_head4160_under_default_proof_and_c_export(self):
        source = dynamic_source(shape=(1, 65, 64), reduce="head", dtype="bf16")
        with tempfile.TemporaryDirectory(prefix="atlas-quantizer-head4160-") as directory:
            compiled = Path(directory) / "compiled"
            compile_source(None, source, target_source=ROOT, destination=compiled, limits=None)
            region = prepare_c_package(source, compiled, "head4160", target_source=ROOT)
            self.assertEqual(region.description["launches_per_invocation"], 1)


@unittest.skipUnless(os.environ.get("ATLAS_RUN_NPU_MODEL") == "1", "select pinned npu_model")
class QuantizerApplicationModelTest(unittest.TestCase):
    def test_large_irregular_groups_codes_exponents_padding_and_guards(self):
        import torch
        from atlas_mlir_native import vector_quantize as device
        from atlas_mlir_native.limits import functional_instruction_limit

        model = Path(os.environ["ATLAS_NPU_MODEL_ROOT"])
        sys.path[:0] = [str(model), str(ROOT), str(ROOT / "tools"),
                       str(ROOT.parent / "merlin/examples/atlas/whole-model/unchanged_fx")]
        from functional_runtime import FunctionalModelRuntime
        from model_runtime import READ, WRITE, MEMORY

        torch.set_num_threads(1)
        # Both groups use 32-lane tiles; the second is close to the default
        # machine-bound limit. The complete group must select one exponent.
        for length in (4128, 27680):
            with self.subTest(length=length), tempfile.TemporaryDirectory(prefix="atlas-quantizer-model-") as directory:
                root = Path(directory)
                source = dynamic_source(shape=(1, 1, length), reduce="head", rule="ref")
                compiled = root / "compiled"
                manifest = compile_source(None, source, target_source=ROOT, destination=compiled, limits=None)
                exported = prepare_c_package(source, compiled, "quant_test", target_source=ROOT)
                for filename, data in exported.files.items():
                    path = root / filename
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(data)
                (root / "harness.c").write_text('''#include "quant_test.h"
static quant_test_context region; static quant_test_source_context source;
static uint8_t workspace[QUANT_TEST_WORKSPACE_BYTES]; static atlas_host host;
int run(atlas_host_read32 read,atlas_host_write32 write,atlas_tile_write put,atlas_tile_read get,const void *const *inputs,void *const *outputs) {
 if(atlas_host_init(&host,(atlas_host_map){0x20000u,0x40000u,131072u},(atlas_host_mmio){read,write,0},(atlas_host_memory){1,0,0}))return 9;
 if(quant_test_bind(&region,&host,(atlas_tile_memory_io){put,get,0},workspace,sizeof(workspace),8u))return 8;
 if(quant_test_source_bind(&source,&region,0,0))return 7;
 return quant_test_source_dispatch(&source,inputs,outputs);
}
''')
                subprocess.run(["cc", "-std=c11", "-Wall", "-Wextra", "-Werror", "-shared", "-fPIC", "-O2",
                                "quant_test.c", "runtime/atlas_host.c", "harness.c", "-o", "test.so"],
                               cwd=root, check=True, capture_output=True)
                library = ctypes.CDLL(str(root / "test.so"))
                function = library.run
                function.argtypes = [READ, WRITE, MEMORY, MEMORY, ctypes.POINTER(ctypes.c_void_p),
                                     ctypes.POINTER(ctypes.c_void_p)]
                images, plans = [], {}
                for segment in manifest["segments"]:
                    leaf = compiled / segment["path"]
                    binary = (leaf / "program.bin").read_bytes()
                    images.append(binary)
                    plans[sha256(binary).hexdigest()] = json.loads((leaf / "execution_plan.json").read_text())
                plan = device.prepare_plan(prepare_plan(source).bodies[0])
                runtime = FunctionalModelRuntime(ROOT, model, images, root / "run",
                    memory_bytes=plan.workspace_bytes + 64, image_plans=plans, trace_payloads=False,
                    compact_dram_prefix=False,
                    max_instructions=functional_instruction_limit())
                finite = np.resize(np.array([0, 0x8000, 1, 127, 128, 0x3f88, 0x3f98, 0x3ff8,
                                             0xbf88, 0xbff8], dtype="<u2"), length)
                exceptional = finite.copy()
                exceptional[::19], exceptional[::31] = 0x7f80, 0xffc1
                # A finite maximum in a mixed final tile catches both splitting
                # amax and omitting the finite-only exceptional-lane correction.
                exceptional[-3:] = (0x7f80, 0x7f7f, 0xffc1)
                subnormal = np.resize(np.array([0x7f80, 0xff80, 0x7fc1, 0xffff], dtype="<u2"), length)
                subnormal[-2] = 1
                ca, ea = plan.code_address - plan.io_memory_base, plan.exponent_address - plan.io_memory_base
                for payload in (finite, exceptional, subnormal):
                    raw = payload.tobytes()
                    code, exp = host_reference(source, raw)
                    initial = bytearray([0xa5]) * len(runtime.memory)
                    initial[:plan.input_bytes] = raw + bytes(plan.input_bytes - len(raw))
                    runtime.memory[:] = initial
                    expected = bytearray(initial)
                    expected[ca:ea] = code + bytes(ea - ca - len(code))
                    expected[ea:plan.workspace_bytes] = exp + bytes(plan.exponent_bytes - len(exp))
                    input_ = ctypes.create_string_buffer(raw)
                    codes = ctypes.create_string_buffer(bytes([0xa5]) * len(code))
                    exponents = ctypes.create_string_buffer(bytes([0xa5]) * len(exp))
                    inputs = (ctypes.c_void_p * 1)(ctypes.addressof(input_))
                    outputs = (ctypes.c_void_p * 2)(ctypes.addressof(codes), ctypes.addressof(exponents))
                    self.assertEqual(function(*runtime.callbacks, inputs, outputs), 0, runtime.errors)
                    self.assertEqual((codes.raw[:-1], exponents.raw[:-1]), (code, exp))
                    self.assertEqual(bytes(runtime.memory), bytes(expected))
                result = runtime.finish()
                self.assertFalse(result["errors"])
                self.assertEqual(len(result["launches"]), 3)


if __name__ == "__main__":
    unittest.main()
