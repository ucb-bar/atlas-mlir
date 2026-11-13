"""Exact source grouping, slicing, publication, and C export for quantizers."""
from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import numpy as np

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
    def test_static_full_and_tail_chunks_keep_every_byte(self):
        source = quantizer_source(count=4103, fixed=-9)
        plan = prepare_plan(source)
        self.assertEqual([item["code_bytes"] for item in plan.launches], [2048, 2048, 7])
        self.assertEqual(len(plan.bodies), 2)
        raw = np.resize(np.array([0, 0x8000, 1, 127, 0x3f88, 0x3ff8, 0x7f7f, 0x7f80, 0xff80, 0x7fc1], dtype="<u2"), 4103).tobytes()
        self.assertEqual(scheduled_reference(source, raw), host_reference(source, raw))

    def test_dynamic_whole_groups_all_reductions_rules_and_tails(self):
        for reduction in ("tensor", "head", "token", "block", "head_block"):
            for rule in ("ref", "nc"):
                with self.subTest(reduction=reduction, rule=rule):
                    source = dynamic_source(shape=(2, 5, 7), reduce=reduction, rows=3, rule=rule)
                    raw = np.resize(np.array([0, 0x8000, 1, 127, 0x3f88, 0x3ff8, 0x7f7f, 0x7f80, 0xff80, 0x7fc1], dtype="<u2"), 70).tobytes()
                    self.assertEqual(scheduled_reference(source, raw), host_reference(source, raw))
        plan = prepare_plan(dynamic_source(shape=(2, 5, 7), reduce="head_block", rows=3))
        self.assertEqual([item["code_bytes"] for item in plan.launches], [21, 14, 21, 14])
        self.assertEqual([item["exponent_offset_bytes"] for item in plan.launches], [0, 12, 20, 32])

    def test_f32_and_reversed_source_output_order(self):
        source = dynamic_source(shape=(2, 3, 7), reduce="head", dtype="bf16")
        source = replace(source, outputs=tuple(reversed(source.outputs)))
        raw = np.resize(np.array([0, 0x80000000, 0x3f888000, 0x7f7fffff, 0x7f800000, 0x7f800001], dtype="<u4"), 42).tobytes()
        self.assertEqual(scheduled_reference(source, raw), host_reference(source, raw))
        self.assertEqual(prepare_plan(source).source.outputs, source.outputs)

    def test_large_application_geometry_reuses_bodies(self):
        from merlin.semantic_compiler.codegen import CodegenOptions, codegen_options
        with codegen_options(CodegenOptions(io_memory_bytes=4325376)):
            static = prepare_plan(quantizer_source(count=64 * 5632))
        self.assertEqual(len(static.launches), 1)
        self.assertEqual(len(static.bodies), 1)
        dynamic = prepare_plan(dynamic_source(shape=(1, 32, 64, 64), reduce="head"))
        self.assertEqual(len(dynamic.launches), 4)
        self.assertEqual(len(dynamic.bodies), 1)
        self.assertTrue(all(item["code_bytes"] == 32768 and item["exponent_bytes"] == 32 for item in dynamic.launches))

    def test_irregular_head_tile_counts_do_not_use_full_tile_batch_budget(self):
        for length in (1056, 2016, 4064):
            plan = prepare_plan(dynamic_source(shape=(10, 1, length), reduce="head"))
            self.assertEqual(len(plan.launches), 10)
            self.assertTrue(all(item["code_bytes"] == length for item in plan.launches))
        for length in (64, 96, 320, 512, 992, 1024, 3072, 4096):
            plan = prepare_plan(dynamic_source(shape=(10, 1, length), reduce="head"))
            self.assertEqual(len(plan.launches), 2)

    def test_largest_static_prefix_extent_keeps_measured_whole_body(self):
        from merlin.semantic_compiler.codegen import CodegenOptions, codegen_options
        with codegen_options(CodegenOptions(io_memory_bytes=4325376)):
            plan = prepare_plan(quantizer_source(count=524256))
        self.assertEqual(len(plan.launches), 1)
        self.assertEqual(plan.launches[0]["code_bytes"], 524256)

    def test_packed_heads_preserve_independent_exponents_and_remainder(self):
        source = dynamic_source(shape=(10, 16, 64), reduce="head", rule="nc")
        plan = prepare_plan(source)
        self.assertEqual([item["exponent_bytes"] for item in plan.launches], [32, 8])
        values = np.resize(np.array([0, 1, 127, 128, 0x3f88, 0x3ff8, 0x7f7f, 0x7f80, 0xff80, 0x7fc1], dtype="<u2"), 10240)
        self.assertEqual(scheduled_reference(source, values.tobytes()), host_reference(source, values.tobytes()))

    def test_small_selected_io_retains_static_chunk_fallback(self):
        from merlin.semantic_compiler.codegen import CodegenOptions, codegen_options
        with codegen_options(CodegenOptions(io_memory_bytes=8192)):
            plan = prepare_plan(quantizer_source(count=65536))
        self.assertEqual(len(plan.launches), 32)
        self.assertTrue(all(item['code_bytes'] == 2048 for item in plan.launches))

    def test_oversized_dynamic_group_refused_without_splitting_amax(self):
        for source in (dynamic_source(shape=(2, 5000), reduce="token"),
                       dynamic_source(shape=(2, 1, 4128), reduce="head"),
                       dynamic_source(shape=(2, 33, 1024), reduce="head"),
                       dynamic_source(shape=(2, 17, 1024), reduce="tensor")):
            with self.assertRaisesRegex(ValueError, "indivisible dynamic reduction group"):
                prepare_plan(source)

    def test_window512_heads_keep_one_complete_group_per_launch(self):
        source = dynamic_source(shape=(1, 32, 512, 64), reduce="head")
        plan = prepare_plan(source)
        self.assertEqual(len(plan.launches), 32)
        self.assertEqual(len(plan.bodies), 1)
        self.assertEqual(plan.bodies[0].nodes[0].type.shape, (1, 512, 64))
        for head, launch in enumerate(plan.launches):
            self.assertEqual(launch["input_offset_bytes"], head * 65536)
            self.assertEqual(launch["input_bytes"], 65536)
            self.assertEqual(launch["code_offset_bytes"], head * 32768)
            self.assertEqual(launch["code_bytes"], 32768)
            self.assertEqual(launch["exponent_offset_bytes"], head * 4)
            self.assertEqual(launch["exponent_bytes"], 4)

    def test_large_head_batching_scales_the_complete_group_budget(self):
        for length, packed in ((8192, 4), (16384, 2), (24576, 1), (32768, 1)):
            with self.subTest(length=length):
                plan = prepare_plan(dynamic_source(shape=(8, 1, length), reduce="head"))
                self.assertEqual(len(plan.launches), 8 // packed)
                self.assertTrue(all(item["code_bytes"] == packed * length for item in plan.launches))


@unittest.skipUnless((BIN / "atlas-opt").is_file(), "select built Atlas tools")
class QuantizerApplicationCompilationTest(unittest.TestCase):
    def test_multiple_bodies_default_proof_c_export_and_source_seal(self):
        from atlas_mlir_native.activation_quantize_application import _static_body_capacity
        from atlas_mlir_native.limits import functional_instruction_limit
        source = quantizer_source(count=_static_body_capacity(functional_instruction_limit()) + 32)
        with tempfile.TemporaryDirectory(prefix="atlas-quantizer-schedule-") as directory:
            root = Path(directory)
            compiled = root / "compiled"
            manifest = compile_source(None, source, target_source=ROOT, destination=compiled, limits=None)
            self.assertEqual(manifest["artifact_kind"], "program_set")
            self.assertEqual(len(manifest["segments"]), 2)
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
            compile_source(None, source, target_source=ROOT, destination=compiled, limits=None)
            region = prepare_c_package(source, compiled, "head4096", target_source=ROOT)
            self.assertEqual(region.description["launches_per_invocation"], 1)


if __name__ == "__main__":
    unittest.main()
