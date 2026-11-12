"""Explicit activation reuse keeps typed programs and ordered BF16 panels."""
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import tempfile
import shutil
import subprocess
from types import SimpleNamespace
from unittest.mock import patch
import unittest

from atlas_mlir_native.looped_tiling import _operand_reuse_policy, _virtual, _virtual_schedule, prepare_plan, tuning_key
from atlas_mlir_native.resident_looped_tiling import RESIDENT_A_TILES, schedule_record, virtual_resident_a
from merlin.semantic_compiler.target_binding import NativeCompilationError
from test_looped_mnk import chunked_case_data, seeded_case_data, seeded_source_request, source_request
from test_virtual_mxu_lowering import instructions
from test_virtual_ssa import BIN, ROOT, run


class ResidentLoopedMNKTest(unittest.TestCase):
    def test_policy_is_explicit_and_rejects_unrecognized_values(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(_operand_reuse_policy(), "ordered")
        for policy in ("ordered", "resident-a"):
            with patch.dict(os.environ, {"ATLAS_LOOPED_OPERAND_REUSE": policy}):
                self.assertEqual(_operand_reuse_policy(), policy)
        with patch.dict(os.environ, {"ATLAS_LOOPED_OPERAND_REUSE": "unknown"}):
            with self.assertRaisesRegex(NativeCompilationError, "ordered or resident-a"):
                _operand_reuse_policy()

    def test_panels_exceed_dma_windows_and_preserve_exact_bounded_allocation(self):
        for shape, seeded, relu in (((33, 35, 777), False, True),
                                    ((3, 35, 1025), True, True),
                                    ((1, 256, 2048), False, False)):
            with self.subTest(shape=shape, seeded=seeded, relu=relu):
                source = seeded_source_request(shape, relu) if seeded else source_request(*shape, relu=relu)
                plan = prepare_plan(source)
                record = schedule_record(plan)
                self.assertEqual(record["resident_a_panel_tiles"], 24)
                self.assertEqual(record["resident_a_panels"], (plan.counts[2] + 23) // 24)
                a = 0x90000000
                w = a + plan.buffer_bytes[0] + 2048
                seed = w + plan.buffer_bytes[1] + 2048 if seeded else None
                output = (seed + plan.buffer_bytes[2] if seeded else w + plan.buffer_bytes[1]) + 2048
                virtual = virtual_resident_a(plan, (a, w), output, source.nodes[-1] if relu else None,
                                            seed_address=seed)
                self.assertEqual(virtual.count('kind = "relu"'), int(relu))
                self.assertEqual(virtual.count('"atlas.virtual_mxu_readout_bf16"'), record["resident_a_panels"])
                compiled = run("atlas-opt", virtual, "--atlas-compile-unscheduled")
                self.assertEqual(compiled.returncode, 0, compiled.stderr)
                for check in ("--verify-atlas-generated-schedule", "--verify-atlas-machine-stream", "--verify-atlas-timing"):
                    checked = run("atlas-opt", compiled.stdout, check)
                    self.assertEqual(checked.returncode, 0, checked.stderr)
                entries = instructions(compiled.stdout)
                # Static MAC count is the complete ordered reduction; M/N
                # iteration stays in CFG loops rather than duplicating it.
                self.assertEqual(sum(row["operation"] == "atlas.mxu_matmul" for row in entries), plan.counts[2])
                self.assertEqual(sum(row["operation"] == "atlas.mxu_pop" for row in entries), record["resident_a_panels"])

    def test_default_bytes_single_column_fallback_and_optimized_refusal(self):
        for n in (7, 35):
            plan = prepare_plan(source_request(3, n, 777))
            addresses, output = (0x90000000, 0x90010000), 0x90030000
            with patch.dict(os.environ, {"ATLAS_LOOPED_OPERAND_REUSE": "ordered"}):
                virtual, record = _virtual_schedule(plan, addresses, output, k_unroll=4)
                self.assertEqual(virtual, _virtual(plan, addresses, output, k_unroll=4))
                self.assertEqual(record["operand_reuse"], "ordered")
            with patch.dict(os.environ, {"ATLAS_LOOPED_OPERAND_REUSE": "resident-a", "ATLAS_NATIVE_PIPELINE": "mandatory"}):
                virtual, record = _virtual_schedule(plan, addresses, output, k_unroll=4)
                self.assertEqual(record["operand_reuse"], "ordered" if n == 7 else "resident-a")
                if n == 7:
                    self.assertIn("operand_reuse_fallback", record)
                    self.assertEqual(virtual, _virtual(plan, addresses, output, k_unroll=4))
            if n > 32:
                with patch.dict(os.environ, {"ATLAS_LOOPED_OPERAND_REUSE": "resident-a", "ATLAS_NATIVE_PIPELINE": "optimized"}):
                    with self.assertRaisesRegex(NativeCompilationError, "mandatory only"):
                        _virtual_schedule(plan, addresses, output, k_unroll=4)

    def test_panel_size_obeys_fp8_register_budget(self):
        plan = prepare_plan(source_request(1, 33, 777))
        for bad in (0, RESIDENT_A_TILES + 1, True, 2.0):
            with self.subTest(panel_tiles=bad), self.assertRaises(ValueError):
                virtual_resident_a(plan, (0x90000000, 0x90010000), 0x90020000, panel_tiles=bad)

    def test_tuning_identity_records_operand_policy_and_resident_provider(self):
        from merlin.semantic_compiler.search import SearchLimits

        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as directory:
            root = Path(directory)
            (root / "manifest.json").write_text("snapshot identity")
            keys = []
            for policy in ("ordered", "resident-a"):
                with patch.dict(os.environ, {"ATLAS_LOOPED_OPERAND_REUSE": policy,
                                             "ATLAS_NATIVE_PIPELINE": "mandatory", "ATLAS_OOT_BIN_DIR": str(BIN)}):
                    keys.append(tuning_key(SimpleNamespace(root=root), source_request(3, 35, 777),
                                           {"a": 0, "w": 25}, (50,), SearchLimits()))
            self.assertNotEqual(keys[0], keys[1])
            self.assertEqual(keys[1]["operand_reuse"], "resident-a")
            self.assertIn("resident_looped_tiling.py", keys[1]["provider_sources"])

    @unittest.skipUnless(os.environ.get("ATLAS_RUN_NPU_MODEL") == "1", "set ATLAS_RUN_NPU_MODEL=1 for original47")
    def test_ordered_panel_cancellation_seed_relu_and_all_memory_bytes_on_original47(self):
        import torch
        from npu_model.configs import isa_definition as isa
        from npu_model.configs.hardware.default import DefaultHardwareConfig
        from npu_model.logging import LoggerConfig
        from npu_model.simulation import Simulation
        from npu_model.software.program import InstantiableProgram
        from atlas_mlir_native.layout import pack_bf16_matrix
        from merlin.semantic_compiler.ordered_mnk_tiling import materialize_looped_mnk_inputs
        from test_mxu1_cap448 import authenticate_model
        from tools.check_program_decode import decode_binary
        from tools.model_instructions import model_instruction

        torch.set_num_threads(1)
        model_root = Path(os.environ["NPU_MODEL_ROOT"])
        before = authenticate_model(model_root)
        inventory = json.loads((ROOT / "docs/selected-variant-inventory.json").read_text())
        shape = (3, 35, 777)  # 25 K tiles, with a short first and second panel.
        m, n, _ = shape

        def packed(logical):
            rows, columns = (m + 31) // 32 * 32, (n + 31) // 32 * 32
            padded = bytearray(rows * columns * 2)
            for row in range(m):
                padded[row * columns * 2:row * columns * 2 + n * 2] = logical[row * n * 2:(row + 1) * n * 2]
            return pack_bf16_matrix(bytes(padded), (rows, columns))

        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as directory:
            for seeded in (False, True):
                with self.subTest(seeded=seeded):
                    if seeded:
                        a, w, seed, expected = seeded_case_data(shape, shift=1, relu=True)
                        source = seeded_source_request(shape, True)
                        seed = packed(seed)
                    else:
                        a, w, expected = chunked_case_data(shape, relu=True)
                        source, seed = source_request(*shape, relu=True), None
                    plan = prepare_plan(source)
                    inputs = materialize_looped_mnk_inputs(plan, {"a": a, "w": w})
                    expected = packed(expected)
                    base = 0x90000000
                    ap = base + 2048
                    wp = ap + len(inputs[0]) + 2048
                    sp = wp + len(inputs[1]) + 2048 if seeded else None
                    output = (sp + len(seed) if seeded else wp + len(inputs[1])) + 2048
                    virtual = virtual_resident_a(plan, (ap, wp), output, source.nodes[-1], seed_address=sp)
                    compiled = run("atlas-opt", virtual, "--atlas-compile-unscheduled")
                    self.assertEqual(compiled.returncode, 0, compiled.stderr)
                    for check in ("--verify-atlas-generated-schedule", "--verify-atlas-machine-stream", "--verify-atlas-timing"):
                        checked = run("atlas-opt", compiled.stdout, check)
                        self.assertEqual(checked.returncode, 0, checked.stderr)
                    binary = b"".join(row["word_u32"].to_bytes(4, "little") for row in instructions(compiled.stdout))
                    decoded = decode_binary(inventory, binary)
                    program = InstantiableProgram([model_instruction(row, inventory, isa) for row in decoded["instructions"]])
                    program.dram_base = base
                    length = output - base + len(expected) + 4096
                    initial = bytearray((i * 19 + 31) % 256 for i in range(length))
                    for address, payload in zip((ap, wp, sp), (*inputs, seed)):
                        if payload is not None:
                            initial[address - base:address - base + len(payload)] = payload
                    program.memory_regions = [(0, torch.tensor(list(initial), dtype=torch.uint8))]
                    config = DefaultHardwareConfig()
                    config.arch_state_config = replace(config.arch_state_config, numerics="rtl", randomize_init=True, init_seed=213)
                    simulation = Simulation(config, LoggerConfig(filename=str(Path(directory) / f"model-seeded-{seeded}.json")), program, verbose=False)
                    state = simulation.core.arch_state
                    try:
                        state.vmem.fill_(0xcd)
                        simulation.run(max_cycles=500000)
                        self.assertTrue(state.halted)
                        self.assertEqual(simulation.get_stats().runtime_errors, 0)
                        initial[output - base:output - base + len(expected)] = expected
                        actual = bytes(state.read_dram(base, len(initial)).tolist())
                        self.assertEqual(actual, bytes(initial))
                    finally:
                        state.close()
        self.assertEqual(authenticate_model(model_root), before)


@unittest.skipUnless(os.environ.get("MERLIN_NATIVE_ROOT") and os.environ.get("ATLAS_OOT_BIN_DIR"),
                     "set MERLIN_NATIVE_ROOT and ATLAS_OOT_BIN_DIR for compile/export")
class ResidentLoopedCPackageTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from atlas_mlir_native.binding import AtlasNativeBinding
        from merlin.semantic_compiler.codegen import CodegenOptions, codegen_options
        from merlin.semantic_compiler.search import SearchLimits
        from merlin.semantic_compiler.snapshot import build_native_snapshot, packaged_egg_bridge_path

        cls.temporary = tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR"))
        cls.root = Path(cls.temporary.name)
        cls.compiled = cls.root / "compiled"
        snapshot = build_native_snapshot(AtlasNativeBinding.profile(), destination=cls.root / "snapshot",
            crate=packaged_egg_bridge_path(), cargo_target_dir=Path(os.environ.get("CARGO_TARGET_DIR", cls.root / "cargo")),
            source_revision="resident-looped-c-export-test")
        request = seeded_source_request((3, 35, 777), True)
        abi = AtlasNativeBinding.plan_boundary_abi(request)
        with patch.dict(os.environ, {"ATLAS_LOOPED_OPERAND_REUSE": "resident-a", "ATLAS_NATIVE_PIPELINE": "mandatory"}), \
                codegen_options(CodegenOptions(reduction_unroll="4")):
            AtlasNativeBinding.compile(snapshot, request, fixed_inputs=abi["fixed_inputs"],
                fixed_outputs=tuple(abi["fixed_outputs"]), target_source=ROOT, destination=cls.compiled,
                limits=SearchLimits())

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_complete_compile_export_and_native_c_compile_ignores_current_policy(self):
        from atlas_mlir_native.looped_c import prepare_c_package

        # Export must replay the authenticated publication rather than the
        # current environment, which may belong to a different compilation.
        with patch.dict(os.environ, {"ATLAS_LOOPED_OPERAND_REUSE": "unknown", "ATLAS_NATIVE_PIPELINE": "optimized"}):
            record, files = prepare_c_package(self.compiled / "source-request.json", self.compiled,
                                              "resident", target_source=ROOT)
        self.assertTrue(record)
        package = self.root / "c-package"
        package.mkdir(exist_ok=True)
        for name, data in files.items():
            path = package / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        subprocess.run([shutil.which("cc"), "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
                        "-c", "resident.c", "-o", "resident.o"], cwd=package, capture_output=True, check=True)

    def test_resealed_resident_policy_metadata_is_recomputed(self):
        from atlas_mlir_native.looped_c import prepare_c_package

        faults = {"operand_reuse": "ordered", "operand_reuse_requested": "unknown",
                  "resident_a_panel_tiles": 23, "resident_a_panels": 3,
                  "reduction_unroll_scope": "ordered_k_loop", "partial_accumulator": "fp8",
                  "operand_reuse_fallback": "unproved fallback"}
        original_manifest = json.loads((self.compiled / "manifest.json").read_text())
        for key, value in faults.items():
            with self.subTest(field=key), tempfile.TemporaryDirectory(dir=self.root) as directory:
                copied = Path(directory) / "compiled"
                copied.mkdir()
                for name in ("manifest.json", "program.bin", *original_manifest["stage_artifacts"]):
                    (copied / name).write_bytes((self.compiled / name).read_bytes())
                execution = json.loads((copied / "execution_plan.json").read_text())
                execution["codegen"][key] = value
                payload = (json.dumps(execution, sort_keys=True, indent=2) + "\n").encode()
                (copied / "execution_plan.json").write_bytes(payload)
                manifest = json.loads((copied / "manifest.json").read_text())
                manifest["execution_plan_sha256"] = hashlib.sha256(payload).hexdigest()
                manifest["stage_artifacts"]["execution_plan.json"] = manifest["execution_plan_sha256"]
                (copied / "manifest.json").write_text(json.dumps(manifest, sort_keys=True) + "\n")
                with self.assertRaisesRegex(ValueError, "operand reuse"):
                    prepare_c_package(copied / "source-request.json", copied, "changed", target_source=ROOT)

    def test_resealed_resident_source_fingerprint_and_pipeline_are_rejected(self):
        from atlas_mlir_native.looped_c import prepare_c_package

        original_manifest = json.loads((self.compiled / "manifest.json").read_text())
        for key, value in (("resident_loop_lowering_source_sha256", "00" * 32), ("lowering_pipeline", "optimized")):
            with self.subTest(field=key), tempfile.TemporaryDirectory(dir=self.root) as directory:
                copied = Path(directory) / "compiled"
                copied.mkdir()
                for name in ("manifest.json", "program.bin", *original_manifest["stage_artifacts"]):
                    (copied / name).write_bytes((self.compiled / name).read_bytes())
                manifest = json.loads((copied / "manifest.json").read_text())
                manifest[key] = value
                (copied / "manifest.json").write_text(json.dumps(manifest, sort_keys=True) + "\n")
                with self.assertRaisesRegex(ValueError, "resident loop"):
                    prepare_c_package(copied / "source-request.json", copied, "changed", target_source=ROOT)


if __name__ == "__main__":
    unittest.main()
