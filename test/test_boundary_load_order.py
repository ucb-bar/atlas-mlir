"""Native ordering fallback for shared, long-lived BF16 boundary values."""

import dataclasses
import hashlib
import importlib.metadata
import json
import os
import tempfile
import unittest
from pathlib import Path

from test_looped_mnk import ROOT, build_snapshot


def shared_normalization():
    from atlas_mlir_native.binding import (
        BF16_BINARY_ATTRS,
        BF16_ROW_REDUCTION_ATTRS,
        BF16_UNARY_ATTRS,
        TARGET_IDENTITY,
        _row_reduce_maps,
    )

    from merlin.semantic_compiler.model import (
        KernelRequest,
        SemanticNode,
        TensorType,
        pointwise_identity_maps,
    )

    tensor = TensorType((32, 32), "bf16", "raw-bit-pattern")
    attrs = {**BF16_BINARY_ATTRS, **BF16_ROW_REDUCTION_ATTRS, **BF16_UNARY_ATTRS}

    def source(name, user):
        return SemanticNode(name, "input", (), tensor, (attrs[user][0],), "input")

    def operation(name, op, inputs):
        maps = _row_reduce_maps() if op == "row_sum_bf16" else pointwise_identity_maps(2, len(inputs))
        return SemanticNode(name, op, inputs, tensor, attrs[op], index_maps=maps)

    nodes = (
        *(source(name, "mul_bf16_zero") for name in ("a", "b", "c", "d", "tile", "mask")),
        operation("masked", "mul_bf16_zero", ("tile", "mask")),
        operation("sum", "row_sum_bf16", ("masked",)),
        source("subtotal", "add_bf16_rne"),
        operation("denominator", "add_bf16_rne", ("subtotal", "sum")),
        operation("inverse", "recip_bf16_wide", ("denominator",)),
        *(operation(f"p{i}", "mul_bf16_zero", (name, "inverse")) for i, name in enumerate("abcd")),
    )
    return KernelRequest(
        nodes, ("p0", "p1", "p2", "p3", "masked", "inverse"), ("external",) * 6,
        tuple((node.id, "external") for node in nodes if node.effect == "input"),
        TARGET_IDENTITY, lowering_policy="diagnostic",
    )


@unittest.skipUnless(os.environ.get("MERLIN_NATIVE_ROOT") and os.environ.get("ATLAS_OOT_BIN_DIR"),
                     "select Merlin and Atlas tools")
class BoundaryLoadOrderTest(unittest.TestCase):
    def test_alternative_order_preserves_shared_values_and_all_outputs(self):
        from atlas_mlir_native.binding import AtlasNativeBinding

        from merlin.semantic_compiler.search import SearchLimits
        from merlin.semantic_compiler.target_binding import NativeCompilationError, verify_native_publication

        with tempfile.TemporaryDirectory(prefix="boundary-load-order-") as temporary:
            work = Path(temporary)
            snapshot = build_snapshot(work)
            request = shared_normalization()
            inputs = dict(zip((n.id for n in request.nodes if n.effect == "input"), (11, 12, 16, 17, 18, 6, 19)))
            outputs = (7, 8, 9, 10, 14, 15)

            def compile_to(name, limits):
                return AtlasNativeBinding.compile(
                    snapshot, request, fixed_inputs=inputs, fixed_outputs=outputs,
                    target_source=ROOT, destination=work / name, limits=limits, allow_staging=False,
                )

            with self.assertRaises(NativeCompilationError):
                compile_to("one-order", SearchLimits(orders_per_candidate=1, candidates=1))
            manifest = compile_to("compiled", SearchLimits())
            self.assertEqual(manifest["ordering_attempts"], 2)
            self.assertEqual(manifest["rejected_materialization"], 1)
            verify_native_publication(work / "compiled", manifest, engine="merlin_native",
                                      request_digest=request.digest(), target_identity=request.target_identity)
            if os.environ.get("ATLAS_RUN_NPU_MODEL") == "1":
                self.execute_model(work, inputs, outputs)

    def execute_model(self, work, inputs, outputs):
        # Runtime panels are created after compilation. Powers of two keep the
        # reference independent of the target's approximate reciprocal table.
        import numpy as np
        import torch
        from atlas_mlir_native.binding import DRAM_BASE
        from npu_model.configs import isa_definition as isa
        from npu_model.configs.hardware.default import DefaultHardwareConfig
        from npu_model.logging import LoggerConfig
        from npu_model.simulation import Simulation
        from npu_model.software.program import InstantiableProgram
        from test_spill_model_execution import MODEL_REVISION, model_instruction
        from tools.check_program_decode import decode_binary

        origin = json.loads(importlib.metadata.distribution("npu-model").read_text("direct_url.json") or "{}")
        self.assertTrue(MODEL_REVISION in origin.get("url", "") or
                        origin.get("vcs_info", {}).get("commit_id") == MODEL_REVISION)
        torch.set_num_threads(1)
        inventory = json.loads((ROOT / "docs/selected-variant-inventory.json").read_text())
        binary = (work / "compiled/program.bin").read_bytes()
        program = InstantiableProgram([model_instruction(row, inventory, isa)
                                       for row in decode_binary(inventory, binary)["instructions"]])
        program.dram_base = DRAM_BASE

        def packed(value):
            bits = np.asarray(value, dtype="<f4").view("<u4")
            self.assertTrue(np.all((bits & 0xFFFF) == 0))
            codes = (bits >> 16).astype("<u2")
            return np.stack((codes[:, :16], codes[:, 16:])).tobytes()

        reports = []
        for panel in (0, 1, 0):
            arrays = {name: np.full((32, 32), value) for name, value in
                      zip(("a", "b", "c", "d"), (.125, .25, .0625, .5))}
            arrays["tile"] = np.full((32, 32), .125 if panel == 0 else .25)
            arrays["mask"] = np.broadcast_to(np.arange(32) < (3 if panel == 0 else 2), (32, 32))
            arrays["subtotal"] = np.full((32, 32), 3.625 if panel == 0 else 1.5)
            inverse = np.full((32, 32), .25 if panel == 0 else .5)
            wanted = [packed(arrays[name] * inverse) for name in "abcd"]
            wanted += [packed(arrays["tile"] * arrays["mask"]), packed(inverse)]
            initial = bytearray([0xA5]) * 65536
            for name, slot in inputs.items():
                initial[slot * 2048:(slot + 1) * 2048] = packed(arrays[name])
            expected = bytearray(initial)
            for slot, value in zip(outputs, wanted, strict=True):
                expected[slot * 2048:(slot + 1) * 2048] = value
            program.memory_regions = [(0, torch.tensor(list(initial), dtype=torch.uint8))]
            config = DefaultHardwareConfig()
            config.arch_state_config = dataclasses.replace(config.arch_state_config, numerics="rtl",
                                                          randomize_init=True, init_seed=87)
            simulation = Simulation(config, LoggerConfig(filename=str(work / "model.json")), program, verbose=False)
            state = simulation.core.arch_state
            try:
                state.vmem.fill_(0xCD)
                simulation.run(max_cycles=500000)
                self.assertTrue(state.halted)
                self.assertEqual(simulation.get_stats().runtime_errors, 0)
                observed = bytes(state.read_dram(DRAM_BASE, len(initial)).tolist())
                self.assertEqual(observed, bytes(expected))
                reports.append({"panel": panel, "cycles": simulation.get_stats().cycles,
                                "checked_bytes": len(initial), "sha256": hashlib.sha256(observed).hexdigest()})
            finally:
                state.close()
        if os.environ.get("ATLAS_ORDER_FIXTURES"):
            import shutil
            destination = Path(os.environ["ATLAS_ORDER_FIXTURES"])
            shutil.copytree(work / "compiled", destination)
            (destination / "model-runs.json").write_text(json.dumps(reports, indent=2) + "\n")
