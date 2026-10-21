"""Compile guarded masked softmax and execute empty/nonempty rows on the model."""

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


def policy_for(source):
    from atlas_mlir_native.pointwise_math import MATH_POLICY
    from atlas_mlir_native.reduction import REDUCTION_POLICY
    from atlas_mlir_native.row_softmax import GUARDED_SOFTMAX_POLICY, MASKED_SOFTMAX_POLICY
    from atlas_mlir_native.sigmoid import SIGMOID_POLICY
    from test_application_c_package import ApplicationLayoutPolicyTest

    return ApplicationLayoutPolicyTest().policy() | {
        "schema": "atlas.diagnostic_application_precision.v15",
        "source_sha256": source.source_identity,
        "reduction": REDUCTION_POLICY,
        "pointwise_math": MATH_POLICY,
        "sigmoid": SIGMOID_POLICY,
        "masked_softmax": MASKED_SOFTMAX_POLICY,
        "guarded_masked_softmax": GUARDED_SOFTMAX_POLICY,
        "host_predicates": ["mask"],
        "host_integer_ops": ["i64_add"],
    }


@unittest.skipUnless(
    os.environ.get("MERLIN_NATIVE_ROOT") and os.environ.get("ATLAS_OOT_BIN_DIR"), "select Merlin and Atlas tools"
)
class GuardedSoftmaxTest(unittest.TestCase):
    shape = (2, 33)
    candidate_nodes = 256
    model_cycle_limit = 500000

    @classmethod
    def setUpClass(cls):
        from atlas_mlir_native.application import AtlasApplicationAdapter
        from atlas_mlir_native.binding import TARGET_IDENTITY

        from merlin.compile.semantic_application import compile_semantic_application, verify_semantic_application
        from merlin.semantic_compiler.search import SearchLimits
        from merlin.tests.ir.test_guarded_softmax_source import guarded_graph

        cls.temporary = tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR"))
        cls.root = Path(cls.temporary.name)
        cls.source = dataclasses.replace(
            guarded_graph(cls.shape), target_identity=TARGET_IDENTITY, lowering_policy="diagnostic"
        )
        cls.policy = policy_for(cls.source)
        cls.adapter = AtlasApplicationAdapter(cls.policy, original_source_sha256=cls.source.source_identity)
        cls.snapshot = build_snapshot(cls.root)
        cls.artifact = cls.root / "compiled"
        cls.manifest = compile_semantic_application(
            cls.source,
            adapter=cls.adapter,
            snapshot=cls.snapshot,
            target_source=ROOT,
            destination=cls.artifact,
            limits=SearchLimits(candidate_nodes=cls.candidate_nodes),
            name="guarded",
            capacity=1 << 20,
        )
        cls.check = verify_semantic_application(
            cls.source, adapter=cls.adapter, snapshot=cls.snapshot, target_source=ROOT, artifact=cls.artifact
        )

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_explicit_policy_one_program_and_mixed_source_C_interface(self):
        from atlas_mlir_native.application import AtlasApplicationAdapter

        from merlin.semantic_compiler.target_binding import NativeCompilationError

        self.assertEqual(self.check["status"], "checked")
        self.assertEqual(len(self.manifest["segments"]), 1)
        source = self.artifact / "c-package/combined/guarded_combined.c"
        commands = [
            [
                shutil.which("cc"),
                "-std=c11",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-c",
                str(source),
                "-o",
                str(self.root / "native.o"),
            ]
        ]
        if cross := os.environ.get("MERLIN_RISCV_GCC"):
            commands.append(
                [
                    cross,
                    "-std=c11",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    "-ffreestanding",
                    "-march=rv64gcv",
                    "-mabi=lp64d",
                    "-mstrict-align",
                    "-c",
                    str(source),
                    "-o",
                    str(self.root / "rv64.o"),
                ]
            )
        for command in commands:
            subprocess.run(command, check=True, capture_output=True)
        old = self.policy | {"schema": "atlas.diagnostic_application_precision.v14"}
        del old["guarded_masked_softmax"]
        with self.assertRaisesRegex(NativeCompilationError, "no compiled application route"):
            AtlasApplicationAdapter(old, original_source_sha256=self.source.source_identity).plan(self.source)
        with self.assertRaisesRegex(NativeCompilationError, "guarded masked softmax policy"):
            AtlasApplicationAdapter(
                self.policy | {"guarded_masked_softmax": "unqualified"},
                original_source_sha256=self.source.source_identity,
            )

    @unittest.skipUnless(os.environ.get("ATLAS_RUN_NPU_MODEL") == "1", "enable the pinned model")
    def test_empty_and_nonempty_rows_preserve_memory_on_model(self):
        import numpy as np
        import torch
        from atlas_mlir_native.row_softmax import prepare_inputs, prepare_row_softmax
        from atlas_mlir_native.softmax_precision import lower_semantic_softmax
        from npu_model.configs import isa_definition as isa
        from npu_model.configs.hardware.default import DefaultHardwareConfig
        from npu_model.logging import LoggerConfig
        from npu_model.simulation import Simulation
        from npu_model.software.program import InstantiableProgram
        from test_tiled_softmax import reference, scores_for
        from tools.check_program_decode import decode_binary
        from tools.model_instructions import model_instruction

        origin = json.loads(importlib.metadata.distribution("npu-model").read_text("direct_url.json"))
        self.assertIn("337e9f249f938a1dff007ddf8a763e5cd4b3aa2b", origin.get("url", ""))
        plan = prepare_row_softmax(lower_semantic_softmax(self.source, allow_guarded=True).request)
        program_root = self.artifact / "segments/region_000"
        body_root = program_root / "body"
        binary = (program_root / "program.bin").read_bytes()
        execution = json.loads((body_root / "execution_plan.json").read_text())
        inventory = json.loads((ROOT / "docs/selected-variant-inventory.json").read_text())
        instructions = [
            model_instruction(row, inventory, isa) for row in decode_binary(inventory, binary)["instructions"]
        ]
        torch.set_num_threads(1)
        self.model_results = []
        for panel in range(2):
            scores = scores_for(self.shape, "split" if panel == 0 else "span")
            keep = np.ones(self.shape, dtype=bool)
            keep[panel] = False
            keep[1 - panel, 1::3] = False
            scores[~keep] = 0xC200
            (group,) = prepare_inputs(
                plan, scores.astype("<u2").tobytes(), np.where(keep, 0x3F80, 0).astype("<u2").tobytes()
            )
            # The independent normal-row reference is evaluated on a legal
            # nonempty row. The source guard independently defines the other
            # row as positive zero, rather than evaluating a zero reciprocal.
            safe_keep = keep.copy()
            safe_keep[panel, 0] = True
            expected = [bytearray(raw) for raw in reference(plan, scores, safe_keep)[0]]
            for raw in expected:
                raw[panel * 32 : panel * 32 + 32] = bytes(32)
                raw[1024 + panel * 32 : 1024 + panel * 32 + 32] = bytes(32)
            payloads = dict(group)
            for item in execution["constant_inputs"]:
                payloads[item["node_id"]] = (body_root / item["path"]).read_bytes()
            leaves = [n.id for n in plan.regions[0].request.nodes if n.effect in {"input", "constant"}]
            base = min(execution["input_addresses"]) - 64
            end = max(execution["output_addresses"]) + 2048 + 64
            staging = execution.get("staging_workspace")
            if staging:
                end = max(end, staging["address"] + staging["bytes"] + 64)
            memory = bytearray((i * 19 + 31) % 256 for i in range(end - base))
            for address, key in zip(execution["input_addresses"], leaves, strict=True):
                memory[address - base : address - base + 2048] = payloads[key]
            program = InstantiableProgram(instructions)
            program.dram_base = base
            program.memory_regions = [(0, torch.tensor(list(memory), dtype=torch.uint8))]
            config = DefaultHardwareConfig()
            config.arch_state_config = dataclasses.replace(
                config.arch_state_config, numerics="rtl", randomize_init=True, init_seed=149 + panel
            )
            simulation = Simulation(
                config, LoggerConfig(filename=str(self.root / f"model-{panel}.log")), program, verbose=False
            )
            state = simulation.core.arch_state
            try:
                state.vmem.fill_(0xCD)
                simulation.run(max_cycles=self.model_cycle_limit)
                self.assertTrue(state.halted, f"watchdog at {simulation.get_stats().cycles} cycles")
                self.assertEqual(state.halt_reason, "ecall")
                self.assertEqual(simulation.get_stats().runtime_errors, 0)
                observed = bytes(state.read_dram(base, len(memory)).tolist())
                scratch_bytes = 0
                if staging:
                    start = staging["address"] - base
                    memory[start : start + staging["bytes"]] = observed[start : start + staging["bytes"]]
                    scratch_bytes = staging["bytes"] - 2048 * len(expected)
                for address, raw in zip(execution["output_addresses"], expected, strict=True):
                    memory[address - base : address - base + len(raw)] = raw
                self.assertEqual(observed, bytes(memory))
                self.model_results.append(
                    {
                        "panel": panel,
                        "cycles": simulation.get_stats().cycles,
                        "checked_memory_bytes": len(memory) - scratch_bytes,
                        "mutable_scratch_bytes": scratch_bytes,
                        "program_bytes": len(binary),
                    }
                )
            finally:
                state.close()


class WideGuardedSoftmaxTest(GuardedSoftmaxTest):
    """Stage one 1024-column row group and retain one device launch."""

    shape = (32, 1024)
    candidate_nodes = 1024
    # The independently executed uniform-row screen takes 540,052 cycles.
    model_cycle_limit = 1000000
