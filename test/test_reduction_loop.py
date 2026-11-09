"""Execute compact seeded reduction loops with independent FP32-tree rounding."""

import dataclasses
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from test_looped_mnk import ROOT, build_snapshot


@unittest.skipUnless(
    os.environ.get("MERLIN_NATIVE_ROOT") and os.environ.get("ATLAS_RUN_NPU_MODEL") == "1",
    "select native compiler tools and npu_model",
)
class ReductionLoopTest(unittest.TestCase):
    def test_seeded_loops_axes_tails_and_tree_order(self):
        import numpy as np
        import torch
        from atlas_mlir_native.binding import DRAM_BASE, AtlasNativeBinding
        from atlas_mlir_native.layout import pack_bf16_pair_halves
        from atlas_mlir_native.reduction import (
            NONNEGATIVE_NORMAL_REDUCTION_POLICY,
            REDUCTION_POLICY,
            input_spec,
            prepare_plan,
            prepare_source,
            request_for,
        )
        from atlas_mlir_native.reduction_c import prepare_c_package
        from atlas_mlir_native.reference import evaluate_seeded_bf16_reduction
        from npu_model.configs import isa_definition as isa
        from npu_model.configs.hardware.default import DefaultHardwareConfig
        from npu_model.logging import LoggerConfig
        from npu_model.simulation import Simulation
        from npu_model.software.program import InstantiableProgram
        from test_pointwise_loop import check_source_c_call
        from test_softmax_primitives import _rounded_sum
        from test_spill_model_execution import model_instruction

        MODEL_REVISION = "47b8599d8ffbe5ddd96059ffe2b5be2427a410f5"
        from test_vpu_row_sum_reference import _tree_row
        from tools.check_program_decode import decode_binary

        from merlin.semantic_compiler.reduction_tiling import (
            assemble_reduction_output,
            materialize_reduction_inputs,
        )
        from merlin.semantic_compiler.reference import TensorValue, evaluate_graph
        from merlin.semantic_compiler.search import SearchLimits
        from merlin.semantic_compiler.target_binding import verify_native_publication

        try:
            origin = json.loads(importlib.metadata.distribution("npu-model").read_text("direct_url.json") or "{}")
        except importlib.metadata.PackageNotFoundError:
            origin = {}
        if not (MODEL_REVISION in origin.get("url", "")
                or origin.get("vcs_info", {}).get("commit_id") == MODEL_REVISION):
            # A pinned, clean source checkout is also an independent original backend.
            model_root = Path(os.environ["ATLAS_NPU_MODEL_ROOT"]).resolve()
            head = subprocess.check_output(["git", "-C", str(model_root), "rev-parse", "HEAD"], text=True).strip()
            self.assertEqual(head, MODEL_REVISION)
            subprocess.run(["git", "-C", str(model_root), "diff", "--exit-code", "HEAD", "--", "."],
                           check=True, capture_output=True)
            import npu_model
            self.assertEqual(Path(npu_model.__file__).resolve().parents[1], model_root)
        torch.set_num_threads(1)
        from npu_model.configs.numerics import NUMERICS

        # Qualify every positive normal BF16 code and +0 on the unchanged
        # original backend. A singleton tree must preserve the code exactly.
        all_codes = torch.tensor([0, *range(0x80, 0x7f80)], dtype=torch.int16)
        singleton_rows = torch.zeros((len(all_codes), 32), dtype=torch.bfloat16)
        singleton_rows.view(torch.int16)[:, 0] = all_codes
        partials = NUMERICS["rtl"].row_sum(singleton_rows)
        combined = NUMERICS["rtl"].add(torch.zeros_like(partials), partials)
        self.assertTrue(torch.equal(combined.view(torch.int16).reshape(-1), all_codes))
        inventory = json.loads((ROOT / "docs/selected-variant-inventory.json").read_text())
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as directory:
            root = Path(directory)
            snapshot = build_snapshot(root)
            for shape, axes in [
                ((3, 7), (1,)),
                ((33, 65), (1,)),
                ((2, 3, 35), (0, 2)),
                ((1, 8, 16, 16), (2, 3)),
                ((8, 33), (1,)),
                ((8, 1), (1,)),
            ]:
                with self.subTest(shape=shape, axes=axes):
                    positive = shape in {(8, 33), (8, 1)}
                    policy = NONNEGATIVE_NORMAL_REDUCTION_POLICY if positive else REDUCTION_POLICY
                    spec = input_spec(reduction_policy=policy)
                    raw_request = request_for(shape, axes, reduction_policy=policy)
                    data_node, seed_node, result_node = raw_request.nodes
                    float_type = lambda node: dataclasses.replace(
                        node.type, dtype="f32", numerical_policy="raw-bit-pattern"
                    )
                    source = dataclasses.replace(
                        raw_request,
                        nodes=(
                            dataclasses.replace(data_node, type=float_type(data_node), attrs=()),
                            dataclasses.replace(seed_node, type=float_type(seed_node), attrs=()),
                            dataclasses.replace(
                                result_node,
                                type=float_type(result_node),
                                op="source_f32_reduce_add_unqualified",
                                attrs=(
                                    ("axes", ",".join(map(str, axes))),
                                    ("reduction_order", "unresolved"),
                                    (
                                        "numeric_semantics",
                                        "source-f32-add-unqualified-v1",
                                    ),
                                    ("scalar_add_order", "input-accumulator"),
                                ),
                            ),
                        ),
                    )
                    request = prepare_source(source, reduction_policy=policy).request
                    plan = prepare_plan(request)
                    abi = AtlasNativeBinding.plan_boundary_abi(request)
                    target = root / ("x".join(map(str, shape)))
                    manifest = AtlasNativeBinding.compile(
                        snapshot,
                        request,
                        fixed_inputs=abi["fixed_inputs"],
                        fixed_outputs=tuple(abi["fixed_outputs"]),
                        target_source=ROOT,
                        destination=target,
                        limits=SearchLimits(),
                    )
                    verify_native_publication(
                        target,
                        manifest,
                        engine="merlin_native",
                        request_digest=request.digest(),
                        target_identity=request.target_identity,
                    )
                    # Values and expectations are constructed only after compilation.
                    values = ((np.arange(np.prod(shape)).reshape(shape) % 29) - 14).astype(np.float32) / 4
                    seed = ((np.arange(np.prod(plan.output_shape)).reshape(plan.output_shape) % 7) - 3).astype(
                        np.float32
                    ) / 2
                    values.reshape(-1)[0] = -0.0
                    if shape == (33, 65):
                        values[0, :5] = [2.0**25, 1, -(2.0**25), 1, 2]
                    if positive:
                        codes = np.zeros(shape, dtype="<u4")
                        codes[:, 0] = [0, 0x80, 0xff, 0x1ff5, 0x3f80, 0x7b00, 0x7f7e, 0x7f7f]
                        if shape[1] > 1:
                            codes[4, 16] = 0x3bc0  # final BF16 RNE tie discriminator
                            codes[4, 32] = 0x3b80  # seeded add chops
                        values = (codes << 16).view("<f4")
                        seed = np.zeros(plan.output_shape, dtype="<f4")
                    data = np.frombuffer(spec.encode(values), dtype="<u2").reshape(shape)
                    initial_codes = np.frombuffer(spec.encode(seed), dtype="<u2").reshape(plan.output_shape)
                    matrix = data.transpose(plan.kept_axes + axes).reshape(plan.matrix_shape)
                    wanted = np.zeros((plan.matrix_shape[0], 32), dtype="<u2")
                    wanted[:, 0] = initial_codes.reshape(-1)
                    for row in range(matrix.shape[0]):
                        for start in range(0, matrix.shape[1], 32):
                            group = [int(x) for x in matrix[row, start : start + 32]]
                            partial = _tree_row(group + [0] * (32 - len(group)))
                            wanted[row] = [_rounded_sum(int(code), partial) for code in wanted[row]]
                    if shape == (33, 65):
                        self.assertEqual(
                            _tree_row([0x4C00, 0x3F80, 0xCC00, 0x3F80, 0x4000] + [0] * 27),
                            0x4000,
                        )
                    interpreted = evaluate_graph(
                        request,
                        {
                            node.id: TensorValue(
                                node.type,
                                tuple(int(code) for code in codes.reshape(-1)),
                            )
                            for node, codes in zip(request.nodes[:2], (data, initial_codes), strict=True)
                        },
                        primitives={
                            (
                                "reduce_sum",
                                request.nodes[-1].type.numerical_policy,
                            ): evaluate_seeded_bf16_reduction
                        },
                    )
                    self.assertEqual(
                        interpreted[0].elements,
                        tuple(int(code) for code in wanted[:, 0]),
                    )
                    expected = bytearray()
                    for start in range(0, wanted.shape[0], 32):
                        tile = np.zeros((32, 32), dtype="<u2")
                        part = wanted[start : start + 32]
                        tile[: len(part)] = part
                        expected.extend(pack_bf16_pair_halves(tile.tobytes()))
                    expected = bytes(expected)
                    inputs = materialize_reduction_inputs(plan, data.tobytes(), initial_codes.tobytes())
                    physical = [
                        b"".join(pack_bf16_pair_halves(raw[at : at + 2048]) for at in range(0, len(raw), 2048))
                        for raw in inputs
                    ]
                    execution = json.loads((target / "execution_plan.json").read_text())
                    binary = (target / "program.bin").read_bytes()
                    program = InstantiableProgram(
                        [
                            model_instruction(row, inventory, isa)
                            for row in decode_binary(inventory, binary)["instructions"]
                        ]
                    )
                    program.dram_base = DRAM_BASE
                    initial = bytearray([0xA5]) * (128 * 2048)
                    for address, payload in zip(execution["input_addresses"], physical, strict=True):
                        at = address - DRAM_BASE
                        initial[at : at + len(payload)] = payload

                    def execute(
                        memory,
                        program=program,
                        target=target,
                        execution=execution,
                        expected=expected,
                        wanted=wanted,
                        plan=plan,
                    ):
                        program.memory_regions = [(0, torch.tensor(list(memory), dtype=torch.uint8))]
                        config = DefaultHardwareConfig()
                        config.arch_state_config = dataclasses.replace(
                            config.arch_state_config,
                            numerics="rtl",
                            randomize_init=True,
                            init_seed=73,
                        )
                        sim = Simulation(
                            config,
                            LoggerConfig(filename=str(target / "model.json")),
                            program,
                            verbose=False,
                        )
                        state = sim.core.arch_state
                        try:
                            state.vmem.fill_(0xCD)
                            sim.run(max_cycles=500000)
                            self.assertTrue(state.halted)
                            self.assertEqual(sim.get_stats().runtime_errors, 0)
                            actual = bytes(state.read_dram(DRAM_BASE, len(memory)).tolist())
                            at = execution["output_addresses"][0] - DRAM_BASE
                            wanted_memory = bytearray(memory)
                            wanted_memory[at : at + len(expected)] = expected
                            self.assertEqual(actual, bytes(wanted_memory))
                            logical = b"".join(wanted[row, 0].tobytes() for row in range(wanted.shape[0]))
                            from atlas_mlir_native.layout import unpack_bf16_pair_halves

                            raw = b"".join(
                                unpack_bf16_pair_halves(actual[at + i : at + i + 2048])
                                for i in range(0, len(expected), 2048)
                            )
                            self.assertEqual(assemble_reduction_output(plan, raw), logical)
                            return actual
                        finally:
                            state.close()

                    execute(initial)
                    _, files = prepare_c_package(
                        source,
                        target,
                        "pointwise",
                        target_source=ROOT,
                        runtime_linkage="embedded",
                        reduction_policy=policy,
                    )
                    package = target / "c-package"
                    package.mkdir()
                    for name, payload in files.items():
                        (package / name).write_bytes(payload)
                    wanted_float = (wanted[:, 0].astype(np.uint32) << 16).astype("<u4").tobytes()
                    invalid = []
                    if positive:
                        # Reject before any program load or device-memory write.
                        for code in (0x8000, 1, 0x7f80, 0x7fc0):
                            bad = values.copy()
                            bad.view("<u4").reshape(-1)[0] = code << 16
                            invalid.append((bad, seed))
                        bad_seed = seed.copy()
                        bad_seed.view("<u4").reshape(-1)[-1] = 0x7f7f0000
                        invalid.append((values, bad_seed))  # seeded FP32 overflow
                        if shape[1] > 1:
                            for code in (0x7f7f, 0x7b00):
                                bad = values.copy()
                                bad.view("<u4")[-1, 1] = code << 16
                                invalid.append((bad, seed))  # tree or BF16-rounding overflow
                    check_source_c_call(
                        self,
                        package,
                        initial,
                        binary,
                        execute,
                        [values, seed],
                        wanted_float,
                        invalid_values=invalid,
                    )
                    if os.environ.get("ATLAS_LLVM_BIN"):
                        subprocess.run(
                            [
                                str(Path(os.environ["ATLAS_LLVM_BIN"]) / "clang"),
                                "--target=riscv64-unknown-elf",
                                "-march=rv64gcv",
                                "-mabi=lp64d",
                                "-std=c11",
                                "-O2",
                                "-ffreestanding",
                                "-Wall",
                                "-Wextra",
                                "-Werror",
                                "-c",
                                "pointwise.c",
                                "-o",
                                "pointwise.rv64.o",
                            ],
                            cwd=package,
                            check=True,
                            capture_output=True,
                        )
                    if os.environ.get("ATLAS_REDUCTION_FIXTURES"):
                        fixture = Path(os.environ["ATLAS_REDUCTION_FIXTURES"]) / target.name
                        fixture.mkdir(parents=True, exist_ok=False)
                        (fixture / "program.bin").write_bytes(binary)
                        for member in (
                            "pointwise.c",
                            "pointwise.h",
                            "pointwise.rv64.o",
                        ):
                            if (package / member).exists():
                                shutil.copyfile(package / member, fixture / member)
                        (fixture / "expected.bin").write_bytes(expected)
                        rows = []
                        for address, payload in zip(execution["input_addresses"], physical, strict=True):
                            rows.extend((address + i, payload[i : i + 2048]) for i in range(0, len(payload), 2048))
                        props = [
                            "schema=atlas.rtl_tile_fixture.v2",
                            f"name=reduction_{target.name}",
                            f"program_sha256={hashlib.sha256(binary).hexdigest()}",
                            f"input_count={len(rows)}",
                            f"output_count={len(expected) // 2048}",
                            f"output.address={execution['output_addresses'][0]:#x}",
                        ]
                        for i, (address, payload) in enumerate(rows):
                            (fixture / f"input-{i}.bin").write_bytes(payload)
                            props.extend(
                                [
                                    f"input.{i}.address={address:#x}",
                                    f"input.{i}.read_count=1",
                                ]
                            )
                        (fixture / "manifest.properties").write_text("\n".join(props) + "\n")


if __name__ == "__main__":
    unittest.main()
