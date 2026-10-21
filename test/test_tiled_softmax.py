"""Compile whole-row softmax with independently ordered numerical checks."""

import argparse
import dataclasses
import hashlib
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from test_looped_mnk import ROOT, build_snapshot
from test_softmax_primitives import _reciprocal, _rounded_sum
from test_vpu_row_sum_reference import _tree_row


def source_request(shape, masked=False):
    from atlas_mlir_native.binding import TARGET_IDENTITY
    from atlas_mlir_native.row_softmax import SOFTMAX_DOMAIN, SOFTMAX_POLICY

    from merlin.semantic_compiler.model import KernelRequest, SemanticNode, TensorType

    request = KernelRequest(
        (
            SemanticNode(
                "scores",
                "input",
                (),
                TensorType(shape, "bf16", "raw-bit-pattern"),
                (("input_domain", SOFTMAX_DOMAIN),),
                "input",
            ),
            SemanticNode(
                "probabilities",
                "softmax",
                ("scores",),
                TensorType(shape, "bf16", SOFTMAX_POLICY),
                (("axis", 1),),
            ),
        ),
        ("probabilities",),
        ("external",),
        (("scores", "external"),),
        TARGET_IDENTITY,
        lowering_policy="diagnostic",
    )

    if masked:
        from atlas_mlir_native.row_softmax import MASKED_SOFTMAX_POLICY

        scores, result = request.nodes
        keep = SemanticNode("keep", "input", (), scores.type, (("input_domain", "exact-zero-or-one"),), "input")
        request = dataclasses.replace(
            request,
            nodes=(
                scores,
                keep,
                dataclasses.replace(
                    result,
                    op="masked_softmax",
                    inputs=(scores.id, keep.id),
                    type=dataclasses.replace(result.type, numerical_policy=MASKED_SOFTMAX_POLICY),
                ),
            ),
            input_storages=((scores.id, "external"), (keep.id, "external")),
        )
    return request


def reference(plan, scores, keep=None):
    """Direct full-row reference, independent of the generated instruction graph.

    The exponential lane port is shared with its separate RTL qualification.
    Max, tree sums, cross-tile adds, reciprocal LUT and products use the
    independent public primitive checks, preserving every rounding boundary.
    """
    from atlas_mlir_native.exp_lane import vexp_bf16_bits
    from atlas_mlir_native.layout import pack_bf16_pair_halves
    from rtl.make_ordered_tile_fixture import _bf16, _round_bf16

    m, n = plan.shape
    tm, tn = plan.tile_type.shape
    result = []
    for region in plan.regions:
        tiles = [bytearray(tm * tn * 2) for _ in range(plan.column_tiles)]
        for r in range(tm):
            row = list(scores[region.row_start + r]) if region.row_start + r < m else [0xC200] * n
            def key(code):
                return (~int(code) & 65535) if int(code) & 32768 else int(code) ^ 32768
            maximum = max(row, key=key)
            exponentials = [vexp_bf16_bits(_rounded_sum(int(code), int(maximum), True)) for code in row]
            if plan.masked:
                if keep is None:
                    raise ValueError("masked reference requires independent keep values")
                active = list(keep[region.row_start + r]) if region.row_start + r < m else [True] * n
                exponentials = [code if flag else 0 for code, flag in zip(exponentials, active, strict=True)]
            partials = [
                _tree_row(exponentials[start : start + tn] + [0] * max(0, tn - len(exponentials[start : start + tn])))
                for start in range(0, n, tn)
            ]
            denominator = partials[0]
            for partial in partials[1:]:
                denominator = _rounded_sum(denominator, partial)
            inverse = _reciprocal(denominator)
            for col, exponential in enumerate(exponentials):
                value = _round_bf16(_bf16(exponential) * _bf16(inverse))
                struct.pack_into("<H", tiles[col // tn], (r * tn + col % tn) * 2, value)
        result.append(tuple(pack_bf16_pair_halves(bytes(tile)) for tile in tiles))
    return tuple(result)


def scores_for(shape, panel):
    rows, columns = np.indices(shape)
    if panel == "uniform":
        values = np.zeros(shape, dtype=np.float32)
    elif panel == "split":
        values = (((rows * 3 + columns) % 13) - 6).astype(np.float32) / 4
        values[:, columns.shape[1] // 2] = 4
    elif panel == "span":
        values = np.full(shape, -32, dtype=np.float32)
        values[:, 0] = 32
        values[:, 1] = np.float32(2**-119)
        values[:, 2] = -0.0
    else:
        raise ValueError("unknown public score panel")
    bits = values.view(np.uint32)
    assert np.all((bits & 65535) == 0)
    return (bits >> 16).astype(np.uint16)


def build_case(snapshot, root, shape, panel, *, masked=False, candidate_nodes=128):
    from atlas_mlir_native.binding import AtlasNativeBinding
    from atlas_mlir_native.row_softmax import prepare_inputs, prepare_row_softmax

    from merlin.semantic_compiler.search import SearchLimits
    from merlin.semantic_compiler.target_binding import verify_native_publication

    source = source_request(shape, masked=masked)
    plan = prepare_row_softmax(source)
    (root / "source-request.json").write_text(json.dumps(source.record(), sort_keys=True, indent=2) + "\n")
    (root / "row-softmax-plan.json").write_text(json.dumps(plan.record(), sort_keys=True, indent=2) + "\n")
    compiled = []
    for index, region in enumerate(plan.regions):
        destination = root / f"compiled-{index}"
        (root / f"request-row-{index}.json").write_text(
            json.dumps(region.request.record(), sort_keys=True, indent=2) + "\n"
        )
        abi = AtlasNativeBinding.plan_boundary_abi(region.request)
        manifest = AtlasNativeBinding.compile(
            snapshot,
            region.request,
            fixed_inputs=abi["fixed_inputs"],
            fixed_outputs=tuple(abi["fixed_outputs"]),
            target_source=ROOT,
            destination=destination,
            limits=SearchLimits(candidate_nodes=candidate_nodes),
        )
        verify_native_publication(
            destination,
            manifest,
            engine="merlin_native",
            request_digest=region.request.digest(),
            target_identity=source.target_identity,
        )
        compiled.append(destination)
    # Runtime scores and reference outputs are constructed after compilation.
    scores = scores_for(shape, panel)
    keep = None
    if masked:
        rows, cols = np.indices(shape)
        keep = (cols % 3 != 1) & ((cols <= rows % shape[1]) | (cols == 0))
        scores[~keep] = 0xC200
    inputs = prepare_inputs(
        plan, scores.astype("<u2").tobytes(), np.where(keep, 0x3F80, 0).astype("<u2").tobytes() if masked else None
    )
    expected = reference(plan, scores, keep)
    fixtures = []
    for index, (destination, group, goldens) in enumerate(zip(compiled, inputs, expected, strict=True)):
        fixture = root / f"fixture-{index}"
        fixture.mkdir()
        execution = json.loads((destination / "execution_plan.json").read_text())
        manifest = json.loads((destination / "manifest.json").read_text())
        request = plan.regions[index].request
        payloads = dict(group)
        for constant in manifest.get("constant_inputs", []):
            payloads[constant["node_id"]] = (destination / constant["path"]).read_bytes()
        sources = [node.id for node in request.nodes if node.effect in {"input", "constant"}]
        binary = (destination / "program.bin").read_bytes()
        (fixture / "program.bin").write_bytes(binary)
        (fixture / "expected.bin").write_bytes(b"".join(goldens))
        properties = [
            "schema=atlas.rtl_tile_fixture.v1",
            f"name=softmax_{shape[0]}_{shape[1]}_{panel}_{index}",
            f"input_count={len(sources)}",
            f"output_count={len(goldens)}",
            f"program_sha256={hashlib.sha256(binary).hexdigest()}",
            f"output.address={execution['output_addresses'][0]:#x}",
        ]
        for ordinal, name in enumerate(sources):
            (fixture / f"input-{ordinal}.bin").write_bytes(payloads[name])
            properties.append(f"input.{ordinal}.address={execution['input_addresses'][ordinal]:#x}")
        (fixture / "manifest.properties").write_text("\n".join(properties) + "\n")
        fixtures.append(fixture)
    return plan, compiled, fixtures


@unittest.skipUnless(
    os.environ.get("MERLIN_NATIVE_ROOT") and os.environ.get("ATLAS_OOT_BIN_DIR"),
    "select Merlin and Atlas tools",
)
class TiledSoftmaxTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR"))
        cls.root = Path(cls.temporary.name)
        cls.snapshot = build_snapshot(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_score_domain_and_initialized_tail_mask(self):
        from atlas_mlir_native.row_softmax import prepare_inputs, prepare_row_softmax

        plan = prepare_row_softmax(source_request((1, 65)))
        self.assertEqual(len(prepare_inputs(plan, bytes(65 * 2))), 1)
        for code in (0x0001, 0x0080, 0x0380, 0x4201, 0x7F80, 0x7FC1):
            with (
                self.subTest(code=code),
                self.assertRaisesRegex(ValueError, "score is outside"),
            ):
                prepare_inputs(plan, struct.pack("<H", code) + bytes(64 * 2))
        (mask,) = plan.regions[0].request.constants
        codes = np.frombuffer(bytes.fromhex(mask.data_hex), dtype="<u2")
        self.assertEqual(np.count_nonzero(codes == 0x3F80), plan.tile_type.shape[0])
        self.assertEqual(np.count_nonzero(codes == 0), codes.size - plan.tile_type.shape[0])
        with self.assertRaisesRegex(ValueError, "tiling differs"):
            prepare_inputs(dataclasses.replace(plan, pad_element=b"\0\0"), bytes(65 * 2))
        from merlin.semantic_compiler.target_binding import NativeCompilationError

        with self.assertRaises(NativeCompilationError):
            prepare_row_softmax(
                dataclasses.replace(
                    plan.source,
                    boundary_layouts=(("scores", "column_major"),),
                    output_layouts=("column_major",),
                    physical_extents=(("scores", 1),),
                )
            )

    def test_installed_cli_binds_the_source_row_group_and_rejects_a_changed_index(self):
        source = self.root / "source.json"
        source.write_text(json.dumps(source_request((33, 65)).record()))
        limits = self.root / "limits.json"
        from merlin.semantic_compiler.search import SearchLimits

        limits.write_text(json.dumps(SearchLimits(candidate_nodes=128).record()))
        common = [
            "--engine",
            "merlin_native",
            "--support",
            "atlas",
            "--snapshot",
            str(self.snapshot.root),
            "--request",
            str(source),
            "--mode",
            "diagnostic",
            "--row-softmax-tile-index",
            "1",
        ]
        destination = self.root / "cli"
        command = [
            sys.executable,
            "-m",
            "merlin.targetgen.cli",
            "native-compile",
            *common,
            "--abi",
            "auto",
            "--target-source",
            str(ROOT),
            "--search-limits",
            str(limits),
            "--out",
            str(destination),
        ]
        result = subprocess.run(command, text=True, capture_output=True, check=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        verification = [
            sys.executable,
            "-m",
            "merlin.targetgen.cli",
            "native-verify",
            *common,
            "--artifact",
            str(destination),
            "--out",
            str(self.root / "verified.json"),
        ]
        result = subprocess.run(verification, text=True, capture_output=True, check=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        verification[verification.index("--row-softmax-tile-index") + 1] = "0"
        result = subprocess.run(verification, text=True, capture_output=True, check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertNotEqual(
            json.loads((self.root / "verified.json").read_text())["status"],
            "verified_artifacts",
        )

    @unittest.skipUnless(os.environ.get("ATLAS_RUN_NPU_MODEL") == "1", "enable the selected model")
    def test_complete_rows_tails_and_physical_memory_on_model(self):
        import torch
        from npu_model.configs import isa_definition as isa
        from npu_model.configs.hardware.default import DefaultHardwareConfig
        from npu_model.logging import LoggerConfig
        from npu_model.simulation import Simulation
        from npu_model.software.program import InstantiableProgram
        from test_spill_model_execution import model_instruction
        from tools.check_program_decode import decode_binary

        torch.set_num_threads(1)
        inventory = json.loads((ROOT / "docs/selected-variant-inventory.json").read_text())
        for shape, panel, masked in (
            ((1, 64), "uniform", False),
            ((31, 65), "split", False),
            ((33, 96), "span", False),
            ((3, 65), "split", True),
            ((33, 35), "span", True),
        ):
            root = self.root / (f"case-{shape[0]}-{shape[1]}")
            root.mkdir()
            _, compiled, fixtures = build_case(self.snapshot, root, shape, panel, masked=masked)
            for destination, fixture in zip(compiled, fixtures, strict=True):
                with self.subTest(shape=shape, panel=panel, fixture=fixture.name):
                    fields = dict(
                        line.split("=", 1) for line in (fixture / "manifest.properties").read_text().splitlines()
                    )
                    program = InstantiableProgram(
                        [
                            model_instruction(row, inventory, isa)
                            for row in decode_binary(inventory, (destination / "program.bin").read_bytes())[
                                "instructions"
                            ]
                        ]
                    )
                    program.dram_base = 0x90000000
                    initial = bytearray([0xA5]) * (256 * 1024)
                    for index in range(int(fields["input_count"])):
                        address = int(fields[f"input.{index}.address"], 0) - program.dram_base
                        payload = (fixture / f"input-{index}.bin").read_bytes()
                        initial[address : address + len(payload)] = payload
                    program.memory_regions = [(0, torch.tensor(list(initial), dtype=torch.uint8))]
                    config = DefaultHardwareConfig()
                    config.arch_state_config = dataclasses.replace(
                        config.arch_state_config,
                        numerics="rtl",
                        randomize_init=True,
                        init_seed=733,
                    )
                    simulation = Simulation(
                        config,
                        LoggerConfig(filename=str(fixture / "model.json")),
                        program,
                        verbose=False,
                    )
                    state = simulation.core.arch_state
                    try:
                        state.vmem.fill_(0xCD)
                        simulation.run(max_cycles=500000)
                        self.assertTrue(state.halted)
                        self.assertEqual(simulation.get_stats().runtime_errors, 0)
                        expected = (fixture / "expected.bin").read_bytes()
                        address = int(fields["output.address"], 0) - program.dram_base
                        initial[address : address + len(expected)] = expected
                        self.assertEqual(
                            bytes(state.read_dram(program.dram_base, len(initial)).tolist()),
                            bytes(initial),
                        )
                    finally:
                        state.close()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--fixtures":
        parser = argparse.ArgumentParser()
        parser.add_argument("--fixtures", type=Path, required=True)
        args = parser.parse_args()
        args.fixtures.mkdir(parents=True, exist_ok=False)
        snapshot = build_snapshot(args.fixtures)
        rows = []
        for shape, panel in (
            ((1, 64), "uniform"),
            ((31, 65), "split"),
            ((33, 96), "span"),
        ):
            root = args.fixtures / f"case-{shape[0]}-{shape[1]}"
            root.mkdir()
            plan, compiled, fixtures = build_case(snapshot, root, shape, panel)
            rows.extend({"directory": str(path)} for path in fixtures)
        (args.fixtures / "cases.json").write_text(json.dumps(rows, indent=2) + "\n")
        print(json.dumps(rows))
    else:
        unittest.main()
