"""Parsed source graph to Atlas diagnostic precision lowering."""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

ROOT = Path(__file__).resolve().parents[1]
MERLIN = os.environ.get("MERLIN_NATIVE_ROOT")


@unittest.skipUnless(MERLIN, "set MERLIN_NATIVE_ROOT to test the OOT precision provider")
class GraphPrecisionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        sys.path.insert(0, str(Path(MERLIN) / "src"))
        sys.path.insert(0, str(ROOT / "python"))
        from atlas_mlir_native.graph_precision import lower_diagnostic_graph

        cls.lower = staticmethod(lower_diagnostic_graph)
        cls.source = (ROOT / "test/examples/captured_attention_mlp32_two_outputs_linalg.mlir").read_text()
        cls.policy = json.loads((ROOT / "test/examples/attention_mlp_graph_precision.json").read_text())

    @classmethod
    def tearDownClass(cls) -> None:
        sys.path.remove(str(ROOT / "python"))
        sys.path.remove(str(Path(MERLIN) / "src"))

    def assert_shared_source_constants(self, parsed):
        # The fixture initializes ten tensors, but equal typed splats share
        # their semantic value. Computations and every source binding remain.
        self.assertEqual(sum(op in {"linalg.fill", "tensor.splat"}
                             for op in parsed.source_operations), 10)
        self.assertEqual(sum(node.effect == "input" for node in parsed.request.nodes), 6)
        self.assertEqual(sum(node.op == "source_f32_matmul_unqualified"
                             for node in parsed.request.nodes), 7)
        constants = [parsed.request.node(item.node_id) for item in parsed.request.constants]
        self.assertEqual(len(constants), 4)
        identities = {(node.type, item.encoding, item.data_hex)
                      for node, item in zip(constants, parsed.request.constants, strict=True)}
        self.assertEqual(len(identities), len(constants))

    def test_all_source_nodes_and_two_outputs_are_accounted(self) -> None:
        from atlas_mlir_native.binding import _mxu_bf16_dag_request
        from merlin.semantic_compiler.linalg_bridge import translate_linalg_text

        result = self.lower(self.source, entry="forward", policy=self.policy)
        parsed = translate_linalg_text(
            self.source, entry="forward", target_identity=result.request.target_identity,
            lowering_policy="diagnostic",
        )
        self.assert_shared_source_constants(parsed)
        self.assertEqual(len(result.request.nodes), 29)
        self.assertEqual({name for name, _ in result.source_to_target},
                         {node.id for node in parsed.request.nodes})
        self.assertEqual(result.request.outputs, parsed.request.outputs)
        self.assertTrue(_mxu_bf16_dag_request(result.request))
        self.assertEqual(result.request.constants[0].data_hex, "353e" * 1024)
        self.assertEqual(len(result.boundary_conversions), 8)
        self.assertIn("original_model_quality_unmeasured", result.record()["qualification"])

    def test_residual_add_preserves_three_ordered_outputs(self) -> None:
        from atlas_mlir_native.binding import BF16_ADD_ATTRS, _mxu_bf16_dag_request
        from merlin.semantic_compiler.linalg_bridge import translate_linalg_text

        source = (ROOT / "test/examples/attention_mlp_residual_add_linalg.mlir").read_text()
        result = self.lower(source, entry="forward", policy=self.policy)
        parsed = translate_linalg_text(
            source, entry="forward", target_identity=result.request.target_identity,
            lowering_policy="diagnostic",
        )
        self.assert_shared_source_constants(parsed)
        self.assertEqual(len(result.request.nodes), 30)
        self.assertEqual(result.request.outputs, parsed.request.outputs)
        self.assertEqual(len(result.request.outputs), 3)
        self.assertEqual(result.request.nodes[-1].op, "add_bf16")
        self.assertEqual(result.request.nodes[-1].attrs, BF16_ADD_ATTRS)
        self.assertTrue(_mxu_bf16_dag_request(result.request))
        self.assertEqual(
            {name for name, _ in result.source_to_target},
            {node.id for node in parsed.request.nodes},
        )

    def test_node_ids_and_provenance_do_not_choose_the_rewrite(self) -> None:
        extra = "    %91 = arith.constant 0.000000e+00 : f32\n"
        changed = self.source.replace("    %6 = tensor.empty()", extra + "    %6 = tensor.empty()", 1)
        self.assertNotEqual(changed, self.source)
        original = self.lower(self.source, entry="forward", policy=self.policy)
        shifted = self.lower(changed, entry="forward", policy=self.policy)
        self.assertEqual(len(shifted.request.nodes), len(original.request.nodes))
        self.assertNotEqual(shifted.request.outputs, original.request.outputs)
        self.assertEqual(tuple(node.op for node in shifted.request.nodes),
                         tuple(node.op for node in original.request.nodes))

    def test_wrong_transpose_and_nonzero_initializer_fail(self) -> None:
        from merlin.semantic_compiler.linalg_bridge import LinalgBridgeError

        transpose = self.source.replace("permutation = [1, 0]", "permutation = [0, 1]", 1)
        initializer = self.source.replace(
            'prov.module = "query"} 0.000000e+00',
            'prov.module = "query"} 1.000000e+00', 1,
        )
        for changed in (transpose, initializer):
            with self.subTest(changed=changed != self.source), self.assertRaises(LinalgBridgeError):
                self.lower(changed, entry="forward", policy=self.policy)

    def test_output_order_and_policy_identity(self) -> None:
        from merlin.semantic_compiler.linalg_bridge import LinalgBridgeError

        changed = self.source.replace("func.return %80, %90", "func.return %90, %80", 1)
        original = self.lower(self.source, entry="forward", policy=self.policy)
        swapped = self.lower(changed, entry="forward", policy=self.policy)
        self.assertEqual(swapped.request.outputs, tuple(reversed(original.request.outputs)))
        self.assertNotEqual(swapped.request.digest(), original.request.digest())
        scaled = {**self.policy, "pack_scale_code": 126}
        changed_scale = self.lower(self.source, entry="forward", policy=scaled)
        self.assertNotEqual(changed_scale.request.digest(), original.request.digest())
        self.assertNotEqual(changed_scale.policy_sha256, original.policy_sha256)
        self.assertEqual(
            {dict(node.attrs)["scale_code"] for node in changed_scale.request.nodes if node.op == "pack_fp8"},
            {126},
        )
        for invalid in (121, 129, True):
            with self.subTest(invalid=invalid), self.assertRaises(LinalgBridgeError):
                self.lower(self.source, entry="forward", policy={**self.policy, "pack_scale_code": invalid})
        bad = copy.deepcopy(self.policy)
        bad["interlayer_pack"] = []
        with self.assertRaises(LinalgBridgeError):
            self.lower(self.source, entry="forward", policy=bad)

    def test_rounded_pack_policy_has_distinct_graph_identity(self) -> None:
        from atlas_mlir_native.binding import (
            ROUNDED_PACK_DOMAIN,
            ROUNDED_PACK_POLICY,
            _mxu_bf16_dag_request,
        )

        rounded_policy = json.loads((
            ROOT / "test/examples/attention_mlp_graph_rounded_pack.json"
        ).read_text())
        exact = self.lower(self.source, entry="forward", policy=self.policy)
        rounded = self.lower(self.source, entry="forward", policy=rounded_policy)
        self.assertNotEqual(rounded.request.digest(), exact.request.digest())
        self.assertNotEqual(rounded.policy_sha256, exact.policy_sha256)
        self.assertTrue(_mxu_bf16_dag_request(rounded.request))
        packs = [node for node in rounded.request.nodes if node.op == "pack_fp8"]
        self.assertEqual(len(packs), 6)
        self.assertTrue(all(node.type.numerical_policy == ROUNDED_PACK_POLICY
                            and dict(node.attrs)["input_domain"] == ROUNDED_PACK_DOMAIN
                            for node in packs))

    def test_native_compile_uses_lowered_graph(self) -> None:
        if not os.environ.get("ATLAS_OOT_BIN_DIR"):
            self.skipTest("set ATLAS_OOT_BIN_DIR for physical emission")
        from atlas_mlir_native.binding import AtlasNativeBinding
        from atlas_mlir_native.graph_abi import prepare_graph_inputs
        from merlin.semantic_compiler.snapshot import (
            build_native_snapshot,
            packaged_egg_bridge_path,
        )

        result = self.lower(self.source, entry="forward", policy=self.policy)
        cli = Path(sys.executable).with_name("merlin-targetgen")
        if not cli.is_file():
            self.skipTest("install merlin-targetgen for the CLI integration")
        environment = os.environ.copy()
        environment["PYTHONPATH"] = os.pathsep.join((str(ROOT / "python"), str(Path(MERLIN) / "src")))
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as directory:
            root = Path(directory)
            snapshot = build_native_snapshot(
                AtlasNativeBinding.profile(), destination=root / "snapshot",
                crate=packaged_egg_bridge_path(), cargo_target_dir=root / "cargo",
                source_revision="parsed-graph-precision-test",
            )
            common = [
                "--engine", "merlin_native", "--support", "atlas",
                "--snapshot", str(snapshot.root),
                "--linalg", str(ROOT / "test/examples/captured_attention_mlp32_two_outputs_linalg.mlir"),
                "--linalg-entry", "forward",
                "--diagnostic-graph-policy", str(ROOT / "test/examples/attention_mlp_graph_precision.json"),
                "--mode", "diagnostic",
            ]
            compiled = root / "compiled"
            compile_run = subprocess.run(
                [str(cli), "native-compile", *common,
                 "--abi", str(ROOT / "test/examples/attention_mlp_graph_abi.json"),
                 "--search-limits", str(ROOT / "test/examples/attention_mlp_graph_search_limits.json"),
                 "--target-source", str(ROOT), "--out", str(compiled)],
                env=environment, capture_output=True, text=True, timeout=120, check=False,
            )
            self.assertEqual(compile_run.returncode, 0, compile_run.stderr + compile_run.stdout)
            self.assertEqual(json.loads(compile_run.stdout)["request_digest"], result.request.digest())
            verify_run = subprocess.run(
                [str(cli), "native-verify", *common, "--artifact", str(compiled),
                 "--out", str(root / "verify.json")],
                env=environment, capture_output=True, text=True, timeout=30, check=False,
            )
            self.assertEqual(verify_run.returncode, 0, verify_run.stderr + verify_run.stdout)
            self.assertEqual(json.loads((root / "verify.json").read_text())["status"], "verified_artifacts")
            self.assertGreater(len((compiled / "program.bin").read_bytes()), 0)
            staged = prepare_graph_inputs(
                result, self.policy,
                json.loads((compiled / "precision-plan.json").read_text()),
                json.loads((compiled / "manifest.json").read_text()),
                json.loads((compiled / "execution_plan.json").read_text()),
                json.loads((ROOT / "test/examples/captured_attention_mlp32_arguments.json").read_text()),
                weights=load_file(ROOT / "test/examples/captured_attention_mlp32_weights.safetensors"),
                invocation={"tokens": np.full((32, 32), 0.5, dtype=np.float32)},
            )
            self.assertEqual(len(staged), 6)
            self.assertEqual(staged[-1].payload, bytes([0x30]) * 1024)

            scaled_policy = {**self.policy, "pack_scale_code": 126}
            scaled_policy_path = ROOT / "test/examples/attention_mlp_graph_scale126_precision.json"
            self.assertEqual(json.loads(scaled_policy_path.read_text()), scaled_policy)
            scaled_common = common.copy()
            scaled_common[scaled_common.index("--diagnostic-graph-policy") + 1] = str(scaled_policy_path)
            scaled_result = self.lower(self.source, entry="forward", policy=scaled_policy)
            scaled_compiled = root / "compiled-scale126"
            scaled_run = subprocess.run(
                [str(cli), "native-compile", *scaled_common,
                 "--abi", str(ROOT / "test/examples/attention_mlp_graph_abi.json"),
                 "--search-limits", str(ROOT / "test/examples/attention_mlp_graph_search_limits.json"),
                 "--target-source", str(ROOT), "--out", str(scaled_compiled)],
                env=environment, capture_output=True, text=True, timeout=120, check=False,
            )
            self.assertEqual(scaled_run.returncode, 0, scaled_run.stderr + scaled_run.stdout)
            self.assertEqual(json.loads(scaled_run.stdout)["request_digest"], scaled_result.request.digest())
            self.assertNotEqual(scaled_result.request.digest(), result.request.digest())
            self.assertNotEqual(
                (scaled_compiled / "program.bin").read_bytes(),
                (compiled / "program.bin").read_bytes(),
            )
            self.assertIn("scale_code = 126 : i32", (scaled_compiled / "atlas-virtual.mlir").read_text())
            scaled_verify = subprocess.run(
                [str(cli), "native-verify", *scaled_common, "--artifact", str(scaled_compiled),
                 "--out", str(root / "scaled-verify.json")],
                env=environment, capture_output=True, text=True, timeout=30, check=False,
            )
            self.assertEqual(scaled_verify.returncode, 0, scaled_verify.stderr + scaled_verify.stdout)
            self.assertEqual(
                json.loads((root / "scaled-verify.json").read_text())["status"],
                "verified_artifacts",
            )
            fixture_env = environment.copy()
            fixture_env["PYTHONPATH"] += os.pathsep + str(ROOT / "test")
            fixture = subprocess.run(
                [sys.executable, str(ROOT / "test/rtl/make_graph_target_fixture.py"),
                 "--compiled", str(scaled_compiled),
                 "--suite", str(ROOT / "test/examples/attention_mlp_graph_scale126_rtl_suite.json"),
                 "--policy", str(scaled_policy_path),
                 "--case", "exact_scale126_half", "--out", str(root / "scale126-half")],
                env=fixture_env, capture_output=True, text=True, timeout=30, check=False,
            )
            self.assertEqual(fixture.returncode, 0, fixture.stderr + fixture.stdout)
            self.assertEqual(json.loads(fixture.stdout)["expected_bf16_cells"], 2048)
            wrong_policy = subprocess.run(
                [sys.executable, str(ROOT / "test/rtl/make_graph_target_fixture.py"),
                 "--compiled", str(scaled_compiled),
                 "--suite", str(ROOT / "test/examples/attention_mlp_graph_scale126_rtl_suite.json"),
                 "--policy", str(ROOT / "test/examples/attention_mlp_graph_rounded_pack.json"),
                 "--case", "exact_scale126_half", "--out", str(root / "wrong-policy")],
                env=fixture_env, capture_output=True, text=True, timeout=30, check=False,
            )
            self.assertNotEqual(wrong_policy.returncode, 0)
            self.assertIn("frozen policy differs", wrong_policy.stderr)

            residual_source = ROOT / "test/examples/attention_mlp_residual_add_linalg.mlir"
            residual_common = common.copy()
            residual_common[residual_common.index("--linalg") + 1] = str(residual_source)
            residual = root / "compiled-residual"
            residual_run = subprocess.run(
                [str(cli), "native-compile", *residual_common,
                 "--abi", str(ROOT / "test/examples/attention_mlp_residual_add_abi.json"),
                 "--search-limits", str(ROOT / "test/examples/attention_mlp_graph_search_limits.json"),
                 "--target-source", str(ROOT), "--out", str(residual)],
                env=environment, capture_output=True, text=True, timeout=120, check=False,
            )
            self.assertEqual(residual_run.returncode, 0, residual_run.stderr + residual_run.stdout)
            self.assertEqual(json.loads(residual_run.stdout)["status"], "emitted")
            residual_verify = subprocess.run(
                [str(cli), "native-verify", *residual_common,
                 "--artifact", str(residual), "--out", str(root / "residual-verify.json")],
                env=environment, capture_output=True, text=True, timeout=30, check=False,
            )
            self.assertEqual(residual_verify.returncode, 0, residual_verify.stderr + residual_verify.stdout)
            self.assertEqual(
                json.loads((root / "residual-verify.json").read_text())["status"],
                "verified_artifacts",
            )
            residual_fixture = subprocess.run(
                [sys.executable, str(ROOT / "test/rtl/make_graph_target_fixture.py"),
                 "--compiled", str(residual), "--source", str(residual_source),
                 "--suite", str(ROOT / "test/examples/attention_mlp_residual_add_rtl_suite.json"),
                 "--policy", str(ROOT / "test/examples/attention_mlp_graph_precision.json"),
                 "--case", "residual_add_half", "--out", str(root / "residual-half")],
                env=fixture_env, capture_output=True, text=True, timeout=30, check=False,
            )
            self.assertEqual(residual_fixture.returncode, 0,
                             residual_fixture.stderr + residual_fixture.stdout)
            self.assertEqual(json.loads(residual_fixture.stdout)["expected_bf16_cells"], 3072)


if __name__ == "__main__":
    unittest.main()
