"""Optional Merlin-native raw movement integration; no ACT compiler is used."""

from __future__ import annotations

import copy
import ctypes
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
MERLIN = os.environ.get("MERLIN_NATIVE_ROOT")
ATLAS_BIN = os.environ.get("ATLAS_OOT_BIN_DIR")


@unittest.skipUnless(MERLIN and ATLAS_BIN, "set MERLIN_NATIVE_ROOT and ATLAS_OOT_BIN_DIR")
class NativeBindingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        sys.path.insert(0, str(ROOT))
        sys.path.insert(0, str(ROOT / "python"))
        sys.path.insert(0, str(ROOT / "test"))
        sys.path.insert(0, str(Path(MERLIN) / "src"))
        from atlas_mlir_native.binding import AtlasNativeBinding
        from merlin.semantic_compiler.snapshot import (
            build_native_snapshot,
            packaged_egg_bridge_path,
        )

        cls.binding = AtlasNativeBinding
        cls.temporary = tempfile.TemporaryDirectory()
        cls.work = Path(cls.temporary.name)
        cls.snapshot = build_native_snapshot(
            cls.binding.profile(), destination=cls.work / "snapshot",
            crate=packaged_egg_bridge_path(), cargo_target_dir=cls.work / "cargo",
            source_revision="selected-rtl-copy-integration",
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()
        sys.path.remove(str(Path(MERLIN) / "src"))
        sys.path.remove(str(ROOT / "test"))
        sys.path.remove(str(ROOT / "python"))
        sys.path.remove(str(ROOT))

    def setUp(self) -> None:
        self._test_directory = tempfile.TemporaryDirectory(dir=self.temporary.name)
        self.work = Path(self._test_directory.name)

    def tearDown(self) -> None:
        self._test_directory.cleanup()

    def compile_verified(
        self, request, *, fixed_inputs, fixed_outputs, destination, limits,
    ):
        from merlin.semantic_compiler.target_binding import verify_native_publication

        manifest = self.binding.compile(
            self.snapshot, request, fixed_inputs=fixed_inputs, fixed_outputs=fixed_outputs,
            target_source=ROOT, destination=destination, limits=limits,
        )
        verify_native_publication(
            destination, manifest, engine="merlin_native",
            request_digest=request.digest(), target_identity=request.target_identity,
        )
        return manifest

    def request(self, *, shape: tuple[int, int] = (32, 32), mode: str = "diagnostic"):
        from merlin.semantic_compiler.model import (
            KernelRequest,
            SemanticNode,
            TensorType,
        )

        tensor = TensorType(shape, "bf16", "raw-bit-pattern")
        return KernelRequest(
            (SemanticNode("x", "input", (), tensor, effect="input"),
             SemanticNode("y", "identity", ("x",), tensor)),
            ("y",), ("external",), (("x", "external"),),
            self.binding.profile().target_identity, lowering_policy=mode,
        )

    def transpose_request(self, *, policy: str = "raw-bit-pattern", maps=None):
        from atlas_mlir_native.binding import _transpose_maps
        from merlin.semantic_compiler.model import (
            KernelRequest,
            SemanticNode,
            TensorType,
        )

        tensor = TensorType((32, 32), "fp8_e4m3", policy)
        return KernelRequest(
            (SemanticNode("x", "input", (), tensor, effect="input"),
             SemanticNode("y", "transpose", ("x",), tensor,
                          index_maps=_transpose_maps() if maps is None else maps)),
            ("y",), ("external",), (("x", "external"),),
            self.binding.profile().target_identity, lowering_policy="diagnostic",
        )

    def relu_request(self):
        from atlas_mlir_native.binding import BF16_RELU_ATTRS
        from merlin.semantic_compiler.model import (
            KernelRequest,
            SemanticNode,
            TensorType,
            pointwise_identity_maps,
        )

        tensor = TensorType((32, 32), "bf16", "raw-bit-pattern")
        return KernelRequest(
            (
                SemanticNode("x", "input", (), tensor, effect="input"),
                SemanticNode("y", "relu", ("x",), tensor, attrs=BF16_RELU_ATTRS,
                             index_maps=pointwise_identity_maps(2, 1)),
            ),
            ("y",), ("external",), (("x", "external"),),
            self.binding.profile().target_identity, lowering_policy="diagnostic",
        )

    def add_request(self):
        from atlas_mlir_native.binding import BF16_ADD_ATTRS
        from merlin.semantic_compiler.model import (
            KernelRequest,
            SemanticNode,
            TensorType,
            pointwise_identity_maps,
        )

        tensor = TensorType((32, 32), "bf16", "raw-bit-pattern")
        return KernelRequest(
            (
                SemanticNode("a", "input", (), tensor, attrs=(BF16_ADD_ATTRS[0],), effect="input"),
                SemanticNode("b", "input", (), tensor, attrs=(BF16_ADD_ATTRS[0],), effect="input"),
                SemanticNode("sum", "add_bf16", ("a", "b"), tensor,
                             attrs=BF16_ADD_ATTRS, index_maps=pointwise_identity_maps(2, 2)),
            ),
            ("sum",), ("external",), (("a", "external"), ("b", "external")),
            self.binding.profile().target_identity, lowering_policy="diagnostic",
        )

    def mul_request(self):
        from atlas_mlir_native.binding import BF16_MUL_ATTRS
        from merlin.semantic_compiler.model import (
            KernelRequest,
            SemanticNode,
            TensorType,
            pointwise_identity_maps,
        )

        tensor = TensorType((32, 32), "bf16", "raw-bit-pattern")
        return KernelRequest(
            (
                SemanticNode("a", "input", (), tensor, attrs=(BF16_MUL_ATTRS[0],), effect="input"),
                SemanticNode("b", "input", (), tensor, attrs=(BF16_MUL_ATTRS[0],), effect="input"),
                SemanticNode("product", "mul_bf16", ("a", "b"), tensor,
                             attrs=BF16_MUL_ATTRS, index_maps=pointwise_identity_maps(2, 2)),
            ),
            ("product",), ("external",), (("a", "external"), ("b", "external")),
            self.binding.profile().target_identity, lowering_policy="diagnostic",
        )

    def test_native_optimized_pipeline_compiles_and_tracks_qualification(self) -> None:
        from merlin.semantic_compiler.search import SearchLimits

        with patch.dict(os.environ, ATLAS_NATIVE_PIPELINE="optimized"):
            manifest = self.compile_verified(
                self.request(), fixed_inputs={"x": 3}, fixed_outputs=(35,),
                destination=self.work / "optimized", limits=SearchLimits(),
            )
        self.assertEqual(manifest["lowering_pipeline"], "optimized")
        self.assertEqual(manifest["selected_registers_scope"], "mandatory_qualification")
        self.assertIn("mandatory-qualification-machine.mlir", manifest["stage_artifacts"])
        self.assertIn("mandatory-qualification-program.json", manifest["stage_artifacts"])

    def test_native_bf16_mul_selects_checks_and_emits(self) -> None:
        from atlas_mlir_native.binding import check_bf16_tile_stream
        from atlas_mlir_native.reference import evaluate_bf16_mul
        from merlin.semantic_compiler.reference import TensorValue, evaluate_graph
        from merlin.semantic_compiler.search import SearchLimits
        from merlin.semantic_compiler.target_binding import (
            NativeCompilationError,
        )

        from tools.check_program_decode import audit_program
        from tools.check_variant_inventory import load_inventory

        request = self.mul_request()
        pairs = ((0x3F82, 0x3FA0, 0x3FA2), (0x3F81, 0x3FC0, 0x3FC2),
                 (0xBF82, 0x3FA0, 0xBFA2), (0xBF80, 0xBF80, 0x3F80))
        lhs = tuple(pairs[index % len(pairs)][0] for index in range(1024))
        rhs = tuple(pairs[index % len(pairs)][1] for index in range(1024))
        expected = tuple(pairs[index % len(pairs)][2] for index in range(1024))
        evaluated = evaluate_graph(
            request,
            {"a": TensorValue(request.nodes[0].type, lhs),
             "b": TensorValue(request.nodes[1].type, rhs)},
            primitives={("mul_bf16", "raw-bit-pattern"): evaluate_bf16_mul},
        )[0]
        self.assertEqual(evaluated.elements, expected)
        invalid = list(lhs)
        invalid[0] = 0x7F80
        with self.assertRaisesRegex(ValueError, "finite normal"):
            evaluate_graph(
                request,
                {"a": TensorValue(request.nodes[0].type, tuple(invalid)),
                 "b": TensorValue(request.nodes[1].type, rhs)},
                primitives={("mul_bf16", "raw-bit-pattern"): evaluate_bf16_mul},
            )
        for boundary in (0x0080, 0x7F00):
            with self.subTest(boundary=boundary), self.assertRaisesRegex(ValueError, "normal range"):
                values = (boundary,) * 1024
                evaluate_graph(
                    request,
                    {"a": TensorValue(request.nodes[0].type, values),
                     "b": TensorValue(request.nodes[1].type, values)},
                    primitives={("mul_bf16", "raw-bit-pattern"): evaluate_bf16_mul},
                )

        destination = self.work / "native-bf16-mul"
        self.compile_verified(
            request, fixed_inputs={"a": 3, "b": 4},
            fixed_outputs=(35,),
            destination=destination, limits=SearchLimits(),
        )
        program = json.loads((destination / "physical-program.json").read_text())
        audit = audit_program(load_inventory(), program, (destination / "program.bin").read_bytes())
        self.assertEqual((audit["status"], audit["modes"]["VMUL_BF16"]), ("PASS", 1))
        plan = json.loads((destination / "execution_plan.json").read_text())
        self.assertEqual((plan["operation"], plan["numerical_policy"], plan["input_domain"]),
                         ("bf16_mul", "bf16-product-rne-v1", "finite-normal-bf16-product-normal-v1"))
        changed = copy.deepcopy(program)
        next(row for row in changed["instructions"] if row["operation"] == "atlas.vpu_binary")["fields"]["kind"] = "add"
        selected = self.snapshot.select(request, fixed_inputs={"a": 3, "b": 4}, fixed_outputs=(35,))
        self.assertEqual(selected.status, "selected")
        assert selected.graph and selected.allocation and selected.rules
        names = [selected.rules.symbols[value.symbol].get("descriptor", {}).get("name")
                 for value in selected.graph.values]
        with self.assertRaisesRegex(NativeCompilationError, "BF16 stream non-DMA effect .* differs"):
            check_bf16_tile_stream(
                changed, graph=selected.graph, order=selected.allocation.order,
                names=names, assignment=selected.allocation.addresses,
                source_ids=("a", "b"), input_slots=(3, 4), output_slots=(35,),
            )

    def test_parsed_bf16_mul_uses_the_generated_native_rule(self) -> None:
        from merlin.semantic_compiler.linalg_bridge import (
            LinalgBridgeError,
            translate_linalg_text,
        )
        from merlin.semantic_compiler.search import SearchLimits

        from tools.check_program_decode import audit_program
        from tools.check_variant_inventory import load_inventory

        source = (ROOT / "test/examples/native_bf16_mul_linalg.mlir").read_text()
        request = translate_linalg_text(
            source, entry="mul_tile", target_identity=self.binding.profile().target_identity,
            lowering_policy="diagnostic",
        ).request
        self.assertEqual(tuple(node.op for node in request.nodes), ("input", "input", "mul_bf16"))
        destination = self.work / "parsed-bf16-mul"
        self.compile_verified(
            request, fixed_inputs={"arg0": 3, "arg1": 4},
            fixed_outputs=(35,),
            destination=destination, limits=SearchLimits(),
        )
        program = json.loads((destination / "physical-program.json").read_text())
        audit = audit_program(load_inventory(), program, (destination / "program.bin").read_bytes())
        self.assertEqual((audit["word_count"], audit["modes"]["VMUL_BF16"]), (49, 1))
        changed = source.replace("arith.constant 32767 : i32", "arith.constant 32766 : i32", 1)
        self.assertNotEqual(changed, source)
        with self.assertRaises(LinalgBridgeError):
            translate_linalg_text(
                changed, entry="mul_tile", target_identity=request.target_identity,
                lowering_policy="diagnostic",
            )

    def test_parsed_bf16_mul_chain_shares_producer_and_preserves_both_outputs(self) -> None:
        from atlas_mlir_native.binding import check_bf16_tile_stream
        from atlas_mlir_native.reference import evaluate_bf16_mul
        from merlin.semantic_compiler.linalg_bridge import translate_linalg_text
        from merlin.semantic_compiler.reference import TensorValue, evaluate_graph
        from merlin.semantic_compiler.search import SearchLimits
        from merlin.semantic_compiler.target_binding import (
            NativeCompilationError,
        )
        from test_vpu_mul_reference import _panel, _product_rne

        from tools.check_program_decode import audit_program
        from tools.check_variant_inventory import load_inventory

        request = translate_linalg_text(
            (ROOT / "test/examples/native_bf16_mul_chain_linalg.mlir").read_text(),
            entry="mul_chain", target_identity=self.binding.profile().target_identity,
            lowering_policy="diagnostic",
        ).request
        self.assertEqual(tuple(node.op for node in request.nodes),
                         ("input", "input", "mul_bf16", "mul_bf16"))
        self.assertEqual(request.nodes[-1].inputs,
                         (request.nodes[-2].id, request.nodes[1].id))
        left, right, _ = _panel(0)
        lhs = tuple(struct.unpack_from("<H", left, index)[0] for index in range(0, len(left), 2))
        rhs = tuple(struct.unpack_from("<H", right, index)[0] for index in range(0, len(right), 2))
        expected_first = tuple(_product_rne(a, b) for a, b in zip(lhs, rhs))
        expected_second = tuple(_product_rne(p, b) for p, b in zip(expected_first, rhs))
        observed = evaluate_graph(
            request,
            {"arg0": TensorValue(request.nodes[0].type, lhs),
             "arg1": TensorValue(request.nodes[1].type, rhs)},
            primitives={("mul_bf16", "raw-bit-pattern"): evaluate_bf16_mul},
        )
        self.assertEqual(tuple(value.elements for value in observed),
                         (expected_first, expected_second))

        destination = self.work / "parsed-bf16-mul-chain"
        self.compile_verified(
            request, fixed_inputs={"arg0": 3, "arg1": 4},
            fixed_outputs=(35, 36),
            destination=destination, limits=SearchLimits(),
        )
        plan = json.loads((destination / "execution_plan.json").read_text())
        self.assertEqual(plan["producer_domain_obligations"], [{
            "producer": request.nodes[2].id,
            "consumer": request.nodes[3].id,
            "required_domain": dict(request.nodes[3].attrs)["input_domain"],
            "status": "caller_premise_unchecked",
        }])
        self.assertEqual({row["input"] for row in plan["boundary_domain_obligations"]},
                         {request.nodes[0].id, request.nodes[1].id})
        selection = json.loads((destination / "native-selection.json").read_text())
        names = [value["descriptor"] for value in selection["values"]]
        self.assertEqual(names.count("atlas.vpu_mul_bf16_tile"), 2)
        self.assertEqual(names.count("atlas.load_bf16_tile"), 2)
        program = json.loads((destination / "physical-program.json").read_text())
        audit = audit_program(load_inventory(), program, (destination / "program.bin").read_bytes())
        self.assertEqual((audit["status"], audit["modes"]["VMUL_BF16"]), ("PASS", 2))
        self.assertEqual(audit["modes"]["DMA_STORE_ANY"], 4)
        selected = self.snapshot.select(
            request, fixed_inputs={"arg0": 3, "arg1": 4}, fixed_outputs=(35, 36),
        )
        self.assertEqual(selected.status, "selected")
        assert selected.graph and selected.allocation and selected.rules
        selected_names = [selected.rules.symbols[value.symbol].get("descriptor", {}).get("name")
                          for value in selected.graph.values]
        changed = copy.deepcopy(program)
        binaries = [row for row in changed["instructions"]
                    if row["operation"] == "atlas.vpu_binary"]
        binaries[1]["fields"]["lhs"] = binaries[0]["fields"]["lhs"]
        with self.assertRaisesRegex(NativeCompilationError, "BF16 stream non-DMA effect .* differs"):
            check_bf16_tile_stream(
                changed, graph=selected.graph, order=selected.allocation.order,
                names=selected_names, assignment=selected.allocation.addresses,
                source_ids=("arg0", "arg1"), input_slots=(3, 4), output_slots=(35, 36),
            )

    def test_three_input_bf16_chain_uses_one_native_dag(self) -> None:
        from atlas_mlir_native.binding import BF16_MUL_ATTRS
        from atlas_mlir_native.reference import evaluate_bf16_mul
        from merlin.semantic_compiler.model import (
            KernelRequest,
            SemanticNode,
            TensorType,
            pointwise_identity_maps,
        )
        from merlin.semantic_compiler.reference import TensorValue, evaluate_graph
        from merlin.semantic_compiler.search import SearchLimits
        from test_vpu_mul_reference import _panel, _product_rne

        from tools.check_program_decode import audit_program
        from tools.check_variant_inventory import load_inventory

        tensor = TensorType((32, 32), "bf16", "raw-bit-pattern")
        nodes = tuple(
            SemanticNode(name, "input", (), tensor, attrs=(BF16_MUL_ATTRS[0],), effect="input")
            for name in ("a", "b", "c")
        ) + (
            SemanticNode("first", "mul_bf16", ("a", "b"), tensor,
                         attrs=BF16_MUL_ATTRS, index_maps=pointwise_identity_maps(2, 2)),
            SemanticNode("second", "mul_bf16", ("first", "c"), tensor,
                         attrs=BF16_MUL_ATTRS, index_maps=pointwise_identity_maps(2, 2)),
        )
        request = KernelRequest(
            nodes, ("first", "second"), ("external", "external"),
            (("a", "external"), ("b", "external"), ("c", "external")),
            self.binding.profile().target_identity, lowering_policy="diagnostic",
        )
        a, b, _ = _panel(0)
        c, _, _ = _panel(2)
        operands = tuple(
            tuple(struct.unpack_from("<H", raw, offset)[0] for offset in range(0, len(raw), 2))
            for raw in (a, b, c)
        )
        expected_first = tuple(_product_rne(x, y) for x, y in zip(operands[0], operands[1]))
        expected_second = tuple(_product_rne(x, y) for x, y in zip(expected_first, operands[2]))
        observed = evaluate_graph(
            request,
            {name: TensorValue(tensor, values) for name, values in zip(("a", "b", "c"), operands)},
            primitives={("mul_bf16", "raw-bit-pattern"): evaluate_bf16_mul},
        )
        self.assertEqual(tuple(value.elements for value in observed),
                         (expected_first, expected_second))

        destination = self.work / "three-input-bf16-chain"
        self.compile_verified(
            request, fixed_inputs={"a": 3, "b": 4, "c": 5},
            fixed_outputs=(35, 36),
            destination=destination, limits=SearchLimits(),
        )
        selected = json.loads((destination / "native-selection.json").read_text())
        descriptors = [row["descriptor"] for row in selected["values"]]
        self.assertEqual(descriptors.count("atlas.load_bf16_tile"), 3)
        self.assertEqual(descriptors.count("atlas.vpu_mul_bf16_tile"), 2)
        program = json.loads((destination / "physical-program.json").read_text())
        audit = audit_program(load_inventory(), program, (destination / "program.bin").read_bytes())
        self.assertEqual((audit["status"], audit["modes"]["VMUL_BF16"]), ("PASS", 2))
        self.assertEqual(audit["modes"]["DMA_LOAD_ANY"], 6)
        self.assertEqual(audit["modes"]["DMA_STORE_ANY"], 4)
        sparse = self.work / "noncontiguous-three-inputs"
        self.compile_verified(
            request, fixed_inputs={"a": 3, "b": 4, "c": 6},
            fixed_outputs=(35, 36),
            destination=sparse, limits=SearchLimits(),
        )
        sparse_plan = json.loads((sparse / "execution_plan.json").read_text())
        self.assertEqual(sparse_plan["input_addresses"],
                         [0x90001800, 0x90002000, 0x90003000])
        self.assertIn('"atlas.virtual_dma_load_bf16"',
                      (sparse / "atlas-virtual.mlir").read_text())
        sparse_program = json.loads((sparse / "physical-program.json").read_text())
        sparse_audit = audit_program(load_inventory(), sparse_program,
                                     (sparse / "program.bin").read_bytes())
        self.assertEqual(sparse_audit["status"], "PASS")

    def test_mixed_bf16_add_then_mul_preserves_shared_result(self) -> None:
        from atlas_mlir_native.reference import evaluate_bf16_add, evaluate_bf16_mul
        from merlin.semantic_compiler.linalg_bridge import translate_linalg_text
        from merlin.semantic_compiler.reference import TensorValue, evaluate_graph
        from merlin.semantic_compiler.search import SearchLimits

        from tools.check_program_decode import audit_program
        from tools.check_variant_inventory import load_inventory

        translated = translate_linalg_text(
            (ROOT / "test/examples/native_bf16_add_mul_linalg.mlir").read_text(),
            entry="add_mul", target_identity=self.binding.profile().target_identity,
            lowering_policy="diagnostic",
        )
        request = translated.request
        self.assertEqual(tuple(node.op for node in request.nodes),
                         ("input", "input", "input", "add_bf16", "mul_bf16"))
        tensor = request.nodes[0].type
        values = {
            "arg0": TensorValue(tensor, (0x3F80,) * 1024),
            "arg1": TensorValue(tensor, (0x3F00,) * 1024),
            "arg2": TensorValue(tensor, (0x4000,) * 1024),
        }
        evaluated = evaluate_graph(
            request, values,
            primitives={
                ("add_bf16", "raw-bit-pattern"): evaluate_bf16_add,
                ("mul_bf16", "raw-bit-pattern"): evaluate_bf16_mul,
            },
        )
        self.assertEqual(tuple(value.elements[0] for value in evaluated), (0x3FC0, 0x4040))

        destination = self.work / "mixed-add-mul"
        self.compile_verified(
            request, fixed_inputs={"arg0": 3, "arg1": 4, "arg2": 5},
            fixed_outputs=(35, 36),
            destination=destination, limits=SearchLimits(),
        )
        selected = json.loads((destination / "native-selection.json").read_text())
        descriptors = [row["descriptor"] for row in selected["values"]]
        self.assertEqual(descriptors.count("atlas.vpu_add_bf16_tile"), 1)
        self.assertEqual(descriptors.count("atlas.vpu_mul_bf16_tile"), 1)
        self.assertEqual(descriptors.count("atlas.store_bf16_tile"), 2)
        plan = json.loads((destination / "execution_plan.json").read_text())
        self.assertEqual(plan["operation"], "bf16_mixed")
        self.assertEqual(plan["input_domain"], "per-operation-declared")
        self.assertEqual(plan["producer_domain_obligations"], [{
            "producer": request.nodes[3].id,
            "consumer": request.nodes[4].id,
            "required_domain": dict(request.nodes[4].attrs)["input_domain"],
            "status": "caller_premise_unchecked",
        }])
        self.assertEqual({row["input"] for row in plan["boundary_domain_obligations"]},
                         {node.id for node in request.nodes if node.op == "input"})
        program = json.loads((destination / "physical-program.json").read_text())
        audit = audit_program(load_inventory(), program, (destination / "program.bin").read_bytes())
        self.assertEqual((audit["status"], audit["modes"]["VADD_BF16"],
                          audit["modes"]["VMUL_BF16"]), ("PASS", 1, 1))
        self.assertEqual(audit["modes"]["DMA_STORE_ANY"], 4)

    def test_mixed_bf16_mul_then_add_uses_same_generated_rules(self) -> None:
        from atlas_mlir_native.binding import BF16_ADD_ATTRS, BF16_MUL_ATTRS
        from atlas_mlir_native.reference import evaluate_bf16_add, evaluate_bf16_mul
        from merlin.semantic_compiler.model import (
            KernelRequest,
            SemanticNode,
            TensorType,
            pointwise_identity_maps,
        )
        from merlin.semantic_compiler.reference import TensorValue, evaluate_graph
        from merlin.semantic_compiler.search import SearchLimits

        from tools.check_program_decode import audit_program
        from tools.check_variant_inventory import load_inventory

        tensor = TensorType((32, 32), "bf16", "raw-bit-pattern")
        request = KernelRequest(
            (
                SemanticNode("a", "input", (), tensor, attrs=(BF16_MUL_ATTRS[0],), effect="input"),
                SemanticNode("b", "input", (), tensor, attrs=(BF16_MUL_ATTRS[0],), effect="input"),
                SemanticNode("c", "input", (), tensor, attrs=(BF16_ADD_ATTRS[0],), effect="input"),
                SemanticNode("product", "mul_bf16", ("a", "b"), tensor,
                             attrs=BF16_MUL_ATTRS, index_maps=pointwise_identity_maps(2, 2)),
                SemanticNode("sum", "add_bf16", ("product", "c"), tensor,
                             attrs=BF16_ADD_ATTRS, index_maps=pointwise_identity_maps(2, 2)),
            ),
            ("product", "sum"), ("external", "external"),
            (("a", "external"), ("b", "external"), ("c", "external")),
            self.binding.profile().target_identity, lowering_policy="diagnostic",
        )
        evaluated = evaluate_graph(
            request,
            {"a": TensorValue(tensor, (0x3F80,) * 1024),
             "b": TensorValue(tensor, (0x3F00,) * 1024),
             "c": TensorValue(tensor, (0x4000,) * 1024)},
            primitives={
                ("add_bf16", "raw-bit-pattern"): evaluate_bf16_add,
                ("mul_bf16", "raw-bit-pattern"): evaluate_bf16_mul,
            },
        )
        self.assertEqual(tuple(value.elements[0] for value in evaluated), (0x3F00, 0x4020))
        destination = self.work / "mixed-mul-add"
        self.compile_verified(
            request, fixed_inputs={"a": 3, "b": 4, "c": 5},
            fixed_outputs=(35, 36),
            destination=destination, limits=SearchLimits(),
        )
        program = json.loads((destination / "physical-program.json").read_text())
        audit = audit_program(load_inventory(), program, (destination / "program.bin").read_bytes())
        self.assertEqual((audit["status"], audit["modes"]["VMUL_BF16"],
                          audit["modes"]["VADD_BF16"]), ("PASS", 1, 1))
        plan = json.loads((destination / "execution_plan.json").read_text())
        self.assertEqual((plan["operation"], plan["output_numerical_policies"]),
                         ("bf16_mixed", ["bf16-product-rne-v1", "selected-f32-rne-bf16-chop-v1"]))

    def test_native_bf16_relu_result_feeds_binary_ops(self) -> None:
        from dataclasses import replace

        from atlas_mlir_native.binding import (
            BF16_ADD_ATTRS,
            BF16_MUL_ATTRS,
            BF16_RELU_ATTRS,
        )
        from atlas_mlir_native.reference import evaluate_bf16_add, evaluate_bf16_mul
        from merlin.semantic_compiler.linalg_bridge import translate_linalg_text
        from merlin.semantic_compiler.model import (
            KernelRequest,
            SemanticNode,
            TensorType,
            pointwise_identity_maps,
        )
        from merlin.semantic_compiler.reference import TensorValue, evaluate_graph
        from merlin.semantic_compiler.search import SearchLimits
        from merlin.semantic_compiler.target_binding import verify_native_publication

        from tools.atlas_virtual_evaluator import (
            MemoryRegion,
            RuntimeInputs,
            evaluate,
            parse_program,
        )
        from tools.check_program_decode import audit_program
        from tools.check_variant_inventory import load_inventory

        tensor = TensorType((32, 32), "bf16", "raw-bit-pattern")
        request = KernelRequest(
            (
                SemanticNode("a", "input", (), tensor, attrs=(BF16_ADD_ATTRS[0],), effect="input"),
                SemanticNode("b", "input", (), tensor, attrs=(BF16_ADD_ATTRS[0],), effect="input"),
                SemanticNode("positive", "relu", ("a",), tensor,
                             attrs=BF16_RELU_ATTRS, index_maps=pointwise_identity_maps(2, 1)),
                SemanticNode("sum", "add_bf16", ("positive", "b"), tensor,
                             attrs=BF16_ADD_ATTRS, index_maps=pointwise_identity_maps(2, 2)),
            ),
            ("positive", "sum"), ("external", "external"),
            (("a", "external"), ("b", "external")),
            self.binding.profile().target_identity, lowering_policy="diagnostic",
        )
        values = {
            "a": TensorValue(tensor, (0x3F80, 0x4000) * 512),
            "b": TensorValue(tensor, (0x3F00,) * 1024),
        }
        expected = evaluate_graph(
            request, values,
            primitives={("add_bf16", "raw-bit-pattern"): evaluate_bf16_add},
        )
        self.assertEqual(tuple(value.elements[:2] for value in expected),
                         ((0x3F80, 0x4000), (0x3FC0, 0x4020)))
        # ReLU can also produce zero, outside this add rule's admitted normal
        # operand domain. The diagnostic request does not install a runtime guard.
        with self.assertRaises(ValueError):
            evaluate_graph(
                request,
                {"a": TensorValue(tensor, (0xBF80,) * 1024), "b": values["b"]},
                primitives={("add_bf16", "raw-bit-pattern"): evaluate_bf16_add},
            )

        destination = self.work / "relu-then-add"
        manifest = self.binding.compile(
            self.snapshot, request, fixed_inputs={"a": 3, "b": 5},
            fixed_outputs=(35, 37), target_source=ROOT, destination=destination,
            limits=SearchLimits(),
        )
        plan = json.loads((destination / "execution_plan.json").read_text())
        self.assertFalse(plan["runtime_domain_guard"])
        self.assertEqual(plan["producer_domain_obligations"], [{
            "producer": "positive", "consumer": "sum",
            "required_domain": BF16_ADD_ATTRS[0][1],
            "status": "caller_premise_unchecked",
        }])
        verify_native_publication(
            destination, manifest, engine="merlin_native",
            request_digest=request.digest(), target_identity=request.target_identity,
        )
        program = json.loads((destination / "physical-program.json").read_text())
        self.assertEqual(
            audit_program(load_inventory(), program, (destination / "program.bin").read_bytes())["status"],
            "PASS",
        )
        operations = [row["operation"] for row in program["instructions"]]
        self.assertEqual(operations.count("atlas.vpu_unary"), 1)
        self.assertEqual(operations.count("atlas.vpu_binary"), 1)
        self.assertEqual(operations.count("atlas.dma"), 4)
        source_memory = (
            MemoryRegion(0x90001800, struct.pack("<1024H", *values["a"].elements)),
            MemoryRegion(0x90002800, struct.pack("<1024H", *values["b"].elements)),
            MemoryRegion(0x900117F0, b"L" * 16 + b"?" * 2048 + b"R" * 16),
            MemoryRegion(0x90012010, b"G" * 2016),
            MemoryRegion(0x900127F0, b"X" * 16 + b"?" * 2048 + b"Y" * 16),
        )
        virtual = parse_program((destination / "atlas-virtual.mlir").read_text(),
                                function="native_tile")
        result = evaluate(virtual, RuntimeInputs(memory=source_memory))
        observed = {region.address: region.data for region in result.memory}
        self.assertEqual(observed[0x900117F0],
                         b"L" * 16 + struct.pack("<1024H", *expected[0].elements) + b"R" * 16)
        self.assertEqual(observed[0x900127F0],
                         b"X" * 16 + struct.pack("<1024H", *expected[1].elements) + b"Y" * 16)
        self.assertEqual(observed[0x90012010], b"G" * 2016)
        self.assertEqual(observed[0x90001800], source_memory[0].data)
        self.assertEqual(observed[0x90002800], source_memory[1].data)
        # The physical model can execute a zero-producing ReLU, but that
        # observation does not discharge the selected-core add premise.
        negative = struct.pack("<1024H", *((0xBF80, 0x8000) * 512))
        zero_case = evaluate(virtual, RuntimeInputs(memory=(
            MemoryRegion(0x90001800, negative), source_memory[1],
            source_memory[2], source_memory[3], source_memory[4],
        )))
        zero_memory = {region.address: region.data for region in zero_case.memory}
        self.assertEqual(zero_memory[0x900117F0], b"L" * 16 + b"\x00" * 2048 + b"R" * 16)
        self.assertEqual(zero_memory[0x900127F0],
                         b"X" * 16 + source_memory[1].data + b"Y" * 16)
        self.assertEqual(zero_memory[0x90001800], negative)
        self.assertEqual(zero_memory[0x90012010], b"G" * 2016)
        translated = translate_linalg_text(
            (ROOT / "test/examples/native_bf16_relu_add_linalg.mlir").read_text(),
            entry="relu_then_add", target_identity=request.target_identity,
            lowering_policy="diagnostic",
        )
        self.assertEqual(tuple(node.op for node in translated.request.nodes),
                         ("input", "input", "relu", "add_bf16"))
        parsed_destination = self.work / "parsed-relu-then-add"
        parsed_manifest = self.binding.compile(
            self.snapshot, translated.request, fixed_inputs={"arg0": 3, "arg1": 5},
            fixed_outputs=(35, 37), target_source=ROOT, destination=parsed_destination,
            limits=SearchLimits(),
        )
        verify_native_publication(
            parsed_destination, parsed_manifest, engine="merlin_native",
            request_digest=translated.request.digest(), target_identity=request.target_identity,
        )
        self.assertEqual((parsed_destination / "program.bin").read_bytes(),
                         (destination / "program.bin").read_bytes())

        multiply = replace(
            request,
            nodes=(request.nodes[0],
                   replace(request.nodes[1], attrs=(BF16_MUL_ATTRS[0],)),
                   request.nodes[2],
                   replace(request.nodes[3], op="mul_bf16", attrs=BF16_MUL_ATTRS)),
        )
        product = evaluate_graph(
            multiply, values,
            primitives={("mul_bf16", "raw-bit-pattern"): evaluate_bf16_mul},
        )
        self.assertEqual(tuple(value.elements[:2] for value in product),
                         ((0x3F80, 0x4000), (0x3F00, 0x3F80)))
        mul_destination = self.work / "relu-then-mul"
        mul_manifest = self.binding.compile(
            self.snapshot, multiply, fixed_inputs={"a": 3, "b": 5},
            fixed_outputs=(35, 37), target_source=ROOT, destination=mul_destination,
            limits=SearchLimits(),
        )
        verify_native_publication(
            mul_destination, mul_manifest, engine="merlin_native",
            request_digest=multiply.digest(), target_identity=request.target_identity,
        )
        mul_program = json.loads((mul_destination / "physical-program.json").read_text())
        mul_plan = json.loads((mul_destination / "execution_plan.json").read_text())
        self.assertEqual(mul_plan["producer_domain_obligations"], [{
            "producer": "positive", "consumer": "sum",
            "required_domain": BF16_MUL_ATTRS[0][1],
            "status": "caller_premise_unchecked",
        }])
        mul_audit = audit_program(load_inventory(), mul_program,
                                  (mul_destination / "program.bin").read_bytes())
        self.assertEqual((mul_audit["status"], mul_audit["modes"]["VMUL_BF16"]),
                         ("PASS", 1))

    def test_native_bf16_add_selects_checks_and_emits(self) -> None:
        from atlas_mlir_native.binding import check_bf16_tile_stream
        from atlas_mlir_native.reference import evaluate_bf16_add
        from merlin.semantic_compiler.reference import TensorValue, evaluate_graph
        from merlin.semantic_compiler.search import SearchLimits
        from merlin.semantic_compiler.target_binding import (
            NativeCompilationError,
        )

        from tools.check_program_decode import audit_program
        from tools.check_variant_inventory import load_inventory

        request = self.add_request()
        left = tuple((0x3F80, 0x3F80, 0xBF80)[index % 3] for index in range(1024))
        right = tuple((0x3BC0, 0x4000, 0x3F00)[index % 3] for index in range(1024))
        expected = (0x3F80, 0x4040, 0xBF00)
        evaluated = evaluate_graph(
            request,
            {"a": TensorValue(request.nodes[0].type, left),
             "b": TensorValue(request.nodes[1].type, right)},
            primitives={("add_bf16", "raw-bit-pattern"): evaluate_bf16_add},
        )[0]
        self.assertEqual(evaluated.elements, tuple(expected[index % 3] for index in range(1024)))
        invalid = list(right)
        invalid[0] = 0x7F80
        with self.assertRaisesRegex(ValueError, "finite normal"):
            evaluate_graph(
                request,
                {"a": TensorValue(request.nodes[0].type, left),
                 "b": TensorValue(request.nodes[1].type, tuple(invalid))},
                primitives={("add_bf16", "raw-bit-pattern"): evaluate_bf16_add},
            )

        destination = self.work / "native-bf16-add"
        manifest = self.compile_verified(
            request, fixed_inputs={"a": 3, "b": 4},
            fixed_outputs=(35,),
            destination=destination, limits=SearchLimits(),
        )
        program = json.loads((destination / "physical-program.json").read_text())
        self.assertEqual(sum(row["operation"] == "atlas.vpu_binary" for row in program["instructions"]), 1)
        plan = json.loads((destination / "execution_plan.json").read_text())
        self.assertEqual((plan["operation"], plan["input_domain"]),
                         ("bf16_add", "finite-normal-exact-fp32-sum-v1"))
        self.assertEqual(manifest["selected_registers"], [0, 1, 2, 3, 4, 5])
        audit = audit_program(load_inventory(), program, (destination / "program.bin").read_bytes())
        self.assertEqual((audit["status"], audit["modes"]["VADD_BF16"]), ("PASS", 1))
        selection = self.snapshot.select(request, fixed_inputs={"a": 3, "b": 4}, fixed_outputs=(35,))
        self.assertEqual(selection.status, "selected")
        assert selection.graph and selection.allocation and selection.rules
        names = [selection.rules.symbols[value.symbol].get("descriptor", {}).get("name")
                 for value in selection.graph.values]
        checker = {"graph": selection.graph, "order": selection.allocation.order,
                   "names": names, "assignment": selection.allocation.addresses,
                   "source_ids": ("a", "b"), "input_slots": (3, 4),
                   "output_slots": (35,)}
        check_bf16_tile_stream(program, **checker)
        altered = copy.deepcopy(program)
        next(row for row in altered["instructions"] if row["operation"] == "atlas.vpu_binary")["fields"]["rhs"] = 0
        with self.assertRaisesRegex(NativeCompilationError, "BF16 stream non-DMA effect .* differs"):
            check_bf16_tile_stream(altered, **checker)

    def test_parsed_bf16_sub_selects_checks_and_emits(self) -> None:
        from atlas_mlir_native.binding import BF16_SUB_ATTRS, check_bf16_tile_stream
        from atlas_mlir_native.reference import evaluate_bf16_sub
        from merlin.semantic_compiler.linalg_bridge import translate_linalg_text
        from merlin.semantic_compiler.reference import TensorValue, evaluate_graph
        from merlin.semantic_compiler.search import SearchLimits
        from merlin.semantic_compiler.target_binding import (
            NativeCompilationError,
        )

        from tools.check_program_decode import audit_program
        from tools.check_variant_inventory import load_inventory

        source = (ROOT / "test/examples/native_bf16_sub_linalg.mlir").read_text()
        request = translate_linalg_text(
            source, entry="sub_tile", target_identity=self.binding.profile().target_identity,
            lowering_policy="diagnostic",
        ).request
        self.assertEqual(tuple(node.op for node in request.nodes), ("input", "input", "sub_bf16"))
        self.assertEqual(request.nodes[-1].attrs, BF16_SUB_ATTRS)
        pairs = ((0x3F80, 0x3F80, 0), (0xBF80, 0xBF80, 0),
                 (0x3F80, 0xBBC0, 0x3F80), (0x3F00, 0x3F80, 0xBF00))
        lhs = tuple(pairs[index % len(pairs)][0] for index in range(1024))
        rhs = tuple(pairs[index % len(pairs)][1] for index in range(1024))
        expected = tuple(pairs[index % len(pairs)][2] for index in range(1024))
        result = evaluate_graph(request, {
            "arg0": TensorValue(request.nodes[0].type, lhs),
            "arg1": TensorValue(request.nodes[1].type, rhs),
        }, primitives={("sub_bf16", "raw-bit-pattern"): evaluate_bf16_sub})
        self.assertEqual(result[0].elements, expected)
        bad = list(rhs)
        bad[0] = 0x7F80
        with self.assertRaisesRegex(ValueError, "finite normal"):
            evaluate_graph(request, {
                "arg0": TensorValue(request.nodes[0].type, lhs),
                "arg1": TensorValue(request.nodes[1].type, tuple(bad)),
            }, primitives={("sub_bf16", "raw-bit-pattern"): evaluate_bf16_sub})

        fixed_inputs = {"arg0": 3, "arg1": 4}
        destination = self.work / "parsed-bf16-sub"
        self.compile_verified(
            request, fixed_inputs=fixed_inputs,
            fixed_outputs=(35,),
            destination=destination, limits=SearchLimits(),
        )
        plan = json.loads((destination / "execution_plan.json").read_text())
        self.assertEqual(plan["operation"], "bf16_sub")
        self.assertEqual(plan["input_domain"], BF16_SUB_ATTRS[0][1])
        program = json.loads((destination / "physical-program.json").read_text())
        self.assertEqual(audit_program(
            load_inventory(), program, (destination / "program.bin").read_bytes(),
        )["modes"]["VSUB_BF16"], 1)
        selected = self.snapshot.select(
            request, fixed_inputs=fixed_inputs, fixed_outputs=(35,), limits=SearchLimits(),
        )
        self.assertEqual(selected.status, "selected")
        assert selected.graph and selected.allocation and selected.rules
        arguments = {
            "graph": selected.graph, "order": selected.allocation.order,
            "names": [selected.rules.symbols[value.symbol].get("descriptor", {}).get("name")
                      for value in selected.graph.values],
            "assignment": selected.allocation.addresses,
            "source_ids": ("arg0", "arg1"), "input_slots": (3, 4), "output_slots": (35,),
        }
        check_bf16_tile_stream(program, **arguments)
        changed = copy.deepcopy(program)
        next(row for row in changed["instructions"]
             if row["operation"] == "atlas.vpu_binary")["fields"]["kind"] = "add"
        with self.assertRaises(NativeCompilationError):
            check_bf16_tile_stream(changed, **arguments)

    def test_native_bf16_row_max_selects_and_preserves_raw_winners(self) -> None:
        from atlas_mlir_native.binding import (
            BF16_ROW_MAX_ATTRS,
            _row_reduce_maps,
            check_bf16_tile_stream,
        )
        from atlas_mlir_native.reference import evaluate_bf16_row_max
        from merlin.semantic_compiler.model import (
            KernelRequest,
            SemanticNode,
            TensorType,
        )
        from merlin.semantic_compiler.reference import TensorValue, evaluate_graph
        from merlin.semantic_compiler.search import SearchLimits
        from merlin.semantic_compiler.target_binding import (
            NativeCompilationError,
        )

        from tools.check_program_decode import audit_program
        from tools.check_variant_inventory import load_inventory

        tensor = TensorType((32, 32), "bf16", "raw-bit-pattern")
        request = KernelRequest(
            (SemanticNode("scores", "input", (), tensor,
                          attrs=(BF16_ROW_MAX_ATTRS[0],), effect="input"),
             SemanticNode("maximum", "row_max_bf16", ("scores",), tensor,
                          attrs=BF16_ROW_MAX_ATTRS, index_maps=_row_reduce_maps())),
            ("maximum",), ("external",), (("scores", "external"),),
            self.binding.profile().target_identity, lowering_policy="diagnostic",
        )
        fixture = KernelRequest.from_record(json.loads(
            (ROOT / "test/examples/native_bf16_row_max_request.json").read_text()
        ))
        self.assertEqual(fixture.digest(), request.digest())
        cases = ((0xBF80, 0xC000, 0xBF80),
                 (0x3F80, 0x4000, 0x4000),
                 (0x7F80, 0x7FC1, 0x7FC1),
                 (0x8000, 0x0000, 0x0000))
        source = tuple(code for row in range(32)
                       for code in (cases[row % len(cases)][0],) * 31
                       + (cases[row % len(cases)][1],))
        expected = tuple(cases[row % len(cases)][2] for row in range(32)
                         for _ in range(32))
        result = evaluate_graph(
            request, {"scores": TensorValue(tensor, source)},
            primitives={("row_max_bf16", "raw-bit-pattern"): evaluate_bf16_row_max},
        )
        self.assertEqual(result[0].elements, expected)

        destination = self.work / "native-bf16-row-max"
        self.compile_verified(
            request, fixed_inputs={"scores": 3},
            fixed_outputs=(35,),
            destination=destination, limits=SearchLimits(),
        )
        plan = json.loads((destination / "execution_plan.json").read_text())
        self.assertEqual((plan["operation"], plan["input_domain"]),
                         ("bf16_row_max", BF16_ROW_MAX_ATTRS[0][1]))
        program = json.loads((destination / "physical-program.json").read_text())
        self.assertEqual(audit_program(
            load_inventory(), program, (destination / "program.bin").read_bytes(),
        )["modes"]["VREDMAX_ROW_BF16"], 1)
        selected = self.snapshot.select(
            request, fixed_inputs={"scores": 3}, fixed_outputs=(35,), limits=SearchLimits(),
        )
        self.assertEqual(selected.status, "selected")
        assert selected.graph and selected.allocation and selected.rules
        names = [selected.rules.symbols[value.symbol].get("descriptor", {}).get("name")
                 for value in selected.graph.values]
        self.assertEqual(names.count("atlas.vpu_row_max_bf16_tile"), 1)
        checker = {
            "graph": selected.graph, "order": selected.allocation.order,
            "names": names, "assignment": selected.allocation.addresses,
            "source_ids": ("scores",), "input_slots": (3,), "output_slots": (35,),
        }
        check_bf16_tile_stream(program, **checker)
        altered = copy.deepcopy(program)
        next(row for row in altered["instructions"]
             if row["operation"] == "atlas.vpu_reduce")["fields"]["kind"] = "col_sum"
        with self.assertRaisesRegex(NativeCompilationError, "unadmitted reduction mode"):
            check_bf16_tile_stream(altered, **checker)
        altered = copy.deepcopy(program)
        next(row for row in altered["instructions"]
             if row["operation"] == "atlas.vpu_reduce")["fields"]["kind"] = "row_sum"
        with self.assertRaises(NativeCompilationError):
            check_bf16_tile_stream(altered, **checker)

        wrong_map = copy.deepcopy(request.record())
        wrong_map["nodes"][1]["index_maps"][0]["coefficients"][1] = [0, 1, 0]
        changed = KernelRequest.from_record(wrong_map)
        with self.assertRaises(NativeCompilationError):
            self.binding.compile(
                self.snapshot, changed, fixed_inputs={"scores": 3}, fixed_outputs=(35,),
                target_source=ROOT, destination=self.work / "row-max-wrong-map",
                limits=SearchLimits(),
            )

    def test_native_bf16_row_sum_uses_selected_fp32_tree(self) -> None:
        from atlas_mlir_native.binding import (
            BF16_ROW_SUM_ATTRS,
            _row_reduce_maps,
            check_bf16_tile_stream,
        )
        from atlas_mlir_native.reference import evaluate_bf16_row_sum
        from merlin.semantic_compiler.model import (
            KernelRequest,
            SemanticNode,
            TensorType,
        )
        from merlin.semantic_compiler.reference import TensorValue, evaluate_graph
        from merlin.semantic_compiler.search import SearchLimits
        from merlin.semantic_compiler.target_binding import (
            NativeCompilationError,
        )
        from test_vpu_row_sum_reference import _panel

        from tools.check_program_decode import audit_program
        from tools.check_variant_inventory import load_inventory

        tensor = TensorType((32, 32), "bf16", "raw-bit-pattern")
        request = KernelRequest(
            (SemanticNode("scores", "input", (), tensor,
                          attrs=(BF16_ROW_SUM_ATTRS[0],), effect="input"),
             SemanticNode("summed", "row_sum_bf16", ("scores",), tensor,
                          attrs=BF16_ROW_SUM_ATTRS, index_maps=_row_reduce_maps())),
            ("summed",), ("external",), (("scores", "external"),),
            self.binding.profile().target_identity, lowering_policy="diagnostic",
        )
        fixture = KernelRequest.from_record(json.loads(
            (ROOT / "test/examples/native_bf16_row_sum_request.json").read_text()
        ))
        self.assertEqual(fixture.digest(), request.digest())

        def logical(physical: bytes) -> tuple[int, ...]:
            halves = struct.unpack("<1024H", physical)
            return tuple(halves[row * 16 + column] if column < 16
                         else halves[512 + row * 16 + column - 16]
                         for row in range(32) for column in range(32))

        for phase in (0, 2):
            source, expected = _panel(phase)
            result = evaluate_graph(
                request, {"scores": TensorValue(tensor, logical(source))},
                primitives={("row_sum_bf16", "raw-bit-pattern"): evaluate_bf16_row_sum},
            )
            self.assertEqual(result[0].elements, logical(expected))
            self.assertEqual(result[0].elements[:32], (0x4000,) * 32)
        for code in (0x8000, 0x7F80, 0x0001):
            with self.subTest(invalid_code=code), self.assertRaises(ValueError):
                evaluate_graph(
                    request, {"scores": TensorValue(tensor, (code,) + (0,) * 1023)},
                    primitives={("row_sum_bf16", "raw-bit-pattern"): evaluate_bf16_row_sum},
                )
        with self.assertRaisesRegex(ValueError, "overflows FP32"):
            evaluate_graph(
                request, {"scores": TensorValue(tensor, (0x7F7F,) * 1024)},
                primitives={("row_sum_bf16", "raw-bit-pattern"): evaluate_bf16_row_sum},
            )

        destination = self.work / "native-bf16-row-sum"
        self.compile_verified(
            request, fixed_inputs={"scores": 3},
            fixed_outputs=(35,),
            destination=destination, limits=SearchLimits(),
        )
        plan = json.loads((destination / "execution_plan.json").read_text())
        self.assertEqual((plan["operation"], plan["input_domain"]),
                         ("bf16_row_sum", BF16_ROW_SUM_ATTRS[0][1]))
        program = json.loads((destination / "physical-program.json").read_text())
        self.assertEqual(audit_program(
            load_inventory(), program, (destination / "program.bin").read_bytes(),
        )["modes"]["VREDSUM_ROW_BF16"], 1)
        selected = self.snapshot.select(
            request, fixed_inputs={"scores": 3}, fixed_outputs=(35,), limits=SearchLimits(),
        )
        self.assertEqual(selected.status, "selected")
        assert selected.graph and selected.allocation and selected.rules
        names = [selected.rules.symbols[value.symbol].get("descriptor", {}).get("name")
                 for value in selected.graph.values]
        self.assertEqual(names.count("atlas.vpu_row_sum_bf16_tile"), 1)
        checker = {
            "graph": selected.graph, "order": selected.allocation.order,
            "names": names, "assignment": selected.allocation.addresses,
            "source_ids": ("scores",), "input_slots": (3,), "output_slots": (35,),
        }
        check_bf16_tile_stream(program, **checker)
        altered = copy.deepcopy(program)
        next(row for row in altered["instructions"]
             if row["operation"] == "atlas.vpu_reduce")["fields"]["kind"] = "row_max"
        with self.assertRaises(NativeCompilationError):
            check_bf16_tile_stream(altered, **checker)

        wrong_policy = copy.deepcopy(request.record())
        wrong_policy["nodes"][1]["attrs"]["numeric_semantics"] = "left-to-right-fp32-sum"
        with self.assertRaises(NativeCompilationError):
            self.binding.compile(
                self.snapshot, KernelRequest.from_record(wrong_policy),
                fixed_inputs={"scores": 3}, fixed_outputs=(35,), target_source=ROOT,
                destination=self.work / "row-sum-wrong-policy", limits=SearchLimits(),
            )

    def test_native_bf16_exp_selects_lut_rule_and_checks_emission(self) -> None:
        from atlas_mlir_native.binding import BF16_EXP_ATTRS, check_bf16_tile_stream
        from atlas_mlir_native.exp_lane import vexp_bf16_bits
        from atlas_mlir_native.reference import evaluate_bf16_exp
        from merlin.semantic_compiler.model import (
            KernelRequest,
            SemanticNode,
            TensorType,
            pointwise_identity_maps,
        )
        from merlin.semantic_compiler.reference import TensorValue, evaluate_graph
        from merlin.semantic_compiler.search import SearchLimits
        from merlin.semantic_compiler.target_binding import (
            NativeCompilationError,
        )

        from tools.atlas_virtual_evaluator import (
            RuntimeInputs,
            Tile,
            evaluate,
            parse_program,
        )
        from tools.check_program_decode import audit_program
        from tools.check_variant_inventory import load_inventory

        tensor = TensorType((32, 32), "bf16", "raw-bit-pattern")
        request = KernelRequest(
            (SemanticNode("x", "input", (), tensor, attrs=(BF16_EXP_ATTRS[0],),
                          effect="input"),
             SemanticNode("y", "exp_bf16", ("x",), tensor,
                          attrs=BF16_EXP_ATTRS,
                          index_maps=pointwise_identity_maps(2, 1))),
            ("y",), ("external",), (("x", "external"),),
            self.binding.profile().target_identity, lowering_policy="diagnostic",
        )
        fixture = KernelRequest.from_record(json.loads(
            (ROOT / "test/examples/native_bf16_exp_request.json").read_text()
        ))
        self.assertEqual(fixture.digest(), request.digest())
        codes = (0x0000, 0x8000, 0x3F00, 0xBF00, 0x3F80, 0xBF80,
                 0x3FC0, 0xBFC0, 0x42A0, 0xC2A0)
        source = tuple(codes[(i * 7 + i // 16) % len(codes)] for i in range(1024))
        expected = tuple(vexp_bf16_bits(code) for code in source)
        evaluated = evaluate_graph(
            request, {"x": TensorValue(tensor, source)},
            primitives={("exp_bf16", "raw-bit-pattern"): evaluate_bf16_exp},
        )
        self.assertEqual(evaluated[0].elements, expected)
        with self.assertRaises(ValueError):
            evaluate_graph(
                request, {"x": TensorValue(tensor, (0x7F80,) + source[1:])},
                primitives={("exp_bf16", "raw-bit-pattern"): evaluate_bf16_exp},
            )

        destination = self.work / "native-bf16-exp"
        self.compile_verified(
            request, fixed_inputs={"x": 3},
            fixed_outputs=(35,),
            destination=destination, limits=SearchLimits(),
        )
        plan = json.loads((destination / "execution_plan.json").read_text())
        self.assertEqual((plan["operation"], plan["input_domain"]),
                         ("bf16_exp", BF16_EXP_ATTRS[0][1]))
        program = json.loads((destination / "physical-program.json").read_text())
        self.assertEqual(audit_program(
            load_inventory(), program, (destination / "program.bin").read_bytes(),
        )["modes"]["VEXP"], 1)
        virtual = parse_program((destination / "atlas-virtual.mlir").read_text(),
                                function="native_tile")
        self.assertEqual(evaluate(virtual, RuntimeInputs(tiles={0: Tile("bf16", source)}))
                         .outputs[0].bits, expected)
        selected = self.snapshot.select(
            request, fixed_inputs={"x": 3}, fixed_outputs=(35,), limits=SearchLimits(),
        )
        self.assertEqual(selected.status, "selected")
        assert selected.graph and selected.allocation and selected.rules
        names = [selected.rules.symbols[value.symbol].get("descriptor", {}).get("name")
                 for value in selected.graph.values]
        self.assertEqual(names.count("atlas.vpu_exp_bf16_tile"), 1)
        checker = {
            "graph": selected.graph, "order": selected.allocation.order,
            "names": names, "assignment": selected.allocation.addresses,
            "source_ids": ("x",), "input_slots": (3,), "output_slots": (35,),
        }
        check_bf16_tile_stream(program, **checker)
        altered = copy.deepcopy(program)
        next(row for row in altered["instructions"]
             if row["operation"] == "atlas.vpu_unary")["fields"]["kind"] = "exp2"
        with self.assertRaises(NativeCompilationError):
            check_bf16_tile_stream(altered, **checker)

        wrong_policy = copy.deepcopy(request.record())
        wrong_policy["nodes"][1]["attrs"]["numeric_semantics"] = "exact-real-exp-rne-bf16"
        with self.assertRaises(NativeCompilationError):
            self.binding.compile(
                self.snapshot, KernelRequest.from_record(wrong_policy),
                fixed_inputs={"x": 3}, fixed_outputs=(35,), target_source=ROOT,
                destination=self.work / "native-bf16-exp-wrong-policy",
                limits=SearchLimits(),
            )

    def test_native_bf16_recip_selects_lut_rule_and_checks_emission(self) -> None:
        from atlas_mlir_native.binding import BF16_RECIP_ATTRS, check_bf16_tile_stream
        from atlas_mlir_native.recip_lane import vrecip_bf16_bits
        from atlas_mlir_native.reference import evaluate_bf16_recip
        from merlin.semantic_compiler.model import KernelRequest
        from merlin.semantic_compiler.reference import TensorValue, evaluate_graph
        from merlin.semantic_compiler.search import SearchLimits
        from merlin.semantic_compiler.target_binding import (
            NativeCompilationError,
        )

        from tools.atlas_virtual_evaluator import (
            RuntimeInputs,
            Tile,
            evaluate,
            parse_program,
        )
        from tools.check_program_decode import audit_program
        from tools.check_variant_inventory import load_inventory

        request = KernelRequest.from_record(json.loads(
            (ROOT / "test/examples/native_bf16_recip_request.json").read_text()
        ))
        self.assertEqual(request.nodes[1].attrs, BF16_RECIP_ATTRS)
        codes = tuple(range(0x3F80, 0x4201))
        source = tuple(codes[(i * 113) % len(codes)] for i in range(1024))
        expected = tuple(vrecip_bf16_bits(code) for code in source)
        evaluated = evaluate_graph(
            request, {"x": TensorValue(request.nodes[0].type, source)},
            primitives={("recip_bf16", "raw-bit-pattern"): evaluate_bf16_recip},
        )
        self.assertEqual(evaluated[0].elements, expected)
        with self.assertRaises(ValueError):
            evaluate_graph(
                request, {"x": TensorValue(request.nodes[0].type, (0x0000,) + source[1:])},
                primitives={("recip_bf16", "raw-bit-pattern"): evaluate_bf16_recip},
            )

        destination = self.work / "native-bf16-recip"
        self.compile_verified(
            request, fixed_inputs={"x": 3},
            fixed_outputs=(35,),
            destination=destination, limits=SearchLimits(),
        )
        plan = json.loads((destination / "execution_plan.json").read_text())
        self.assertEqual((plan["operation"], plan["input_domain"]),
                         ("bf16_recip", BF16_RECIP_ATTRS[0][1]))
        program = json.loads((destination / "physical-program.json").read_text())
        self.assertEqual(audit_program(
            load_inventory(), program, (destination / "program.bin").read_bytes(),
        )["modes"]["VRECIP_BF16"], 1)
        virtual = parse_program((destination / "atlas-virtual.mlir").read_text(),
                                function="native_tile")
        self.assertEqual(evaluate(virtual, RuntimeInputs(tiles={0: Tile("bf16", source)}))
                         .outputs[0].bits, expected)
        selected = self.snapshot.select(
            request, fixed_inputs={"x": 3}, fixed_outputs=(35,), limits=SearchLimits(),
        )
        self.assertEqual(selected.status, "selected")
        assert selected.graph and selected.allocation and selected.rules
        names = [selected.rules.symbols[value.symbol].get("descriptor", {}).get("name")
                 for value in selected.graph.values]
        self.assertEqual(names.count("atlas.vpu_recip_bf16_tile"), 1)
        checker = {
            "graph": selected.graph, "order": selected.allocation.order,
            "names": names, "assignment": selected.allocation.addresses,
            "source_ids": ("x",), "input_slots": (3,), "output_slots": (35,),
        }
        check_bf16_tile_stream(program, **checker)
        altered = copy.deepcopy(program)
        next(row for row in altered["instructions"]
             if row["operation"] == "atlas.vpu_unary")["fields"]["kind"] = "exp"
        with self.assertRaises(NativeCompilationError):
            check_bf16_tile_stream(altered, **checker)

        wrong_policy = copy.deepcopy(request.record())
        wrong_policy["nodes"][1]["attrs"]["numeric_semantics"] = "real-recip-rne-bf16"
        with self.assertRaises(NativeCompilationError):
            self.binding.compile(
                self.snapshot, KernelRequest.from_record(wrong_policy),
                fixed_inputs={"x": 3}, fixed_outputs=(35,), target_source=ROOT,
                destination=self.work / "native-bf16-recip-wrong-policy",
                limits=SearchLimits(),
            )

    def test_parsed_bf16_add_lowers_to_the_same_checked_tile(self) -> None:
        from atlas_mlir_native.reference import evaluate_bf16_add
        from merlin.semantic_compiler.linalg_bridge import (
            LinalgBridgeError,
            translate_linalg_text,
        )
        from merlin.semantic_compiler.reference import TensorValue, evaluate_graph
        from merlin.semantic_compiler.search import SearchLimits

        source = (ROOT / "test/examples/native_bf16_add_linalg.mlir").read_text()
        request = translate_linalg_text(
            source, entry="add_tile", target_identity=self.binding.profile().target_identity,
            lowering_policy="diagnostic",
        ).request
        self.assertEqual(request.nodes[-1].op, "add_bf16")
        evaluated = evaluate_graph(
            request,
            {"arg0": TensorValue(request.nodes[0].type, (0x3F80,) * 1024),
             "arg1": TensorValue(request.nodes[1].type, (0x3BC0,) * 1024)},
            primitives={("add_bf16", "raw-bit-pattern"): evaluate_bf16_add},
        )
        self.assertEqual(evaluated[0].elements, (0x3F80,) * 1024)
        destination = self.work / "parsed-bf16-add"
        self.compile_verified(
            request, fixed_inputs={"arg0": 3, "arg1": 4},
            fixed_outputs=(35,),
            destination=destination, limits=SearchLimits(),
        )
        program = json.loads((destination / "physical-program.json").read_text())
        self.assertEqual(sum(row["operation"] == "atlas.vpu_binary" for row in program["instructions"]), 1)
        self.assertEqual(len((destination / "program.bin").read_bytes()) // 4, 49)
        changed = source.replace("arith.constant 16 : i32", "arith.constant 15 : i32", 1)
        with self.assertRaises(LinalgBridgeError):
            translate_linalg_text(
                changed, entry="add_tile", target_identity=request.target_identity,
                lowering_policy="diagnostic",
            )

    def test_parsed_bf16_add_then_relu_rechecks_physical_pairs(self) -> None:
        from atlas_mlir_native.binding import (
            BF16_ADD_ATTRS,
            _realized_bf16_assignment,
            check_bf16_tile_stream,
        )
        from atlas_mlir_native.reference import evaluate_bf16_add
        from merlin.semantic_compiler.linalg_bridge import translate_linalg_text
        from merlin.semantic_compiler.reference import TensorValue, evaluate_graph
        from merlin.semantic_compiler.search import SearchLimits
        from merlin.semantic_compiler.target_binding import (
            NativeCompilationError,
            verify_native_publication,
        )

        source = (ROOT / "test/examples/native_bf16_add_relu_linalg.mlir").read_text()
        request = translate_linalg_text(
            source, entry="add_and_relu", target_identity=self.binding.profile().target_identity,
            lowering_policy="diagnostic",
        ).request
        self.assertEqual(tuple(node.op for node in request.nodes),
                         ("input", "input", "add_bf16", "relu"))
        self.assertEqual(len(request.outputs), 2)
        left = tuple((0x3F80, 0xBF80)[index % 2] for index in range(1024))
        right = tuple((0x3BC0, 0x3F00)[index % 2] for index in range(1024))
        raw, relu = evaluate_graph(
            request,
            {"arg0": TensorValue(request.nodes[0].type, left),
             "arg1": TensorValue(request.nodes[1].type, right)},
            primitives={("add_bf16", "raw-bit-pattern"): evaluate_bf16_add},
        )
        self.assertEqual(raw.elements, tuple((0x3F80, 0xBF00)[index % 2] for index in range(1024)))
        self.assertEqual(relu.elements, tuple((0x3F80, 0)[index % 2] for index in range(1024)))

        destination = self.work / "parsed-bf16-add-relu"
        manifest = self.compile_verified(
            request, fixed_inputs={"arg0": 3, "arg1": 4},
            fixed_outputs=(35, 36),
            destination=destination, limits=SearchLimits(),
        )
        program = json.loads((destination / "physical-program.json").read_text())
        self.assertEqual((program["word_count"], len(manifest["selected_registers"])), (66, 6))
        self.assertEqual(sum(row["operation"] == "atlas.vpu_binary" for row in program["instructions"]), 1)
        self.assertEqual(sum(row["operation"] == "atlas.vpu_unary" for row in program["instructions"]), 1)
        plan = json.loads((destination / "execution_plan.json").read_text())
        self.assertEqual((plan["operation"], plan["numerical_policy"], plan["input_domain"]),
                         ("bf16_add_relu", "per-output", "finite-normal-exact-fp32-sum-v1"))
        self.assertEqual(plan["output_numerical_policies"],
                         [BF16_ADD_ATTRS[1][1], "bf16-sign-clamp-v1"])

        selected = self.snapshot.select(request, fixed_inputs={"arg0": 3, "arg1": 4}, fixed_outputs=(35, 36))
        self.assertEqual(selected.status, "selected")
        assert selected.graph and selected.allocation and selected.rules
        names = [selected.rules.symbols[value.symbol].get("descriptor", {}).get("name")
                 for value in selected.graph.values]
        physical = _realized_bf16_assignment(
            program, graph=selected.graph, order=selected.allocation.order,
            names=names, selected=selected.allocation.addresses,
        )
        self.assertNotEqual(physical, selected.allocation.addresses)
        check_bf16_tile_stream(
            program, graph=selected.graph, order=selected.allocation.order,
            names=names, assignment=physical, source_ids=("arg0", "arg1"),
            input_slots=(3, 4), output_slots=(35, 36),
        )
        changed = copy.deepcopy(program)
        next(row for row in changed["instructions"] if row["operation"] == "atlas.vpu_binary")["fields"]["rhs"] = 0
        with self.assertRaisesRegex(NativeCompilationError, "different source pairs"):
            _realized_bf16_assignment(
                changed, graph=selected.graph, order=selected.allocation.order,
                names=names, selected=selected.allocation.addresses,
            )
        selection_path = destination / "native-selection.json"
        selection = json.loads(selection_path.read_text())
        self.assertEqual(selection["check_fingerprint"], manifest["native_check_fingerprint"])
        self.assertEqual(selection["graph_outputs"], list(selected.graph.outputs))
        self.assertEqual(
            {row["id"]: row["address"] for row in selection["values"]}, physical
        )
        selection_bytes = selection_path.read_bytes()
        selection_path.write_bytes(selection_bytes + b" ")
        with self.assertRaisesRegex(ValueError, "native stage artifact changed"):
            verify_native_publication(
                destination, manifest, engine="merlin_native",
                request_digest=request.digest(), target_identity=request.target_identity,
            )
        selection_path.write_bytes(selection_bytes)
        machine_path = destination / "atlas-machine.mlir"
        machine_bytes = machine_path.read_bytes()
        machine_path.write_bytes(machine_bytes + b" ")
        with self.assertRaisesRegex(ValueError, "native stage artifact changed"):
            verify_native_publication(
                destination, manifest, engine="merlin_native",
                request_digest=request.digest(), target_identity=request.target_identity,
            )
        machine_path.write_bytes(machine_bytes)
        verify_native_publication(
            destination, manifest, engine="merlin_native",
            request_digest=request.digest(), target_identity=request.target_identity,
        )

    def pack_request(self, *, scale_code: int = 127, policy: str | None = None):
        from atlas_mlir_native.binding import (
            PACK_ATTRS,
            PACK_POLICY,
            ROUNDED_PACK_DOMAIN,
            ROUNDED_PACK_POLICY,
            SCALED_PACK_DOMAIN,
            SCALED_PACK_POLICY,
        )
        from merlin.semantic_compiler.model import (
            KernelRequest,
            SemanticNode,
            TensorType,
            pointwise_identity_maps,
        )

        source = TensorType((32, 32), "bf16", "raw-bit-pattern")
        selected_policy = policy or PACK_POLICY
        packed = TensorType((32, 32), "fp8_e4m3", selected_policy)
        domain = {
            SCALED_PACK_POLICY: SCALED_PACK_DOMAIN,
            ROUNDED_PACK_POLICY: ROUNDED_PACK_DOMAIN,
        }.get(selected_policy)
        scaled = domain is not None
        attributes = tuple((key, scale_code if key == "scale_code" else value)
                           for key, value in (
                               (("input_domain", domain), ("scale_code", scale_code))
                               if scaled else PACK_ATTRS
                           ))
        return KernelRequest(
            (SemanticNode("x", "input", (), source,
                          attrs=(("input_domain", domain),) if scaled else (),
                          effect="input"),
             SemanticNode("y", "pack_fp8", ("x",), packed, attrs=attributes,
                          index_maps=pointwise_identity_maps(2, 1))),
            ("y",), ("external",), (("x", "external"),),
            self.binding.profile().target_identity, lowering_policy="diagnostic",
        )

    def test_native_scaled_pack_binds_source_code_through_emission(self) -> None:
        from atlas_mlir_native.binding import (
            SCALED_PACK_POLICY,
            check_pack_stream,
        )
        from atlas_mlir_native.reference import evaluate_scaled_pack
        from merlin.semantic_compiler.linalg_bridge import translate_linalg_text
        from merlin.semantic_compiler.reference import TensorValue, evaluate_graph
        from merlin.semantic_compiler.search import SearchLimits
        from merlin.semantic_compiler.target_binding import (
            NativeCompilationError,
            verify_native_publication,
        )

        from tools.check_program_decode import audit_program
        from tools.check_variant_inventory import load_inventory

        baseline = None
        for code in range(122, 129):
            request = self.pack_request(scale_code=code, policy=SCALED_PACK_POLICY)
            bits = (0,) + tuple(
                ((code + exponent) << 7) | sign
                for exponent in (-1, 0, 1) for sign in (0, 0x8000)
            )
            source = tuple(bits[(row * 5 + column * 3) % len(bits)]
                           for row in range(32) for column in range(32))
            expected_codes = (0, 0x30, 0xB0, 0x38, 0xB8, 0x40, 0xC0)
            expected = tuple(expected_codes[(row * 5 + column * 3) % len(bits)]
                             for row in range(32) for column in range(32))
            checked = evaluate_graph(
                request, {"x": TensorValue(request.nodes[0].type, source)},
                primitives={("pack_fp8", SCALED_PACK_POLICY): evaluate_scaled_pack},
            )
            self.assertEqual(checked[0].elements, expected)
            with self.assertRaisesRegex(ValueError, "exact scaled-pack domain"):
                evaluate_graph(
                    request,
                    {"x": TensorValue(request.nodes[0].type, (0x8000,) + source[1:])},
                    primitives={("pack_fp8", SCALED_PACK_POLICY): evaluate_scaled_pack},
                )
            destination = self.work / f"scaled-pack-{code}"
            manifest = self.binding.compile(
                self.snapshot, request, fixed_inputs={"x": 3}, fixed_outputs=(35,),
                target_source=ROOT, destination=destination, limits=SearchLimits(),
            )
            verify_native_publication(
                destination, manifest, engine="merlin_native",
                request_digest=request.digest(), target_identity=request.target_identity,
            )
            program = json.loads((destination / "physical-program.json").read_text())
            check_pack_stream(program, input_address=0x90001800,
                              output_address=0x90011800, scale_code=code)
            self.assertEqual(audit_program(
                load_inventory(), program, (destination / "program.bin").read_bytes()
            )["word_count"], 73)
            plan = json.loads((destination / "execution_plan.json").read_text())
            self.assertEqual((plan["operation"], plan["numerical_policy"], plan["pack_scale_code"]),
                             ("bf16_to_fp8_scaled_pack", SCALED_PACK_POLICY, code))
            self.assertIn(f"scale_code = {code} : i32", (destination / "atlas-virtual.mlir").read_text())
            selected = json.loads((destination / "native-selection.json").read_text())
            self.assertEqual(selected["engine"], "merlin_native")
            if baseline is not None:
                self.assertEqual(len(baseline), len(program["instructions"]))
                changed = [index for index, (left, right) in enumerate(zip(baseline, program["instructions"]))
                           if left["word_u32"] != right["word_u32"]]
                self.assertEqual(len(changed), 1)
                self.assertEqual(program["instructions"][changed[0]]["fields"]["kind"], "seli")
            baseline = program["instructions"]
            wrong = copy.deepcopy(program)
            next(row for row in wrong["instructions"] if row["fields"].get("kind") == "seli")["fields"]["offset"] = 126 if code == 127 else 127
            with self.assertRaises(NativeCompilationError):
                check_pack_stream(wrong, input_address=0x90001800,
                                  output_address=0x90011800, scale_code=code)

        linalg = (ROOT / "test/examples/native_bf16_scaled_pack_linalg.mlir").read_text()
        translated = translate_linalg_text(
            linalg, entry="pack_tile", target_identity=self.binding.profile().target_identity,
            lowering_policy="diagnostic",
        )
        self.assertEqual(translated.request.nodes[-1].type.numerical_policy, SCALED_PACK_POLICY)
        self.assertEqual(dict(translated.request.nodes[-1].attrs)["scale_code"], 122)
        parsed_destination = self.work / "scaled-pack-parsed"
        self.binding.compile(
            self.snapshot, translated.request, fixed_inputs={"arg0": 3}, fixed_outputs=(35,),
            target_source=ROOT, destination=parsed_destination, limits=SearchLimits(),
        )
        self.assertIn("scale_code = 122 : i32", (parsed_destination / "atlas-virtual.mlir").read_text())
        for invalid in (121, 129, True):
            changed = self.pack_request(scale_code=invalid, policy=SCALED_PACK_POLICY)
            with self.assertRaises(NativeCompilationError) as caught:
                self.binding.compile(
                    self.snapshot, changed, fixed_inputs={"x": 3}, fixed_outputs=(35,),
                    target_source=ROOT, destination=self.work / f"invalid-scaled-pack-{invalid}",
                    limits=SearchLimits(),
                )
            self.assertEqual(caught.exception.status, "unsupported_semantics")

    def test_native_rounded_pack_keeps_distinct_policy_and_emits_same_opcode(self) -> None:
        from dataclasses import replace

        from atlas_mlir_native.binding import (
            ROUNDED_PACK_POLICY,
            SCALED_PACK_DOMAIN,
            check_pack_stream,
        )
        from atlas_mlir_native.reference import evaluate_rounded_scaled_pack
        from merlin.semantic_compiler.reference import TensorValue, evaluate_graph
        from merlin.semantic_compiler.search import SearchLimits
        from merlin.semantic_compiler.target_binding import (
            NativeCompilationError,
        )

        request = self.pack_request(policy=ROUNDED_PACK_POLICY)
        cases = (0x3f88, 0x3f98, 0x3ff8, 0x3c70, 0x3c78, 0x43f0, 0x8000, 0x7fc1)
        bits = tuple(cases[index % len(cases)] for index in range(1024))
        expected = (0x38, 0x3a, 0x40, 0x00, 0x08, 0x7e, 0x00, 0x00)
        checked = evaluate_graph(
            request, {"x": TensorValue(request.nodes[0].type, bits)},
            primitives={("pack_fp8", ROUNDED_PACK_POLICY): evaluate_rounded_scaled_pack},
        )
        self.assertEqual(checked[0].elements, expected * 128)

        destination = self.work / "rounded-pack"
        self.compile_verified(
            request, fixed_inputs={"x": 3},
            fixed_outputs=(35,),
            destination=destination, limits=SearchLimits(),
        )
        program = json.loads((destination / "physical-program.json").read_text())
        check_pack_stream(program, input_address=0x90001800,
                          output_address=0x90011800, scale_code=127)
        plan = json.loads((destination / "execution_plan.json").read_text())
        self.assertEqual(plan["numerical_policy"], ROUNDED_PACK_POLICY)
        selected = json.loads((destination / "native-selection.json").read_text())
        self.assertIn("atlas.vpu_pack_rounded_fp8_tile", json.dumps(selected))

        wrong_pack = replace(request.nodes[1], attrs=(
            ("input_domain", SCALED_PACK_DOMAIN), ("scale_code", 127),
        ))
        wrong_request = replace(request, nodes=(request.nodes[0], wrong_pack))
        with self.assertRaises(NativeCompilationError) as caught:
            self.binding.compile(
                self.snapshot, wrong_request, fixed_inputs={"x": 3}, fixed_outputs=(35,),
                target_source=ROOT, destination=self.work / "wrong-rounded-domain",
                limits=SearchLimits(),
            )
        self.assertEqual(caught.exception.status, "unsupported_semantics")

    def test_native_unit_pack_selects_and_checks_scalar_relayout(self) -> None:
        from atlas_mlir_native.binding import check_pack_stream
        from atlas_mlir_native.reference import evaluate_unit_pack
        from merlin.semantic_compiler.model import KernelRequest
        from merlin.semantic_compiler.reference import TensorValue, evaluate_graph
        from merlin.semantic_compiler.search import SearchLimits
        from merlin.semantic_compiler.target_binding import (
            NativeCompilationError,
        )

        request = self.pack_request()
        fixture = KernelRequest.from_record(json.loads(
            (ROOT / "test/examples/native_bf16_to_fp8_pack_request.json").read_text()
        ))
        self.assertEqual(fixture.digest(), request.digest())
        codes = (0x0000, 0x3F00, 0x3F80, 0x4000, 0xBF00, 0xBF80, 0xC000)
        fp8_codes = (0x00, 0x30, 0x38, 0x40, 0xB0, 0xB8, 0xC0)
        bits = tuple(codes[(row * 5 + col * 3) % len(codes)]
                     for row in range(32) for col in range(32))
        expected = tuple(fp8_codes[(row * 5 + col * 3) % len(codes)]
                         for row in range(32) for col in range(32))

        self.assertEqual(evaluate_graph(
            request, {"x": TensorValue(request.nodes[0].type, bits)},
            primitives={("pack_fp8", request.nodes[1].type.numerical_policy): evaluate_unit_pack},
        )[0].elements, expected)
        for changed_bits in (0x8000, 0x4040, 0x7F80):
            changed = (changed_bits,) + bits[1:]
            with self.subTest(bits=changed_bits), self.assertRaisesRegex(ValueError, "seven-code"):
                evaluate_graph(
                    request, {"x": TensorValue(request.nodes[0].type, changed)},
                    primitives={("pack_fp8", request.nodes[1].type.numerical_policy): evaluate_unit_pack},
                )
        destination = self.work / "native-pack"
        manifest = self.compile_verified(
            request, fixed_inputs={"x": 3},
            fixed_outputs=(35,),
            destination=destination, limits=SearchLimits(),
        )
        program = json.loads((destination / "physical-program.json").read_text())
        self.assertEqual(program["word_count"], 73)
        self.assertEqual(sum(row["operation"] == "atlas.vpu_pack" for row in program["instructions"]), 1)
        self.assertEqual(manifest["selected_registers"], [32, 33, 0])
        self.assertEqual(json.loads((destination / "execution_plan.json").read_text())[
            "output_physical_layouts"], [None])
        check_pack_stream(program, input_address=0x90001800, output_address=0x90011800)
        for operation, field, new_value in (
            ("atlas.vpu_pack", "src", 0),
            ("atlas.branch", "offset_bytes", -52),
            ("atlas.scalar_store", "offset", 4),
        ):
            changed = copy.deepcopy(program)
            next(row for row in changed["instructions"] if row["operation"] == operation)["fields"][field] = new_value
            with self.subTest(operation=operation), self.assertRaises(NativeCompilationError):
                check_pack_stream(changed, input_address=0x90001800,
                                  output_address=0x90011800)
        for operation, bit in (("atlas.vpu_pack", 1 << 24),
                               ("atlas.branch", 1 << 8)):
            changed = copy.deepcopy(program)
            next(row for row in changed["instructions"] if row["operation"] == operation)["word_u32"] ^= bit
            with self.subTest(encoded_operation=operation), self.assertRaises(NativeCompilationError):
                check_pack_stream(changed, input_address=0x90001800,
                                  output_address=0x90011800)
        for changed in (self.pack_request(scale_code=128),
                        self.pack_request(policy="raw-bit-pattern")):
            with self.assertRaises(NativeCompilationError) as caught:
                self.binding.compile(
                    self.snapshot, changed, fixed_inputs={"x": 3}, fixed_outputs=(35,),
                    target_source=ROOT, destination=self.work / f"invalid-pack-{changed.digest()[:8]}",
                    limits=SearchLimits(),
                )
            self.assertEqual(caught.exception.status, "unsupported_semantics")

    def test_parsed_bf16_fp8_cast_reaches_native_pack(self) -> None:
        from atlas_mlir_native.binding import PACK_POLICY
        from atlas_mlir_native.reference import evaluate_unit_pack
        from merlin.semantic_compiler.linalg_bridge import translate_linalg_text
        from merlin.semantic_compiler.reference import TensorValue, evaluate_graph
        from merlin.semantic_compiler.search import SearchLimits
        from merlin.semantic_compiler.target_binding import (
            NativeCompilationError,
        )

        source = (ROOT / "test/examples/native_bf16_unit_pack_linalg.mlir").read_text()
        translated = translate_linalg_text(
            source, entry="pack_tile", target_identity=self.binding.profile().target_identity,
            lowering_policy="diagnostic",
        )
        self.assertEqual(translated.source_operations,
                         ("tensor.empty", "linalg.generic", "func.return"))
        self.assertEqual((translated.request.nodes[-1].op,
                          translated.request.nodes[-1].type.numerical_policy),
                         ("pack_fp8", PACK_POLICY))
        source_codes = (0x0000, 0x3F00, 0x3F80, 0x4000, 0xBF00, 0xBF80, 0xC000)
        expected_codes = (0x00, 0x30, 0x38, 0x40, 0xB0, 0xB8, 0xC0)
        source_bits = tuple(source_codes[(row * 5 + col * 3) % 7]
                            for row in range(32) for col in range(32))
        expected_bits = tuple(expected_codes[(row * 5 + col * 3) % 7]
                              for row in range(32) for col in range(32))
        self.assertEqual(evaluate_graph(
            translated.request,
            {"arg0": TensorValue(translated.request.nodes[0].type, source_bits)},
            primitives={("pack_fp8", PACK_POLICY): evaluate_unit_pack},
        )[0].elements, expected_bits)
        destination = self.work / "native-linalg-pack"
        self.compile_verified(
            translated.request, fixed_inputs={"arg0": 3},
            fixed_outputs=(35,),
            destination=destination, limits=SearchLimits(),
        )
        plan = json.loads((destination / "execution_plan.json").read_text())
        self.assertEqual((plan["operation"], plan["numerical_policy"], plan["input_bytes"]),
                         ("bf16_to_fp8_unit_pack", PACK_POLICY, 2048))
        self.assertEqual(json.loads((destination / "physical-program.json").read_text())["word_count"], 73)
        nonunit_source = source.replace("merlin.scale_code = 127", "merlin.scale_code = 128").replace(
            "%converted = arith.truncf %value : bf16 to f8E4M3FN",
            "%inverse_scale = arith.constant 5.000000e-01 : bf16\n"
            "        %scaled = arith.mulf %value, %inverse_scale : bf16\n"
            "        %converted = arith.truncf %scaled : bf16 to f8E4M3FN",
        )
        changed = translate_linalg_text(
            nonunit_source,
            entry="pack_tile", target_identity=self.binding.profile().target_identity,
            lowering_policy="diagnostic",
        )
        with self.assertRaises(NativeCompilationError) as caught:
            self.binding.compile(
                self.snapshot, changed.request, fixed_inputs={"arg0": 3},
                fixed_outputs=(35,), target_source=ROOT,
                destination=self.work / "unsupported-linalg-scale",
                limits=SearchLimits(),
            )
        self.assertEqual(caught.exception.status, "unsupported_semantics")

    def test_parsed_relu_then_unit_pack_checks_domain_and_selected_stream(self) -> None:
        from atlas_mlir_native.binding import PACK_POLICY, check_pack_stream
        from atlas_mlir_native.reference import evaluate_unit_pack
        from merlin.semantic_compiler.linalg_bridge import translate_linalg_text
        from merlin.semantic_compiler.reference import TensorValue, evaluate_graph
        from merlin.semantic_compiler.search import SearchLimits
        from merlin.semantic_compiler.target_binding import NativeCompilationError

        from tools.check_program_decode import audit_program
        from tools.check_variant_inventory import load_inventory

        source = (ROOT / "test/examples/native_bf16_relu_unit_pack_linalg.mlir").read_text()
        translated = translate_linalg_text(
            source, entry="relu_pack_tile", target_identity=self.binding.profile().target_identity,
            lowering_policy="diagnostic",
        )
        request = translated.request
        self.assertEqual(tuple(node.op for node in request.nodes), ("input", "relu", "pack_fp8"))
        codes = (0x0000, 0x3F00, 0x3F80, 0x4000, 0xBF00, 0xBF80, 0xC000)
        bits = tuple(codes[(row * 5 + col * 3) % len(codes)]
                     for row in range(32) for col in range(32))
        expected = {0x0000: 0, 0x3F00: 0x30, 0x3F80: 0x38, 0x4000: 0x40}
        self.assertEqual(evaluate_graph(
            request, {"arg0": TensorValue(request.nodes[0].type, bits)},
            primitives={("pack_fp8", PACK_POLICY): evaluate_unit_pack},
        )[0].elements, tuple(expected.get(value, 0) for value in bits))
        destination = self.work / "native-linalg-relu-pack"
        manifest = self.binding.compile(
            self.snapshot, request, fixed_inputs={"arg0": 3}, fixed_outputs=(35,),
            target_source=ROOT, destination=destination, limits=SearchLimits(),
        )
        self.assertEqual(manifest["engine"], "merlin_native")
        self.assertEqual(manifest["selected_registers"], [32, 33, 34, 35, 0])
        program = json.loads((destination / "physical-program.json").read_text())
        check_pack_stream(program, input_address=0x90001800,
                          output_address=0x90011800, source_pair=34, relu_pair=(32, 34))
        binary = (destination / "program.bin").read_bytes()
        inventory = load_inventory()
        audit = audit_program(inventory, program, binary)
        self.assertEqual(audit["word_count"], 75)
        self.assertEqual((audit["modes"]["VRELU"], audit["modes"]["VFP8PACK"]), (1, 1))
        pack_index = next(index for index, row in enumerate(program["instructions"])
                          if row["operation"] == "atlas.vpu_pack")
        wrong_mode = copy.deepcopy(program)
        changed_word = wrong_mode["instructions"][pack_index]["word_u32"] ^ (1 << 25)
        wrong_mode["instructions"][pack_index]["word_u32"] = changed_word
        wrong_mode["instructions"][pack_index]["word_hex"] = f"{changed_word:08x}"
        wrong_binary = bytearray(binary)
        struct.pack_into("<I", wrong_binary, pack_index * 4, changed_word)
        with self.assertRaisesRegex(ValueError, "binary and structured stream differ"):
            audit_program(inventory, program, bytes(wrong_binary))
        with self.assertRaisesRegex(ValueError, "VFP8UNPACK"):
            audit_program(inventory, wrong_mode, bytes(wrong_binary))
        wrong_fields = copy.deepcopy(program)
        wrong_fields["instructions"][pack_index]["fields"]["direction"] = "fp8_to_bf16"
        with self.assertRaisesRegex(ValueError, "fields disagree"):
            audit_program(inventory, wrong_fields, binary)
        wrong_operand = copy.deepcopy(program)
        wrong_operand["instructions"][pack_index]["fields"]["src"] = 36
        with self.assertRaisesRegex(ValueError, "src field differs from binary operand"):
            audit_program(inventory, wrong_operand, binary)
        with self.assertRaisesRegex(ValueError, "partial instruction"):
            audit_program(inventory, program, binary[:-1])
        wrong_relu = copy.deepcopy(program)
        next(row for row in wrong_relu["instructions"] if row["operation"] == "atlas.vpu_unary")["fields"]["dst"] = 36
        with self.assertRaises(NativeCompilationError):
            check_pack_stream(wrong_relu, input_address=0x90001800,
                              output_address=0x90011800, source_pair=34, relu_pair=(32, 34))
        wrong_pack_word = copy.deepcopy(program)
        next(row for row in wrong_pack_word["instructions"] if row["operation"] == "atlas.vpu_pack")["word_u32"] ^= 1 << 19
        with self.assertRaises(NativeCompilationError):
            check_pack_stream(wrong_pack_word, input_address=0x90001800,
                              output_address=0x90011800, source_pair=34, relu_pair=(32, 34))
        plan = json.loads((destination / "execution_plan.json").read_text())
        self.assertEqual(plan["operation"], "bf16_relu_pack")
        self.assertEqual(plan["producer_domain_obligations"], [{
            "producer": request.nodes[1].id,
            "consumer": request.nodes[2].id,
            "required_domain": dict(request.nodes[2].attrs)["input_domain"],
            "status": "caller_premise_unchecked",
        }])
        altered = source.replace('merlin.input_domain = "bf16-seven-powers-and-zero-v1"',
                                 'merlin.input_domain = "unqualified-domain"', 1)
        assert altered != source
        changed = translate_linalg_text(
            altered, entry="relu_pack_tile", target_identity=self.binding.profile().target_identity,
            lowering_policy="diagnostic",
        )
        with self.assertRaises(NativeCompilationError) as caught:
            self.binding.compile(
                self.snapshot, changed.request, fixed_inputs={"arg0": 3}, fixed_outputs=(35,),
                target_source=ROOT, destination=self.work / "unqualified-relu-pack",
                limits=SearchLimits(),
            )
        self.assertEqual(caught.exception.status, "unsupported_semantics")

    def test_native_bf16_relu_uses_vpu_and_checks_pair_timing(self) -> None:
        from dataclasses import replace

        from atlas_mlir_native.binding import check_bf16_tile_stream
        from merlin.semantic_compiler.model import KernelRequest
        from merlin.semantic_compiler.reference import TensorValue, evaluate_graph
        from merlin.semantic_compiler.search import SearchLimits
        from merlin.semantic_compiler.target_binding import (
            NativeCompilationError,
        )

        request = self.relu_request()
        fixture = KernelRequest.from_record(json.loads(
            (ROOT / "test/examples/native_bf16_relu_request.json").read_text()
        ))
        self.assertEqual(fixture.digest(), request.digest())
        source = tuple((0x8000, 0, 0xBF80, 0x3F80, 0xC000, 0x4000)[i % 6]
                       for i in range(1024))
        expected = tuple(0 if bits & 0x8000 else bits for bits in source)
        self.assertEqual(evaluate_graph(request, {"x": TensorValue(request.nodes[0].type, source)})[0].elements,
                         expected)
        destination = self.work / "native-relu"
        self.compile_verified(
            request, fixed_inputs={"x": 3},
            fixed_outputs=(35,),
            destination=destination, limits=SearchLimits(),
        )
        plan = json.loads((destination / "execution_plan.json").read_text())
        self.assertEqual((plan["operation"], plan["input_domain"]),
                         ("bf16_relu", "finite-normal-or-zero-bf16"))
        program = json.loads((destination / "physical-program.json").read_text())
        self.assertEqual(sum(row["operation"] == "atlas.vpu_unary" for row in program["instructions"]), 1)
        selected = self.snapshot.select(
            request, fixed_inputs={"x": 3}, fixed_outputs=(35,), limits=SearchLimits(),
        )
        self.assertEqual(selected.status, "selected")
        assert selected.graph and selected.allocation and selected.rules
        names = [selected.rules.symbols[value.symbol].get("descriptor", {}).get("name")
                 for value in selected.graph.values]
        kwargs = {
            "graph": selected.graph, "order": selected.allocation.order,
            "names": names, "assignment": selected.allocation.addresses,
            "source_ids": ("x",), "input_slots": (3,), "output_slots": (35,),
        }
        check_bf16_tile_stream(program, **kwargs)
        changed = copy.deepcopy(program)
        next(row for row in changed["instructions"] if row["operation"] == "atlas.vpu_unary")["fields"]["dst"] += 2
        with self.assertRaises(NativeCompilationError):
            check_bf16_tile_stream(changed, **kwargs)
        invalid = replace(request, nodes=(request.nodes[0], replace(request.nodes[1], attrs=())))
        refused = self.work / "relu-without-domain"
        with self.assertRaises(NativeCompilationError) as caught:
            self.binding.compile(
                self.snapshot, invalid, fixed_inputs={"x": 3}, fixed_outputs=(35,),
                target_source=ROOT, destination=refused, limits=SearchLimits(),
            )
        self.assertEqual(caught.exception.status, "unsupported_semantics")
        self.assertFalse(refused.exists())
        changed = copy.deepcopy(program)
        for row in changed["instructions"]:
            if row["operation"] == "atlas.delay":
                row["fields"]["cycles"] = 0
        with self.assertRaises(NativeCompilationError):
            check_bf16_tile_stream(changed, **kwargs)

    def test_native_shared_bf16_input_keeps_raw_and_relu_outputs_distinct(self) -> None:
        from dataclasses import replace

        from merlin.semantic_compiler.model import SemanticNode
        from merlin.semantic_compiler.reference import TensorValue, evaluate_graph
        from merlin.semantic_compiler.search import SearchLimits

        base = self.relu_request()
        tensor = base.nodes[0].type
        request = replace(
            base,
            nodes=(base.nodes[0], SemanticNode("raw", "identity", ("x",), tensor), base.nodes[1]),
            outputs=("raw", "y"), output_storages=("external", "external"),
        )
        source = tuple((0xBF80, 0x3F80, 0x8000, 0x0000)[i % 4] for i in range(1024))
        original, clamped = evaluate_graph(request, {"x": TensorValue(tensor, source)})
        self.assertEqual(original.elements, source)
        self.assertEqual(clamped.elements, tuple(0 if bits & 0x8000 else bits for bits in source))

        destination = self.work / "native-shared-relu"
        manifest = self.compile_verified(
            request, fixed_inputs={"x": 3},
            fixed_outputs=(35, 36),
            destination=destination, limits=SearchLimits(),
        )
        plan = json.loads((destination / "execution_plan.json").read_text())
        self.assertEqual(plan["output_numerical_policies"],
                         ["raw-bit-pattern", "bf16-sign-clamp-v1"])
        program = json.loads((destination / "physical-program.json").read_text())
        operations = [row["operation"] for row in program["instructions"]]
        self.assertEqual(operations.count("atlas.vload"), 2)
        self.assertEqual(operations.count("atlas.vpu_unary"), 1)
        self.assertEqual(operations.count("atlas.vstore"), 4)
        self.assertNotEqual(manifest["selected_registers"], [0, 1])

    def test_parsed_bf16_relu_linalg_reaches_native_vpu(self) -> None:
        from merlin.semantic_compiler.linalg_bridge import translate_linalg_text
        from merlin.semantic_compiler.search import SearchLimits

        source = (ROOT / "test/examples/native_bf16_relu_linalg.mlir").read_text()
        translated = translate_linalg_text(
            source, entry="relu_tile", target_identity=self.binding.profile().target_identity,
            lowering_policy="diagnostic",
        )
        self.assertEqual(translated.source_operations,
                         ("tensor.empty", "linalg.generic", "func.return"))
        self.assertEqual(translated.request.nodes[-1].op, "relu")
        destination = self.work / "linalg-bf16-relu"
        self.compile_verified(
            translated.request, fixed_inputs={"arg0": 3},
            fixed_outputs=(35,),
            destination=destination, limits=SearchLimits(),
        )
        plan = json.loads((destination / "execution_plan.json").read_text())
        self.assertEqual(plan["boundary_domain_obligations"], [{
            "input": translated.request.nodes[0].id,
            "required_domain": dict(translated.request.nodes[-1].attrs)["input_domain"],
            "status": "caller_premise_unchecked",
        }])
        self.assertEqual(plan["producer_domain_obligations"], [])
        program = json.loads((destination / "physical-program.json").read_text())
        self.assertEqual(program["word_count"], 36)
        self.assertEqual(sum(row["operation"] == "atlas.vpu_unary" for row in program["instructions"]), 1)

    def test_parsed_raw_and_relu_preserves_both_ordered_outputs(self) -> None:
        from merlin.semantic_compiler.linalg_bridge import translate_linalg_text
        from merlin.semantic_compiler.reference import TensorValue, evaluate_graph
        from merlin.semantic_compiler.search import SearchLimits

        source = (ROOT / "test/examples/native_bf16_raw_relu_linalg.mlir").read_text()
        translated = translate_linalg_text(
            source, entry="raw_and_relu", target_identity=self.binding.profile().target_identity,
            lowering_policy="diagnostic",
        )
        self.assertEqual(translated.request.outputs, ("op1r0", "op3r0"))
        tensor = translated.request.nodes[0].type
        bits = tuple((0xBF80, 0x3F80, 0x8000, 0x0000)[i % 4] for i in range(1024))
        raw, relu = evaluate_graph(translated.request, {"arg0": TensorValue(tensor, bits)})
        self.assertEqual(raw.elements, bits)
        self.assertEqual(relu.elements, tuple(0 if value & 0x8000 else value for value in bits))

        destination = self.work / "linalg-raw-and-relu"
        self.compile_verified(
            translated.request, fixed_inputs={"arg0": 3},
            fixed_outputs=(35, 36),
            destination=destination, limits=SearchLimits(),
        )
        plan = json.loads((destination / "execution_plan.json").read_text())
        self.assertEqual(plan["output_numerical_policies"],
                         ["raw-bit-pattern", "bf16-sign-clamp-v1"])
        program = json.loads((destination / "physical-program.json").read_text())
        self.assertEqual(program["word_count"], 50)
        self.assertEqual(sum(row["operation"] == "atlas.vpu_unary" for row in program["instructions"]), 1)

    def mxu0_request(self, *, policy: str | None = None, maps=None):
        from atlas_mlir_native.binding import MXU0_INPUT_DOMAIN, MXU0_POLICY, _mxu0_maps
        from merlin.semantic_compiler.model import (
            KernelRequest,
            SemanticNode,
            TensorType,
        )

        raw = TensorType((32, 32), "fp8_e4m3", "raw-bit-pattern")
        result = TensorType((32, 32), "bf16", policy or MXU0_POLICY)
        return KernelRequest(
            (SemanticNode("x", "input", (), raw,
                          attrs=(("input_domain", MXU0_INPUT_DOMAIN),), effect="input"),
             SemanticNode("w", "input", (), raw,
                          attrs=(("input_domain", MXU0_INPUT_DOMAIN),), effect="input"),
             SemanticNode("y", "matmul", ("x", "w"), result,
                          index_maps=_mxu0_maps() if maps is None else maps)),
            ("y",), ("external",), (("x", "external"), ("w", "external")),
            self.binding.profile().target_identity, lowering_policy="diagnostic",
        )

    def mxu0_k_request(self, tile_count: int):
        from atlas_mlir_native.binding import (
            MXU0_INPUT_DOMAIN,
            MXU0_POLICY,
            _mxu0_accumulate_maps,
            _mxu0_maps,
        )
        from merlin.semantic_compiler.model import (
            KernelRequest,
            SemanticNode,
            TensorType,
        )

        raw = TensorType((32, 32), "fp8_e4m3", "raw-bit-pattern")
        result = TensorType((32, 32), "bf16", MXU0_POLICY)
        sources = tuple(
            SemanticNode(f"{kind}{index}", "input", (), raw,
                         attrs=(("input_domain", MXU0_INPUT_DOMAIN),), effect="input")
            for index in range(tile_count) for kind in ("a", "w")
        )
        reductions = (
            SemanticNode("s0", "matmul", ("a0", "w0"), result, index_maps=_mxu0_maps()),
            *(SemanticNode(
                f"s{index}", "matmul_accumulate",
                (f"a{index}", f"w{index}", f"s{index - 1}"), result,
                index_maps=_mxu0_accumulate_maps(),
            ) for index in range(1, tile_count)),
        )
        return KernelRequest(
            (*sources, *reductions), (reductions[-1].id,), ("external",),
            tuple((source.id, "external") for source in sources),
            self.binding.profile().target_identity, lowering_policy="diagnostic",
        )

    def test_native_mxu0_k2_keeps_one_accumulator_until_readout(self) -> None:
        from atlas_mlir_native.binding import check_mxu0_k_chain_stream
        from merlin.semantic_compiler.search import SearchLimits

        request = self.mxu0_k_request(2)
        destination = self.work / "mxu0-k2"
        manifest = self.compile_verified(
            request, fixed_inputs={"a0": 3, "w0": 4, "a1": 5, "w1": 6},
            fixed_outputs=(35,),
            destination=destination, limits=SearchLimits(),
        )
        virtual = (destination / "atlas-virtual.mlir").read_text()
        self.assertEqual(virtual.count('"atlas.virtual_mxu_reset"'), 1)
        self.assertEqual(virtual.count('"atlas.virtual_mxu_accumulate"'), 1)
        self.assertEqual(virtual.count('"atlas.virtual_mxu_readout_bf16"'), 1)
        program = json.loads((destination / "physical-program.json").read_text())
        self.assertEqual(
            [row["fields"]["accumulate"] for row in program["instructions"]
            if row["operation"] == "atlas.mxu_matmul"], [False, True],
        )
        plan = json.loads((destination / "execution_plan.json").read_text())
        self.assertEqual(plan["operation"], "mxu0_k2")
        self.assertEqual(manifest["selected_registers"], [0, 1, 2, 3])
        check_mxu0_k_chain_stream(
            program, input_addresses=(0x90001800, 0x90002000, 0x90002800, 0x90003000),
            output_address=0x90011800, source_registers=(2, 3, 0, 1),
        )
        from merlin.semantic_compiler.target_binding import NativeCompilationError

        arguments = {
            "input_addresses": (0x90001800, 0x90002000, 0x90002800, 0x90003000),
            "output_address": 0x90011800,
            "source_registers": (2, 3, 0, 1),
        }
        for fault in ("reset_later_k", "early_continuation", "changed_weight"):
            changed = copy.deepcopy(program)
            rows = changed["instructions"]
            if fault == "reset_later_k":
                [row for row in rows if row["operation"] == "atlas.mxu_matmul"][1]["fields"]["accumulate"] = False
            elif fault == "early_continuation":
                second = [index for index, row in enumerate(rows)
                          if row["operation"] == "atlas.mxu_matmul"][1]
                delay = max((row for row in rows[:second] if row["operation"] == "atlas.delay"),
                            key=lambda row: row["fields"]["cycles"])
                delay["fields"]["cycles"] = 0
            else:
                [row for row in rows if row["operation"] == "atlas.mxu_push"][1]["fields"]["src"] = 3
            with self.subTest(fault=fault), self.assertRaises(NativeCompilationError):
                check_mxu0_k_chain_stream(changed, **arguments)

    def test_parsed_ordered_k2_linalg_reaches_native_accumulator_chain(self) -> None:
        from merlin.semantic_compiler.linalg_bridge import translate_linalg_text
        from merlin.semantic_compiler.search import SearchLimits

        from tools.check_program_decode import audit_program
        from tools.check_variant_inventory import load_inventory

        request = translate_linalg_text(
            (ROOT / "test/examples/native_mxu0_k2_linalg.mlir").read_text(),
            entry="mxu0_k2_tile", target_identity=self.binding.profile().target_identity,
            lowering_policy="diagnostic",
        ).request
        self.assertEqual(tuple(node.op for node in request.nodes),
                         ("input", "input", "input", "input", "matmul", "matmul_accumulate"))
        destination = self.work / "parsed-mxu0-k2"
        manifest = self.binding.compile(
            self.snapshot, request,
            fixed_inputs={"arg0": 3, "arg1": 4, "arg2": 5, "arg3": 6},
            fixed_outputs=(35,), target_source=ROOT,
            destination=destination, limits=SearchLimits(),
        )
        self.assertEqual(manifest["engine"], "merlin_native")
        plan = json.loads((destination / "execution_plan.json").read_text())
        self.assertEqual(plan["operation"], "mxu0_k2")
        self.assertEqual(plan["weight_addresses"], [0x90002000, 0x90003000])
        program = json.loads((destination / "physical-program.json").read_text())
        self.assertEqual(audit_program(
            load_inventory(), program, (destination / "program.bin").read_bytes(),
        )["status"], "PASS")

    def test_parsed_ordered_k3_linalg_retains_both_continuations(self) -> None:
        from merlin.semantic_compiler.linalg_bridge import translate_linalg_text
        from merlin.semantic_compiler.search import SearchLimits

        from tools.check_program_decode import audit_program
        from tools.check_variant_inventory import load_inventory

        request = translate_linalg_text(
            (ROOT / "test/examples/native_mxu0_k3_linalg.mlir").read_text(),
            entry="mxu0_k3_tile", target_identity=self.binding.profile().target_identity,
            lowering_policy="diagnostic",
        ).request
        self.assertEqual(tuple(node.op for node in request.nodes),
                         ("input",) * 6 + ("matmul", "matmul_accumulate", "matmul_accumulate"))
        destination = self.work / "parsed-mxu0-k3"
        self.compile_verified(
            request, fixed_inputs={f"arg{index}": index + 3 for index in range(6)},
            fixed_outputs=(35,),
            destination=destination, limits=SearchLimits(),
        )
        plan = json.loads((destination / "execution_plan.json").read_text())
        self.assertEqual(plan["operation"], "mxu0_k3")
        self.assertEqual(plan["weight_addresses"], [0x90002000, 0x90003000, 0x90004000])
        program = json.loads((destination / "physical-program.json").read_text())
        self.assertEqual(
            [row["fields"]["accumulate"] for row in program["instructions"]
             if row["operation"] == "atlas.mxu_matmul"],
            [False, True, True],
        )
        self.assertEqual(audit_program(
            load_inventory(), program, (destination / "program.bin").read_bytes(),
        )["status"], "PASS")
        self.assertEqual(
            (destination / "atlas-virtual.mlir").read_text().count('"atlas.virtual_mxu_readout_bf16"'), 1,
        )

    def test_generated_ordered_k_chain_uses_resource_bound_instead_of_shape_list(self) -> None:
        from merlin.semantic_compiler.search import SearchLimits

        from tools.check_program_decode import audit_program, decode_binary
        from tools.check_variant_inventory import load_inventory

        tile_count = 4
        request = self.mxu0_k_request(tile_count)
        destination = self.work / "generated-mxu0-k4"
        self.compile_verified(
            request, fixed_inputs={node.id: index + 3 for index, node in enumerate(request.nodes[:2 * tile_count])},
            fixed_outputs=(35,),
            destination=destination, limits=SearchLimits(),
        )
        binary = (destination / "program.bin").read_bytes()
        program = json.loads((destination / "physical-program.json").read_text())
        inventory = load_inventory()
        self.assertEqual(audit_program(inventory, program, binary)["status"], "PASS")
        decoded = decode_binary(inventory, binary)
        self.assertEqual(
            [row["fields"]["accumulate"] for row in decoded["instructions"]
             if row["operation"] == "atlas.mxu_matmul"],
            [False, True, True, True],
        )
        self.assertEqual(program["word_count"], 90)
        self.assertEqual(json.loads((destination / "execution_plan.json").read_text())["operation"], "mxu0_k4")

    def test_parsed_m64_n96_k64_linalg_compiles_six_native_tiles(self) -> None:
        from merlin.semantic_compiler.linalg_bridge import translate_linalg_text
        from merlin.semantic_compiler.ordered_k_tiling import (
            materialize_row_major_k_slices,
        )
        from merlin.semantic_compiler.ordered_mn_tiling import (
            materialize_row_major_mn_inputs,
        )
        from merlin.semantic_compiler.search import SearchLimits
        from merlin.semantic_compiler.target_binding import verify_native_publication

        from tools.check_program_decode import audit_program
        from tools.check_variant_inventory import load_inventory

        source = (ROOT / "test/examples/native_mxu0_m64_n96_k64_linalg.mlir").read_text()
        logical = translate_linalg_text(
            source, entry="mxu0_m64_n96_k64",
            target_identity=self.binding.profile().target_identity,
            lowering_policy="diagnostic",
        ).request
        outer = self.binding.prepare_ordered_mn(logical, 32, 32)
        self.assertEqual([(tile.m_start, tile.n_start) for tile in outer.tiles],
                         [(0, 0), (0, 32), (0, 64), (32, 0), (32, 32), (32, 64)])
        inputs = {"arg0": bytes([0x40]) * (64 * 64),
                  "arg1": bytes([0x38]) * (96 * 64)}
        outer_inputs = materialize_row_major_mn_inputs(outer, inputs, element_bytes=1)
        for index, (tile, tile_inputs) in enumerate(zip(outer.tiles, outer_inputs)):
            inner = self.binding.prepare_ordered_k(tile.request, 32)
            packed = materialize_row_major_k_slices(inner, tile_inputs, element_bytes=1)
            self.assertEqual(tuple(map(len, packed.values())), (1024,) * 4)
            destination = self.work / f"m64-n96-k64-tile-{index}"
            manifest = self.binding.compile(
                self.snapshot, inner.request,
                fixed_inputs={item.tile_input: slot + 3
                              for slot, item in enumerate(inner.source_slices)},
                fixed_outputs=(35,), target_source=ROOT,
                destination=destination, limits=SearchLimits(),
            )
            verify_native_publication(
                destination, manifest, engine="merlin_native",
                request_digest=inner.request.digest(),
                target_identity=inner.request.target_identity,
            )
            program = json.loads((destination / "physical-program.json").read_text())
            self.assertEqual(audit_program(
                load_inventory(), program, (destination / "program.bin").read_bytes(),
            )["status"], "PASS")
            self.assertEqual(json.loads((destination / "execution_plan.json").read_text())["operation"],
                             "mxu0_k2")

    def test_parsed_m33_n31_k65_compiles_checked_mnk_tails(self) -> None:
        from atlas_mlir_native.layout import unpack_native_bf16_output
        from merlin.semantic_compiler.linalg_bridge import translate_linalg_text
        from merlin.semantic_compiler.ordered_mnk_tiling import (
            OrderedMNKTiling,
            assemble_row_major_mnk_outputs,
            materialize_row_major_mnk_inputs,
        )
        from merlin.semantic_compiler.search import SearchLimits
        from merlin.semantic_compiler.target_binding import verify_native_publication

        from tools.check_program_decode import audit_program
        from tools.check_variant_inventory import load_inventory

        source = (ROOT / "test/examples/native_mxu0_m33_n31_k65_linalg.mlir").read_text()
        logical = translate_linalg_text(
            source, entry="mxu0_m33_n31_k65",
            target_identity=self.binding.profile().target_identity,
            lowering_policy="diagnostic",
        ).request
        outer = self.binding.prepare_ordered_mn(logical, 32, 32)
        reductions = tuple(self.binding.prepare_ordered_k(tile.request, 32)
                           for tile in outer.tiles)
        plan = OrderedMNKTiling(
            logical, outer, reductions, 32,
            logical.nodes[-1].type.numerical_policy, b"\x00",
        )
        self.assertEqual([(tile.m_start, tile.n_start) for tile in outer.tiles], [(0, 0), (32, 0)])
        self.assertEqual([reduction.tile_count for reduction in reductions], [3, 3])
        packed = materialize_row_major_mnk_inputs(
            plan,
            {"arg0": bytes([0x38]) * (33 * 65), "arg1": bytes([0x38]) * (31 * 65)},
            element_bytes=1,
        )
        self.assertEqual(len(packed), 2)
        self.assertEqual(packed[0]["arg0__k0"][:31], bytes(31))
        self.assertEqual(packed[1]["arg0__k0"][32:], bytes(31 * 32))
        self.assertEqual(packed[0]["arg1__k0"][31 * 32:], bytes(32))

        tile_outputs = []
        for index, (request, tile_inputs) in enumerate(zip(plan.requests, packed)):
            destination = self.work / f"m33-n31-k65-tile-{index}"
            manifest = self.binding.compile(
                self.snapshot, request,
                fixed_inputs={name: slot + 3 for slot, name in enumerate(tile_inputs)},
                fixed_outputs=(35,), target_source=ROOT,
                destination=destination, limits=SearchLimits(),
            )
            verify_native_publication(
                destination, manifest, engine="merlin_native",
                request_digest=request.digest(), target_identity=request.target_identity,
            )
            physical = json.loads((destination / "physical-program.json").read_text())
            self.assertEqual(audit_program(
                load_inventory(), physical, (destination / "program.bin").read_bytes(),
            )["status"], "PASS")
            execution = json.loads((destination / "execution_plan.json").read_text())
            self.assertEqual(execution["operation"], "mxu0_k3")
            self.assertEqual(execution["output_physical_layouts"], ["bf16_pair_halves"])
            tile_outputs.append(unpack_native_bf16_output(
                execution, (0x4282).to_bytes(2, "little") * (32 * 32),
            ))

        self.assertEqual(
            assemble_row_major_mnk_outputs(plan, tile_outputs, element_bytes=2),
            (0x4282).to_bytes(2, "little") * (33 * 31),
        )

    def test_parsed_m31_n33_k64_uses_checked_outer_tails(self) -> None:
        from merlin.semantic_compiler.linalg_bridge import translate_linalg_text
        from merlin.semantic_compiler.ordered_mn_tiling import (
            assemble_row_major_mn_outputs,
            materialize_row_major_mn_inputs,
        )

        source = (ROOT / "test/examples/native_mxu0_m64_n96_k64_linalg.mlir").read_text()
        source = (source.replace("m64_n96", "m31_n33")
                        .replace("64x64", "31x64").replace("96x64", "33x64")
                        .replace("64x96", "31x33"))
        logical = translate_linalg_text(
            source, entry="mxu0_m31_n33_k64",
            target_identity=self.binding.profile().target_identity,
            lowering_policy="diagnostic",
        ).request
        outer = self.binding.prepare_ordered_mn(logical, 32, 32)
        self.assertEqual(outer.record()["schema"], "merlin.ordered_mn_tiling.v2")
        self.assertEqual(outer.record()["tail_pad_element_hex"], "00")
        self.assertEqual([(tile.m_start, tile.n_start) for tile in outer.tiles],
                         [(0, 0), (0, 32)])
        self.assertEqual([(tile.input_slices[0].rows, tile.input_slices[1].rows)
                          for tile in outer.tiles], [(31, 32), (31, 1)])
        self.assertEqual(outer.tiles[0].request.digest(), outer.tiles[1].request.digest())
        packed = materialize_row_major_mn_inputs(
            outer, {"arg0": bytes([0x40]) * (31 * 64),
                    "arg1": bytes([0x38]) * (33 * 64)}, element_bytes=1,
        )
        self.assertEqual([len(row["arg0"]) for row in packed], [2048, 2048])
        self.assertEqual([len(row["arg1"]) for row in packed], [2048, 2048])
        self.assertEqual(packed[0]["arg0"][-64:], bytes(64))
        self.assertEqual(packed[1]["arg1"][64:], bytes(31 * 64))
        first = bytes([1, 2]) * (32 * 32)
        second = bytes([3, 4]) * (32 * 32)
        result = assemble_row_major_mn_outputs(outer, (first, second), element_bytes=2)
        self.assertEqual(len(result), 31 * 33 * 2)
        self.assertEqual(result[:64], first[:64])
        self.assertEqual(result[64:66], second[:2])

    def _check_mn_example_route(self, files: dict[str, bytes], *, tails: bool) -> None:
        with tempfile.TemporaryDirectory(prefix="mn-example-route-", dir=self.work) as directory:
            output = Path(directory)
            for name, payload in files.items():
                (output / name).write_bytes(payload)
            obj = output / "host.o"
            subprocess.run([
                "clang", "--target=riscv64-unknown-elf", "-march=rv64gc",
                "-mabi=lp64d", "-std=c11", "-O1", "-ffreestanding",
                "-fno-builtin", "-Wall", "-Wextra", "-Werror",
                "-DATLAS_DIAGNOSTIC_ORDERED_K", "-I", str(output), "-c",
                str(ROOT / "examples/handoff/ordered_mn/host.c"),
                "-o", str(obj),
            ], check=True, capture_output=True, text=True)
            symbols = {
                line.split()[-1]
                for line in subprocess.run(
                    ["nm", "-u", str(obj)], check=True, capture_output=True, text=True,
                ).stdout.splitlines() if line.split()
            }
            selected = "atlas_ordered_mn_run_tails" if tails else "atlas_ordered_mn_run"
            other = "atlas_ordered_mn_run" if tails else "atlas_ordered_mn_run_tails"
            self.assertIn(selected, symbols)
            self.assertNotIn(other, symbols)

    def test_mn_k_cli_rechecks_selected_source_tile(self) -> None:
        from merlin.targetgen.cli import main as targetgen_main

        from tools.atlas_ordered_mn_host_data import prepare_mn

        source = ROOT / "test/examples/native_mxu0_m64_n96_k64_linalg.mlir"
        abi = self.work / "m64-n96-k64-abi.json"
        abi.write_text(json.dumps({
            "fixed_inputs": {"arg0__k0": 3, "arg1__k0": 4,
                             "arg0__k1": 5, "arg1__k1": 6},
            "fixed_outputs": [35],
        }))
        common = [
            "--engine", "merlin_native", "--support", "atlas",
            "--snapshot", str(self.snapshot.root),
            "--linalg", str(source), "--linalg-entry", "mxu0_m64_n96_k64",
            "--ordered-m-tile", "32", "--ordered-n-tile", "32",
            "--outer-tile-index", "4", "--ordered-k-tile", "32",
            "--mode", "diagnostic",
        ]
        output = self.work / "m64-n96-k64-cli-tile-4"
        status = self.work / "m64-n96-k64-cli-status.json"
        self.assertEqual(targetgen_main([
            "native-compile", *common, "--abi", str(abi),
            "--target-source", str(ROOT), "--out", str(output),
            "--status-file", str(status),
        ]), 0)
        manifest = json.loads((output / "manifest.json").read_text())
        outer = json.loads((output / "ordered-mn-tiling.json").read_text())
        inner = json.loads((output / "ordered-k-tiling.json").read_text())
        self.assertEqual(outer["selected_tile_index"], 4)
        self.assertEqual(len(outer["tiles"]), 6)
        self.assertEqual(manifest["outer_source_request_digest"], outer["source_request_digest"])
        self.assertEqual(manifest["source_request_digest"], inner["source_request_digest"])
        self.assertEqual(inner["source_request_digest"], outer["selected_request_digest"])
        checked = self.work / "m64-n96-k64-cli-verified.json"
        self.assertEqual(targetgen_main([
            "native-verify", *common, "--artifact", str(output), "--out", str(checked),
        ]), 0)
        self.assertEqual(json.loads(checked.read_text())["status"], "verified_artifacts")
        files, host_record = prepare_mn(
            output, checked, ROOT / "runtime/selected-host-map.json",
            Path(MERLIN) / "merlin/runtime/c/row_major_k_window.h",
            Path(MERLIN) / "merlin/runtime/c/finite_float_boundary.h",
        )
        self.assertEqual((host_record["m_rows"], host_record["n_rows"],
                          host_record["k_columns"], host_record["tile_count"]),
                         (64, 96, 64, 6))
        self.assertIn(b"ATLAS_ORDERED_MN_ROWS 64u", files["atlas_ordered_mn_data.h"])
        self.assertEqual(host_record["files"]["atlas_ordered_mn.c"],
                         hashlib.sha256(files["atlas_ordered_mn.c"]).hexdigest())
        self._check_mn_example_route(files, tails=False)
        outer_path = output / "ordered-mn-tiling.json"
        outer_path.write_text(outer_path.read_text().replace('"selected_tile_index": 4',
                                                       '"selected_tile_index": 3'))
        with self.assertRaisesRegex(ValueError, "outer matrix partition"):
            prepare_mn(
                output, checked, ROOT / "runtime/selected-host-map.json",
                Path(MERLIN) / "merlin/runtime/c/row_major_k_window.h",
                Path(MERLIN) / "merlin/runtime/c/finite_float_boundary.h",
            )
        rejected = self.work / "m64-n96-k64-cli-rejected.json"
        self.assertEqual(targetgen_main([
            "native-verify", *common, "--artifact", str(output), "--out", str(rejected),
        ]), 2)
        self.assertEqual(json.loads(rejected.read_text())["status"], "compile_error")

    def test_mn_k_cli_exports_partial_output_tiles(self) -> None:
        from merlin.targetgen.cli import main as targetgen_main

        from tools.atlas_ordered_mn_host_data import prepare_mn

        source = (ROOT / "test/examples/native_mxu0_m64_n96_k64_linalg.mlir").read_text()
        source = (source.replace("m64_n96", "m31_n33")
                        .replace("64x64", "31x64").replace("96x64", "33x64")
                        .replace("64x96", "31x33"))
        source_path = self.work / "m31-n33-k64.mlir"
        source_path.write_text(source)
        abi = self.work / "m31-n33-k64-abi.json"
        abi.write_text(json.dumps({
            "fixed_inputs": {"arg0__k0": 3, "arg1__k0": 4,
                             "arg0__k1": 5, "arg1__k1": 6},
            "fixed_outputs": [35],
        }))
        common = [
            "--engine", "merlin_native", "--support", "atlas",
            "--snapshot", str(self.snapshot.root),
            "--linalg", str(source_path), "--linalg-entry", "mxu0_m31_n33_k64",
            "--ordered-m-tile", "32", "--ordered-n-tile", "32",
            "--outer-tile-index", "1", "--ordered-k-tile", "32",
            "--mode", "diagnostic",
        ]
        output = self.work / "m31-n33-k64-compiled"
        self.assertEqual(targetgen_main([
            "native-compile", *common, "--abi", str(abi),
            "--target-source", str(ROOT), "--out", str(output),
        ]), 0)
        checked = self.work / "m31-n33-k64-verified.json"
        self.assertEqual(targetgen_main([
            "native-verify", *common, "--artifact", str(output), "--out", str(checked),
        ]), 0)
        files, host_record = prepare_mn(
            output, checked, ROOT / "runtime/selected-host-map.json",
            Path(MERLIN) / "merlin/runtime/c/row_major_k_window.h",
            Path(MERLIN) / "merlin/runtime/c/finite_float_boundary.h",
        )
        outer = json.loads((output / "ordered-mn-tiling.json").read_text())
        self.assertEqual(outer["schema"], "merlin.ordered_mn_tiling.v2")
        self.assertEqual(outer["tiles"][1]["valid_n"], 1)
        self.assertEqual(host_record["schema"], "atlas.ordered_mn_host_data.v2")
        self.assertEqual((host_record["m_rows"], host_record["n_rows"],
                          host_record["tile_count"]), (31, 33, 2))
        self.assertIn(b"ATLAS_ORDERED_MN_HAS_TAILS 1u", files["atlas_ordered_mn_data.h"])
        self._check_mn_example_route(files, tails=True)
        outer_path = output / "ordered-mn-tiling.json"
        outer_path.write_text(outer_path.read_text().replace(
            '"tail_pad_element_hex": "00"', '"tail_pad_element_hex": "80"',
        ))
        with self.assertRaisesRegex(ValueError, "outer matrix partition"):
            prepare_mn(
                output, checked, ROOT / "runtime/selected-host-map.json",
                Path(MERLIN) / "merlin/runtime/c/row_major_k_window.h",
                Path(MERLIN) / "merlin/runtime/c/finite_float_boundary.h",
            )

    def test_mn_cli_exports_single_k_window_with_output_tails(self) -> None:
        from merlin.targetgen.cli import main as targetgen_main

        from tools.atlas_ordered_mn_host_data import prepare_mn

        source = (ROOT / "test/examples/native_mxu0_m64_n96_k64_linalg.mlir").read_text()
        source = (source.replace("m64_n96_k64", "m31_n33_k32")
                        .replace("64x64", "31x32").replace("96x64", "33x32")
                        .replace("64x96", "31x33"))
        source_path = self.work / "m31-n33-k32.mlir"
        source_path.write_text(source)
        abi = self.work / "m31-n33-k32-abi.json"
        abi.write_text(json.dumps({"fixed_inputs": {"arg0": 3, "arg1": 4},
                                   "fixed_outputs": [35]}))
        common = [
            "--engine", "merlin_native", "--support", "atlas",
            "--snapshot", str(self.snapshot.root),
            "--linalg", str(source_path), "--linalg-entry", "mxu0_m31_n33_k32",
            "--ordered-m-tile", "32", "--ordered-n-tile", "32",
            "--outer-tile-index", "1", "--mode", "diagnostic",
        ]
        output = self.work / "m31-n33-k32-compiled"
        self.assertEqual(targetgen_main([
            "native-compile", *common, "--abi", str(abi),
            "--target-source", str(ROOT), "--out", str(output),
        ]), 0)
        self.assertFalse((output / "ordered-k-tiling.json").exists())
        checked = self.work / "m31-n33-k32-verified.json"
        self.assertEqual(targetgen_main([
            "native-verify", *common, "--artifact", str(output), "--out", str(checked),
        ]), 0)
        files, host_record = prepare_mn(
            output, checked, ROOT / "runtime/selected-host-map.json",
            Path(MERLIN) / "merlin/runtime/c/row_major_k_window.h",
            Path(MERLIN) / "merlin/runtime/c/finite_float_boundary.h",
        )
        self.assertEqual(host_record["schema"], "atlas.ordered_mn_host_data.v2")
        self.assertEqual((host_record["m_rows"], host_record["n_rows"],
                          host_record["k_columns"], host_record["tile_count"]),
                         (31, 33, 32, 2))
        self.assertIn(b"ATLAS_ORDERED_K_WINDOW_COUNT 2u", files["atlas_ordered_k_data.h"])
        self.assertIn(b"ATLAS_ORDERED_K_PROGRAM_WORD_COUNT 38u", files["atlas_ordered_k_data.h"])

    def test_mn_cli_exports_head_padded_k_tails(self) -> None:
        from merlin.targetgen.cli import main as targetgen_main

        from tools.atlas_ordered_mn_host_data import prepare_mn

        template = (ROOT / "test/examples/native_mxu0_m64_n96_k64_linalg.mlir").read_text()
        cases = ((31, 33, 1, 2, 38), (31, 33, 31, 2, 38),
                 (31, 33, 65, 6, 72), (33, 65, 31, 2, 38),
                 (1, 7, 65, 6, 72), (7, 33, 1, 2, 38))
        for m, n, k, windows, words in cases:
            with self.subTest(m=m, n=n, k=k):
                source = (template.replace("m64_n96_k64", f"m{m}_n{n}_k{k}")
                                  .replace("64x64", f"{m}x{k}")
                                  .replace("96x64", f"{n}x{k}")
                                  .replace("64x96", f"{m}x{n}"))
                source_path = self.work / f"m{m}-n{n}-k{k}.mlir"
                source_path.write_text(source)
                abi = self.work / f"m{m}-n{n}-k{k}-abi.json"
                abi.write_text(json.dumps({
                    "fixed_inputs": {
                        f"arg{operand}__k{tile}": 3 + tile * 2 + operand
                        for tile in range(windows // 2) for operand in range(2)
                    },
                    "fixed_outputs": [35],
                }))
                common = [
                    "--engine", "merlin_native", "--support", "atlas",
                    "--snapshot", str(self.snapshot.root),
                    "--linalg", str(source_path), "--linalg-entry", f"mxu0_m{m}_n{n}_k{k}",
                    "--ordered-m-tile", "32", "--ordered-n-tile", "32",
                    "--outer-tile-index", str(((m + 31) // 32) * ((n + 31) // 32) - 1),
                    "--ordered-k-tile", "32",
                    "--mode", "diagnostic",
                ]
                output = self.work / f"m{m}-n{n}-k{k}-compiled"
                self.assertEqual(targetgen_main([
                    "native-compile", *common, "--abi", str(abi),
                    "--target-source", str(ROOT), "--out", str(output),
                ]), 0)
                checked = self.work / f"m{m}-n{n}-k{k}-verified.json"
                self.assertEqual(targetgen_main([
                    "native-verify", *common, "--artifact", str(output), "--out", str(checked),
                ]), 0)
                files, record = prepare_mn(
                    output, checked, ROOT / "runtime/selected-host-map.json",
                    Path(MERLIN) / "merlin/runtime/c/row_major_k_window.h",
                    Path(MERLIN) / "merlin/runtime/c/finite_float_boundary.h",
                )
                inner = json.loads((output / "ordered-k-tiling.json").read_text())
                self.assertEqual(inner["schema"], "merlin.ordered_k_tiling.v2")
                self.assertEqual(inner["head_pad_element_hex"], "00")
                self.assertEqual(inner["source_slices"][0]["source_count"], k % 32)
                self.assertEqual(inner["source_slices"][0]["destination_column_start"],
                                 32 - k % 32)
                self.assertEqual((record["m_rows"], record["n_rows"],
                                  record["k_columns"], record["tile_count"]),
                                 (m, n, k, ((m + 31) // 32) * ((n + 31) // 32)))
                self.assertIn(
                    f"ATLAS_ORDERED_K_WINDOW_COUNT {windows}u".encode(),
                    files["atlas_ordered_k_data.h"],
                )
                self.assertIn(
                    f"ATLAS_ORDERED_K_PROGRAM_WORD_COUNT {words}u".encode(),
                    files["atlas_ordered_k_data.h"],
                )
                path = output / "ordered-k-tiling.json"
                path.write_text(path.read_text().replace(
                    '"head_pad_element_hex": "00"', '"head_pad_element_hex": "80"',
                ))
                with self.assertRaisesRegex(ValueError, "artifact or selected host map differs"):
                    prepare_mn(
                        output, checked, ROOT / "runtime/selected-host-map.json",
                        Path(MERLIN) / "merlin/runtime/c/row_major_k_window.h",
                        Path(MERLIN) / "merlin/runtime/c/finite_float_boundary.h",
                    )

    def test_fresh_k160_native_compilation_does_not_access_act(self) -> None:
        blocker = self.work / "act-unavailable"
        blocker.mkdir()
        marker = blocker / "loaded.txt"
        (blocker / "sitecustomize.py").write_text(
            "import builtins, importlib.abc, os, subprocess, sys\n"
            "with open(os.environ['ATLAS_ACT_BLOCK_MARKER'], 'a') as marker:\n"
            "    marker.write(str(os.getpid()) + '\\n')\n"
            "_denied = {'act', 'act_backend', 'taidl', 'taidl_to'}\n"
            "_original_import = builtins.__import__\n"
            "def _import(name, *args, **kwargs):\n"
            "    if name.split('.')[0] in _denied: raise ImportError('ACT access blocked')\n"
            "    return _original_import(name, *args, **kwargs)\n"
            "builtins.__import__ = _import\n"
            "class _Finder(importlib.abc.MetaPathFinder):\n"
            "    def find_spec(self, fullname, path=None, target=None):\n"
            "        if fullname.split('.')[0] in _denied: raise ImportError('ACT access blocked')\n"
            "sys.meta_path.insert(0, _Finder())\n"
            "_original_popen = subprocess.Popen\n"
            "def _popen(args, *other, **kwargs):\n"
            "    executable = args[0] if isinstance(args, (list, tuple)) else args.split()[0]\n"
            "    if os.path.basename(executable) in {'act', 'act-backend', 'act_backend'}:\n"
            "        raise OSError('ACT executable access blocked')\n"
            "    return _original_popen(args, *other, **kwargs)\n"
            "subprocess.Popen = _popen\n"
        )
        environment = os.environ.copy()
        environment["ATLAS_ACT_BLOCK_MARKER"] = str(marker)
        environment["PYTHONNOUSERSITE"] = "1"
        environment["PYTHONPATH"] = os.pathsep.join((
            str(blocker), str(Path(MERLIN) / "src"), str(ROOT / "python"),
            environment.get("PYTHONPATH", ""),
        ))
        executable = Path(sys.executable).with_name("merlin-targetgen")
        self.assertTrue(executable.is_file())

        def invoke(*args: str) -> subprocess.CompletedProcess[str]:
            result = subprocess.run(
                [str(executable), *args], env=environment,
                text=True, capture_output=True, timeout=180, check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            return result

        source = self.work / "act-unavailable-k160.mlir"
        source.write_text((ROOT / "test/examples/native_mxu0_k64_linalg.mlir")
                          .read_text().replace("64", "160"))
        self.assertIn("@mxu0_k160_tile", source.read_text())
        abi = self.work / "act-unavailable-k160-abi.json"
        abi.write_text(json.dumps({
            "fixed_inputs": {
                f"arg{operand}__k{tile}": 3 + 2 * tile + operand
                for tile in range(5) for operand in range(2)
            },
            "fixed_outputs": [35],
        }))
        snapshot = self.work / "act-unavailable-snapshot"
        invoke(
            "native-build", "--engine", "merlin_native", "--support", "atlas",
            "--cargo-target-dir", str(self.work / "act-unavailable-cargo"),
            "--source-revision", self.binding.profile().target_identity.split(":", 1)[0]
            .removeprefix("atlas-rtl-"), "--out", str(snapshot),
        )
        common = (
            "--engine", "merlin_native", "--support", "atlas",
            "--snapshot", str(snapshot), "--linalg", str(source),
            "--linalg-entry", "mxu0_k160_tile", "--ordered-k-tile", "32",
            "--mode", "diagnostic",
        )
        output = self.work / "act-unavailable-k160-compiled"
        invoke(
            "native-compile", *common, "--abi", str(abi),
            "--target-source", str(ROOT), "--out", str(output),
        )
        checked = self.work / "act-unavailable-k160-verified.json"
        invoke("native-verify", *common, "--artifact", str(output), "--out", str(checked))
        self.assertEqual(json.loads(checked.read_text())["status"], "verified_artifacts")
        manifest = json.loads((output / "manifest.json").read_text())
        self.assertEqual(manifest["engine"], "merlin_native")
        self.assertGreater(len((output / "program.bin").read_bytes()), 90 * 4)
        self.assertGreaterEqual(len(marker.read_text().splitlines()), 3)

    def test_logical_k64_linalg_partitions_into_checked_native_chain(self) -> None:
        from merlin.semantic_compiler.linalg_bridge import translate_linalg_text
        from merlin.semantic_compiler.ordered_k_tiling import (
            materialize_row_major_k_slices,
        )
        from merlin.semantic_compiler.search import SearchLimits

        from tools.check_program_decode import audit_program
        from tools.check_variant_inventory import load_inventory

        source = (ROOT / "test/examples/native_mxu0_k64_linalg.mlir").read_text()
        logical = translate_linalg_text(
            source, entry="mxu0_k64_tile", target_identity=self.binding.profile().target_identity,
            lowering_policy="diagnostic",
        ).request
        self.assertEqual(tuple(node.type.shape for node in logical.nodes),
                         ((32, 64), (32, 64), (32, 32)))
        tiling = self.binding.prepare_ordered_k(logical, 32)
        self.assertEqual(tiling.tile_count, 2)
        self.assertEqual([(item.source_input, item.k_start) for item in tiling.source_slices],
                         [("arg0", 0), ("arg1", 0), ("arg0", 32), ("arg1", 32)])
        activation = bytes(0x38 if k % 2 else 0x40 for _ in range(32) for k in range(64))
        weight = bytes(0x38 if k % 3 else 0xB8 for _ in range(32) for k in range(64))
        packed = materialize_row_major_k_slices(
            tiling, {"arg0": activation, "arg1": weight}, element_bytes=1,
        )
        self.assertEqual(tuple(map(len, packed.values())), (1024,) * 4)
        self.assertEqual(packed["arg0__k1"][:32], activation[32:64])
        destination = self.work / "logical-k64"
        request = tiling.request
        self.compile_verified(
            request, fixed_inputs={item.tile_input: index + 3 for index, item in enumerate(tiling.source_slices)},
            fixed_outputs=(35,),
            destination=destination, limits=SearchLimits(),
        )
        program = json.loads((destination / "physical-program.json").read_text())
        self.assertEqual(audit_program(
            load_inventory(), program, (destination / "program.bin").read_bytes(),
        )["status"], "PASS")
        self.assertEqual(json.loads((destination / "execution_plan.json").read_text())["operation"], "mxu0_k2")

    def test_ordered_k_cli_rechecks_source_windows_and_refuses_bad_geometry(self) -> None:
        from atlas_mlir_native.binding import diagnostic_mxu0_f32_profile
        from merlin.targetgen.cli import main as targetgen_main

        source = ROOT / "test/examples/native_mxu0_k64_linalg.mlir"
        abi = self.work / "logical-k64-abi.json"
        abi.write_text(json.dumps({
            "fixed_inputs": {"arg0__k0": 3, "arg1__k0": 4,
                             "arg0__k1": 5, "arg1__k1": 6},
            "fixed_outputs": [35],
        }))
        common = [
            "--engine", "merlin_native", "--support", "atlas",
            "--snapshot", str(self.snapshot.root),
            "--linalg", str(source), "--linalg-entry", "mxu0_k64_tile",
            "--ordered-k-tile", "32", "--mode", "diagnostic",
        ]
        output = self.work / "logical-k64-cli"
        status = self.work / "logical-k64-cli-status.json"
        self.assertEqual(targetgen_main([
            "native-compile", *common, "--abi", str(abi),
            "--target-source", str(ROOT), "--out", str(output),
            "--status-file", str(status),
        ]), 0)
        compiled = json.loads(status.read_text())
        manifest = json.loads((output / "manifest.json").read_text())
        record = json.loads((output / "ordered-k-tiling.json").read_text())
        self.assertEqual((compiled["status"], record["tile_count"]), ("emitted", 2))
        self.assertEqual(manifest["source_request_digest"], record["source_request_digest"])
        self.assertEqual(manifest["request_digest"], record["tiled_request_digest"])
        self.assertEqual(
            manifest["stage_artifacts"]["ordered-k-tiling.json"],
            hashlib.sha256((output / "ordered-k-tiling.json").read_bytes()).hexdigest(),
        )
        verified = self.work / "logical-k64-cli-verified.json"
        self.assertEqual(targetgen_main([
            "native-verify", *common, "--artifact", str(output), "--out", str(verified),
        ]), 0)
        self.assertEqual(json.loads(verified.read_text())["status"], "verified_artifacts")

        host_map = self.work / "logical-k64-host-map.json"
        host_map.write_text(json.dumps({
            "schema": "atlas.host-map.v1",
            "rtl_revision": manifest["target_identity"].split(":", 1)[0].removeprefix("atlas-rtl-"),
            "soc_config_source": "generators/chipyard/src/main/scala/config/AtlasConfigs.scala::EE290SimConfig",
            "soc_source_revision": "1" * 40,
            "bitstream_identity": None,
            "imem_base": 0x20000, "imem_capacity_bytes": 0x20000,
            "csr_base": 0x40000, "start_csr_offset": 0x18,
        }))
        generator = ROOT / "tools/atlas_ordered_k_host_data.py"
        packer_header = Path(MERLIN) / "merlin/runtime/c/row_major_k_window.h"
        boundary_header = Path(MERLIN) / "merlin/runtime/c/finite_float_boundary.h"
        host_data = self.work / "logical-k64-host-data"
        generated = subprocess.run([
            sys.executable, str(generator), "--compiled", str(output),
            "--verification", str(verified), "--host-map", str(host_map),
            "--packer-header", str(packer_header), "--boundary-header", str(boundary_header),
            "--f32-boundary", "diagnostic-mxu0-f32",
            "--out", str(host_data),
        ], capture_output=True, text=True, check=False)
        self.assertEqual(generated.returncode, 0, generated.stderr)
        handoff = json.loads((host_data / "manifest.json").read_text())
        header = (host_data / "atlas_ordered_k_data.h").read_bytes()
        self.assertEqual((handoff["window_count"], handoff["status"]), (4, "diagnostic_unbound"))
        self.assertEqual(handoff["source_domain_coverage"], "runtime_subset_unqualified")
        self.assertEqual(handoff["f32_boundary"]["source_dtype"], "f32")
        self.assertEqual(
            handoff["f32_boundary"]["status"],
            "diagnostic_approximation_unqualified_model_quality",
        )
        self.assertEqual(handoff["runtime_admitted_code_ranges"], [[0, 0], [8, 119], [136, 247]])
        self.assertEqual(
            handoff["runtime_admitted_code_ranges"],
            [list(pair) for pair in diagnostic_mxu0_f32_profile().conversion_specs[0].admitted_code_ranges],
        )
        self.assertEqual(handoff["program_sha256"], manifest["binary_sha256"])
        self.assertEqual(handoff["header_sha256"], hashlib.sha256(header).hexdigest())
        self.assertEqual(handoff["packer_sha256"], hashlib.sha256(packer_header.read_bytes()).hexdigest())
        self.assertEqual((host_data / "row_major_k_window.h").read_bytes(), packer_header.read_bytes())
        self.assertEqual(handoff["boundary_sha256"], hashlib.sha256(boundary_header.read_bytes()).hexdigest())
        self.assertEqual((host_data / "finite_float_boundary.h").read_bytes(), boundary_header.read_bytes())
        self.assertIn(b"#define ATLAS_ORDERED_K_PROGRAM_WORD_COUNT 54u", header)
        self.assertIn(b"#define ATLAS_ORDERED_K_SOURCE_F32_DIAGNOSTIC 1", header)
        self.assertIn(b"atlas_ordered_k_encode_source_f32", header)
        self.assertIn(b".pack = merlin_pack_row_major_k_window", header)
        self.assertIn(b"merlin_unpack_split_bf16_f32(physical, logical, 32u, 32u, 16u)", header)
        self.assertIn(b"#ifndef ATLAS_DIAGNOSTIC_ORDERED_K", header)
        raw_data = self.work / "logical-k64-raw-host-data"
        raw_generated = subprocess.run([
            sys.executable, str(generator), "--compiled", str(output),
            "--verification", str(verified), "--host-map", str(host_map),
            "--packer-header", str(packer_header), "--boundary-header", str(boundary_header),
            "--out", str(raw_data),
        ], capture_output=True, text=True, check=False)
        self.assertEqual(raw_generated.returncode, 0, raw_generated.stderr)
        self.assertNotIn("f32_boundary", json.loads((raw_data / "manifest.json").read_text()))
        self.assertNotIn(b"ATLAS_ORDERED_K_SOURCE_F32_DIAGNOSTIC",
                         (raw_data / "atlas_ordered_k_data.h").read_bytes())

        without_tile = common[:common.index("--ordered-k-tile")] + common[common.index("--ordered-k-tile") + 2:]
        missing_flag = self.work / "logical-k64-cli-missing-flag.json"
        self.assertEqual(targetgen_main([
            "native-verify", *without_tile, "--artifact", str(output), "--out", str(missing_flag),
        ]), 2)
        self.assertIn("unrequested ordered-K partition", json.loads(missing_flag.read_text())["reason"])

        tampered = self.work / "logical-k64-cli-tampered"
        shutil.copytree(output, tampered)
        changed = json.loads((tampered / "ordered-k-tiling.json").read_text())
        changed["source_slices"][2]["k_start"] = 16
        (tampered / "ordered-k-tiling.json").write_text(json.dumps(changed))
        failure = self.work / "logical-k64-cli-tampered-status.json"
        self.assertEqual(targetgen_main([
            "native-verify", *common, "--artifact", str(tampered), "--out", str(failure),
        ]), 2)
        self.assertIn("ordered-K partition differs", json.loads(failure.read_text())["reason"])
        rejected = subprocess.run([
            sys.executable, str(generator), "--compiled", str(tampered),
            "--verification", str(verified), "--host-map", str(host_map),
            "--packer-header", str(packer_header), "--boundary-header", str(boundary_header),
            "--out", str(self.work / "tampered-host-data"),
        ], capture_output=True, text=True, check=False)
        self.assertNotEqual(rejected.returncode, 0)
        self.assertFalse((self.work / "tampered-host-data").exists())

        changed_map = self.work / "wrong-rtl-host-map.json"
        wrong_map = json.loads(host_map.read_text())
        wrong_map["rtl_revision"] = "0" * 40
        changed_map.write_text(json.dumps(wrong_map))
        rejected = subprocess.run([
            sys.executable, str(generator), "--compiled", str(output),
            "--verification", str(verified), "--host-map", str(changed_map),
            "--packer-header", str(packer_header), "--boundary-header", str(boundary_header),
            "--out", str(self.work / "wrong-map-host-data"),
        ], capture_output=True, text=True, check=False)
        self.assertNotEqual(rejected.returncode, 0)
        self.assertFalse((self.work / "wrong-map-host-data").exists())

        wrong = self.work / "logical-k64-wrong-tile.json"
        wrong_output = self.work / "logical-k64-wrong-tile"
        wrong_common = ["16" if item == "32" else item for item in common]
        self.assertEqual(targetgen_main([
            "native-compile", *wrong_common, "--abi", str(abi),
            "--target-source", str(ROOT), "--out", str(wrong_output),
            "--status-file", str(wrong),
        ]), 2)
        self.assertEqual(json.loads(wrong.read_text())["status"], "unsupported_semantics")
        self.assertFalse(wrong_output.exists())

    def test_ordered_k_cli_extends_source_and_host_windows(self) -> None:
        from merlin.targetgen.cli import main as targetgen_main

        source = (ROOT / "test/examples/native_mxu0_k64_linalg.mlir").read_text()
        generator = ROOT / "tools/atlas_ordered_k_host_data.py"
        runtime_headers = Path(MERLIN) / "merlin/runtime/c"
        host_map = self.work / "extended-k-host-map.json"
        host_map.write_text(json.dumps({
            "schema": "atlas.host-map.v1",
            "rtl_revision": self.binding.profile().target_identity.split(":", 1)[0].removeprefix("atlas-rtl-"),
            "soc_config_source": "generators/chipyard/src/main/scala/config/AtlasConfigs.scala::EE290SimConfig",
            "soc_source_revision": "1" * 40, "bitstream_identity": None,
            "imem_base": 0x20000, "imem_capacity_bytes": 0x20000,
            "csr_base": 0x40000, "start_csr_offset": 0x18,
        }))
        for k, expected_words in ((96, 72), (128, 90)):
            with self.subTest(k=k):
                selected = self.work / f"logical-k{k}.mlir"
                selected.write_text(source.replace("64", str(k)))
                abi = self.work / f"logical-k{k}-abi.json"
                abi.write_text(json.dumps({
                    "fixed_inputs": {f"arg{operand}__k{index}": 3 + 2 * index + operand
                                     for index in range(k // 32) for operand in (0, 1)},
                    "fixed_outputs": [35],
                }))
                common = [
                    "--engine", "merlin_native", "--support", "atlas",
                    "--snapshot", str(self.snapshot.root), "--linalg", str(selected),
                    "--linalg-entry", f"mxu0_k{k}_tile", "--ordered-k-tile", "32",
                    "--mode", "diagnostic",
                ]
                compiled = self.work / f"logical-k{k}-compiled"
                status = self.work / f"logical-k{k}-status.json"
                self.assertEqual(targetgen_main([
                    "native-compile", *common, "--abi", str(abi),
                    "--target-source", str(ROOT), "--out", str(compiled),
                    "--status-file", str(status),
                ]), 0)
                program = json.loads((compiled / "physical-program.json").read_text())
                self.assertEqual(program["word_count"], expected_words)
                receipt = self.work / f"logical-k{k}-verification.json"
                self.assertEqual(targetgen_main([
                    "native-verify", *common, "--artifact", str(compiled),
                    "--out", str(receipt),
                ]), 0)
                handoff = self.work / f"logical-k{k}-host-data"
                generated = subprocess.run([
                    sys.executable, str(generator), "--compiled", str(compiled),
                    "--verification", str(receipt), "--host-map", str(host_map),
                    "--packer-header", str(runtime_headers / "row_major_k_window.h"),
                    "--boundary-header", str(runtime_headers / "finite_float_boundary.h"),
                    "--f32-boundary", "diagnostic-mxu0-f32",
                    "--out", str(handoff),
                ], capture_output=True, text=True, check=False)
                self.assertEqual(generated.returncode, 0, generated.stderr)
                record = json.loads((handoff / "manifest.json").read_text())
                self.assertEqual(record["window_count"], 2 * (k // 32))
                self.assertEqual(record["status"], "diagnostic_unbound")
                self.assertEqual(record["source_domain_coverage"], "runtime_subset_unqualified")
                self.assertEqual(record["f32_boundary"]["source_dtype"], "f32")

    def mxu1_request(self):
        from dataclasses import replace

        from atlas_mlir_native.binding import MXU1_INPUT_DOMAIN, MXU1_POLICY
        from merlin.semantic_compiler.model import TensorType

        mxu0 = self.mxu0_request()
        return replace(mxu0, nodes=(
            replace(mxu0.nodes[0], attrs=(("input_domain", MXU1_INPUT_DOMAIN),)),
            replace(mxu0.nodes[1], attrs=(("input_domain", MXU1_INPUT_DOMAIN),)),
            replace(mxu0.nodes[2], type=TensorType((32, 32), "bf16", MXU1_POLICY)),
        ))

    def test_native_mxu0_tile_from_source_bound_f32_contraction(self) -> None:
        from atlas_mlir_native.binding import diagnostic_mxu0_f32_profile
        from merlin.semantic_compiler.precision_transform import (
            build_diagnostic_matmul_tile,
        )
        from merlin.semantic_compiler.search import SearchLimits

        source = (ROOT / "test/examples/captured_attention_mlp32_linalg.mlir").read_text()
        policy_bytes = (ROOT / "test/examples/atlas_attention_mlp_source_bound_policy.json").read_bytes()
        plan = build_diagnostic_matmul_tile(
            source, entry="forward", policy_document=policy_bytes,
            original_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
            region_id="matmul_0", profile=diagnostic_mxu0_f32_profile(),
        )
        self.assertEqual(plan.precision_policy_sha256, hashlib.sha256(policy_bytes).hexdigest())
        self.assertEqual(plan.inputs[1].target_axes_from_source, (1, 0))
        self.assertEqual(plan.inputs[1].source_value.record(), {
            "kind": "operation_result", "index": 1, "result_index": 0,
        })
        self.assertEqual(plan.request.node("rhs").type.shape, (32, 32))
        self.assertEqual(plan.record()["quality_status"], "unqualified")
        destination = self.work / "source-bound-mxu0"
        self.compile_verified(
            plan.request, fixed_inputs={"lhs": 3, "rhs": 4},
            fixed_outputs=(35,),
            destination=destination, limits=SearchLimits(),
        )
        program = json.loads((destination / "physical-program.json").read_text())
        self.assertGreater(program["word_count"], 0)
        self.assertIn("atlas.mxu_matmul", [row["operation"] for row in program["instructions"]])

    def test_source_bound_conversion_respects_each_mxu_input_domain(self) -> None:
        import numpy as np
        from atlas_mlir_native.binding import (
            diagnostic_mxu0_f32_profile,
            diagnostic_mxu0_scaled_f32_profile,
            diagnostic_mxu1_f32_profile,
        )

        mxu0 = diagnostic_mxu0_f32_profile().conversion_specs[0]
        mxu1 = diagnostic_mxu1_f32_profile().conversion_specs[0]
        self.assertEqual(mxu0.encode(np.array([1.0, 240.0], dtype=np.float32)).tolist(), [56, 119])
        self.assertEqual(mxu1.encode(np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)).tolist(),
                         [192, 184, 0, 56, 64])
        with self.assertRaisesRegex(ValueError, "input domain"):
            mxu0.encode(np.array([-0.001], dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "input domain"):
            mxu1.encode(np.array([1.125], dtype=np.float32))
        self.assertEqual(
            tuple(spec.pre_encode_scale_exponent
                  for spec in diagnostic_mxu0_scaled_f32_profile(128).conversion_specs[:2]),
            (-1, 1),
        )
        with self.assertRaisesRegex(ValueError, "scale code"):
            diagnostic_mxu0_scaled_f32_profile(121)

    def test_small_source_bound_matmul_compiles_full_physical_tile(self) -> None:
        from atlas_mlir_native.binding import diagnostic_mxu0_f32_profile
        from merlin.semantic_compiler.precision_transform import (
            build_diagnostic_matmul_tile,
        )
        from merlin.semantic_compiler.search import SearchLimits

        source = (ROOT / "test/examples/source_bound_small_matmul.mlir").read_text()
        profile = diagnostic_mxu0_f32_profile()
        source_sha = hashlib.sha256(source.encode()).hexdigest()
        policy = {
            **dict(profile.required_policy_fields),
            "source_sha256": source_sha,
            "matmul_regions": {"small": dict(profile.selected_decision)},
        }
        plan = build_diagnostic_matmul_tile(
            source, entry="forward", policy_document=json.dumps(policy, sort_keys=True).encode(),
            original_source_sha256=source_sha, region_id="small", profile=profile,
        )
        self.assertEqual(plan.inputs[0].source_type.shape, (1, 2))
        self.assertEqual(plan.request.node("result").type.shape, (32, 32))
        destination = self.work / "source-bound-small-mxu0"
        self.compile_verified(
            plan.request, fixed_inputs={"lhs": 3, "rhs": 4},
            fixed_outputs=(35,),
            destination=destination, limits=SearchLimits(),
        )
        program = json.loads((destination / "physical-program.json").read_text())
        self.assertIn("atlas.mxu_matmul", [row["operation"] for row in program["instructions"]])
        self.assertEqual((destination / "program.bin").stat().st_size, program["word_count"] * 4)

    def test_source_bound_mn_tiles_use_distinct_cli_source_windows(self) -> None:
        import numpy as np
        from atlas_mlir_native.binding import diagnostic_mxu0_f32_profile
        from atlas_mlir_native.precision_abi import pack_source_operand
        from merlin.frontends.compile_inputs import (
            load_compile_inputs,
            stage_compile_inputs,
        )
        from merlin.targetgen.cli import main as targetgen_main
        from safetensors.numpy import save

        source = (ROOT / "test/examples/source_bound_mn_matmul.mlir").read_bytes()
        source_sha = hashlib.sha256(source).hexdigest()
        profile = diagnostic_mxu0_f32_profile()
        policy = {
            **dict(profile.required_policy_fields),
            "source_sha256": source_sha,
            "matmul_regions": {"large": dict(profile.selected_decision)},
        }
        policy_path = self.work / "source-bound-mn-policy.json"
        policy_path.write_text(json.dumps(policy))
        trace = json.dumps({"status": "complete", "mlir": {"sha256": source_sha}}).encode()
        capture = self.work / "source-bound-mn-capture"
        capture.mkdir()
        members = {
            "model.mlir": source,
            "weights.safetensors": save({}),
            "weights.safetensors.manifest.json": json.dumps({
                "0": {"kind": "input", "name": "lhs"},
                "1": {"kind": "input", "name": "rhs"},
            }).encode(),
            "input_order.json": b'{"lhs": 0, "rhs": 1}',
            "frontend-trace.json": trace,
            "meta.json": json.dumps({
                "ok": True, "opaque": 0, "weights": "weights.safetensors",
                "frontend_trace": {"sha256": hashlib.sha256(trace).hexdigest()},
                "func_name": "forward",
                "input_abi": [{"shape": [3, 2], "dtype": "f32"},
                              {"shape": [2, 35], "dtype": "f32"}],
                "output_abi": [{"shape": [3, 35], "dtype": "f32"}],
                "dtype": "f32",
            }).encode(),
        }
        for name, payload in members.items():
            (capture / name).write_bytes(payload)
        (capture / "capture_receipt.json").write_text(json.dumps({
            "schema": "m2m.capture-receipt.v1",
            "materialized_abi": {"complete": True, "inputs": 2, "lifted_constants": []},
            "lifted_constants": {},
            "artifacts": {name: {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
                          for name, payload in members.items()},
        }))
        staged_path = self.work / "source-bound-mn-staged"
        stage_compile_inputs(capture, staged_path, precision_policy=policy_path)
        staged = load_compile_inputs(staged_path)
        left = self.binding.prepare_precision_tile(staged, "large", tile_index=0)
        right = self.binding.prepare_precision_tile(staged, "large", tile_index=1)
        self.assertEqual((left.tile_origin, right.tile_origin), ((0, 0), (0, 32)))
        self.assertNotEqual(left.request.digest(), right.request.digest())
        rhs = np.ones((2, 35), dtype=np.float32)
        rhs[:, 32:] = 2
        self.assertNotEqual(pack_source_operand(left, "rhs", rhs),
                            pack_source_operand(right, "rhs", rhs))
        self.assertEqual(right.output.window_shape, (3, 3))

        abi = self.work / "source-bound-mn-abi.json"
        abi.write_text(json.dumps({"fixed_inputs": {"lhs": 3, "rhs": 4}, "fixed_outputs": [35]}))
        common = ["--engine", "merlin_native", "--support", "atlas",
                  "--snapshot", str(self.snapshot.root), "--compile-inputs", str(staged_path),
                  "--diagnostic-region", "large", "--mode", "diagnostic"]
        for index, expected in enumerate((left, right)):
            tile_args = [*common, "--diagnostic-tile-index", str(index)]
            selected = self.work / f"source-bound-mn-select-{index}.json"
            self.assertEqual(targetgen_main(["native-select", *tile_args,
                                             "--abi", str(abi), "--out", str(selected)]), 0)
            self.assertEqual(json.loads(selected.read_text())["request_digest"],
                             expected.request.digest())
            output = self.work / f"source-bound-mn-compiled-{index}"
            self.assertEqual(targetgen_main(["native-compile", *tile_args, "--abi", str(abi),
                                             "--target-source", str(ROOT), "--out", str(output)]), 0)
            plan = json.loads((output / "precision-plan.json").read_text())
            self.assertEqual((plan["schema"], plan["tile_origin"]),
                             ("merlin.diagnostic_precision_tile.v2", list(expected.tile_origin)))
            receipt = self.work / f"source-bound-mn-verify-{index}.json"
            self.assertEqual(targetgen_main(["native-verify", *tile_args,
                                             "--artifact", str(output), "--out", str(receipt)]), 0)
            self.assertEqual(json.loads(receipt.read_text())["status"], "verified_artifacts")
        wrong = self.work / "source-bound-mn-wrong-tile.json"
        self.assertEqual(targetgen_main(["native-verify", *common, "--diagnostic-tile-index", "1",
                                         "--artifact", str(self.work / "source-bound-mn-compiled-0"),
                                         "--out", str(wrong)]), 2)
        self.assertNotEqual(json.loads(wrong.read_text())["status"], "verified_artifacts")
        out_of_range = self.work / "source-bound-mn-out-of-range.json"
        self.assertEqual(targetgen_main(["native-select", *common, "--diagnostic-tile-index", "2",
                                         "--abi", str(abi), "--out", str(out_of_range)]), 2)
        self.assertNotEqual(json.loads(out_of_range.read_text())["status"], "selected")

    def test_batched_source_tiles_compile_with_distinct_checked_boundaries(self) -> None:
        import numpy as np
        from atlas_mlir_native.binding import diagnostic_mxu0_f32_profile
        from merlin.frontends.compile_inputs import (
            load_compile_inputs,
            stage_compile_inputs,
        )
        from merlin.semantic_compiler.precision_constants import (
            load_declared_static_operand,
        )
        from merlin.targetgen.cli import main as targetgen_main
        from safetensors.numpy import save

        source = (ROOT / "test/examples/source_bound_batched_matmul.mlir").read_bytes()
        source_sha = hashlib.sha256(source).hexdigest()
        profile = diagnostic_mxu0_f32_profile()
        policy = {
            **dict(profile.required_policy_fields),
            "schema": "atlas.oot.quantized_linalg_policy.v8",
            "zero_sign_policy": "canonicalize_encoded_negative_zero",
            "source_sha256": source_sha,
            "matmul_regions": {},
            "batched_matmul_regions": {"batch_matmul#0": dict(profile.selected_decision)},
        }
        policy_path = self.work / "batched-source-policy.json"
        policy_path.write_text(json.dumps(policy))
        trace = json.dumps({"status": "complete", "mlir": {"sha256": source_sha}}).encode()
        capture = self.work / "batched-source-capture"
        capture.mkdir()
        weights = np.ones((2, 2, 2), dtype=np.float32)
        weights[1] = 2.0
        members = {
            "model.mlir": source,
            "weights.safetensors": save({"rhs_weight": weights}),
            "weights.safetensors.manifest.json": json.dumps({
                "0": {"kind": "input", "name": "lhs"},
                "1": {"kind": "param", "weight": "rhs_weight", "shape": [2, 2, 2], "dtype": "float32"},
            }).encode(),
            "input_order.json": b'{"lhs": 0}',
            "frontend-trace.json": trace,
            "meta.json": json.dumps({
                "ok": True, "opaque": 0, "weights": "weights.safetensors",
                "frontend_trace": {"sha256": hashlib.sha256(trace).hexdigest()},
                "func_name": "forward",
                "input_abi": [{"shape": [2, 2, 2], "dtype": "f32"}],
                "output_abi": [{"shape": [2, 2, 2], "dtype": "f32"}],
                "dtype": "f32",
            }).encode(),
        }
        for name, payload in members.items():
            (capture / name).write_bytes(payload)
        (capture / "capture_receipt.json").write_text(json.dumps({
            "schema": "m2m.capture-receipt.v1",
            "materialized_abi": {"complete": True, "inputs": 1, "lifted_constants": []},
            "lifted_constants": {},
            "artifacts": {name: {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
                          for name, payload in members.items()},
        }))
        staged_path = self.work / "batched-source-staged"
        stage_compile_inputs(capture, staged_path, precision_policy=policy_path)
        staged = load_compile_inputs(staged_path)
        plans = tuple(
            self.binding.prepare_precision_tile(staged, "batch_matmul#0", tile_index=index)
            for index in range(2)
        )
        self.assertEqual(tuple(plan.batch_index for plan in plans), ((0,), (1,)))
        self.assertNotEqual(plans[0].request.digest(), plans[1].request.digest())
        self.assertEqual(
            tuple(load_declared_static_operand(staged_path, plan, "rhs").value.shape for plan in plans),
            ((2, 2, 2), (2, 2, 2)),
        )
        abi = self.work / "batched-source-abi.json"
        abi.write_text(json.dumps({"fixed_inputs": {"lhs": 3, "rhs": 4}, "fixed_outputs": [35]}))
        common = ["--engine", "merlin_native", "--support", "atlas",
                  "--snapshot", str(self.snapshot.root), "--compile-inputs", str(staged_path),
                  "--diagnostic-region", "batch_matmul#0", "--mode", "diagnostic"]
        packed_weights = []
        for index, expected in enumerate(plans):
            tile_args = [*common, "--diagnostic-tile-index", str(index)]
            selected = self.work / f"batched-source-select-{index}.json"
            self.assertEqual(targetgen_main(["native-select", *tile_args,
                                             "--abi", str(abi), "--out", str(selected)]), 0)
            self.assertEqual(json.loads(selected.read_text())["request_digest"],
                             expected.request.digest())
            output = self.work / f"batched-source-compiled-{index}"
            self.assertEqual(targetgen_main(["native-compile", *tile_args, "--abi", str(abi),
                                             "--target-source", str(ROOT), "--out", str(output)]), 0)
            precision = json.loads((output / "precision-plan.json").read_text())
            self.assertEqual(precision["batch_index"], [index])
            self.assertEqual(precision["schema"], "merlin.diagnostic_precision_tile.v4")
            packed_weights.append((output / "constants/rhs.bin").read_bytes())
            verified = self.work / f"batched-source-verify-{index}.json"
            self.assertEqual(targetgen_main(["native-verify", *tile_args,
                                             "--artifact", str(output), "--out", str(verified)]), 0)
            self.assertEqual(json.loads(verified.read_text())["status"], "verified_artifacts")
        self.assertNotEqual(*packed_weights)
        from tools.atlas_source_mn_host_data import prepare as prepare_source_mn

        grid = self.work / "batched-static-grid"
        grid.mkdir()
        for index in range(2):
            artifact = grid / f"tile-{index}"
            tile_args = [*common, "--diagnostic-tile-index", str(index)]
            self.assertEqual(targetgen_main(["native-compile", *tile_args, "--abi", str(abi),
                                             "--target-source", str(ROOT),
                                             "--out", str(artifact)]), 0)
            self.assertEqual(targetgen_main(["native-verify", *tile_args,
                                             "--artifact", str(artifact),
                                             "--out", str(grid / f"tile-{index}-verification.json")]), 0)
        files, record = prepare_source_mn(
            grid, ROOT / "runtime/selected-host-map.json",
            Path(MERLIN) / "merlin/runtime/c/row_major_k_window.h",
            Path(MERLIN) / "merlin/runtime/c/finite_float_boundary.h",
            staged_root=staged_path,
        )
        self.assertEqual(record["batch_count"], 2)
        self.assertEqual(record["frozen_rhs"]["tensor_sha256"],
                         hashlib.sha256(weights.tobytes()).hexdigest())
        self.assertIn(b"atlas_source_mn_frozen_rhs_bits[8]", files["atlas_source_mn_data.h"])
        self.assertIn(b"0x40000000u", files["atlas_source_mn_data.h"])
        bundle = self.work / "batched-static-host-data"
        bundle.mkdir()
        for name, payload in files.items():
            (bundle / name).write_bytes(payload)
        (bundle / "manifest.json").write_text(json.dumps(record, sort_keys=True) + "\n")
        executable = self.work / "batched-static-fake-host"
        compile_result = subprocess.run([
            "cc", "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
            "-DATLAS_TEST_MN_SOURCE_GENERATED=1",
            "-DATLAS_TEST_MN_MEMREF=1",
            "-DATLAS_USE_GENERATED_SOURCE_MN_DATA=1",
            "-DATLAS_TEST_MN_POSITIVE_ZERO=1",
            "-DATLAS_DIAGNOSTIC_SOURCE_MN=1", f"-I{bundle}", f"-I{ROOT / 'runtime'}",
            *(str(bundle / name) for name in (
                "atlas_host.c", "atlas_ordered_k.c", "atlas_ordered_mn.c",
                "atlas_source_mn.c")),
            str(ROOT / "runtime/atlas_source_mn_memref.c"),
            str(ROOT / "test/ordered-mn-host-test.c"), "-o", str(executable),
        ], check=False, capture_output=True, text=True)
        self.assertEqual(compile_result.returncode, 0, compile_result.stderr)
        run_result = subprocess.run([str(executable)], check=False, capture_output=True, text=True)
        self.assertEqual(run_result.returncode, 0, run_result.stderr)
        if shutil.which("clang") and shutil.which("ld.lld"):
            host_build = self.work / "batched-static-host-elf"
            build_result = subprocess.run([
                "bash", str(ROOT / "examples/handoff/source_mn/build.sh"),
                str(bundle), str(host_build),
            ], check=False, capture_output=True, text=True)
            self.assertEqual(build_result.returncode, 0, build_result.stderr)
            self.assertEqual((host_build / "atlas-source-mn.elf").read_bytes()[:4], b"\x7fELF")

    def test_batched_source_grid_exports_one_checked_host_image(self) -> None:
        from atlas_mlir_native.binding import diagnostic_mxu0_f32_profile
        from merlin.frontends.compile_inputs import stage_compile_inputs
        from merlin.targetgen.cli import main as targetgen_main
        from safetensors.numpy import save

        from tools.atlas_source_mn_host_data import prepare as prepare_source_mn

        source = (ROOT / "test/examples/source_bound_batched_matmul_m_tail.mlir").read_bytes()
        source_sha = hashlib.sha256(source).hexdigest()
        profile = diagnostic_mxu0_f32_profile()
        policy = {
            **dict(profile.required_policy_fields),
            "schema": "atlas.oot.quantized_linalg_policy.v8",
            "zero_sign_policy": "canonicalize_encoded_negative_zero",
            "source_sha256": source_sha,
            "matmul_regions": {},
            "batched_matmul_regions": {"batch_matmul#0": dict(profile.selected_decision)},
        }
        policy_path = self.work / "batched-grid-policy.json"
        policy_path.write_text(json.dumps(policy))
        trace = json.dumps({"status": "complete", "mlir": {"sha256": source_sha}}).encode()
        capture = self.work / "batched-grid-capture"
        capture.mkdir()
        members = {
            "model.mlir": source,
            "weights.safetensors": save({}),
            "weights.safetensors.manifest.json": json.dumps({
                "0": {"kind": "input", "name": "lhs"},
                "1": {"kind": "input", "name": "rhs"},
            }).encode(),
            "input_order.json": b'{"lhs": 0, "rhs": 1}',
            "frontend-trace.json": trace,
            "meta.json": json.dumps({
                "ok": True, "opaque": 0, "weights": "weights.safetensors",
                "frontend_trace": {"sha256": hashlib.sha256(trace).hexdigest()},
                "func_name": "forward",
                "input_abi": [{"shape": [2, 33, 2], "dtype": "f32"},
                              {"shape": [2, 2, 2], "dtype": "f32"}],
                "output_abi": [{"shape": [2, 33, 2], "dtype": "f32"}],
                "dtype": "f32",
            }).encode(),
        }
        for name, payload in members.items():
            (capture / name).write_bytes(payload)
        (capture / "capture_receipt.json").write_text(json.dumps({
            "schema": "m2m.capture-receipt.v1",
            "materialized_abi": {"complete": True, "inputs": 2, "lifted_constants": []},
            "lifted_constants": {},
            "artifacts": {name: {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
                          for name, payload in members.items()},
        }))
        staged = self.work / "batched-grid-staged"
        stage_compile_inputs(capture, staged, precision_policy=policy_path)
        abi = self.work / "batched-grid-abi.json"
        abi.write_text(json.dumps({"fixed_inputs": {"lhs": 3, "rhs": 4},
                                   "fixed_outputs": [35]}))
        common = ["--engine", "merlin_native", "--support", "atlas",
                  "--snapshot", str(self.snapshot.root), "--compile-inputs", str(staged),
                  "--diagnostic-region", "batch_matmul#0", "--mode", "diagnostic"]
        grid = self.work / "batched-grid"
        grid.mkdir()
        for index in range(4):
            tile_args = [*common, "--diagnostic-tile-index", str(index)]
            artifact = grid / f"tile-{index}"
            self.assertEqual(targetgen_main(["native-compile", *tile_args, "--abi", str(abi),
                                             "--target-source", str(ROOT),
                                             "--out", str(artifact)]), 0)
            self.assertEqual(json.loads((artifact / "manifest.json").read_text())[
                "constant_inputs"], [])
            receipt = grid / f"tile-{index}-verification.json"
            self.assertEqual(targetgen_main(["native-verify", *tile_args,
                                             "--artifact", str(artifact),
                                             "--out", str(receipt)]), 0)
        files, record = prepare_source_mn(
            grid, ROOT / "runtime/selected-host-map.json",
            Path(MERLIN) / "merlin/runtime/c/row_major_k_window.h",
            Path(MERLIN) / "merlin/runtime/c/finite_float_boundary.h",
        )
        self.assertEqual(
            (record["batch_count"], record["tile_count"], record["matrix_tile_count"]),
            (2, 4, 2),
        )
        self.assertEqual([tile["batch_index"] for tile in record["tiles"]], [0, 0, 1, 1])
        self.assertEqual([tile["matrix_tile_index"] for tile in record["tiles"]], [0, 1, 0, 1])
        self.assertIn(b"#define ATLAS_SOURCE_MN_BATCHES 2u", files["atlas_source_mn_data.h"])
        self.assertIn(b"#define ATLAS_SOURCE_MN_TILE_COUNT 2u", files["atlas_source_mn_data.h"])
        bundle = self.work / "batched-grid-host-data"
        bundle.mkdir()
        for name, payload in files.items():
            (bundle / name).write_bytes(payload)
        (bundle / "manifest.json").write_text(json.dumps(record, sort_keys=True) + "\n")
        from merlin.frontends.linalg_mlir import parse_mlir_text
        from merlin.semantic_compiler.precision_intake import (
            survey_source_f32_batched_matmuls,
        )
        from merlin.targetgen.application_graph import _program_graph
        from merlin.targetgen.contract.model_host_bridge import diagnostic_host_bridge

        from tools.atlas_source_mn_model_bridge import prepare as prepare_model_wrapper

        (batched_region,), _ = survey_source_f32_batched_matmuls(
            source.decode(), entry="forward",
        )
        graph = _program_graph(parse_mlir_text(source.decode()), source_sha)
        operation = next(row for row in graph["operations"]
                         if row["mlir_operation"] == "linalg.generic")
        region = {
            "region_id": batched_region.region_id,
            "operation_id": operation["operation_id"],
            "operation_ordinal": batched_region.operation_ordinal,
            "batch_shape": [2], "status": "isolated_tiles_emitted",
            "static_inputs_status": "packed_or_runtime", "tile_count": 4,
        }
        model_probe = self.work / "batched-grid-model-probe"
        model_probe.mkdir()
        batched_tiles = []
        for index in range(4):
            artifact = f"regions/batched-0000-tile-{index:04d}"
            shutil.copytree(grid / f"tile-{index}", model_probe / artifact)
            tile_manifest = json.loads((grid / f"tile-{index}/manifest.json").read_text())
            batched_tiles.append({
                "region_id": region["region_id"],
                "operation_id": region["operation_id"],
                "tile_index": index, "status": "emitted",
                "static_inputs_status": "packed_or_runtime",
                "artifact": artifact,
                "request_digest": tile_manifest["request_digest"],
                "binary_sha256": tile_manifest["binary_sha256"],
            })
        bridge, bridge_text = diagnostic_host_bridge(
            source.decode(), "forward", [],
            batched_regions=[region], batched_tiles=batched_tiles,
        )
        self.assertEqual(bridge["bindings"][0]["selected_route"], "tiled_native")
        (model_probe / "host-bridge.mlir").write_text(bridge_text)
        bridge.update({"path": "host-bridge.mlir",
                       "sha256": hashlib.sha256(bridge_text.encode()).hexdigest()})
        probe_record = {
            "schema": "merlin.native_model_region_probe.v1",
            "engine": "merlin_native", "status": "incomplete",
            "source_operation_count": len(graph["operations"]),
            "target_identity": record["target_identity"],
            **{key: record["source_identity"][key] for key in (
                "original_source_sha256", "staged_program_sha256",
                "precision_policy_sha256",
            )},
            "regions": [], "batched_contractions": [region],
            "batched_tiles": batched_tiles, "host_bridge": bridge,
        }
        (model_probe / "manifest.json").write_text(json.dumps(probe_record, sort_keys=True) + "\n")
        wrapper = prepare_model_wrapper(model_probe, region["region_id"], bundle)
        self.assertEqual(wrapper["record"]["rank"], 3)
        self.assertEqual(wrapper["record"]["batch_count"], 2)
        self.assertIn("atlas_source_mn_run_memref3", wrapper["files"]["atlas_source_mn_model_wrapper.c"])
        wrapper_dir = self.work / "batched-grid-model-wrapper"
        wrapper_dir.mkdir()
        for name, content in wrapper["files"].items():
            (wrapper_dir / name).write_text(content)
        wrapper_object = wrapper_dir / "wrapper.o"
        wrapper_build = subprocess.run([
            "cc", "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
            "-DATLAS_DIAGNOSTIC_ALLOW_UNBOUND=1",
            "-DATLAS_DIAGNOSTIC_ASSUME_COHERENT=1",
            "-DATLAS_DIAGNOSTIC_PRECISION_POLICY=1",
            "-DATLAS_DIAGNOSTIC_SOURCE_MN=1",
            f"-I{bundle}", f"-I{wrapper_dir}",
            "-c", str(wrapper_dir / "atlas_source_mn_model_wrapper.c"),
            "-o", str(wrapper_object),
        ], check=False, capture_output=True, text=True)
        self.assertEqual(wrapper_build.returncode, 0, wrapper_build.stderr)
        wrapper_record = {
            **wrapper["record"],
            "files": {name: hashlib.sha256(content.encode()).hexdigest()
                      for name, content in wrapper["files"].items()},
        }
        (wrapper_dir / "manifest.json").write_text(json.dumps(wrapper_record, sort_keys=True) + "\n")
        from tools.atlas_model_link import _check_native_reset_words, _selection

        symbol = bridge["bindings"][0]["symbol"]
        host_object = self.work / "batched-grid-model-host.o"
        host_object.write_bytes(b"selected outlined host object")
        source_count = len(graph["operations"])
        host_check = {
            "schema": "merlin.native_host_object_check.v1",
            "status": "object_symbols_checked",
            "selected_device_symbols": [symbol],
            "required_native_unmet": [],
            "host_object_sha256": hashlib.sha256(host_object.read_bytes()).hexdigest(),
            "entry_dataflow": {
                "schema": "merlin.host_entry_dataflow_check.v2",
                "status": "source_to_host_body_checked",
                "operation_count": source_count,
                "native_external_call_count": 1,
                "host_defined_call_count": 0,
                "host_linalg_body_correspondence_checked_count": 0,
                "checked_operand_edges": 0,
                "initialization_semantics": "source_drop_and_host_initializer_checked",
                "native_reset_semantics": "not_checked",
                "native_zero_initialization_obligation_count": 1,
            },
            "source_route_accounting": {
                "schema": "merlin.model_source_route_accounting.v1",
                "operation_count": source_count,
                "state_counts": {
                    "native_external_unexecuted": 1,
                    "entry_support_numerics_unqualified": source_count - 2,
                    "model_output_unexecuted": 1,
                },
                "operations": [{"operation_ordinal": index}
                               for index in range(source_count)],
            },
            "entry_abi": {
                "source_input_count": 2, "source_result_count": 1,
                "input_descriptor_positions": [0, 1],
                "result_descriptor_positions": [2],
                "descriptor_types": [
                    "memref<2x33x2xf32>", "memref<2x2x2xf32>",
                    "memref<2x33x2xf32>",
                ],
            },
        }
        rows = _selection(
            probe_record, host_check, host_object,
            [(wrapper_dir, wrapper_record)], probe_path=model_probe,
            source_bundles=[bundle],
        )
        self.assertEqual([row["kind"] for row in rows], ["source_grid"])
        resets = _check_native_reset_words(model_probe, probe_record, host_check)
        self.assertEqual([row["tile_index"] for row in resets], list(range(4)))
        clang = shutil.which("clang")
        llvm_nm = shutil.which("llvm-nm") or "/usr/lib/llvm-18/bin/llvm-nm"
        if clang and Path(llvm_nm).is_file():
            from argparse import Namespace

            from tools.atlas_model_link import build as build_model_link

            host_source = self.work / "batched-grid-link-stub.c"
            host_source.write_text(
                '#include "atlas_model_tile.h"\n'
                'extern void device(atlas_f32_memref3 *, const atlas_f32_memref3 *, '
                'const atlas_f32_memref3 *) __asm__("_mlir_ciface_' + symbol + '");\n'
                'void _mlir_ciface_forward(const atlas_f32_memref3 *lhs, '
                'const atlas_f32_memref3 *rhs, atlas_f32_memref3 *result) {\n'
                '  device(result, lhs, rhs);\n}\n'
            )
            subprocess.run([
                clang, "--target=riscv64-unknown-elf", "-march=rv64gcv",
                "-mabi=lp64d", "-ffreestanding", "-I", str(ROOT / "runtime"),
                "-c", str(host_source), "-o", str(host_object),
            ], check=True, capture_output=True, text=True)
            host_check["host_object_sha256"] = hashlib.sha256(host_object.read_bytes()).hexdigest()
            host_check["probe_manifest_sha256"] = hashlib.sha256(
                (model_probe / "manifest.json").read_bytes()).hexdigest()
            host_check_path = self.work / "batched-grid-host-check.json"
            host_check_path.write_text(json.dumps(host_check, sort_keys=True) + "\n")
            linked = build_model_link(Namespace(
                out=str(self.work / "batched-grid-linked"),
                probe=str(model_probe), host_check=str(host_check_path),
                host_object=str(host_object), wrapper=[str(wrapper_dir)],
                source_bundle=[str(bundle)],
                merlin_runtime_include=str(Path(MERLIN) / "merlin/runtime/c"),
                clang=clang, nm=llvm_nm,
            ))
            self.assertEqual(linked["status"], "relocatable_link_checked")
            self.assertEqual(len(linked["checked_native_reset_words"]), 4)
            self.assertEqual((self.work / "batched-grid-linked/combined.o").read_bytes()[:4],
                             b"\x7fELF")
        changed_probe = copy.deepcopy(probe_record)
        changed_probe["batched_tiles"][0]["tile_index"] = 3
        (model_probe / "manifest.json").write_text(json.dumps(changed_probe, sort_keys=True) + "\n")
        with self.assertRaisesRegex(ValueError, "missing selected model tiles"):
            prepare_model_wrapper(model_probe, region["region_id"], bundle)
        (model_probe / "manifest.json").write_text(json.dumps(probe_record, sort_keys=True) + "\n")
        executable = self.work / "batched-grid-fake-host"
        compile_result = subprocess.run([
            "cc", "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
            "-DATLAS_TEST_MN_SOURCE_GENERATED=1",
            "-DATLAS_USE_GENERATED_SOURCE_MN_DATA=1",
            "-DATLAS_TEST_MN_POSITIVE_ZERO=1",
            "-DATLAS_DIAGNOSTIC_SOURCE_MN=1", f"-I{bundle}",
            *(str(bundle / name) for name in (
                "atlas_host.c", "atlas_ordered_k.c", "atlas_ordered_mn.c",
                "atlas_source_mn.c")),
            str(ROOT / "test/ordered-mn-host-test.c"), "-o", str(executable),
        ], check=False, capture_output=True, text=True)
        self.assertEqual(compile_result.returncode, 0, compile_result.stderr)
        run_result = subprocess.run([str(executable)], check=False, capture_output=True, text=True)
        self.assertEqual(run_result.returncode, 0, run_result.stderr)
        if shutil.which("clang") and shutil.which("ld.lld"):
            host_build = self.work / "batched-grid-host-elf"
            build_result = subprocess.run([
                "bash", str(ROOT / "examples/handoff/source_mn/build.sh"),
                str(bundle), str(host_build),
            ], check=False, capture_output=True, text=True)
            self.assertEqual(build_result.returncode, 0, build_result.stderr)
            image = (host_build / "atlas-source-mn.elf").read_bytes()
            self.assertEqual(image[:4], b"\x7fELF")
            self.assertEqual(struct.unpack_from("<H", image, 18)[0], 243)

    def test_source_bound_k72_tile_uses_checked_native_mxu0_continuation(self) -> None:
        from dataclasses import replace

        import numpy as np
        from atlas_mlir_native.binding import (
            BF16_DRAM_PAIR_LAYOUT,
            FP8_DRAM_TILE_LAYOUT,
            diagnostic_mxu0_f32_profile,
        )
        from atlas_mlir_native.precision_abi import pack_source_operand
        from merlin.frontends.compile_inputs import (
            StagedCompileInputs,
            load_compile_inputs,
            stage_compile_inputs,
        )
        from merlin.semantic_compiler.linalg_bridge import LinalgBridgeError
        from merlin.semantic_compiler.search import SearchLimits
        from merlin.targetgen.cli import main as targetgen_main
        from safetensors.numpy import save

        source = (ROOT / "test/examples/source_bound_mn_k72_matmul.mlir").read_text()
        source_sha = hashlib.sha256(source.encode()).hexdigest()
        profile = diagnostic_mxu0_f32_profile()
        policy = {
            **dict(profile.required_policy_fields),
            "source_sha256": source_sha,
            "matmul_regions": {"large": dict(profile.selected_decision)},
        }
        policy_bytes = json.dumps(policy).encode()
        staged = StagedCompileInputs(
            program=source, entry="forward", precision_policy=policy,
            precision_policy_document=policy_bytes,
            original_source_sha256=source_sha,
            program_sha256=source_sha,
            precision_policy_sha256=hashlib.sha256(policy_bytes).hexdigest(),
        )
        plan = self.binding.prepare_precision_tile(staged, "large", tile_index=1)
        mxu1_policy = {
            **policy,
            "matmul_regions": {"large": {"mxu_unit": 1, "activation_pack_scale_code": 127}},
        }
        mxu1_bytes = json.dumps(mxu1_policy).encode()
        with self.assertRaisesRegex(LinalgBridgeError, "shape exceeds"):
            self.binding.prepare_precision_tile(replace(
                staged, precision_policy=mxu1_policy,
                precision_policy_document=mxu1_bytes,
                precision_policy_sha256=hashlib.sha256(mxu1_bytes).hexdigest(),
            ), "large", tile_index=1)
        self.assertEqual(plan.tile_origin, (0, 32))
        self.assertEqual(plan.record()["schema"], "merlin.diagnostic_precision_tile.v3")
        self.assertEqual([item.source_offsets for item in plan.inputs], [
            (0, 0), (0, 32), (0, 8), (8, 32), (0, 40), (40, 32),
        ])
        self.assertEqual([item.target_offsets for item in plan.inputs], [
            (0, 24), (0, 24), (0, 0), (0, 0), (0, 0), (0, 0),
        ])
        lhs = np.ones((3, 72), dtype=np.float32)
        rhs = np.ones((72, 35), dtype=np.float32)
        self.assertEqual(pack_source_operand(plan, "lhs__k0", lhs)[:24], b"\x00" * 24)
        self.assertEqual(pack_source_operand(plan, "rhs__k0", rhs)[:24], b"\x00" * 24)
        destination = self.work / "source-bound-k72-mxu0"
        self.compile_verified(
            plan.request, fixed_inputs={item.source_role: index + 3 for index, item in enumerate(plan.inputs)},
            fixed_outputs=(35,),
            destination=destination, limits=SearchLimits(),
        )
        execution = json.loads((destination / "execution_plan.json").read_text())
        self.assertEqual(execution["operation"], "mxu0_k3")
        program = json.loads((destination / "physical-program.json").read_text())
        self.assertEqual([row["fields"]["accumulate"] for row in program["instructions"]
                          if row["operation"] == "atlas.mxu_matmul"], [False, True, True])

        trace = json.dumps({"status": "complete", "mlir": {"sha256": source_sha}}).encode()
        capture = self.work / "source-bound-k72-capture"
        capture.mkdir()
        members = {
            "model.mlir": source.encode(),
            "weights.safetensors": save({}),
            "weights.safetensors.manifest.json": json.dumps({
                "0": {"kind": "input", "name": "lhs"},
                "1": {"kind": "input", "name": "rhs"},
            }).encode(),
            "input_order.json": b'{"lhs": 0, "rhs": 1}',
            "frontend-trace.json": trace,
            "meta.json": json.dumps({
                "ok": True, "opaque": 0, "weights": "weights.safetensors",
                "frontend_trace": {"sha256": hashlib.sha256(trace).hexdigest()},
                "func_name": "forward",
                "input_abi": [{"shape": [3, 72], "dtype": "f32"},
                              {"shape": [72, 35], "dtype": "f32"}],
                "output_abi": [{"shape": [3, 35], "dtype": "f32"}],
                "dtype": "f32",
            }).encode(),
        }
        for name, payload in members.items():
            (capture / name).write_bytes(payload)
        (capture / "capture_receipt.json").write_text(json.dumps({
            "schema": "m2m.capture-receipt.v1",
            "materialized_abi": {"complete": True, "inputs": 2, "lifted_constants": []},
            "lifted_constants": {},
            "artifacts": {name: {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
                          for name, payload in members.items()},
        }))
        policy_path = self.work / "source-bound-k72-policy.json"
        policy_path.write_bytes(policy_bytes)
        staged_root = self.work / "source-bound-k72-staged"
        stage_compile_inputs(capture, staged_root, precision_policy=policy_path)
        abi = self.work / "source-bound-k72-abi.json"
        abi.write_text(json.dumps({
            "fixed_inputs": {item.source_role: index + 3 for index, item in enumerate(plan.inputs)},
            "fixed_outputs": [35],
        }))
        common = ["--engine", "merlin_native", "--support", "atlas",
                  "--snapshot", str(self.snapshot.root), "--compile-inputs", str(staged_root),
                  "--diagnostic-region", "large", "--diagnostic-tile-index", "1",
                  "--mode", "diagnostic"]
        compiled = self.work / "source-bound-k72-cli"
        self.assertEqual(targetgen_main(["native-compile", *common, "--abi", str(abi),
                                         "--target-source", str(ROOT), "--out", str(compiled)]), 0)
        self.assertEqual(json.loads((compiled / "precision-plan.json").read_text())["schema"],
                         "merlin.diagnostic_precision_tile.v3")
        self.assertEqual(json.loads((compiled / "manifest.json").read_text())["constant_inputs"], [])
        receipt = self.work / "source-bound-k72-cli-verification.json"
        self.assertEqual(targetgen_main(["native-verify", *common, "--artifact", str(compiled),
                                         "--out", str(receipt)]), 0)
        self.assertEqual(json.loads(receipt.read_text())["status"], "verified_artifacts")
        staged_plan = self.binding.prepare_precision_tile(
            load_compile_inputs(staged_root), "large", tile_index=1,
        )
        self.assertEqual(json.loads((compiled / "precision-plan.json").read_text())
                         ["request_digest"], staged_plan.request.digest())
        exact_layouts = self.work / "source-bound-k72-exact-layouts.json"
        exact_layouts.write_text(json.dumps({
            "schema": "merlin.physical_boundary_layouts.v2",
            "target_identity": staged_plan.request.target_identity,
            "source_identity": staged_plan.request.source_identity,
            "request_digest": staged_plan.request.digest(),
            "physical_extents": dict(staged_plan.request.physical_extents),
            "boundary_layouts": {
                item.source_role: FP8_DRAM_TILE_LAYOUT for item in staged_plan.inputs
            },
            "output_layouts": [BF16_DRAM_PAIR_LAYOUT],
        }))
        stale_layouts = self.work / "source-bound-k72-stale-layouts.json"
        stale_layouts.write_text(json.dumps({
            **json.loads(exact_layouts.read_text()),
            "request_digest": plan.request.digest(),
        }))
        stale_status = self.work / "source-bound-k72-stale-selection.json"
        self.assertEqual(targetgen_main(["native-select", *common,
                                         "--physical-layouts", str(stale_layouts),
                                         "--out", str(stale_status)]), 2)
        self.assertEqual(json.loads(stale_status.read_text())["status"], "compile_error")
        exact_common = [*common, "--physical-layouts", str(exact_layouts)]
        exact_compiled = self.work / "source-bound-k72-exact-layout-cli"
        self.assertEqual(targetgen_main(["native-compile", *exact_common,
                                         "--abi", str(abi), "--target-source", str(ROOT),
                                         "--out", str(exact_compiled)]), 0)
        self.assertEqual((exact_compiled / "program.bin").read_bytes(),
                         (compiled / "program.bin").read_bytes())
        exact_receipt = self.work / "source-bound-k72-exact-layout-verification.json"
        self.assertEqual(targetgen_main(["native-verify", *exact_common,
                                         "--artifact", str(exact_compiled),
                                         "--out", str(exact_receipt)]), 0)
        self.assertEqual(json.loads(exact_receipt.read_text())["status"], "verified_artifacts")

        static_capture = self.work / "source-bound-k72-static-capture"
        static_capture.mkdir()
        static_members = {
            **members,
            "weights.safetensors": save({"rhs.weight": np.ones((72, 35), dtype=np.float32)}),
            "weights.safetensors.manifest.json": json.dumps({
                "0": {"kind": "input", "name": "lhs"},
                "1": {"kind": "param", "weight": "rhs.weight", "dtype": "float32",
                      "shape": [72, 35]},
            }).encode(),
            "input_order.json": b'{"lhs": 0}',
        }
        static_meta = json.loads(members["meta.json"])
        static_meta["input_abi"] = static_meta["input_abi"][:1]
        static_members["meta.json"] = json.dumps(static_meta).encode()
        for name, payload in static_members.items():
            (static_capture / name).write_bytes(payload)
        (static_capture / "capture_receipt.json").write_text(json.dumps({
            "schema": "m2m.capture-receipt.v1",
            "materialized_abi": {"complete": True, "inputs": 1, "lifted_constants": []},
            "lifted_constants": {},
            "artifacts": {name: {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
                          for name, payload in static_members.items()},
        }))
        static_staged = self.work / "source-bound-k72-static-staged"
        stage_compile_inputs(static_capture, static_staged, precision_policy=policy_path)
        static_common = [
            "--engine", "merlin_native", "--support", "atlas",
            "--snapshot", str(self.snapshot.root), "--compile-inputs", str(static_staged),
            "--diagnostic-region", "large", "--diagnostic-tile-index", "1",
            "--mode", "diagnostic",
        ]
        static_compiled = self.work / "source-bound-k72-static-cli"
        self.assertEqual(targetgen_main(["native-compile", *static_common, "--abi", str(abi),
                                         "--target-source", str(ROOT), "--out", str(static_compiled)]), 0)
        static_manifest = json.loads((static_compiled / "manifest.json").read_text())
        self.assertEqual([row["role"] for row in static_manifest["constant_inputs"]],
                         ["rhs__k0", "rhs__k1", "rhs__k2"])
        self.assertTrue(all(row["status"] == "packed" for row in static_manifest["constant_inputs"]))
        for index in range(3):
            self.assertEqual(
                (static_compiled / f"constants/rhs__k{index}.bin").read_bytes(),
                pack_source_operand(plan, f"rhs__k{index}", rhs),
            )
        static_receipt = self.work / "source-bound-k72-static-verification.json"
        self.assertEqual(targetgen_main(["native-verify", *static_common,
                                         "--artifact", str(static_compiled),
                                         "--out", str(static_receipt)]), 0)
        self.assertEqual(json.loads(static_receipt.read_text())["status"], "verified_artifacts")

        from tools.atlas_source_mn_host_data import prepare as prepare_source_mn

        grid = self.work / "source-bound-k72-grid"
        grid.mkdir()
        for index in range(2):
            tile_args = list(common)
            tile_args[tile_args.index("--diagnostic-tile-index") + 1] = str(index)
            destination = grid / f"tile-{index}"
            self.assertEqual(targetgen_main(["native-compile", *tile_args, "--abi", str(abi),
                                             "--target-source", str(ROOT),
                                             "--out", str(destination)]), 0)
            receipt_path = grid / f"tile-{index}-verification.json"
            self.assertEqual(targetgen_main(["native-verify", *tile_args,
                                             "--artifact", str(destination),
                                             "--out", str(receipt_path)]), 0)
        files, host_record = prepare_source_mn(
            grid, ROOT / "runtime/selected-host-map.json",
            Path(MERLIN) / "merlin/runtime/c/row_major_k_window.h",
            Path(MERLIN) / "merlin/runtime/c/finite_float_boundary.h",
        )
        self.assertEqual((host_record["m_rows"], host_record["n_rows"],
                          host_record["k_columns"], host_record["tile_count"]),
                         (3, 35, 72, 2))
        self.assertEqual(len(host_record["tiles"]), 2)
        self.assertIn(b"atlas_source_mn_encode_rhs", files["atlas_source_mn_data.h"])
        bundle = self.work / "source-bound-k72-host-data"
        bundle.mkdir()
        for name, payload in files.items():
            (bundle / name).write_bytes(payload)
        executable = self.work / "source-bound-k72-fake-host"
        subprocess.run([
            "cc", "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
            "-DATLAS_TEST_MN_SOURCE_GENERATED=1",
            "-DATLAS_USE_GENERATED_SOURCE_MN_DATA=1",
            "-DATLAS_DIAGNOSTIC_SOURCE_MN=1", f"-I{bundle}",
            *(str(bundle / name) for name in (
                "atlas_host.c", "atlas_ordered_k.c", "atlas_ordered_mn.c",
                "atlas_source_mn.c")),
            str(ROOT / "test/ordered-mn-host-test.c"), "-o", str(executable),
        ], check=True, capture_output=True, text=True)
        subprocess.run([str(executable)], check=True, capture_output=True, text=True)
        (bundle / "manifest.json").write_text(json.dumps(host_record, sort_keys=True) + "\n")
        if shutil.which("clang") and shutil.which("ld.lld"):
            host_build = self.work / "source-bound-k72-host-elf"
            build_result = subprocess.run([
                "bash", str(ROOT / "examples/handoff/source_mn/build.sh"),
                str(bundle), str(host_build),
            ], check=False, capture_output=True, text=True)
            self.assertEqual(build_result.returncode, 0, build_result.stderr)
            self.assertEqual((host_build / "atlas-source-mn.elf").read_bytes()[:4], b"\x7fELF")
        static_grid = self.work / "source-bound-k72-static-grid"
        static_grid.mkdir()
        for index in range(2):
            tile_args = [*static_common, "--diagnostic-tile-index", str(index)]
            artifact = static_grid / f"tile-{index}"
            self.assertEqual(targetgen_main(["native-compile", *tile_args, "--abi", str(abi),
                                             "--target-source", str(ROOT),
                                             "--out", str(artifact)]), 0)
            receipt_path = static_grid / f"tile-{index}-verification.json"
            self.assertEqual(targetgen_main(["native-verify", *tile_args,
                                             "--artifact", str(artifact),
                                             "--out", str(receipt_path)]), 0)
        with self.assertRaisesRegex(ValueError, "require --compile-inputs"):
            prepare_source_mn(
                static_grid, ROOT / "runtime/selected-host-map.json",
                Path(MERLIN) / "merlin/runtime/c/row_major_k_window.h",
                Path(MERLIN) / "merlin/runtime/c/finite_float_boundary.h",
            )
        static_files, static_record = prepare_source_mn(
            static_grid, ROOT / "runtime/selected-host-map.json",
            Path(MERLIN) / "merlin/runtime/c/row_major_k_window.h",
            Path(MERLIN) / "merlin/runtime/c/finite_float_boundary.h",
            staged_root=static_staged,
        )
        self.assertEqual(static_record["frozen_rhs"]["weight_key"], "rhs.weight")
        self.assertEqual(static_record["frozen_rhs"]["tensor_sha256"],
                         hashlib.sha256(np.ones((72, 35), dtype=np.float32).tobytes()).hexdigest())
        self.assertIn(b"#define ATLAS_SOURCE_MN_FROZEN_RHS 1", static_files["atlas_source_mn_data.h"])
        self.assertIn(b"atlas_source_mn_frozen_rhs_bits[2520]", static_files["atlas_source_mn_data.h"])
        weights_path = static_staged / "weights.safetensors"
        original_weights = weights_path.read_bytes()
        try:
            weights_path.write_bytes(original_weights + b"changed")
            with self.assertRaises((ValueError, OSError)):
                prepare_source_mn(
                    static_grid, ROOT / "runtime/selected-host-map.json",
                    Path(MERLIN) / "merlin/runtime/c/row_major_k_window.h",
                    Path(MERLIN) / "merlin/runtime/c/finite_float_boundary.h",
                    staged_root=static_staged,
                )
        finally:
            weights_path.write_bytes(original_weights)
        static_bundle = self.work / "source-bound-k72-static-host-data"
        static_bundle.mkdir()
        for name, payload in static_files.items():
            (static_bundle / name).write_bytes(payload)
        (static_bundle / "manifest.json").write_text(json.dumps(static_record, sort_keys=True) + "\n")

        from merlin.frontends.linalg_mlir import parse_mlir_text
        from merlin.targetgen.application_graph import _program_graph
        from merlin.targetgen.contract.model_host_bridge import diagnostic_host_bridge

        from tools.atlas_source_mn_model_bridge import (
            prepare as prepare_source_model_wrapper,
        )

        graph = _program_graph(parse_mlir_text(source), source_sha)
        (source_operation,) = [row for row in graph["operations"]
                               if row["mlir_operation"] == "linalg.matmul"]
        region = {
            "region_id": "large", "operation_id": source_operation["operation_id"],
            "status": "unsupported_semantics", "static_inputs_status": "not_emitted",
            "tiling_status": "isolated_tiles_emitted", "tile_count": 2,
        }
        model_probe = self.work / "source-bound-k72-model-probe"
        model_probe.mkdir()
        named_tiles = []
        for index in range(2):
            artifact = f"regions/named-0000-tile-{index:04d}"
            shutil.copytree(static_grid / f"tile-{index}", model_probe / artifact)
            tile_manifest = json.loads((static_grid / f"tile-{index}/manifest.json").read_text())
            named_tiles.append({
                "region_id": "large", "operation_id": source_operation["operation_id"],
                "tile_index": index, "status": "emitted", "artifact": artifact,
                "static_inputs_status": (
                    "packed_or_runtime"
                    if all(item["status"] == "packed" for item in tile_manifest["constant_inputs"])
                    else "blocked"
                ),
                "request_digest": tile_manifest["request_digest"],
                "binary_sha256": tile_manifest["binary_sha256"],
            })
        bridge, bridge_text = diagnostic_host_bridge(
            source, "forward", [region], named_tiles,
        )
        self.assertEqual(bridge["bindings"][0]["selected_route"], "tiled_native")
        (model_probe / "host-bridge.mlir").write_text(bridge_text)
        bridge.update({"path": "host-bridge.mlir",
                       "sha256": hashlib.sha256(bridge_text.encode()).hexdigest()})
        (model_probe / "manifest.json").write_text(json.dumps({
            "schema": "merlin.native_model_region_probe.v1",
            "engine": "merlin_native", "status": "incomplete",
            "source_operation_count": len(graph["operations"]),
            "target_identity": static_record["target_identity"],
            **{key: static_record["source_identity"][key] for key in (
                "original_source_sha256", "staged_program_sha256",
                "precision_policy_sha256",
            )},
            "regions": [region], "named_tiles": named_tiles, "host_bridge": bridge,
        }, sort_keys=True) + "\n")
        model_wrapper = prepare_source_model_wrapper(model_probe, "large", static_bundle)
        wrapper_dir = self.work / "source-bound-k72-model-wrapper"
        wrapper_dir.mkdir()
        for name, payload in model_wrapper["files"].items():
            (wrapper_dir / name).write_text(payload)
        wrapper_record = {
            **model_wrapper["record"],
            "files": {name: hashlib.sha256(payload.encode()).hexdigest()
                      for name, payload in model_wrapper["files"].items()},
        }
        (wrapper_dir / "manifest.json").write_text(json.dumps(wrapper_record, sort_keys=True) + "\n")
        self.assertTrue(model_wrapper["record"]["frozen_rhs"])
        self.assertEqual(len(model_wrapper["record"]["native_request_digests"]), 2)
        from tools.atlas_model_link import _check_native_reset_words, _selection

        probe_record = json.loads((model_probe / "manifest.json").read_text())
        source_count = probe_record["source_operation_count"]
        symbol = bridge["bindings"][0]["symbol"]
        host_object = self.work / "source-bound-k72-selected-host.o"
        host_object.write_bytes(b"selected outlined model host object")
        host_check = {
            "schema": "merlin.native_host_object_check.v1",
            "status": "object_symbols_checked",
            "selected_device_symbols": [symbol],
            "required_native_unmet": bridge["required_native_unmet"],
            "host_object_sha256": hashlib.sha256(host_object.read_bytes()).hexdigest(),
            "entry_dataflow": {
                "schema": "merlin.host_entry_dataflow_check.v2",
                "status": "source_to_host_body_checked",
                "operation_count": source_count,
                "native_external_call_count": 1,
                "host_defined_call_count": 0,
                "host_linalg_body_correspondence_checked_count": 0,
                "checked_operand_edges": 0,
                "initialization_semantics": "source_drop_and_host_initializer_checked",
                "native_reset_semantics": "not_checked",
                "native_zero_initialization_obligation_count": 1,
            },
            "source_route_accounting": {
                "schema": "merlin.model_source_route_accounting.v1",
                "operation_count": source_count,
                "state_counts": {
                    "native_external_unexecuted": 1,
                    "entry_support_numerics_unqualified": source_count - 2,
                    "model_output_unexecuted": 1,
                },
                "operations": [{"operation_ordinal": index}
                               for index in range(source_count)],
            },
            "entry_abi": {
                "source_input_count": 2, "source_result_count": 1,
                "input_descriptor_positions": [0, 1],
                "result_descriptor_positions": [2],
                "descriptor_types": ["memref<3x72xf32>", "memref<72x35xf32>",
                                     "memref<3x35xf32>"],
            },
        }
        selected = _selection(
            probe_record, host_check, host_object, [(wrapper_dir, wrapper_record)],
            probe_path=model_probe, source_bundles=[static_bundle],
        )
        self.assertEqual([row["kind"] for row in selected], ["source_grid"])
        self.assertEqual(len(_check_native_reset_words(model_probe, probe_record, host_check)), 2)
        with self.assertRaisesRegex(ValueError, "no selected host bundle"):
            _selection(probe_record, host_check, host_object,
                       [(wrapper_dir, wrapper_record)], probe_path=model_probe)
        clang = shutil.which("clang")
        llvm_nm = shutil.which("llvm-nm") or "/usr/lib/llvm-18/bin/llvm-nm"
        if clang and Path(llvm_nm).is_file():
            # A small host object checks RV64 symbol resolution; it does not
            # stand in for Merlin's source-to-entry lowering check.
            from argparse import Namespace

            from tools.atlas_model_link import build as build_model_link

            host_source = self.work / "source-bound-k72-link-stub.c"
            host_source.write_text(
                '#include "atlas_model_tile.h"\n'
                'extern void device(atlas_f32_memref2 *, const atlas_f32_memref2 *, '
                'const atlas_f32_memref2 *) __asm__("_mlir_ciface_' + symbol + '");\n'
                'void _mlir_ciface_forward(const atlas_f32_memref2 *lhs, '
                'const atlas_f32_memref2 *rhs, atlas_f32_memref2 *result) {\n'
                '  device(result, lhs, rhs);\n}\n'
            )
            subprocess.run([
                clang, "--target=riscv64-unknown-elf", "-march=rv64gcv",
                "-mabi=lp64d", "-ffreestanding", "-I", str(ROOT / "runtime"),
                "-c", str(host_source), "-o", str(host_object),
            ], check=True, capture_output=True, text=True)
            host_check["host_object_sha256"] = hashlib.sha256(host_object.read_bytes()).hexdigest()
            host_check["probe_manifest_sha256"] = hashlib.sha256(
                (model_probe / "manifest.json").read_bytes()).hexdigest()
            host_check_path = self.work / "source-bound-k72-link-host-check.json"
            host_check_path.write_text(json.dumps(host_check, sort_keys=True) + "\n")
            linked = build_model_link(Namespace(
                out=str(self.work / "source-bound-k72-model-linked"),
                probe=str(model_probe), host_check=str(host_check_path),
                host_object=str(host_object), wrapper=[str(wrapper_dir)],
                source_bundle=[str(static_bundle)],
                merlin_runtime_include=str(Path(MERLIN) / "merlin/runtime/c"),
                clang=clang, nm=llvm_nm,
            ))
            self.assertEqual(linked["status"], "relocatable_link_checked")
            self.assertEqual(len(linked["checked_native_reset_words"]), 2)
            self.assertEqual(linked["selected_wrappers"][0]["kind"], "source_grid")
        changed_model_tile = model_probe / "regions/named-0000-tile-0001/manifest.json"
        original_model_tile = changed_model_tile.read_bytes()
        try:
            changed_model_tile.write_bytes(original_model_tile + b"\n")
            with self.assertRaisesRegex(ValueError, "tile differs"):
                prepare_source_model_wrapper(model_probe, "large", static_bundle)
        finally:
            changed_model_tile.write_bytes(original_model_tile)
        changed_header = static_bundle / "atlas_source_mn_data.h"
        original_header = changed_header.read_bytes()
        try:
            changed_header.write_bytes(original_header + b"\n")
            with self.assertRaisesRegex(ValueError, "source host member changed"):
                prepare_source_model_wrapper(model_probe, "large", static_bundle)
        finally:
            changed_header.write_bytes(original_header)
        suffix = hashlib.sha256(bridge["bindings"][0]["symbol"].encode()).hexdigest()[:12]
        callback = f"atlas_source_mn_ciface_{suffix}"
        (wrapper_dir / "main.c").write_text(
            '#include "atlas_source_mn_model_wrapper.h"\n'
            '#include "atlas_source_mn_data.h"\n'
            '#include <assert.h>\n#include <stdlib.h>\n'
            'static unsigned calls;\n'
            'atlas_ordered_k_status atlas_source_mn_run_memref2('
            'const atlas_source_mn_memref_binding *binding, '
            'const atlas_f32_memref2 *lhs, const atlas_f32_memref2 *rhs, '
            'atlas_f32_memref2 *result, atlas_host_result *device) {\n'
            '  (void)lhs; (void)rhs; (void)result; (void)device;\n'
            '  assert(binding->frozen_rhs_bits != 0);\n'
            '  assert(binding->frozen_rhs_elements == 72u * 35u);\n'
            '  assert(binding->frozen_rhs_bits[0] == atlas_source_mn_frozen_rhs_bits[0]);\n'
            '  assert(binding->frozen_rhs_bits[72u * 35u - 1u] == '
            'atlas_source_mn_frozen_rhs_bits[72u * 35u - 1u]);\n'
            '  ++calls; return ATLAS_ORDERED_K_CONSTANT_MISMATCH;\n}\n'
            '__attribute__((noreturn)) void atlas_model_tile_fail('
            'atlas_model_tile_status status, atlas_host_result device) {\n'
            '  assert(status == ATLAS_MODEL_TILE_CONSTANT_MISMATCH);\n'
            '  assert(device == ATLAS_HOST_INVALID && calls == 2);\n'
            '  _Exit(42);\n}\n'
            'static int write_bytes(void *c, uintptr_t a, const uint8_t *p, size_t n) '
            '{ (void)c; (void)a; (void)p; (void)n; return 0; }\n'
            'static int read_bytes(void *c, uintptr_t a, uint8_t *p, size_t n) '
            '{ (void)c; (void)a; (void)p; (void)n; return 0; }\n'
            f'void {callback}(atlas_f32_memref2 *, const atlas_f32_memref2 *, '
            f'const atlas_f32_memref2 *) __asm__("_mlir_ciface_{bridge["bindings"][0]["symbol"]}");\n'
            'int main(void) {\n'
            '  atlas_host host = {0}; atlas_f32_memref2 input = {0}, result = {0};\n'
            '  atlas_host_result device = ATLAS_HOST_INVALID;\n'
            '  atlas_tile_memory_io io = {write_bytes, read_bytes, 0};\n'
            f'  assert(atlas_bind_source_mn_{suffix}(&host, io, 1) == 0);\n'
            f'  assert(atlas_source_mn_checked_{suffix}(&result, &input, &input, &device) '
            '== ATLAS_ORDERED_K_CONSTANT_MISMATCH && calls == 1);\n'
            f'  {callback}(&result, &input, &input);\n'
            '  return 99;\n}\n'
        )
        wrapper_executable = self.work / "source-bound-k72-model-wrapper-test"
        wrapper_build = subprocess.run([
            "cc", "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
            "-DATLAS_DIAGNOSTIC_SOURCE_MN=1", "-DATLAS_DIAGNOSTIC_ALLOW_UNBOUND=1",
            "-DATLAS_DIAGNOSTIC_ASSUME_COHERENT=1",
            "-DATLAS_DIAGNOSTIC_PRECISION_POLICY=1", f"-I{static_bundle}",
            f"-I{wrapper_dir}", str(wrapper_dir / "main.c"),
            str(wrapper_dir / "atlas_source_mn_model_wrapper.c"),
            "-o", str(wrapper_executable),
        ], check=False, capture_output=True, text=True)
        self.assertEqual(wrapper_build.returncode, 0, wrapper_build.stderr)
        wrapper_run = subprocess.run([str(wrapper_executable)], check=False,
                                     capture_output=True, text=True)
        self.assertEqual(wrapper_run.returncode, 42, wrapper_run.stderr)
        if shutil.which("clang"):
            rv64_object = self.work / "source-bound-k72-model-wrapper.o"
            rv64_build = subprocess.run([
                "clang", "--target=riscv64-unknown-elf", "-march=rv64gc", "-mabi=lp64d",
                "-mcmodel=medany", "-std=c11", "-O2", "-ffreestanding",
                "-Wall", "-Wextra", "-Werror", "-DATLAS_DIAGNOSTIC_SOURCE_MN=1",
                "-DATLAS_DIAGNOSTIC_ALLOW_UNBOUND=1",
                "-DATLAS_DIAGNOSTIC_ASSUME_COHERENT=1",
                "-DATLAS_DIAGNOSTIC_PRECISION_POLICY=1", f"-I{static_bundle}",
                f"-I{wrapper_dir}", "-c",
                str(wrapper_dir / "atlas_source_mn_model_wrapper.c"),
                "-o", str(rv64_object),
            ], check=False, capture_output=True, text=True)
            self.assertEqual(rv64_build.returncode, 0, rv64_build.stderr)
            self.assertEqual(struct.unpack_from("<H", rv64_object.read_bytes(), 18)[0], 243)
        static_executable = self.work / "source-bound-k72-static-fake-host"
        compile_result = subprocess.run([
            "cc", "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
            "-DATLAS_TEST_MN_SOURCE_GENERATED=1",
            "-DATLAS_TEST_MN_MEMREF=1",
            "-DATLAS_USE_GENERATED_SOURCE_MN_DATA=1",
            "-DATLAS_DIAGNOSTIC_SOURCE_MN=1", f"-I{static_bundle}", f"-I{ROOT / 'runtime'}",
            *(str(static_bundle / name) for name in (
                "atlas_host.c", "atlas_ordered_k.c", "atlas_ordered_mn.c",
                "atlas_source_mn.c")),
            str(ROOT / "runtime/atlas_source_mn_memref.c"),
            str(ROOT / "test/ordered-mn-host-test.c"), "-o", str(static_executable),
        ], check=False, capture_output=True, text=True)
        self.assertEqual(compile_result.returncode, 0, compile_result.stderr)
        run_result = subprocess.run([str(static_executable)], check=False,
                                    capture_output=True, text=True)
        self.assertEqual(run_result.returncode, 0, run_result.stderr)
        if shutil.which("clang") and shutil.which("ld.lld"):
            static_host_build = self.work / "source-bound-k72-static-host-elf"
            build_result = subprocess.run([
                "bash", str(ROOT / "examples/handoff/source_mn/build.sh"),
                str(static_bundle), str(static_host_build),
            ], check=False, capture_output=True, text=True)
            self.assertEqual(build_result.returncode, 0, build_result.stderr)
            self.assertEqual((static_host_build / "atlas-source-mn.elf").read_bytes()[:4], b"\x7fELF")
        changed = grid / "tile-1/precision-plan.json"
        original = changed.read_bytes()
        try:
            changed.write_bytes(original + b"\n")
            with self.assertRaisesRegex(ValueError, "unverified or changed"):
                prepare_source_mn(
                    grid, ROOT / "runtime/selected-host-map.json",
                    Path(MERLIN) / "merlin/runtime/c/row_major_k_window.h",
                    Path(MERLIN) / "merlin/runtime/c/finite_float_boundary.h",
                )
        finally:
            changed.write_bytes(original)
        changed_stage = grid / "tile-1/atlas-machine.mlir"
        original_stage = changed_stage.read_bytes()
        try:
            changed_stage.write_bytes(original_stage + b"\n")
            with self.assertRaisesRegex(ValueError, "stage changed"):
                prepare_source_mn(
                    grid, ROOT / "runtime/selected-host-map.json",
                    Path(MERLIN) / "merlin/runtime/c/row_major_k_window.h",
                    Path(MERLIN) / "merlin/runtime/c/finite_float_boundary.h",
                )
        finally:
            changed_stage.write_bytes(original_stage)
        (grid / "tile-0-verification.json").rename(grid / "missing-receipt.json")
        with self.assertRaisesRegex(ValueError, "verification receipts are missing"):
            prepare_source_mn(
                grid, ROOT / "runtime/selected-host-map.json",
                Path(MERLIN) / "merlin/runtime/c/row_major_k_window.h",
                Path(MERLIN) / "merlin/runtime/c/finite_float_boundary.h",
            )

    def test_finite_mxu1_candidate_is_separate_and_emits_scaled_tile(self) -> None:
        import numpy as np
        from atlas_mlir_native.binding import (
            MXU1_FINITE_POLICY,
            diagnostic_mxu1_finite_f32_profile,
        )
        from merlin.frontends.compile_inputs import StagedCompileInputs
        from merlin.semantic_compiler.search import SearchLimits
        from merlin.semantic_compiler.target_binding import (
            NativeCompilationError,
        )

        examples = ROOT / "test/examples"
        source = (examples / "captured_attention_mlp32_linalg.mlir").read_bytes()
        source_sha = hashlib.sha256(source).hexdigest()

        def staged(policy_bytes: bytes) -> StagedCompileInputs:
            return StagedCompileInputs(
                source.decode(), "forward", json.loads(policy_bytes), policy_bytes,
                source_sha, source_sha, hashlib.sha256(policy_bytes).hexdigest(),
            )

        old_policy = (examples / "atlas_attention_mlp_source_bound_policy.json").read_bytes()
        with self.assertRaisesRegex(NativeCompilationError, "no qualified tile profile"):
            self.binding.prepare_precision_tile(staged(old_policy), "matmul_5")

        policy_bytes = (examples / "atlas_attention_mlp_finite_anchor_candidate.json").read_bytes()
        compiler_inputs = staged(policy_bytes)
        unit_plan = self.binding.prepare_precision_tile(compiler_inputs, "matmul_1")
        scaled_plan = self.binding.prepare_precision_tile(compiler_inputs, "matmul_5")
        self.assertEqual(unit_plan.request.node("result").type.numerical_policy, MXU1_FINITE_POLICY)
        self.assertEqual(scaled_plan.request.node("result").type.numerical_policy, MXU1_FINITE_POLICY)
        self.assertEqual(tuple(item.spec.pre_encode_scale_exponent for item in scaled_plan.inputs),
                         (5, -5))
        self.assertEqual(
            diagnostic_mxu1_finite_f32_profile(122).conversion_specs[1].encode(
                np.array([0.5], dtype=np.float32)
            ).tolist(), [0x08],
        )
        for value in (-0.0, -0.001, float("nan")):
            with self.assertRaises(ValueError):
                unit_plan.inputs[0].spec.encode(np.array([value], dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "scale code"):
            diagnostic_mxu1_finite_f32_profile(121)

        altered = json.loads(policy_bytes)
        altered["mxu1_reset_arithmetic"] = "exact-sum"
        with self.assertRaisesRegex(ValueError, "mxu1_reset_arithmetic.*differs"):
            self.binding.prepare_precision_tile(staged(json.dumps(altered).encode()), "matmul_5")
        altered["schema"] = {"unexpected": "object"}
        with self.assertRaisesRegex(NativeCompilationError, "policy schema differs"):
            self.binding.prepare_precision_tile(staged(json.dumps(altered).encode()), "matmul_5")

        destination = self.work / "finite-mxu1-scaled"
        self.compile_verified(
            scaled_plan.request, fixed_inputs={"lhs": 3, "rhs": 4},
            fixed_outputs=(35,),
            destination=destination, limits=SearchLimits(),
        )
        execution = json.loads((destination / "execution_plan.json").read_text())
        self.assertEqual((execution["operation"], execution["numerical_policy"]),
                         ("mxu1_reset", MXU1_FINITE_POLICY))
        self.assertFalse(execution["runtime_domain_guard"])

    def test_positive_zero_boundary_is_a_separate_source_policy(self) -> None:
        import numpy as np
        from merlin.frontends.compile_inputs import StagedCompileInputs
        from merlin.semantic_compiler.target_binding import NativeCompilationError

        examples = ROOT / "test/examples"
        source = (examples / "captured_attention_mlp32_linalg.mlir").read_bytes()
        source_sha = hashlib.sha256(source).hexdigest()
        original = json.loads((examples / "atlas_attention_mlp_source_bound_policy.json").read_bytes())

        def staged(document: dict) -> StagedCompileInputs:
            data = json.dumps(document, sort_keys=True).encode()
            return StagedCompileInputs(
                source.decode(), "forward", document, data,
                source_sha, source_sha, hashlib.sha256(data).hexdigest(),
            )

        old = self.binding.prepare_precision_tile(staged(original), "matmul_0")
        selected = copy.deepcopy(original)
        selected["schema"] = "atlas.oot.quantized_linalg_policy.v5"
        selected["zero_sign_policy"] = "canonicalize_encoded_negative_zero"
        candidate = self.binding.prepare_precision_tile(staged(selected), "matmul_0")
        self.assertNotEqual(old.request.digest(), candidate.request.digest())
        self.assertNotEqual(old.precision_policy_sha256, candidate.precision_policy_sha256)
        self.assertFalse(old.inputs[0].spec.canonicalize_negative_zero)
        self.assertTrue(candidate.inputs[0].spec.canonicalize_negative_zero)
        self.assertEqual(candidate.inputs[0].spec.encode(
            np.asarray([-0.0, -2.0**-20, -1.0], dtype=np.float32)
        ).tolist(), [0, 0, 0xB8])
        with self.assertRaisesRegex(NativeCompilationError, "currently needs MXU0"):
            self.binding.prepare_precision_tile(staged(selected), "matmul_1")
        del selected["zero_sign_policy"]
        with self.assertRaisesRegex(ValueError, "missing or unknown fields"):
            self.binding.prepare_precision_tile(staged(selected), "matmul_0")

    def test_mixed_positive_zero_policy_keeps_both_mxu_arithmetic_contracts(self) -> None:
        import numpy as np
        from merlin.frontends.compile_inputs import StagedCompileInputs
        from merlin.semantic_compiler.search import SearchLimits

        examples = ROOT / "test/examples"
        source = (examples / "captured_attention_mlp32_linalg.mlir").read_bytes()
        source_sha = hashlib.sha256(source).hexdigest()
        baseline = json.loads((examples / "atlas_attention_mlp_finite_anchor_candidate.json").read_bytes())
        selected = json.loads((examples / "atlas_attention_mlp_mixed_positive_zero_candidate.json").read_bytes())

        def staged(document: dict) -> StagedCompileInputs:
            data = json.dumps(document, sort_keys=True).encode()
            return StagedCompileInputs(
                source.decode(), "forward", document, data,
                source_sha, source_sha, hashlib.sha256(data).hexdigest(),
            )

        assert selected["source_sha256"] == source_sha
        assert selected["matmul_regions"] == baseline["matmul_regions"]
        mxu1_candidate = None
        for region in selected["matmul_regions"]:
            old = self.binding.prepare_precision_tile(staged(baseline), region)
            candidate = self.binding.prepare_precision_tile(staged(selected), region)
            self.assertNotEqual(old.request.digest(), candidate.request.digest())
            self.assertEqual(old.request.nodes[-1].type.numerical_policy,
                             candidate.request.nodes[-1].type.numerical_policy)
            self.assertEqual(candidate.region_id, region)
            for operand in candidate.inputs:
                self.assertTrue(operand.spec.canonicalize_negative_zero)
                self.assertEqual(operand.spec.encode(
                    np.asarray([-0.0, -2.0**-30], dtype=np.float32)
                ).tolist(), [0, 0])
            if region == "matmul_1":
                mxu1_candidate = candidate

        self.assertIsNotNone(mxu1_candidate)
        destination = self.work / "mixed-positive-zero-mxu1"
        self.compile_verified(
            mxu1_candidate.request, fixed_inputs={"lhs": 3, "rhs": 4},
            fixed_outputs=(35,),
            destination=destination, limits=SearchLimits(),
        )
        self.assertEqual(json.loads((destination / "execution_plan.json").read_text())["operation"],
                         "mxu1_reset")

        missing = copy.deepcopy(selected)
        del missing["zero_sign_policy"]
        with self.assertRaisesRegex(ValueError, "missing or unknown fields"):
            self.binding.prepare_precision_tile(staged(missing), "matmul_1")
        changed = copy.deepcopy(selected)
        changed["mxu1_reset_arithmetic"] = "exact-sum"
        with self.assertRaisesRegex(ValueError, "mxu1_reset_arithmetic.*differs"):
            self.binding.prepare_precision_tile(staged(changed), "matmul_1")

    def test_staged_source_bound_tile_runs_through_installed_native_cli(self) -> None:
        import numpy as np
        from atlas_mlir_native.binding import (
            BF16_DRAM_PAIR_LAYOUT,
            FP8_DRAM_TILE_LAYOUT,
            RTL_REVISION,
        )
        from atlas_mlir_native.precision_abi import (
            decode_source_result,
            pack_source_operand,
        )
        from merlin.frontends.compile_inputs import (
            load_compile_inputs,
            stage_compile_inputs,
        )
        from merlin.semantic_compiler.precision_constants import (
            load_declared_static_operand,
        )
        from merlin.targetgen.cli import main as targetgen_main

        examples = ROOT / "test/examples"
        source = (examples / "captured_attention_mlp32_linalg.mlir").read_bytes()
        source_digest = hashlib.sha256(source).hexdigest()
        trace = json.dumps({"status": "complete", "mlir": {"sha256": source_digest}}).encode()
        capture = self.work / "source-bound-capture"
        capture.mkdir()
        members = {
            "model.mlir": source,
            "weights.safetensors": (examples / "captured_attention_mlp32_weights.safetensors").read_bytes(),
            "weights.safetensors.manifest.json": (
                examples / "captured_attention_mlp32_arguments.json"
            ).read_bytes(),
            "input_order.json": b'{"tokens": 0}',
            "frontend-trace.json": trace,
            "meta.json": json.dumps({
                "ok": True, "opaque": 0,
                "weights": "captured_attention_mlp32_weights.safetensors",
                "frontend_trace": {"sha256": hashlib.sha256(trace).hexdigest()},
                "func_name": "forward",
                "input_abi": [{"shape": [32, 32], "dtype": "f32"}],
                "output_abi": [{"shape": [32, 32], "dtype": "f32"}],
                "dtype": "f32",
            }).encode(),
            "golden.npy": b"private evaluator-only bytes",
        }
        for name, data in members.items():
            (capture / name).write_bytes(data)
        (capture / "capture_receipt.json").write_text(json.dumps({
            "schema": "m2m.capture-receipt.v1",
            "materialized_abi": {"complete": True, "inputs": 1, "lifted_constants": []},
            "lifted_constants": {},
            "artifacts": {name: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
                          for name, data in members.items()},
        }))
        staged = self.work / "source-bound-staged"
        stage_compile_inputs(
            capture, staged, precision_policy=examples / "atlas_attention_mlp_source_bound_policy.json",
        )
        self.assertFalse((staged / "golden.npy").exists())
        compiler_inputs = load_compile_inputs(staged)
        query_plan = self.binding.prepare_precision_tile(compiler_inputs, "matmul_0")
        self.assertIsNone(load_declared_static_operand(staged, query_plan, "lhs"))
        query_weight = load_declared_static_operand(staged, query_plan, "rhs")
        self.assertEqual((query_weight.argument_index, query_weight.weight_key, query_weight.views),
                         (0, "query.weight", ((1, 0),)))
        packed_weight = pack_source_operand(query_plan, "rhs", query_weight.value)
        self.assertEqual((len(packed_weight), packed_weight.count(0x30)), (1024, 32))
        tampered = self.work / "source-bound-tampered"
        shutil.copytree(staged, tampered)
        with (tampered / "weights.safetensors").open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaisesRegex(ValueError, "differs from manifest"):
            load_declared_static_operand(tampered, query_plan, "rhs")
        key_plan = self.binding.prepare_precision_tile(compiler_inputs, "matmul_1")
        key_weight = load_declared_static_operand(staged, key_plan, "rhs")
        with self.assertRaisesRegex(ValueError, "input domain"):
            pack_source_operand(key_plan, "rhs", key_weight.value)
        abi = self.work / "source-bound-abi.json"
        abi.write_text(json.dumps({"fixed_inputs": {"lhs": 3, "rhs": 4}, "fixed_outputs": [35]}))
        common = ["--engine", "merlin_native", "--support", "atlas",
                  "--snapshot", str(self.snapshot.root), "--compile-inputs", str(staged),
                  "--diagnostic-region", "matmul_0", "--mode", "diagnostic"]
        selection = self.work / "source-bound-selection.json"
        self.assertEqual(targetgen_main(["native-select", *common, "--abi", str(abi),
                                         "--out", str(selection)]), 0)
        selected = json.loads(selection.read_text())
        self.assertEqual(selected["status"], "selected")
        self.assertEqual(selected["precision_plan"]["quality_status"], "unqualified")
        output = self.work / "source-bound-cli-program"
        self.assertEqual(targetgen_main(["native-compile", *common, "--abi", str(abi),
                                         "--target-source", str(ROOT), "--out", str(output)]), 0)
        plan = json.loads((output / "precision-plan.json").read_text())
        self.assertEqual(plan["request_digest"], selected["request_digest"])
        native_manifest = json.loads((output / "manifest.json").read_text())
        self.assertEqual(native_manifest["stage_artifacts"]
                         ["precision-plan.json"], hashlib.sha256((output / "precision-plan.json").read_bytes()).hexdigest())
        self.assertEqual(native_manifest["constant_inputs"][0]["status"], "packed")
        self.assertEqual(native_manifest["constant_inputs"][0]["address"],
                         json.loads((output / "execution_plan.json").read_text())["weight_address"])
        self.assertEqual((output / "constants/rhs.bin").read_bytes(), packed_weight)
        verified = self.work / "source-bound-verification.json"
        self.assertEqual(targetgen_main(["native-verify", *common, "--artifact", str(output),
                                         "--out", str(verified)]), 0)
        self.assertEqual(json.loads(verified.read_text())["status"], "verified_artifacts")
        exact_layouts = self.work / "source-bound-exact-layouts.json"
        exact_layouts.write_text(json.dumps({
            "schema": "merlin.physical_boundary_layouts.v2",
            "target_identity": query_plan.request.target_identity,
            "source_identity": query_plan.request.source_identity,
            "request_digest": query_plan.request.digest(),
            "physical_extents": dict(query_plan.request.physical_extents),
            "boundary_layouts": {"lhs": FP8_DRAM_TILE_LAYOUT, "rhs": FP8_DRAM_TILE_LAYOUT},
            "output_layouts": [BF16_DRAM_PAIR_LAYOUT],
        }))
        exact_common = [*common, "--physical-layouts", str(exact_layouts)]
        exact_output = self.work / "source-bound-exact-layout-program"
        self.assertEqual(targetgen_main(["native-compile", *exact_common, "--abi", str(abi),
                                         "--target-source", str(ROOT), "--out", str(exact_output)]), 0)
        self.assertEqual((exact_output / "program.bin").read_bytes(), (output / "program.bin").read_bytes())
        self.assertEqual(json.loads((exact_output / "execution_plan.json").read_text())
                         ["output_physical_layouts"], ["bf16_pair_halves"])
        exact_verified = self.work / "source-bound-exact-layout-verification.json"
        self.assertEqual(targetgen_main(["native-verify", *exact_common,
                                         "--artifact", str(exact_output),
                                         "--out", str(exact_verified)]), 0)
        self.assertEqual(json.loads(exact_verified.read_text())["status"], "verified_artifacts")
        wrong_layouts = self.work / "source-bound-wrong-layouts.json"
        wrong_layouts.write_text(json.dumps({
            **json.loads(exact_layouts.read_text()),
            "boundary_layouts": {"lhs": FP8_DRAM_TILE_LAYOUT, "rhs": BF16_DRAM_PAIR_LAYOUT},
        }))
        wrong_selection = self.work / "source-bound-wrong-layout-selection.json"
        self.assertEqual(targetgen_main(["native-select", *common,
                                         "--physical-layouts", str(wrong_layouts),
                                         "--abi", str(abi), "--out", str(wrong_selection)]), 2)
        self.assertNotEqual(json.loads(wrong_selection.read_text())["status"], "selected")
        from tools.atlas_boot_pack import pack_direct
        from tools.atlas_captured_host_data import render as render_host_data

        layout = self.work / "source-bound-boot-layout.json"
        layout.write_text(json.dumps({
            "schema": "atlas.reset-entry-layout.v1",
            "selected_rtl_revision": RTL_REVISION,
            "dram_window_base": 0x90000000,
            "regions": [
                {"name": "inputs", "kind": "input", "address": 0x90001800,
                 "size_bytes": 4096},
                {"name": "output", "kind": "output", "address": 0x90011800,
                 "size_bytes": 2048},
                {"name": "output_guard", "kind": "guard", "address": 0x90012000,
                 "size_bytes": 32},
            ],
        }))
        capsule = self.work / "source-bound-boot-capsule"
        boot = pack_direct(output / "atlas-machine.mlir", Path(os.environ["ATLAS_OOT_BIN_DIR"]) / "atlas-emit",
                           layout, capsule)
        self.assertEqual(boot["program_sha256"], native_manifest["binary_sha256"])
        header = render_host_data(output, capsule, native_verification=verified)
        self.assertIn("#define ATLAS_CAPTURED_INPUT_ADDRESS 0x90001800u", header)
        self.assertIn("{0x90002000u, 1024u, atlas_constant_0}", header)
        self.assertIn("atlas_pack_runtime_input_f32", header)
        self.assertIn("atlas_unpack_output_bf16_f32", header)
        self.assertIn("112u, 0x3c800000u", header)
        missing_merlin_header = self.work / "source-bound-missing-merlin.h"
        missing_merlin = subprocess.run(
            [sys.executable, "-I", "-S", str(ROOT / "tools/atlas_captured_host_data.py"),
             "--compiled", str(output), "--capsule", str(capsule),
             "--native-verification", str(verified), "--out", str(missing_merlin_header)],
            capture_output=True, text=True, check=False,
        )
        self.assertNotEqual(missing_merlin.returncode, 0)
        self.assertEqual(json.loads(missing_merlin.stderr)["status"], "FAIL")
        self.assertFalse(missing_merlin_header.exists())
        (self.work / "atlas_captured_data.h").write_text(header)
        host_source = self.work / "source-bound-host-conversion.c"
        host_source.write_text(
            '#include "atlas_captured_data.h"\n'
            'int pack(const float *source, uint8_t *output) {\n'
            '  return atlas_pack_runtime_input_f32(source, output);\n'
            '}\n'
            'int unpack(const volatile uint8_t *source, float *output) {\n'
            '  return atlas_unpack_output_bf16_f32(source, output);\n'
            '}\n'
        )
        host_library = self.work / "source-bound-host-conversion.so"
        subprocess.run(
            ["cc", "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "-fPIC", "-shared",
             "-I", str(Path(MERLIN) / "merlin/runtime/c"), "-I", str(self.work),
             str(host_source), "-o", str(host_library)],
            check=True, capture_output=True, text=True,
        )
        pack = ctypes.CDLL(str(host_library)).pack
        pack.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_uint8)]
        pack.restype = ctypes.c_int
        runtime_values = np.full((32, 32), 0.5, dtype=np.float32)
        output_bytes = (ctypes.c_uint8 * 1024)()
        self.assertEqual(pack(runtime_values.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                              output_bytes), 0)
        self.assertEqual(bytes(output_bytes), pack_source_operand(query_plan, "lhs", runtime_values))
        runtime_values[0, 0] = np.float32(-0.0)
        unchanged = (ctypes.c_uint8 * 1024)(*([0xA5] * 1024))
        self.assertEqual(pack(runtime_values.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                              unchanged), 2)
        self.assertEqual(bytes(unchanged), b"\xA5" * 1024)
        readback = ctypes.CDLL(str(host_library)).unpack
        readback.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_float)]
        readback.restype = ctypes.c_int
        rng = np.random.default_rng(0xA71A5)
        codes = rng.integers(0, 1 << 16, size=(32, 32), dtype=np.uint16)
        codes[0, :4] = [0, 0x8000, 0x7F80, 0x7FC1]
        physical = np.concatenate((codes[:, :16].ravel(), codes[:, 16:].ravel()))
        payload = physical.astype("<u2").tobytes()
        source = (ctypes.c_uint8 * len(payload)).from_buffer_copy(payload)
        decoded = (ctypes.c_float * 1024)()
        self.assertEqual(readback(source, decoded), 0)
        self.assertEqual(ctypes.string_at(ctypes.addressof(decoded), 4096),
                         decode_source_result(query_plan, payload).tobytes())
        self.assertEqual(bytes(source), payload)
        wrong_readback = self.work / "source-bound-wrong-readback"
        shutil.copytree(output, wrong_readback)
        changed_plan = json.loads((wrong_readback / "precision-plan.json").read_text())
        changed_plan["output"]["target_axes_from_source"] = [1, 0]
        changed_plan_bytes = json.dumps(changed_plan).encode()
        (wrong_readback / "precision-plan.json").write_bytes(changed_plan_bytes)
        changed_manifest = json.loads((wrong_readback / "manifest.json").read_text())
        changed_manifest["stage_artifacts"]["precision-plan.json"] = hashlib.sha256(
            changed_plan_bytes
        ).hexdigest()
        changed_manifest_bytes = json.dumps(changed_manifest).encode()
        (wrong_readback / "manifest.json").write_bytes(changed_manifest_bytes)
        forged_receipt = json.loads(verified.read_text())
        forged_receipt["artifact"] = str(wrong_readback)
        forged_receipt["manifest_sha256"] = hashlib.sha256(changed_manifest_bytes).hexdigest()
        forged_receipt_path = self.work / "source-bound-forged-readback.json"
        forged_receipt_path.write_text(json.dumps(forged_receipt))
        with self.assertRaisesRegex(ValueError, "exact BF16-to-f32 readback"):
            render_host_data(wrong_readback, capsule, native_verification=forged_receipt_path)
        stale_receipt = self.work / "source-bound-stale-verification.json"
        changed_receipt = json.loads(verified.read_text())
        changed_receipt["request_digest"] = "0" * 64
        stale_receipt.write_text(json.dumps(changed_receipt))
        with self.assertRaisesRegex(ValueError, "verification does not match"):
            render_host_data(output, capsule, native_verification=stale_receipt)
        manifest_path = output / "manifest.json"
        original_manifest = manifest_path.read_bytes()
        changed_manifest = json.loads(original_manifest)
        changed_manifest["constant_inputs"][0]["source"]["weight_key"] = "other.weight"
        manifest_path.write_text(json.dumps(changed_manifest))
        with self.assertRaisesRegex(ValueError, "verification does not match"):
            render_host_data(output, capsule, native_verification=verified)
        manifest_path.write_bytes(original_manifest)
        (output / "constants/rhs.bin").write_bytes(b"\x00" * len(packed_weight))
        with self.assertRaisesRegex(ValueError, "constant payload differs"):
            render_host_data(output, capsule, native_verification=verified)
        changed_constant = self.work / "source-bound-changed-constant.json"
        self.assertEqual(targetgen_main(["native-verify", *common, "--artifact", str(output),
                                         "--out", str(changed_constant)]), 2)
        self.assertEqual(json.loads(changed_constant.read_text())["status"], "compile_error")
        (output / "constants/rhs.bin").write_bytes(packed_weight)
        mxu1 = self.work / "source-bound-mxu1-selection.json"
        self.assertEqual(targetgen_main(["native-select", *common[:-4],
                                         "--diagnostic-region", "matmul_1", "--mode", "diagnostic",
                                         "--abi", str(abi), "--out", str(mxu1)]), 0)
        self.assertEqual(json.loads(mxu1.read_text())["status"], "selected")
        mxu1_common = [*common[:-4], "--diagnostic-region", "matmul_1", "--mode", "diagnostic"]
        mxu1_output = self.work / "source-bound-mxu1-program"
        self.assertEqual(targetgen_main(["native-compile", *mxu1_common,
                                         "--abi", str(abi), "--target-source", str(ROOT),
                                         "--out", str(mxu1_output)]), 0)
        mxu1_plan = json.loads((mxu1_output / "execution_plan.json").read_text())
        self.assertEqual(
            (mxu1_plan["operation"], mxu1_plan["numerical_policy"],
             mxu1_plan["input_domain"]),
            ("mxu1_reset", "mxu1-e4m3-anchor-bf16-bounded-v1",
             "e4m3-positive-zero-or-signed-one-two-v1"),
        )
        mxu1_constants = json.loads((mxu1_output / "manifest.json").read_text())["constant_inputs"]
        self.assertEqual((len(mxu1_constants), mxu1_constants[0]["status"]), (1, "blocked"))
        self.assertFalse((mxu1_output / "constants/rhs.bin").exists())
        self.assertEqual(targetgen_main(["native-verify", *mxu1_common,
                                         "--artifact", str(mxu1_output),
                                         "--out", str(self.work / "source-bound-mxu1-verified.json")]), 0)
        with self.assertRaisesRegex(ValueError, "constant is blocked"):
            render_host_data(mxu1_output, capsule,
                             native_verification=self.work / "source-bound-mxu1-verified.json")
        scaled_common = [*common[:-4], "--diagnostic-region", "matmul_6", "--mode", "diagnostic"]
        scaled_plan = self.binding.prepare_precision_tile(compiler_inputs, "matmul_6")
        self.assertEqual(tuple(item.spec.pre_encode_scale_exponent for item in scaled_plan.inputs),
                         (5, -5))
        scaled_output = self.work / "source-bound-scaled-mxu0"
        self.assertEqual(targetgen_main(["native-compile", *scaled_common,
                                         "--abi", str(abi), "--target-source", str(ROOT),
                                         "--out", str(scaled_output)]), 0)
        scaled_manifest = json.loads((scaled_output / "manifest.json").read_text())
        self.assertEqual(scaled_manifest["constant_inputs"][0]["status"], "packed")
        scaled_weight = load_declared_static_operand(staged, scaled_plan, "rhs")
        self.assertEqual((scaled_output / "constants/rhs.bin").read_bytes(),
                         pack_source_operand(scaled_plan, "rhs", scaled_weight.value))
        self.assertEqual((scaled_output / "constants/rhs.bin").read_bytes().count(0x08), 32)
        scaled_verified = self.work / "source-bound-scaled-verification.json"
        self.assertEqual(targetgen_main(["native-verify", *scaled_common,
                                         "--artifact", str(scaled_output),
                                         "--out", str(scaled_verified)]), 0)
        scaled_capsule = self.work / "source-bound-scaled-capsule"
        pack_direct(scaled_output / "atlas-machine.mlir",
                    Path(os.environ["ATLAS_OOT_BIN_DIR"]) / "atlas-emit", layout, scaled_capsule)
        scaled_header = render_host_data(scaled_output, scaled_capsule,
                                         native_verification=scaled_verified)
        self.assertIn("atlas_pack_runtime_input_0_f32", scaled_header)
        self.assertIn("atlas_pack_runtime_input_1_f32", scaled_header)
        self.assertNotIn("atlas_pack_runtime_input_f32(", scaled_header)
        scaled_host = self.work / "scaled-host"
        scaled_host.mkdir()
        (scaled_host / "atlas_captured_data.h").write_text(scaled_header)
        (scaled_host / "conversion.c").write_text(
            '#include "atlas_captured_data.h"\n'
            'int pack_lhs(const float *src, uint8_t *dst) {\n'
            '  return atlas_pack_runtime_input_0_f32(src, dst);\n}\n'
            'int pack_rhs(const float *src, uint8_t *dst) {\n'
            '  return atlas_pack_runtime_input_1_f32(src, dst);\n}\n'
        )
        subprocess.run(
            ["cc", "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "-fPIC", "-shared",
             "-I", str(Path(MERLIN) / "merlin/runtime/c"), "-I", str(scaled_host),
             str(scaled_host / "conversion.c"), "-o", str(scaled_host / "conversion.so")],
            check=True, capture_output=True, text=True,
        )
        scale_library = ctypes.CDLL(str(scaled_host / "conversion.so"))
        half = np.full((32, 32), 0.5, dtype=np.float32)
        for role, expected in (("lhs", 0x58), ("rhs", 0x08)):
            packed = (ctypes.c_uint8 * 1024)()
            function = getattr(scale_library, f"pack_{role}")
            function.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_uint8)]
            function.restype = ctypes.c_int
            self.assertEqual(function(half.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), packed), 0)
            self.assertEqual(bytes(packed), pack_source_operand(scaled_plan, role, half))
            self.assertEqual(packed[0], expected)
        refused = self.work / "scaled-mxu1-refused.json"
        self.assertEqual(targetgen_main(["native-select", *common[:-4],
                                         "--diagnostic-region", "matmul_5", "--mode", "diagnostic",
                                         "--abi", str(abi), "--out", str(refused)]), 2)
        refused_status = json.loads(refused.read_text())
        self.assertEqual(refused_status["status"], "unqualified_target")
        self.assertIn("no qualified tile profile", refused_status["reason"])
        strict = self.work / "source-bound-strict-refused.json"
        self.assertEqual(targetgen_main(["native-select", *common[:-2], "--mode", "strict-native",
                                         "--abi", str(abi), "--out", str(strict)]), 2)
        self.assertEqual(json.loads(strict.read_text())["status"], "unsupported_semantics")
        (output / "precision-plan.json").write_text("{}\n")
        changed = self.work / "source-bound-changed-verification.json"
        self.assertEqual(targetgen_main(["native-verify", *common, "--artifact", str(output),
                                         "--out", str(changed)]), 2)
        self.assertEqual(json.loads(changed.read_text())["status"], "compile_error")

    def test_native_mxu0_result_feeds_bf16_relu(self) -> None:
        from dataclasses import replace

        from atlas_mlir_native.binding import BF16_RELU_ATTRS, check_mxu0_stream
        from merlin.semantic_compiler.model import (
            KernelRequest,
            SemanticNode,
            TensorType,
            pointwise_identity_maps,
        )
        from merlin.semantic_compiler.search import SearchLimits
        from merlin.semantic_compiler.target_binding import (
            NativeCompilationError,
        )

        base = self.mxu0_request()
        relu = SemanticNode(
            "z", "relu", ("y",), TensorType((32, 32), "bf16", "raw-bit-pattern"),
            attrs=BF16_RELU_ATTRS, index_maps=pointwise_identity_maps(2, 1),
        )
        request = replace(base, nodes=(*base.nodes, relu), outputs=("z",))
        fixture = KernelRequest.from_record(json.loads(
            (ROOT / "test/examples/native_mxu0_relu_request.json").read_text()
        ))
        self.assertEqual(fixture.digest(), request.digest())
        destination = self.work / "mxu0-relu"
        manifest = self.compile_verified(
            request, fixed_inputs={"x": 3, "w": 4},
            fixed_outputs=(35,),
            destination=destination, limits=SearchLimits(),
        )
        program = json.loads((destination / "physical-program.json").read_text())
        pair, relu_pair = manifest["selected_registers"][2:4], manifest["selected_registers"][4:6]
        self.assertNotEqual(pair, relu_pair)
        self.assertEqual([row["operation"] for row in program["instructions"]].count("atlas.vpu_unary"), 1)
        kwargs = {
            "activation_address": 0x90001800, "weight_address": 0x90002000,
            "output_address": 0x90011800,
            "activation": manifest["selected_registers"][0],
            "weight": manifest["selected_registers"][1],
            "pair": pair[0], "relu_pair": relu_pair[0],
        }
        check_mxu0_stream(program, **kwargs)
        changed = copy.deepcopy(program)
        next(row for row in changed["instructions"] if row["operation"] == "atlas.vpu_unary")["fields"]["dst"] = pair[0]
        with self.assertRaises(NativeCompilationError):
            check_mxu0_stream(changed, **kwargs)
        changed = copy.deepcopy(program)
        next(row for row in changed["instructions"] if row["operation"] == "atlas.vpu_unary")["fields"]["src"] = relu_pair[0]
        with self.assertRaises(NativeCompilationError):
            check_mxu0_stream(changed, **kwargs)
        plan = json.loads((destination / "execution_plan.json").read_text())
        self.assertEqual((plan["operation"], plan["input_domain"], plan["output_numerical_policies"]),
                         ("mxu0_bf16_dag", "per-operation-declared", ["bf16-sign-clamp-v1"]))
        self.assertEqual(
            {(row["input"], row["required_domain"]) for row in plan["boundary_domain_obligations"]},
            {("x", "finite-normal-e4m3-or-positive-zero"),
             ("w", "finite-normal-e4m3-or-positive-zero")},
        )
        self.assertEqual(plan["producer_domain_obligations"], [{
            "producer": "y", "consumer": "z",
            "required_domain": "finite-normal-or-zero-bf16",
            "status": "caller_premise_unchecked",
        }])
        invalid = replace(request, nodes=(*base.nodes, replace(relu, attrs=())))
        with self.assertRaises(NativeCompilationError):
            self.binding.compile(
                self.snapshot, invalid, fixed_inputs={"x": 3, "w": 4}, fixed_outputs=(35,),
                target_source=ROOT, destination=self.work / "invalid-mxu0-relu", limits=SearchLimits(),
            )
        wrong_domain = replace(request, nodes=(replace(base.nodes[0], attrs=()), *request.nodes[1:]))
        with self.assertRaises(NativeCompilationError):
            self.binding.compile(
                self.snapshot, wrong_domain, fixed_inputs={"x": 3, "w": 4}, fixed_outputs=(35,),
                target_source=ROOT, destination=self.work / "invalid-mxu0-relu-domain", limits=SearchLimits(),
            )

    def test_native_mxu0_result_feeds_bf16_multiply(self) -> None:
        from atlas_mlir_native.binding import (
            BF16_MUL_ATTRS,
            MXU0_INPUT_DOMAIN,
            MXU0_POLICY,
            _mxu0_maps,
            check_mxu_bf16_binary_stream,
        )
        from merlin.semantic_compiler.model import (
            KernelRequest,
            SemanticNode,
            TensorType,
            pointwise_identity_maps,
        )
        from merlin.semantic_compiler.search import SearchLimits
        from merlin.semantic_compiler.target_binding import (
            NativeCompilationError,
        )

        fp8 = TensorType((32, 32), "fp8_e4m3", "raw-bit-pattern")
        bf16 = TensorType((32, 32), "bf16", "raw-bit-pattern")
        request = KernelRequest(
            (
                SemanticNode("x", "input", (), fp8,
                             attrs=(("input_domain", MXU0_INPUT_DOMAIN),), effect="input"),
                SemanticNode("w", "input", (), fp8,
                             attrs=(("input_domain", MXU0_INPUT_DOMAIN),), effect="input"),
                SemanticNode("scale", "input", (), bf16,
                             attrs=(BF16_MUL_ATTRS[0],), effect="input"),
                SemanticNode("mm", "matmul", ("x", "w"),
                             TensorType((32, 32), "bf16", MXU0_POLICY),
                             index_maps=_mxu0_maps()),
                SemanticNode("product", "mul_bf16", ("mm", "scale"), bf16,
                             attrs=BF16_MUL_ATTRS,
                             index_maps=pointwise_identity_maps(2, 2)),
            ),
            ("product",), ("external",),
            (("x", "external"), ("w", "external"), ("scale", "external")),
            self.binding.profile().target_identity, lowering_policy="diagnostic",
        )
        destination = self.work / "mxu0-bf16-mul"
        self.compile_verified(
            request, fixed_inputs={"x": 3, "w": 4, "scale": 5},
            fixed_outputs=(35,),
            destination=destination, limits=SearchLimits(),
        )
        selection = json.loads((destination / "native-selection.json").read_text())
        program = json.loads((destination / "physical-program.json").read_text())
        names = [value["descriptor"] for value in selection["values"]]
        self.assertEqual(program["word_count"], 53)
        self.assertEqual([name for name in names if name == "atlas.mxu0_reset_tile"],
                         ["atlas.mxu0_reset_tile"])
        self.assertEqual([name for name in names if name == "atlas.vpu_mul_bf16_tile"],
                         ["atlas.vpu_mul_bf16_tile"])
        plan = json.loads((destination / "execution_plan.json").read_text())
        self.assertIsNone(plan["input_bytes"])
        self.assertEqual(plan["input_payload_bytes"], [1024, 1024, 2048])
        self.assertEqual(plan["producer_domain_obligations"], [{
            "producer": "mm", "consumer": "product",
            "required_domain": BF16_MUL_ATTRS[0][1],
            "status": "caller_premise_unchecked",
        }])

        from merlin.semantic_compiler.linalg_bridge import translate_linalg_text

        translated = translate_linalg_text(
            (ROOT / "test/examples/native_mxu0_bf16_mul_linalg.mlir").read_text(),
            entry="mxu0_bf16_mul_tile",
            target_identity=self.binding.profile().target_identity,
            lowering_policy="diagnostic",
        )
        self.assertEqual(translated.request.nodes[3].type.numerical_policy, MXU0_POLICY)
        parsed_destination = self.work / "mxu0-bf16-mul-parsed"
        self.binding.compile(
            self.snapshot, translated.request,
            fixed_inputs={"arg0": 3, "arg1": 4, "arg2": 5},
            fixed_outputs=(35,), target_source=ROOT,
            destination=parsed_destination, limits=SearchLimits(),
        )
        self.assertEqual((destination / "program.bin").read_bytes(),
                         (parsed_destination / "program.bin").read_bytes())

        # The independent stream replay is run on the actual emitted words
        # during compile. Corrupting effects or issue gaps must fail it here.
        selected = self.snapshot.select(
            request, fixed_inputs={"x": 3, "w": 4, "scale": 5}, fixed_outputs=(35,),
            limits=SearchLimits(),
        )
        self.assertEqual(selected.status, "selected")
        assert selected.graph and selected.allocation and selected.rules
        kwargs = {
            "graph": selected.graph, "order": selected.allocation.order,
            "names": [selected.rules.symbols[value.symbol].get("descriptor", {}).get("name")
                      for value in selected.graph.values],
            "assignment": selected.allocation.addresses,
            "source_ids": ("x", "w", "scale"), "input_slots": (3, 4, 5), "output_slot": 35,
        }
        realized = check_mxu_bf16_binary_stream(program, **kwargs)
        self.assertEqual(realized[4], 32)
        for kind in ("wrong_vpu_source", "wrong_weight_slot", "missing_wait",
                     "short_delay", "physical_alias"):
            changed = copy.deepcopy(program)
            rows = changed["instructions"]
            if kind == "wrong_vpu_source":
                next(row for row in rows if row["operation"] == "atlas.vpu_binary")["fields"]["lhs"] = 34
            elif kind == "wrong_weight_slot":
                next(row for row in rows if row["operation"] == "atlas.mxu_matmul")["fields"]["weight_slot"] = 1
            elif kind == "missing_wait":
                rows.remove(next(row for row in rows if row["operation"] == "atlas.dma_wait"))
            elif kind == "short_delay":
                delays = [row for row in rows if row["operation"] == "atlas.delay"]
                max(delays, key=lambda row: row["fields"]["cycles"])["fields"]["cycles"] = 0
            else:
                bf16_loads = [row for row in rows if row["operation"] == "atlas.vload"]
                bf16_loads[2]["fields"]["dst"] = 0
                bf16_loads[3]["fields"]["dst"] = 1
            with self.subTest(kind=kind), self.assertRaises(NativeCompilationError):
                check_mxu_bf16_binary_stream(changed, **kwargs)

    def test_parsed_mxu0_readout_then_bias_add(self) -> None:
        from atlas_mlir_native.binding import (
            BF16_ADD_ATTRS,
            MXU0_INPUT_DOMAIN,
            MXU0_POLICY,
            check_mxu_bf16_binary_stream,
        )
        from merlin.semantic_compiler.linalg_bridge import translate_linalg_text
        from merlin.semantic_compiler.model import TensorType
        from merlin.semantic_compiler.reference import TensorValue, evaluate_graph
        from merlin.semantic_compiler.search import SearchLimits
        from merlin.semantic_compiler.target_binding import (
            NativeCompilationError,
        )

        from tools.atlas_virtual_evaluator import (
            RuntimeInputs,
            Tile,
            evaluate,
            parse_program,
        )

        sys.path.insert(0, str(ROOT / "test"))
        try:
            from test_mxu_numerics import dot
            from test_vpu_add_reference import _add_then_chop
        finally:
            sys.path.remove(str(ROOT / "test"))

        translated = translate_linalg_text(
            (ROOT / "test/examples/native_mxu0_bf16_add_linalg.mlir").read_text(),
            entry="mxu0_bf16_add_tile", target_identity=self.binding.profile().target_identity,
            lowering_policy="diagnostic",
        )
        request = translated.request
        self.assertEqual(tuple(node.op for node in request.nodes),
                         ("input", "input", "input", "matmul", "add_bf16"))
        self.assertEqual(request.nodes[3].type.numerical_policy, MXU0_POLICY)
        self.assertEqual(request.nodes[4].attrs, BF16_ADD_ATTRS)
        fp8 = TensorType((32, 32), "fp8_e4m3", "raw-bit-pattern")
        bf16 = TensorType((32, 32), "bf16", "raw-bit-pattern")
        one = (0x38,) * 1024
        bias = (0x3F00,) * 1024

        def check_matmul(node, operands):
            self.assertEqual(node.type.numerical_policy, MXU0_POLICY)
            self.assertEqual(tuple(value.type for value in operands), (fp8, fp8))
            cell = dot(operands[1].elements[:32], operands[0].elements[:32])
            self.assertEqual(cell, 0x4200)
            return TensorValue(node.type, (cell,) * 1024)

        def check_add(node, operands):
            self.assertEqual((node.attrs, node.type), (BF16_ADD_ATTRS, bf16))
            return TensorValue(bf16, tuple(_add_then_chop(left, right)
                                           for left, right in zip(operands[0].elements,
                                                                  operands[1].elements)))

        expected = evaluate_graph(
            request,
            {"arg0": TensorValue(fp8, one), "arg1": TensorValue(fp8, one),
             "arg2": TensorValue(bf16, bias)},
            primitives={("matmul", MXU0_POLICY): check_matmul,
                        ("add_bf16", "raw-bit-pattern"): check_add},
        )[0]
        self.assertEqual(expected.elements, (0x4202,) * 1024)

        destination = self.work / "mxu0-bf16-bias-add"
        self.compile_verified(
            request, fixed_inputs={"arg0": 3, "arg1": 4, "arg2": 5},
            fixed_outputs=(35,),
            destination=destination, limits=SearchLimits(),
        )
        plan = json.loads((destination / "execution_plan.json").read_text())
        self.assertEqual(plan["producer_domain_obligations"], [{
            "producer": request.nodes[3].id, "consumer": request.nodes[4].id,
            "required_domain": BF16_ADD_ATTRS[0][1],
            "status": "caller_premise_unchecked",
        }])
        self.assertEqual(
            {row["input"]: row["required_domain"] for row in plan["boundary_domain_obligations"]},
            {request.nodes[0].id: MXU0_INPUT_DOMAIN,
             request.nodes[1].id: MXU0_INPUT_DOMAIN,
             request.nodes[2].id: BF16_ADD_ATTRS[0][1]},
        )
        program = json.loads((destination / "physical-program.json").read_text())
        binary = (destination / "program.bin").read_bytes()
        self.assertEqual((program["word_count"], len(binary)), (53, 212))
        vpu = [row for row in program["instructions"] if row["operation"] == "atlas.vpu_binary"]
        self.assertEqual([row["fields"]["kind"] for row in vpu], ["add"])

        virtual = parse_program((destination / "atlas-virtual.mlir").read_text(),
                                function="native_tile")
        result = evaluate(virtual, RuntimeInputs(tiles={
            0: Tile("fp8", one), 1: Tile("fp8", one), 2: Tile("bf16", bias),
        }))
        self.assertEqual(result.outputs[0].bits, expected.elements)

        selected = self.snapshot.select(
            request, fixed_inputs={"arg0": 3, "arg1": 4, "arg2": 5},
            fixed_outputs=(35,), limits=SearchLimits(),
        )
        self.assertEqual(selected.status, "selected")
        assert selected.graph and selected.allocation and selected.rules
        changed = copy.deepcopy(program)
        next(row for row in changed["instructions"]
             if row["operation"] == "atlas.vpu_binary")["fields"]["kind"] = "mul"
        with self.assertRaises(NativeCompilationError):
            check_mxu_bf16_binary_stream(
                changed, graph=selected.graph, order=selected.allocation.order,
                names=[selected.rules.symbols[value.symbol].get("descriptor", {}).get("name")
                       for value in selected.graph.values],
                assignment=selected.allocation.addresses,
                source_ids=("arg0", "arg1", "arg2"), input_slots=(3, 4, 5), output_slot=35,
            )

    def test_parsed_mxu0_add_relu_preserves_both_outputs(self) -> None:
        from atlas_mlir_native.binding import (
            BF16_ADD_ATTRS,
            BF16_RELU_ATTRS,
            check_mxu_bf16_dag_stream,
        )
        from merlin.semantic_compiler.linalg_bridge import translate_linalg_text
        from merlin.semantic_compiler.search import SearchLimits
        from merlin.semantic_compiler.target_binding import NativeCompilationError

        from tools.atlas_virtual_evaluator import (
            MemoryRegion,
            RuntimeInputs,
            Tile,
            evaluate,
            parse_program,
        )

        sys.path.insert(0, str(ROOT / "test"))
        try:
            from test_mxu_numerics import dot
            from test_vpu_add_reference import _add_then_chop
        finally:
            sys.path.remove(str(ROOT / "test"))

        translated = translate_linalg_text(
            (ROOT / "test/examples/native_mxu0_add_relu_linalg.mlir").read_text(),
            entry="mxu0_add_relu_tile", target_identity=self.binding.profile().target_identity,
            lowering_policy="diagnostic",
        )
        request = translated.request
        self.assertEqual(tuple(node.op for node in request.nodes),
                         ("input", "input", "input", "matmul", "add_bf16", "relu"))
        self.assertEqual(request.outputs, (request.nodes[4].id, request.nodes[5].id))
        self.assertEqual((request.nodes[4].attrs, request.nodes[5].attrs),
                         (BF16_ADD_ATTRS, BF16_RELU_ATTRS))

        destination = self.work / "mxu0-add-relu-dag"
        fixed_inputs = {"arg0": 3, "arg1": 4, "arg2": 5}
        manifest = self.binding.compile(
            self.snapshot, request, fixed_inputs=fixed_inputs, fixed_outputs=(35, 36),
            target_source=ROOT, destination=destination, limits=SearchLimits(),
        )
        plan = json.loads((destination / "execution_plan.json").read_text())
        self.assertEqual((plan["operation"], plan["input_payload_bytes"]),
                         ("mxu0_bf16_dag", [1024, 1024, 2048]))
        self.assertEqual(
            [(item["producer"], item["consumer"]) for item in plan["producer_domain_obligations"]],
            [(request.nodes[3].id, request.nodes[4].id),
             (request.nodes[4].id, request.nodes[5].id)],
        )
        self.assertEqual(len(plan["output_addresses"]), 2)
        program = json.loads((destination / "physical-program.json").read_text())
        self.assertEqual((program["word_count"], len((destination / "program.bin").read_bytes())),
                         (70, 280))
        self.assertEqual(sum(row["operation"] == "atlas.vpu_binary"
                             for row in program["instructions"]), 1)
        self.assertEqual(sum(row["operation"] == "atlas.vpu_unary"
                             for row in program["instructions"]), 1)

        virtual = parse_program((destination / "atlas-virtual.mlir").read_text(),
                                function="native_tile")
        cell = dot((0x38,) * 32, (0x38,) * 32)
        self.assertEqual(cell, 0x4200)
        base = 0x90000000
        guards = (MemoryRegion(base + 2 * 2048, b"\xa5" * 2048),
                  MemoryRegion(base + 34 * 2048, b"\x5a" * 2048))
        for bias in (0x3F00, 0xC280):
            with self.subTest(bias=bias):
                expected_sum = _add_then_chop(cell, bias)
                expected_relu = 0 if expected_sum & 0x8000 else expected_sum
                result = evaluate(virtual, RuntimeInputs(tiles={
                    0: Tile("fp8", (0x38,) * 1024),
                    1: Tile("fp8", (0x38,) * 1024),
                    2: Tile("bf16", (bias,) * 1024),
                }, memory=guards))
                self.assertEqual(result.outputs[0].bits, (expected_sum,) * 1024)
                self.assertEqual(result.outputs[1].bits, (expected_relu,) * 1024)
                self.assertEqual(result.memory, guards)

        selected = self.snapshot.select(
            request, fixed_inputs=fixed_inputs, fixed_outputs=(35, 36),
            limits=SearchLimits(),
        )
        self.assertEqual(selected.status, "selected")
        assert selected.graph and selected.allocation and selected.rules
        kwargs = {
            "graph": selected.graph,
            "order": selected.allocation.order,
            "names": [selected.rules.symbols[value.symbol].get("descriptor", {}).get("name")
                      for value in selected.graph.values],
            "assignment": selected.allocation.addresses,
            "source_ids": ("arg0", "arg1", "arg2"),
            "mxu_inputs": {request.nodes[3].id: ("arg0", "arg1")},
            "input_slots": (3, 4, 5),
            "output_slots": (35, 36),
        }
        check_mxu_bf16_dag_stream(program, **kwargs)
        for kind in ("wrong_relu_source", "wrong_output_half", "short_delay"):
            changed = copy.deepcopy(program)
            rows = changed["instructions"]
            if kind == "wrong_relu_source":
                next(row for row in rows if row["operation"] == "atlas.vpu_unary")["fields"]["src"] = 36
            elif kind == "wrong_output_half":
                stores = [row for row in rows if row["operation"] == "atlas.vstore"]
                stores[-1]["fields"]["src"] = stores[0]["fields"]["src"]
            else:
                delays = [row for row in rows if row["operation"] == "atlas.delay"]
                max(delays, key=lambda row: row["fields"]["cycles"])["fields"]["cycles"] = 0
            with self.subTest(kind=kind), self.assertRaises(NativeCompilationError):
                check_mxu_bf16_dag_stream(changed, **kwargs)

        self.assertIn("native-selection.json", manifest["stage_artifacts"])

    def test_mxu0_vpu_dag_keeps_requested_output_order_and_shared_readout(self) -> None:
        from dataclasses import replace

        from merlin.semantic_compiler.linalg_bridge import translate_linalg_text
        from merlin.semantic_compiler.search import SearchLimits

        from tools.atlas_virtual_evaluator import (
            RuntimeInputs,
            Tile,
            evaluate,
            parse_program,
        )

        parsed = translate_linalg_text(
            (ROOT / "test/examples/native_mxu0_add_relu_linalg.mlir").read_text(),
            entry="mxu0_add_relu_tile", target_identity=self.binding.profile().target_identity,
            lowering_policy="diagnostic",
        ).request
        mxu, summed, relu = (node.id for node in parsed.nodes[-3:])
        for label, outputs, expected, policies in (
            (
                "reversed", (relu, summed), (0, 0xC200),
                ("bf16-sign-clamp-v1", "selected-f32-rne-bf16-chop-v1"),
            ),
            (
                "shared-mxu", (mxu, summed, relu), (0x4200, 0xC200, 0),
                ("mxu0-e4m3-ordered-bf16-finite-v1",
                 "selected-f32-rne-bf16-chop-v1", "bf16-sign-clamp-v1"),
            ),
        ):
            with self.subTest(label=label):
                request = replace(parsed, outputs=outputs,
                                  output_storages=("external",) * len(outputs))
                destination = self.work / f"mxu-dag-{label}"
                self.binding.compile(
                    self.snapshot, request,
                    fixed_inputs={"arg0": 3, "arg1": 4, "arg2": 5},
                    fixed_outputs=tuple(range(35, 35 + len(outputs))),
                    target_source=ROOT, destination=destination,
                    limits=SearchLimits(),
                )
                selected = json.loads((destination / "native-selection.json").read_text())
                self.assertEqual(sum(value["descriptor"] == "atlas.mxu0_reset_tile"
                                     for value in selected["values"]), 1)
                program = json.loads((destination / "physical-program.json").read_text())
                self.assertEqual(sum(row["operation"] == "atlas.mxu_matmul"
                                     for row in program["instructions"]), 1)
                plan = json.loads((destination / "execution_plan.json").read_text())
                self.assertEqual(plan["output_numerical_policies"], list(policies))
                virtual = parse_program((destination / "atlas-virtual.mlir").read_text(),
                                        function="native_tile")
                result = evaluate(virtual, RuntimeInputs(tiles={
                    0: Tile("fp8", (0x38,) * 1024),
                    1: Tile("fp8", (0x38,) * 1024),
                    2: Tile("bf16", (0xC280,) * 1024),
                }))
                self.assertEqual(tuple(result.outputs), tuple(range(len(outputs))))
                self.assertEqual(tuple(result.outputs[index].bits[0] for index in result.outputs),
                                 expected)
                self.assertTrue(all(set(tile.bits) == {expected[index]}
                                    for index, tile in result.outputs.items()))

    def test_two_mxu0_results_feed_one_vpu_add(self) -> None:
        from atlas_mlir_native.binding import (
            BF16_ADD_ATTRS,
            MXU0_INPUT_DOMAIN,
            MXU0_POLICY,
            _mxu0_maps,
            check_mxu_bf16_dag_stream,
        )
        from merlin.semantic_compiler.model import (
            KernelRequest,
            SemanticNode,
            TensorType,
            pointwise_identity_maps,
        )
        from merlin.semantic_compiler.reference import TensorValue, evaluate_graph
        from merlin.semantic_compiler.search import SearchLimits
        from merlin.semantic_compiler.target_binding import (
            NativeCompilationError,
        )

        from tools.check_program_decode import audit_program
        from tools.check_variant_inventory import load_inventory

        sys.path.insert(0, str(ROOT / "test"))
        try:
            from test_mxu_numerics import dot
            from test_vpu_add_reference import _add_then_chop
        finally:
            sys.path.remove(str(ROOT / "test"))

        fp8 = TensorType((32, 32), "fp8_e4m3", "raw-bit-pattern")
        mxu_result = TensorType((32, 32), "bf16", MXU0_POLICY)
        bf16 = TensorType((32, 32), "bf16", "raw-bit-pattern")
        inputs = ("a", "weight_a", "b", "weight_b")
        request = KernelRequest(
            (
                *(SemanticNode(name, "input", (), fp8,
                                attrs=(("input_domain", MXU0_INPUT_DOMAIN),), effect="input")
                  for name in inputs),
                SemanticNode("first", "matmul", ("a", "weight_a"), mxu_result,
                             index_maps=_mxu0_maps()),
                SemanticNode("second", "matmul", ("b", "weight_b"), mxu_result,
                             index_maps=_mxu0_maps()),
                SemanticNode("sum", "add_bf16", ("first", "second"), bf16,
                             attrs=BF16_ADD_ATTRS, index_maps=pointwise_identity_maps(2, 2)),
            ),
            ("sum",), ("external",), tuple((name, "external") for name in inputs),
            self.binding.profile().target_identity, lowering_policy="diagnostic",
        )

        def matmul(node, operands):
            self.assertEqual(tuple(value.type for value in operands), (fp8, fp8))
            self.assertEqual(node.type, mxu_result)
            cell = dot(operands[1].elements[:32], operands[0].elements[:32])
            return TensorValue(mxu_result, (cell,) * 1024)

        def add(node, operands):
            self.assertEqual(node.attrs, BF16_ADD_ATTRS)
            return TensorValue(bf16, tuple(
                _add_then_chop(left, right)
                for left, right in zip(operands[0].elements, operands[1].elements)
            ))

        values = {name: TensorValue(fp8, (0x38,) * 1024) for name in inputs}
        self.assertEqual(evaluate_graph(
            request, values,
            primitives={("matmul", MXU0_POLICY): matmul, ("add_bf16", "raw-bit-pattern"): add},
        )[0].elements, (0x4280,) * 1024)

        fixed_inputs = dict(zip(inputs, (3, 4, 5, 6)))
        destination = self.work / "two-mxu0-add"
        self.compile_verified(
            request, fixed_inputs=fixed_inputs,
            fixed_outputs=(35,),
            destination=destination, limits=SearchLimits(),
        )
        plan = json.loads((destination / "execution_plan.json").read_text())
        self.assertEqual(plan["weight_addresses"], [0x90002000, 0x90003000])
        self.assertIsNone(plan["weight_address"])
        program = json.loads((destination / "physical-program.json").read_text())
        self.assertEqual(sum(row["operation"] == "atlas.mxu_matmul"
                             for row in program["instructions"]), 2)
        self.assertEqual(sum(row["operation"] == "atlas.vpu_binary"
                             for row in program["instructions"]), 1)
        self.assertEqual(audit_program(
            load_inventory(), program, (destination / "program.bin").read_bytes(),
        )["status"], "PASS")

        selected = self.snapshot.select(
            request, fixed_inputs=fixed_inputs, fixed_outputs=(35,), limits=SearchLimits(),
        )
        self.assertEqual(selected.status, "selected")
        assert selected.graph and selected.allocation and selected.rules
        arguments = {
            "graph": selected.graph, "order": selected.allocation.order,
            "names": [selected.rules.symbols[value.symbol].get("descriptor", {}).get("name")
                      for value in selected.graph.values],
            "assignment": selected.allocation.addresses,
            "source_ids": inputs,
            "mxu_inputs": {"first": ("a", "weight_a"), "second": ("b", "weight_b")},
            "input_slots": (3, 4, 5, 6), "output_slots": (35,),
        }
        check_mxu_bf16_dag_stream(program, **arguments)
        for kind in ("second_readout_alias", "second_weight_slot"):
            changed = copy.deepcopy(program)
            if kind == "second_readout_alias":
                pops = [row for row in changed["instructions"] if row["operation"] == "atlas.mxu_pop"]
                pops[1]["fields"]["dst"] = pops[0]["fields"]["dst"]
            else:
                matmuls = [row for row in changed["instructions"] if row["operation"] == "atlas.mxu_matmul"]
                matmuls[1]["fields"]["weight_slot"] = 1
            with self.subTest(kind=kind), self.assertRaises(NativeCompilationError):
                check_mxu_bf16_dag_stream(changed, **arguments)

    def test_mxu0_unit_pack_feeds_later_mxu0(self) -> None:
        from atlas_mlir_native.binding import (
            MXU0_INPUT_DOMAIN,
            MXU0_POLICY,
            PACK_ATTRS,
            PACK_POLICY,
            _mxu0_maps,
            check_mxu_bf16_dag_stream,
        )
        from atlas_mlir_native.reference import evaluate_unit_pack
        from merlin.semantic_compiler.model import (
            KernelRequest,
            SemanticNode,
            TensorType,
            pointwise_identity_maps,
        )
        from merlin.semantic_compiler.reference import TensorValue, evaluate_graph
        from merlin.semantic_compiler.search import SearchLimits
        from merlin.semantic_compiler.target_binding import NativeCompilationError

        from tools.atlas_virtual_evaluator import (
            RuntimeInputs,
            Tile,
            evaluate,
            parse_program,
        )
        from tools.check_program_decode import audit_program
        from tools.check_variant_inventory import load_inventory

        sys.path.insert(0, str(ROOT / "test"))
        try:
            from test_mxu_numerics import dot
        finally:
            sys.path.remove(str(ROOT / "test"))

        fp8 = TensorType((32, 32), "fp8_e4m3", "raw-bit-pattern")
        bf16 = TensorType((32, 32), "bf16", MXU0_POLICY)
        packed = TensorType((32, 32), "fp8_e4m3", PACK_POLICY)
        inputs = ("a", "first_weight", "second_weight")
        request = KernelRequest(
            (
                *(SemanticNode(name, "input", (), fp8,
                                attrs=(("input_domain", MXU0_INPUT_DOMAIN),), effect="input")
                  for name in inputs),
                SemanticNode("first", "matmul", ("a", "first_weight"), bf16,
                             index_maps=_mxu0_maps()),
                SemanticNode("packed", "pack_fp8", ("first",), packed,
                             attrs=PACK_ATTRS, index_maps=pointwise_identity_maps(2, 1)),
                SemanticNode("second", "matmul", ("packed", "second_weight"), bf16,
                             index_maps=_mxu0_maps()),
            ),
            ("second",), ("external",), tuple((name, "external") for name in inputs),
            self.binding.profile().target_identity, lowering_policy="diagnostic",
        )
        fixed_inputs = dict(zip(inputs, (3, 4, 5)))
        destination = self.work / "mxu0-pack-mxu0"
        self.binding.compile(
            self.snapshot, request, fixed_inputs=fixed_inputs, fixed_outputs=(35,),
            target_source=ROOT, destination=destination, limits=SearchLimits(),
        )
        program = json.loads((destination / "physical-program.json").read_text())
        self.assertEqual(sum(row["operation"] == "atlas.mxu_matmul"
                             for row in program["instructions"]), 2)
        self.assertEqual(sum(row["operation"] == "atlas.vpu_pack"
                             for row in program["instructions"]), 1)
        plan = json.loads((destination / "execution_plan.json").read_text())
        self.assertIn({
            "producer": "first", "consumer": "packed",
            "required_domain": PACK_ATTRS[0][1],
            "status": "caller_premise_unchecked",
        }, plan["producer_domain_obligations"])
        self.assertIn({
            "producer": "packed", "consumer": "second",
            "required_domain": MXU0_INPUT_DOMAIN,
            "status": "caller_premise_unchecked",
        }, plan["producer_domain_obligations"])
        def matmul(node, operands):
            return TensorValue(node.type, (
                dot(operands[1].elements[:32], operands[0].elements[:32]),
            ) * 1024)

        def pack(node, operands):
            return evaluate_unit_pack(node, (TensorValue(
                TensorType((32, 32), "bf16", "raw-bit-pattern"),
                operands[0].elements,
            ),))

        result = evaluate_graph(request, {
            "a": TensorValue(fp8, (0x38,) * 1024),
            "first_weight": TensorValue(fp8, (0x10,) * 1024),
            "second_weight": TensorValue(fp8, (0x38,) * 1024),
        }, primitives={
            ("matmul", MXU0_POLICY): matmul,
            ("pack_fp8", PACK_POLICY): pack,
        })
        self.assertEqual(set(result[0].elements), {0x4200})
        with self.assertRaisesRegex(ValueError, "outside the seven-code"):
            evaluate_graph(request, {
                "a": TensorValue(fp8, (0x38,) * 1024),
                "first_weight": TensorValue(fp8, (0x38,) * 1024),
                "second_weight": TensorValue(fp8, (0x38,) * 1024),
            }, primitives={
                ("matmul", MXU0_POLICY): matmul,
                ("pack_fp8", PACK_POLICY): pack,
            })
        virtual = parse_program((destination / "atlas-virtual.mlir").read_text(),
                                function="native_tile")
        executed = evaluate(virtual, RuntimeInputs(tiles={
            0: Tile("fp8", (0x38,) * 1024),
            1: Tile("fp8", (0x10,) * 1024),
            2: Tile("fp8", (0x38,) * 1024),
        }))
        self.assertEqual(set(executed.outputs[0].bits), {0x4200})
        self.assertEqual(audit_program(
            load_inventory(), program, (destination / "program.bin").read_bytes(),
        )["status"], "PASS")
        selected = self.snapshot.select(
            request, fixed_inputs=fixed_inputs, fixed_outputs=(35,), limits=SearchLimits(),
        )
        self.assertEqual(selected.status, "selected")
        assert selected.graph and selected.allocation and selected.rules
        arguments = {
            "graph": selected.graph, "order": selected.allocation.order,
            "names": [selected.rules.symbols[value.symbol].get("descriptor", {}).get("name")
                      for value in selected.graph.values],
            "assignment": selected.allocation.addresses, "source_ids": inputs,
            "mxu_inputs": {"first": ("a", "first_weight"),
                           "second": ("packed", "second_weight")},
            "input_slots": (3, 4, 5), "output_slots": (35,),
        }
        check_mxu_bf16_dag_stream(program, **arguments)
        for kind in ("scale_register", "branch_word", "premature_row_read"):
            changed = copy.deepcopy(program)
            rows = changed["instructions"]
            if kind == "scale_register":
                next(row for row in rows if row["operation"] == "atlas.vpu_pack")[
                    "fields"
                ]["scale_reg"] = 2
            elif kind == "branch_word":
                next(row for row in rows if row["operation"] == "atlas.branch")[
                    "word_u32"
                ] ^= 1 << 8
            else:
                first_row_read = next(index for index, row in enumerate(rows)
                                      if row["operation"] == "atlas.scalar_load"
                                      and row["fields"].get("kind") == "lw")
                rows[first_row_read - 1]["fields"]["cycles"] = 0
            with self.subTest(kind=kind), self.assertRaises(NativeCompilationError):
                check_mxu_bf16_dag_stream(changed, **arguments)

    def test_two_packed_projections_feed_both_mxu_operands(self) -> None:
        from atlas_mlir_native.binding import (
            MXU0_INPUT_DOMAIN,
            MXU0_POLICY,
            PACK_ATTRS,
            PACK_POLICY,
            _mxu0_maps,
            check_mxu_bf16_dag_stream,
        )
        from merlin.semantic_compiler.model import (
            KernelRequest,
            SemanticNode,
            TensorType,
            pointwise_identity_maps,
        )
        from merlin.semantic_compiler.search import SearchLimits
        from merlin.semantic_compiler.target_binding import (
            NativeCompilationError,
        )

        from tools.atlas_virtual_evaluator import (
            RuntimeInputs,
            Tile,
            evaluate,
            parse_program,
        )
        from tools.check_program_decode import audit_program
        from tools.check_variant_inventory import load_inventory

        fp8 = TensorType((32, 32), "fp8_e4m3", "raw-bit-pattern")
        bf16 = TensorType((32, 32), "bf16", MXU0_POLICY)
        packed = TensorType((32, 32), "fp8_e4m3", PACK_POLICY)
        inputs = ("tokens", "wq", "wk")
        request = KernelRequest(
            (
                *(SemanticNode(name, "input", (), fp8,
                                attrs=(("input_domain", MXU0_INPUT_DOMAIN),), effect="input")
                  for name in inputs),
                SemanticNode("q", "matmul", ("tokens", "wq"), bf16,
                             index_maps=_mxu0_maps()),
                SemanticNode("k", "matmul", ("tokens", "wk"), bf16,
                             index_maps=_mxu0_maps()),
                SemanticNode("q_fp8", "pack_fp8", ("q",), packed,
                             attrs=PACK_ATTRS, index_maps=pointwise_identity_maps(2, 1)),
                SemanticNode("k_fp8", "pack_fp8", ("k",), packed,
                             attrs=PACK_ATTRS, index_maps=pointwise_identity_maps(2, 1)),
                SemanticNode("scores", "matmul", ("q_fp8", "k_fp8"), bf16,
                             index_maps=_mxu0_maps()),
            ),
            ("scores",), ("external",), tuple((name, "external") for name in inputs),
            self.binding.profile().target_identity, lowering_policy="diagnostic",
        )
        fixed_inputs = dict(zip(inputs, (3, 4, 5)))
        destination = self.work / "two-packed-projections"
        self.compile_verified(
            request, fixed_inputs=fixed_inputs,
            fixed_outputs=(35,),
            destination=destination, limits=SearchLimits(),
        )
        plan = json.loads((destination / "execution_plan.json").read_text())
        self.assertEqual(plan["weight_source_nodes"], ["wq", "wk", "k_fp8"])
        self.assertEqual(plan["weight_addresses"], [0x90002000, 0x90002800, None])
        program = json.loads((destination / "physical-program.json").read_text())
        self.assertEqual(sum(row["operation"] == "atlas.mxu_matmul"
                             for row in program["instructions"]), 3)
        packs = [row for row in program["instructions"]
                 if row["operation"] == "atlas.vpu_pack"]
        self.assertEqual(len(packs), 2)
        self.assertEqual(len({row["fields"]["dst"] for row in packs}), 2)
        self.assertEqual(audit_program(
            load_inventory(), program, (destination / "program.bin").read_bytes(),
        )["status"], "PASS")
        virtual = parse_program((destination / "atlas-virtual.mlir").read_text(),
                                function="native_tile")
        for weight_code, expected in ((0x10, 0x4200), (0x18, 0x4280)):
            with self.subTest(weight_code=weight_code):
                executed = evaluate(virtual, RuntimeInputs(tiles={
                    0: Tile("fp8", (0x38,) * 1024),
                    1: Tile("fp8", (0x10,) * 1024),
                    2: Tile("fp8", (weight_code,) * 1024),
                }))
                self.assertEqual(set(executed.outputs[0].bits), {expected})

        selected = self.snapshot.select(
            request, fixed_inputs=fixed_inputs, fixed_outputs=(35,), limits=SearchLimits(),
        )
        self.assertEqual(selected.status, "selected")
        assert selected.graph and selected.allocation and selected.rules
        arguments = {
            "graph": selected.graph, "order": selected.allocation.order,
            "names": [selected.rules.symbols[value.symbol].get("descriptor", {}).get("name")
                      for value in selected.graph.values],
            "assignment": selected.allocation.addresses, "source_ids": inputs,
            "mxu_inputs": {"q": ("tokens", "wq"), "k": ("tokens", "wk"),
                           "scores": ("q_fp8", "k_fp8")},
            "input_slots": (3, 4, 5), "output_slots": (35,),
        }
        check_mxu_bf16_dag_stream(program, **arguments)
        changed = copy.deepcopy(program)
        next(row for row in changed["instructions"]
             if row["operation"] == "atlas.vpu_pack")["word_u32"] ^= 1 << 7
        with self.assertRaises(NativeCompilationError):
            check_mxu_bf16_dag_stream(changed, **arguments)

    def test_parsed_qk_pack_score_keeps_computed_weight(self) -> None:
        from merlin.semantic_compiler.linalg_bridge import translate_linalg_text
        from merlin.semantic_compiler.search import SearchLimits

        from tools.atlas_virtual_evaluator import (
            RuntimeInputs,
            Tile,
            evaluate,
            parse_program,
        )

        source = (ROOT / "test/examples/native_qk_pack_score_linalg.mlir").read_text()
        request = translate_linalg_text(
            source, entry="qk_pack_score_tile",
            target_identity=self.binding.profile().target_identity,
            lowering_policy="diagnostic",
        ).request
        self.assertEqual(len(request.outputs), 1)
        self.assertEqual(sum(node.op == "matmul" for node in request.nodes), 3)
        self.assertEqual(sum(node.op == "pack_fp8" for node in request.nodes), 2)
        fixed_inputs = {"arg0": 3, "arg1": 4, "arg2": 5}
        destination = self.work / "parsed-qk-pack-score"
        self.compile_verified(
            request, fixed_inputs=fixed_inputs,
            fixed_outputs=(35,),
            destination=destination, limits=SearchLimits(),
        )
        plan = json.loads((destination / "execution_plan.json").read_text())
        self.assertEqual(plan["weight_addresses"], [0x90002000, 0x90002800, None])
        self.assertEqual(plan["weight_source_nodes"][:2], ["arg1", "arg2"])
        virtual = parse_program((destination / "atlas-virtual.mlir").read_text(),
                                function="native_tile")
        result = evaluate(virtual, RuntimeInputs(tiles={
            0: Tile("fp8", (0x38,) * 1024),
            1: Tile("fp8", (0x10,) * 1024),
            2: Tile("fp8", (0x18,) * 1024),
        }))
        self.assertEqual(set(result.outputs[0].bits), {0x4280})

    def test_native_qk_score_row_max_and_centered_difference(self) -> None:
        from atlas_mlir_native.binding import (
            BF16_EXP_ATTRS,
            BF16_MUL_ATTRS,
            BF16_RECIP_ATTRS,
            BF16_ROW_MAX_ATTRS,
            BF16_ROW_SUM_ATTRS,
            BF16_SUB_ATTRS,
            _row_reduce_maps,
            check_mxu_bf16_dag_stream,
        )
        from merlin.semantic_compiler.linalg_bridge import translate_linalg_text
        from merlin.semantic_compiler.model import (
            KernelRequest,
            SemanticNode,
            TensorType,
            pointwise_identity_maps,
        )
        from merlin.semantic_compiler.search import SearchLimits
        from merlin.semantic_compiler.target_binding import (
            NativeCompilationError,
        )

        from tools.atlas_virtual_evaluator import (
            RuntimeInputs,
            Tile,
            evaluate,
            parse_program,
        )
        from tools.check_program_decode import audit_program
        from tools.check_variant_inventory import load_inventory

        parsed = translate_linalg_text(
            (ROOT / "test/examples/native_qk_pack_score_linalg.mlir").read_text(),
            entry="qk_pack_score_tile",
            target_identity=self.binding.profile().target_identity,
            lowering_policy="diagnostic",
        ).request
        score = parsed.outputs[0]
        bf16 = TensorType((32, 32), "bf16", "raw-bit-pattern")
        request = KernelRequest(
            (*parsed.nodes,
             SemanticNode("maximum", "row_max_bf16", (score,), bf16,
                          attrs=BF16_ROW_MAX_ATTRS, index_maps=_row_reduce_maps()),
             SemanticNode("centered", "sub_bf16", (score, "maximum"), bf16,
                          attrs=BF16_SUB_ATTRS, index_maps=pointwise_identity_maps(2, 2))),
            ("centered",), ("external",), parsed.input_storages,
            parsed.target_identity, lowering_policy="diagnostic",
        )
        fixed_inputs = {"arg0": 3, "arg1": 4, "arg2": 5}
        destination = self.work / "qk-row-max-centered"
        self.compile_verified(
            request, fixed_inputs=fixed_inputs,
            fixed_outputs=(35,),
            destination=destination, limits=SearchLimits(),
        )
        plan = json.loads((destination / "execution_plan.json").read_text())
        self.assertIn({
            "producer": score, "consumer": "maximum",
            "required_domain": BF16_ROW_MAX_ATTRS[0][1],
            "status": "caller_premise_unchecked",
        }, plan["producer_domain_obligations"])
        self.assertIn({
            "producer": score, "consumer": "centered",
            "required_domain": BF16_SUB_ATTRS[0][1],
            "status": "caller_premise_unchecked",
        }, plan["producer_domain_obligations"])
        program = json.loads((destination / "physical-program.json").read_text())
        modes = audit_program(
            load_inventory(), program, (destination / "program.bin").read_bytes(),
        )["modes"]
        self.assertEqual((modes["VREDMAX_ROW_BF16"], modes["VSUB_BF16"]), (1, 1))
        virtual = parse_program((destination / "atlas-virtual.mlir").read_text(),
                                function="native_tile")
        result = evaluate(virtual, RuntimeInputs(tiles={
            0: Tile("fp8", (0x38,) * 1024),
            1: Tile("fp8", (0x10,) * 1024),
            2: Tile("fp8", (0x18,) * 1024),
        }))
        self.assertEqual(set(result.outputs[0].bits), {0})

        selected = self.snapshot.select(
            request, fixed_inputs=fixed_inputs, fixed_outputs=(35,), limits=SearchLimits(),
        )
        self.assertEqual(selected.status, "selected")
        assert selected.graph and selected.allocation and selected.rules
        checker = {
            "graph": selected.graph, "order": selected.allocation.order,
            "names": [selected.rules.symbols[value.symbol].get("descriptor", {}).get("name")
                      for value in selected.graph.values],
            "assignment": selected.allocation.addresses,
            "source_ids": ("arg0", "arg1", "arg2"),
            "mxu_inputs": {node.id: node.inputs for node in request.nodes if node.op == "matmul"},
            "input_slots": (3, 4, 5), "output_slots": (35,),
        }
        check_mxu_bf16_dag_stream(program, **checker)
        changed = copy.deepcopy(program)
        next(row for row in changed["instructions"]
             if row["operation"] == "atlas.vpu_reduce")["fields"]["kind"] = "row_sum"
        with self.assertRaises(NativeCompilationError):
            check_mxu_bf16_dag_stream(changed, **checker)

        # The catalog admits a centered-score reduction in one candidate.
        # This omits exponentiation and therefore is not a softmax.
        normalizer = SemanticNode(
            "normalizer", "row_sum_bf16", ("centered",), bf16,
            attrs=BF16_ROW_SUM_ATTRS, index_maps=_row_reduce_maps(),
        )
        extended = KernelRequest(
            (*request.nodes, normalizer), ("normalizer",), ("external",),
            request.input_storages, request.target_identity,
            lowering_policy="diagnostic",
        )
        extended_dir = self.work / "qk-row-max-center-row-sum"
        self.compile_verified(
            extended, fixed_inputs=fixed_inputs,
            fixed_outputs=(35,),
            destination=extended_dir, limits=SearchLimits(),
        )
        extended_plan = json.loads((extended_dir / "execution_plan.json").read_text())
        self.assertIn({
            "producer": "centered", "consumer": "normalizer",
            "required_domain": BF16_ROW_SUM_ATTRS[0][1],
            "status": "caller_premise_unchecked",
        }, extended_plan["producer_domain_obligations"])
        extended_program = json.loads((extended_dir / "physical-program.json").read_text())
        extended_modes = audit_program(
            load_inventory(), extended_program, (extended_dir / "program.bin").read_bytes(),
        )["modes"]
        self.assertEqual((extended_modes["VREDMAX_ROW_BF16"], extended_modes["VSUB_BF16"],
                          extended_modes["VREDSUM_ROW_BF16"]), (1, 1, 1))
        extended_virtual = parse_program((extended_dir / "atlas-virtual.mlir").read_text(),
                                         function="native_tile")
        extended_result = evaluate(extended_virtual, RuntimeInputs(tiles={
            0: Tile("fp8", (0x38,) * 1024),
            1: Tile("fp8", (0x10,) * 1024),
            2: Tile("fp8", (0x18,) * 1024),
        }))
        self.assertEqual(set(extended_result.outputs[0].bits), {0})

        exponential = SemanticNode(
            "exponential", "exp_bf16", ("centered",), bf16,
            attrs=BF16_EXP_ATTRS, index_maps=pointwise_identity_maps(2, 1),
        )
        exp_sum = SemanticNode(
            "exp_sum", "row_sum_bf16", ("exponential",), bf16,
            attrs=BF16_ROW_SUM_ATTRS, index_maps=_row_reduce_maps(),
        )
        exp_request = KernelRequest(
            (*request.nodes, exponential, exp_sum), ("exp_sum",), ("external",),
            request.input_storages, request.target_identity,
            lowering_policy="diagnostic",
        )
        exp_dir = self.work / "qk-row-max-center-exp-row-sum"
        self.compile_verified(
            exp_request, fixed_inputs=fixed_inputs,
            fixed_outputs=(35,),
            destination=exp_dir, limits=SearchLimits(),
        )
        exp_plan = json.loads((exp_dir / "execution_plan.json").read_text())
        self.assertIn({
            "producer": "centered", "consumer": "exponential",
            "required_domain": BF16_EXP_ATTRS[0][1],
            "status": "caller_premise_unchecked",
        }, exp_plan["producer_domain_obligations"])
        exp_program = json.loads((exp_dir / "physical-program.json").read_text())
        exp_modes = audit_program(
            load_inventory(), exp_program, (exp_dir / "program.bin").read_bytes(),
        )["modes"]
        self.assertEqual((exp_modes["VREDMAX_ROW_BF16"], exp_modes["VSUB_BF16"],
                          exp_modes["VEXP"], exp_modes["VREDSUM_ROW_BF16"]),
                         (1, 1, 1, 1))
        exp_virtual = parse_program((exp_dir / "atlas-virtual.mlir").read_text(),
                                    function="native_tile")
        exp_result = evaluate(exp_virtual, RuntimeInputs(tiles={
            0: Tile("fp8", (0x38,) * 1024),
            1: Tile("fp8", (0x10,) * 1024),
            2: Tile("fp8", (0x18,) * 1024),
        }))
        self.assertEqual(set(exp_result.outputs[0].bits), {0x4200})

        inverse = SemanticNode(
            "inverse", "recip_bf16", ("exp_sum",), bf16,
            attrs=BF16_RECIP_ATTRS, index_maps=pointwise_identity_maps(2, 1),
        )
        probability = SemanticNode(
            "probability", "mul_bf16", ("exponential", "inverse"), bf16,
            attrs=BF16_MUL_ATTRS, index_maps=pointwise_identity_maps(2, 2),
        )
        softmax_request = KernelRequest(
            (*exp_request.nodes, inverse, probability), ("probability",),
            ("external",), request.input_storages, request.target_identity,
            lowering_policy="diagnostic",
        )
        fixture_request = KernelRequest.from_record(json.loads(
            (ROOT / "test/examples/native_qk_softmax_request.json").read_text()
        ))
        self.assertEqual(fixture_request.digest(), softmax_request.digest())
        softmax_dir = self.work / "qk-row-softmax"
        self.compile_verified(
            softmax_request, fixed_inputs=fixed_inputs,
            fixed_outputs=(35,),
            destination=softmax_dir, limits=SearchLimits(),
        )
        softmax_plan = json.loads((softmax_dir / "execution_plan.json").read_text())
        self.assertIn({
            "producer": "exp_sum", "consumer": "inverse",
            "required_domain": BF16_RECIP_ATTRS[0][1],
            "status": "caller_premise_unchecked",
        }, softmax_plan["producer_domain_obligations"])
        softmax_program = json.loads((softmax_dir / "physical-program.json").read_text())
        softmax_modes = audit_program(
            load_inventory(), softmax_program, (softmax_dir / "program.bin").read_bytes(),
        )["modes"]
        self.assertEqual((softmax_modes["VEXP"], softmax_modes["VREDSUM_ROW_BF16"],
                          softmax_modes["VRECIP_BF16"], softmax_modes["VMUL_BF16"]),
                         (1, 1, 1, 1))
        softmax_virtual = parse_program(
            (softmax_dir / "atlas-virtual.mlir").read_text(), function="native_tile",
        )
        softmax_result = evaluate(softmax_virtual, RuntimeInputs(tiles={
            0: Tile("fp8", (0x38,) * 1024),
            1: Tile("fp8", (0x10,) * 1024),
            2: Tile("fp8", (0x18,) * 1024),
        }))
        self.assertEqual(set(softmax_result.outputs[0].bits), {0x3D00})
        from test.rtl.make_native_qk_softmax_fixture import _panel

        payloads, expected_physical = _panel("patterned")
        patterned_result = evaluate(softmax_virtual, RuntimeInputs(tiles={
            index: Tile("fp8", tuple(payload))
            for index, payload in enumerate(payloads)
        }))
        logical = patterned_result.outputs[0].bits
        physical = tuple(logical[row * 32 + column + half * 16]
                         for half in range(2) for row in range(32) for column in range(16))
        self.assertEqual(struct.pack("<1024H", *physical), expected_physical)

    def test_parsed_mxu0_pack_mxu0_compiles_without_losing_the_pack_edge(self) -> None:
        from merlin.semantic_compiler.linalg_bridge import (
            LinalgBridgeError,
            translate_linalg_text,
        )
        from merlin.semantic_compiler.search import SearchLimits

        from tools.check_program_decode import audit_program
        from tools.check_variant_inventory import load_inventory

        source = (ROOT / "test/examples/native_mxu0_unit_pack_mxu0_linalg.mlir").read_text()
        request = translate_linalg_text(
            source, entry="mxu0_pack_mxu0_tile",
            target_identity=self.binding.profile().target_identity,
            lowering_policy="diagnostic",
        ).request
        self.assertEqual(tuple(node.op for node in request.nodes), (
            "input", "input", "input", "matmul", "pack_fp8", "matmul",
        ))
        self.assertEqual(request.nodes[-1].inputs[0], request.nodes[-2].id)
        fixed_inputs = {"arg0": 3, "arg1": 4, "arg2": 5}
        destination = self.work / "parsed-mxu0-pack-mxu0"
        self.compile_verified(
            request, fixed_inputs=fixed_inputs,
            fixed_outputs=(35,),
            destination=destination, limits=SearchLimits(),
        )
        program = json.loads((destination / "physical-program.json").read_text())
        self.assertEqual(program["word_count"], 96)
        self.assertEqual(sum(row["operation"] == "atlas.mxu_matmul"
                             for row in program["instructions"]), 2)
        self.assertEqual(sum(row["operation"] == "atlas.vpu_pack"
                             for row in program["instructions"]), 1)
        self.assertEqual(audit_program(
            load_inventory(), program, (destination / "program.bin").read_bytes(),
        )["status"], "PASS")
        plan = json.loads((destination / "execution_plan.json").read_text())
        self.assertEqual(len(plan["producer_domain_obligations"]), 2)

        for changed in (
            source.replace(
                '%second_w: tensor<32x32xf8E4M3FN> '
                '{merlin.input_domain = "finite-normal-e4m3-or-positive-zero"}',
                '%second_w: tensor<32x32xf8E4M3FN> '
                '{merlin.input_domain = "all-bit-patterns"}',
            ),
            source.replace('"fp8-e4m3-unit-e8m0-seven-codes-v1"', '"raw-bit-pattern"'),
            source.replace("affine_map<(m,n,k)->(n,k)>", "affine_map<(m,n,k)->(k,n)>"),
        ):
            self.assertNotEqual(changed, source)
            with self.assertRaises(LinalgBridgeError):
                translate_linalg_text(
                    changed, entry="mxu0_pack_mxu0_tile",
                    target_identity=self.binding.profile().target_identity,
                    lowering_policy="diagnostic",
                )

    def test_parsed_mxu0_relu_pack_mxu0_preserves_both_outputs(self) -> None:
        from atlas_mlir_native.binding import check_mxu_bf16_dag_stream
        from merlin.semantic_compiler.linalg_bridge import (
            LinalgBridgeError,
            translate_linalg_text,
        )
        from merlin.semantic_compiler.search import SearchLimits
        from merlin.semantic_compiler.target_binding import (
            NativeCompilationError,
        )

        from tools.atlas_virtual_evaluator import (
            RuntimeInputs,
            Tile,
            evaluate,
            parse_program,
        )
        from tools.check_program_decode import audit_program
        from tools.check_variant_inventory import load_inventory

        source = (ROOT / "test/examples/native_mxu0_relu_pack_mxu0_linalg.mlir").read_text()
        request = translate_linalg_text(
            source, entry="mxu0_relu_pack_mxu0_tile",
            target_identity=self.binding.profile().target_identity,
            lowering_policy="diagnostic",
        ).request
        self.assertEqual(tuple(node.op for node in request.nodes), (
            "input", "input", "input", "matmul", "relu", "pack_fp8", "matmul",
        ))
        self.assertEqual(len(request.outputs), 2)
        fixed_inputs = {"arg0": 3, "arg1": 4, "arg2": 5}
        destination = self.work / "parsed-mxu0-relu-pack-mxu0"
        self.compile_verified(
            request, fixed_inputs=fixed_inputs,
            fixed_outputs=(35, 36),
            destination=destination, limits=SearchLimits(),
        )
        program = json.loads((destination / "physical-program.json").read_text())
        operations = [row["operation"] for row in program["instructions"]]
        self.assertEqual(operations.count("atlas.mxu_matmul"), 2)
        self.assertEqual(operations.count("atlas.vpu_unary"), 1)
        self.assertEqual(operations.count("atlas.vpu_pack"), 1)
        self.assertEqual(audit_program(
            load_inventory(), program, (destination / "program.bin").read_bytes(),
        )["status"], "PASS")
        plan = json.loads((destination / "execution_plan.json").read_text())
        self.assertEqual(len(plan["producer_domain_obligations"]), 3)
        self.assertEqual({entry["status"] for entry in plan["producer_domain_obligations"]},
                         {"caller_premise_unchecked"})

        virtual = parse_program((destination / "atlas-virtual.mlir").read_text(),
                                function="native_tile")
        for first_weight, expected in ((0x10, (0x3f80, 0x4200)),
                                       (0x90, (0, 0))):
            with self.subTest(first_weight=first_weight):
                result = evaluate(virtual, RuntimeInputs(tiles={
                    0: Tile("fp8", (0x38,) * 1024),
                    1: Tile("fp8", (first_weight,) * 1024),
                    2: Tile("fp8", (0x38,) * 1024),
                }))
                self.assertEqual(tuple(result.outputs), (0, 1))
                for index, bits in enumerate(expected):
                    self.assertEqual(set(result.outputs[index].bits), {bits})

        selected = self.snapshot.select(
            request, fixed_inputs=fixed_inputs, fixed_outputs=(35, 36),
            limits=SearchLimits(),
        )
        self.assertEqual(selected.status, "selected")
        assert selected.graph and selected.allocation and selected.rules
        arguments = {
            "graph": selected.graph, "order": selected.allocation.order,
            "names": [selected.rules.symbols[value.symbol].get("descriptor", {}).get("name")
                      for value in selected.graph.values],
            "assignment": selected.allocation.addresses,
            "source_ids": tuple(fixed_inputs),
            "mxu_inputs": {node.id: node.inputs for node in request.nodes if node.op == "matmul"},
            "input_slots": tuple(fixed_inputs.values()), "output_slots": (35, 36),
        }
        check_mxu_bf16_dag_stream(program, **arguments)
        changed_program = copy.deepcopy(program)
        next(row for row in changed_program["instructions"]
             if row["operation"] == "atlas.vpu_pack")["word_u32"] ^= 1 << 19
        with self.assertRaises(NativeCompilationError):
            check_mxu_bf16_dag_stream(changed_program, **arguments)
        changed_source = source.replace('"bf16-sign-clamp-v1"', '"raw-bit-pattern"')
        self.assertNotEqual(changed_source, source)
        with self.assertRaises(LinalgBridgeError):
            translate_linalg_text(
                changed_source, entry="mxu0_relu_pack_mxu0_tile",
                target_identity=self.binding.profile().target_identity,
                lowering_policy="diagnostic",
            )

    def test_scaled_interlayer_pack_keeps_its_code_and_computed_operand(self) -> None:
        from atlas_mlir_native.binding import (
            SCALED_PACK_POLICY,
            _tile_effects,
            check_mxu_bf16_dag_stream,
        )
        from atlas_mlir_native.reference import evaluate_scaled_pack
        from merlin.semantic_compiler.linalg_bridge import translate_linalg_text
        from merlin.semantic_compiler.model import TensorType
        from merlin.semantic_compiler.reference import TensorValue, evaluate_graph
        from merlin.semantic_compiler.search import SearchLimits
        from merlin.semantic_compiler.target_binding import NativeCompilationError

        from tools.atlas_virtual_evaluator import (
            RuntimeInputs,
            Tile,
            UnsupportedVirtualMode,
            evaluate,
            parse_program,
        )
        from tools.check_program_decode import audit_program
        from tools.check_variant_inventory import load_inventory

        sys.path.insert(0, str(ROOT / "test"))
        try:
            from test_mxu_numerics import dot
        finally:
            sys.path.remove(str(ROOT / "test"))

        source = (ROOT / "test/examples/native_mxu0_scaled_pack_mxu0_linalg.mlir").read_text()
        request = translate_linalg_text(
            source, entry="mxu0_scaled_pack_mxu0_tile",
            target_identity=self.binding.profile().target_identity,
            lowering_policy="diagnostic",
        ).request
        self.assertEqual(request.nodes[4].type.numerical_policy, SCALED_PACK_POLICY)
        self.assertEqual(dict(request.nodes[4].attrs)["scale_code"], 122)
        fixed_inputs = {"arg0": 3, "arg1": 4, "arg2": 5}
        destination = self.work / "scaled-interlayer-pack"
        self.binding.compile(
            self.snapshot, request, fixed_inputs=fixed_inputs, fixed_outputs=(35,),
            target_source=ROOT, destination=destination, limits=SearchLimits(),
        )
        program = json.loads((destination / "physical-program.json").read_text())
        self.assertEqual(program["word_count"], 96)
        self.assertEqual(sum(row["operation"] == "atlas.mxu_matmul"
                             for row in program["instructions"]), 2)
        self.assertIn(("seli", 3, 122), _tile_effects(program))
        self.assertEqual(audit_program(
            load_inventory(), program, (destination / "program.bin").read_bytes(),
        )["status"], "PASS")
        plan = json.loads((destination / "execution_plan.json").read_text())
        self.assertEqual(plan["interlayer_pack_scale_codes"], [
            {"producer": request.nodes[4].id, "scale_code": 122},
        ])
        self.assertEqual(len(plan["producer_domain_obligations"]), 2)

        fp8 = TensorType((32, 32), "fp8_e4m3", "raw-bit-pattern")

        def matmul(node, operands):
            return TensorValue(node.type, (
                dot(operands[1].elements[:32], operands[0].elements[:32]),
            ) * 1024)

        def pack(node, operands):
            return evaluate_scaled_pack(node, (TensorValue(
                TensorType((32, 32), "bf16", "raw-bit-pattern"), operands[0].elements,
            ),))

        inputs = {
            "arg0": TensorValue(fp8, (0x38,) * 1024),
            "arg1": TensorValue(fp8, (0x10,) * 1024),
            "arg2": TensorValue(fp8, (0x10,) * 1024),
        }
        result = evaluate_graph(request, inputs, primitives={
            ("matmul", request.nodes[3].type.numerical_policy): matmul,
            ("pack_fp8", SCALED_PACK_POLICY): pack,
        })
        self.assertEqual(set(result[0].elements), {0x4200})
        virtual = parse_program((destination / "atlas-virtual.mlir").read_text(),
                                function="native_tile")
        with self.assertRaises(UnsupportedVirtualMode):
            parse_program((destination / "atlas-virtual.mlir").read_text().replace(
                "scale_code = 122 : i32", "scale_code = 121 : i32"
            ), function="native_tile")
        executed = evaluate(virtual, RuntimeInputs(tiles={
            0: Tile("fp8", inputs["arg0"].elements),
            1: Tile("fp8", inputs["arg1"].elements),
            2: Tile("fp8", inputs["arg2"].elements),
        }))
        self.assertEqual(set(executed.outputs[0].bits), {0x4200})

        def sparse_panel(code, column_of_row):
            panel = [0] * 1024
            for row in range(32):
                panel[32 * row + column_of_row(row)] = code
            return tuple(panel)

        sparse = evaluate(virtual, RuntimeInputs(tiles={
            0: Tile("fp8", sparse_panel(0x38, lambda row: row)),
            1: Tile("fp8", sparse_panel(0x10, lambda row: row)),
            2: Tile("fp8", sparse_panel(0x38, lambda row: (row + 1) % 32)),
        }))
        self.assertEqual(sparse.outputs[0].bits, tuple(
            0x3F80 if row == (column + 1) % 32 else 0
            for row in range(32) for column in range(32)
        ))

        selected = self.snapshot.select(
            request, fixed_inputs=fixed_inputs, fixed_outputs=(35,), limits=SearchLimits(),
        )
        assert selected.graph and selected.allocation and selected.rules
        arguments = {
            "graph": selected.graph, "order": selected.allocation.order,
            "names": [selected.rules.symbols[value.symbol].get("descriptor", {}).get("name")
                      for value in selected.graph.values],
            "assignment": selected.allocation.addresses,
            "source_ids": ("arg0", "arg1", "arg2"),
            "mxu_inputs": {
                request.nodes[3].id: ("arg0", "arg1"),
                request.nodes[5].id: (request.nodes[4].id, "arg2"),
            },
            "input_slots": (3, 4, 5), "output_slots": (35,),
            "pack_scale_codes": {request.nodes[4].id: 122},
        }
        check_mxu_bf16_dag_stream(program, **arguments)
        changed = copy.deepcopy(program)
        next(row for row in changed["instructions"]
             if row["operation"] == "atlas.scalar_load"
             and row["fields"].get("kind") == "seli")["fields"]["offset"] = 127
        with self.assertRaises(NativeCompilationError):
            check_mxu_bf16_dag_stream(changed, **arguments)

    def test_parsed_two_mxu0_results_share_one_weight_load(self) -> None:
        from merlin.semantic_compiler.linalg_bridge import translate_linalg_text
        from merlin.semantic_compiler.search import SearchLimits

        from tools.check_program_decode import audit_program
        from tools.check_variant_inventory import load_inventory

        source = (ROOT / "test/examples/native_two_mxu0_add_linalg.mlir").read_text()
        request = translate_linalg_text(
            source, entry="two_mxu0_add_tile",
            target_identity=self.binding.profile().target_identity,
            lowering_policy="diagnostic",
        ).request
        self.assertEqual(tuple(node.op for node in request.nodes),
                         ("input", "input", "input", "matmul", "matmul", "add_bf16"))
        self.assertEqual(request.nodes[3].inputs[1], request.nodes[4].inputs[1])

        fixed_inputs = {"arg0": 3, "arg1": 4, "arg2": 5}
        destination = self.work / "parsed-two-mxu0-shared-weight"
        self.compile_verified(
            request, fixed_inputs=fixed_inputs,
            fixed_outputs=(35,),
            destination=destination, limits=SearchLimits(),
        )
        selected = json.loads((destination / "native-selection.json").read_text())
        values = selected["values"]
        weight_loads = [value for value in values
                        if value["source_node"] == "arg2" and value["descriptor"] == "atlas.load_fp8_tile"]
        mxus = [value for value in values if value["descriptor"] == "atlas.mxu0_reset_tile"]
        self.assertEqual((len(weight_loads), len(mxus)), (1, 2))
        self.assertTrue(all(weight_loads[0]["id"] in mxu["children"] for mxu in mxus))
        plan = json.loads((destination / "execution_plan.json").read_text())
        self.assertEqual(plan["weight_addresses"], [0x90002800, 0x90002800])
        program = json.loads((destination / "physical-program.json").read_text())
        self.assertEqual(sum(row["operation"] == "atlas.vload"
                             for row in program["instructions"]), 3)
        self.assertEqual(sum(row["operation"] == "atlas.mxu_matmul"
                             for row in program["instructions"]), 2)
        self.assertEqual(audit_program(
            load_inventory(), program, (destination / "program.bin").read_bytes(),
        )["status"], "PASS")

    def test_parsed_ordered_fp8_linalg_reaches_native_mxu0_and_relu(self) -> None:
        from merlin.semantic_compiler.linalg_bridge import translate_linalg_text
        from merlin.semantic_compiler.model import KernelRequest
        from merlin.semantic_compiler.search import SearchLimits
        from merlin.semantic_compiler.target_binding import verify_native_publication

        for filename, entry, operations, words in (
            ("native_mxu0_ordered_linalg.mlir", "mxu0_tile", ("atlas.mxu_matmul",), 38),
            ("native_mxu0_relu_linalg.mlir", "mxu0_relu_tile",
             ("atlas.mxu_matmul", "atlas.vpu_unary"), 40),
        ):
            with self.subTest(filename=filename):
                source = (ROOT / "test/examples" / filename).read_text()
                translated = translate_linalg_text(
                    source, entry=entry, target_identity=self.binding.profile().target_identity,
                    lowering_policy="diagnostic",
                )
                request = translated.request
                self.assertEqual(request.source_identity, hashlib.sha256(source.encode()).hexdigest())
                self.assertEqual(tuple(node.op for node in request.nodes),
                                 ("input", "input", "matmul") + (("relu",) if "relu" in filename else ()))
                self.assertEqual(len(translated.constants), 0)
                destination = self.work / filename.removesuffix(".mlir")
                manifest = self.binding.compile(
                    self.snapshot, request, fixed_inputs={"arg0": 3, "arg1": 4},
                    fixed_outputs=(35,), target_source=ROOT, destination=destination,
                    limits=SearchLimits(),
                )
                verify_native_publication(
                    destination, manifest, engine="merlin_native",
                    request_digest=request.digest(), target_identity=request.target_identity,
                )
                program = json.loads((destination / "physical-program.json").read_text())
                self.assertEqual(program["word_count"], words)
                emitted = [row["operation"] for row in program["instructions"]]
                for operation in operations:
                    self.assertEqual(emitted.count(operation), 1)
                if "relu" in filename:
                    baseline = KernelRequest.from_record(json.loads(
                        (ROOT / "test/examples/native_mxu0_relu_request.json").read_text()
                    ))
                else:
                    baseline = self.mxu0_request()
                baseline_destination = self.work / (filename.removesuffix(".mlir") + "-typed")
                self.binding.compile(
                    self.snapshot, baseline, fixed_inputs={"x": 3, "w": 4},
                    fixed_outputs=(35,), target_source=ROOT, destination=baseline_destination,
                    limits=SearchLimits(),
                )
                self.assertEqual((destination / "program.bin").read_bytes(),
                                 (baseline_destination / "program.bin").read_bytes())

    def test_parsed_mxu0_relu_graph_against_exact_numeric_reference(self) -> None:
        from dataclasses import replace
        from functools import lru_cache

        from atlas_mlir_native.binding import MXU0_POLICY, _mxu0_maps
        from merlin.semantic_compiler.linalg_bridge import translate_linalg_text
        from merlin.semantic_compiler.model import TensorType
        from merlin.semantic_compiler.reference import TensorValue, evaluate_graph

        sys.path.insert(0, str(ROOT / "test"))
        try:
            from test_mxu_numerics import dot
        finally:
            sys.path.remove(str(ROOT / "test"))

        source = (ROOT / "test/examples/native_mxu0_relu_linalg.mlir").read_text()
        request = translate_linalg_text(
            source, entry="mxu0_relu_tile", target_identity=self.binding.profile().target_identity,
            lowering_policy="diagnostic",
        ).request
        request = replace(request, outputs=(request.nodes[2].id, request.nodes[3].id),
                          output_storages=("external", "external"))
        activation = tuple(0x38 if row % 2 == 0 else 0xB8
                           for row in range(32) for _ in range(32))
        weight = tuple(0x38 if col < 16 else 0xB8
                       for col in range(32) for _ in range(32))
        fp8_type = TensorType((32, 32), "fp8_e4m3", "raw-bit-pattern")

        @lru_cache(maxsize=4)
        def ordered_dot(acts: tuple[int, ...], weights: tuple[int, ...]) -> int:
            return dot(weights, acts)

        def verify_matmul(node, operands):
            self.assertEqual((node.op, node.type.numerical_policy, node.index_maps),
                             ("matmul", MXU0_POLICY, _mxu0_maps()))
            a, w = operands
            self.assertEqual((a.type, w.type), (fp8_type, fp8_type))
            result = tuple(ordered_dot(a.elements[32 * row:32 * row + 32],
                                       w.elements[32 * col:32 * col + 32])
                           for row in range(32) for col in range(32))
            return TensorValue(node.type, result)

        matmul, relu = evaluate_graph(
            request, {"arg0": TensorValue(fp8_type, activation),
                      "arg1": TensorValue(fp8_type, weight)},
            primitives={("matmul", MXU0_POLICY): verify_matmul},
        )
        expected = tuple(0x4200 if (row % 2 == 0) == (col < 16) else 0xC200
                         for row in range(32) for col in range(32))
        self.assertEqual(matmul.elements, expected)
        self.assertEqual(relu.elements, tuple(0 if bits & 0x8000 else bits for bits in expected))

    def test_native_two_outputs_share_one_physical_producer(self) -> None:
        from atlas_mlir_native.binding import (
            _tile_effects,
            check_bf16_tile_stream,
            check_copy_twice_stream,
        )
        from merlin.semantic_compiler.linalg_bridge import translate_linalg_text
        from merlin.semantic_compiler.search import SearchLimits
        from merlin.semantic_compiler.target_binding import (
            NativeCompilationError,
        )

        translated = translate_linalg_text(
            (ROOT / "test/examples/native_bf16_two_outputs_linalg.mlir").read_text(),
            entry="copy_twice", target_identity=self.binding.profile().target_identity,
            lowering_policy="diagnostic",
        )
        self.assertEqual(translated.source_operations,
                         ("tensor.empty", "linalg.copy", "tensor.empty", "linalg.copy", "func.return"))
        request = translated.request
        destination = self.work / "two-outputs"
        manifest = self.compile_verified(
            request, fixed_inputs={"arg0": 3},
            fixed_outputs=(35, 36),
            destination=destination, limits=SearchLimits(),
        )
        self.assertEqual(manifest["selected_registers"], [0, 1])
        plan = json.loads((destination / "execution_plan.json").read_text())
        self.assertEqual(plan["output_addresses"], [0x90011800, 0x90012000])
        program = json.loads((destination / "physical-program.json").read_text())
        self.assertEqual(sum(row["operation"] == "atlas.vload" for row in program["instructions"]), 2)
        self.assertEqual(sum(row["operation"] == "atlas.vstore" for row in program["instructions"]), 4)
        check_copy_twice_stream(program, input_address=0x90001800,
                                output_addresses=(0x90011800, 0x90012000), pair=0)
        changed = copy.deepcopy(program)
        stores = [row for row in changed["instructions"] if row["operation"] == "atlas.vstore"]
        stores[2]["fields"]["src"] = 2
        with self.assertRaises(NativeCompilationError):
            check_copy_twice_stream(changed, input_address=0x90001800,
                                    output_addresses=(0x90011800, 0x90012000), pair=0)
        sparse = self.work / "noncontiguous-two-outputs"
        self.compile_verified(
            request, fixed_inputs={"arg0": 3},
            fixed_outputs=(35, 37),
            destination=sparse, limits=SearchLimits(),
        )
        sparse_plan = json.loads((sparse / "execution_plan.json").read_text())
        self.assertEqual(sparse_plan["output_addresses"], [0x90011800, 0x90012800])
        self.assertIn('"atlas.virtual_dma_store_bf16"',
                      (sparse / "atlas-virtual.mlir").read_text())
        sparse_program = json.loads((sparse / "physical-program.json").read_text())
        self.assertEqual(
            [(effect[1], effect[2], effect[4]) for effect in _tile_effects(sparse_program)
             if effect[0] == "dma" and effect[1] == "store"],
            [("store", 0x90011800, 2048), ("store", 0x90012800, 2048)],
        )
        selected = self.snapshot.select(
            request, fixed_inputs={"arg0": 3}, fixed_outputs=(35, 37),
            limits=SearchLimits(),
        )
        self.assertEqual(selected.status, "selected")
        assert selected.graph and selected.allocation and selected.rules
        names = [selected.rules.symbols[value.symbol].get("descriptor", {}).get("name")
                 for value in selected.graph.values]
        kwargs = {
            "graph": selected.graph, "order": selected.allocation.order,
            "names": names, "assignment": selected.allocation.addresses,
            "source_ids": ("arg0",), "input_slots": (3,),
            "output_slots": (35, 37),
        }
        check_bf16_tile_stream(sparse_program, **kwargs)
        changed = copy.deepcopy(sparse_program)
        first_store = next(row for row in changed["instructions"]
                           if row["operation"] == "atlas.dma"
                           and row["fields"]["direction"] == "store")
        first_store["fields"]["size"] = 8
        with self.assertRaises(NativeCompilationError):
            check_bf16_tile_stream(changed, **kwargs)

    def test_native_two_inputs_reuse_pair_after_first_output(self) -> None:
        from atlas_mlir_native.binding import check_bf16_tile_stream
        from merlin.semantic_compiler.linalg_bridge import translate_linalg_text
        from merlin.semantic_compiler.search import SearchLimits
        from merlin.semantic_compiler.target_binding import (
            NativeCompilationError,
        )

        from tools.atlas_virtual_evaluator import (
            MemoryRegion,
            RuntimeInputs,
            evaluate,
            parse_program,
        )

        translated = translate_linalg_text(
            (ROOT / "test/examples/native_bf16_two_inputs_linalg.mlir").read_text(),
            entry="two_inputs", target_identity=self.binding.profile().target_identity,
            lowering_policy="diagnostic",
        )
        request = translated.request
        abi = {"arg0": 3, "arg1": 4}
        destination = self.work / "two-independent-inputs"
        manifest = self.compile_verified(
            request, fixed_inputs=abi,
            fixed_outputs=(35, 36),
            destination=destination, limits=SearchLimits(),
        )
        self.assertEqual(manifest["selected_registers"], [0, 1])
        plan = json.loads((destination / "execution_plan.json").read_text())
        self.assertEqual(plan["input_addresses"], [0x90001800, 0x90002000])
        self.assertEqual(plan["output_addresses"], [0x90011800, 0x90012000])
        virtual = (destination / "atlas-virtual.mlir").read_text()
        self.assertLess(virtual.index('"atlas.virtual_output_bf16"'),
                        virtual.index('"atlas.virtual_input_bf16"(%s2) {index = 1'))
        program = json.loads((destination / "physical-program.json").read_text())
        self.assertEqual(sum(row["operation"] == "atlas.vload" for row in program["instructions"]), 4)
        self.assertEqual(sum(row["operation"] == "atlas.vstore" for row in program["instructions"]), 4)
        selected = self.snapshot.select(
            request, fixed_inputs=abi, fixed_outputs=(35, 36), limits=SearchLimits(),
        )
        self.assertEqual(selected.status, "selected")
        assert selected.graph and selected.allocation and selected.rules
        names = [selected.rules.symbols[value.symbol].get("descriptor", {}).get("name")
                 for value in selected.graph.values]
        kwargs = {
            "graph": selected.graph, "order": selected.allocation.order,
            "names": names, "assignment": selected.allocation.addresses,
            "source_ids": ("arg0", "arg1"), "input_slots": (3, 4),
            "output_slots": (35, 36),
        }
        check_bf16_tile_stream(program, **kwargs)
        changed = copy.deepcopy(program)
        loads = [row for row in changed["instructions"] if row["operation"] == "atlas.vload"]
        loads[2]["fields"]["dst"] = 2
        with self.assertRaises(NativeCompilationError):
            check_bf16_tile_stream(changed, **kwargs)
        changed = copy.deepcopy(program)
        for row in changed["instructions"]:
            if row["operation"] == "atlas.delay":
                row["fields"]["cycles"] = 0
        with self.assertRaises(NativeCompilationError):
            check_bf16_tile_stream(changed, **kwargs)
        sparse = self.work / "spaced-input-slots"
        self.compile_verified(
            request, fixed_inputs={"arg0": 3, "arg1": 5},
            fixed_outputs=(35, 36),
            destination=sparse, limits=SearchLimits(),
        )
        sparse_plan = json.loads((sparse / "execution_plan.json").read_text())
        self.assertEqual(sparse_plan["input_addresses"], [0x90001800, 0x90002800])
        self.assertIn('"atlas.virtual_dma_load_bf16"',
                      (sparse / "atlas-virtual.mlir").read_text())
        spaced = self.work / "spaced-input-and-output-slots"
        self.compile_verified(
            request, fixed_inputs={"arg0": 3, "arg1": 5},
            fixed_outputs=(35, 37),
            destination=spaced, limits=SearchLimits(),
        )
        spaced_plan = json.loads((spaced / "execution_plan.json").read_text())
        self.assertEqual(spaced_plan["input_addresses"], [0x90001800, 0x90002800])
        self.assertEqual(spaced_plan["output_addresses"], [0x90011800, 0x90012800])
        first = bytes((index * 13 + 7) & 255 for index in range(2048))
        second = bytes((index * 29 + 3) & 255 for index in range(2048))
        source_memory = (
            MemoryRegion(0x90001800, first),
            MemoryRegion(0x90002800, second),
            MemoryRegion(0x900117F0, b"L" * 16 + b"?" * 2048 + b"R" * 16),
            MemoryRegion(0x90012010, b"G" * 2016),
            MemoryRegion(0x900127F0, b"X" * 16 + b"?" * 2048 + b"Y" * 16),
        )
        virtual = parse_program((spaced / "atlas-virtual.mlir").read_text(), function="native_tile")
        result = evaluate(virtual, RuntimeInputs(memory=source_memory))
        actual = {region.address: region.data for region in result.memory}
        self.assertEqual(actual[0x900117F0], b"L" * 16 + first + b"R" * 16)
        self.assertEqual(actual[0x900127F0], b"X" * 16 + second + b"Y" * 16)
        self.assertEqual(actual[0x90012010], b"G" * 2016)
        self.assertEqual(actual[0x90001800], first)
        self.assertEqual(actual[0x90002800], second)

    def test_physical_rejection_tries_another_checked_order(self) -> None:
        from atlas_mlir_native import binding as atlas_binding
        from merlin.semantic_compiler.linalg_bridge import translate_linalg_text
        from merlin.semantic_compiler.search import SearchLimits
        from merlin.semantic_compiler.target_binding import (
            NativeCompilationError,
            verify_native_publication,
        )

        request = translate_linalg_text(
            (ROOT / "test/examples/native_bf16_two_inputs_linalg.mlir").read_text(),
            entry="two_inputs", target_identity=self.binding.profile().target_identity,
            lowering_policy="diagnostic",
        ).request
        original = atlas_binding._virtual_from_selected
        attempts = 0

        def reject_first(*args, **kwargs):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise NativeCompilationError("unqualified_target", "first physical order is unavailable")
            return original(*args, **kwargs)

        destination = self.work / "materialization-fallback"
        with patch.object(atlas_binding, "_virtual_from_selected", side_effect=reject_first):
            manifest = self.binding.compile(
                self.snapshot, request, fixed_inputs={"arg0": 3, "arg1": 4},
                fixed_outputs=(35, 36), target_source=ROOT,
                destination=destination, limits=SearchLimits(),
            )
        self.assertGreater(attempts, 1)
        self.assertEqual(manifest["rejected_materialization"], 1)
        verify_native_publication(
            destination, manifest, engine="merlin_native",
            request_digest=request.digest(), target_identity=request.target_identity,
        )

        rejected = self.work / "all-physical-orders-rejected"
        with (
            patch.object(
                atlas_binding, "_virtual_from_selected",
                side_effect=NativeCompilationError("unqualified_target", "physical order rejected"),
            ),
            self.assertRaisesRegex(NativeCompilationError, "physical order rejected"),
        ):
            self.binding.compile(
                self.snapshot, request, fixed_inputs={"arg0": 3, "arg1": 4},
                fixed_outputs=(35, 36), target_source=ROOT,
                destination=rejected, limits=SearchLimits(),
            )
        self.assertFalse(rejected.exists())

    def test_native_distinct_outputs_keep_copy_and_transpose_order(self) -> None:
        from dataclasses import replace

        from atlas_mlir_native.binding import (
            FP8_DRAM_TILE_LAYOUT,
            check_copy_transpose_stream,
        )
        from merlin.semantic_compiler.linalg_bridge import translate_linalg_text
        from merlin.semantic_compiler.search import SearchLimits
        from merlin.semantic_compiler.target_binding import (
            NativeCompilationError,
        )
        from merlin.targetgen.cli import main as targetgen_main

        source = ROOT / "test/examples/native_fp8_copy_transpose_linalg.mlir"
        translated = translate_linalg_text(
            source.read_text(),
            entry="copy_transpose", target_identity=self.binding.profile().target_identity,
            lowering_policy="diagnostic",
        )
        self.assertEqual(translated.source_operations,
                         ("tensor.empty", "linalg.copy", "tensor.empty", "linalg.transpose", "func.return"))
        request = replace(
            translated.request,
            physical_extents=(("arg0", 1),),
            boundary_layouts=(("arg0", FP8_DRAM_TILE_LAYOUT),),
            output_layouts=(FP8_DRAM_TILE_LAYOUT, FP8_DRAM_TILE_LAYOUT),
        )
        self.assertEqual(request.record()["schema"], "merlin.semantic_kernel.v4")
        destination = self.work / "copy-transpose-outputs"
        manifest = self.compile_verified(
            request, fixed_inputs={"arg0": 3},
            fixed_outputs=(35, 36),
            destination=destination, limits=SearchLimits(),
        )
        self.assertEqual(manifest["selected_registers"], [0, 1])
        program = json.loads((destination / "physical-program.json").read_text())
        self.assertEqual(sum(row["operation"] == "atlas.vload" for row in program["instructions"]), 1)
        self.assertEqual(sum(row["operation"] == "atlas.xlu_transpose" for row in program["instructions"]), 1)
        kwargs = {"input_address": 0x90001800,
                  "output_addresses": (0x90011800, 0x90012000),
                  "source": 0, "result": 1}
        check_copy_transpose_stream(program, **kwargs)
        legacy_destination = self.work / "copy-transpose-legacy-request"
        self.binding.compile(
            self.snapshot, translated.request,
            fixed_inputs={"arg0": 3}, fixed_outputs=(35, 36),
            target_source=ROOT, destination=legacy_destination, limits=SearchLimits(),
        )
        self.assertEqual(
            (destination / "program.bin").read_bytes(),
            (legacy_destination / "program.bin").read_bytes(),
        )
        layouts = ROOT / "test/examples/native_fp8_copy_transpose_layouts.json"
        layout_record = json.loads(layouts.read_text())
        self.assertEqual(layout_record["source_identity"], request.source_identity)
        self.assertEqual(layout_record["target_identity"], request.target_identity)
        self.assertEqual(layout_record["boundary_layouts"], {"arg0": FP8_DRAM_TILE_LAYOUT})
        abi = self.work / "copy-transpose-abi.json"
        abi.write_text(json.dumps({"fixed_inputs": {"arg0": 3}, "fixed_outputs": [35, 36]}))
        cli_output = self.work / "copy-transpose-linalg-layout-cli"
        arguments = [
            "--engine", "merlin_native", "--support", "atlas",
            "--snapshot", str(self.snapshot.root),
            "--linalg", str(source), "--linalg-entry", "copy_transpose",
            "--physical-layouts", str(layouts), "--mode", "diagnostic",
        ]
        self.assertEqual(targetgen_main([
            "native-compile", *arguments, "--abi", str(abi),
            "--target-source", str(ROOT), "--out", str(cli_output),
        ]), 0)
        self.assertEqual((cli_output / "program.bin").read_bytes(), (destination / "program.bin").read_bytes())
        layout_stage = cli_output / "physical-boundary-layouts.json"
        self.assertEqual(json.loads(layout_stage.read_text()), layout_record)
        cli_manifest = json.loads((cli_output / "manifest.json").read_text())
        self.assertEqual(
            cli_manifest["stage_artifacts"][layout_stage.name],
            hashlib.sha256(layout_stage.read_bytes()).hexdigest(),
        )
        verify_args = [
            "native-verify", *arguments, "--artifact", str(cli_output),
            "--out", str(self.work / "copy-transpose-layout-verify.json"),
        ]
        self.assertEqual(targetgen_main(verify_args), 0)
        original_layout_stage = layout_stage.read_bytes()
        layout_stage.write_bytes(original_layout_stage + b" ")
        self.assertEqual(targetgen_main(verify_args), 2)
        layout_stage.write_bytes(original_layout_stage)
        self.assertEqual(targetgen_main(verify_args), 0)
        exact_layouts = self.work / "copy-transpose-exact-layouts.json"
        exact_record = {**layout_record, "schema": "merlin.physical_boundary_layouts.v2",
                        "request_digest": translated.request.digest()}
        exact_layouts.write_text(json.dumps(exact_record))
        exact_arguments = [
            "--engine", "merlin_native", "--support", "atlas",
            "--snapshot", str(self.snapshot.root),
            "--linalg", str(source), "--linalg-entry", "copy_transpose",
            "--physical-layouts", str(exact_layouts), "--mode", "diagnostic",
        ]
        exact_output = self.work / "copy-transpose-exact-layout-cli"
        self.assertEqual(targetgen_main([
            "native-compile", *exact_arguments, "--abi", str(abi),
            "--target-source", str(ROOT), "--out", str(exact_output),
        ]), 0)
        self.assertEqual((exact_output / "program.bin").read_bytes(), (destination / "program.bin").read_bytes())
        self.assertEqual(targetgen_main([
            "native-verify", *exact_arguments, "--artifact", str(exact_output),
            "--out", str(self.work / "copy-transpose-exact-layout-verify.json"),
        ]), 0)
        for kind in ("output_order", "delay"):
            changed = copy.deepcopy(program)
            if kind == "output_order":
                stores = [row for row in changed["instructions"] if row["operation"] == "atlas.vstore"]
                stores[0]["fields"]["src"] = 1
            else:
                delays = [row for row in changed["instructions"] if row["operation"] == "atlas.delay"]
                max(delays, key=lambda row: row["fields"]["cycles"])["fields"]["cycles"] = 0
            with self.subTest(kind=kind), self.assertRaises(NativeCompilationError):
                check_copy_transpose_stream(changed, **kwargs)

        wrong_layout = replace(request, output_layouts=(FP8_DRAM_TILE_LAYOUT, "unknown-output-layout"))
        rejected = self.work / "copy-transpose-wrong-layout"
        with self.assertRaises(NativeCompilationError):
            self.binding.compile(
                self.snapshot, wrong_layout, fixed_inputs={"arg0": 3}, fixed_outputs=(35, 36),
                target_source=ROOT, destination=rejected, limits=SearchLimits(),
            )
        self.assertFalse(rejected.exists())

    def test_native_mxu0_reset_selects_and_emits_explicit_handles(self) -> None:
        from atlas_mlir_native.binding import check_mxu0_stream
        from merlin.semantic_compiler.search import SearchLimits

        request = self.mxu0_request()
        destination = self.work / "mxu0"
        manifest = self.compile_verified(
            request, fixed_inputs={"x": 3, "w": 4},
            fixed_outputs=(35,),
            destination=destination, limits=SearchLimits(),
        )
        self.assertEqual(manifest["selected_registers"], [0, 1, 32, 33])
        virtual = (destination / "atlas-virtual.mlir").read_text()
        for op in ("virtual_mxu_load_weight", "virtual_mxu_reset", "virtual_mxu_readout_bf16"):
            self.assertIn(op, virtual)
        self.assertNotIn("virtual_mxu_matmul", virtual)
        program = json.loads((destination / "physical-program.json").read_text())
        self.assertGreater(program["word_count"], 0)
        check_mxu0_stream(program, activation_address=0x90001800,
                          weight_address=0x90002000, output_address=0x90011800,
                          activation=0, weight=1, pair=32)
        self.assertEqual((destination / "program.bin").read_bytes(),
                         struct.pack(f'<{program["word_count"]}I',
                                     *(row["word_u32"] for row in program["instructions"])))

    def test_native_mxu1_reset_keeps_distinct_policy_unit_and_timing(self) -> None:
        from dataclasses import replace

        from atlas_mlir_native.binding import (
            MXU1_POLICY,
            check_mxu_stream,
        )
        from merlin.semantic_compiler.search import SearchLimits
        from merlin.semantic_compiler.target_binding import NativeCompilationError

        from tools.check_program_decode import audit_program
        from tools.check_variant_inventory import load_inventory

        mxu0 = self.mxu0_request()
        request = self.mxu1_request()
        wrong_domain = replace(request, nodes=(mxu0.nodes[0], *request.nodes[1:]))
        with self.assertRaises(NativeCompilationError):
            self.binding.compile(
                self.snapshot, wrong_domain, fixed_inputs={"x": 3, "w": 4},
                fixed_outputs=(35,), target_source=ROOT,
                destination=self.work / "mxu1-wrong-domain", limits=SearchLimits(),
            )
        destination = self.work / "mxu1"
        manifest = self.binding.compile(
            self.snapshot, request, fixed_inputs={"x": 3, "w": 4},
            fixed_outputs=(35,), target_source=ROOT, destination=destination,
            limits=SearchLimits(),
        )
        self.assertEqual(manifest["selected_registers"], [0, 1, 32, 33])
        virtual = (destination / "atlas-virtual.mlir").read_text()
        self.assertIn("!atlas.virtual_mxu_weight<1>", virtual)
        self.assertIn("!atlas.virtual_mxu_acc<1>", virtual)
        program = json.loads((destination / "physical-program.json").read_text())
        audit = audit_program(load_inventory(), program, (destination / "program.bin").read_bytes())
        self.assertEqual(audit["status"], "PASS")
        for mode in ("VMATPUSH_W_MXU1", "VMATMUL_MXU1", "VMATPOP_BF16_MXU1"):
            self.assertEqual(audit["modes"][mode], 1)
        self.assertEqual((program["word_count"],
                          json.loads((destination / "execution_plan.json").read_text())
                          ["numerical_policy"]), (39, MXU1_POLICY))
        kwargs = {"unit": 1, "activation_address": 0x90001800,
                  "weight_address": 0x90002000, "output_address": 0x90011800,
                  "activation": 0, "weight": 1, "pair": 32}
        check_mxu_stream(program, **kwargs)
        cycle = 0
        issued = {}
        for row in program["instructions"]:
            issued[row["operation"]] = cycle
            cycle += 1 + (row["fields"]["cycles"] if row["operation"] == "atlas.delay" else 0)
        self.assertEqual(issued["atlas.mxu_pop"] - issued["atlas.mxu_matmul"], 4)
        for kind in ("unit", "early_readout"):
            changed = copy.deepcopy(program)
            rows = changed["instructions"]
            if kind == "unit":
                next(row for row in rows if row["operation"] == "atlas.mxu_pop")["fields"]["unit"] = 0
            else:
                pop_index = next(index for index, row in enumerate(rows)
                                 if row["operation"] == "atlas.mxu_pop")
                rows[pop_index - 1]["fields"]["cycles"] -= 1
            with self.subTest(kind=kind), self.assertRaises(NativeCompilationError):
                check_mxu_stream(changed, **kwargs)

    def test_native_mxu1_binary_on_selected_standalone_core(self) -> None:
        import random

        from merlin.semantic_compiler.search import SearchLimits
        from test_mxu1_reference import _expected

        keys = ("ATLAS_ARC_MODEL", "ATLAS_ARC_STATE", "ATLAS_MODELIR_ROOT")
        if not all(os.environ.get(key) for key in keys):
            self.skipTest("set selected-source-linked standalone AtlasCore paths")
        request = self.mxu1_request()
        destination = self.work / "mxu1-selected-core"
        self.binding.compile(
            self.snapshot, request, fixed_inputs={"x": 3, "w": 4},
            fixed_outputs=(35,), target_source=ROOT, destination=destination,
            limits=SearchLimits(),
        )
        binary = (destination / "program.bin").read_bytes()
        words = struct.unpack(f"<{len(binary) // 4}I", binary)
        model, state, modelir = (Path(os.environ[key]).resolve(strict=True) for key in keys)
        old_cwd = Path.cwd()
        sys.path.insert(0, str(modelir))
        try:
            os.chdir(modelir)
            from mlc.backends import cosim_atlas

            original = cosim_atlas.CosimCore

            class SelectedCore(original):
                def peek(self, name: str) -> int:
                    return super().peek("scalar/halt_now" if name == "io_halted" else name)

                def poke(self, name: str, value: int) -> None:
                    if name in ("io_dmaTL_d_bits_opcode", "io_dmaTL_d_bits_size"):
                        if name in self._S:
                            raise AssertionError(f"unexpected live ARC input {name}")
                        return
                    super().poke(name, value)

            cosim_atlas.CosimCore = SelectedCore
            try:
                rng = random.Random(0xC011A)
                palette = (0, 0x38, 0xB8, 0x40, 0xC0)
                panels = (
                    ("ones", bytes([0x38]) * 1024, bytes([0x38]) * 1024),
                    ("mixed", bytes(rng.choice(palette) for _ in range(1024)),
                     bytes(rng.choice(palette) for _ in range(1024))),
                )
                for name, weights, activations in panels:
                    with self.subTest(panel=name):
                        result = cosim_atlas.run_program(
                            model, state, words,
                            preload=[(0x90001800, activations),
                                     (0x90002000, weights),
                                     (0x90011800, b"\xA5" * 2048),
                                     (0x90012000, b"\x5A" * 32)],
                            max_cycles=5000,
                        )
                        self.assertTrue(result.halted)
                        self.assertEqual((result.reads, result.writes), (64, 64))
                        self.assertEqual(result.slave.captured(0x90011800, 2048),
                                         _expected(weights, activations))
                        self.assertEqual(result.slave.captured(0x90001800, 1024), activations)
                        self.assertEqual(result.slave.captured(0x90002000, 1024), weights)
                        self.assertEqual(result.slave.captured(0x90012000, 32), b"\x5A" * 32)
            finally:
                cosim_atlas.CosimCore = original
        finally:
            os.chdir(old_cwd)
            sys.path.remove(str(modelir))

    def test_native_mxu0_refuses_wrong_policy_map_and_placement(self) -> None:
        from dataclasses import replace

        from atlas_mlir_native.binding import _mxu0_maps, check_mxu0_stream
        from merlin.semantic_compiler.model import IndexMap
        from merlin.semantic_compiler.search import SearchLimits
        from merlin.semantic_compiler.target_binding import NativeCompilationError

        wrong_map = list(_mxu0_maps())
        wrong_map[1] = IndexMap(3, ((0, 0, 1), (0, 1, 0)), (0, 0))
        declared = self.mxu0_request()
        no_domain = replace(declared, nodes=(replace(declared.nodes[0], attrs=()),
                                             *declared.nodes[1:]))
        for request in (self.mxu0_request(policy="real-matmul"),
                        self.mxu0_request(maps=tuple(wrong_map)), no_domain):
            with self.assertRaises(NativeCompilationError):
                self.binding.compile(
                    self.snapshot, request, fixed_inputs={"x": 3, "w": 4},
                    fixed_outputs=(35,), target_source=ROOT,
                    destination=self.work / "invalid-mxu0", limits=SearchLimits(),
                )
        with self.assertRaisesRegex(NativeCompilationError, "adjacent"):
            self.binding.compile(
                self.snapshot, self.mxu0_request(), fixed_inputs={"x": 3, "w": 5},
                fixed_outputs=(35,), target_source=ROOT,
                destination=self.work / "invalid-mxu0", limits=SearchLimits(),
            )
        checker_destination = self.work / "mxu0-checker"
        self.binding.compile(
            self.snapshot, self.mxu0_request(), fixed_inputs={"x": 3, "w": 4},
            fixed_outputs=(35,), target_source=ROOT,
            destination=checker_destination, limits=SearchLimits(),
        )
        original = json.loads((checker_destination / "physical-program.json").read_text())
        for kind in ("pop_register", "weight_slot", "wait", "delay", "config_word"):
            changed = copy.deepcopy(original)
            rows = changed["instructions"]
            if kind == "pop_register":
                next(row for row in rows if row["operation"] == "atlas.mxu_pop")["fields"]["dst"] = 34
            elif kind == "weight_slot":
                next(row for row in rows if row["operation"] == "atlas.mxu_matmul")["fields"]["weight_slot"] = 1
            elif kind == "wait":
                rows.remove(next(row for row in rows if row["operation"] == "atlas.dma_wait"))
            elif kind == "delay":
                delays = [row for row in rows if row["operation"] == "atlas.delay"]
                max(delays, key=lambda row: row["fields"]["cycles"])["fields"]["cycles"] = 0
            else:
                rows[0]["word_u32"] = 0x0200007F
            with self.subTest(kind=kind), self.assertRaises(NativeCompilationError):
                check_mxu0_stream(changed, activation_address=0x90001800,
                                  weight_address=0x90002000, output_address=0x90011800,
                                  activation=0, weight=1, pair=32)

    def test_native_mxu0_binary_on_selected_standalone_core(self) -> None:
        from merlin.semantic_compiler.search import SearchLimits

        keys = ("ATLAS_ARC_MODEL", "ATLAS_ARC_STATE", "ATLAS_MODELIR_ROOT")
        if not all(os.environ.get(key) for key in keys):
            self.skipTest("set selected-source-linked standalone AtlasCore paths")
        from test.test_mxu_reference import _tile

        destination = self.work / "mxu0-selected-core"
        self.binding.compile(
            self.snapshot, self.mxu0_request(), fixed_inputs={"x": 3, "w": 4},
            fixed_outputs=(35,), target_source=ROOT, destination=destination,
            limits=SearchLimits(),
        )
        words = struct.unpack(f'<{(destination / "program.bin").stat().st_size // 4}I',
                              (destination / "program.bin").read_bytes())
        model, state, modelir = (Path(os.environ[key]).resolve(strict=True) for key in keys)
        old_cwd = Path.cwd()
        sys.path.insert(0, str(modelir))
        try:
            os.chdir(modelir)
            from mlc.backends import cosim_atlas

            original = cosim_atlas.CosimCore

            class SelectedCore(original):
                def peek(self, name: str) -> int:
                    return super().peek("scalar/halt_now" if name == "io_halted" else name)

                def poke(self, name: str, value: int) -> None:
                    if name in ("io_dmaTL_d_bits_opcode", "io_dmaTL_d_bits_size"):
                        if name in self._S:
                            raise AssertionError(f"unexpected live ARC input {name}")
                        return
                    super().poke(name, value)

            cosim_atlas.CosimCore = SelectedCore
            try:
                one, small = 0x38, 0x18
                weight = _tile(((0, 0, one), (0, 1, small), (0, 2, small)))
                activation = _tile(((0, 0, one), (0, 1, small), (0, 2, small)))
                result = cosim_atlas.run_program(
                    model, state, words,
                    preload=[(0x90001800, activation), (0x90002000, weight),
                             (0x90011800, b"\xA5" * 2048),
                             (0x90012000, b"\x5A" * 32)],
                    max_cycles=5000,
                )
                output = result.slave.captured(0x90011800, 2048)
                self.assertTrue(result.halted)
                self.assertEqual(struct.unpack_from("<H", output)[0], 0x3F80)
                self.assertEqual(output[2:], bytes(2046))
                self.assertEqual(result.slave.captured(0x90001800, 1024), activation)
                self.assertEqual(result.slave.captured(0x90002000, 1024), weight)
                self.assertEqual(result.slave.captured(0x90012000, 32), b"\x5A" * 32)
            finally:
                cosim_atlas.CosimCore = original
        finally:
            os.chdir(old_cwd)
            sys.path.remove(str(modelir))

    def test_native_selection_matches_oot_physical_pair_and_binary(self) -> None:
        from merlin.semantic_compiler.search import SearchLimits

        request = self.request()
        destination = self.work / "emitted"
        manifest = self.compile_verified(
            request, fixed_inputs={"x": 0},
            fixed_outputs=(32,),
            destination=destination, limits=SearchLimits(),
        )
        physical = json.loads((destination / "physical-program.json").read_text())
        words = [row["word_u32"] for row in physical["instructions"]]
        self.assertEqual((destination / "program.bin").read_bytes(),
                         struct.pack(f"<{len(words)}I", *words))
        self.assertEqual(manifest["qualification_scope"], "diagnostic_only")
        self.assertTrue(manifest["native_check_fingerprint"])
        self.assertGreaterEqual(manifest["candidate_attempts"], 1)
        self.assertEqual(manifest["selected_registers"], [0, 1])
        self.assertEqual(manifest["atlas_opt_sha256"],
                         hashlib.sha256((Path(ATLAS_BIN) / "atlas-opt").read_bytes()).hexdigest())
        self.assertEqual(manifest["atlas_emit_sha256"],
                         hashlib.sha256((Path(ATLAS_BIN) / "atlas-emit").read_bytes()).hexdigest())
        self.assertGreater(len(words), 0)
        self.assertEqual(words[-1], 0x73)
        plan = json.loads((destination / "execution_plan.json").read_text())
        self.assertEqual((plan["input_address"], plan["output_address"]),
                         (0x90000000, 0x90010000))
        self.assertEqual(hashlib.sha256((destination / "atlas-machine.mlir").read_bytes()).hexdigest(),
                         manifest["machine_sha256"])

    def test_relocated_dram_slots_regenerate_addresses(self) -> None:
        from atlas_mlir_native.binding import check_copy_stream
        from merlin.semantic_compiler.search import SearchLimits

        request = self.request()
        baseline = self.work / "baseline"
        self.compile_verified(
            request, fixed_inputs={"x": 0}, fixed_outputs=(32,),
            destination=baseline, limits=SearchLimits(),
        )
        destination = self.work / "relocated"
        self.compile_verified(
            request, fixed_inputs={"x": 3}, fixed_outputs=(35,),
            destination=destination, limits=SearchLimits(),
        )
        plan = json.loads((destination / "execution_plan.json").read_text())
        self.assertEqual(plan["input_address"], 0x90001800)
        self.assertEqual(plan["output_address"], 0x90011800)
        program = json.loads((destination / "physical-program.json").read_text())
        check_copy_stream(program, input_address=plan["input_address"],
                          output_address=plan["output_address"], pair=0)
        self.assertNotEqual((destination / "program.bin").read_bytes(),
                            (baseline / "program.bin").read_bytes())

    def test_native_fp8_transpose_selects_xlu_and_emits_checked_stream(self) -> None:
        from atlas_mlir_native.binding import check_transpose_stream
        from merlin.semantic_compiler.linalg_bridge import translate_linalg_text
        from merlin.semantic_compiler.search import SearchLimits

        translated = translate_linalg_text(
            (ROOT / "test/examples/native_fp8_transpose_linalg.mlir").read_text(),
            entry="transpose", target_identity=self.binding.profile().target_identity,
            lowering_policy="diagnostic",
        )
        self.assertEqual(translated.source_operations,
                         ("tensor.empty", "linalg.transpose", "func.return"))
        request = translated.request
        destination = self.work / "transpose"
        manifest = self.compile_verified(
            request, fixed_inputs={"arg0": 3},
            fixed_outputs=(35,),
            destination=destination, limits=SearchLimits(),
        )
        plan = json.loads((destination / "execution_plan.json").read_text())
        self.assertEqual((plan["operation"], plan["dtype"], plan["input_bytes"]),
                         ("transpose", "fp8_e4m3", 1024))
        self.assertEqual((plan["input_address"], plan["output_address"]),
                         (0x90001800, 0x90011800))
        self.assertEqual(manifest["selected_registers"], [0, 1])
        program = json.loads((destination / "physical-program.json").read_text())
        self.assertEqual(sum(row["operation"] == "atlas.xlu_transpose"
                             for row in program["instructions"]), 1)
        check_transpose_stream(program, input_address=plan["input_address"],
                               output_address=plan["output_address"],
                               source=0, result=1)
        words = [row["word_u32"] for row in program["instructions"]]
        self.assertEqual((destination / "program.bin").read_bytes(),
                         struct.pack(f"<{len(words)}I", *words))

    def test_native_packed_value_transpose_feeds_later_mxu(self) -> None:
        from atlas_mlir_native.binding import (
            MXU0_INPUT_DOMAIN,
            MXU0_POLICY,
            PACK_ATTRS,
            PACK_POLICY,
            SCALED_PACK_DOMAIN,
            SCALED_PACK_POLICY,
            _mxu0_maps,
            _transpose_maps,
            check_mxu_bf16_dag_stream,
        )
        from merlin.semantic_compiler.model import (
            KernelRequest,
            SemanticNode,
            TensorType,
            pointwise_identity_maps,
        )
        from merlin.semantic_compiler.search import SearchLimits
        from merlin.semantic_compiler.target_binding import NativeCompilationError

        from tools.atlas_virtual_evaluator import (
            RuntimeInputs,
            Tile,
            evaluate,
            parse_program,
        )

        fp8 = TensorType((32, 32), "fp8_e4m3", "raw-bit-pattern")
        packed = TensorType((32, 32), "fp8_e4m3", SCALED_PACK_POLICY)
        bf16 = TensorType((32, 32), "bf16", MXU0_POLICY)
        inputs = tuple(SemanticNode(
            name, "input", (), fp8,
            attrs=(("input_domain", MXU0_INPUT_DOMAIN),), effect="input",
        ) for name in ("a", "w", "b"))
        request = KernelRequest(
            (*inputs,
             SemanticNode("v", "matmul", ("a", "w"), bf16,
                          index_maps=_mxu0_maps()),
             SemanticNode("vpack", "pack_fp8", ("v",), packed,
                          attrs=(("input_domain", SCALED_PACK_DOMAIN),
                                 ("scale_code", 127)),
                          index_maps=pointwise_identity_maps(2, 1)),
             SemanticNode("vt", "transpose", ("vpack",), packed,
                          index_maps=_transpose_maps()),
             SemanticNode("pv", "matmul", ("b", "vt"), bf16,
                          index_maps=_mxu0_maps())),
            ("pv",), ("external",),
            (("a", "external"), ("w", "external"), ("b", "external")),
            self.binding.profile().target_identity, lowering_policy="diagnostic",
        )
        destination = self.work / "native-packed-value-transpose"
        self.binding.compile(
            self.snapshot, request, fixed_inputs={"a": 3, "w": 4, "b": 5},
            fixed_outputs=(35,), target_source=ROOT, destination=destination,
            limits=SearchLimits(),
        )
        program = json.loads((destination / "physical-program.json").read_text())
        self.assertEqual(sum(row["operation"] == "atlas.xlu_transpose"
                             for row in program["instructions"]), 1)
        selected = self.snapshot.select(
            request, fixed_inputs={"a": 3, "w": 4, "b": 5},
            fixed_outputs=(35,), limits=SearchLimits(),
        )
        self.assertEqual(selected.status, "selected")
        assert selected.graph and selected.allocation and selected.rules
        checker = {
            "graph": selected.graph, "order": selected.allocation.order,
            "names": [selected.rules.symbols[value.symbol].get("descriptor", {}).get("name")
                      for value in selected.graph.values],
            "assignment": selected.allocation.addresses,
            "source_ids": ("a", "w", "b"),
            "mxu_inputs": {"v": ("a", "w"), "pv": ("b", "vt")},
            "input_slots": (3, 4, 5), "output_slots": (35,),
            "pack_scale_codes": {"vpack": 127},
            "transpose_inputs": {"vt": "vpack"},
        }
        check_mxu_bf16_dag_stream(program, **checker)
        changed = copy.deepcopy(program)
        next(row for row in changed["instructions"]
             if row["operation"] == "atlas.xlu_transpose")["fields"]["src"] += 1
        with self.assertRaises(NativeCompilationError):
            check_mxu_bf16_dag_stream(changed, **checker)
        changed = copy.deepcopy(program)
        xlu_index = next(index for index, row in enumerate(changed["instructions"])
                         if row["operation"] == "atlas.xlu_transpose")
        self.assertEqual(changed["instructions"][xlu_index + 1]["operation"], "atlas.delay")
        changed["instructions"][xlu_index + 1]["fields"]["cycles"] = 0
        with self.assertRaises(NativeCompilationError):
            check_mxu_bf16_dag_stream(changed, **checker)
        virtual = parse_program((destination / "atlas-virtual.mlir").read_text(),
                                function="native_tile")
        result = evaluate(virtual, RuntimeInputs(tiles={
            0: Tile("fp8", (0x38,) * 1024),
            1: Tile("fp8", (0x10,) * 1024),
            2: Tile("fp8", (0x38,) * 1024),
        }))
        self.assertEqual(set(result.outputs[0].bits), {0x4200})
        unit_packed = TensorType((32, 32), "fp8_e4m3", PACK_POLICY)
        unit_request = KernelRequest(
            (*request.nodes[:4],
             SemanticNode("vpack", "pack_fp8", ("v",), unit_packed,
                          attrs=PACK_ATTRS,
                          index_maps=pointwise_identity_maps(2, 1)),
             SemanticNode("vt", "transpose", ("vpack",), unit_packed,
                          index_maps=_transpose_maps()),
             request.nodes[-1]),
            request.outputs, request.output_storages, request.input_storages,
            request.target_identity, lowering_policy="diagnostic",
        )
        unit_destination = self.work / "native-unit-packed-value-transpose"
        self.binding.compile(
            self.snapshot, unit_request, fixed_inputs={"a": 3, "w": 4, "b": 5},
            fixed_outputs=(35,), target_source=ROOT, destination=unit_destination,
            limits=SearchLimits(),
        )
        unit_selection = json.loads((unit_destination / "native-selection.json").read_text())
        self.assertIn("atlas.xlu_transpose_unit_fp8_tile",
                      [row["descriptor"] for row in unit_selection["values"]])
        unit_virtual = parse_program((unit_destination / "atlas-virtual.mlir").read_text(),
                                     function="native_tile")
        unit_result = evaluate(unit_virtual, RuntimeInputs(tiles={
            0: Tile("fp8", (0x38,) * 1024),
            1: Tile("fp8", (0x10,) * 1024),
            2: Tile("fp8", (0x38,) * 1024),
        }))
        self.assertEqual(set(unit_result.outputs[0].bits), {0x4200})
        for index, altered in enumerate((
            SemanticNode("vt", "transpose", ("vpack",), fp8,
                         index_maps=_transpose_maps()),
            SemanticNode("vt", "transpose", ("vpack",), packed,
                         index_maps=pointwise_identity_maps(2, 1)),
        )):
            invalid = KernelRequest(
                (*request.nodes[:-2], altered, request.nodes[-1]),
                request.outputs, request.output_storages, request.input_storages,
                request.target_identity, lowering_policy="diagnostic",
            )
            with self.subTest(index=index), self.assertRaisesRegex(
                NativeCompilationError, "only bounded BF16/FP8 tile graphs"
            ):
                self.binding.compile(
                    self.snapshot, invalid, fixed_inputs={"a": 3, "w": 4, "b": 5},
                    fixed_outputs=(35,), target_source=ROOT,
                    destination=self.work / f"invalid-packed-transpose-{index}",
                    limits=SearchLimits(),
                )

    def test_native_qk_softmax_value_attention_compiles_as_one_graph(self) -> None:
        from atlas_mlir_native.binding import (
            MXU0_INPUT_DOMAIN,
            MXU0_POLICY,
            SCALED_PACK_DOMAIN,
            SCALED_PACK_POLICY,
            _mxu0_maps,
            _transpose_maps,
        )
        from merlin.semantic_compiler.model import (
            KernelRequest,
            SemanticNode,
            TensorType,
            pointwise_identity_maps,
        )
        from merlin.semantic_compiler.search import SearchLimits

        from tools.atlas_virtual_evaluator import (
            RuntimeInputs,
            Tile,
            evaluate,
            parse_program,
        )

        softmax = KernelRequest.from_record(json.loads(
            (ROOT / "test/examples/native_qk_softmax_request.json").read_text()
        ))
        fp8 = TensorType((32, 32), "fp8_e4m3", "raw-bit-pattern")
        packed = TensorType((32, 32), "fp8_e4m3", SCALED_PACK_POLICY)
        bf16 = TensorType((32, 32), "bf16", MXU0_POLICY)
        pack_attrs = (("input_domain", SCALED_PACK_DOMAIN), ("scale_code", 127))
        request = KernelRequest(
            (*softmax.nodes,
             SemanticNode("value_weight", "input", (), fp8,
                          attrs=(("input_domain", MXU0_INPUT_DOMAIN),), effect="input"),
             SemanticNode("value", "matmul", ("arg0", "value_weight"), bf16,
                          index_maps=_mxu0_maps()),
             SemanticNode("probability_fp8", "pack_fp8", ("probability",), packed,
                          attrs=pack_attrs, index_maps=pointwise_identity_maps(2, 1)),
             SemanticNode("value_fp8", "pack_fp8", ("value",), packed,
                          attrs=pack_attrs, index_maps=pointwise_identity_maps(2, 1)),
             SemanticNode("value_transposed", "transpose", ("value_fp8",), packed,
                          index_maps=_transpose_maps()),
             SemanticNode("attention", "matmul",
                          ("probability_fp8", "value_transposed"), bf16,
                          index_maps=_mxu0_maps())),
            ("attention",), ("external",),
            (*softmax.input_storages, ("value_weight", "external")),
            softmax.target_identity, lowering_policy="diagnostic",
        )
        fixture_request = KernelRequest.from_record(json.loads(
            (ROOT / "test/examples/native_qk_softmax_value_request.json").read_text()
        ))
        self.assertEqual(fixture_request.digest(), request.digest())
        destination = self.work / "native-qk-softmax-value"
        self.compile_verified(
            request, fixed_inputs={"arg0": 3, "arg1": 4, "arg2": 5, "value_weight": 6},
            fixed_outputs=(35,),
            destination=destination, limits=SearchLimits(),
        )
        plan = json.loads((destination / "execution_plan.json").read_text())
        self.assertEqual(plan["transpose_source_nodes"], [
            {"result": "value_transposed", "source": "value_fp8"},
        ])
        self.assertIn({
            "producer": "value_transposed", "consumer": "attention",
            "required_domain": MXU0_INPUT_DOMAIN,
            "status": "caller_premise_unchecked",
        }, plan["producer_domain_obligations"])
        program = json.loads((destination / "physical-program.json").read_text())
        self.assertEqual(sum(row["operation"] == "atlas.xlu_transpose"
                             for row in program["instructions"]), 1)
        self.assertEqual(sum(row["operation"] == "atlas.mxu_matmul"
                             for row in program["instructions"]), 5)
        virtual = parse_program((destination / "atlas-virtual.mlir").read_text(),
                                function="native_tile")
        result = evaluate(virtual, RuntimeInputs(tiles={
            0: Tile("fp8", (0x38,) * 1024),
            1: Tile("fp8", (0x10,) * 1024),
            2: Tile("fp8", (0x10,) * 1024),
            3: Tile("fp8", (0x10,) * 1024),
        }))
        self.assertEqual(set(result.outputs[0].bits), {0x3F80})

    def test_native_attention_mlp_two_outputs_use_one_selected_graph(self) -> None:
        from merlin.semantic_compiler.model import KernelRequest
        from merlin.semantic_compiler.search import SearchLimits

        from tools.atlas_virtual_evaluator import (
            RuntimeInputs,
            Tile,
            evaluate,
            parse_program,
        )
        from tools.check_program_decode import audit_program
        from tools.check_variant_inventory import load_inventory

        attention = KernelRequest.from_record(json.loads(
            (ROOT / "test/examples/native_qk_softmax_value_request.json").read_text()
        ))
        request = KernelRequest.from_record(json.loads(
            (ROOT / "test/examples/native_attention_mlp_request.json").read_text()
        ))
        self.assertEqual(request.nodes[:len(attention.nodes)], attention.nodes)
        self.assertEqual(request.outputs, ("mlp_relu", "mlp_output"))
        self.assertEqual(sum(node.op == "matmul" for node in request.nodes), 7)
        self.assertEqual(sum(node.op == "pack_fp8" for node in request.nodes), 6)
        self.assertEqual(sum(node.op == "transpose" for node in request.nodes), 1)
        limits = SearchLimits.from_record(json.loads(
            (ROOT / "test/examples/native_attention_mlp_search_limits.json").read_text()
        ))
        self.assertEqual((limits.candidate_nodes, limits.candidates,
                          limits.orders_per_candidate), (48, 512, 64))
        abi = json.loads((ROOT / "test/examples/native_attention_mlp_abi.json").read_text())
        self.assertEqual(abi["fixed_inputs"], {
            "arg0": 3, "arg1": 4, "arg2": 5, "value_weight": 6,
            "mlp_weight1": 7, "mlp_weight2": 8,
        })
        self.assertEqual(abi["fixed_outputs"], [35, 36])
        destination = self.work / "native-attention-mlp-two-outputs"
        self.compile_verified(
            request, fixed_inputs=abi["fixed_inputs"],
            fixed_outputs=tuple(abi["fixed_outputs"]),
            destination=destination, limits=limits,
        )
        plan = json.loads((destination / "execution_plan.json").read_text())
        self.assertEqual(plan["output_addresses"], [0x90011800, 0x90012000])
        self.assertEqual(plan["output_physical_layouts"], ["bf16_pair_halves"] * 2)
        program = json.loads((destination / "physical-program.json").read_text())
        operations = [row["operation"] for row in program["instructions"]]
        self.assertEqual((operations.count("atlas.mxu_matmul"),
                          operations.count("atlas.vpu_pack"),
                          operations.count("atlas.xlu_transpose")), (7, 6, 1))
        audit = audit_program(
            load_inventory(), program, (destination / "program.bin").read_bytes(),
        )
        self.assertEqual((audit["status"], audit["word_count"]),
                         ("PASS", len(program["instructions"])))
        virtual = parse_program((destination / "atlas-virtual.mlir").read_text(),
                                function="native_tile")
        token = Tile("fp8", (0x38,) * 1024)
        projection_weight = Tile("fp8", (0x10,) * 1024)
        identity_weight = Tile("fp8", tuple(
            0x38 if row == column else 0
            for row in range(32) for column in range(32)
        ))
        for striped in (False, True):
            value_weight = (Tile("fp8", tuple(
                0x10 if feature % 2 == 0 else 0x08
                for feature in range(32) for _ in range(32)
            )) if striped else projection_weight)
            outputs = evaluate(virtual, RuntimeInputs(tiles={
                0: token, 1: projection_weight, 2: projection_weight,
                3: value_weight, 4: identity_weight, 5: identity_weight,
            })).outputs
            expected = tuple(0x3F80 if not striped or column % 2 == 0 else 0x3F00
                             for _row in range(32) for column in range(32))
            self.assertEqual(tuple(outputs), (0, 1))
            self.assertEqual(outputs[0].bits, expected)
            self.assertEqual(outputs[1].bits, expected)

    def test_transpose_checker_rejects_register_scratch_and_wait_mutations(self) -> None:
        from atlas_mlir_native.binding import check_transpose_stream
        from merlin.semantic_compiler.search import SearchLimits
        from merlin.semantic_compiler.target_binding import NativeCompilationError

        destination = self.work / "transpose-checker"
        self.binding.compile(
            self.snapshot, self.transpose_request(), fixed_inputs={"x": 3},
            fixed_outputs=(35,), target_source=ROOT, destination=destination,
            limits=SearchLimits(),
        )
        program = json.loads((destination / "physical-program.json").read_text())
        kwargs = {"input_address": 0x90001800, "output_address": 0x90011800,
                  "source": 0, "result": 1}
        for kind in ("source", "result", "scratch", "wait", "delay", "model_config_word"):
            changed = copy.deepcopy(program)
            rows = changed["instructions"]
            if kind in {"source", "result"}:
                xlu = next(row for row in rows if row["operation"] == "atlas.xlu_transpose")
                xlu["fields"]["src" if kind == "source" else "dst"] += 1
            elif kind == "scratch":
                next(row for row in rows if row["operation"] == "atlas.vstore")["fields"]["base"] = 0
            elif kind == "delay":
                next(row for row in rows if row["operation"] == "atlas.delay")["fields"]["cycles"] = 0
            elif kind == "model_config_word":
                rows[0]["word_u32"] = 0x0200007F
            else:
                rows.remove(next(row for row in rows if row["operation"] == "atlas.dma_wait"))
            with self.subTest(kind=kind), self.assertRaises(NativeCompilationError):
                check_transpose_stream(changed, **kwargs)

    def test_fp8_transpose_refuses_wrong_maps_and_policy(self) -> None:
        from atlas_mlir_native.binding import _transpose_maps
        from merlin.semantic_compiler.model import IndexMap
        from merlin.semantic_compiler.search import SearchLimits
        from merlin.semantic_compiler.target_binding import NativeCompilationError

        wrong_maps = (_transpose_maps()[1], _transpose_maps()[0])
        for index, request in enumerate((
            self.transpose_request(policy="ordered-bf16"),
            self.transpose_request(maps=wrong_maps),
            self.transpose_request(maps=(IndexMap(2, ((1, 0), (0, 1)), (0, 1)),
                                         _transpose_maps()[1])),
        )):
            destination = self.work / f"transpose-refused-{index}"
            with self.subTest(index=index), self.assertRaises(NativeCompilationError) as caught:
                self.binding.compile(
                    self.snapshot, request, fixed_inputs={"x": 0}, fixed_outputs=(32,),
                    target_source=ROOT, destination=destination, limits=SearchLimits(),
                )
            self.assertEqual(caught.exception.status, "unsupported_semantics")
            self.assertFalse(destination.exists())

    def test_unqualified_and_unsupported_requests_refuse_before_emission(self) -> None:
        from merlin.semantic_compiler.search import SearchLimits
        from merlin.semantic_compiler.target_binding import NativeCompilationError

        cases = (
            (self.request(mode="strict-native"), {"x": 0}, (32,), "unqualified_target"),
            (self.request(shape=(31, 32)), {"x": 0}, (32,), "unsupported_semantics"),
            (self.request(), {"x": 0}, (0,), "unqualified_target"),
        )
        for index, (request, inputs, outputs, status) in enumerate(cases):
            destination = self.work / f"refused-{index}"
            with self.subTest(status=status), self.assertRaises(NativeCompilationError) as caught:
                self.binding.compile(
                    self.snapshot, request, fixed_inputs=inputs, fixed_outputs=outputs,
                    target_source=ROOT, destination=destination, limits=SearchLimits(),
                )
            self.assertEqual(caught.exception.status, status)
            self.assertFalse(destination.exists())

    def test_emitted_stream_checker_rejects_changed_address_pair_scratch_wait_and_completion(self) -> None:
        from atlas_mlir_native.binding import check_copy_stream
        from merlin.semantic_compiler.search import SearchLimits
        from merlin.semantic_compiler.target_binding import NativeCompilationError

        destination = self.work / "checker-source"
        self.binding.compile(
            self.snapshot, self.request(), fixed_inputs={"x": 0}, fixed_outputs=(32,),
            target_source=ROOT, destination=destination, limits=SearchLimits(),
        )
        program = json.loads((destination / "physical-program.json").read_text())
        kwargs = {"input_address": 0x90000000, "output_address": 0x90010000, "pair": 0}
        check_copy_stream(program, **kwargs)
        for kind in ("address", "pair", "scratch", "wait", "completion"):
            changed = copy.deepcopy(program)
            rows = changed["instructions"]
            if kind == "address":
                next(row for row in rows if row["operation"] == "atlas.upper")["fields"]["immediate"] += 1
            elif kind == "pair":
                next(row for row in rows if row["operation"] == "atlas.vload")["fields"]["dst"] = 4
            elif kind == "scratch":
                next(row for row in rows if row["operation"] == "atlas.dma")["fields"]["reg"] = 2
            elif kind == "completion":
                next(row for row in rows if row["operation"] == "atlas.csr")["fields"]["source"] = 0
            else:
                rows.remove(next(row for row in rows if row["operation"] == "atlas.dma_wait"))
            with self.subTest(kind=kind), self.assertRaises(NativeCompilationError):
                check_copy_stream(changed, **kwargs)

    def test_declared_bf16_multiplier_constant_is_placed_and_checked(self) -> None:
        from dataclasses import replace

        from atlas_mlir_native.binding import (
            BF16_MUL_ATTRS,
            MXU0_INPUT_DOMAIN,
            MXU0_POLICY,
            _mxu0_maps,
        )
        from atlas_mlir_native.layout import (
            pack_bf16_pair_halves,
            unpack_bf16_pair_halves,
        )
        from merlin.semantic_compiler.model import (
            ConstantBinding,
            KernelRequest,
            SemanticNode,
            TensorType,
            pointwise_identity_maps,
        )
        from merlin.semantic_compiler.search import SearchLimits
        from merlin.semantic_compiler.target_binding import (
            NativeCompilationError,
            verify_native_publication,
        )

        from tools.atlas_virtual_evaluator import (
            RuntimeInputs,
            Tile,
            evaluate,
            parse_program,
        )

        fp8 = TensorType((32, 32), "fp8_e4m3", "raw-bit-pattern")
        bf16 = TensorType((32, 32), "bf16", "raw-bit-pattern")
        logical = struct.pack("<1024H", *(0x3e00 + index % 64 for index in range(1024)))
        request = KernelRequest(
            (
                SemanticNode("x", "input", (), fp8,
                             attrs=(("input_domain", MXU0_INPUT_DOMAIN),), effect="input"),
                SemanticNode("w", "input", (), fp8,
                             attrs=(("input_domain", MXU0_INPUT_DOMAIN),), effect="input"),
                SemanticNode("scale", "constant", (), bf16,
                             attrs=(BF16_MUL_ATTRS[0],), effect="constant"),
                SemanticNode("mm", "matmul", ("x", "w"),
                             TensorType((32, 32), "bf16", MXU0_POLICY),
                             index_maps=_mxu0_maps()),
                SemanticNode("product", "mul_bf16", ("mm", "scale"), bf16,
                             attrs=BF16_MUL_ATTRS,
                             index_maps=pointwise_identity_maps(2, 2)),
            ),
            ("product",), ("external",),
            (("x", "external"), ("w", "external")),
            self.binding.profile().target_identity, lowering_policy="diagnostic",
            constants=(ConstantBinding("scale", "bf16-le-raw", logical.hex()),),
            physical_extents=(("x", 1), ("w", 1), ("scale", 1)),
        )
        destination = self.work / "mxu0-declared-bf16-constant"
        fixed = {"x": 3, "w": 4, "scale": 5}
        manifest = self.compile_verified(
            request, fixed_inputs=fixed,
            fixed_outputs=(35,),
            destination=destination, limits=SearchLimits(),
        )
        rows = manifest["constant_inputs"]
        self.assertEqual(len(rows), 1)
        row = rows[0]
        payload = (destination / row["path"]).read_bytes()
        self.assertEqual(payload, pack_bf16_pair_halves(logical))
        self.assertEqual(unpack_bf16_pair_halves(payload), logical)
        self.assertEqual(payload[:2], struct.pack("<H", 0x3e00))
        self.assertEqual(payload[32:34], struct.pack("<H", 0x3e20))
        self.assertEqual(payload[1024:1026], struct.pack("<H", 0x3e10))
        self.assertEqual(row["address"], 0x90000000 + 5 * 2048)
        self.assertEqual(manifest["stage_artifacts"][row["path"]], hashlib.sha256(payload).hexdigest())
        plan = json.loads((destination / "execution_plan.json").read_text())
        self.assertEqual(plan["input_addresses"], [
            0x90000000 + slot * 2048 for slot in (3, 4, 5)
        ])
        self.assertEqual(plan["input_payload_bytes"], [1024, 1024, 2048])
        self.assertEqual(plan["constant_inputs"], rows)
        virtual = parse_program((destination / "atlas-virtual.mlir").read_text(),
                                function="native_tile")
        result = evaluate(virtual, RuntimeInputs(tiles={
            0: Tile("fp8", (0x38,) * 1024),
            1: Tile("fp8", tuple(
                0x38 if row == column else 0
                for row in range(32) for column in range(32)
            )),
            2: Tile("bf16", struct.unpack("<1024H", logical)),
        }))
        self.assertEqual(result.outputs[0].bits, struct.unpack("<1024H", logical))
        (destination / row["path"]).write_bytes(bytes((payload[0] ^ 1,)) + payload[1:])
        with self.assertRaisesRegex(ValueError, "stage artifact changed"):
            verify_native_publication(
                destination, manifest, engine="merlin_native",
                request_digest=request.digest(), target_identity=request.target_identity,
            )
        (destination / row["path"]).write_bytes(payload)

        wrong = replace(request, constants=(ConstantBinding(
            "scale", "bf16-le-raw", (b"\x00\x00" + logical[2:]).hex(),
        ),))
        with self.assertRaisesRegex(NativeCompilationError, "finite-normal domain"):
            self.binding.compile(
                self.snapshot, wrong, fixed_inputs=fixed, fixed_outputs=(35,),
                target_source=ROOT, destination=self.work / "invalid-bf16-constant",
                limits=SearchLimits(),
            )
        self.assertFalse((self.work / "invalid-bf16-constant").exists())

    def test_native_copy_executes_on_selected_standalone_core(self) -> None:
        keys = ("ATLAS_ARC_MODEL", "ATLAS_ARC_STATE", "ATLAS_MODELIR_ROOT")
        if not all(os.environ.get(key) for key in keys):
            self.skipTest("set selected-source-linked standalone AtlasCore paths")
        from merlin.semantic_compiler.search import SearchLimits

        destination = self.work / "executed"
        self.binding.compile(
            self.snapshot, self.request(), fixed_inputs={"x": 0}, fixed_outputs=(32,),
            target_source=ROOT, destination=destination, limits=SearchLimits(),
        )
        data = (destination / "program.bin").read_bytes()
        words = struct.unpack(f"<{len(data) // 4}I", data)
        old_cwd = Path.cwd()
        modelir = Path(os.environ["ATLAS_MODELIR_ROOT"]).resolve(strict=True)
        sys.path.insert(0, str(modelir))
        try:
            os.chdir(modelir)
            from mlc.backends import cosim_atlas

            original = cosim_atlas.CosimCore

            class SelectedCore(original):
                def peek(self, name: str) -> int:
                    return super().peek("scalar/halt_now" if name == "io_halted" else name)

                def poke(self, name: str, value: int) -> None:
                    if name in ("io_dmaTL_d_bits_opcode", "io_dmaTL_d_bits_size"):
                        if name in self._S:
                            raise AssertionError(f"unexpected ARC input {name}")
                        return
                    super().poke(name, value)

            cosim_atlas.CosimCore = SelectedCore
            try:
                for seed in (3, 17):
                    source = bytes((seed + 37 * index) & 255 for index in range(2048))
                    guard = b"\x5a" * 32
                    result = cosim_atlas.run_program(
                        Path(os.environ["ATLAS_ARC_MODEL"]), Path(os.environ["ATLAS_ARC_STATE"]),
                        words,
                        preload=[(0x90000000, source), (0x90010000, b"\xa5" * 2048),
                                 (0x90010800, guard)],
                        max_cycles=10000,
                    )
                    self.assertTrue(result.halted)
                    self.assertEqual(result.slave.captured(0x90010000, 2048), source)
                    self.assertEqual(result.slave.captured(0x90000000, 2048), source)
                    self.assertEqual(result.slave.captured(0x90010800, 32), guard)
            finally:
                cosim_atlas.CosimCore = original
        finally:
            os.chdir(old_cwd)
            sys.path.remove(str(modelir))


if __name__ == "__main__":
    unittest.main()
