"""DMA placement refusals before tool execution, plus scratchpad export replay."""
from contextlib import ExitStack
from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from atlas_mlir_native.binding import DRAM_BASE, TILE_BYTES
from atlas_mlir_native.exact_float_device import OPS, prepare_plan
from atlas_mlir_native.pointwise import (
    BF16_POINTWISE_POLICY, compile_pointwise, plan_abi, prepare_execution_plan, prepare_source,
)
from atlas_mlir_native.pointwise_c import prepare_precision_c_package
from merlin.semantic_compiler.search import SearchLimits
from merlin.semantic_compiler.target_binding import NativeCompilationError
from test_exact_float_device import request as float_request
from test_pointwise_streaming import request

ROOT = Path(__file__).resolve().parents[1]
BIN = Path(os.environ.get("ATLAS_OOT_BIN_DIR", ROOT / "build/bin"))


def precision_source(shape=(32, 32)):
    source = request(shape)
    a, b, result = source.nodes
    return replace(source, nodes=(replace(a, attrs=()), replace(b, attrs=()),
                   replace(result, op="bf16_mul", attrs=(("numeric_semantics", "ieee754-bf16-mul-rne-v1"),))))


class LeafMemoryWindowsTest(unittest.TestCase):
    def test_exact_float_refuses_invalid_complete_spans_without_tools(self):
        with patch("atlas_mlir_native.binding._run", side_effect=AssertionError("tool invoked")):
            for op in OPS:
                for base in (True, -32, 0x08000001, 0x07ffffe0, 0x08010000,
                             0x0800ffe0, 0xffffffe0, 1 << 32):
                    with self.subTest(op=op, base=base), self.assertRaisesRegex(ValueError, "DMA memory windows"):
                        prepare_plan(float_request(op), io_memory_base=base)

    def test_exact_float_default_records_match_before_relocation_support(self):
        # Captured from the parent of the scratchpad-leaf commit, for every op.
        expected = json.loads((ROOT / "test/fixtures/exact-float-default-records.json").read_text())
        for op in OPS:
            with self.subTest(op=op):
                self.assertEqual(prepare_plan(float_request(op)).record(), expected[op])
                self.assertEqual(prepare_plan(float_request(op), io_memory_base=DRAM_BASE).record(), expected[op])

    def test_exact_float_scratchpad_span_may_end_at_window_boundary(self):
        for op in OPS:
            source = float_request(op)
            size = prepare_plan(source).workspace_bytes
            plan = prepare_plan(source, io_memory_base=0x08010000 - size)
            self.assertEqual(plan.status_address + 32, 0x08010000)
            with self.assertRaises(ValueError):
                prepare_plan(source, io_memory_base=plan.io_memory_base + 32)

    def test_pointwise_refuses_before_tools_or_destination_creation(self):
        source = request((32, 32))
        with tempfile.TemporaryDirectory() as tmp, patch(
            "atlas_mlir_native.pointwise.AtlasNativeBinding.compile", side_effect=AssertionError("tool invoked")
        ):
            destination = Path(tmp) / "compiled"
            for base in (True, "invalid", -32, 0x08000001, 0x07ffffe0,
                         0x08010000, 0x0800f800, 0xfffff800, 1 << 32):
                with self.subTest(base=base), self.assertRaisesRegex(NativeCompilationError, "DMA memory windows"):
                    compile_pointwise(None, source, **plan_abi(source), target_source=ROOT,
                                      destination=destination, limits=SearchLimits(), io_memory_base=base)
                self.assertFalse(destination.exists())

    def package_placement(self, addresses, output, *, shape=(32, 32)):
        precision = prepare_source(precision_source(shape), bf16_policy=BF16_POINTWISE_POLICY)
        _, plan, _ = prepare_execution_plan(precision.request)
        # Isolate placement validation from earlier inventory/hash validation.
        # All address checks and subsequent source replay remain real.
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            compiled = Path(tmp)
            (compiled / "body").mkdir()
            (compiled / "body/manifest.json").write_text('{}')
            (compiled / "manifest.json").write_text(json.dumps({
                "schema": "atlas.native_pointwise_compilation.v1", "selected_body_descriptor": {},
            }))
            (compiled / "execution_plan.json").write_text(json.dumps({
                "schema": "atlas.pointwise_execution_plan.v1", "shape": list(plan.shape),
                "input_addresses": addresses, "input_buffer_bytes": list(plan.input_packed_bytes),
                "output_addresses": [output], "output_bytes": plan.packed_bytes,
                "output_physical_layouts": ["mn_tiles_bf16_pair_halves"], "tiling": plan.record(),
            }))
            for name in ("check_io_memory_manifest", "verify_native_publication"):
                stack.enter_context(patch("atlas_mlir_native.pointwise_c." + name))
            stack.enter_context(patch("atlas_mlir_native.pointwise_c.hashlib.sha256")).return_value.hexdigest.return_value = None
            stack.enter_context(patch("atlas_mlir_native.pointwise_c.selected_body_descriptor", return_value={}))
            stack.enter_context(patch("atlas_mlir_native.pointwise_c._run", side_effect=AssertionError("tool invoked")))
            (compiled / "atlas-virtual.mlir").write_text('')
            stack.enter_context(patch("atlas_mlir_native.pointwise_c.virtual_program", side_effect=ReplayReached))
            prepare_precision_c_package(precision, compiled, "leaf", target_source=ROOT)

    def test_packaging_refuses_bad_addresses_and_complete_spans_without_tools(self):
        for address in (True, -TILE_BYTES, 0x08000020, 0x07fff800,
                        0x08010000, 1 << 32):
            with self.subTest(address=address), self.assertRaisesRegex(ValueError, "byte address"):
                self.package_placement([address, 0x08002000], 0x08004000)
        for address in (0x0800f800, 0xfffff800):
            with self.subTest(crossing=address), self.assertRaisesRegex(ValueError, "byte address"):
                self.package_placement([address, 0x08002000], 0x08004000, shape=(32, 64))
        with self.assertRaisesRegex(ValueError, "overlap"):
            self.package_placement([0x08000000, 0x08000000], 0x08004000)
        with self.assertRaisesRegex(ValueError, "selected memory window"):
            self.package_placement([DRAM_BASE, DRAM_BASE + 0x40000], DRAM_BASE + 0x42000)

    def test_packaging_admits_dram_and_scratchpad_before_source_replay(self):
        for base in (DRAM_BASE, 0x80000000, 0x08000000, 0x0800e800):
            with self.subTest(base=base), self.assertRaises(ReplayReached):
                self.package_placement([base, base + TILE_BYTES], base + 2 * TILE_BYTES)


class ReplayReached(Exception):
    pass


@unittest.skipUnless((BIN / "atlas-opt").is_file(), "select built Atlas tools")
class ScratchpadPointwiseExportTest(unittest.TestCase):
    def test_compile_and_package_scratchpad_with_full_replay(self):
        from test_looped_mnk import build_snapshot
        from atlas_mlir_native.pointwise_c import prepare_c_package
        source = precision_source()
        precision = prepare_source(source, bf16_policy=BF16_POINTWISE_POLICY)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = build_snapshot(root / "native")
            compiled = root / "compiled"
            compile_pointwise(snapshot, precision.request, **plan_abi(precision.request),
                              target_source=ROOT, destination=compiled, limits=SearchLimits(),
                              io_memory_base=0x08000000)
            self.assertEqual(json.loads((compiled / "execution_plan.json").read_text())["input_addresses"][0], 0x08000000)
            prepare_c_package(source, compiled, "scratchpad_leaf", target_source=ROOT,
                              bf16_policy=BF16_POINTWISE_POLICY)
