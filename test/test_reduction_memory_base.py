"""Reduction io_memory_base: invalid bases and incomplete DMA spans refuse before any tool runs."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from atlas_mlir_native.binding import AtlasNativeBinding, DRAM_BASE, dram_slots, validate_explicit_io_base, _tile_io_base, io_memory_base
from atlas_mlir_native.reduction import compile_reduction, plan_abi, request_for
from atlas_mlir_native.options import CodegenOptions, codegen_options
from merlin.semantic_compiler.target_binding import NativeCompilationError

ROOT = Path(__file__).resolve().parents[1]


class Reached(Exception):
    pass


class ReductionMemoryBaseTest(unittest.TestCase):
    def setUp(self):
        self.options = codegen_options(CodegenOptions(io_memory_bytes=67108864))
        self.options.__enter__()
        self.addCleanup(self.options.__exit__, None, None, None)
        self.request = request_for((32, 32), (1,))
        self.abi = plan_abi(self.request)

    def compile(self, base, destination):
        return compile_reduction(None, self.request, fixed_inputs=self.abi["fixed_inputs"],
                                 fixed_outputs=tuple(self.abi["fixed_outputs"]), target_source=ROOT,
                                 destination=destination, limits=None, io_memory_base=base)

    def test_invalid_bases_and_spans_refuse_before_tools(self):
        with tempfile.TemporaryDirectory() as tmp, patch(
                "atlas_mlir_native.reduction.AtlasNativeBinding.compile", side_effect=AssertionError("tool invoked")):
            destination = Path(tmp) / "compiled"
            for base in (True, "0x200c0000", -2048, 0x200C0001, 0x200C0400, 0x07FFF800, 0x0800F800,
                         0x08010000, 0x20000000, 0x200BF800, 0x2017F800, 0x20180000, 1 << 32):
                with self.subTest(base=base), self.assertRaises(NativeCompilationError):
                    self.compile(base, destination)
                self.assertFalse(destination.exists())

    def test_native_dispatch_forwards_base_for_nontile_reduction(self):
        with codegen_options(CodegenOptions(io_memory_bytes=786432)):
            snapshot = SimpleNamespace(profile=SimpleNamespace(
                banks=[SimpleNamespace(name="external", capacity=dram_slots())]))
            with patch("atlas_mlir_native.resident_graph.is_resident_graph", return_value=False), patch(
                    "atlas_mlir_native.reduction.compile_reduction", return_value={}) as compile_body:
                AtlasNativeBinding.compile(snapshot, self.request, fixed_inputs=self.abi["fixed_inputs"],
                    fixed_outputs=tuple(self.abi["fixed_outputs"]), target_source=ROOT,
                    destination=ROOT / "unused", limits=None, io_memory_base=0x200C0000)
        self.assertEqual(compile_body.call_args.kwargs["io_memory_base"], 0x200C0000)
        self.assertIsNone(_tile_io_base.get())
        self.assertEqual(io_memory_base(), DRAM_BASE)

    def test_c_export_refuses_invalid_bases_before_publication_reads(self):
        from atlas_mlir_native.reduction_c import prepare_c_package
        for base in (True, "0x200c0000", -2048, 0x200C0001, 0x20000000, 0x20180000, 1 << 32):
            with self.subTest(base=base), self.assertRaises(ValueError):
                prepare_c_package(None, None, "reduction", target_source=ROOT, io_memory_base=base)

    def test_none_base_uses_invocation_default_before_tools_or_publication_reads(self):
        from atlas_mlir_native.memory_windows import dma_span_in_window
        from atlas_mlir_native.reduction_c import prepare_c_package

        for base in (DRAM_BASE, 0x200C0000):
            with self.subTest(base=hex(base)), codegen_options(
                    CodegenOptions(io_memory_base=base, io_memory_bytes=786432)):
                with tempfile.TemporaryDirectory() as tmp, patch(
                        "atlas_mlir_native.reduction.AtlasNativeBinding.compile", side_effect=Reached), patch(
                        "atlas_mlir_native.binding.validate_explicit_io_base",
                        wraps=validate_explicit_io_base) as validate:
                    with self.assertRaises(Reached):
                        self.compile(None, Path(tmp) / "compiled")
                    validate.assert_called_once_with(base, alignment=2048)
                with patch("atlas_mlir_native.reduction_c.prepare_source", side_effect=Reached), patch(
                        "atlas_mlir_native.reduction_c.dma_span_in_window", wraps=dma_span_in_window) as span:
                    with self.assertRaises(Reached):
                        prepare_c_package(self.request, None, "reduction", target_source=ROOT, io_memory_base=None)
                    span.assert_called_once_with(base, 1)

    def test_admitted_windows_reach_body_compilation(self):
        for base in (DRAM_BASE, 0x80000000, 0x08000000, 0x0800E800, 0x200C0000, 0x2017E800):
            with tempfile.TemporaryDirectory() as tmp, patch(
                    "atlas_mlir_native.reduction.AtlasNativeBinding.compile", side_effect=Reached), \
                    self.subTest(base=hex(base)), self.assertRaises(Reached):
                self.compile(base, Path(tmp) / "compiled")


if __name__ == "__main__":
    unittest.main()
