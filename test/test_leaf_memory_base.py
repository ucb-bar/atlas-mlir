"""Custom leaf bases preserve sizes and refuse invalid complete DMA spans."""
import unittest
from dataclasses import replace
from atlas_mlir_native import activation_quantize_device as quant, row_loop
from atlas_mlir_native.binding import DRAM_BASE
from merlin.semantic_compiler.target_binding import NativeCompilationError
from merlin.semantic_compiler.codegen import CodegenOptions, codegen_options
from test_already_quantized import quantizer_source
from test_tiled_softmax import source_request


class LeafMemoryBaseTest(unittest.TestCase):
    def setUp(self):
        self.options = codegen_options(CodegenOptions(io_memory_bytes=67108864))
        self.options.__enter__()
        self.addCleanup(self.options.__exit__, None, None, None)
        self.cases = [(quant.prepare_plan, quantizer_source(count=2048)),
                      (row_loop.prepare_plan, source_request((32, 65), masked=True))]

    def test_invalid_bases_and_full_span_refusal(self):
        for prepare, request in self.cases:
            for base in (True, None, 0, -32, 0x40000, 0x08008001,
                         0x0800ffe0, 0xffffffe0, 1 << 32):
                with self.subTest(prepare=prepare.__module__, base=base):
                    with self.assertRaises((ValueError, NativeCompilationError)):
                        prepare(request, io_memory_base=base)

    def test_default_and_relocated_geometry(self):
        for prepare, request in self.cases:
            original = prepare(request)
            self.assertEqual(original, prepare(request, io_memory_base=DRAM_BASE))
            moved = prepare(request, io_memory_base=0x08008000)
            self.assertEqual(moved, replace(original, io_memory_base=0x08008000))
        original = row_loop.prepare_plan(self.cases[1][1])
        moved = row_loop.prepare_plan(self.cases[1][1], io_memory_base=0x08008000)
        for offset in (0, original.readonly_bytes, original.readonly_bytes+original.output_bytes):
            self.assertEqual(moved.address(DRAM_BASE+offset, 0), original.address(DRAM_BASE+offset, 0)-DRAM_BASE+0x08008000)
        for offset in (-32, original.end-DRAM_BASE):
            with self.assertRaises(ValueError):
                moved.address(DRAM_BASE+offset, 0)


if __name__ == '__main__':
    unittest.main()
