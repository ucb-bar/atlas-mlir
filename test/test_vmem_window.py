"""VMEM banks 3-5 as a DMA window: admitted spans, relocated frames, and refusals."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from atlas_mlir_native import quantized_loop as loop
from atlas_mlir_native.already_quantized import prepare_source
from atlas_mlir_native.binding import DRAM_BASE, _tile_io_base, _with_tile_io_base
from atlas_mlir_native.memory_windows import dma_span_in_window
from merlin.semantic_compiler.codegen import CodegenOptions, codegen_options
from merlin.semantic_compiler.target_binding import NativeCompilationError
from test_already_quantized import source_request

ROOT = Path(__file__).resolve().parents[1]
VMEM_WINDOW = 0x200C0000


class VmemWindowTest(unittest.TestCase):
    def test_only_banks_3_to_5_are_a_dma_window(self):
        for address, size, admitted in ((0x200C0000, 0xC0000, True), (0x200C0000, 32, True), (0x2017FFE0, 32, True),
                                        (0x200BFFE0, 32, False), (0x200BFFE0, 64, False), (0x20000000, 32, False),
                                        (0x2017FFE0, 64, False), (0x20180000, 32, False), (0x200C0000, 0, False)):
            with self.subTest(address=hex(address), size=size):
                self.assertIs(dma_span_in_window(address, size), admitted)

    def test_native_tile_base_must_keep_all_slots_in_the_window(self):
        @_with_tile_io_base
        def emit(snapshot, request, **kwargs):
            return _tile_io_base.get()
        request = SimpleNamespace(nodes=[SimpleNamespace(type=SimpleNamespace(shape=(32, 32)))])
        args = dict(fixed_inputs={'a': 0, 'w': 1}, fixed_outputs=(2,))
        self.assertEqual(emit(None, request, io_memory_base=VMEM_WINDOW, **args), VMEM_WINDOW)
        for base in (0x20000000, 0x200BF800, 0x2017F000):
            with self.subTest(base=hex(base)), self.assertRaises(NativeCompilationError):
                emit(None, request, io_memory_base=base, **args)
        self.assertEqual(_tile_io_base.get(), DRAM_BASE)

    def test_quantized_frame_moves_with_identical_offsets(self):
        plan = prepare_source(source_request(1, 256, 2048, batch=()))
        with codegen_options(CodegenOptions(io_memory_bytes=67108864)):
            default = loop.execution_layout(plan)
            token = _tile_io_base.set(VMEM_WINDOW)
            try:
                moved = loop.execution_layout(plan)
            finally:
                _tile_io_base.reset(token)
        self.assertEqual(default['input_addresses'][0], DRAM_BASE)
        self.assertEqual(moved['input_addresses'][0], VMEM_WINDOW)
        shift = VMEM_WINDOW - DRAM_BASE
        for key in ('input_addresses', 'exponent_addresses', 'output_addresses'):
            self.assertEqual(moved[key], [address + shift for address in default[key]])
        self.assertEqual(moved['status_address'], default['status_address'] + shift)
        self.assertEqual(moved['status_address'] + moved['status_bytes'] - VMEM_WINDOW, 626688)
        self.assertEqual({k: v for k, v in moved.items() if 'address' not in k},
                         {k: v for k, v in default.items() if 'address' not in k})

    def test_quantized_loop_refuses_bases_before_tools(self):
        source = source_request(1, 256, 2048, batch=())
        with codegen_options(CodegenOptions(io_memory_bytes=67108864)), \
             patch('atlas_mlir_native.quantized_loop_sizing.size_counted', side_effect=AssertionError('sized')), \
             tempfile.TemporaryDirectory(prefix='atlas-vmem-window-') as directory:
            # 0x20170000 starts in the window but its 626,688-byte frame leaves it.
            for base in (True, 1.0, '0x200c0000', -2048, VMEM_WINDOW + 32, 0x20000000, 0x200B0000, 0x20170000, 0x0800b000):
                with self.subTest(base=base), self.assertRaises(NativeCompilationError):
                    loop.compile_source('snapshot', source, target_source=ROOT, destination=Path(directory)/'x',
                                        limits='limits', io_memory_base=base)
                self.assertEqual(_tile_io_base.get(), DRAM_BASE)
            self.assertFalse((Path(directory)/'x').exists())

    def test_custom_base_refuses_tiled_fallback(self):
        source = source_request(1, 256, 2048, batch=())
        tiled = SimpleNamespace(record=lambda: dict(route='k32_tiles', reason='physical_instruction_bound'))
        with codegen_options(CodegenOptions(io_memory_bytes=67108864)), \
             patch('atlas_mlir_native.quantized_loop_sizing.size_counted', return_value=tiled), \
             patch('atlas_mlir_native.already_quantized.compile_source', side_effect=AssertionError('tiles compiled')):
            with self.assertRaisesRegex(NativeCompilationError, 'requires the counted'):
                loop.compile_source('snapshot', source, target_source=ROOT, destination='unused',
                                    limits='limits', io_memory_base=VMEM_WINDOW)
        self.assertEqual(_tile_io_base.get(), DRAM_BASE)

    def test_reduction_refuses_bases_before_tools(self):
        from atlas_mlir_native import reduction
        request = reduction.request_for((1, 2048), (1,))
        abi = reduction.plan_abi(request)
        with codegen_options(CodegenOptions(io_memory_bytes=67108864)), \
             patch('atlas_mlir_native.binding.AtlasNativeBinding.compile', side_effect=AssertionError('compiled')), \
             tempfile.TemporaryDirectory(prefix='atlas-vmem-reduction-') as directory:
            # 0x2017F000 starts in the window, but the complete reduction frame leaves it.
            for base in (True, 1.0, VMEM_WINDOW + 32, 0x20000000, 0x200B0000, 0x2017F000, 0x0800F000):
                with self.subTest(base=base), self.assertRaises(NativeCompilationError):
                    reduction.compile_reduction('snapshot', request, fixed_inputs=abi['fixed_inputs'],
                        fixed_outputs=tuple(abi['fixed_outputs']), target_source=ROOT,
                        destination=Path(directory)/'x', limits='limits', io_memory_base=base)
            self.assertFalse((Path(directory)/'x').exists())
            # A complete frame inside the window passes the address checks and reaches the body compile.
            with self.assertRaisesRegex(AssertionError, 'compiled'):
                reduction.compile_reduction('snapshot', request, fixed_inputs=abi['fixed_inputs'],
                    fixed_outputs=tuple(abi['fixed_outputs']), target_source=ROOT,
                    destination=Path(directory)/'y', limits='limits', io_memory_base=VMEM_WINDOW)


if __name__ == '__main__':
    unittest.main()
