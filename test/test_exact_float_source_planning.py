"""Logical prefill float regions slice before physical frame admission."""

import pytest

from atlas_mlir_native import exact_float_device
from atlas_mlir_native.binding import DRAM_BASE
from atlas_mlir_native.options import CodegenOptions, codegen_options
from test_exact_float_device import request as float_source


@pytest.mark.parametrize("budget,chunk,launches", [(1000000, 1024, 128), (128000000, 130048, 2)])
@pytest.mark.parametrize("base", [DRAM_BASE, 0x200C0000])
def test_exact_float_logical_prefill_is_sliced_before_physical_window_checks(budget, chunk, launches, base):
    from dataclasses import replace
    from atlas_mlir_native import exact_float_application
    from atlas_mlir_native.memory_windows import dma_span_in_window
    from merlin.compiler.semantic.model import pointwise_identity_maps

    # The prefill residual-add geometry has two BF16 operands and 131072
    # results. Its unsliced frame exceeds the selected 768-KiB window by 32.
    source = float_source("bf16_add", 131072)
    source = replace(source, nodes=tuple(replace(node,
        type=replace(node.type, shape=(1, 32, 64, 64)),
        index_maps=() if node.effect == "input" else pointwise_identity_maps(4, 2))
        for node in source.nodes))
    options = CodegenOptions(io_memory_base=base, io_memory_bytes=786432,
        execution_instruction_limit=budget, io_memory_capacity_base=base,
        io_memory_capacity_bytes=786432)
    with codegen_options(options):
        logical = exact_float_device.prepare_source_plan(source)
        assert logical.workspace_bytes == 786464
        with pytest.raises(ValueError, match="DMA memory windows"):
            exact_float_device.prepare_plan(source)
        schedule = exact_float_application.prepare_plan(source)
        assert (schedule.chunk_elements, len(schedule.launches)) == (chunk, launches)
        assert sum(call["output_bytes"] for call in schedule.launches) == 131072 * 2
        for body in schedule.bodies:
            physical = exact_float_device.prepare_plan(body)
            assert physical.workspace_bytes <= 786432
            assert base <= physical.io_memory_base < base + physical.workspace_bytes <= base + 786432
            assert dma_span_in_window(base, physical.workspace_bytes)


def test_largest_f32_neg_leaf_refuses_default_window_before_tools():
    from unittest.mock import patch

    source = float_source("f32_neg", exact_float_device.MAX_LEAF_ELEMENTS["f32_neg"])
    logical = exact_float_device.prepare_source_plan(source)
    assert logical.count == 43008
    assert logical.workspace_bytes == 344096
    assert logical.descriptor()["io_memory_bytes"] == 262144
    with patch("atlas_mlir_native.binding._run", side_effect=AssertionError("tool invoked")):
        with pytest.raises(ValueError, match="^exact float buffers leave selected DMA memory windows$"):
            exact_float_device.prepare_plan(source)
    bounded_source = float_source("f32_neg", exact_float_device.chunk_elements(logical))
    bounded = exact_float_device.prepare_plan(bounded_source)
    assert bounded.count == 32256
    assert bounded.workspace_bytes <= logical.descriptor()["io_memory_bytes"]
