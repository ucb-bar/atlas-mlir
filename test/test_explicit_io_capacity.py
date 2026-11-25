"""Explicit leaf bases respect declared geometry without losing legacy DMA APIs."""

import pytest

from atlas_mlir_native import (
    activation_quantize_device,
    exact_float_device,
    layout_device,
    pointwise,
    reduction,
)
from atlas_mlir_native.binding import DRAM_BASE, io_memory_base
from atlas_mlir_native.options import CodegenOptions, codegen_options
from merlin.compiler.semantic.target_binding import NativeCompilationError
from test_activation_quantize_device import dynamic_source
from test_bf16_pointwise import request as pointwise_source
from test_exact_float_device import request as float_source
from test_layout_device import request as layout_source


class ReachedPublication(Exception):
    pass


class NoPublication:
    def mkdir(self, **kwargs):
        raise ReachedPublication("address validation completed before publication or tools")


def prepare_route(kind, base):
    if kind == "layout":
        return layout_device.prepare_source(layout_source(), io_memory_base=base)
    if kind == "float":
        return exact_float_device.prepare_plan(float_source("bf16_neg", 32), io_memory_base=base)
    if kind == "quantize":
        return activation_quantize_device.prepare_plan(
            dynamic_source((1, 32), reduce="tensor"), io_memory_base=base)
    if kind == "pointwise":
        module = pointwise
        request = pointwise.prepare_source(pointwise_source((32, 32)),
                                          bf16_policy=pointwise.BF16_POINTWISE_POLICY).request
        compile_route = module.compile_pointwise
    else:
        module = reduction
        request = reduction.request_for((32, 32), (1,))
        compile_route = module.compile_reduction
    abi = module.plan_abi(request)
    return compile_route(None, request, fixed_inputs=abi["fixed_inputs"],
                         fixed_outputs=tuple(abi["fixed_outputs"]), target_source=None,
                         destination=NoPublication(), limits=None, io_memory_base=base)


@pytest.mark.parametrize("kind", ["layout", "float", "quantize", "pointwise", "reduction"])
@pytest.mark.parametrize("base", [DRAM_BASE, 0xA0000800])
def test_explicit_leaf_base_cannot_escape_complete_declared_window(kind, base):
    with codegen_options(CodegenOptions(io_memory_base=0xA0000000, io_memory_bytes=262144,
            io_memory_capacity_base=0xA0000000, io_memory_capacity_bytes=262144)):
        # The second base starts inside capacity but shifts the full window out.
        with pytest.raises(NativeCompilationError, match="capacity"):
            prepare_route(kind, base)
        assert io_memory_base() == 0xA0000000


@pytest.mark.parametrize("kind", ["layout", "float", "quantize", "pointwise", "reduction"])
def test_explicit_leaf_base_respects_requested_alignment(kind):
    with codegen_options(CodegenOptions(io_memory_alignment=4096)):
        with pytest.raises(NativeCompilationError, match="alignment"):
            prepare_route(kind, DRAM_BASE + 2048)


@pytest.mark.parametrize("kind", ["layout", "float", "quantize", "pointwise", "reduction"])
def test_legacy_leaf_alignment_and_small_dma_apertures_remain_supported(kind):
    base = 0x08000000 if kind == "reduction" else 0x08000020
    with codegen_options(CodegenOptions(io_memory_bytes=64 << 20)):
        if kind in {"pointwise", "reduction"}:
            with pytest.raises(ReachedPublication):
                prepare_route(kind, base)
        else:
            assert prepare_route(kind, base).io_memory_base == base
        assert io_memory_base() == DRAM_BASE


def test_row_plan_keeps_legacy_32_byte_explicit_alignment():
    from atlas_mlir_native.row_loop import prepare_plan
    from test_tiled_softmax import source_request

    with codegen_options(CodegenOptions(io_memory_bytes=64 << 20)):
        plan = prepare_plan(source_request((33, 32)), io_memory_base=0x80000020)
        assert plan.io_memory_base == 0x80000020


def test_boundary_capacity_selection_without_base_option_checks_selected_bounds():
    from atlas_mlir_native.device_boundary import virtual_transition

    with codegen_options(CodegenOptions(io_memory_capacity_base=DRAM_BASE,
                                      io_memory_capacity_bytes=262144)):
        with pytest.raises(ValueError, match="selected I/O window"):
            virtual_transition(source_address=DRAM_BASE + 262144,
                               destination_address=DRAM_BASE + 264192,
                               status_address=DRAM_BASE + 266240, rows=32, columns=32)
        with pytest.raises(NativeCompilationError, match="capacity"):
            virtual_transition(source_address=0x80000000, destination_address=0x80000800,
                               status_address=0x80001000, rows=32, columns=32,
                               io_memory_base=0x80000000)
