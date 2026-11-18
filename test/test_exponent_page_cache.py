"""Invariant exponent pages retain guards, batch identity and exact results."""
from collections import Counter
from dataclasses import replace
import json
from math import prod
import os
import subprocess
from types import SimpleNamespace

import pytest

from atlas_mlir_native.options import CodegenOptions, codegen_options, current_codegen_options
from atlas_mlir_native.already_quantized import prepare_source
from atlas_mlir_native.binding import DRAM_BASE
from atlas_mlir_native.exponent_page_cache import enabled
from atlas_mlir_native.quantized_loop import execution_layout, virtual_source
from atlas_mlir_native.restoration_budget import instruction_bound

from merlin.semantic_compiler.target_binding import NativeCompilationError
from test_already_quantized import source_request
from test_exponent_restoration import compile_fixture
from test_quantized_prefix_reuse import memory_fixture


OPTIONS = CodegenOptions(io_memory_bytes=67108864, execution_instruction_limit=128000000)


def fixture(m, n, k, batch, left, right, shared):
    source = source_request(m=m, n=n, k=k, batch=batch)
    shapes = {'a': (*batch, m, k), 'w': (n, k) if shared else (*batch, n, k)}
    for key, mode, rows, dims in (
            ('sx', left, m, batch), ('sw', right, n, () if shared else batch)):
        shapes[key] = () if mode == 'scalar' else (
            (*dims, 1, 1) if mode == 'batch' else (*dims, rows, 1))
    nodes = tuple(replace(node, type=replace(node.type, shape=shapes[node.id]))
                  if node.id in shapes else node for node in source.nodes)
    if not shared:
        output = nodes[-1]
        nodes = (*nodes[:-1], replace(
            output, op='atlas.qk_matmul', type=replace(output.type, dtype='f32'),
            attrs=tuple((key, 'float32' if key == 'output_dtype' else value)
                        for key, value in output.attrs)))
    return replace(source, nodes=nodes)


def test_explicit_policy_identity_recorded_replay_and_scope_tampering(monkeypatch, select_codegen_options):
    from atlas_mlir_native.application import AtlasApplicationAdapter

    source = fixture(35, 65, 65, (2,), 'batch', 'batch', False)
    plan = prepare_source(source)
    choice = SimpleNamespace(to_dict=lambda: dict(
        id='test', nodes=[], inputs=[], outputs=[], resident_inputs=[], resident_outputs=[]))
    adapter = AtlasApplicationAdapter.__new__(AtlasApplicationAdapter)
    with codegen_options(replace(current_codegen_options(), io_memory_bytes=OPTIONS.io_memory_bytes, execution_instruction_limit=OPTIONS.execution_instruction_limit)):
        select_codegen_options(exponent_page_cache=False)
        old = execution_layout(plan)
        plain = virtual_source(plan, old)
        identity = adapter.materialization_identity(source, choice)
        assert not enabled() and 'exponent_page_cache' not in old
        select_codegen_options(exponent_page_cache=True)
        cached = execution_layout(plan)
        selected = virtual_source(plan, cached)
        assert cached['exponent_page_cache_scopes'] == ['batch', 'batch']
        changed = adapter.materialization_identity(source, choice)
        assert changed['compilation_identity'].pop('exponent_page_cache') is True
        assert changed == identity
        assert virtual_source(plan, old) == plain
        select_codegen_options(exponent_page_cache=False)
        assert virtual_source(plan, cached) == selected
        with pytest.raises(ValueError):
            select_codegen_options(exponent_page_cache='invalid')
        for key, value in (
                ('exponent_page_cache', False), ('exponent_page_cache', 1),
                ('exponent_page_cache_scopes', ['invocation', 'batch'])):
            with pytest.raises(ValueError):
                virtual_source(plan, {**cached, key: value})


def test_native_binding_and_export_replay_cache_selection(tmp_path, select_codegen_options):
    from test_quantized_panel_geometry import test_selected_panels_survive_native_rebinding_and_c_export
    from atlas_mlir_native import quantized_loop_c
    from test_exponent_restoration import ROOT

    # Uncached panels are test_quantized_panel_geometry's default-option case.
    select_codegen_options(exponent_page_cache=True)
    test_selected_panels_survive_native_rebinding_and_c_export(tmp_path)
    source = source_request(m=97, n=35, k=257, batch=())
    select_codegen_options(exponent_page_cache=False)
    with codegen_options(replace(current_codegen_options(), io_memory_bytes=OPTIONS.io_memory_bytes, execution_instruction_limit=OPTIONS.execution_instruction_limit)):
        record, files = quantized_loop_c.prepare_c_package(
            source, tmp_path / 'bound', 'replay', target_source=ROOT)
    assert files and record


def test_native_initialization_proof_rejects_missing_prefetch(monkeypatch):
    from atlas_mlir_native.exponent_page_cache import ExponentPageCache
    from atlas_mlir_native.exponent_restoration import STATE

    def missing(self, scope, addresses, *, input_state, output_state, prefix):
        return [f'    cf.br ^{prefix}_ready(%{input_state} : {STATE})',
                f'  ^{prefix}_ready(%{output_state}: {STATE}):']

    monkeypatch.setattr(ExponentPageCache, 'prefetch', missing)
    with codegen_options(replace(current_codegen_options(), io_memory_bytes=OPTIONS.io_memory_bytes, execution_instruction_limit=OPTIONS.execution_instruction_limit)):
        plan = prepare_source(fixture(3, 35, 65, (), 'scalar', 'row', True))
        layout = execution_layout(plan, exponent_page_cache=True)
        with pytest.raises(AssertionError, match='initializ'):
            compile_fixture(virtual_source(plan, layout), pipeline='optimized')
