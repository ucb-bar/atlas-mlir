"""Row-relative admission changes the proof domain, never the device arithmetic."""
from dataclasses import replace
import struct
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from atlas_mlir_native.row_softmax import (
    ROWSPAN80_DOMAIN, SOFTMAX_DOMAIN, prepare_inputs, prepare_row_softmax, validate_row_span80,
)
from atlas_mlir_native.softmax_precision import input_spec, lower_semantic_softmax
from test_tiled_softmax import source_request


def payload(values):
    return b''.join(struct.pack('<H', struct.unpack('<I', struct.pack('<f', x))[0] >> 16) for x in values)


def plan(shape, *, masked=False, relative=True):
    request = source_request(shape, masked)
    if relative:
        request = replace(request, nodes=(replace(request.nodes[0], attrs=(("input_domain", ROWSPAN80_DOMAIN),)),
                                          *request.nodes[1:]))
    return prepare_row_softmax(request)


def test_row_domain_keeps_exact_primitive_graph_and_padding():
    old, new = plan((2, 65), relative=False), plan((2, 65))
    assert old.pad_element == new.pad_element == b'\x00\xc2'
    assert old.policy == new.policy
    for a, b in zip(old.regions, new.regions):
        assert a.request.nodes == b.request.nodes
        assert a.request.constants == b.request.constants
    assert old.source.digest() != new.source.digest()
    assert input_spec().admitted_code_ranges == ((0, 0), (0x0400, 0x4200), (0x8000, 0x8000), (0x8400, 0xc200))
    assert input_spec(ROWSPAN80_DOMAIN).admitted_code_ranges[-1] == (0x8400, 0xc300)
    with pytest.raises(ValueError, match='domain'):
        input_spec('unknown')


@pytest.mark.parametrize('values', [[-44.5, 4.53125], [-112, -32], [-32, 48], [-80, 0], [-0.0, 0.0]])
def test_admits_exact_relative_edges(values):
    prepare_inputs(plan((1, len(values))), payload(values))


@pytest.mark.parametrize('values', [[-112.5, -32], [-32.25, 48], [-80, 2.0**-119],
                                   [-33, -32.25], [48.25, 48], [129, 0], [2.0**-120, 0]])
def test_refuses_span_maximum_and_code_violations(values):
    with pytest.raises(ValueError, match='domain'):
        prepare_inputs(plan((1, len(values))), payload(values))


def test_old_domain_still_refuses_wider_scores_and_masked_values_are_proved():
    with pytest.raises(ValueError, match='domain'):
        prepare_inputs(plan((1, 2), relative=False), payload([-44.5, 4.53125]))
    selected = plan((1, 3), masked=True)
    prepare_inputs(selected, payload([-44.5, 4.53125, -32]), payload([1, 1, 0]))
    with pytest.raises(ValueError, match='padding'):
        prepare_inputs(selected, payload([-44.5, 4.53125, 0]), payload([1, 1, 0]))
    with pytest.raises(ValueError, match='nonempty'):
        prepare_inputs(selected, payload([-32]*3), payload([0]*3))


def test_semantic_lowering_fingerprints_selected_domain():
    from atlas_mlir_native.binding import TARGET_IDENTITY
    from merlin.tests.ir.test_softmax_source import source_graph
    graph = replace(source_graph((1, 3)), target_identity=TARGET_IDENTITY, lowering_policy='diagnostic')
    original = lower_semantic_softmax(graph)
    relative = lower_semantic_softmax(graph, score_domain=ROWSPAN80_DOMAIN)
    assert dict(original.request.nodes[0].attrs)['input_domain'] == SOFTMAX_DOMAIN
    assert dict(relative.request.nodes[0].attrs)['input_domain'] == ROWSPAN80_DOMAIN
    assert original.request.digest() != relative.request.digest()
    assert original.record() != relative.record()


def test_row_relative_column_limit_preserves_original_domain_capacity():
    from atlas_mlir_native.softmax_c_source import render_source_interface
    from merlin.semantic_compiler.target_binding import NativeCompilationError

    selected = plan((1, 1024))
    prepare_inputs(selected, payload([0]*1024))
    with pytest.raises(NativeCompilationError, match='policy'):
        plan((1, 1025))
    with pytest.raises(ValueError, match='1024'):
        validate_row_span80(payload([0]*1025), (1, 1025))
    # Gate the legacy upper limit without constructing thousands of column nodes.
    with patch('atlas_mlir_native.row_softmax.decompose_row_softmax', return_value='accepted') as decompose:
        assert plan((1, 65536), relative=False) == 'accepted'
        decompose.assert_called_once()
    unsafe = source_request((1, 1025))
    unsafe = replace(unsafe, nodes=(
        replace(unsafe.nodes[0], attrs=(("input_domain", ROWSPAN80_DOMAIN),)), unsafe.nodes[1],
    ))
    with pytest.raises(ValueError, match='1024'):
        render_source_interface(SimpleNamespace(request=unsafe), 'unsafe', 1, score_domain=ROWSPAN80_DOMAIN)


def test_semantic_row_relative_column_limit_is_checked_before_export():
    from atlas_mlir_native.binding import TARGET_IDENTITY
    from merlin.semantic_compiler.linalg_bridge import LinalgBridgeError
    from merlin.tests.ir.test_softmax_source import source_graph

    graph = replace(source_graph((1, 1024)), target_identity=TARGET_IDENTITY, lowering_policy='diagnostic')
    precision = lower_semantic_softmax(graph, score_domain=ROWSPAN80_DOMAIN)
    assert precision.request.nodes[0].type.shape == (1, 1024)
    graph = replace(source_graph((1, 1025)), target_identity=TARGET_IDENTITY, lowering_policy='diagnostic')
    with pytest.raises(LinalgBridgeError, match='1024'):
        lower_semantic_softmax(graph, score_domain=ROWSPAN80_DOMAIN)
    legacy = replace(source_graph((1, 65536)), target_identity=TARGET_IDENTITY, lowering_policy='diagnostic')
    assert lower_semantic_softmax(legacy).request.nodes[0].type.shape == (1, 65536)
