import hashlib
import json

import pytest

from atlas_mlir_native.reference_eval import decisions


def _write(path, value):
    path.write_text(json.dumps(value))
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def selected(tmp_path, monkeypatch):
    app, capture = tmp_path / 'app', tmp_path / 'capture'
    app.mkdir()
    capture.mkdir()
    model = capture / 'model.mlir'
    model.write_text('authenticated source provenance only')
    receipt_sha = _write(capture / 'capture_receipt.json', {'artifacts': {'model.mlir': {
        'bytes': model.stat().st_size, 'sha256': hashlib.sha256(model.read_bytes()).hexdigest()}}})
    rows = [
        {'id': 'constant', 'implementation': 'source_constant_only', 'placement': 'host',
         'outputs': [{'buffer': 'op8r0'}]},
        {'id': 'sum', 'implementation': 'source_reduction', 'placement': 'accelerator',
         'outputs': [{'buffer': 'op9r0_s0'}]},
        {'id': 'divide', 'implementation': 'source_pointwise', 'placement': 'accelerator',
         'outputs': [{'buffer': 'op10r0'}]},
        {'id': 'quantized', 'implementation': 'source_already_quantized', 'placement': 'accelerator',
         'outputs': [{'buffer': 'op11r0'}]},
    ]
    plan_sha = _write(app / 'selected-plan.json', {'schema': 'global_plan_v1', 'selected': rows})
    manifest_sha = _write(app / 'manifest.json', {'stage_artifacts': {'selected-plan.json': plan_sha}})
    monkeypatch.setattr(decisions, '_source_provenance', lambda text, bindings=False: {} if bindings else {
        8: ['g:quantized:root:n1'], 9: ['g:quantized:root:n1'],
        10: ['g:quantized:root:n1'], 11: ['g:quantized:root:n2']})
    return decisions.SelectedDecisions(app, manifest_sha, capture, receipt_sha), app


def test_split_mean_ignores_host_constant(selected):
    policy, _ = selected
    base = {'host_math': {'supplier': 'pinned'}}
    rules = policy.node_rules({'id': 'g:quantized:root:n1', 'op': 'call_function',
                               'target': 'aten.mean.dim'}, base)
    assert rules == {**base, 'execution': 'device', 'authenticated_placement': True,
                     'selected_routes': ['source_pointwise', 'source_reduction']}
    assert base == {'host_math': {'supplier': 'pinned'}}
    policy.verify()


def test_mxu_absence_uses_recorded_shared_route(selected):
    policy, _ = selected
    rules = policy.node_rules({'id': 'g:quantized:root:n2', 'op': 'call_function',
                               'target': 'atlas.linear.default'}, {})
    assert rules['selected_unit'] == 'mxu1'
    assert policy.record['applied']['g:quantized:root:n2']['unit']['explicit_plan_unit_absent']


def test_conflicting_arithmetic_refuses(selected):
    policy, _ = selected
    policy._by_node['g:quantized:root:n1'][1]['placement'] = 'host'
    with pytest.raises(ValueError, match='numerical stage graph'):
        policy.node_rules({'id': 'g:quantized:root:n1'}, {})


def test_missing_arithmetic_refuses_but_layout_passes(selected):
    policy, _ = selected
    with pytest.raises(ValueError, match='no authenticated arithmetic'):
        policy.node_rules({'id': 'missing', 'op': 'call_function', 'target': 'aten.mul.Tensor'}, {})
    assert policy.node_rules({'id': 'missing', 'op': 'call_function', 'target': 'aten.view.default'}, {}) == {}


def test_host_dense_buffer_view_is_layout_not_arithmetic(selected):
    policy, _ = selected
    view = {'id': 'view', 'implementation': 'merlin.dense_buffer_view', 'placement': 'host',
            'inputs': ['op10r0'], 'buffers': ['op12r0']}
    policy._by_node['g:quantized:root:n1'].append(view)
    rules = policy.node_rules({'id': 'g:quantized:root:n1', 'op': 'call_function', 'target': 'aten.mean.dim'}, {})
    assert rules['execution'] == 'device'
    assert rules['selected_routes'] == ['source_pointwise', 'source_reduction']


def test_verification_detects_mutated_plan(selected):
    policy, app = selected
    (app / 'selected-plan.json').write_text('{}')
    with pytest.raises(ValueError, match='changed during evaluation'):
        policy.verify()


def test_buffer_join_keeps_full_origin_ids():
    rows = [{'id': 'x', 'placement': 'host', 'implementation': 'source_pointwise',
             'outputs': [{'buffer': 'op9r2_s0'}]}]
    result = decisions._join_rows(rows, {9: ['g:quantized:root:n1']})
    assert list(result) == ['g:quantized:root:n1']


def test_constant_only_source_arithmetic_uses_host(selected):
    policy, _ = selected
    policy._by_node['g:quantized:root:n3'] = [policy._by_node['g:quantized:root:n1'][0]]
    rules = policy.node_rules({'id': 'g:quantized:root:n3', 'op': 'call_function',
                               'target': 'aten.cos.default'}, {})
    assert rules['execution'] == 'host'


def test_source_parser_uses_top_level_quantized_origins_only():
    text = '''"builtin.module"() ({
      "func.func"() <{sym_name = "forward", function_type = () -> ()}> ({
        "test.source"() {prov.origin_node_ids = ["g:original:root:n1", "g:quantized:root:n2"]} : () -> ()
        "func.return"() : () -> ()
      }) : () -> ()
    }) : () -> ()'''
    assert decisions._source_provenance(text) == {0: ['g:quantized:root:n2']}


def test_host_placed_atlas_op_cannot_use_device_adapter():
    from atlas_mlir_native.reference_eval.evaluator import _call
    with pytest.raises(ValueError, match='no host source adapter'):
        _call('atlas.qlinear.default', [], {},
              {'execution': 'host', 'authenticated_placement': True})


def test_authenticated_exact_route_is_passed_and_recorded(selected):
    policy, _ = selected
    row = {'id': 'exact', 'placement': 'accelerator',
           'implementation': 'source_exact_float_device', 'buffers': ['op12r0']}
    policy._by_node['exact-node'] = [row]
    rules = policy.node_rules({'id': 'exact-node', 'target': 'aten.add.Tensor'}, {})
    assert rules['selected_routes'] == ['source_exact_float_device']
    assert policy.record['applied']['exact-node']['numerical_routes'] == rules['selected_routes']
    policy._by_node['exact-node'].append({**row, 'implementation': 'source_pointwise'})
    with pytest.raises(ValueError, match='conflicting exact-float'):
        policy.node_rules({'id': 'exact-node'}, {})


def test_composite_silu_keeps_both_constituent_routes(selected):
    policy, _ = selected
    policy._by_node['silu'] = [
        {'id': str(i), 'placement': 'accelerator', 'implementation': route, 'buffers': []}
        for i, route in enumerate(['source_exact_float_device', 'source_pointwise_math'])]
    rules = policy.node_rules({'id': 'silu', 'target': 'aten.silu.default'}, {})
    assert rules['selected_routes'] == ['source_exact_float_device', 'source_pointwise_math']
    assert rules['execution'] == 'device'


def test_host_silu_between_exact_casts_runs_on_the_host(selected):
    policy, _ = selected
    routes = [('accelerator', 'source_exact_float_device'), ('host', 'source_host_pointwise'),
              ('accelerator', 'source_exact_float_device')]
    policy._by_node['silu'] = [{'id': str(i), 'placement': placement, 'implementation': route, 'buffers': []}
                               for i, (placement, route) in enumerate(routes)]
    rules = policy.node_rules({'id': 'silu', 'target': 'aten.silu.default'}, {})
    assert rules['execution'] == 'host' and 'numerical_stages' not in rules
    assert rules['selected_routes'] == ['source_exact_float_device', 'source_host_pointwise']
    policy._by_node['gelu'] = policy._by_node['silu']
    with pytest.raises(ValueError, match='missing or conflicting source operand'):
        policy.node_rules({'id': 'gelu', 'target': 'aten.gelu.default'}, {})


def _binding(target='atlas.qlinear.default'):
    return {'operands': ['op0r0', 'op0r1', 'arg1', 'arg2', 'arg0'],
            'results': ['op1r0'], 'target': target, 'bias_order': 'after_output_cast'}


def _stage(name, route, placement, inputs, output):
    return {'id': name, 'implementation': route, 'placement': placement,
            'inputs': inputs, 'buffers': [output]}


def _split_rows():
    return [
        _stage('readout', 'source_already_quantized', 'accelerator',
               ['arg1', 'arg2', 'op0r0', 'op0r1'], 'temporary-a'),
        _stage('bias', 'source_layout', 'host', ['arg0'], 'temporary-b'),
        _stage('add', 'source_float_arithmetic', 'host',
               ['temporary-a', 'temporary-b'], 'op1r0')]


def _install_split(policy, rows):
    policy._bindings['split'] = _binding()
    policy._by_node['split'] = rows
    return policy.node_rules({'id': 'split', 'target': 'atlas.qlinear.default'}, {})


def test_split_decision_records_actual_edges(selected):
    policy, _ = selected
    rules = _install_split(policy, _split_rows())
    assert rules['execution'] == 'device'
    assert rules['selected_unit'] == 'mxu1'
    assert rules['numerical_stages']['stages'][2]['inputs'] == ['temporary-a', 'temporary-b']
    assert policy.record['applied']['split']['numerical_stages'] == rules['numerical_stages']


@pytest.mark.parametrize('mutation, reason', [
    ('missing_stage', 'missing stage or unbound source operand temporary-b'),
    ('duplicate_producer', 'duplicate producer or overwritten source operand'),
    ('undeclared_route', 'undeclared route or placement'),
    ('wrong_operand', 'missing stage or unbound source operand external-not-source'),
    ('wrong_result', 'missing source result'),
    ('wrong_placement', 'undeclared route or placement'),
    ('cycle', 'cycle'),
    ('unordered', 'edge violates declared stage order'),
    ('disconnected', 'incomplete or disconnected stages'),
    ('wrong_bias_binding', 'stage expression differs from source operation or rounding order'),
    ('wrong_source_contract', 'source contraction requires declared post-cast bias'),
])
def test_stage_graph_refuses_invalid_contracts(mutation, reason):
    from atlas_mlir_native.reference_eval import stages
    rows, binding = _split_rows(), _binding()
    if mutation == 'missing_stage': rows.pop(1)
    elif mutation == 'duplicate_producer':
        rows.insert(1, {**rows[0], 'id': 'duplicate'})
    elif mutation == 'undeclared_route': rows[2]['implementation'] = 'source_unqualified'
    elif mutation == 'wrong_operand': rows[0]['inputs'][0] = 'external-not-source'
    elif mutation == 'wrong_result': rows[2]['buffers'] = ['other-result']
    elif mutation == 'wrong_placement': rows[1]['placement'] = 'accelerator'
    elif mutation == 'cycle': rows[1]['inputs'] = ['op1r0']
    elif mutation == 'unordered': rows.reverse()
    elif mutation == 'disconnected':
        rows.insert(0, _stage('orphan', 'source_layout', 'host', ['arg0'], 'unused'))
    elif mutation == 'wrong_bias_binding': rows[1]['inputs'] = ['arg1']
    elif mutation == 'wrong_source_contract': binding['bias_order'] = 'before_output_cast'
    with pytest.raises(ValueError, match='^invalid numerical stage graph: ' + reason + '$'):
        stages.build(rows, binding, 'atlas.qlinear.default')


@pytest.mark.parametrize('route, placement, kind', [
    ('source_float_arithmetic', 'host', 'float_arithmetic'),
    ('source_exact_float_device', 'accelerator', 'exact_float_device'),
])
@pytest.mark.parametrize('metadata', [None, {}, {'kind': 'expected'}])
def test_add_stage_accepts_absent_or_declared_kind(route, placement, kind, metadata):
    from atlas_mlir_native.reference_eval import stages
    binding = {'operands': ['arg0', 'arg1'], 'results': ['op7r0'], 'target': 'aten.add.Tensor'}
    row = {'id': 'sum', 'implementation': route, 'placement': placement,
           'inputs': [{'buffer': 'arg0'}, {'buffer': 'arg1'}],
           'outputs': [{'buffer': 'op7r0'}]}
    if metadata is not None:
        row['metadata'] = {'kind': kind} if metadata else {}
    joined = decisions._join_rows([row], {7: ['sum-node']})['sum-node']
    if metadata:
        assert joined[0]['kind'] == kind
    else:
        assert 'kind' not in joined[0]
    graph = stages.build(joined, binding, 'aten.add.Tensor')
    assert graph['stages'][0]['adapter'] == 'add'


@pytest.mark.parametrize('route, placement', [
    ('source_float_arithmetic', 'host'),
    ('source_exact_float_device', 'accelerator'),
])
@pytest.mark.parametrize('kind', ['add', 'layout', 'already_quantized', '', None])
def test_add_stage_refuses_kind_differing_from_declared_route(route, placement, kind):
    from atlas_mlir_native.reference_eval import stages
    binding = {'operands': ['arg0', 'arg1'], 'results': ['op7r0'], 'target': 'aten.add.Tensor'}
    row = {'id': 'sum', 'implementation': route, 'placement': placement,
           'inputs': [{'buffer': 'arg0'}, {'buffer': 'arg1'}],
           'outputs': [{'buffer': 'op7r0'}], 'metadata': {'kind': kind}}
    joined = decisions._join_rows([row], {7: ['sum-node']})['sum-node']
    assert joined[0]['kind'] == kind
    with pytest.raises(ValueError, match='^invalid numerical stage graph: stage kind differs from declared adapter$'):
        stages.build(joined, binding, 'aten.add.Tensor')


@pytest.mark.parametrize('dtype', ['bf16', 'f32'])
def test_split_qlinear_cast_then_host_bias(selected, monkeypatch, dtype):
    import numpy as np
    from atlas_mlir_native.reference_eval import device
    from atlas_mlir_native.reference_eval.evaluator import Tensor, _call
    policy, _ = selected
    rules = _install_split(policy, _split_rows())
    # Contraction readout has already rounded to BF16 before output cast/add.
    monkeypatch.setattr(device, '_rules', lambda *a, **kw: 'mxu1')
    monkeypatch.setattr(device, '_contraction', lambda *a: np.array([[0x3f80, 0xc000]], np.uint16))
    bias = Tensor(np.array([0x3f00, 0x4040], np.uint16) if dtype == 'bf16' else
                  np.array([0.5, 3], np.float32), dtype)
    args = [Tensor(np.ones((1, 32), np.uint8), 'u8'), Tensor(np.array(0, np.int32), 'i32'),
            Tensor(np.ones((2, 32), np.uint8), 'u8'), Tensor(np.zeros((2, 1), np.int32), 'i32'),
            bias, 'bfloat16' if dtype == 'bf16' else 'float32', 'atlas_e4m3']
    actual = _call('atlas.qlinear.default', args, {}, rules)
    np.testing.assert_array_equal(actual.raw, np.array([[0x3fc0, 0x3f80]], np.uint16) if dtype == 'bf16'
                                  else np.array([[1.5, 1]], np.float32).view(np.uint32))
    args[4] = None
    with pytest.raises(ValueError, match='^invalid numerical stage graph: requires a source bias$'):
        _call('atlas.qlinear.default', args, {}, rules)


def test_add_composition_accepts_renamed_intermediates_and_additional_layouts(monkeypatch):
    import numpy as np
    from atlas_mlir_native.reference_eval import stages
    from atlas_mlir_native.reference_eval.evaluator import Tensor, _call
    binding = {'operands': ['arg0', 'arg1'], 'results': ['op7r0'], 'target': 'aten.add.Tensor'}
    rows = [
        _stage('copy1', 'source_layout', 'host', ['arg0'], 'any-name'),
        _stage('copy2', 'source_layout', 'host', ['any-name'], 'another-name'),
        _stage('sum', 'source_exact_float_device', 'accelerator', ['another-name', 'arg1'], 'sum-result'),
        _stage('copy3', 'source_layout', 'host', ['sum-result'], 'op7r0')]
    graph = stages.build(rows, binding, 'aten.add.Tensor')
    from atlas_mlir_native.reference_eval import placed_device
    # Target admission is qualified separately; exercise the exact route adapter here.
    monkeypatch.setattr(placed_device, '_rtl', lambda rules: None)
    rules = {'execution': 'device', 'authenticated_placement': True,
             'numerical_stages': graph}
    # Exact-device addition uses RNE: raw chopped addition would produce 0x3da6.
    left = Tensor(np.array([0x3d96], np.uint16), 'bf16')
    right = Tensor(np.array([0x3c07], np.uint16), 'bf16')
    actual = _call('aten.add.Tensor', [left, right], {}, rules)
    np.testing.assert_array_equal(actual.raw, np.array([0x3da7], np.uint16))


def test_unsupported_source_composition_refuses():
    from atlas_mlir_native.reference_eval import stages
    with pytest.raises(ValueError, match=r'^invalid numerical stage graph: unsupported source composition another\.source\.op$'):
        stages.build(_split_rows(), _binding('another.source.op'), 'another.source.op')


def test_source_parser_binds_actual_ssa_operands_and_results():
    text = '''"builtin.module"() ({
      "func.func"() <{sym_name = "forward", function_type = (tensor<2xbf16>, tensor<2xbf16>) -> tensor<2xbf16>}> ({
      ^bb0(%0: tensor<2xbf16>, %1: tensor<2xbf16>):
        %2 = "test.add"(%1, %0) {prov.aten = "aten.add.Tensor", prov.origin_node_ids = ["g:quantized:root:n2"]} : (tensor<2xbf16>, tensor<2xbf16>) -> tensor<2xbf16>
        "func.return"(%2) : (tensor<2xbf16>) -> ()
      }) : () -> ()
    }) : () -> ()'''
    bindings = decisions._source_provenance(text, bindings=True)
    assert bindings['g:quantized:root:n2']['operands'] == ['arg1', 'arg0']
    assert bindings['g:quantized:root:n2']['results'] == ['op0r0']
    assert bindings['g:quantized:root:n2']['target'] == 'aten.add.Tensor'


def test_buffer_join_follows_provider_edges_with_renamed_intermediates():
    rows = [{**row, 'inputs': [{'buffer': x} for x in row['inputs']],
             'outputs': [{'buffer': x} for x in row['buffers']]}
            for row in _split_rows()]
    joined = decisions._join_rows(rows, {1: ['split']})
    assert [row['id'] for row in joined['split']] == ['readout', 'bias', 'add']
    assert joined['split'][2]['inputs'] == ['temporary-a', 'temporary-b']


def test_fused_region_owns_operations_no_output_names():
    rows = [{'id': 'fused', 'implementation': 'source_host_softmax', 'placement': 'host',
             'inputs': [{'buffer': 'op3r0'}], 'outputs': [{'buffer': 'op19r0'}]},
            {'id': 'cast', 'implementation': 'source_float_cast', 'placement': 'host',
             'outputs': [{'buffer': 'op3r0'}]}]
    provenance = {3: ['cast-node'], 5: ['add-node'], 19: ['softmax-node']}
    assert 'add-node' not in decisions._join_rows(rows, provenance)
    joined = decisions._join_rows(rows, provenance, {'fused': ['op5r0_s1', 'op19r0'], 'cast': ['op3r0']})
    assert [row['id'] for row in joined['add-node']] == ['fused']
    assert [row['id'] for row in joined['softmax-node']] == ['fused']
    assert [row['id'] for row in joined['cast-node']] == ['cast']


def test_operation_inside_two_fused_regions_refuses():
    rows = [{'id': name, 'implementation': 'source_softmax', 'placement': 'accelerator',
             'outputs': [{'buffer': f'op{ordinal}r0'}]} for name, ordinal in (('first', 7), ('second', 9))]
    provenance = {5: ['add-node'], 7: ['first-node'], 9: ['second-node']}
    joined = decisions._join_rows(rows, provenance, {'first': ['op5r0'], 'second': ['op5r0']})
    assert joined['add-node'] is None
    policy = object.__new__(decisions.SelectedDecisions)
    policy._by_node = joined
    with pytest.raises(ValueError, match='more than one fused region: add-node'):
        policy.node_rules({'id': 'add-node', 'op': 'call_function', 'target': 'aten.add.Tensor'}, {})


def _install_request(app, rows, request_nodes):
    request_sha = _write(app / 'source-request.json', {'nodes': request_nodes})
    plan_sha = _write(app / 'selected-plan.json', {'schema': 'global_plan_v1', 'selected': rows})
    return _write(app / 'manifest.json', {'stage_artifacts': {'selected-plan.json': plan_sha,
                                                               'source-request.json': request_sha}})


@pytest.mark.parametrize('indices', [[2], [-1], [True], 0])
def test_region_nodes_must_index_the_authenticated_request(selected, indices):
    _, app = selected
    capture = app.parent / 'capture'
    receipt_sha = hashlib.sha256((capture / 'capture_receipt.json').read_bytes()).hexdigest()
    nodes = [{'id': 'arg0', 'effect': 'input'}, {'id': 'op8r0', 'effect': 'pure'}, {'id': 'op11r0', 'effect': 'pure'}]
    rows = [{'id': 'fused', 'implementation': 'source_masked_softmax', 'placement': 'accelerator',
             'outputs': [{'buffer': 'op11r0'}], 'nodes': [0, 1]}]
    accepted = decisions.SelectedDecisions(app, _install_request(app, rows, nodes), capture, receipt_sha)
    # n1 (ordinal 8) is named by no output; the request places it inside the fused region.
    assert [r['id'] for r in accepted._by_node['g:quantized:root:n1']] == ['fused']
    assert 'selected.nodes' in accepted.record['selected_fields_read']
    rows[0]['nodes'] = indices
    with pytest.raises(ValueError, match='leave the authenticated source request'):
        decisions.SelectedDecisions(app, _install_request(app, rows, nodes), capture, receipt_sha)


def _install_softmax(app, *, request=True, mask=0xff7f0000, domain='bf16-row-relative-span80-pad32-v1'):
    rows = [{'id': 'region_7', 'implementation': 'source_masked_softmax', 'placement': 'accelerator',
             'outputs': [{'buffer': 'op11r0'}]}]
    plan_sha = _write(app / 'selected-plan.json', {'schema': 'global_plan_v1', 'selected': rows})
    (app / 'segments/region_006').mkdir(parents=True, exist_ok=True)
    request_sha = _write(app / 'segments/region_006/source-request.json', {'nodes': [
        {'id': 'op9r0', 'op': 'input', 'attrs': {'input_domain': domain}},
        {'id': 'op10r0', 'op': 'input', 'attrs': {'input_domain': 'exact-zero-or-one'}},
        {'id': 'op11r0', 'op': 'masked_softmax', 'attrs': {'axis': 1},
         'type': {'numerical_policy': 'bf16-tiled-masked-row-softmax-zero-neginf-v1'}}]})
    export = {'package_name': 'region_006', 'source_interface': {
        'additive_mask': {'zero_keeps': True, 'finite_exclusion_bits': mask}} if mask is not None else {}}
    artifacts = {'selected-plan.json': plan_sha, **({'segments/region_006/source-request.json': request_sha} if request else {})}
    return _write(app / 'manifest.json', {'stage_artifacts': artifacts,
                                          'regions': [{'id': 'region_7', 'description': {'native_export': export}}]})


def test_softmax_declaration_comes_from_the_selected_region(selected):
    _, app = selected
    capture = app.parent / 'capture'
    receipt_sha = hashlib.sha256((capture / 'capture_receipt.json').read_bytes()).hexdigest()
    node = {'id': 'g:quantized:root:n2', 'op': 'call_function', 'target': 'aten.softmax.int'}
    policy = decisions.SelectedDecisions(app, _install_softmax(app), capture, receipt_sha)
    rules = policy.node_rules(node, {'host_order': 'lexicographic-f32'})
    declared = {'region': 'region_7', 'policy': 'bf16-tiled-masked-row-softmax-zero-neginf-v1',
                'score_domain': 'bf16-row-relative-span80-pad32-v1', 'exclusion_bits': 0xff7f0000}
    assert rules['softmax'] == policy.record['applied'][node['id']]['softmax'] == declared
    assert str(app / 'segments/region_006/source-request.json') in policy.record['identities']
    policy.verify()
    # The declared domain and mask encoding are read, not assumed.
    policy = decisions.SelectedDecisions(app, _install_softmax(app, mask=None, domain='other-domain'), capture, receipt_sha)
    assert policy.node_rules(node, {})['softmax'] == {**declared, 'score_domain': 'other-domain', 'exclusion_bits': None}
    policy = decisions.SelectedDecisions(app, _install_softmax(app, request=False), capture, receipt_sha)
    with pytest.raises(ValueError, match='declares no authenticated row-softmax request: region_7'):
        policy.node_rules(node, {})
    policy = decisions.SelectedDecisions(app, _install_softmax(app), capture, receipt_sha)
    policy.node_rules(node, {})
    (app / 'segments/region_006/source-request.json').write_text('{}')
    with pytest.raises(ValueError, match='identity changed'):
        policy.verify()
