"""Differential checks for generic grouped quantizer and column-row staging rules."""
import ctypes
from dataclasses import replace
import json
import os
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from atlas_mlir_native.dynamic_quantize import emit_c, prepare_source
from atlas_mlir_native.reference_eval import device
from merlin.compiler.build.staging import leading_rows, row_slice, stage
from merlin.compiler.build.regions import CompiledCRegion, compile_semantic_application
from merlin.compiler.partition.semantic_regions import outline_semantic_regions
from merlin.compiler.semantic.search.exploration import SearchLimits
import _prefill_staging_fixtures as helpers
from _prefill_staging_fixtures import publish


def gather(blob, item):
    return b''.join(blob[item['offset'] + j * item['stride']:][:item['bytes']]
                    for j in range(item['count']))


def scatter(output, item, data, coverage):
    width = item['bytes']
    for j in range(item['count']):
        start = item['offset'] + j * item['stride']
        output[start:start + width] = data[j * width:(j + 1) * width]
        coverage[start:start + width] += 1


def authored(reduce, rule='ref', dtype='f32', rows=7):
    """A grouped quantizer with the full contract the Atlas host quantizer accepts."""
    source, _, _ = helpers.quantizer(reduce)
    attrs = dict(mode='dynamic', reduce=reduce, rule=rule, rows=rows, semantic_kind='dynamic_activation_quantizer',
                 code_dtype='uint8', weight_format='atlas_e4m3', exponent_dtype='int32',
                 dequantization='e4m3(code)*2^exponent')
    source = replace(source, nodes=tuple(
        replace(n, type=replace(n.type, dtype=dtype)) if n.id == 'x' else replace(n, attrs=tuple(attrs.items()))
        for n in source.nodes))
    if reduce in ('block', 'head_block'):
        # These source modes return an exponent per row; staging admits only
        # chunks that hold whole declared reduction groups.
        shape = source.node('x').type.shape[:-1] + (1,)
        source = replace(source, nodes=tuple(replace(n, type=replace(n.type, shape=shape))
                                             if n.id == 'exponent' else n for n in source.nodes))
    plan = helpers.plan_for(source, ((0, 1),))
    return source, plan, outline_semantic_regions(source, entry='application', plan=plan)


def native_quantizer(request, root, label):
    """Compile and execute the emitted host quantizer, rather than a grouping surrogate."""
    _, files = emit_c(request, label)
    sources = []
    for name, data in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        if path.suffix == '.c':
            sources.append(str(path))
    library = root / (label + '.so')
    subprocess.run([os.environ.get('ATLAS_HOST_CC', 'cc'), '-shared', '-fPIC', '-O2', '-I', str(root), *sources,
                    '-o', str(library)], check=True, capture_output=True)
    function = getattr(ctypes.CDLL(str(library)), label)
    function.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)]
    function.restype = ctypes.c_int

    def run(data):
        x = ctypes.create_string_buffer(data)
        outputs = [ctypes.create_string_buffer(np.prod(request.node(k).type.shape, dtype=int).item() * size)
                   for k, size in [('codes', 1), ('exponent', 4)]]
        inputs = (ctypes.c_void_p * 1)(ctypes.addressof(x))
        pointers = (ctypes.c_void_p * 2)(*(ctypes.addressof(v) for v in outputs))
        assert function(None, inputs, pointers) == 0
        return dict(zip(request.outputs, (v.raw for v in outputs)))
    return run


@pytest.mark.parametrize('reduce,count,rows', [('head', 4, 7), ('token', 8, 7), ('head_block', 4, 7),
                                               ('head_block', 8, 4), ('block', 8, 4)])
@pytest.mark.parametrize('rule', ['ref', 'nc'])
@pytest.mark.parametrize('dtype', ['f32', 'bf16'])
def test_grouped_quantizer_actual_codes_and_exponents(tmp_path, reduce, count, rows, rule, dtype):
    source, plan, outlined = authored(reduce, rule, dtype, rows)
    key, region = outlined.regions[0]
    compiled = []

    def compile_slice(index, key, region, request, label, root):
        compiled.append(request)
        return helpers.package(request, programs=publish(root / 'segments', request, label))

    with helpers.onchip():
        steps, regions, *_ = stage(source, helpers.Provider(), outlined, plan, [helpers.package(region)],
            capacity=helpers.WINDOW, native_root=tmp_path, compile_slice=compile_slice, counts={key: count})
    assert regions[0]['slices'] == count and len(compiled) == 1
    shape = source.node('x').type.shape
    rng = np.random.default_rng(20261119)
    # Different maxima for each token and head, with both signs, zeros, and rounding ties.
    scale = np.exp2((np.arange(shape[1])[:, None] * 3 + np.arange(shape[2])[None, :] % 4)
                    .reshape(1, shape[1], shape[2], 1) - 7)
    x = rng.uniform(-1.9, 1.9, shape).astype(np.float32)
    x *= scale
    x[..., :4] = np.array([0., -0., 1.0625, -1.1875]) * scale
    if dtype == 'bf16':
        x = device.bf16_bits(x)
    codes, exponents = device.quantize_act(x, 'dynamic', reduce, rule, None, rows)
    expected = dict(codes=codes.tobytes(), exponent=exponents.astype('<i4').tobytes())
    assert np.unique(exponents).size > 1 and np.count_nonzero(codes) > codes.size // 2
    whole = native_quantizer(region, tmp_path / 'whole', 'whole')
    chunk = native_quantizer(compiled[0], tmp_path / 'chunk', 'chunk')
    assert whole(x.tobytes()) == expected
    output = {k: bytearray(len(v)) for k, v in expected.items()}
    coverage = {k: np.zeros(len(v), np.uint8) for k, v in expected.items()}
    for step in steps:
        load, = step['loads']
        results = chunk(gather(x.tobytes(), load['source']))
        for read in step['reads']:
            target = read['destination']
            scatter(output[target['value']], target, results[target['value']], coverage[target['value']])
    assert all(np.all(v == 1) for v in coverage.values())
    assert {k: bytes(v) for k, v in output.items()} == expected
    if reduce == 'token' or (reduce, rows) == ('head_block', 4):
        assert compiled[0].node('x').type.shape[2] < shape[2]  # Split inside a head.


@pytest.mark.parametrize('reduce,counts', [('tensor', (2, 4, 8, 16)), ('block', (2, 4, 8, 16)),
                                           ('head_block', (8, 16))])
def test_chunks_that_cut_a_dynamic_group_refuse(reduce, counts):
    region = authored(reduce)[2].regions[0][1]
    prepare_source(region)  # Refuse a valid quantizer contract, not a malformed exponent shape.
    for count in counts:
        assert leading_rows(region, count) is None
        with pytest.raises(ValueError):
            row_slice(region, count)
        # The refused split would change bits: quantize its chunks directly.
        shape = region.node('x').type.shape
        rows = shape[1] * shape[2]
        rng = np.random.default_rng(20261121)
        x = rng.uniform(1, 1.9, (rows, shape[3])) * rng.choice([-1., 1.], (rows, shape[3]))
        x = device.bf16_bits((x * np.exp2(np.arange(rows)[:, None] - 64.)).astype(np.float32))
        whole = device.quantize_act(x.reshape(shape), 'dynamic', reduce, 'ref', None, 7)[0].tobytes()
        chunk = (1, max(shape[1] // count, 1), shape[2] // max(count // shape[1], 1), shape[3])
        parts = np.split(x, count)
        assert b''.join(device.quantize_act(part.reshape(chunk), 'dynamic', reduce, 'ref', None, 7)[0].tobytes()
                        for part in parts) != whole


def tile_source(per_row, m=64, k=64, n=64):
    """The device contraction alone, so its output is the application output."""
    source, plan = helpers.contraction(m, k, n)
    source = replace(source, nodes=source.nodes[:-1], outputs=('y',))
    plan = replace(plan, selected=plan.selected[:1])
    if per_row:
        source = replace(source, nodes=tuple(replace(node, type=replace(node.type, shape=(m, 1)))
                                            if node.id == 'xe' else node for node in source.nodes))
    return source, plan


@pytest.mark.parametrize('unit', ['mxu0', 'mxu1'])
@pytest.mark.parametrize('per_row', [False, True])
def test_tiled_bf16_contraction_exponents_scatter_and_published_replay(tmp_path, per_row, unit):
    root = tmp_path / 'application'
    (root / 'segments').mkdir(parents=True)
    m = k = n = 64
    source, plan = tile_source(per_row, m, k, n)
    outlined, (steps, regions, packages, *_) = helpers.staged(source, plan, helpers.Provider(groups=False), [], root)
    split = regions[0]
    assert split['staging'] == 'tiles'
    width, rows = split['columns'], split['rows']
    # Replay the JSON round trip of the published tile, and require the same schedule.
    published = json.loads(json.dumps(regions))
    choice = {r['id']: (r['columns'], r['row_axis'], r['rows']) for r in published if r['staging'] == 'tiles'}
    compiled = [helpers.package(region, programs=() if i else helpers.program(region, 'region_000', root))
                for i, (_, region) in enumerate(outlined.regions)]
    labels = []

    def compile_slice(index, key, region, request, label, root):
        labels.append(label)
        return helpers.package(request, helpers._width(region, request), helpers.program(request, label, root))

    with helpers.onchip():
        again, again_regions, *_ = stage(source, helpers.Provider(groups=False), outlined, plan, compiled,
            capacity=helpers.WINDOW, native_root=root, compile_slice=compile_slice, tiles=choice)
    assert again == steps and again_regions == regions and len(labels) == 1
    rng = np.random.default_rng(718)
    # Legal signed E4M3 codes, with no subnormal or reserved encodings.
    codes = np.array([0x18, 0x29, 0x38, 0x49, 0x58, 0x98, 0xa9, 0xb8, 0xc9], np.uint8)
    x, w = rng.choice(codes, (m, k)), rng.choice(codes, (n, k))
    xe = ((np.arange(m) % 5 - 8).astype('<i4').reshape(m, 1) if per_row else np.array(-6, '<i4'))
    we = (np.arange(n) % 7 - 9).astype('<i4').reshape(n, 1)
    expected = device._contraction(x, xe, w, we, unit)
    assert expected.dtype == np.uint16 and np.unique(expected).size > 100
    output = bytearray(expected.nbytes)
    coverage = np.zeros(expected.nbytes, np.uint8)
    store = {k: v.tobytes() for k, v in dict(x=x, xe=xe, w=w, we=we).items()}
    for step in steps[:split['slices']]:
        inputs = {load['port']: gather(store[load['port']], load['source']) for load in step['loads']}
        result = device._contraction(np.frombuffer(inputs['x'], np.uint8).reshape(rows, k),
            np.frombuffer(inputs['xe'], '<i4').reshape((rows, 1) if per_row else ()),
            np.frombuffer(inputs['w'], np.uint8).reshape(width, k),
            np.frombuffer(inputs['we'], '<i4').reshape(width, 1), unit)
        target = step['reads'][0]['destination']
        assert target['count'] > 1 and target['stride'] > target['bytes']
        scatter(output, target, result.tobytes(), coverage)
    assert bytes(output) == expected.tobytes()
    assert np.all(coverage == 1)


def test_tile_publication_replays_through_front_door(tmp_path):
    source, plan = tile_source(True)

    class FrontDoor(helpers.Provider):
        def plan(self, request):
            return plan

        def export_region(self, region, alternative, label, segments, target_source):
            inputs = tuple(n.id for n in region.nodes if n.effect == 'input')
            return CompiledCRegion(label, label + '.h', inputs, region.outputs,
                {label + '.c': b'int x;\n', label + '.h': b'#pragma once\n'}, (),
                (f'segments/{label}',) if alternative.placement != 'host' else (), {})

        def compile_region(self, snapshot, region, alternative, label, segments, target_source, limits):
            if alternative.placement != 'host':
                publish(segments, region, label)
            return self.export_region(region, alternative, label, segments, target_source)

    snapshot = tmp_path / 'snapshot'
    snapshot.mkdir()
    (snapshot / 'manifest.json').write_text('{}')
    with helpers.onchip():
        result = compile_semantic_application(source, adapter=FrontDoor(groups=False),
            snapshot=SimpleNamespace(root=snapshot), target_source=tmp_path, destination=tmp_path / 'front',
            limits=SearchLimits(), name='tiles', capacity=helpers.WINDOW)
    assert result['staged_deployment']
    record = json.loads((tmp_path / 'front/deployment.json').read_text())
    assert record['regions'][0]['staging'] == 'tiles'


@pytest.mark.parametrize('shape,kind', [((128, 32, 8), 'rows'), ((2, 64, 128), 'output_columns')])
def test_single_axis_choices_precede_tiles(tmp_path, shape, kind):
    source, plan = tile_source(False, *shape)
    outlined = outline_semantic_regions(source, entry='application', plan=plan)

    def compile_slice(index, key, region, request, label, root):
        return helpers.package(request, programs=publish(root / 'segments', request, label))

    with helpers.onchip():
        _, regions, *_ = stage(source, helpers.Provider(groups=False), outlined, plan,
            [helpers.package(r) for _, r in outlined.regions], capacity=helpers.WINDOW, native_root=tmp_path,
            compile_slice=compile_slice)
    assert regions[0]['staging'] == kind
