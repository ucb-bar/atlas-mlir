"""Pointwise affine-read folding preserves code validation and selected results."""
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np
import pytest

from atlas_mlir_native import pointwise_input_layout as fused
from atlas_mlir_native.composition_adapter import AdjacentDeviceAdapter
from merlin.semantic_compiler.model import SemanticNode, IndexMap, pointwise_identity_maps
from test_pointwise_quantize import case as quantized_case
from test_bf16_pointwise import request as pointwise_case

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = ('pointwise_quantize', 'pointwise_input_layout')


def adapter():
    from atlas_mlir_native.application import AtlasApplicationAdapter
    from atlas_mlir_native.already_quantized import POLICY
    from atlas_mlir_native.pointwise import BF16_POINTWISE_POLICY
    from test_application_float_arithmetic import policy
    selected = {**policy(), 'already_quantized': POLICY, 'bf16_pointwise': BF16_POINTWISE_POLICY,
                'host_support': ['transpose','reshape','copy','gather']}
    return AtlasApplicationAdapter(selected, original_source_sha256=selected['source_sha256'])


def case(shape=(33, 35), kind='broadcast', packed=False):
    source = quantized_case(shape) if packed else pointwise_case(shape)
    a, b, *operations = source.nodes
    m,n = int(np.prod(shape[:-1])),shape[-1]
    rank = len(shape)
    physical, mapping = {
        'broadcast': ((n,), IndexMap(rank, (tuple([0]*(rank-1)+[1]),), (0,))),
        'transpose': ((n,m), IndexMap(2, ((0,1),(1,0)), (0,0))),
        'reverse': (shape, IndexMap(2, ((1,0),(0,-1)), (0,n-1))),
        'slice': ((m,n+1), IndexMap(2, ((1,0),(0,1)), (0,0))),
    }[kind]
    base = replace(b, id='base', type=replace(b.type, shape=physical))
    view = SemanticNode(b.id, 'gather', ('base',), b.type,
                        index_maps=(mapping, pointwise_identity_maps(rank,1)[-1]))
    return replace(source, nodes=(a,base,view,*operations), input_storages=(('a','external'),('base','external')))


@pytest.mark.parametrize('kind', ['broadcast','transpose','reverse'])
@pytest.mark.parametrize('packed', [False, True])
def test_real_selector_preserves_affine_semantics(kind, packed):
    source = case(kind=kind, packed=packed)
    group = fused.prepare(source, adapter(), 'pointwise_quantize' if packed else 'pointwise')
    selected, = AdjacentDeviceAdapter(adapter(), patterns=PATTERNS).plan(source).selected
    assert dict(selected.metadata)['kind'] == 'pointwise_input_layout'
    assert group.consumer.outputs == source.outputs
    assert group.consumer.node('result').inputs == ('a','base')
    assert len(group.consumer.nodes) == len(source.nodes)-1


def test_slices_and_shared_views_retain_materialization():
    source = case()
    public = replace(source, outputs=(*source.outputs,'b'), output_storages=('external',)*2)
    for value in (case(kind='slice'), public):
        with pytest.raises(ValueError):
            fused.prepare(value, adapter(), 'pointwise')
        assert all(dict(r.metadata)['kind'] != 'pointwise_input_layout'
                   for r in AdjacentDeviceAdapter(adapter(), patterns=PATTERNS).plan(value).selected)


@pytest.mark.skipif(not all(os.environ.get(k) for k in ('ATLAS_RUST_NPU_MODEL', 'ATLAS_OOT_BIN_DIR')),
                    reason='select Rust model and Atlas tools')
@pytest.mark.parametrize('shape,kind,packed', [
    ((1,7), 'broadcast', False), ((33,35), 'broadcast', True),
    ((32,64), 'transpose', True), ((33,35), 'reverse', False),
    ((2,3,35), 'broadcast', True)])
def test_rust_codes_tails_and_input_guards(tmp_path, shape, kind, packed):
    import torch
    from npu_model.configs.numerics import RtlNumerics
    from atlas_mlir_native.binding import DRAM_BASE
    from atlas_mlir_native.pointwise import prepare_plan
    from atlas_mlir_native.static_quantize import pack_bf16
    from merlin.semantic_compiler.pointwise_tiling import materialize_pointwise_input
    from merlin.semantic_compiler.search import SearchLimits
    from test_looped_mnk import build_snapshot
    from test_pointwise_loop import check_source_c_call

    source = case(shape, kind, packed)
    group = fused.prepare(source, adapter(), 'pointwise_quantize' if packed else 'pointwise')
    compiled = tmp_path/'compiled'
    fused.compile_group(group, build_snapshot(tmp_path/'snapshot'), target_source=ROOT,
                        destination=compiled, limits=SearchLimits())
    exported = fused.export_group(group, compiled, 'pointwise', target_source=ROOT)
    assert exported.inputs == ('a','base')
    package = tmp_path/'package'
    for name,data in exported.files.items():
        path = package/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    with (package/'pointwise.c').open('a') as stream:
        stream.write('\n#include "runtime/atlas_host.c"\n')
    choices = np.array([0,0x8000,0x3f80,0x3f81,0xc049,0x3f00], '<u2')
    a = np.resize(choices, shape)
    base = np.resize(choices[::-1], source.node('base').type.shape)
    view = {'broadcast': lambda: np.broadcast_to(base, shape), 'transpose': lambda: base.T,
            'reverse': lambda: base[:,::-1]}[kind]()
    expected = RtlNumerics.mul(torch.from_numpy(a.copy()).view(torch.bfloat16),
        torch.from_numpy(view.copy()).view(torch.bfloat16)).contiguous().view(torch.uint16).numpy()
    wanted = pack_bf16(expected, -7).tobytes()+(-7).to_bytes(4,'little',signed=True) if packed else expected.tobytes()
    precision = group.group.precision if packed else group.group
    plan = prepare_plan(precision.request)
    program = compiled/'consumer/pointwise' if packed else compiled/'consumer'
    execution = json.loads((program/'execution_plan.json').read_text())
    output = execution['output_addresses'][0]-DRAM_BASE
    end = output+execution['output_bytes']
    initial = bytearray([0xa5])*((end+4095)//4096*4096+4096)
    for index,values in enumerate((a,base)):
        payload = materialize_pointwise_input(plan, index, values.tobytes())
        payload = np.frombuffer(payload,'<u2').reshape(-1,32,2,16).transpose(0,2,1,3).copy().tobytes()
        at = execution['input_addresses'][index]-DRAM_BASE
        initial[at:at+len(payload)] = payload
    config = tmp_path/'config.json'
    config.write_text(json.dumps(dict(dram_base=DRAM_BASE,dram_size=len(initial),numerics='rtl',pack_layout='register_rows')))

    def execute(memory):
        with tempfile.TemporaryDirectory(dir=tmp_path) as directory:
            directory = Path(directory)
            (directory/'initial.bin').write_bytes(memory)
            subprocess.run([os.environ['ATLAS_RUST_NPU_MODEL'], str(program/'program.bin'),
                '--config', str(config), '--compute', 'cpu', '--load', f'{DRAM_BASE}:{directory/"initial.bin"}',
                '--dump-state', str(directory/'state'), '--max-cycles', '10000000'], check=True,capture_output=True)
            actual = b''.join((directory/'state'/f'dram_{addr:016x}.bin').read_bytes()
                             for addr in range(DRAM_BASE,DRAM_BASE+len(memory),4096))
            assert actual[:output] == memory[:output]
            assert actual[end:] == memory[end:]
            return actual

    invalid = base.copy()
    invalid.flat[-1] = 0x7fc0
    check_source_c_call(unittest.TestCase(),package,initial,(program/'program.bin').read_bytes(),execute,
        (a,base),wanted,((a,invalid),),source_dtype='opaque',output_dtype='i8' if packed else 'bf16',
        output_exponent=-7 if packed else None)


def test_selected_rounding_policy_is_never_replaced_to_accept_a_layout():
    from atlas_mlir_native.qualified_rne_add import RNE_ADD_POLICY
    source = case()
    nodes = tuple(replace(node, type=replace(node.type, dtype='f32')) for node in source.nodes)
    addition = replace(nodes[-1], op='f32_add', attrs=(('numeric_semantics','ieee754-f32-add-rne-v1'),))
    source = replace(source, nodes=(*nodes[:-1],addition))
    base = adapter()
    base.qualified_rne_add = RNE_ADD_POLICY
    # This qualified producer admits only identity maps. The unqualified
    # chopped-add implementation can accept broadcasts, but is not a fallback.
    with pytest.raises(ValueError, match='RNE add'):
        fused.prepare(source, base, 'pointwise')


def test_source_indexing_is_replayed_before_compilation_or_export(tmp_path):
    group = fused.prepare(case(), adapter(), 'pointwise')
    changed = replace(group.consumer, outputs=('base',))
    forged = replace(group, consumer=changed)
    with pytest.raises(ValueError, match='composed source indexing'):
        fused.compile_group(forged, None, target_source=ROOT, destination=tmp_path/'compiled', limits=None)
    assert not (tmp_path/'compiled').exists()
    with pytest.raises(ValueError, match='composed source indexing'):
        fused.export_group(forged, tmp_path/'absent', 'pointwise', target_source=ROOT)
