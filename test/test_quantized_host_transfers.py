"""Generated C transports useful exponent pages into real checked device code."""
import ctypes
from dataclasses import replace
import json
import os

import numpy as np
import pytest

from atlas_mlir_native.already_quantized import prepare_source
from atlas_mlir_native.binding import DRAM_BASE
from atlas_mlir_native.options import codegen_options, current_codegen_options
from test_exponent_page_cache import fixture


@pytest.mark.skipif(not (os.environ.get('ATLAS_RUST_NPU_MODEL') and os.environ.get('ATLAS_OOT_BIN_DIR')),
                    reason='select Rust simulator and Atlas tools')
@pytest.mark.parametrize('pipeline',['mandatory','optimized'])
@pytest.mark.parametrize('shape',[
    (35,37,65,(2,),'row','row',True),
    (3,35,65,(2,),'scalar','batch',False),
    (3,5,65,(2,),'row','row',True),
    (35,3,65,(2,),'row','batch',False),
])
def test_host_prefixes_keep_padding_and_late_refusal_atomic(tmp_path,pipeline,shape):
    from atlas_mlir_native.application import AtlasApplicationAdapter
    from merlin.compiler.build.regions import compile_semantic_application, verify_semantic_application
    from merlin.compiler.semantic.search.exploration import SearchLimits
    from rust_model import Application
    from test_already_quantized import _policy
    from test_looped_mnk import ROOT, build_snapshot

    source = fixture(*shape)
    plan = prepare_source(source)
    adapter = AtlasApplicationAdapter(_policy(),original_source_sha256='0'*64)
    artifact = tmp_path/'application'
    with codegen_options(replace(current_codegen_options(),io_memory_bytes=1048576,
                                 native_pipeline=pipeline)):
        snapshot = build_snapshot(tmp_path)
        manifest = compile_semantic_application(source,adapter=adapter,snapshot=snapshot,
            target_source=ROOT,destination=artifact,limits=SearchLimits(),name='transport',capacity=1048576)
        assert len(manifest['regions']) == 1
        assert verify_semantic_application(source,adapter=adapter,snapshot=snapshot,
            target_source=ROOT,artifact=artifact)['status'] == 'checked'
    layout = json.loads((artifact/'segments/region_000/execution_plan.json').read_text())
    arrays = {}
    for key in plan.input_ids:
        dimensions = source.node(key).type.shape
        arrays[key] = (np.full(dimensions,0x38,np.uint8) if key in plan.operand_ids else
                       (np.arange(int(np.prod(dimensions)),dtype='<i4')%3-1).reshape(dimensions))
    left,right = (arrays[key] for key in plan.exponent_ids)
    right = right.swapaxes(-1,-2) if right.ndim else right
    factors = np.broadcast_to(np.broadcast_to(left,(*plan.batch_shape,plan.m,1)) + right,
                              (*plan.batch_shape,plan.m,plan.n))
    words = int(np.float32(plan.k).view(np.uint32))//65536 + factors*128
    expected = (words.astype('<u2') if plan.output_dtype == 'bf16' else
                (words.astype('<u4') << 16)).tobytes()
    width = {'bf16':2,'f32':4}[source.node(plan.output_id).type.dtype]
    assert len(expected) == int(np.prod(source.node(plan.output_id).type.shape))*width
    arguments = [ctypes.create_string_buffer(arrays[key].tobytes()) for key in plan.input_ids]
    output = ctypes.create_string_buffer(bytes([0x7a])*(len(expected)+64))
    app = Application(artifact/'c-package','transport',tmp_path/'runtime',
                      base=DRAM_BASE-64,size=1048704)
    app.invoke(arguments,[output])
    assert output.raw[:len(expected)] == expected
    assert output.raw[len(expected):len(expected)+64] == bytes([0x7a])*64
    assert len(app.stats) == 1
    assert app.upload_bytes == sum(layout['input_buffer_bytes']) + sum(
        geometry['pages']*128 for geometry in layout['exponent_layouts'])
    assert app.download_bytes == layout['output_bytes']//2048*layout.get('output_tile_store_bytes',2048)+4
    for address,geometry in zip(layout['exponent_addresses'],layout['exponent_layouts'],strict=True):
        for page in range(geometry['pages']):
            offset = address-app.base+page*2048
            assert app.memory[offset+128:offset+2048] == bytes([0xa5])*1920
    stored = layout.get('output_tile_store_bytes',2048)
    for tile in range(layout['output_bytes']//2048):
        offset = layout['output_addresses'][0]-app.base+tile*2048
        assert app.memory[offset+stored:offset+2048] == bytes([0xa5])*(2048-stored)
    assert app.memory[:64] == app.memory[-64:] == bytes([0xa5])*64
    for argument,key in zip(arguments,plan.input_ids,strict=True):
        assert argument.raw[:-1] == arrays[key].tobytes()

    # A bad last exponent reaches the real device and prevents publication.
    last = plan.input_ids.index(plan.exponent_ids[1])
    ctypes.memmove(ctypes.addressof(arguments[last])+arrays[plan.exponent_ids[1]].nbytes-4,
                   (1000).to_bytes(4,'little'),4)
    ctypes.memset(output,0x7a,ctypes.sizeof(output))
    before = app.download_bytes
    pointers = (ctypes.c_void_p*len(arguments))(*(ctypes.addressof(value) for value in arguments))
    result = app.library.invoke(pointers,(ctypes.c_void_p*1)(ctypes.addressof(output)))
    assert result != 0 and not app.errors
    assert len(app.stats) == 2 and app.download_bytes-before == 4
    assert output.raw == bytes([0x7a])*ctypes.sizeof(output)
