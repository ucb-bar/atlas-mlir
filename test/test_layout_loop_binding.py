"""Share byte-preserving layout bodies only after complete current-source replay."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from atlas_mlir_native.application import AtlasApplicationAdapter
from atlas_mlir_native import layout_device,layout_prefix_application
from atlas_mlir_native.options import CodegenOptions,codegen_options
from test_layout_device import cases
from test_layout_prefix_application import transpose
from test_loop_body_binding import ROOT,rename

CHOICE=SimpleNamespace(metadata=(("kind","layout_device"),),placement='accelerator')


@pytest.fixture(params=['dense_dma','xlu','prefix'])
def template(request,tmp_path):
    source=transpose() if request.param=='prefix' else cases()[request.param]
    options=CodegenOptions(device_kv_quantizer=True,io_memory_bytes=8388608)
    adapter=AtlasApplicationAdapter.__new__(AtlasApplicationAdapter)
    segments=tmp_path/'segments'
    with codegen_options(options):
        package=adapter.compile_region(None,source,CHOICE,'body',segments,ROOT,None)
    return source,options,adapter,package,segments


def test_source_replay_shares_native_layouts_without_compiling(template,monkeypatch):
    source,options,adapter,package,segments=template
    current,names=rename(source)
    def forbidden(*args,**kwargs):pytest.fail('binding selected or compiled a layout')
    monkeypatch.setattr(adapter,'plan',forbidden)
    monkeypatch.setattr(adapter,'compile_region',forbidden)
    monkeypatch.setattr(layout_device,'compile_source',forbidden)
    monkeypatch.setattr(layout_prefix_application,'compile_source',forbidden)
    with codegen_options(options):
        bound=adapter.bind_loop_region(package,current,CHOICE,names,segments,ROOT)
        again=adapter.bind_loop_region(package,current,CHOICE,names,segments,ROOT)
    assert bound is not None and bound==again
    assert bound.files==package.files and bound.programs==package.programs
    assert bound.inputs==tuple(names[key] for key in package.inputs)
    assert bound.outputs==tuple(names[key] for key in package.outputs)
    proof=bound.description['source_binding_proof']
    assert proof['schema']=='atlas.layout_loop_source_binding.v1'
    assert proof['source_request_digest']==current.digest()
    assert proof['template_source_request_digest']==source.digest()
    assert bound.description['native_manifest_sha256']==package.description['native_manifest_sha256']
    assert bound.description['launches_per_invocation']==package.description['launches_per_invocation']
    assert bound.description.get('device_launch_schedule')==package.description.get('device_launch_schedule')
    assert adapter._loop_body_packages[package.symbol]['root']==(segments/'body').resolve()


def test_unissued_changed_policy_wrong_correspondence_and_deploy_decline(template):
    source,options,adapter,package,segments=template
    current,names=rename(source)
    with codegen_options(options):
        assert adapter.bind_loop_region(replace(package,files=dict(package.files)),current,CHOICE,names,segments,ROOT) is None
        assert adapter.bind_loop_region(package,current,CHOICE,{},segments,ROOT) is None
        different=replace(current,nodes=tuple(replace(node,type=replace(node.type,dtype='f32')) for node in current.nodes))
        assert adapter.bind_loop_region(package,different,CHOICE,names,segments,ROOT) is None
    with codegen_options(replace(options,execution_instruction_limit=1000001)):
        assert adapter.bind_loop_region(package,current,CHOICE,names,segments,ROOT) is None
    with codegen_options(replace(options,application_deployment='onchip')):
        assert adapter.bind_loop_region(package,current,CHOICE,names,segments,ROOT) is None
