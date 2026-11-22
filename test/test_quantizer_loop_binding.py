"""Reuse checked quantizer programs while retaining each current source ABI."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from atlas_mlir_native.application import AtlasApplicationAdapter
from atlas_mlir_native import activation_quantize_application, activation_quantize_device
from atlas_mlir_native.options import CodegenOptions, codegen_options
from test_already_quantized import quantizer_source
from test_activation_quantize_device import dynamic_source
from test_loop_body_binding import ROOT, rename

CHOICE=SimpleNamespace(metadata=(("kind","activation_quantize_device"),),placement='accelerator')


@pytest.fixture(params=['small','tail','dynamic'])
def template(request,tmp_path):
    source=(quantizer_source(count=64) if request.param=='small' else
            quantizer_source(count=4160) if request.param=='tail' else
            dynamic_source(shape=(2,64,64)))
    options=CodegenOptions(device_kv_quantizer=True)
    adapter=AtlasApplicationAdapter.__new__(AtlasApplicationAdapter)
    segments=tmp_path/'segments'
    with codegen_options(options):
        package=adapter.compile_region(None,source,CHOICE,'body',segments,ROOT,None)
    return source,options,adapter,package,segments


def test_source_replay_shares_all_native_images_without_compiling(template,monkeypatch):
    source,options,adapter,package,segments=template
    current,names=rename(source)
    def forbidden(*args,**kwargs):
        pytest.fail('binding selected or compiled a quantizer')
    monkeypatch.setattr(adapter,'plan',forbidden)
    monkeypatch.setattr(adapter,'compile_region',forbidden)
    monkeypatch.setattr(activation_quantize_application,'compile_source',forbidden)
    monkeypatch.setattr(activation_quantize_device,'compile_source',forbidden)
    with codegen_options(options):
        bound=adapter.bind_loop_region(package,current,CHOICE,names,segments,ROOT)
        again=adapter.bind_loop_region(package,current,CHOICE,names,segments,ROOT)
    assert bound is not None and again==bound
    assert bound.files==package.files and bound.programs==package.programs
    assert bound.inputs==tuple(names[k] for k in package.inputs)
    assert bound.outputs==tuple(names[k] for k in package.outputs)
    proof=bound.description['source_binding_proof']
    assert proof['schema']=='atlas.activation_quantize_loop_source_binding.v1'
    assert proof['source_request_digest']==current.digest()
    assert proof['template_source_request_digest']==source.digest()
    assert bound.description['native_manifest_sha256']==package.description['native_manifest_sha256']
    assert bound.description['device_launch_schedule']==package.description['device_launch_schedule']
    assert adapter._loop_body_packages[package.symbol]['root']==(segments/'body').resolve()


def test_unissued_changed_policy_wrong_correspondence_and_deploy_decline(template):
    source,options,adapter,package,segments=template
    current,names=rename(source)
    with codegen_options(options):
        assert adapter.bind_loop_region(replace(package,files=dict(package.files)),current,CHOICE,names,segments,ROOT) is None
        assert adapter.bind_loop_region(package,current,CHOICE,{},segments,ROOT) is None
    with codegen_options(replace(options,execution_instruction_limit=1000001)):
        assert adapter.bind_loop_region(package,current,CHOICE,names,segments,ROOT) is None
    with codegen_options(replace(options,application_deployment='onchip')):
        assert adapter.bind_loop_region(package,current,CHOICE,names,segments,ROOT) is None
    # A different static exponent must never reuse an equal-shaped template.
    static=quantizer_source(count=64,fixed=-6)
    with codegen_options(options):
        assert adapter.bind_loop_region(package,static,CHOICE,names,segments,ROOT) is None
