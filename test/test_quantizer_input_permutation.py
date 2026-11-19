"""Folded packing preserves quantizer groups, all output bytes and admission."""
from dataclasses import replace

import numpy as np
import pytest

from atlas_mlir_native import quantizer_input_permutation as folded
from merlin.semantic_compiler.model import IndexMap, SemanticNode, pointwise_identity_maps
from test_activation_quantize_device import dynamic_source
from test_already_quantized import quantizer_source


def fixture(shape, axes, mode):
    logical_shape = tuple(shape[axes.index(i)] for i in range(len(shape)))
    if mode == 'static':
        logical = quantizer_source(count=int(np.prod(shape)), fixed=-8)
        logical = replace(logical, nodes=tuple(replace(n,type=replace(n.type,shape=logical_shape))
            if n.id != logical.nodes[-1].id else n for n in logical.nodes))
    else:
        logical = dynamic_source(shape=logical_shape,reduce=mode,rows=2,rule='ref')
    x = logical.nodes[0]
    physical = replace(x,id='physical',type=replace(x.type,shape=shape))
    rank = len(shape)
    view = SemanticNode(x.id,'transpose',(physical.id,),x.type,
        index_maps=(IndexMap(rank,tuple(tuple(int(j==a) for j in range(rank)) for a in axes),(0,)*rank),
                    pointwise_identity_maps(rank,1)[0]))
    source = replace(logical,nodes=(physical,view,*logical.nodes[1:]),input_storages=((physical.id,'external'),))
    return logical,source


def test_source_refuses_escape_nonpermutation_and_changed_type():
    _,source=fixture((3,5,32),(1,0,2),'head')
    group=folded.prepare_source(source)
    assert group.physical.id=='physical' and group.quantizer.nodes[0].id==group.logical.id
    view=group.logical
    bad=[replace(source,outputs=(*source.outputs,view.id),output_storages=('external',)*3)]
    for changed in (replace(view,attrs=(('unknown',True),)),
                    replace(view,type=replace(view.type,dtype='f32')),
                    replace(view,index_maps=(replace(view.index_maps[0],offsets=(1,0,0)),view.index_maps[1]))):
        bad.append(replace(source,nodes=tuple(changed if n.id==view.id else n for n in source.nodes)))
    for request in bad:
        with pytest.raises(ValueError): folded.prepare_source(request)


def test_selector_requires_private_complete_permutation_and_device_consumer():
    from atlas_mlir_native.composition_adapter import AdjacentDeviceAdapter
    from test_pointwise_narrowing import adapter
    _,source=fixture((3,5,32),(1,0,2),'head')
    base=adapter()
    assert len(base.plan(source).selected)==2
    wrapper=AdjacentDeviceAdapter(base,patterns=('quantizer_input_permutation',))
    selected,=wrapper.plan(source).selected
    assert dict(selected.metadata)['kind']=='quantizer_input_permutation'
    escaped=replace(source,outputs=(*source.outputs,source.nodes[1].id),output_storages=('external',)*3)
    assert len(wrapper.plan(escaped).selected)==2
    # A second external reader forces the original permutation to remain.
    view=source.nodes[1]
    other=SemanticNode('copy','copy',(view.id,),view.type,index_maps=pointwise_identity_maps(len(view.type.shape),1))
    shared=replace(source,nodes=(*source.nodes,other),outputs=(*source.outputs,other.id),output_storages=('external',)*3)
    assert all(dict(row.metadata)['kind']!='quantizer_input_permutation' for row in wrapper.plan(shared).selected)
