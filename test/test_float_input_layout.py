"""Packed affine inputs retain exact arithmetic, guards and atomic publication."""
from dataclasses import replace

import numpy as np
import pytest

from atlas_mlir_native import float_input_layout as folded
from merlin.semantic_compiler.model import IndexMap, SemanticNode, pointwise_identity_maps
from test_exact_float_device import request


def fixture(op, layout, port=0, shape=(33,35)):
    rows,columns=shape
    original=request(op,int(np.prod(shape)))
    logical=replace(original,nodes=tuple(replace(n,type=replace(n.type,shape=shape),
        index_maps=() if n.effect=='input' else pointwise_identity_maps(2,len(n.inputs))) for n in original.nodes))
    operand=logical.nodes[port]
    if layout=='broadcast':
        physical_shape,coefficients,offsets=(columns,),((0,1),),(0,)
    elif layout=='slice':
        physical_shape,coefficients,offsets=(rows,2*columns),((1,0),(0,1)),(0,columns)
    elif layout=='reverse':
        physical_shape,coefficients,offsets=(rows,2*columns),((1,0),(0,-2)),(0,2*columns-1)
    elif layout=='transpose':
        physical_shape,coefficients,offsets=(columns,rows),((0,1),(1,0)),(0,0)
    else: raise ValueError(layout)
    physical=replace(operand,id='physical',type=replace(operand.type,shape=physical_shape))
    view=SemanticNode(operand.id,'gather',(physical.id,),operand.type,
        index_maps=(IndexMap(2,coefficients,offsets),pointwise_identity_maps(2,1)[0]))
    nodes=tuple(item for n in logical.nodes for item in ((physical,view) if n.id==operand.id else (n,)))
    source=replace(logical,nodes=nodes,input_storages=tuple((n.id,'external') for n in nodes if n.effect=='input'))
    return logical,source


def test_private_selection_and_refused_source_boundaries():
    from atlas_mlir_native.composition_adapter import AdjacentDeviceAdapter
    from test_pointwise_narrowing import adapter
    _,source=fixture('bf16_add','broadcast',1)
    base=adapter(host_support=['transpose','reshape','copy','gather'],
                 host_float_arithmetic=['bf16_add','bf16_sub','bf16_neg','f32_neg'])
    wrapper=AdjacentDeviceAdapter(base,patterns=('float_input_layout',))
    selected,=wrapper.plan(source).selected
    assert dict(selected.metadata)['kind']=='float_input_layout'
    view=next(n for n in source.nodes if n.op=='gather')
    escaped=replace(source,outputs=(*source.outputs,view.id),output_storages=('external',)*2)
    assert len(wrapper.plan(escaped).selected)==2
    other=SemanticNode('copy','copy',(view.id,),view.type,index_maps=pointwise_identity_maps(2,1))
    shared=replace(source,nodes=(*source.nodes,other),outputs=(*source.outputs,other.id),output_storages=('external',)*2)
    assert all(dict(row.metadata)['kind']!='float_input_layout' for row in wrapper.plan(shared).selected)
    changes=[escaped]
    for changed in (replace(view,attrs=(('unknown',True),)),
        replace(view,type=replace(view.type,dtype='f32')),
        replace(view,index_maps=(replace(view.index_maps[0],offsets=(1,)),view.index_maps[1]))):
        changes.append(replace(source,nodes=tuple(changed if n.id==view.id else n for n in source.nodes)))
    for changed in changes:
        with pytest.raises(ValueError): folded.prepare_source(changed)
    # A source port used both through the view and directly must not silently
    # change alias/dedup semantics when the view port is replaced.
    _,same=fixture('bf16_add','transpose')
    physical=replace(same.nodes[0],type=replace(same.nodes[0].type,shape=(33,35)))
    identity=replace(same.nodes[1],index_maps=pointwise_identity_maps(2,1))
    out=replace(same.nodes[-1],inputs=(identity.id,physical.id))
    same=replace(same,nodes=(physical,identity,out),input_storages=((physical.id,'external'),))
    with pytest.raises(ValueError): folded.prepare_source(same)
