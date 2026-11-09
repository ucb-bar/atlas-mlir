"""Explicit standalone source exp through the selected bounded BF16 LUT."""
from dataclasses import dataclass
import numpy as np
from merlin.semantic_compiler.model import KernelRequest, SemanticNode, TensorType, pointwise_identity_maps
from merlin.semantic_compiler.precision_values import FloatBoundarySpec
from .binding import BF16_EXP_ATTRS, TARGET_IDENTITY

POLICY = "f32-to-bf16-bounded32-exp-lut-v1"


def input_spec():
    return FloatBoundarySpec(POLICY,"bf16","encode",admitted_code_ranges=(
        (0,0),(0x0080,0x4200),(0x8000,0x8000),(0x8080,0xc200)))


@dataclass(frozen=True)
class ExpPrecision:
    source: KernelRequest
    request: KernelRequest

    @property
    def operand_specs(self):
        return (input_spec(),)

    def record(self):
        return dict(schema="atlas.source_exp_precision.v1",policy=POLICY,
            source_request_digest=self.source.digest(),request_digest=self.request.digest(),
            inputs=list(self.source.nodes[-1].inputs),input_conversions=[input_spec().record()],
            input_domain="BF16 finite normal or signed zero, magnitude at most 32 after RNE",
            arithmetic="selected-rtl-bf16-exp-q9p12-lut-rne-v1",output_conversion="bf16-to-f32-exact",
            qualification="diagnostic_precision_change; source exp remains a separate operation")


def prepare_source(source,policy):
    if policy!=POLICY:
        raise ValueError("standalone source exp needs its explicitly selected precision policy")
    pure=[node for node in source.nodes if node.effect=='pure']
    if len(pure)!=1:
        raise ValueError("standalone source exp requires one closed computation")
    output=pure[0]
    if (source.target_identity!=TARGET_IDENTITY or source.lowering_policy!='diagnostic' or
        source.physical_extents or source.boundary_layouts or source.output_layouts or
        output.op!='source_f32_exp_unqualified' or len(output.inputs)!=1 or source.outputs!=(output.id,) or
        dict(output.attrs)!={'numeric_semantics':'source-f32-exp-unqualified-v1'} or
        output.index_maps!=pointwise_identity_maps(len(output.type.shape),1) or
        {node.id for node in source.nodes}!={*output.inputs,output.id} or
        any(node.type.dtype!='f32' or node.type.numerical_policy!='raw-bit-pattern' for node in source.nodes)):
        raise ValueError("standalone source exp operation, metadata, maps, or source accounting differ")
    operand=source.node(output.inputs[0])
    if operand.effect not in ('input','constant') or operand.inputs or operand.index_maps or operand.type!=output.type:
        raise ValueError("standalone source exp requires one matching dense boundary operand")
    arg=SemanticNode('arg0','input',(),TensorType(operand.type.shape,'bf16','raw-bit-pattern'),(BF16_EXP_ATTRS[0],),'input')
    result=SemanticNode('result','exp_bf16',('arg0',),arg.type,BF16_EXP_ATTRS,index_maps=output.index_maps)
    request=KernelRequest((arg,result),('result',),('external',),(('arg0','external'),),TARGET_IDENTITY,'diagnostic',source.digest())
    if operand.effect=='constant':
        binding=next(item for item in source.constants if item.node_id==operand.id)
        if binding.encoding!='f32-le-raw':
            raise ValueError("standalone source exp parameter requires exact f32 bytes")
        input_spec().encode(np.frombuffer(bytes.fromhex(binding.data_hex),dtype='<f4'))
    return ExpPrecision(source,request)
