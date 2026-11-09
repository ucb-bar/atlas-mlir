"""Explicit BF16 square, grouped-tree RMSNorm precision for captured source."""
from dataclasses import dataclass, replace

from merlin.semantic_compiler.model import KernelRequest, SemanticNode, TensorType
from merlin.semantic_compiler.qualified_rmsnorm_source import recognize_rmsnorm
from merlin.semantic_compiler.tiled_rmsnorm import decompose_rmsnorm

from .binding import TARGET_IDENTITY, AtlasNativeBinding

RMSNORM_POLICY = "bf16-square-grouped32-fp32-tree-rne-sqrt-recip-gamma-v1"


def input_spec():
    from .pointwise import input_spec as base

    return replace(base(), policy=RMSNORM_POLICY + "-input",
                   admitted_code_ranges=((0, 0), (0x2f80, 0x4780), (0x8000, 0x8000), (0xaf80, 0xc780)))


@dataclass(frozen=True)
class RMSNormPrecision:
    source: KernelRequest
    request: KernelRequest
    source_region: object

    @property
    def input_ids(self):
        return self.source_region.data, self.source_region.gamma

    @property
    def operand_specs(self):
        return input_spec(), input_spec()

    def record(self):
        return dict(schema="atlas.rmsnorm_precision.v1", source_request_digest=self.source.digest(),
                    request_digest=self.request.digest(), source_region=self.source_region.record(),
                    input_conversions=[s.record() for s in self.operand_specs],
                    arithmetic=RMSNORM_POLICY, output_conversion="bf16-to-f32-exact",
                    qualification="diagnostic_precision_change; original_model_quality_unmeasured")


def prepare_source(source):
    if (source.target_identity != TARGET_IDENTITY or source.lowering_policy != "diagnostic" or len(source.outputs) != 1
            or source.physical_extents or source.boundary_layouts or source.output_layouts):
        raise ValueError("RMSNorm needs one selected diagnostic source result")
    region = recognize_rmsnorm(source, source.outputs[0])
    accounted = {region.data, region.gamma, *region.computations, *region.initializers}
    if {n.id for n in source.nodes} != accounted or any(source.node(k).effect not in {"input", "constant"} for k in (region.data, region.gamma)):
        raise ValueError("RMSNorm source has unaccounted computations or boundaries")
    constants = {c.node_id: c for c in source.constants}
    for key in (region.data, region.gamma):
        if source.node(key).effect == "constant":
            import numpy as np

            binding = constants[key]
            if binding.encoding != "f32-le-raw":
                raise ValueError("RMSNorm parameter must contain exact f32 bytes")
            input_spec().encode(np.frombuffer(bytes.fromhex(binding.data_hex), dtype="<f4"))
    epsilon = (region.epsilon_f32_bits + 0x7fff + ((region.epsilon_f32_bits >> 16) & 1)) >> 16
    request = replace(request_for(region.shape, epsilon_bf16_bits=epsilon), source_identity=source.digest())
    prepare_rmsnorm(request)
    return RMSNormPrecision(source, request, region)


def request_for(shape, *, epsilon_bf16_bits=0x3728):
    data = SemanticNode("data", "input", (), TensorType(tuple(shape), "bf16", "raw-bit-pattern"), effect="input")
    gamma = SemanticNode("gamma", "input", (), TensorType((shape[1],), "bf16", "raw-bit-pattern"), effect="input")
    result = SemanticNode("result", "rmsnorm", (data.id, gamma.id), TensorType(tuple(shape), "bf16", RMSNORM_POLICY),
                          (("axis", 1), ("epsilon_bf16_bits", epsilon_bf16_bits)))
    return KernelRequest((data, gamma, result), (result.id,), ("external",),
                         ((data.id, "external"), (gamma.id, "external")), TARGET_IDENTITY, "diagnostic")


def prepare_rmsnorm(source):
    if source.target_identity != TARGET_IDENTITY or source.lowering_policy != "diagnostic":
        raise ValueError("RMSNorm needs the selected diagnostic target")
    catalog = {d.computation: d for d in AtlasNativeBinding.profile().descriptors}
    roles = {"multiply": "mul_bf16_zero", "add": "add_bf16_zero", "row_sum": "row_sum_bf16",
             "sqrt": "sqrt_bf16_bounded", "reciprocal": "recip_bf16_bounded"}
    return decompose_rmsnorm(source, tile_type=TensorType((32, 32), "bf16", "raw-bit-pattern"),
                             operations={role: catalog[name] for role, name in roles.items()}, policy=RMSNORM_POLICY)
