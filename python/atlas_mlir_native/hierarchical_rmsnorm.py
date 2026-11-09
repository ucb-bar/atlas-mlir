"""Source-bound hierarchical BF16 RMSNorm using existing native tile primitives.

The transport ledger copies completed bytes into partial columns. Arithmetic
is exclusively in typed native requests; this module emits no host arithmetic,
assembly, register assignments, or schedules. C dispatch is a separate task.
"""

from dataclasses import dataclass
from hashlib import sha256
import json
from math import prod
from pathlib import Path

from merlin.semantic_compiler.model import ConstantBinding, KernelRequest, SemanticNode, TensorType, pointwise_identity_maps
from merlin.semantic_compiler.precision_values import FloatBoundarySpec
from merlin.semantic_compiler.rmsnorm_source import recognize_rmsnorm

from .binding import (AtlasNativeBinding, TARGET_IDENTITY, BF16_BINARY_ATTRS,
                      BF16_UNARY_ATTRS, BF16_ROW_REDUCTION_ATTRS, _row_reduce_maps)
from .pack_lane import ROUNDED_PACK_DOMAIN, ROUNDED_PACK_POLICY

POLICY = "f32-private-rmsnorm-bf16-tree32-group1024-mean-chop-lut-fp8-cap448-v1"
EPSILON_SOURCE_BITS = "0x3727c5ac"
EPSILON_BF16 = 0x3728
BF16 = TensorType((32, 32), "bf16", "raw-bit-pattern")


def operand_spec():
    """Conservative operands; per-operation obligations remain mandatory."""
    return FloatBoundarySpec(POLICY + "-finite-bounded-input", "bf16", "encode",
        canonicalize_negative_zero=True, admitted_code_ranges=(
            (0, 0), ((127-24)<<7, (127+8)<<7), (0x8000, 0x8000),
            (0x8000|((127-24)<<7), 0x8000|((127+8)<<7))))


def runtime_admission(width):
    """Sufficient comparison-only admission before the first native launch.

    Positive tree rounding is monotone: a row sum cannot fall below its
    largest squared lane, or exceed the all-maximum-lane bound. Thus these
    comparisons imply mean0 or[2^-33,4] without host norm arithmetic. Epsilon
    is exactly21*2^-21, so its addition to every BF16 mean in that range fits
    at most24 FP32 significand bits before the selected BF16 chop.
    """
    if type(width) is not int or width not in (1024,2048,4096,8192,16384,32768):
        raise ValueError("unsupported hierarchical normalization width")
    exponent=(width.bit_length()-1-33+1)//2
    return dict(when="before first MMIO; after exact declared BF16 input conversion",
        hidden_abs_max_bf16=0x4000, nonzero_row_abs_max_min_bf16=(127+exponent)<<7,
        all_zero_row_admitted=True, gamma_conversion=operand_spec().record(),
        checks="word comparisons, row maxima, and zero tests only; no host normalization arithmetic",
        proof=dict(mean="zero or [2**-33,4]",epsilon="21*2**-21; every admitted BF16 mean+epsilon is exact FP32",
            sqrt_input="positive normal within[2**-63,2**63]", sqrt_output="[2**-9,4]",
            reciprocal_output="[2**-2,2**9]", products="all nonzero square, normalization, gamma, and prescale products normal finite BF16"))


def stage_requests(width, output_scale_exponent, source_identity):
    if type(width) is not int or width not in (1024, 2048, 4096, 8192, 16384, 32768):
        raise ValueError("hierarchical RMSNorm needs a power-of-two width1024..32768")
    if type(output_scale_exponent) is not int or not -32 <= output_scale_exponent <= 32:
        raise ValueError("hierarchical RMSNorm output exponent must be in [-32,32]")

    def request(kind, inputs, constants, steps, output):
        catalog = {**BF16_BINARY_ATTRS, **BF16_UNARY_ATTRS, **BF16_ROW_REDUCTION_ATTRS}
        def domain(key):
            operation = next(op for _, op, args in steps if key in args)
            return (("input_domain", ROUNDED_PACK_DOMAIN),) if operation == "pack_fp8" else (catalog[operation][0],)
        nodes = [SemanticNode(key, "input", (), BF16, domain(key), "input")
                 for key in inputs]
        bindings = []
        for key, bits in constants:
            nodes.append(SemanticNode(key, "constant", (), BF16, domain(key), effect="constant"))
            bindings.append(ConstantBinding(key, "bf16-le-raw", bits.to_bytes(2, "little").hex()*1024))
        for key, operation, arguments in steps:
            packed = operation == "pack_fp8"
            attrs = ((("input_domain", ROUNDED_PACK_DOMAIN), ("scale_code", 127)) if packed
                     else {**BF16_BINARY_ATTRS, **BF16_UNARY_ATTRS, **BF16_ROW_REDUCTION_ATTRS}[operation])
            type_ = TensorType((32, 32), "fp8_e4m3", ROUNDED_PACK_POLICY) if packed else BF16
            maps = _row_reduce_maps() if operation in BF16_ROW_REDUCTION_ATTRS else pointwise_identity_maps(2, len(arguments))
            nodes.append(SemanticNode(key, operation, tuple(arguments), type_, attrs, index_maps=maps))
        return KernelRequest(tuple(nodes), (output,), ("external",), tuple((key, "external") for key in inputs),
                             TARGET_IDENTITY, "diagnostic", sha256((source_identity+":"+kind).encode()).hexdigest(), tuple(bindings))

    exponent = width.bit_length()-1
    return {
        "square_reduce": request("square_reduce", ("hidden",), (), (
            ("square", "mul_bf16_zero", ("hidden", "hidden")),
            ("partial", "row_sum_bf16", ("square",))), "partial"),
        "reduce": request("reduce", ("partials",), (), (
            ("total", "row_sum_bf16", ("partials",)),), "total"),
        "inverse": request("inverse", ("total",), (("mean_scale", (127-exponent)<<7), ("epsilon", EPSILON_BF16)), (
            ("mean", "mul_bf16_zero", ("total", "mean_scale")),
            ("shifted", "add_bf16_zero", ("mean", "epsilon")),
            ("root", "sqrt_bf16_bounded", ("shifted",)),
            ("inverse", "recip_bf16_bounded", ("root",))), "inverse"),
        "publish": request("publish", ("hidden", "inverse", "gamma"),
            (("encode_scale", (127-output_scale_exponent)<<7),), (
            ("normalized", "mul_bf16_zero", ("hidden", "inverse")),
            ("weighted", "mul_bf16_zero", ("normalized", "gamma")),
            ("encoded", "mul_bf16_zero", ("weighted", "encode_scale")),
            ("fp8", "pack_fp8", ("encoded",))), "fp8"),
    }


@dataclass(frozen=True)
class HierarchicalRMSNormPlan:
    source: KernelRequest
    region: object
    output_scale_exponent: int
    provenance_sha256: str
    stages: dict

    @property
    def width(self):
        return self.region.matrix_shape[1]

    @property
    def row_groups(self):
        return (self.region.matrix_shape[0]+31)//32

    @property
    def launches_per_row_group(self):
        return 2*(self.width//32) + self.width//1024 + 2

    def record(self):
        tiles, groups = self.width//32, self.width//1024
        return dict(schema="atlas.hierarchical_rmsnorm_plan.v1", policy=POLICY,
            source_request_digest=self.source.digest(), source_region=self.region.record(),
            provenance_sha256=self.provenance_sha256, matrix_shape=list(self.region.matrix_shape),
            row_groups=self.row_groups, channel_tiles=tiles, reduction_groups=groups,
            launches_per_row_group=self.launches_per_row_group,
            launches_per_call=self.row_groups*self.launches_per_row_group,
            output_scale_exponent=self.output_scale_exponent, epsilon_bf16_bits="0x3728",
            operand_conversion=operand_spec().record(),
            sufficient_runtime_admission=runtime_admission(self.width),
            stages={kind:dict(request_digest=body.digest(), inputs=[n.id for n in body.nodes if n.effect=="input"],
                               outputs=list(body.outputs), repeats_per_row_group=tiles if kind in {"square_reduce","publish"}
                               else groups+1 if kind=="reduce" else 1)
                    for kind, body in self.stages.items()},
            transport=[
                dict(source="square_reduce.partial", destination="reduce.partials", instances=groups,
                     source_tiles_per_instance=32, source_column=0, destination_columns=list(range(32)),
                     operation="copy-completed-bf16-row-lane", initialization="positive-zero", arithmetic=False),
                dict(source="reduce.group_total", destination="reduce.partials", instances=1,
                     source_tiles_per_instance=groups, source_column=0, destination_columns=list(range(groups)),
                     operation="copy-completed-bf16-row-lane", initialization="positive-zero", arithmetic=False),
                dict(source="reduce.final_total", destination="inverse.total", operation="tile-byte-copy", arithmetic=False),
                dict(source="inverse.inverse", destination="publish.inverse", operation="tile-byte-copy", arithmetic=False),
                dict(source="gamma", destination="publish.gamma", operation="copy-channel-words-broadcast32rows", arithmetic=False)],
            input_retention="H and gamma immutable through final publication; outside H consumers preserved",
            transport_encoding=dict(logical="32x32 BF16 little-endian row-major,2048bytes",
                native="existing pack_bf16_pair_halves/unpack_bf16_pair_halves byte permutation",
                publication="32x32 FP8 row-major,1024bytes", padded_rows="positive-zero H; gamma broadcast immutable"),
            reduction_order="32 adjacent squared lanes ->32 partials/group1024 ->groups with positive-zero tail",
            arithmetic="BF16-RNE products and tree results; BF16-chop epsilon; sqrt LUT then reciprocal LUT",
            publication=dict(dtype="fp8_e4m3", numerical_policy=ROUNDED_PACK_POLICY,
                operation="exact power-of-two BF16 prescale then native pack127", cap=448,
                underflow="normal-significand-RNE-before-positive-zero-flush",
                optional_source_view="exact FP8 unpack and BF16-to-f32 widen; never omit FP8 boundary"),
            numerical_obligations=[
                "every external H/gamma word satisfies operand_conversion, with positive-zero padding",
                "all nonzero multiply results remain normal BF16 and finite; every tree result is normal or positive zero",
                "each shifted mean addition has an exact normal-or-zero FP32 sum before BF16 chop",
                "sqrt input and reciprocal input are positive normal BF16 in [2**-63,2**63]",
                "encoded publication is finite normal BF16 or signed zero before native pack",
                "runtime must discharge every operation-domain obligation before publishing any result"],
            qualification="diagnostic precision change; C dispatcher and combined decoder qualification PENDING")


def prepare_source(source, policy):
    required = {"policy", "source_request_digest", "provenance_sha256", "output_scale_exponent"}
    if not isinstance(policy, dict) or set(policy)-{"result"} != required:
        raise ValueError("hierarchical RMSNorm needs its exact explicit policy fields")
    if (policy["policy"] != POLICY or policy["source_request_digest"] != source.digest()
            or source.target_identity != TARGET_IDENTITY or source.lowering_policy != "diagnostic"):
        raise ValueError("hierarchical RMSNorm source policy binding differs")
    digest = policy["provenance_sha256"]
    if not isinstance(digest, str) or len(digest)!=64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("hierarchical RMSNorm requires provenance SHA-256")
    result = policy.get("result")
    if result is None:
        if len(source.outputs) != 1:
            raise ValueError("hierarchical RMSNorm needs an explicit result in a larger source graph")
        result = source.outputs[0]
    region = recognize_rmsnorm(source, result, epsilon_bits=EPSILON_SOURCE_BITS)
    identity = sha256(json.dumps(dict(policy=POLICY, source_request_digest=source.digest(),
        provenance_sha256=digest, output_scale_exponent=policy["output_scale_exponent"], result=result),
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    stages = stage_requests(region.matrix_shape[1], policy["output_scale_exponent"], identity)
    return HierarchicalRMSNormPlan(source, region, policy["output_scale_exponent"], digest, stages)


def pack_partial_columns(partials):
    """Byte-only row-lane transport, independent of values and arithmetic."""
    if not 1 <= len(partials) <= 32 or any(type(p) is not bytes or len(p)!=2048 for p in partials):
        raise ValueError("partial transport needs1..32 logical32x32 BF16 tiles")
    result = bytearray(2048)
    for column, tile in enumerate(partials):
        for row in range(32):
            result[(row*32+column)*2:(row*32+column+1)*2] = tile[row*64:row*64+2]
    return bytes(result)


def compile_source(snapshot, source, policy, *, target_source, destination, limits):
    """Compile four reusable native templates; no runtime C is generated here."""
    plan = prepare_source(source, policy)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    compiled = {}
    for kind, request in plan.stages.items():
        abi = AtlasNativeBinding.plan_boundary_abi(request)
        manifest = AtlasNativeBinding.compile(snapshot, request, fixed_inputs=abi["fixed_inputs"],
            fixed_outputs=tuple(abi["fixed_outputs"]), target_source=target_source, destination=destination/kind, limits=limits)
        compiled[kind] = dict(request_digest=request.digest(), program_sha256=manifest["binary_sha256"],
            manifest_path=kind+"/manifest.json",
            manifest_sha256=sha256((destination/kind/"manifest.json").read_bytes()).hexdigest(),
            execution_plan_sha256=manifest["execution_plan_sha256"])
    record = plan.record() | {"compiled_stages": compiled}
    (destination/"hierarchical-rmsnorm-plan.json").write_text(json.dumps(record, sort_keys=True, indent=2)+"\n")
    (destination/"semantic-source.json").write_text(json.dumps(source.record(), sort_keys=True, indent=2)+"\n")
    return record
