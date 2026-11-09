"""Selected exp-pack, denominator, and value-contraction precision.

Build typed semantic DAGs from a recognized private source softmax/PV region.
Merlin selects the target instructions and Atlas composes their verified native
lowerings. This module contains no assembly, register assignment, or schedule.
"""

from dataclasses import dataclass
from hashlib import sha256
import json
from math import prod

from merlin.semantic_compiler.model import ConstantBinding, KernelRequest, SemanticNode, TensorType, pointwise_identity_maps
from merlin.semantic_compiler.precision_values import FloatBoundarySpec
from merlin.semantic_compiler.softmax_value_source import recognize_softmax_value

from .binding import (BF16_BINARY_ATTRS, BF16_ROW_REDUCTION_ATTRS, BF16_UNARY_ATTRS,
                      MXU1_CAP448_INPUT_DOMAIN, MXU1_CAP448_POLICY, TARGET_IDENTITY,
                      _mxu0_accumulate_maps, _mxu0_maps, _row_reduce_maps)
from .pack_lane import ROUNDED_PACK_DOMAIN, ROUNDED_PACK_POLICY

POLICY = "f32-private-softmax-pv-model47-exp256-fp8-denominator-first-bf16-v1"
SCORE_POLICY = "f32-to-bf16-rne-bounded40-row-scores-v1"


def score_spec():
    # A difference of admitted nonzero BF16 scores is normal or zero and lies
    # in [-80,80]. Subtracting the row maximum narrows that to [-80,0].
    return FloatBoundarySpec(SCORE_POLICY, "bf16", "encode", admitted_code_ranges=(
        (0, 0), (0x0400, 0x4220), (0x8000, 0x8000), (0x8400, 0xc220)))


@dataclass(frozen=True)
class QuantizedSoftmaxValuePlan:
    source: KernelRequest
    region: object
    value_scale_exponent: int
    body: KernelRequest
    provenance_sha256: str | None = None

    @property
    def groups(self):
        return prod(self.region.batch_shape) * ((self.region.query_rows + 31) // 32)

    def record(self):
        return {
            "schema": "atlas.quantized_softmax_value_plan.v1", "policy": POLICY,
            "provenance_sha256": self.provenance_sha256,
            "source_request_digest": self.source.digest(), "source_region": self.region.record(),
            "body_request_digest": self.body.digest(), "row_groups": self.groups,
            "tile_shape": [32, 32], "key_tiles": (self.region.keys + 31) // 32,
            "channel_tiles": (self.region.channels + 31) // 32,
            "value_scale_exponent": self.value_scale_exponent, "score_conversion": score_spec().record(),
            "mask_conversion": "source additive zero or negative infinity or lowest finite f32 to BF16 keep one/zero",
            "masked_score": "BF16 -40, with keep zero; <= every admitted valid score",
            "exponential": "selected BF16 EXP, multiply256, E4M3 RNE cap448 positive flushed zero",
            "denominator": "model47 ordered MXU1 FP8 exponential dot FP8 ones; BF16 continuation per32 keys; floor1",
            "numerator": "model47 ordered MXU1 same FP8 exponentials dot FP8 values; BF16 continuation per32 keys",
            "restore": "BF16 reciprocal(denominator), multiply2**value_scale, then BF16 multiply numerator",
            "output": "canonical BF16 widened to source f32; following pack scale remains separately selected",
            "qualification": "diagnostic precision change; full source fidelity and emitted execution required",
        }


def body_request(keys, channels, *, value_scale_exponent, source_identity):
    """One complete 32-row group, retaining all logical key columns.

    The bounded score contract makes the kit's -128 clamp inactive. Disallowed
    lanes use -40 before max and a keep-zero multiply after EXP; they therefore
    cannot affect a nonempty row maximum, denominator, or numerator. Empty and
    padded rows produce positive zero after the denominator floor.
    """
    if any(type(n) is not int or n <= 0 for n in (keys, channels)) or keys > 256:
        raise ValueError("quantized softmax value requires 1..256 keys and positive channels")
    if type(value_scale_exponent) is not int or not -32 <= value_scale_exponent <= 32:
        raise ValueError("quantized softmax value scale must be an integer in [-32,32]")
    bf16 = TensorType((32, 32), "bf16", "raw-bit-pattern")
    fp8 = TensorType((32, 32), "fp8_e4m3", "raw-bit-pattern")
    packed = TensorType((32, 32), "fp8_e4m3", ROUNDED_PACK_POLICY)
    dot = TensorType((32, 32), "bf16", MXU1_CAP448_POLICY)
    key_tiles, channel_tiles = (keys + 31) // 32, (channels + 31) // 32
    boundary, constants, bindings, computations = [], [], [], []

    def input_(name, type_, domain):
        boundary.append(SemanticNode(name, "input", (), type_, (("input_domain", domain),), "input"))
        return name

    scores = [input_(f"scores_{k}", bf16, "bf16-bounded40-row-scores-or-minus40-padding-v1") for k in range(key_tiles)]
    keep = [input_(f"keep_{k}", bf16, "exact-zero-or-one") for k in range(key_tiles)]
    values = [[input_(f"value_{n}_{k}", fp8, MXU1_CAP448_INPUT_DOMAIN) for k in range(key_tiles)] for n in range(channel_tiles)]

    def constant(name, type_, code, encoding):
        constants.append(SemanticNode(name, "constant", (), type_, effect="constant"))
        width = 2 if type_.dtype == "bf16" else 1
        bindings.append(ConstantBinding(name, encoding, (code.to_bytes(width, "little") * 1024).hex()))
        return name

    one = constant("one", bf16, 0x3f80, "bf16-le-raw")
    multiplier = constant("exp_scale", bf16, 0x4380, "bf16-le-raw")
    value_scale = constant("value_scale", bf16, (127 + value_scale_exponent) << 7, "bf16-le-raw")
    ones = constant("denominator_weights", fp8, 0x38, "fp8_e4m3-raw")

    def operation(op, inputs, *, type_=bf16, attrs=None, maps=None):
        name = f"v{len(computations)}"
        if attrs is None:
            attrs = {**BF16_BINARY_ATTRS, **BF16_UNARY_ATTRS, **BF16_ROW_REDUCTION_ATTRS}[op]
        if maps is None:
            maps = _row_reduce_maps() if op in BF16_ROW_REDUCTION_ATTRS else pointwise_identity_maps(2, len(inputs))
        computations.append(SemanticNode(name, op, tuple(inputs), type_, attrs, index_maps=maps))
        return name

    def contraction(lhs, rhs, seed=None):
        return operation("matmul" if seed is None else "matmul_accumulate",
                         (lhs, rhs) if seed is None else (lhs, rhs, seed), type_=dot, attrs=(),
                         maps=_mxu0_maps() if seed is None else _mxu0_accumulate_maps())

    maxima = [operation("row_max_bf16", (x,)) for x in scores]
    maximum = maxima[0]
    for partial in maxima[1:]:
        maximum = operation("max_bf16", (maximum, partial))
    exponentials = []
    denominator = None
    for score, mask in zip(scores, keep, strict=True):
        shifted = operation("sub_bf16_rne", (score, maximum))
        exponential = operation("exp_bf16", (shifted,))
        exponential = operation("mul_bf16_zero", (exponential, mask))
        scaled = operation("mul_bf16_zero", (exponential, multiplier))
        encoded = operation("pack_fp8", (scaled,), type_=packed,
                            attrs=(("input_domain", ROUNDED_PACK_DOMAIN), ("scale_code", 127)))
        exponentials.append(encoded)
        denominator = contraction(encoded, ones, denominator)
    denominator = operation("max_bf16", (denominator, one))
    reciprocal = operation("recip_bf16_wide", (denominator,))
    restore = operation("mul_bf16_zero", (reciprocal, value_scale))
    outputs = []
    for weights in values:
        numerator = None
        for exponential, weight in zip(exponentials, weights, strict=True):
            numerator = contraction(exponential, weight, numerator)
        outputs.append(operation("mul_bf16_zero", (numerator, restore)))
    identity = sha256(json.dumps([source_identity, POLICY, keys, channels, value_scale_exponent]).encode()).hexdigest()
    return KernelRequest(tuple((*boundary, *constants, *computations)), tuple(outputs),
                         ("external",) * len(outputs), tuple((n.id, "external") for n in boundary),
                         TARGET_IDENTITY, "diagnostic", identity, tuple(bindings))


def prepare_source(source, *, value_scale_exponent, masked=True, provenance_sha256=None):
    if source.target_identity != TARGET_IDENTITY or source.lowering_policy != "diagnostic" or len(source.outputs) != 1:
        raise ValueError("quantized softmax value needs one selected diagnostic source region")
    if provenance_sha256 is not None and (not isinstance(provenance_sha256, str) or len(provenance_sha256) != 64
            or any(c not in "0123456789abcdef" for c in provenance_sha256)):
        raise ValueError("quantized softmax value requires a provenance SHA-256")
    region = recognize_softmax_value(source, source.outputs[0], masked=masked)
    body = body_request(region.keys, region.channels, value_scale_exponent=value_scale_exponent,
                        source_identity=source.digest())
    return QuantizedSoftmaxValuePlan(source, region, value_scale_exponent, body, provenance_sha256)


def compile_source(snapshot, source, *, value_scale_exponent, target_source, destination, limits, provenance_sha256=None):
    from .binding import AtlasNativeBinding

    precision = prepare_source(source, value_scale_exponent=value_scale_exponent, provenance_sha256=provenance_sha256)
    abi = AtlasNativeBinding.plan_boundary_abi(precision.body)
    manifest = AtlasNativeBinding.compile(snapshot, precision.body, fixed_inputs=abi["fixed_inputs"],
        fixed_outputs=tuple(abi["fixed_outputs"]), target_source=target_source,
        destination=destination, limits=limits)
    for name, value in (("quantized-softmax-value-plan.json", precision.record()), ("semantic-source.json", source.record())):
        payload = (json.dumps(value, indent=2) + "\n").encode()
        (destination / name).write_bytes(payload)
        manifest["stage_artifacts"][name] = sha256(payload).hexdigest()
    manifest["source_softmax_value_request_digest"] = source.digest()
    manifest["source_softmax_value_plan_sha256"] = manifest["stage_artifacts"]["quantized-softmax-value-plan.json"]
    (destination / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    return manifest


def materialize_row_group(plan, *, scores, keep, values, valid_rows):
    """Prepare logical tile bytes for an independently qualified invocation.

    Inputs are complete BF16 score words, Boolean keep bytes, and already
    admitted FP8 value codes for one independent batch. This diagnostic helper
    never provides compiler inputs or chooses code from sample values.
    """
    import struct

    keys, channels = plan.region.keys, plan.region.channels
    if type(valid_rows) is not int or not 1 <= valid_rows <= 32:
        raise ValueError("row group must contain 1..32 logical rows")
    if (type(scores) is not bytes or len(scores) != valid_rows * keys * 2
            or type(keep) is not bytes or len(keep) != valid_rows * keys
            or type(values) is not bytes or len(values) != keys * channels):
        raise ValueError("row group input extent differs from its logical geometry")
    words = tuple(code for (code,) in struct.iter_unpack("<H", scores))
    for code, visible in zip(words, keep, strict=True):
        magnitude = code & 0x7fff
        if (visible not in (0, 1) or magnitude > 0x4220
                or magnitude and not 8 <= (code >> 7) & 255 <= 132):
            raise ValueError("row score or keep code leaves the selected domain")
    for code in values:
        exponent = code >> 3 & 15
        if code != 0 and (exponent == 0 or code & 127 == 127):
            raise ValueError("value code leaves finite-normal cap448 or positive zero domain")
    result = {}
    for key_tile in range((keys + 31) // 32):
        score, mask = bytearray(), bytearray()
        for row in range(32):
            for column in range(32):
                key = key_tile * 32 + column
                visible = row < valid_rows and key < keys and keep[row * keys + key]
                score.extend((words[row * keys + key] if visible else 0xc220).to_bytes(2, "little"))
                mask.extend((0x3f80 if visible else 0).to_bytes(2, "little"))
        result[f"scores_{key_tile}"] = bytes(score)
        result[f"keep_{key_tile}"] = bytes(mask)
        for channel_tile in range((channels + 31) // 32):
            # MXU weights are logical output-channel by reduction-key.
            result[f"value_{channel_tile}_{key_tile}"] = bytes(
                values[(key_tile * 32 + k) * channels + channel_tile * 32 + n]
                if key_tile * 32 + k < keys and channel_tile * 32 + n < channels else 0
                for n in range(32) for k in range(32))
    result.update({binding.node_id: bytes.fromhex(binding.data_hex) for binding in plan.body.constants})
    return result
