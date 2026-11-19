"""Publish a selected BF16 pointwise result through a private exact cast.

The selected pointwise operation and all input encoders stay unchanged. Only
the host publication omits its exact widening and the following device cast.
This is a target-representation proof, not reassociation of source FP32 math.
"""
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from merlin.compile.result_narrowing import split_private_bf16_cast

from .composition_adapter import c_region, contiguous, device_casts, fuse, kind_of, neighbor_pairs, source_names


@dataclass(frozen=True)
class Narrowing:
    source: object
    precision: object


def prepare(source, adapter):
    from .pointwise import PointwisePrecision, prepare_source
    from .pointwise_math import MathPointwisePrecision
    from .gelu import GeluPrecision
    from .sigmoid import SigmoidPrecision, SiluPrecision

    producer = split_private_bf16_cast(source)
    result = producer.node(producer.outputs[0])
    precision = prepare_source(producer,
        allow_math=True, bf16_policy=adapter.bf16_pointwise,
        rne_add_policy=adapter.qualified_rne_add if result.op == 'f32_add' else None,
        exp_policy=adapter.source_exp_policy)
    # These bounded operators cannot produce NaN: sums/products have finite
    # normal-or-zero inputs of magnitude at most 2**63; LUT division/rsqrt also
    # require a positive denominator of magnitude at least 2**-63. Their selected
    # arithmetic produces BF16, and exact widening followed by RNE is identity
    # for every non-NaN BF16 encoding (including infinities and signed zero).
    bounded_pointwise = (type(precision) in (PointwisePrecision, MathPointwisePrecision)
            and precision.request.nodes[-1].op in {
                'mul_bf16_zero', 'add_bf16_rne', 'sub_bf16_rne',
                'divide_bf16_lut', 'rsqrt_bf16_lut'})
    # The separately selected GELU policy admits |x| <= 2**32. Its cube and
    # scaled inner sum remain finite; tanh lies in [-1, 1], so the final
    # half*x*(1+tanh) is finite as well. All steps retain the existing BF16
    # rounding and coefficient encodings, including both signed zeros.
    bounded_gelu = (type(precision) is GeluPrecision and adapter.gelu
                    and precision.request.nodes[-1].op == 'tanh_gelu_bf16_lut')
    # The selected sigmoid has a finite positive denominator >= 1, and
    # returns finite BF16 in (0, 1]. Its optional product keeps |x| <= 32.
    bounded_sigmoid = (adapter.sigmoid
        and (type(precision) is SigmoidPrecision and precision.request.nodes[-1].op == 'sigmoid_bf16_lut'
             or type(precision) is SiluPrecision and adapter.pointwise
             and precision.request.nodes[-1].op == 'silu_bf16_lut'))
    if not (bounded_pointwise or bounded_gelu or bounded_sigmoid):
        raise ValueError('pointwise narrowing has no non-NaN result proof for this policy')
    return Narrowing(source, precision)


def alternatives(adapter, source, choices):
    from .pointwise import plan_abi

    def prepared(region, _):
        group = prepare(region, adapter)
        plan_abi(group.precision.request)
        return group
    accepts = lambda row: kind_of(row) in ('pointwise', 'pointwise_math') and contiguous(row)
    return fuse(adapter, source, choices, '_pointwise_narrowing_groups', 'pointwise_narrowing',
                neighbor_pairs(choices, device_casts(choices), accepts), prepare=prepared,
                cycles='unchanged pointwise program; exact cast omitted')


def compile_request(snapshot, request, *, target_source, destination, limits, **options):
    """Compile one unchanged pointwise request at its ordinary boundary ABI."""
    from .pointwise import compile_pointwise, plan_abi

    abi = plan_abi(request)
    compile_pointwise(snapshot, request, fixed_inputs=abi['fixed_inputs'], fixed_outputs=tuple(abi['fixed_outputs']),
                      target_source=target_source, destination=destination, limits=limits, **options)


def compile_group(group, snapshot, *, target_source, destination, limits):
    compile_request(snapshot, group.precision.request, target_source=target_source, destination=destination,
                    limits=limits)


def export_precision(precision, compiled, name, *, target_source, outputs, programs, inputs=None, launches=None,
                     manifest=False, input_dtype=None, output_dtype=None, static_output_exponent=None, **fields):
    """Export a compiled pointwise precision as one source-dispatch C region described by `fields`."""
    from .pointwise_c import prepare_precision_c_package

    compiled = Path(compiled)
    record, files = prepare_precision_c_package(precision, compiled, name, target_source=Path(target_source),
        input_dtype=input_dtype, output_dtype=output_dtype, static_output_exponent=static_output_exponent)
    record.update(fields, **source_names(name), launches_per_invocation=launches if launches is not None
                  else record.get('streaming', {}).get('launch_count', 1))
    if manifest:
        record['native_manifest_sha256'] = sha256((compiled / 'manifest.json').read_bytes()).hexdigest()
    return c_region(name, tuple(record['input_node_ids']) if inputs is None else inputs, outputs, files, programs,
                    record)


def export_group(group, compiled, name, *, target_source):
    return export_precision(group.precision, compiled, name, target_source=target_source, output_dtype='bf16',
        outputs=group.source.outputs, programs=('segments/' + name,), manifest=True, kind='pointwise_narrowing',
        output_node_ids=list(group.source.outputs), source_request_digest=group.source.digest(),
        narrowing_proof='non-NaN BF16 result; exact widening followed by RNE is identity')
