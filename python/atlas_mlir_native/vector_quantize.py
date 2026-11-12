"""Exact BF16 activation quantization through the ordinary virtual compiler.

The application schedule owns complete authored groups; each body uses counted
tile loops, vector extrema, and raw PACK. Scalar exponent selection and the
finite-only correction use ordinary allocated i32 and slot operations.
"""
from dataclasses import replace
from . import activation_quantize_device as scalar


def prepare_plan(source):
    return replace(scalar.prepare_plan(source), engine='vector')


def compile_source(snapshot, region, *, target_source, destination, limits):
    return scalar.compile_source(snapshot, region, target_source=target_source,
                                 destination=destination, limits=limits, _plan=prepare_plan(region))


def prepare_c_package(region, compiled, name, *, target_source, runtime_linkage='separate'):
    plan=prepare_plan(region)
    result=scalar.prepare_c_package(region, compiled, name, target_source=target_source,
                                    runtime_linkage=runtime_linkage, _plan=plan)
    vector_path=plan.width==2 and not any(n%32 for n in (plan.count,plan.group_elements,plan.head_elements))
    return replace(result,description={**result.description, 'kind':'vector_quantize',
                   'quantizer_engine':'vector' if vector_path else 'unsupported_device_body',
                   'launches_per_invocation':1,
                   'scalar_correction':'counted indexed finite-only scan on exceptional full tiles; exponent selection once per group',
                   'pipeline':'scheduled common native pipeline and six source contracts'})
