"""Source-float entry for the checked row-softmax C/H package."""

from __future__ import annotations

import json
from pathlib import Path

from merlin.compile.c_conversions import render_f32_operand_c
from merlin.frontends.compile_inputs import load_compile_inputs
from merlin.semantic_compiler.guarded_softmax_source import GuardedMaskedRowSoftmaxSource
from merlin.semantic_compiler.model import TensorType
from merlin.semantic_compiler.precision_intake import SourceValueRef
from merlin.semantic_compiler.precision_transform import BoundaryConversion

from .softmax_precision import input_spec, lower_source_softmax


def prepare_source_c_package(
    source_path: Path,
    policy_path: Path,
    compiled: Path,
    name: str,
    *,
    entry: str | None,
    target_source: Path,
    compile_inputs: Path | None = None,
    runtime_linkage: str = "embedded",
):
    from .softmax_c import _prepare_c_package

    source = source_path.read_text()
    policy_bytes = policy_path.read_bytes()
    policy = json.loads(policy_bytes)
    original = None
    if compile_inputs is not None:
        staged = load_compile_inputs(compile_inputs)
        if (
            staged.program != source
            or staged.precision_policy_document != policy_bytes
            or entry is not None
            and entry != staged.entry
        ):
            raise ValueError("source C export differs from staged compiler inputs")
        entry, original = staged.entry, staged.original_source_sha256
    if not entry:
        raise ValueError("source row-softmax export needs a selected entry")
    precision = lower_source_softmax(source, entry=entry, policy=policy, original_source_sha256=original)
    result, files = _prepare_c_package(
        precision.request,
        compiled,
        name,
        target_source=target_source,
        precision=precision,
        runtime_linkage=runtime_linkage,
    )
    if source_path.read_text() != source or policy_path.read_bytes() != policy_bytes:
        raise ValueError("source softmax inputs changed during C export")
    return result, files


def render_source_interface(precision, name: str, raw_workspace: int):
    shape = precision.request.nodes[0].type.shape
    count = shape[0] * shape[1]
    masked = hasattr(precision.source_region, "mask")
    guarded = isinstance(precision.source_region, GuardedMaskedRowSoftmaxSource)
    input_type = "void" if guarded else "float"
    encoded, floats, scratch = count * 2, count * 4, count * (6 if masked else 4)
    output_offset = encoded * (2 if masked else 1)
    arity = 2 if masked else 1
    mask_checks = ""
    if masked:
        from math import prod

        from merlin.compile.c_softmax_mask import render_masked_softmax_inputs

        region = precision.source_region
        mask_bytes = prod(region.mask_shape) * (1 if guarded else 4)
        helpers = render_masked_softmax_inputs(region, region.mask_shape, input_spec(), b"\x00\xc2", name + "_masked")
        conversion_call = f"{name}_masked_prepare(inputs[0],inputs[1],ctx->workspace,ctx->workspace+{encoded}u)"
        mask_checks = f"""
  if (!{name}_valid(inputs[1],{mask_bytes}u) ||
      {name}_overlap(inputs[1],{mask_bytes}u,out,{floats}u) ||
      {name}_overlap(inputs[1],{mask_bytes}u,ctx->workspace,{scratch}u) ||
      {name}_overlap(inputs[1],{mask_bytes}u,region->workspace,{raw_workspace}u) ||
      {name}_overlap(inputs[1],{mask_bytes}u,ctx,sizeof(*ctx)) ||
      {name}_overlap(inputs[1],{mask_bytes}u,region,sizeof(*region)) ||
      {name}_overlap(inputs[1],{mask_bytes}u,region->host,sizeof(*region->host))) return {name.upper()}_INVALID;
"""
    else:
        conversion_call = f"{name}_encode_source(inputs[0],ctx->workspace)"
    conversion = BoundaryConversion(
        "scores",
        SourceValueRef("argument", 0),
        TensorType(shape, "f32", "raw-bit-pattern"),
        precision.request.nodes[0].type,
        (0, 1),
        input_spec().policy,
        input_spec(),
    )
    upper = name.upper()
    declarations = f"""
#define {upper}_SOURCE_INPUT_COUNT {arity}u
#define {upper}_SOURCE_WORKSPACE_BYTES {scratch}u
typedef struct {{ {name}_context *region; uint8_t *workspace; }} {name}_source_context;
/* Bind separate conversion scratch; both workspaces survive a timed-out call. */
{name}_status {name}_source_bind({name}_source_context *, {name}_context *, void *, size_t);
/* Dense source f32 scores and f32 result; recorded BF16 precision change inside. */
{name}_status {name}_source_run({name}_source_context *, const {input_type} *const *inputs,
                               float *output, atlas_host_result *device_result);
int {name}_source_dispatch(void *, const void *const *inputs, void *const *outputs);
"""
    source = (
        (helpers if masked else render_f32_operand_c(conversion, name + "_encode_source"))
        + f"""
{name}_status {name}_source_bind({name}_source_context *ctx, {name}_context *region,
                                void *workspace, size_t bytes) {{
  if (!ctx || !region || !region->host || region->host->active || !region->workspace ||
      bytes<{scratch}u || !{name}_valid(workspace,{scratch}u) ||
      {name}_overlap(workspace,{scratch}u,ctx,sizeof(*ctx)) ||
      {name}_overlap(workspace,{scratch}u,region,sizeof(*region)) ||
      {name}_overlap(workspace,{scratch}u,region->host,sizeof(*region->host)) ||
      {name}_overlap(workspace,{scratch}u,region->workspace,{raw_workspace}u) ||
      {name}_overlap(ctx,sizeof(*ctx),region,sizeof(*region)) ||
      {name}_overlap(ctx,sizeof(*ctx),region->host,sizeof(*region->host)) ||
      {name}_overlap(ctx,sizeof(*ctx),region->workspace,{raw_workspace}u)) return {upper}_INVALID;
  *ctx=({name}_source_context){{region,workspace}};
  return {upper}_OK;
}}
{name}_status {name}_source_run({name}_source_context *ctx, const {input_type} *const *inputs,
                               float *out, atlas_host_result *device) {{
  if (device) *device=ATLAS_HOST_INVALID;
  if (!ctx || !ctx->region || !ctx->region->host || !ctx->workspace || !inputs)
    return {upper}_INVALID;
  {name}_context *region=ctx->region;
  if (region->host->active) {{ if (device) *device=ATLAS_HOST_BUSY; return {upper}_DEVICE_ERROR; }}
  if (!{name}_valid(inputs[0],{floats}u) || !{name}_valid(out,{floats}u) ||
      {name}_overlap(inputs[0],{floats}u,out,{floats}u) ||
      {name}_overlap(inputs[0],{floats}u,ctx->workspace,{scratch}u) ||
      {name}_overlap(out,{floats}u,ctx->workspace,{scratch}u) ||
      {name}_overlap(inputs[0],{floats}u,region->workspace,{raw_workspace}u) ||
      {name}_overlap(out,{floats}u,region->workspace,{raw_workspace}u) ||
      {name}_overlap(inputs[0],{floats}u,ctx,sizeof(*ctx)) ||
      {name}_overlap(out,{floats}u,ctx,sizeof(*ctx)) ||
      {name}_overlap(inputs[0],{floats}u,region,sizeof(*region)) ||
      {name}_overlap(out,{floats}u,region,sizeof(*region)) ||
      {name}_overlap(inputs[0],{floats}u,region->host,sizeof(*region->host)) ||
      {name}_overlap(out,{floats}u,region->host,sizeof(*region->host))) return {upper}_INVALID;
{mask_checks}  if ({conversion_call}) return {upper}_INPUT_DOMAIN;
  {name}_status status={name}_run(region,ctx->workspace{",ctx->workspace+" + str(encoded) + "u" if masked else ""},
                                ctx->workspace+{output_offset}u,device);
  if (status!={upper}_OK) return status;
  for (size_t i=0; i<{count}u; ++i) {{
    uint32_t bits=((uint32_t)ctx->workspace[{output_offset}u+i*2u] |
                   ((uint32_t)ctx->workspace[{output_offset}u+i*2u+1u]<<8))<<16;
    __builtin_memcpy(out+i,&bits,4u);
  }}
  return {upper}_OK;
}}
int {name}_source_dispatch(void *ctx, const void *const *inputs, void *const *outputs) {{
  if (!inputs || !outputs) return {upper}_INVALID;
  const {input_type} *source[{arity}]={{0}}; source[0]=inputs[0];
  {"source[1]=inputs[1];" if masked else ""}
  return (int){name}_source_run(ctx,source,outputs[0],0);
}}
"""
    )
    interface = {
        "entry_function": name + "_source_run",
        "dispatch_function": name + "_source_dispatch",
        "input_count": arity,
        "shape": list(precision.source_region.shape) if precision.source_region is not None else list(shape),
        **(
            {"dense_row_view": precision.source_region.record()["dense_row_view"]}
            if precision.source_region is not None and len(precision.source_region.shape) != 2
            else {}
        ),
        "workspace_bytes": scratch,
        "input_dtype": "mixed" if guarded else "f32",
        **({"input_dtypes": ["f32", "i1"], "mask_semantics": "true-excludes-score"} if guarded else {}),
        "output_dtype": "f32",
        "precision_plan": precision.record(),
    }
    mask_note = (
        "The second input contains Boolean exclusion bytes: zero keeps a score and one masks it. "
        "Fully masked rows return positive zero.\n"
        if guarded
        else ""
    )
    readme = f"""
## Source-float entry

This package also contains the source-bound interface. After binding the raw
region, call `{name}_source_bind` with separate
`{upper}_SOURCE_WORKSPACE_BYTES` bytes. `{name}_source_run` accepts one dense f32
score tensor through its input-pointer array and returns dense f32 probabilities.
`{name}_source_dispatch` exposes the same call to Merlin's dispatch program.
The input is rounded to BF16 RNE, then checked against the recorded score-code
domain. Values outside that domain are rejected rather than clipped to scores.
The device uses the selected row-softmax policy; output BF16 is widened exactly.
{mask_note}
Keep both contexts and workspaces alive through completion, including timeouts.
The output stays unchanged on failure. This precision change has no model-quality
qualification merely because the C interface was generated.
"""
    return declarations, source, interface, readme
