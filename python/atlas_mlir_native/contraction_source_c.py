"""Reuse source-float conversion around a checked Atlas matrix call."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass


@dataclass(frozen=True)
class SourceWrappers:
    region: dict | None
    declarations: str
    implementation: str
    private_headers: dict[str, str]
    packing: str


def prepare_source_wrappers(
    source,
    *,
    manifest,
    member,
    execution,
    name,
    workspace,
    m,
    n,
    compile_inputs=None,
    semantic_source=None,
    semantic_operand_scale_exponents=None,
    semantic_epilogue=None,
    seed=None,
):
    """Keep conversions outside raw native execution and preserve its C ABI."""
    upper = name.upper()

    def sha(data):
        return hashlib.sha256(data).hexdigest()

    region = None
    region_declarations, region_source = "", ""
    private_headers = {}
    conversion_source = ""
    precision, static, staged = None, (), None
    if seed and (compile_inputs is not None or semantic_source is not None):
        raise ValueError("completed accumulators require the raw BF16 interface, without source conversion")
    if compile_inputs is not None and semantic_source is not None:
        raise ValueError("select staged Linalg or a semantic source region, not both")
    if semantic_operand_scale_exponents is not None and semantic_source is None:
        raise ValueError("semantic operand scales require the exact semantic source")
    if semantic_epilogue is not None and semantic_source is None:
        raise ValueError("semantic epilogue requires the exact semantic source")
    if semantic_source is not None:
        from .application_precision import prepare_contraction

        precision = prepare_contraction(
            semantic_source,
            operand_scale_exponents=semantic_operand_scale_exponents,
            epilogue=semantic_epilogue,
        )
        if precision.request.digest() != source.digest() or manifest.get("constant_inputs"):
            raise ValueError("semantic contraction differs from the checked native source")
    elif compile_inputs is not None:
        from merlin.frontends.compile_inputs import load_compile_inputs

        from .binding import AtlasNativeBinding

        staged = load_compile_inputs(compile_inputs)
        published = json.loads(member("precision-plan.json"))
        precision = AtlasNativeBinding.prepare_precision_tile(
            staged,
            published["region_id"],
            whole_contraction=True,
        )
        if precision.record() != published or precision.request.digest() != source.digest():
            raise ValueError("source-bound region differs from the looped publication")
        static = AtlasNativeBinding.prepare_precision_constants(precision, compile_inputs, execution)
        if [item.record() for item in static] != manifest.get("constant_inputs"):
            raise ValueError("source-bound parameters differ from the looped publication")
        if any(item.payload is None for item in static):
            raise ValueError("source-bound parameter conversion is blocked")
        for item in static:
            if member(item.record()["path"]) != item.payload:
                raise ValueError("source-bound parameter bytes differ from the looped publication")
    if precision is not None:
        from merlin.common.paths import runtime_dir
        from merlin.compile.c_conversions import render_f32_operand_c

        constants = {item.role: item for item in static}
        arguments, embedded, runtime_inputs, runtime_conversions = [], [], [], []
        for conversion in precision.inputs:
            item = constants.get(conversion.source_role)
            if item is None:
                arguments.append(f"inputs[{len(runtime_inputs)}]")
                runtime_inputs.append(conversion.record())
                runtime_conversions.append(conversion)
            else:
                codes = conversion.convert_values(item.source.value).tobytes(order="C")
                symbol = f"{name}_parameter_{len(embedded)}"
                embedded.append(
                    f"static const uint8_t {symbol}[{len(codes)}] = {{" + ",".join(str(value) for value in codes) + "};"
                )
                arguments.append(symbol)
        count = len(runtime_inputs)
        input_check = f"if (!inputs) return {upper}_INVALID;" if count else "(void)inputs;"
        if count:
            pointer_values = ",".join(f"inputs[{i}]" for i in range(count))
            typed_inputs = f"const uint8_t *typed[{count}] = {{{pointer_values}}};"
        else:
            typed_inputs = "(void)inputs;"
        dispatch_check = "!outputs || !inputs" if count else "!outputs"
        dispatch_inputs = "typed" if count else "0"
        region_declarations = f"""
#define {upper}_REGION_INPUT_COUNT {count}u
/* Encoded region inputs in recorded operand order; declared parameters are embedded.
 * Source float conversion remains explicit work outside this byte API. */
{name}_status {name}_region_run({name}_context *context, const uint8_t *const *inputs,
                               uint8_t *output_bf16_le, atlas_host_result *device_result);
int {name}_region_dispatch(void *context, const void *const *inputs, void *const *outputs);
"""
        region_source = (
            "\n".join(embedded)
            + f"""
{name}_status {name}_region_run({name}_context *ctx, const uint8_t *const *inputs,
                               uint8_t *out, atlas_host_result *device) {{
  if (device) *device = ATLAS_HOST_INVALID;
  {input_check}
  return {name}_run(ctx,{arguments[0]},{arguments[1]},out,device);
}}
int {name}_region_dispatch(void *ctx, const void *const *inputs, void *const *outputs) {{
  if ({dispatch_check}) return {upper}_INVALID;
  {typed_inputs}
  return (int){name}_region_run(ctx,{dispatch_inputs},outputs[0],0);
}}
"""
        )
        region = {
            "id": precision.region_id,
            "precision_plan_sha256": sha(member("precision-plan.json"))
            if staged is not None
            else sha(json.dumps(precision.record(), sort_keys=True, separators=(",", ":")).encode()),
            "entry_function": name + "_region_run",
            "dispatch_function": name + "_region_dispatch",
            "runtime_inputs": runtime_inputs,
            "parameters": [item.record() for item in static],
            **(
                {"staged_program_sha256": staged.program_sha256}
                if staged is not None
                else {
                    "semantic_source_request_digest": semantic_source.digest(),
                    "precision_plan": precision.record(),
                }
            ),
            "quality_status": "unqualified",
        }
        # Source conversions are shared Merlin code. This wrapper only binds
        # their scratch and encoded arguments to the checked target region.
        private_headers["finite_float_boundary.h"] = (runtime_dir() / "c/finite_float_boundary.h").read_text()
        conversions, encode_calls, source_checks, encoded_pointers = [], [], [], []
        source_bytes = 0
        for index, conversion in enumerate(runtime_conversions):
            encoder = f"{name}_encode_source_{index}"
            conversions.append(render_f32_operand_c(conversion, encoder))
            elements = conversion.source_type.shape[0] * conversion.source_type.shape[1]
            encoded_pointers.append(f"ctx->workspace+{source_bytes}u")
            encode_calls.append(
                f"  if ({encoder}(inputs[{index}],ctx->workspace+{source_bytes}u) != MERLIN_F32_U8_OK)\n"
                f"    return {upper}_INPUT_DOMAIN;"
            )
            source_checks.append(f"""  if (!{name}_valid_range(inputs[{index}],{elements * 4}u) ||
      {name}_overlaps(inputs[{index}],{elements * 4}u,ctx->workspace,{upper}_SOURCE_WORKSPACE_BYTES) ||
      {name}_overlaps(inputs[{index}],{elements * 4}u,region->workspace,{workspace}u) ||
      {name}_overlaps(inputs[{index}],{elements * 4}u,out,{m * n * 4}u)) return {upper}_INVALID;""")
            source_bytes += elements
        source_workspace = source_bytes + m * n * 2
        encoded_inputs = (
            f"const uint8_t *encoded[{count}] = {{" + ",".join(encoded_pointers) + "};" if count else "(void)inputs;"
        )
        source_typed = (
            f"const float *typed[{count}] = {{" + ",".join(f"inputs[{i}]" for i in range(count)) + "};"
            if count
            else "(void)inputs;"
        )
        source_argument = "encoded" if count else "0"
        conversion_source = """#include <float.h>
_Static_assert(sizeof(float) == 4 && FLT_RADIX == 2 && FLT_MANT_DIG == 24 && FLT_MAX_EXP == 128,
               "source conversion requires binary32 float");
""" + "\n".join(conversions)
        decode_bits = "((uint32_t)result[2*i] | ((uint32_t)result[2*i+1]<<8))<<16"
        if precision.output.spec.post_decode_scale_exponent:
            from merlin.compile.c_conversions import render_bf16_decode_c

            decoder = name + "_decode_source_result"
            conversion_source += render_bf16_decode_c(precision.output.spec, decoder)
            decode_bits = f"{decoder}((uint16_t)result[2*i] | ((uint16_t)result[2*i+1]<<8))"
        region_declarations += f"""
#define {upper}_SOURCE_INPUT_COUNT {count}u
#define {upper}_SOURCE_WORKSPACE_BYTES {source_workspace}u
typedef struct {{ {name}_context *region; uint8_t *workspace; }} {name}_source_context;
/* Bind separate conversion scratch after binding the encoded region context. */
{name}_status {name}_source_bind({name}_source_context *context, {name}_context *region,
                                void *workspace, size_t workspace_bytes);
/* Complete dense f32 region operands in source order; f32 M-by-N output.
 * Parameters are embedded. Both workspaces must survive a timed-out launch. */
{name}_status {name}_source_run({name}_source_context *context, const float *const *inputs,
                               float *output, atlas_host_result *device_result);
int {name}_source_dispatch(void *context, const void *const *inputs, void *const *outputs);
"""
        region_source += f"""
{name}_status {name}_source_bind({name}_source_context *ctx, {name}_context *region,
                                void *scratch, size_t bytes) {{
  if (!ctx || !region || !region->host || region->host->active || !region->workspace ||
      bytes < {upper}_SOURCE_WORKSPACE_BYTES ||
      !{name}_valid_range(scratch,{upper}_SOURCE_WORKSPACE_BYTES) ||
      {name}_overlaps(scratch,{upper}_SOURCE_WORKSPACE_BYTES,region->workspace,{workspace}u) ||
      {name}_overlaps(scratch,{upper}_SOURCE_WORKSPACE_BYTES,region,sizeof(*region)) ||
      {name}_overlaps(scratch,{upper}_SOURCE_WORKSPACE_BYTES,region->host,sizeof(*region->host)) ||
      {name}_overlaps(scratch,{upper}_SOURCE_WORKSPACE_BYTES,ctx,sizeof(*ctx))) return {upper}_INVALID;
  *ctx = ({name}_source_context){{region,scratch}};
  return {upper}_OK;
}}
{name}_status {name}_source_run({name}_source_context *ctx, const float *const *inputs,
                               float *out, atlas_host_result *device) {{
  if (device) *device = ATLAS_HOST_INVALID;
  if (!ctx || !ctx->region || !ctx->region->host || !ctx->workspace) return {upper}_INVALID;
  {name}_context *region = ctx->region;
  if (region->host->active) {{ if (device) *device = ATLAS_HOST_BUSY; return {upper}_DEVICE_ERROR; }}
  {input_check}
  if (!{name}_valid_range(out,{m * n * 4}u) ||
      {name}_overlaps(out,{m * n * 4}u,ctx->workspace,{upper}_SOURCE_WORKSPACE_BYTES) ||
      {name}_overlaps(out,{m * n * 4}u,region->workspace,{workspace}u) ||
      {name}_overlaps(out,{m * n * 4}u,region,sizeof(*region)) ||
      {name}_overlaps(out,{m * n * 4}u,region->host,sizeof(*region->host)) ||
      {name}_overlaps(out,{m * n * 4}u,ctx,sizeof(*ctx))) return {upper}_INVALID;
{chr(10).join(source_checks)}
{chr(10).join(encode_calls)}
  {encoded_inputs}
  uint8_t *result = ctx->workspace+{source_bytes}u;
  {name}_status status = {name}_region_run(region,{source_argument},result,device);
  if (status != {upper}_OK) return status;
  for (size_t i = 0; i < {m * n}u; ++i) {{
    uint32_t bits = {decode_bits};
    __builtin_memcpy(out+i,&bits,4u);
  }}
  return {upper}_OK;
}}
int {name}_source_dispatch(void *ctx, const void *const *inputs, void *const *outputs) {{
  if ({dispatch_check}) return {upper}_INVALID;
  {source_typed}
  return (int){name}_source_run(ctx,{dispatch_inputs},outputs[0],0);
}}
"""
        region["source_interface"] = {
            "entry_function": name + "_source_run",
            "dispatch_function": name + "_source_dispatch",
            "input_dtype": "f32",
            "output_dtype": "f32",
            "workspace_bytes": source_workspace,
            "runtime_inputs": runtime_inputs,
            "precision_conversion": True,
            **(
                {"result_conversion": precision.output.record()}
                if precision.output.spec.post_decode_scale_exponent
                else {}
            ),
        }
    return SourceWrappers(region, region_declarations, region_source, private_headers, conversion_source)
