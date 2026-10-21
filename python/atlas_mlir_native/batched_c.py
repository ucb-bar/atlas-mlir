"""Bind shared batch iteration to the existing synchronous Atlas matrix call."""

from __future__ import annotations

from merlin.compile.batched_c import render_batched_contraction_c
from merlin.compile.c_sources import c_source_pair

from .looped_c import prepare_c_package


def prepare_batched_c_package(precision, compiled, name, *, target_source, operand_scale_exponents):
    record, files = prepare_c_package(
        compiled / "source-request.json",
        compiled,
        name,
        target_source=target_source,
        runtime_linkage="separate",
        semantic_source=precision.plan.matrix_source,
        semantic_operand_scale_exponents=operand_scale_exponents,
    )
    return wrap_batched_c_package(precision, name, record, files)


def wrap_batched_c_package(precision, name, record, files):
    """Reuse batch iteration around either a matrix image or a chunk controller."""
    source_interface = record["source_region"]["source_interface"]
    source = render_batched_contraction_c(precision.plan, name, name + "_source_dispatch")
    source += f"""
int {name}_batched_dispatch(void *context,const void *const *inputs,void *const *outputs) {{
  {name}_source_context *ctx=context;
  if (!ctx || !ctx->region || !ctx->region->host || !ctx->region->workspace || !ctx->workspace) return -1;
  const void *reserved[]={{ctx,ctx->region,ctx->region->host,ctx->workspace,ctx->region->workspace}};
  const size_t sizes[]={{sizeof(*ctx),sizeof(*ctx->region),sizeof(*ctx->region->host),
                         {source_interface["workspace_bytes"]}u,{record["workspace_bytes"]}u}};
  return {name}_batch_run(context,inputs,outputs,reserved,sizes,5);
}}
"""
    combined = c_source_pair(
        name + "_batch",
        public_headers={name + ".h": files[name + ".h"].decode()},
        private_headers={},
        sources={name + ".c": files[name + ".c"].decode(), "batch.c": source},
        declarations=f"int {name}_batched_dispatch(void *,const void *const *,void *const *);\n",
    )
    files = {key: value for key, value in files.items() if key not in {name + ".c", name + ".h"}}
    files.update({key: value.encode() for key, value in combined.items()})
    return {
        **record,
        "schema": "atlas.native_batched_c_package.v1",
        "batching": precision.plan.record(),
        "batch_source_request_digest": precision.plan.source.digest(),
        "launches_per_call": precision.plan.batch_count * record.get("launches_per_call", 1),
        "batch_header": name + "_batch.h",
        "batch_dispatch": name + "_batched_dispatch",
        "batch_interface": {
            "dispatch_function": name + "_batched_dispatch",
            "input_node_ids": list(precision.plan.input_ids),
            "output_node_ids": [precision.plan.result_id],
            "input_count": len(precision.plan.input_ids),
            "workspace_bytes": source_interface["workspace_bytes"],
            "failure_may_write_region_output": True,
            "application_outputs_staged_until_success": True,
        },
    }, files
