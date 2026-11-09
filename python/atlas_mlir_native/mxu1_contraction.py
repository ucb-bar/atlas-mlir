"""Source-bound dense MXU1 model47 contractions with explicit operand scales.

The native K32 body is a typed semantic request selected by Merlin. This module
contains no assembly, physical register choices, schedules, or model execution.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from math import prod
from pathlib import Path
import re

from merlin.semantic_compiler.model import IndexMap, KernelRequest, SemanticNode, TensorType
from merlin.semantic_compiler.precision_values import FloatBoundarySpec, contraction_restore_exponent
from merlin.semantic_compiler.target_binding import verify_native_publication

from .binding import (
    AtlasNativeBinding, DRAM_BASE, MXU1_CAP448_INPUT_DOMAIN, MXU1_CAP448_POLICY,
    TARGET_IDENTITY, TILE_BYTES, _mxu0_maps, _mxu0_accumulate_maps,
)

POLICY = "f32-dense-batched-mxu1-model47-cap448-k32-ordered-bf16-v1"
ENCODE_POLICY = "f32-to-e4m3-cap448-rne-significand-then-flush-positive-zero-v1"


def _digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def operand_spec(exponent):
    return FloatBoundarySpec(
        ENCODE_POLICY, "fp8_e4m3", "encode", max_normal_exponent_code=15,
        flush_below_normal=True, admitted_code_ranges=((0, 0), (8, 126), (136, 254)),
        pre_encode_scale_exponent=exponent, canonicalize_negative_zero=True,
        underflow_rounding="round-normal-significand-then-flush",
    )


@dataclass(frozen=True)
class MXU1ContractionPlan:
    source: KernelRequest
    body: KernelRequest
    operand_ids: tuple[str, str]
    input_ids: tuple[str, ...]
    output_id: str
    batch_shape: tuple[int, ...]
    m: int
    n: int
    k: int
    operand_encode_exponents: tuple[int, int]
    provenance_sha256: str

    @property
    def batch_count(self):
        return prod(self.batch_shape)

    @property
    def operand_specs(self):
        return tuple(operand_spec(exponent) for exponent in self.operand_encode_exponents)

    @property
    def restore_exponent(self):
        return contraction_restore_exponent(*self.operand_encode_exponents)

    @property
    def launches_per_call(self):
        return self.batch_count * ((self.m+31)//32) * ((self.n+31)//32)

    def record(self):
        return dict(
            schema="atlas.source_mxu1_contraction.v1", policy=POLICY,
            source_request_digest=self.source.digest(), body_request_digest=self.body.digest(),
            provenance_sha256=self.provenance_sha256, input_node_ids=list(self.input_ids),
            operand_node_ids=list(self.operand_ids), output_node_ids=[self.output_id],
            batch_shape=list(self.batch_shape), batch_count=self.batch_count, shape=[self.m, self.n, self.k],
            operand_encode_exponents=list(self.operand_encode_exponents),
            operand_conversions=[spec.record() for spec in self.operand_specs],
            rhs_transport="source [...,K,N] to hardware W[N,K], unchanged FP8 codes",
            contraction="model47 anchor BF16 reset then explicit ordered 32-K seeded continuation",
            restore_exponent=self.restore_exponent,
            output_restore="exact BF16 power-of-two exponent adjustment; reject non-normal nonzero result",
            output_conversion="canonical f32 exact expansion of restored BF16",
            tails="positive-zero inactive operand rows and columns; logical output scatter only",
            launches_per_call=self.launches_per_call,
            qualification="diagnostic precision change; source fidelity and device execution required",
        )


def prepare_source(source, policy):
    """Prove a closed canonical source contraction before choosing its policy."""
    if (not isinstance(policy, dict) or set(policy) != {"policy", "source_request_digest", "provenance_sha256", "operand_encode_exponents"}
            or policy["policy"] != POLICY or policy["source_request_digest"] != source.digest()
            or not isinstance(policy["provenance_sha256"], str) or re.fullmatch(r"[0-9a-f]{64}", policy["provenance_sha256"]) is None
            or not isinstance(policy["operand_encode_exponents"], (list, tuple))
            or len(policy["operand_encode_exponents"]) != 2
            or any(type(value) is not int or not -32 <= value <= 32 for value in policy["operand_encode_exponents"])):
        raise ValueError("MXU1 contraction needs an exact source-bound policy and provenance digest")
    pure = [node for node in source.nodes if node.effect == "pure"]
    if (len(pure) != 1 or source.target_identity != TARGET_IDENTITY or source.lowering_policy != "diagnostic"
            or source.physical_extents or source.boundary_layouts or source.output_layouts):
        raise ValueError("MXU1 source needs one plain selected diagnostic contraction")
    result = pure[0]
    arithmetic = dict(result.attrs)
    order = arithmetic.pop("scalar_add_order", None)
    if (result.op != "source_f32_matmul_unqualified" or len(result.inputs) != 3 or source.outputs != (result.id,)
            or order not in (None, "accumulator-product", "product-accumulator")
            or arithmetic != {"reduction_order": "unresolved", "numeric_semantics": "source-f32-mul-add-unqualified-v1"}):
        raise ValueError("MXU1 source contraction arithmetic is not the declared unresolved multiply/add")
    lhs, rhs, seed = (source.node(key) for key in result.inputs)
    rank = len(result.type.shape)
    if (rank < 2 or lhs.id == rhs.id or {node.id for node in source.nodes} != {lhs.id, rhs.id, seed.id, result.id}
            or any(node.type.dtype != "f32" or node.type.numerical_policy != "raw-bit-pattern" or len(node.type.shape) != rank
                   for node in (lhs, rhs, seed, result))
            or any(node.effect != "input" or node.op != "input" or node.inputs or node.attrs or node.index_maps for node in (lhs, rhs))
            or set(source.input_storages) != {(lhs.id, "external"), (rhs.id, "external")}):
        raise ValueError("MXU1 source needs two distinct dense f32 boundary operands and complete source accounting")
    batch = result.type.shape[:-2]
    m, k = lhs.type.shape[-2:]
    n = rhs.type.shape[-1]
    if (lhs.type.shape != (*batch, m, k) or rhs.type.shape != (*batch, k, n)
            or result.type.shape != (*batch, m, n) or seed.type != result.type
            or k % 32 or not 32 <= k <= 256):
        raise ValueError("MXU1 contraction requires identical dense batches and K multiple32 in [32,256]")
    loops = rank+1
    axes = tuple(tuple(int(i == j) for j in range(loops)) for i in range(loops))
    prefix = axes[:rank-2]
    maps = tuple(IndexMap(loops, (*prefix, *rows), (0,)*rank)
                 for rows in ((axes[-3], axes[-1]), (axes[-1], axes[-2]), (axes[-3], axes[-2])))
    if result.index_maps != (*maps, maps[-1]):
        raise ValueError("MXU1 source index maps mix batches or change canonical matrix indexing")
    initialization = next((binding for binding in source.constants if binding.node_id == seed.id), None)
    if (seed.op != "constant" or seed.effect != "constant" or seed.inputs or seed.attrs or seed.index_maps
            or initialization is None or initialization.encoding != "f32-le-raw"
            or any(bytes.fromhex(initialization.data_hex)) or len(source.constants) != 1):
        raise ValueError("MXU1 source requires an exact full positive-zero initializer")
    fp8 = TensorType((32, 32), "fp8_e4m3", "raw-bit-pattern")
    bf16 = TensorType((32, 32), "bf16", MXU1_CAP448_POLICY)
    inputs = tuple(SemanticNode(f"{role}{tile}", "input", (), fp8,
                               (("input_domain", MXU1_CAP448_INPUT_DOMAIN),), "input")
                   for tile in range(k//32) for role in ("a", "w"))
    computations = tuple(SemanticNode(f"s{tile}", "matmul" if tile == 0 else "matmul_accumulate",
        (f"a{tile}", f"w{tile}") + ((f"s{tile-1}",) if tile else ()), bf16,
        index_maps=_mxu0_maps() if tile == 0 else _mxu0_accumulate_maps()) for tile in range(k//32))
    body = KernelRequest((*inputs, *computations), (computations[-1].id,), ("external",),
        tuple((node.id, "external") for node in inputs), TARGET_IDENTITY, "diagnostic",
        _digest([source.digest(), policy]))
    return MXU1ContractionPlan(source, body, (lhs.id, rhs.id), tuple(key for key, _ in source.input_storages),
                               result.id, batch, m, n, k, tuple(policy["operand_encode_exponents"]), policy["provenance_sha256"])


def compile_source(snapshot, source, policy, *, target_source, destination, limits):
    plan = prepare_source(source, policy)
    abi = AtlasNativeBinding.plan_boundary_abi(plan.body)
    manifest = AtlasNativeBinding.compile(snapshot, plan.body, fixed_inputs=abi["fixed_inputs"],
        fixed_outputs=tuple(abi["fixed_outputs"]), target_source=target_source, destination=destination,
        limits=limits, allow_staging=False)
    for name, data in (("semantic-source.json", source.record()), ("mxu1-contraction-plan.json", plan.record())):
        payload = (json.dumps(data, sort_keys=True, indent=2)+"\n").encode()
        (destination/name).write_bytes(payload)
        manifest["stage_artifacts"][name] = sha256(payload).hexdigest()
    manifest["source_contraction_request_digest"] = source.digest()
    manifest["source_contraction_plan_sha256"] = manifest["stage_artifacts"]["mxu1-contraction-plan.json"]
    (destination/"manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2)+"\n")
    return manifest


def prepare_c_package(source, compiled, name, *, policy, target_source, runtime_linkage="separate"):
    """Export source conversion and dense batch/tail iteration around one image."""
    from merlin.common.paths import runtime_dir
    from merlin.compile.c_sources import c_source_pair
    from merlin.compile.c_conversions import render_f32_array_c
    from .c_runtime import program_array, runtime_headers, runtime_members
    plan = prepare_source(source, policy)
    compiled = Path(compiled)
    if compiled.is_symlink() or not compiled.is_dir():
        raise ValueError("MXU1 source compilation must be a real directory")
    def member(name):
        path = compiled/name
        if path.is_symlink() or not path.is_file():
            raise ValueError("missing or linked MXU1 source artifact")
        return path.read_bytes()
    manifest_bytes = member("manifest.json")
    manifest = json.loads(manifest_bytes)
    verify_native_publication(compiled, manifest, engine="merlin_native", request_digest=plan.body.digest(), target_identity=TARGET_IDENTITY)
    if (json.loads(member("mxu1-contraction-plan.json")) != plan.record()
            or KernelRequest.from_record(json.loads(member("semantic-source.json"))).digest() != source.digest()
            or KernelRequest.from_record(json.loads(member("source-request.json"))).digest() != plan.body.digest()
            or manifest.get("source_contraction_request_digest") != source.digest()):
        raise ValueError("MXU1 source or precision provenance differs from its checked publication")
    execution = json.loads(member("execution_plan.json"))
    abi = AtlasNativeBinding.plan_boundary_abi(plan.body)
    addresses = [DRAM_BASE+abi["fixed_inputs"][node.id]*TILE_BYTES for node in plan.body.nodes if node.effect == "input"]
    output_address = DRAM_BASE+abi["fixed_outputs"][0]*TILE_BYTES
    if (execution["input_addresses"] != addresses or execution["output_addresses"] != [output_address]
            or execution.get("operation") != "mxu1_cap448_seeded_ordered" or execution["output_bytes"] != TILE_BYTES):
        raise ValueError("MXU1 body ABI differs from its regenerated exact native binding")
    binary = member("program.bin")
    program = json.loads(member("physical-program.json"))
    if binary != b"".join(row["word_u32"].to_bytes(4, "little") for row in program["instructions"]):
        raise ValueError("MXU1 body binary differs from checked emitted words")
    runtime = target_source/"runtime"
    public = runtime_headers(runtime)
    host_source = (runtime/"atlas_host.c").read_text()
    host_members, runtime_files, runtime_record = runtime_members(public, host_source, runtime_linkage)
    if sha256((target_source/"docs/selected-variant-inventory.json").read_bytes()).hexdigest() != manifest["target_inventory_sha256"]:
        raise ValueError("MXU1 runtime source does not match the compiled inventory")
    m, n, k, batches = plan.m, plan.n, plan.k, plan.batch_count
    tiles = k//32
    raw_workspace = 2*tiles*1024+2048
    a_elements, b_elements, out_elements = batches*m*k, batches*k*n, batches*m*n
    staged_offset = (a_elements+b_elements+3)//4*4
    source_workspace = staged_offset+out_elements*4
    upper = name.upper()
    declarations = f"""
#define {upper}_WORKSPACE_BYTES {raw_workspace}u
#define {upper}_SOURCE_WORKSPACE_BYTES {source_workspace}u
#define {upper}_SOURCE_INPUT_COUNT 2u
#define {upper}_OUTPUT_ELEMENTS {out_elements}u
typedef struct {{ atlas_host *host; atlas_tile_memory_io io; uint8_t *workspace; size_t max_polls;
  atlas_host_buffer buffers[{2*tiles+1}]; }} {name}_context;
typedef struct {{ {name}_context *region; uint8_t *workspace; }} {name}_source_context;
int {name}_bind({name}_context *,atlas_host *,atlas_tile_memory_io,void *,size_t,size_t);
int {name}_source_bind({name}_source_context *,{name}_context *,void *,size_t);
int {name}_source_run({name}_source_context *,const float *const *,float *,atlas_host_result *);
int {name}_source_dispatch(void *,const void *const *,void *const *);
const uint32_t *{name}_program(size_t *);
"""
    encoders = "\n".join(render_f32_array_c(spec, count, f"{name}_encode_{i}")
                          for i, (spec, count) in enumerate(zip(plan.operand_specs, (a_elements, b_elements), strict=True)))
    buffers = ",".join(f"{{{address}u,1024u,ATLAS_HOST_DEVICE_READ}}" for address in addresses)
    buffers += f",{{{output_address}u,2048u,ATLAS_HOST_DEVICE_WRITE}}"
    input_indices = [plan.input_ids.index(key) for key in plan.operand_ids]
    writes = "\n".join(f"      if (region->io.write(region->io.context,{address}u,region->workspace+{index*1024}u,1024u)) return 3;"
                         for index, address in enumerate(addresses))
    implementation = program_array(name+"_words", binary) + encoders + f"""
static int {name}_range(const void *p,size_t n) {{ return p && n && (uintptr_t)p<=UINTPTR_MAX-n; }}
static int {name}_overlap(const void *a,size_t an,const void *b,size_t bn) {{
  return (uintptr_t)a<(uintptr_t)b+bn && (uintptr_t)b<(uintptr_t)a+an;
}}
const uint32_t *{name}_program(size_t *words) {{ if (words) *words={len(binary)//4}u; return {name}_words; }}
int {name}_bind({name}_context *ctx,atlas_host *host,atlas_tile_memory_io io,void *scratch,size_t bytes,size_t polls) {{
  if (!{name}_range(ctx,sizeof(*ctx)) || !{name}_range(host,sizeof(*host)) || host->active ||
      !host->mmio.read32 || !host->mmio.write32 || !io.write || !io.read || !polls ||
      bytes<{raw_workspace}u || !{name}_range(scratch,{raw_workspace}u) ||
      {name}_overlap(ctx,sizeof(*ctx),host,sizeof(*host)) ||
      {name}_overlap(scratch,{raw_workspace}u,ctx,sizeof(*ctx)) ||
      {name}_overlap(scratch,{raw_workspace}u,host,sizeof(*host))) return 1;
  *ctx=({name}_context){{host,io,scratch,polls,{{{buffers}}}}}; return 0;
}}
int {name}_source_bind({name}_source_context *ctx,{name}_context *region,void *scratch,size_t bytes) {{
  if (!{name}_range(ctx,sizeof(*ctx)) || !region || !region->host || region->host->active || !region->workspace ||
      bytes<{source_workspace}u || !{name}_range(scratch,{source_workspace}u) ||
      {name}_overlap(scratch,{source_workspace}u,ctx,sizeof(*ctx)) ||
      {name}_overlap(scratch,{source_workspace}u,region,sizeof(*region)) ||
      {name}_overlap(scratch,{source_workspace}u,region->host,sizeof(*region->host)) ||
      {name}_overlap(scratch,{source_workspace}u,region->workspace,{raw_workspace}u) ||
      {name}_overlap(ctx,sizeof(*ctx),region,sizeof(*region)) ||
      {name}_overlap(ctx,sizeof(*ctx),region->host,sizeof(*region->host)) ||
      {name}_overlap(ctx,sizeof(*ctx),region->workspace,{raw_workspace}u)) return 1;
  *ctx=({name}_source_context){{region,scratch}}; return 0;
}}
int {name}_source_run({name}_source_context *ctx,const float *const *inputs,float *out,atlas_host_result *device) {{
  if (device) *device=ATLAS_HOST_INVALID;
  if (!ctx || !ctx->region || !ctx->workspace || !ctx->region->host || !inputs || !out) return 1;
  {name}_context *region=ctx->region;
  if (!region->workspace || !region->io.write || !region->io.read || !region->max_polls) return 1;
  if (region->host->active) {{ if (device) *device=ATLAS_HOST_BUSY; return 4; }}
  const float *a=inputs[{input_indices[0]}],*b=inputs[{input_indices[1]}];
  if (!{name}_range(a,{a_elements*4}u) || !{name}_range(b,{b_elements*4}u) ||
      !{name}_range(out,{out_elements*4}u) || !{name}_range(inputs,2u*sizeof(*inputs))) return 1;
  const void *reserved[]={{ctx,region,region->host,ctx->workspace,region->workspace,inputs}};
  const size_t reserved_bytes[]={{sizeof(*ctx),sizeof(*region),sizeof(*region->host),{source_workspace}u,{raw_workspace}u,2u*sizeof(*inputs)}};
  for (size_t i=0;i<6u;++i) {{
    if (!{name}_range(reserved[i],reserved_bytes[i]) ||
        {name}_overlap(out,{out_elements*4}u,reserved[i],reserved_bytes[i])) return 1;
    if (i<5u && ({name}_overlap(a,{a_elements*4}u,reserved[i],reserved_bytes[i]) ||
                  {name}_overlap(b,{b_elements*4}u,reserved[i],reserved_bytes[i]))) return 1;
  }}
  if ({name}_overlap(out,{out_elements*4}u,a,{a_elements*4}u) ||
      {name}_overlap(out,{out_elements*4}u,b,{b_elements*4}u) ||
      {name}_overlap(inputs,2u*sizeof(*inputs),ctx->workspace,{source_workspace}u) ||
      {name}_overlap(inputs,2u*sizeof(*inputs),region->workspace,{raw_workspace}u)) return 1;
  uint8_t *encoded_a=ctx->workspace,*encoded_b=ctx->workspace+{a_elements}u;
  uint8_t *staged=ctx->workspace+{staged_offset}u,*tile_output=region->workspace+{2*tiles*1024}u;
  if ({name}_encode_0(a,encoded_a) || {name}_encode_1(b,encoded_b)) return 2;
  atlas_host_result status=atlas_host_ensure_loaded(region->host,{name}_words,{len(binary)//4}u);
  if (device) *device=status;
  if (status!=ATLAS_HOST_OK) return 4;
  for (size_t batch=0;batch<{batches}u;++batch)
    for (size_t first_m=0;first_m<{m}u;first_m+=32u)
      for (size_t first_n=0;first_n<{n}u;first_n+=32u) {{
        __builtin_memset(region->workspace,0,{2*tiles*1024}u);
        for (size_t kt=0;kt<{tiles}u;++kt) for (size_t row=0;row<32u;++row) for (size_t c=0;c<32u;++c) {{
          if (first_m+row<{m}u)
            region->workspace[2u*kt*1024u+row*32u+c]=encoded_a[(batch*{m}u+first_m+row)*{k}u+kt*32u+c];
          if (first_n+row<{n}u)
            region->workspace[(2u*kt+1u)*1024u+row*32u+c]=encoded_b[(batch*{k}u+kt*32u+c)*{n}u+first_n+row];
        }}
{writes}
        status=atlas_host_start(region->host,region->buffers,{2*tiles+1}u);
        if (status==ATLAS_HOST_OK) status=atlas_host_wait(region->host,region->max_polls);
        if (device) *device=status;
        if (status!=ATLAS_HOST_OK) return 4;
        if (region->io.read(region->io.context,{output_address}u,tile_output,2048u)) return 3;
        for (size_t row=0;row<32u;++row) for (size_t c=0;c<32u;++c) {{
          size_t at=(c/16u)*1024u+(row*16u+c%16u)*2u;
          uint32_t word=(uint32_t)tile_output[at]|((uint32_t)tile_output[at+1u]<<8);
          if (word && ((word&0x7f80u)==0u || (word&0x7f80u)==0x7f80u)) return 2;
          if (first_m+row<{m}u && first_n+c<{n}u) {{
            uint32_t bits=word<<16;
            if (word&0x7fffu) {{
              int exponent=(int)((word>>7)&255u)+({plan.restore_exponent});
              if (exponent<=0 || exponent>=255) return 2;
              bits=((word&0x807fu)|((uint32_t)exponent<<7))<<16;
            }}
            size_t index=(batch*{m}u+first_m+row)*{n}u+first_n+c;
            __builtin_memcpy(staged+index*4u,&bits,4u);
          }}
        }}
      }}
  __builtin_memcpy(out,staged,{out_elements*4}u); return 0;
}}
int {name}_source_dispatch(void *context,const void *const *inputs,void *const *outputs) {{
  if (!outputs || !inputs || !{name}_range(outputs,sizeof(*outputs)) ||
      {name}_overlap(outputs,sizeof(*outputs),outputs[0],{out_elements*4}u)) return 1;
  const float *typed[2]={{inputs[0],inputs[1]}};
  return {name}_source_run(context,typed,outputs[0],0);
}}
"""
    files = {key: value.encode() for key, value in c_source_pair(name, public_headers=public,
        private_headers={"finite_float_boundary.h": (runtime_dir()/"c/finite_float_boundary.h").read_text()},
        sources={**host_members, "conversion_and_dispatch.c": implementation}, declarations=declarations).items()}
    files.update(runtime_files)
    record = dict(schema="atlas.source_mxu1_c_package.v1", package_name=name, engine="merlin_native",
        source_request_digest=source.digest(), request_digest=plan.body.digest(), precision_plan=plan.record(),
        context_type=name+"_context", source_context_type=name+"_source_context", binding_function=name+"_bind",
        source_binding_function=name+"_source_bind", dispatch_function=name+"_source_dispatch",
        workspace_bytes=raw_workspace, source_workspace_bytes=source_workspace,
        input_node_ids=list(plan.input_ids), output_node_ids=[plan.output_id], launches_per_call=plan.launches_per_call,
        source_interface=dict(workspace_bytes=source_workspace, input_node_ids=list(plan.input_ids), output_node_ids=[plan.output_id],
                              dispatch_function=name+"_source_dispatch", failure_output="unchanged", completion="synchronous"),
        program_location="program.bin", native_manifest_sha256=sha256(manifest_bytes).hexdigest(),
        device_input_addresses=addresses, device_output_address=output_address, device_output_bytes=2048,
        device_arena_base=min(*addresses, output_address),
        device_arena_bytes=max(*(address+1024 for address in addresses), output_address+2048)-min(*addresses, output_address),
        **runtime_record)
    verify_native_publication(compiled, manifest, engine="merlin_native", request_digest=plan.body.digest(), target_identity=TARGET_IDENTITY)
    if member("manifest.json") != manifest_bytes:
        raise ValueError("MXU1 publication changed during C export")
    return record, files
