"""Export checked seeded reductions through the shared Atlas C launch interface."""

from __future__ import annotations

import hashlib
import json
import os
from math import prod
from pathlib import Path

from merlin.compile.c_conversions import render_f32_array_c
from merlin.compile.c_reduction import render_reduction_c
from merlin.semantic_compiler.target_binding import verify_native_publication

from .binding import (
    DRAM_BASE,
    NATIVE_CHECKS,
    NATIVE_PIPELINES,
    RTL_REVISION,
    TILE_BYTES,
    _run,
    check_io_memory_manifest,
    dram_slots,
)
from .reduction import (
    NONNEGATIVE_NORMAL_REDUCTION_POLICY,
    REDUCTION_POLICY,
    body_request,
    input_spec,
    prepare_plan,
    prepare_source,
    selected_body_descriptors,
    virtual_program,
)
from .tile_io_c import render_tile_io_package


def prepare_c_package(source, compiled, name, *, target_source, runtime_linkage="separate",
                      reduction_policy=REDUCTION_POLICY):
    precision = prepare_source(source, reduction_policy=reduction_policy)
    request = precision.request
    plan = prepare_plan(request)
    manifest = json.loads((compiled / "manifest.json").read_text())
    check_io_memory_manifest(manifest)
    verify_native_publication(
        compiled,
        manifest,
        engine="merlin_native",
        request_digest=request.digest(),
        target_identity=request.target_identity,
    )
    body = json.loads((compiled / "body/manifest.json").read_text())
    verify_native_publication(
        compiled / "body",
        body,
        engine="merlin_native",
        request_digest=body_request(plan).digest(),
        target_identity=request.target_identity,
    )
    inventory = hashlib.sha256((target_source / "docs/selected-variant-inventory.json").read_bytes()).hexdigest()
    if (
        manifest.get("schema") != "atlas.native_reduction_compilation.v1"
        or manifest.get("body_check_fingerprint") != body.get("native_check_fingerprint")
        or manifest.get("target_inventory_sha256") != inventory
        or body.get("target_inventory_sha256") != inventory
        or manifest.get("selected_body_descriptors") != selected_body_descriptors(plan, compiled / "body")
    ):
        raise ValueError("reduction body, inventory or native selection identity differs")
    execution = json.loads((compiled / "execution_plan.json").read_text())
    addresses, outputs = (
        execution.get("input_addresses"),
        execution.get("output_addresses"),
    )
    if not isinstance(addresses, list) or len(addresses) != 2 or not isinstance(outputs, list) or len(outputs) != 1:
        raise ValueError("reduction needs data, seed and output addresses")
    ranges = []
    for address, size in zip([*addresses, *outputs], plan.packed_sizes, strict=True):
        if (
            type(address) is not int
            or address < DRAM_BASE
            or (address - DRAM_BASE) % TILE_BYTES
            or address + size > DRAM_BASE + dram_slots() * TILE_BYTES
        ):
            raise ValueError("reduction address is unaligned or outside the selected window")
        ranges.append((address, address + size))
    if any(a < d and c < b for i, (a, b) in enumerate(ranges) for c, d in ranges[i + 1 :]):
        raise ValueError("reduction physical buffers overlap")
    expected = {
        "schema": "atlas.reduction_execution_plan.v1",
        "input_addresses": addresses,
        "input_buffer_bytes": list(plan.packed_sizes[:2]),
        "output_addresses": outputs,
        "output_bytes": plan.packed_sizes[2],
        "output_physical_layouts": ["row_column_zero_bf16_pair_tiles"],
        "tiling": plan.record(),
    }
    virtual = virtual_program(plan, addresses, outputs[0])
    if execution != expected or (compiled / "atlas-virtual.mlir").read_text() != virtual:
        raise ValueError("reduction program or ABI differs from source replay")
    pipeline = manifest.get("lowering_pipeline")
    if pipeline not in NATIVE_PIPELINES:
        raise ValueError("reduction has an unknown lowering pipeline")
    tools = Path(os.environ.get("ATLAS_OOT_BIN_DIR", target_source / "build/bin"))
    machine = _run([str(tools / "atlas-opt"), *NATIVE_PIPELINES[pipeline], "-"], input_text=virtual)
    _run(
        [str(tools / "atlas-opt"), *NATIVE_CHECKS, "--verify-atlas-timing", "-"],
        input_text=machine,
    )
    replay = json.loads(_run([str(tools / "atlas-emit"), "--program-json", "-"], input_text=machine))
    binary = (compiled / "program.bin").read_bytes()
    if replay.get("selected_rtl_revision") != RTL_REVISION or binary != b"".join(
        row["word_u32"].to_bytes(4, "little") for row in replay["instructions"]
    ):
        raise ValueError("reduction binary differs from checked source lowering")
    from .restoration_budget import instruction_bound
    from .limits import functional_instruction_limit
    from types import SimpleNamespace
    saved_bound = json.loads((compiled / "reduction-instruction-bound.json").read_text())
    cap = functional_instruction_limit(SimpleNamespace(functional_instruction_limit=saved_bound["max_instructions"]))
    bound = instruction_bound(virtual, machine, replay, max_instructions=cap)
    if not bound["admitted"] or saved_bound != bound:
        raise ValueError("reduction instruction bound differs or exceeds default")
    nonnegative = reduction_policy == NONNEGATIVE_NORMAL_REDUCTION_POLICY
    helpers = (
        render_reduction_c(plan, name)
        + "\n"
        + "\n".join(
            (render_nonnegative_encoder_c(prod(node.type.shape), name + f"_encode_{i}") if nonnegative else
             render_f32_array_c(input_spec(), prod(node.type.shape), name + f"_encode_{i}"))
            for i, node in enumerate(request.nodes[:2])
        )
    )
    if nonnegative:
        helpers += "\n" + render_nonnegative_guard_c(plan, name)
    record, files = render_tile_io_package(
        request,
        binary,
        execution,
        name,
        target_source=target_source,
        runtime_linkage=runtime_linkage,
        packed_sizes=plan.packed_sizes,
        tile_bytes=plan.data_plan().tile_bytes,
        helpers=helpers,
        input_guard=f"{name}_domain(operand_0,operand_1)" if nonnegative else None,
    )
    return {
        **record,
        "precision_plan": precision.record(),
        "input_node_ids": list(source.nodes[-1].inputs),
        "output_node_ids": list(source.outputs),
        "native_manifest_sha256": hashlib.sha256((compiled / "manifest.json").read_bytes()).hexdigest(),
    }, files


def render_nonnegative_guard_c(plan, prefix, *, squared=False):
    """Replay positive FP32 tree additions and BF16 rounding before device I/O.

    Integer guard/sticky arithmetic is independent of the host floating-point
    environment. Every tree node, BF16 partial, and seeded add must stay finite.
    With ``squared`` the data codes are a square's admitted input, and each
    reduction operand is replayed as the selected BF16 nearest-even square.
    """
    shape = plan.source.nodes[0].type.shape
    rows, columns = plan.matrix_shape
    coordinates = {}
    for axes, index in ((plan.kept_axes, "row"), (plan.axes, "col")):
        for i, axis in enumerate(axes):
            stride = prod(shape[a] for a in axes[i + 1:])
            coordinates[axis] = f"(({index}/{stride}u)%{shape[axis]}u)"
    at = "0"
    for axis, extent in enumerate(shape):
        at = f"(({at})*{extent}u+{coordinates[axis]})"
    read = f"{prefix}_code(data,{at})"
    square = ""
    if squared:
        read = f"{prefix}_square({read})"
        # Admitted codes are zero or have a normal BF16 square.
        square = f"""static uint32_t {prefix}_square(uint32_t code) {{
  if (!(code&0x7fffu)) return 0;
  uint32_t mantissa=(code&0x7fu)|0x80u, product=mantissa*mantissa;
  uint32_t exponent=2u*((code>>7)&0xffu)-127u, shift=7u;
  if (product>=0x8000u) {{ shift=8u; ++exponent; }}
  uint32_t rounded=product>>shift, rest=product&((1u<<shift)-1u), half=1u<<(shift-1u);
  rounded+=rest>half || (rest==half && (rounded&1u));
  if (rounded==0x100u) {{ rounded=0x80u; ++exponent; }}
  return (exponent<<7)|(rounded&0x7fu);
}}
"""
    return f"""
#include <stdint.h>
#include <stddef.h>
static uint32_t {prefix}_code(const uint8_t *p,size_t i) {{
  return (uint32_t)p[2*i]|((uint32_t)p[2*i+1]<<8);
}}
{square}static uint32_t {prefix}_jam(uint32_t value,unsigned shift) {{
  if (!shift) return value;
  if (shift>=32) return value!=0;
  return (value>>shift)|((value&((1u<<shift)-1u))!=0);
}}
static uint32_t {prefix}_positive_add(uint32_t a,uint32_t b) {{
  if (!a) return b;
  if (!b) return a;
  if (a<b) {{ uint32_t tmp=a; a=b; b=tmp; }}
  unsigned exponent=a>>23, difference=exponent-(b>>23);
  uint32_t sum=(((a&0x7fffffu)|0x800000u)<<3)
      +{prefix}_jam(((b&0x7fffffu)|0x800000u)<<3,difference);
  if (sum>=0x8000000u) {{ sum={prefix}_jam(sum,1); ++exponent; }}
  uint32_t rounded=(sum+3u+((sum>>3)&1u))>>3;
  if (rounded==0x1000000u) {{ rounded>>=1; ++exponent; }}
  if (exponent>=255) return 0x7f800000u;
  return (exponent<<23)|(rounded&0x7fffffu);
}}
static int {prefix}_domain(const uint8_t *data,const uint8_t *seed) {{
  for (size_t row=0;row<{rows}u;++row) {{
    uint32_t accumulated={prefix}_code(seed,row)<<16;
    for (size_t start=0;start<{columns}u;start+=32u) {{
      uint32_t values[32];
      for (size_t lane=0;lane<32u;++lane) {{
        size_t col=start+lane;
        values[lane]=col<{columns}u ? {read}<<16 : 0;
      }}
      for (size_t width=32u;width>1u;width/=2u)
        for (size_t lane=0;lane<width/2u;++lane) {{
          values[lane]={prefix}_positive_add(values[2*lane],values[2*lane+1]);
          if (values[lane]>=0x7f800000u) return 1;
        }}
      uint32_t partial=(values[0]+0x7fffu+((values[0]>>16)&1u))>>16;
      if (partial>0x7f7fu) return 1;
      accumulated={prefix}_positive_add(accumulated,partial<<16);
      if (accumulated>=0x7f800000u) return 1;
      accumulated&=0xffff0000u;
    }}
  }}
  return 0;
}}
"""


def render_nonnegative_encoder_c(count, prefix):
    """Encode RNE directly and reject overflow, subnormals, and negative codes."""
    return f"""
#include <float.h>
#include <stddef.h>
#include <stdint.h>
_Static_assert(sizeof(float)==4 && FLT_RADIX==2 && FLT_MANT_DIG==24 && FLT_MAX_EXP==128,
               "source conversion requires binary32 float");
_Static_assert(SIZE_MAX/4 >= {count}ULL, "source tensor exceeds host size_t");
static int {prefix}(const float *source,uint8_t *output) {{
  if (!source || !output) return 1;
  for (unsigned pass=0;pass<2;++pass) for (size_t i=0;i<{count}u;++i) {{
    uint32_t bits; __builtin_memcpy(&bits,source+i,4u);
    uint32_t magnitude=bits&0x7fffffffu;
    if (magnitude>=0x7f800000u) return 1;
    uint32_t code=((magnitude+0x7fffu+((magnitude>>16)&1u))>>16)|((bits>>16)&0x8000u);
    if (code && (code<0x80u || code>0x7f7fu)) return 2;
    if (pass) {{ output[2*i]=(uint8_t)code; output[2*i+1]=(uint8_t)(code>>8); }}
  }}
  return 0;
}}
"""
