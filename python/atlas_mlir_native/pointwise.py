"""Explicit BF16 precision and compact loops for affine pointwise regions."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, replace
from pathlib import Path

from merlin.runtime.fp8_formats import float_format_params
from merlin.semantic_compiler.model import KernelRequest, SemanticNode, TensorType
from merlin.semantic_compiler.pointwise_tiling import PointwiseTiling
from merlin.semantic_compiler.precision_values import FloatBoundarySpec
from merlin.semantic_compiler.target_binding import NativeCompilationError

from .binding import (
    BF16_BINARY_ATTRS,
    BF16_BINARY_MODES,
    BF16_RELU_ATTRS,
    TARGET_IDENTITY,
    AtlasNativeBinding,
)

POINTWISE_POLICY = "f32-to-bf16-bounded-pointwise-rne-chop-v1"
BF16_POINTWISE_POLICY = "bf16-bounded-pointwise-rne-v1"
_BF16_OPS = {"bf16_mul": "mul_bf16_zero"}
_OPS = {
    "f32_add": "add_bf16_rne",
    "f32_sub": "sub_bf16_rne",
    "f32_mul": "mul_bf16_zero",
}
_UNARY_RULES = {"atlas.vpu_relu_bf16_tile": "relu"}


def input_spec(*, policy=POINTWISE_POLICY):
    # A product of two admitted nonzero BF16 inputs is normal and finite;
    # sums/differences also stay in the normal-or-zero FP32/BF16 domain.
    eb, mb, bias, _ = float_format_params("bf16")
    low = (bias + (1 - bias) // 2) << mb
    high = (bias + (((1 << eb) - 2 - bias) // 2)) << mb
    return FloatBoundarySpec(
        policy,
        "bf16",
        "encode",
        admitted_code_ranges=(
            (0, 0),
            (low, high),
            (0x8000, 0x8000),
            (0x8000 | low, 0x8000 | high),
        ),
    )


@dataclass(frozen=True)
class PointwisePrecision:
    source: KernelRequest
    request: KernelRequest

    @property
    def operand_specs(self):
        policy = BF16_POINTWISE_POLICY if self.source.nodes[-1].type.dtype == "bf16" else POINTWISE_POLICY
        return tuple(input_spec(policy=policy) for _ in self.request.nodes[:-1])

    def record(self):
        source_output, target_output = self.source.nodes[-1], self.request.nodes[-1]
        inputs = [source_output.inputs[target_output.inputs.index(node.id)] for node in self.request.nodes[:-1]]
        if self.source.nodes[-1].type.dtype == "bf16":
            return {
                "schema": "atlas.pointwise_precision.v2",
                "policy": BF16_POINTWISE_POLICY,
                "source_request_digest": self.source.digest(),
                "request_digest": self.request.digest(),
                "inputs": inputs,
                "input_code_domain": input_spec(policy=BF16_POINTWISE_POLICY).record(),
                "input_conversion": "bf16-bit-preserving; checked code domain",
                "output_conversion": "bf16-bit-preserving",
                "arithmetic": "selected-rtl-bf16-product-rne-v1",
                "qualification": "conditional_source_equivalence; bounded finite-normal-or-signed-zero inputs",
            }
        return {
            "schema": "atlas.pointwise_precision.v1",
            "source_request_digest": self.source.digest(),
            "request_digest": self.request.digest(),
            "inputs": inputs,
            "input_conversion": input_spec().record(),
            "output_conversion": "bf16-to-f32-exact",
            "qualification": "diagnostic_precision_change; original_model_quality_unmeasured",
        }


def prepare_source(source, *, allow_math=False, bf16_policy=None):
    from .application_precision import is_source_relu

    if bf16_policy is not None and bf16_policy != BF16_POINTWISE_POLICY:
        raise ValueError("BF16 pointwise precision policy differs")
    if (allow_math and len(source.nodes) == 3 and source.nodes[1].effect == "pure"
            and source.nodes[-1].op == "f32_mul"):
        from .sigmoid import prepare_silu_source

        return prepare_silu_source(source)
    if allow_math and source.nodes[-1].op in {
        "f32_div",
        "source_f32_rsqrt_unqualified",
    }:
        from .pointwise_math import prepare_source as prepare_math

        return prepare_math(source)

    if allow_math and source.nodes[-1].op == "source_f32_composite_unqualified":
        from .gelu import is_source_gelu
        from .gelu import prepare_source as prepare_gelu
        from .sigmoid import prepare_source as prepare_sigmoid

        return prepare_gelu(source) if is_source_gelu(source.nodes[-1]) else prepare_sigmoid(source)

    pure = [node for node in source.nodes if node.effect == "pure"]
    if (
        len(pure) != 1
        or source.lowering_policy != "diagnostic"
        or source.target_identity != TARGET_IDENTITY
        or source.physical_extents
        or source.boundary_layouts
        or source.output_layouts
    ):
        raise ValueError("pointwise precision needs one plain diagnostic source computation")
    output = pure[0]
    relu = is_source_relu(output)
    bf16 = output.type.dtype == "bf16"
    if bf16 and bf16_policy != BF16_POINTWISE_POLICY:
        raise ValueError("BF16 pointwise source needs its separately selected precision policy")
    operations = _BF16_OPS if bf16 else _OPS
    source_dtype = "bf16" if bf16 else "f32"
    if (
        (not relu and output.op not in operations)
        or len(output.inputs) != (1 if relu else 2)
        or source.outputs != (output.id,)
        or (not relu and dict(output.attrs) != {
            "numeric_semantics": f"ieee754-{source_dtype}-{output.op[len(source_dtype) + 1:]}-rne-v1"
        })
        or any(node.type.dtype != source_dtype or node.type.numerical_policy != "raw-bit-pattern"
               for node in source.nodes)
        or {node.id for node in source.nodes} != {*output.inputs, output.id}
    ):
        raise ValueError("pointwise source operation, attributes or accounting differ")
    op = "relu" if relu else operations[output.op]
    attrs = BF16_RELU_ATTRS if relu else BF16_BINARY_ATTRS[op]
    inputs, views, operands = [], [], []
    for key, mapping in zip(output.inputs, output.index_maps[:-1], strict=True):
        # Equal source values may share a prepared tile only when their index
        # maps agree. Distinct views retain distinct storage realizations.
        previous = next((i for i, view in enumerate(views) if view == (key, mapping)), None)
        if previous is None:
            previous = len(inputs)
            views.append((key, mapping))
            inputs.append(SemanticNode(f"arg{previous}", "input", (),
                                       TensorType(source.node(key).type.shape, "bf16", "raw-bit-pattern"),
                                       (attrs[0],), "input"))
        operands.append(inputs[previous].id)
    result = SemanticNode(
        "result",
        op,
        tuple(operands),
        TensorType(output.type.shape, "bf16", "raw-bit-pattern"),
        attrs,
        index_maps=output.index_maps,
    )
    request = KernelRequest(
        (*inputs, result),
        (result.id,),
        ("external",),
        tuple((node.id, "external") for node in inputs),
        TARGET_IDENTITY,
        "diagnostic",
        source.digest(),
    )
    prepare_plan(request)
    import numpy as np

    bindings = {item.node_id: item for item in source.constants}
    for key in output.inputs:
        node = source.node(key)
        if node.effect == "constant":
            binding = bindings[key]
            if binding.encoding != source_dtype + "-le-raw":
                raise ValueError("pointwise parameter bytes differ from the source dtype")
            data = bytes.fromhex(binding.data_hex)
            values = ((np.frombuffer(data, dtype="<u2").astype("<u4") << 16).view("<f4")
                      if bf16 else np.frombuffer(data, dtype="<f4"))
            input_spec(policy=BF16_POINTWISE_POLICY if bf16 else POINTWISE_POLICY).encode(values)
    return PointwisePrecision(source, request)


def _prepare_plan(request):
    if request.nodes[-1].op == "tanh_gelu_bf16_lut":
        from .gelu import prepare_plan as gelu_plan

        return gelu_plan(request)
    if request.nodes[-1].op in {"sigmoid_bf16_lut", "silu_bf16_lut"}:
        from .sigmoid import prepare_plan as sigmoid_plan

        return sigmoid_plan(request)
    if request.nodes[-1].op in {"divide_bf16_lut", "rsqrt_bf16_lut"}:
        from .pointwise_math import prepare_plan as math_plan

        return math_plan(request)
    if (
        request.target_identity != TARGET_IDENTITY
        or request.lowering_policy != "diagnostic"
        or request.nodes[-1].op not in {*_OPS.values(), "relu"}
        or len(request.nodes[-1].inputs) != (1 if request.nodes[-1].op == "relu" else 2)
        or len(request.nodes) != len(set(request.nodes[-1].inputs)) + 1
    ):
        raise NativeCompilationError(
            "unsupported_semantics",
            "pointwise loop needs an admitted BF16 operation",
        )
    *inputs, output = request.nodes
    attrs = BF16_RELU_ATTRS if output.op == "relu" else BF16_BINARY_ATTRS[output.op]
    if (
        output.attrs != attrs
        or output.type.dtype != "bf16"
        or output.type.numerical_policy != "raw-bit-pattern"
        or any(node.type.numerical_policy != "raw-bit-pattern" or node.attrs != (attrs[0],) for node in inputs)
        or any(storage != "external" for _, storage in request.input_storages)
        or request.output_storages != ("external",)
    ):
        raise NativeCompilationError("unsupported_semantics", "pointwise loop policies or storage differ")
    descriptor = next(item for item in AtlasNativeBinding.profile().descriptors if item.computation == output.op)
    shape = tuple(axis.minimum for axis in descriptor.output_axis_bounds)
    return PointwiseTiling(request, shape, 2, b"\0\0")


def prepare_plan(request):
    plan = _prepare_plan(request)
    from .binding import TILE_BYTES

    bank = next(item for item in AtlasNativeBinding.profile().banks if item.name == "external")
    if len(request.nodes) * plan.packed_bytes > bank.capacity * TILE_BYTES:
        # Every admitted body is lane-wise. Broadcast reads can reuse a tile;
        # invalid output-tail lanes are discarded by the checked assembler.
        compact = replace(plan, compact_broadcasts=True)
        if sum(compact.input_packed_bytes) < (len(request.nodes) - 1) * plan.packed_bytes:
            return compact
    return plan


def is_pointwise_request(request):
    if request.nodes[-1].op in {"divide_bf16_lut", "rsqrt_bf16_lut", "sigmoid_bf16_lut", "silu_bf16_lut", "tanh_gelu_bf16_lut"}:
        return True
    return (
        len(request.nodes) in (2, 3)
        and request.nodes[-1].op in {*_OPS.values(), "relu"}
        and (
            any(node.type.shape != (32, 32) for node in request.nodes)
            or any(
                mapping.coefficients != ((1, 0), (0, 1)) or mapping.offsets != (0, 0)
                for mapping in request.nodes[-1].index_maps
            )
        )
    )


def plan_abi(request):
    from merlin.semantic_compiler.buffer_allocation import Buffer, allocate_buffers

    from .binding import TILE_BYTES

    plan = prepare_plan(request)
    bank = next(item for item in AtlasNativeBinding.profile().banks if item.name == "external")
    sizes = (*plan.input_packed_bytes, plan.packed_bytes)
    buffers = tuple(Buffer(node.id, bank.name, (size + TILE_BYTES - 1) // TILE_BYTES, 0, 1)
                    for node, size in zip(request.nodes, sizes, strict=True))
    solved = allocate_buffers(buffers, (bank,))
    if solved.status != "sat":
        raise NativeCompilationError(
            "resource_limit",
            "pointwise packed buffers exceed the selected memory window",
        )
    addresses = dict(solved.addresses)
    return {
        "fixed_inputs": {node.id: addresses[node.id] for node in request.nodes[:-1]},
        "fixed_outputs": [addresses[request.outputs[0]]],
    }


def selected_body_descriptor(plan, directory):
    """Require a selected tile implementation before lifting it into a loop."""
    selected = json.loads((directory / "native-selection.json").read_text())
    if plan.source.nodes[-1].op in {"divide_bf16_lut", "rsqrt_bf16_lut", "sigmoid_bf16_lut", "silu_bf16_lut", "tanh_gelu_bf16_lut"}:
        from .pointwise_math import selected_steps

        return selected_steps(plan, selected)
    names = [
        row["descriptor"]
        for row in selected["values"]
        if row["descriptor"] in BF16_BINARY_MODES or row["descriptor"] in _UNARY_RULES
    ]
    if len(names) != 1:
        raise NativeCompilationError("unsupported_semantics", "pointwise loop needs one selected tile operation")
    descriptor = next(item for item in AtlasNativeBinding.profile().descriptors if item.name == names[0])
    if descriptor.computation != plan.source.nodes[-1].op or selected["request_digest"] != plan.body_request().digest():
        raise NativeCompilationError("compile_error", "selected pointwise tile differs from its source")
    return names[0]


def virtual_program(plan, addresses, output, descriptor):
    composite = isinstance(descriptor, list)
    unary = len(addresses) == 1 if composite else descriptor in _UNARY_RULES
    second_load = len(addresses) == 2
    kind = None if composite else _UNARY_RULES[descriptor] if unary else BF16_BINARY_MODES[descriptor]
    state = "!atlas.virtual_state"
    tensor = "!atlas.virtual_bf16"

    def signed(value):
        return value if value < 2**31 else value - 2**32

    second_base = f"    %bbase = arith.constant {signed(addresses[1])} : i32\n" if second_load else ""
    second_arg = ",%bbase" if second_load else ""
    second_type = ",i32" if second_load else ""
    second_param = ",%bp: i32" if second_load else ""
    if composite:
        from .pointwise_math import virtual_steps

        operation = (
            ""
            if unary
            else f'    %s2, %b = "atlas.virtual_dma_load_bf16"(%s1,%bp,%bytes) : ({state},i32,i32) -> ({state},{tensor})\n'
        )
        operation += virtual_steps(plan, descriptor)
    else:
        load = (f'    %s2, %b = "atlas.virtual_dma_load_bf16"(%s1,%bp,%bytes) : ({state},i32,i32) -> ({state},{tensor})\n'
                if second_load else "")
        right = "%b" if second_load else "%a"
        operation = (
            f'    %r = "atlas.virtual_vpu_unary"(%a) {{kind = "{kind}"}} : ({tensor}) -> {tensor}\n'
            if unary
            else load + f'    %r = "atlas.virtual_vpu_binary"(%a,{right}) {{kind = "{kind}"}} : ({tensor},{tensor}) -> {tensor}\n'
        )
    stored_state = "%s2" if second_load else "%s1"
    if plan.compact_broadcasts:
        return _broadcast_virtual_program(plan, addresses, output, operation, stored_state)
    second_step = "    %bn = arith.addi %bp, %bytes : i32\n" if second_load else ""
    second_next = ",%bn" if second_load else ""

    return f"""module {{
  func.func @pointwise() -> {state} attributes {{atlas.input_dram_base = {addresses[0]} : i64, atlas.output_dram_base = {output} : i64, atlas.output_dram_bytes = {plan.packed_bytes} : i64}} {{
    %s = "atlas.virtual_start"() : () -> {state}
    %zero = arith.constant 0 : i32
    %abase = arith.constant {signed(addresses[0])} : i32
{second_base}    %obase = arith.constant {signed(output)} : i32
    cf.br ^loop(%s,%zero,%abase{second_arg},%obase : {state},i32,i32{second_type},i32)
  ^loop(%token: {state},%i: i32,%ap: i32{second_param},%op: i32):
    %count = arith.constant {plan.tile_counts[0] * plan.tile_counts[1]} : i32
    %more = arith.cmpi ult, %i, %count : i32
    cf.cond_br %more, ^body(%token : {state}), ^done(%token : {state})
  ^body(%t: {state}):
    %bytes = arith.constant {plan.tile_bytes} : i32
    %s1, %a = "atlas.virtual_dma_load_bf16"(%t,%ap,%bytes) : ({state},i32,i32) -> ({state},{tensor})
{operation}    %s3 = "atlas.virtual_dma_store_bf16"({stored_state},%r,%op,%bytes) : ({state},{tensor},i32,i32) -> {state}
    %one = arith.constant 1 : i32
    %next = arith.addi %i, %one : i32
    %an = arith.addi %ap, %bytes : i32
{second_step}    %on = arith.addi %op, %bytes : i32
    cf.br ^loop(%s3,%next,%an{second_next},%on : {state},i32,i32{second_type},i32)
  ^done(%end: {state}):
    return %end : {state}
  }}
}}
"""


def _broadcast_virtual_program(plan, addresses, output, operation, stored_state):
    """Use row/column induction variables; no division or expanded tile table."""
    state, tensor = "!atlas.virtual_state", "!atlas.virtual_bf16"
    names = ("a", "b")[:len(addresses)]
    constants, declarations = {}, []

    def constant(value, hint):
        if value not in constants:
            constants[value] = "%" + hint
            signed = value if value < 2**31 else value - 2**32
            declarations.append(f"    %{hint} = arith.constant {signed} : i32")
        return constants[value]

    zero, one = constant(0, "zero"), constant(1, "one")
    constant(plan.tile_bytes, "bytes")
    rows, columns = (constant(count, hint) for count, hint in zip(plan.tile_counts, ("rows", "columns")))
    output_row_step = constant(plan.tile_counts[1] * plan.tile_bytes, "output_row_step")
    output_base = constant(output, "obase")
    bases = [constant(address, name + "base") for name, address in zip(names, addresses, strict=True)]
    strides = [plan.input_tile_strides(i) for i in range(len(names))]
    row_steps = [constant(row * plan.tile_bytes, name + "rowstep")
                 for name, (row, _) in zip(names, strides, strict=True)]
    column_steps = [constant(column * plan.tile_bytes, name + "colstep")
                    for name, (_, column) in zip(names, strides, strict=True)]
    row_bases = [f"%{name}row" if row else base
                 for name, (row, _), base in zip(names, strides, bases, strict=True)]
    input_pointers = [f"%{name}p" if column else base
                      for name, (_, column), base in zip(names, strides, row_bases, strict=True)]
    if len(names) == 2:
        operation = operation.replace("%bp", input_pointers[1])
    # Do not carry invariant pointers or duplicate constants around backedges:
    # they otherwise create avoidable scalar interference and address spills.
    row_ids = [i for i, (row, _) in enumerate(strides) if row]
    column_ids = [i for i, (_, column) in enumerate(strides) if column]
    row_params = ",".join(["%i: i32", *(f"%{names[i]}row: i32" for i in row_ids), "%orow: i32"])
    column_params = ",".join(["%j: i32", *(f"%{names[i]}p: i32" for i in column_ids), "%op: i32"])
    row_types = ",".join([state] + ["i32"] * (2 + len(row_ids)))
    column_types = ",".join([state] + ["i32"] * (2 + len(column_ids)))
    initial_rows = ",".join([zero, *(bases[i] for i in row_ids), output_base])
    initial_columns = ",".join([zero, *(row_bases[i] for i in column_ids), "%orow"])
    next_columns = ",".join(["%next_column", *(f"%{names[i]}n" for i in column_ids), "%on"])
    next_rows = ",".join(["%next_row", *(f"%{names[i]}rownext" for i in row_ids), "%next_output_row"])
    column_updates = "\n".join(f"    %{names[i]}n = arith.addi %{names[i]}p, {column_steps[i]} : i32"
                               for i in column_ids)
    row_updates = "\n".join(f"    %{names[i]}rownext = arith.addi %{names[i]}row, {row_steps[i]} : i32"
                            for i in row_ids)
    return f"""module {{
  func.func @pointwise() -> {state} attributes {{atlas.input_dram_base = {addresses[0]} : i64, atlas.output_dram_base = {output} : i64, atlas.output_dram_bytes = {plan.packed_bytes} : i64}} {{
    %s = "atlas.virtual_start"() : () -> {state}
{chr(10).join(declarations)}
    cf.br ^rows_loop(%s,{initial_rows} : {row_types})
  ^rows_loop(%token: {state},{row_params}):
    %more_rows = arith.cmpi ult, %i, {rows} : i32
    cf.cond_br %more_rows, ^start_row(%token : {state}), ^done(%token : {state})
  ^start_row(%starttoken: {state}):
    cf.br ^columns_loop(%starttoken,{initial_columns} : {column_types})
  ^columns_loop(%coltoken: {state},{column_params}):
    %more_columns = arith.cmpi ult, %j, {columns} : i32
    cf.cond_br %more_columns, ^body(%coltoken : {state}), ^advance_row(%coltoken : {state})
  ^body(%t: {state}):
    %s1, %a = "atlas.virtual_dma_load_bf16"(%t,{input_pointers[0]},%bytes) : ({state},i32,i32) -> ({state},{tensor})
{operation}    %s3 = "atlas.virtual_dma_store_bf16"({stored_state},%r,%op,%bytes) : ({state},{tensor},i32,i32) -> {state}
    %next_column = arith.addi %j, {one} : i32
{column_updates}
    %on = arith.addi %op, %bytes : i32
    cf.br ^columns_loop(%s3,{next_columns} : {column_types})
  ^advance_row(%rowtoken: {state}):
    %next_row = arith.addi %i, {one} : i32
{row_updates}
    %next_output_row = arith.addi %orow, {output_row_step} : i32
    cf.br ^rows_loop(%rowtoken,{next_rows} : {row_types})
  ^done(%end: {state}):
    return %end : {state}
  }}
}}
"""


def compile_pointwise(
    snapshot,
    request,
    *,
    fixed_inputs,
    fixed_outputs,
    target_source,
    destination,
    limits,
):
    from .binding import (
        DRAM_BASE,
        NATIVE_CHECKS,
        NATIVE_PIPELINES,
        RTL_REVISION,
        TILE_BYTES,
        _native_pipeline,
        _run,
        dram_slots,
    )

    plan = prepare_plan(request)
    if set(fixed_inputs) != {node.id for node in request.nodes[:-1]} or len(fixed_outputs) != 1:
        raise NativeCompilationError("compile_error", "pointwise loop needs every input and one output binding")
    slots = tuple(fixed_inputs[node.id] for node in request.nodes[:-1]) + tuple(fixed_outputs)
    sizes = (*plan.input_packed_bytes, plan.packed_bytes)
    if any(
        type(slot) is not int or slot < 0 or slot * TILE_BYTES + size > dram_slots() * TILE_BYTES
        for slot, size in zip(slots, sizes, strict=True)
    ):
        raise NativeCompilationError("resource_limit", "pointwise buffers exceed the selected memory window")
    addresses = [DRAM_BASE + fixed_inputs[node.id] * TILE_BYTES for node in request.nodes[:-1]]
    output = DRAM_BASE + fixed_outputs[0] * TILE_BYTES
    ranges = [(address, address + size) for address, size in zip([*addresses, output], sizes, strict=True)]
    if any(a < d and c < b for i, (a, b) in enumerate(ranges) for c, d in ranges[i + 1 :]):
        raise NativeCompilationError("infeasible_candidate", "pointwise physical buffers overlap")
    destination.mkdir(parents=True, exist_ok=False)
    body = plan.body_request()
    body_abi = AtlasNativeBinding.plan_boundary_abi(body)
    body_manifest = AtlasNativeBinding.compile(
        snapshot,
        body,
        fixed_inputs=body_abi["fixed_inputs"],
        fixed_outputs=tuple(body_abi["fixed_outputs"]),
        target_source=target_source,
        destination=destination / "body",
        limits=limits,
        allow_staging=False,
    )
    descriptor = selected_body_descriptor(plan, destination / "body")
    virtual = virtual_program(plan, addresses, output, descriptor)
    tools = Path(os.environ.get("ATLAS_OOT_BIN_DIR", target_source / "build/bin"))
    pipeline = _native_pipeline()
    mandatory = _run(
        [str(tools / "atlas-opt"), *NATIVE_PIPELINES["mandatory"], "-"],
        input_text=virtual,
    )
    _run(
        [str(tools / "atlas-opt"), *NATIVE_CHECKS, "--verify-atlas-timing", "-"],
        input_text=mandatory,
    )
    machine = (
        mandatory
        if pipeline == "mandatory"
        else _run(
            [str(tools / "atlas-opt"), *NATIVE_PIPELINES[pipeline], "-"],
            input_text=virtual,
        )
    )
    _run(
        [str(tools / "atlas-opt"), *NATIVE_CHECKS, "--verify-atlas-timing", "-"],
        input_text=machine,
    )
    program = json.loads(_run([str(tools / "atlas-emit"), "--program-json", "-"], input_text=machine))
    if program.get("selected_rtl_revision") != RTL_REVISION:
        raise NativeCompilationError("unqualified_target", "pointwise emitter uses a different RTL")
    binary = b"".join(row["word_u32"].to_bytes(4, "little") for row in program["instructions"])
    execution = {
        "schema": "atlas.pointwise_execution_plan.v1",
        "shape": list(plan.shape),
        "input_addresses": addresses,
        "input_buffer_bytes": list(plan.input_packed_bytes),
        "output_addresses": [output],
        "output_bytes": plan.packed_bytes,
        "output_physical_layouts": ["mn_tiles_bf16_pair_halves"],
        "tiling": plan.record(),
    }
    files = {
        "source-request.json": request.record(),
        "execution_plan.json": execution,
        "pointwise-tiling.json": plan.record(),
        "physical-program.json": program,
    }
    for name, data in files.items():
        (destination / name).write_text(json.dumps(data, sort_keys=True, indent=2) + "\n")
    (destination / "atlas-virtual.mlir").write_text(virtual)
    (destination / "atlas-machine.mlir").write_text(machine)
    (destination / "program.bin").write_bytes(binary)
    if pipeline != "mandatory":
        (destination / "mandatory-qualification-machine.mlir").write_text(mandatory)
        (destination / "mandatory-qualification-program.json").write_text(
            _run([str(tools / "atlas-emit"), "--program-json", "-"], input_text=mandatory)
        )

    def sha(data):
        return hashlib.sha256(data).hexdigest()

    manifest = {
        "schema": "atlas.native_pointwise_compilation.v1",
        "artifact_kind": "program",
        "engine": "merlin_native",
        "compile_mode": "diagnostic",
        "qualification_scope": "diagnostic_only",
        "request_digest": request.digest(),
        "target_identity": request.target_identity,
        "binary_sha256": sha(binary),
        "execution_plan_sha256": sha((destination / "execution_plan.json").read_bytes()),
        "body_check_fingerprint": body_manifest["native_check_fingerprint"],
        "selected_body_descriptor": descriptor,
        "lowering_pipeline": pipeline,
        "semantic_qualification_pipeline": "mandatory",
        "target_inventory_sha256": sha((target_source / "docs/selected-variant-inventory.json").read_bytes()),
        "pointwise_lowering_source_sha256": sha(Path(__file__).read_bytes()),
        "native_snapshot_sha256": sha((snapshot.root / "manifest.json").read_bytes()),
        "io_memory_bytes": dram_slots() * TILE_BYTES,
        "stage_artifacts": {
            path.relative_to(destination).as_posix(): sha(path.read_bytes())
            for path in sorted(destination.rglob("*"))
            if path.is_file() and path != destination / "program.bin"
        },
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    return manifest
