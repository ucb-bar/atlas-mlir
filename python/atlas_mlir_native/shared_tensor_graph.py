"""Explicit diagnostic f32-to-Atlas precision selection over shared tensor graphs.

Source recognition and ordered dense schedules belong to Merlin. All physical
operations, conversions, reduction geometry and approximations below are Atlas
policy. Compilation is not an assertion of f32 equivalence or numeric admission.
"""

import json
from math import isfinite
from pathlib import Path

from merlin.frontends.linalg_tensor_graph import read_tensor_graph
from merlin.runtime.fp8_formats import float_to_codes
from merlin.transforms.ordered_float_dense import OrderedFloatDense

from .shared_dense import (
    INPUT_BASE,
    OUTPUT_BASE,
    POLICY,
    RTL_REVISION,
    SLOT_BYTES,
    compile_virtual,
    expand_dense,
    schedule_dense,
)
from .virtual_builder import VirtualBuilder

PRECISION = "atlas-f32-to-bf16-e4m3-diagnostic-v1"

# These choices belong to this target adapter. The shared reader preserves f32
# operations; none of these replacements is an equality rewrite of that source.
NUMERICS = {
    "input": "f32 to E4M3 RNE/saturate/flush for matrix use; f32 to BF16 saturating RNE for vector use",
    "splat": "finite literal to BF16 saturating RNE; prescribed reduction seeds remain implicit",
    "transpose": "logical transpose; physical FP8 XLU when required",
    "expand_dims": "row-reduction broadcast view",
    "matmul": "E4M3 operands; ascending K; BF16 RNE after each scalar FMA",
    "add": "BF16 operands; FP32 sum RNE; BF16 chop",
    "sub": "BF16 operands; FP32 difference RNE; BF16 chop",
    "mul": "BF16 product RNE",
    "div": "Atlas BF16 reciprocal approximation followed by BF16 multiply",
    "exp": "Atlas BF16 LUT exponential approximation",
    "relu": "BF16 sign clamp",
    "reduce_max": "Atlas row maximum; finite BF16 ordering",
    "reduce_sum": "Atlas adjacent-pair FP32 tree; final BF16 RNE",
}


def lower_tensor_graph(request, *, precision):
    if precision != PRECISION:
        raise ValueError("explicit Atlas diagnostic precision policy required")
    builder = VirtualBuilder(INPUT_BASE, OUTPUT_BASE)
    fp8, bf16 = "!atlas.virtual_fp8", "!atlas.virtual_bf16"
    values, inputs, outputs, schedules, conversions = {}, [], [], {}, []
    for node in request.nodes:
        if node.type.dtype != "f32":
            raise ValueError("selected Atlas source precision is f32")
        if node.op.startswith("reduce_"):
            shape = request.node(node.inputs[0]).type.shape
            if dict(node.attrs)["axis"] != 1 or len(shape) != 2 or shape[1] != 32:
                raise ValueError(
                    "Atlas reduction requires one complete 32-element row; cross-tile reassociation is not admitted"
                )
        if node.op not in ("expand_dims", "reduce_max", "reduce_sum", "splat") and (
            len(node.type.shape) != 2 or any(d % 32 for d in node.type.shape)
        ):
            raise ValueError("Atlas requires full rank-two tiles")

    def pure(op, operands, out=bf16, attrs="", types=None):
        return builder.pure(op, operands, types or [bf16] * len(operands), out, attrs)

    def tile(name, row, col, dtype="bf16", transpose=False):
        key = name, row, col, dtype, transpose
        if key in values:
            return values[key]
        node = request.node(name)
        if node.op == "transpose":
            value = tile(node.inputs[0], col, row, dtype, not transpose)
        elif node.op == "expand_dims":
            value = tile(node.inputs[0], row, 0, dtype, transpose)
        elif transpose:
            if dtype != "fp8":
                raise ValueError(
                    "Atlas only has an FP8 physical transpose in this path"
                )
            value = pure(
                "virtual_xlu_transpose", [tile(name, row, col, dtype)], fp8, types=[fp8]
            )
        elif node.op == "input":
            if len(inputs) >= (OUTPUT_BASE - INPUT_BASE) // SLOT_BYTES:
                raise ValueError("input slots overlap outputs")
            inputs.append({"node": name, "row": row, "col": col, "dtype": dtype})
            value = builder.emit(
                "virtual_input_" + dtype,
                [],
                [],
                fp8 if dtype == "fp8" else bf16,
                f" {{index = {len(inputs) - 1} : i32}}",
            )
        elif dtype == "fp8":
            value = pure(
                "virtual_pack_fp8",
                [tile(name, row, col)],
                fp8,
                " {scale_code = 127 : i32}",
            )
            conversions.append(
                {
                    "node": name,
                    "row": row,
                    "col": col,
                    "conversion": "BF16 to E4M3 RTL RNE/flush/clamp",
                    "scale_code": 127,
                }
            )
        elif node.op == "splat":
            literal = float.fromhex(dict(node.attrs)["value_hex"])
            if not isfinite(literal):
                raise ValueError(
                    "nonfinite literals are supported only as implicit reduction seeds"
                )
            bits = int(float_to_codes([literal], "bf16")[0])
            value = pure("virtual_vpu_splat", [], attrs=f" {{bits = {bits} : i32}}")
        else:
            raise ValueError("producer tile not available: " + name)
        values[key] = value
        return value

    products = 0
    for node in request.nodes:
        kind = node.op
        if kind in ("input", "transpose", "expand_dims", "splat"):
            continue
        if kind == "matmul":
            left, right = (request.node(i) for i in node.inputs)
            m, k = left.type.shape
            rk, n = right.type.shape
            if k != rk:
                raise ValueError("matrix shape differs")
            products += (m // 32) * (k // 32) * (n // 32)
            if products > 64:
                raise ValueError("Atlas dense expansion budget exceeded")
            selected = OrderedFloatDense(
                request.source_identity, "graph", m, k, n, False, POLICY
            )
            scheduled = schedule_dense(selected)
            schedules[node.id] = scheduled

            def load(role, r, c, node=node):
                return (
                    tile(node.inputs[0], r, c, "fp8")
                    if role == "lhs"
                    else tile(node.inputs[1], r, c, "fp8", True)
                )

            def store(r, c, value, node=node):
                values[node.id, r, c, "bf16", False] = value

            expand_dense(selected, scheduled, builder, load, store)
            continue
        shape = node.type.shape
        rows = shape[0] // 32
        cols = 1 if len(shape) == 1 else shape[1] // 32
        for r in range(rows):
            for c in range(cols):
                args = []
                for name in (
                    node.inputs[:1] if kind.startswith("reduce_") else node.inputs
                ):
                    operand = request.node(name)
                    args.append(
                        tile(
                            name,
                            r,
                            0
                            if len(operand.type.shape) == 1
                            or operand.type.shape[-1] == 1
                            else c,
                        )
                    )
                if kind in ("add", "sub", "mul"):
                    value = pure(
                        "virtual_vpu_binary", args, attrs=f' {{kind = "{kind}"}}'
                    )
                elif kind in ("relu", "exp"):
                    value = pure(
                        "virtual_vpu_unary", args, attrs=f' {{kind = "{kind}"}}'
                    )
                elif kind == "div":
                    inverse = pure(
                        "virtual_vpu_unary", [args[1]], attrs=' {kind = "recip"}'
                    )
                    value = pure(
                        "virtual_vpu_binary",
                        [args[0], inverse],
                        attrs=' {kind = "mul"}',
                    )
                elif kind in ("reduce_max", "reduce_sum"):
                    mode = "row_max" if kind == "reduce_max" else "row_sum"
                    value = pure(
                        "virtual_vpu_reduce", args, attrs=f' {{kind = "{mode}"}}'
                    )
                else:
                    raise ValueError("no Atlas lowering for " + kind)
                values[node.id, r, c, "bf16", False] = value
    for name in request.outputs:
        shape = request.node(name).type.shape
        if len(shape) != 2 or any(d % 32 for d in shape):
            raise ValueError("initial output ABI requires full BF16 tiles")
        for r in range(shape[0] // 32):
            for c in range(shape[1] // 32):
                builder.emit(
                    "virtual_output_bf16",
                    [tile(name, r, c)],
                    [bf16],
                    attrs=f" {{index = {len(outputs)} : i32}}",
                )
                outputs.append({"node": name, "row": r, "col": c})
    return (
        builder.text(),
        {"inputs": inputs, "outputs": outputs, "conversions": conversions},
        schedules,
    )


def pack_tensor_inputs(request, abi, arguments):
    import numpy as np

    from .layout import pack_bf16_pair_halves

    if set(arguments) != {n.id for n in request.nodes if n.op == "input"}:
        raise ValueError("source input identities differ")
    for name, value in arguments.items():
        if (
            value.dtype != np.dtype("float32")
            or tuple(value.shape) != request.node(name).type.shape
            or not np.all(np.isfinite(value))
        ):
            raise ValueError("input shape, f32 dtype or finite domain differs")
    payloads = []
    for index, row in enumerate(abi["inputs"]):
        source = arguments[row["node"]][
            row["row"] * 32 : (row["row"] + 1) * 32,
            row["col"] * 32 : (row["col"] + 1) * 32,
        ]
        if row["dtype"] == "fp8":
            # Explicit saturating, normal-only boundary policy. PACK on device
            # remains its own RTL operation and is never replaced by this helper.
            codes = float_to_codes(np.clip(source, -448, 448), "fp8_e4m3").astype(
                np.uint8
            )
            codes[(codes & 0x78) == 0] = 0
            payload = codes.tobytes()
        else:
            payload = pack_bf16_pair_halves(
                float_to_codes(source, "bf16").astype("<u2").tobytes()
            )
        payloads.append((index * SLOT_BYTES, payload))
    return tuple(payloads)


def compile_tensor_graph(
    source, *, entry, precision, destination, atlas_bin_dir, pipeline="source-order"
):
    request = read_tensor_graph(source, entry=entry, target_identity=RTL_REVISION)
    virtual, abi, schedules = lower_tensor_graph(request, precision=precision)
    manifest = compile_virtual(
        source,
        virtual,
        destination=destination,
        atlas_bin_dir=atlas_bin_dir,
        pipeline=pipeline,
    )
    manifest.update(
        schema="atlas.shared-tensor-graph-diagnostic.v1",
        precision=precision,
        abi=abi,
        request=request.record(),
        semantic_status="explicit approximate precision change; f32 equivalence unproved",
        replacements={
            n.id: {"source_operation": n.op, "target_semantics": NUMERICS[n.op]}
            for n in request.nodes
        },
        unresolved_obligations=[
            "runtime intermediate domains and numeric exceptional cases",
            "application quality under the selected precision change",
            "independent RTL timing and integrated execution qualification",
        ],
    )
    destination = Path(destination)
    for name, schedule in schedules.items():
        (destination / (name + ".mk")).write_text(schedule.kernel.text())
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
