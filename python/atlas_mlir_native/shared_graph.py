"""Diagnostic graph lowering through Merlin source checks and dense schedules.

This adapter is trusted support, not an independently generated compiler. Numeric
domain premises are retained in the manifest and require execution-side checks.
"""

from dataclasses import asdict
import json
from pathlib import Path

from merlin.frontends.linalg_float_graph import read_float_graph
from merlin.transforms.ordered_float_dense import OrderedFloatDense

from .layout import pack_bf16_pair_halves
from .shared_dense import INPUT_BASE, OUTPUT_BASE, POLICY, RTL_REVISION, SLOT_BYTES, expand_dense, schedule_dense
from .virtual_builder import VirtualBuilder


def plan_graph(source, *, entry):
    request = read_float_graph(source, entry=entry, target_identity=RTL_REVISION, ordered_policies=(POLICY,))
    schedules = {}
    for node in request.nodes:
        if any(d % 32 for d in node.type.shape):
            raise ValueError("Atlas graph requires full tiles; no implicit padding")
        if node.type.dtype not in ("bf16", "f8E4M3FN"):
            raise ValueError("Atlas graph has an unsupported element format")
        if node.op == "input":
            domain = dict(node.attrs).get("input_domain")
            admitted = POLICY.input_domain if node.type.dtype == "f8E4M3FN" else "finite-normal-or-zero-bf16"
            if domain is not None and domain != admitted:
                raise ValueError("Atlas input declares an unsupported domain")
            continue
        attrs = dict(node.attrs)
        if node.op == "ordered_dot":
            selected = OrderedFloatDense(request.source_identity, entry, attrs["m"], attrs["k"], attrs["n"],
                                         attrs["rhs_transposed"], POLICY)
            schedules[node.id] = (selected, schedule_dense(selected))
        elif node.op == "relu" and node.type.numerical_policy == "bf16-sign-clamp-v1":
            pass
        elif node.op == "cast" and (node.type.numerical_policy, attrs["input_domain"], attrs["scale_code"]) == (
                "fp8-e4m3-unit-e8m0-seven-codes-v1", "bf16-seven-powers-and-zero-v1", 127):
            pass
        else:
            raise ValueError(f"Atlas graph has no qualified lowering for {node.op}: {node.type.numerical_policy}")
    if sum(s.m // 32 * (s.k // 32) * (s.n // 32) for s, _ in schedules.values()) > 64:
        raise ValueError("Atlas graph exceeds its finite dense expansion budget")
    if any(request.node(output).type.dtype != "bf16" for output in request.outputs):
        raise ValueError("Atlas graph output ABI requires BF16")
    return request, schedules


def virtual_graph(request, schedules):
    """Keep producer tiles live in virtual SSA; physical allocation stays in OOT."""
    builder = VirtualBuilder(INPUT_BASE, OUTPUT_BASE)
    values, inputs, outputs = {}, [], []
    fp8, bf16 = "!atlas.virtual_fp8", "!atlas.virtual_bf16"

    def coordinates(node):
        return ((r, c) for r in range(node.type.shape[0] // 32) for c in range(node.type.shape[1] // 32))

    def tile(node_id, row, col, transpose=False):
        key = node_id, row, col, transpose
        if key in values:
            return values[key]
        node = request.node(node_id)
        if node.op != "input":
            raise ValueError("Atlas graph requires a physical transpose of an intermediate tile")
        if len(inputs) >= (OUTPUT_BASE - INPUT_BASE) // SLOT_BYTES:
            raise ValueError("Atlas graph input slots overlap output storage")
        inputs.append(dict(node=node_id, row=row, col=col, transpose=transpose, dtype=node.type.dtype))
        suffix, ty = ("fp8", fp8) if node.type.dtype == "f8E4M3FN" else ("bf16", bf16)
        values[key] = builder.emit(f"virtual_input_{suffix}", [], [], ty,
                                   f" {{index = {len(inputs)-1} : i32}}")
        return values[key]

    for node in request.nodes:
        if node.op == "input":
            continue
        if node.op == "ordered_dot":
            selected, scheduled = schedules[node.id]
            def load(role, row, col):
                if role == "lhs":
                    return tile(node.inputs[0], row, col)
                # Hardware weight bytes are N-major, K-minor.
                return tile(node.inputs[1], col, row) if selected.rhs_transposed else tile(
                    node.inputs[1], row, col, True)
            def store(row, col, value):
                values[node.id, row, col, False] = value
            expand_dense(selected, scheduled, builder, load, store)
        else:
            for row, col in coordinates(node):
                operand = tile(node.inputs[0], row, col)
                if node.op == "relu":
                    value = builder.pure("virtual_vpu_unary", [operand], [bf16], bf16, ' {kind = "relu"}')
                elif node.op == "cast":
                    value = builder.pure("virtual_pack_fp8", [operand], [bf16], fp8, " {scale_code = 127 : i32}")
                else:
                    raise ValueError("unhandled graph operation")
                values[node.id, row, col, False] = value
    for output in request.outputs:
        for row, col in coordinates(request.node(output)):
            value = tile(output, row, col)
            builder.emit("virtual_output_bf16", [value], [bf16], attrs=f" {{index = {len(outputs)} : i32}}")
            outputs.append(dict(node=output, row=row, col=col))
    return builder.text(), dict(inputs=inputs, outputs=outputs)


def pack_graph_inputs(request, abi, arguments):
    """Pack row-major source bytes without conversion or reference evaluation."""
    expected = {n.id for n in request.nodes if n.op == "input"}
    if set(arguments) != expected:
        raise ValueError("graph input identities differ")
    for name, payload in arguments.items():
        node = request.node(name)
        width = 2 if node.type.dtype == "bf16" else 1
        if type(payload) is not bytes or len(payload) != node.type.shape[0] * node.type.shape[1] * width:
            raise ValueError("graph input extent differs")
        # All currently selected consumers require finite normal or zero input.
        if width == 1 and any(v and (v & 0x78 == 0 or v & 0x7f == 0x7f) for v in payload):
            raise ValueError("graph FP8 input outside selected domain")
        if width == 2:
            codes = (int.from_bytes(payload[i:i+2], "little") for i in range(0, len(payload), 2))
            if any((v & 0x7f80 == 0 and v & 0x7f) or v & 0x7f80 == 0x7f80 for v in codes):
                raise ValueError("graph BF16 input outside selected domain")
    packed = []
    for index, record in enumerate(abi["inputs"]):
        node = request.node(record["node"])
        payload = arguments[node.id]
        width = 2 if node.type.dtype == "bf16" else 1
        def element(r, c):
            start = ((record["row"] * 32 + r) * node.type.shape[1] + record["col"] * 32 + c) * width
            return payload[start:start+width]
        data = b"".join(element(c, r) if record["transpose"] else element(r, c)
                        for r in range(32) for c in range(32))
        if width == 2:
            data = pack_bf16_pair_halves(data)
        packed.append((index * SLOT_BYTES, data))
    return tuple(packed)


def compile_graph(source, *, entry, destination, atlas_bin_dir, pipeline="source-order"):
    from .shared_dense import compile_virtual

    request, schedules = plan_graph(source, entry=entry)
    virtual, abi = virtual_graph(request, schedules)
    manifest = compile_virtual(source, virtual, destination=destination, atlas_bin_dir=atlas_bin_dir, pipeline=pipeline)
    manifest.update(schema="atlas.shared-graph-diagnostic.v1", request=request.record(), abi=abi,
                    domain_premises=[dict(node=n.id, domain=dict(n.attrs)["input_domain"], operands=n.inputs)
                                     for n in request.nodes if "input_domain" in dict(n.attrs)],
                    dense_schedules={key: dict(source=asdict(s), kernel_sha256=k.kernel.digest())
                                     for key, (s, k) in schedules.items()})
    destination = Path(destination)
    (destination / "request.json").write_text(json.dumps(request.record(), indent=2) + "\n")
    for key, (_, scheduled) in schedules.items():
        (destination / f"{key}.mk").write_text(scheduled.kernel.text())
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
