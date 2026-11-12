"""Admit bounded source graph families under the selected diagnostic contract.

Compiler-side admission/replay retains its diagnostic scope. Independent reference
and evaluator authority remain separate; no hardware execution is established.
"""

from __future__ import annotations

from merlin.compiler.semantic.model import (
    IndexMap,
    KernelRequest,
    TensorType,
    pointwise_identity_maps,
)

from .pack_lane import ROUNDED_PACK_DOMAIN, ROUNDED_PACK_POLICY
from .policy import (
    BF16_ADD_ATTRS,
    BF16_BINARY_ATTRS,
    BF16_MUL_ATTRS,
    BF16_RELU_ATTRS,
    BF16_ROW_REDUCTION_ATTRS,
    BF16_UNARY_ATTRS,
    DRAM_SLOTS,
    MXU0_INPUT_DOMAIN,
    MXU0_POLICY,
    PACK_ATTRS,
    PACK_POLICY,
    SCALED_PACK_DOMAIN,
    SCALED_PACK_POLICY,
    SCALED_PACK_RANGE,
)


def _scaled_pack_code(node) -> int | None:
    attrs = dict(node.attrs)
    code = attrs.get("scale_code")
    domain = {
        SCALED_PACK_POLICY: SCALED_PACK_DOMAIN,
        ROUNDED_PACK_POLICY: ROUNDED_PACK_DOMAIN,
    }.get(node.type.numerical_policy)
    if (
        domain is not None
        and set(attrs) == {"input_domain", "scale_code"}
        and attrs["input_domain"] == domain
        and type(code) is int
        and SCALED_PACK_RANGE[0] <= code <= SCALED_PACK_RANGE[1]
    ):
        return code
    return None


def _interlayer_pack_code(node) -> int | None:
    if node.type.numerical_policy == PACK_POLICY and node.attrs == PACK_ATTRS:
        return 127
    return _scaled_pack_code(node)


def _bf16_tile_request(request: KernelRequest) -> bool:
    """Admit bounded BF16 movement, binary arithmetic, and ReLU DAGs."""
    if not request.input_storages or not request.outputs:
        return False
    nodes = {node.id: node for node in request.nodes}
    if any(
        nodes[output].op
        not in {
            "identity",
            "relu",
            *BF16_UNARY_ATTRS,
            *BF16_ROW_REDUCTION_ATTRS,
            *BF16_BINARY_ATTRS,
        }
        for output in request.outputs
    ):
        return False
    for node in request.nodes:
        if (
            node.type.shape != (32, 32)
            or node.type.dtype != "bf16"
            or (node.type.numerical_policy != "raw-bit-pattern")
        ):
            return False
        if node.op == "input":
            if (
                node.effect != "input"
                or node.attrs
                not in (
                    (),
                    *((attrs[0],) for attrs in BF16_UNARY_ATTRS.values()),
                    *((attrs[0],) for attrs in BF16_ROW_REDUCTION_ATTRS.values()),
                    *((attrs[0],) for attrs in BF16_BINARY_ATTRS.values()),
                )
                or node.index_maps
            ):
                return False
        elif node.op == "identity":
            if (
                node.effect != "pure"
                or node.attrs
                or node.index_maps
                or len(node.inputs) != 1
                or nodes[node.inputs[0]].type != node.type
            ):
                return False
        elif node.op == "relu":
            if (
                node.effect != "pure"
                or dict(node.attrs) != dict(BF16_RELU_ATTRS)
                or node.index_maps != pointwise_identity_maps(2, 1)
                or len(node.inputs) != 1
                or nodes[node.inputs[0]].type != node.type
            ):
                return False
        elif node.op in BF16_UNARY_ATTRS:
            if (
                node.effect != "pure"
                or node.attrs != BF16_UNARY_ATTRS[node.op]
                or node.index_maps != pointwise_identity_maps(2, 1)
                or len(node.inputs) != 1
                or nodes[node.inputs[0]].type != node.type
            ):
                return False
        elif node.op in BF16_ROW_REDUCTION_ATTRS:
            if (
                node.effect != "pure"
                or node.attrs != BF16_ROW_REDUCTION_ATTRS[node.op]
                or node.index_maps != _row_reduce_maps()
                or len(node.inputs) != 1
                or nodes[node.inputs[0]].type != node.type
                or nodes[node.inputs[0]].op
                not in {
                    "input",
                    "relu",
                    *BF16_UNARY_ATTRS,
                    *BF16_ROW_REDUCTION_ATTRS,
                    *BF16_BINARY_ATTRS,
                }
            ):
                return False
        elif node.op in BF16_BINARY_ATTRS:
            attributes = BF16_BINARY_ATTRS[node.op]
            sources = tuple(nodes[source_id] for source_id in node.inputs)
            # A ReLU can return zero. Its binary consumer's finite-normal
            # operand premise remains unproved until the caller establishes
            # it for the selected invocation; diagnostic compilation records
            # that obligation in the execution plan.
            sources_admitted = all(
                (source.op == "input" and source.attrs == (attributes[0],))
                or (
                    source.op in BF16_BINARY_ATTRS
                    and source.attrs == BF16_BINARY_ATTRS[source.op]
                )
                or (source.op == "relu" and source.attrs == BF16_RELU_ATTRS)
                or (
                    source.op in BF16_UNARY_ATTRS
                    and source.attrs == BF16_UNARY_ATTRS[source.op]
                )
                or (
                    source.op in BF16_ROW_REDUCTION_ATTRS
                    and source.attrs == BF16_ROW_REDUCTION_ATTRS[source.op]
                )
                for source in sources
            )
            if (
                node.effect != "pure"
                or node.attrs != attributes
                or node.index_maps != pointwise_identity_maps(2, 2)
                or len(node.inputs) != 2
                or node.inputs[0] == node.inputs[1]
                or not sources_admitted
            ):
                return False
        else:
            return False
    return True


def _mxu_bf16_binary_request(request: KernelRequest) -> bool:
    """Admit one selected MXU0 readout consumed by a BF16 VPU binary op."""
    if len(request.nodes) != 5 or len(request.outputs) != 1:
        return False
    nodes = {node.id: node for node in request.nodes}
    if len(nodes) != 5:
        return False
    inputs = [node for node in request.nodes if node.op == "input"]
    mxus = [node for node in request.nodes if node.op == "matmul"]
    binaries = [node for node in request.nodes if node.op in {"add_bf16", "mul_bf16"}]
    if len(inputs) != 3 or len(mxus) != 1 or len(binaries) != 1:
        return False
    mxu, binary = mxus[0], binaries[0]
    if (
        request.outputs != (binary.id,)
        or len(mxu.inputs) != 2
        or len(binary.inputs) != 2
    ):
        return False
    fp8 = TensorType((32, 32), "fp8_e4m3", "raw-bit-pattern")
    bf16 = TensorType((32, 32), "bf16", "raw-bit-pattern")
    mxu_type = TensorType((32, 32), "bf16", MXU0_POLICY)
    fp8_inputs = [node for node in inputs if node.type == fp8]
    bf16_inputs = [node for node in inputs if node.type == bf16]
    attributes = BF16_ADD_ATTRS if binary.op == "add_bf16" else BF16_MUL_ATTRS
    return (
        len(fp8_inputs) == 2
        and len(bf16_inputs) == 1
        and all(
            node.effect == "input"
            and node.attrs == (("input_domain", MXU0_INPUT_DOMAIN),)
            and not node.index_maps
            for node in fp8_inputs
        )
        and bf16_inputs[0].effect == "input"
        and bf16_inputs[0].attrs == (attributes[0],)
        and not bf16_inputs[0].index_maps
        and mxu.inputs == tuple(node.id for node in fp8_inputs)
        and mxu.type == mxu_type
        and mxu.effect == "pure"
        and not mxu.attrs
        and mxu.index_maps == _mxu0_maps()
        and set(binary.inputs) == {mxu.id, bf16_inputs[0].id}
        and binary.type == bf16
        and binary.effect == "pure"
        and binary.attrs == attributes
        and binary.index_maps == pointwise_identity_maps(2, 2)
    )


def _mxu_bf16_dag_request(request: KernelRequest) -> bool:
    """Admit MXU0 reset results and a bounded BF16 VPU DAG with ordered outputs."""
    nodes = {node.id: node for node in request.nodes}
    if (
        len(nodes) != len(request.nodes)
        or not request.outputs
        or len(set(request.outputs)) != len(request.outputs)
        or any(output not in nodes for output in request.outputs)
    ):
        return False
    mxus = [node for node in request.nodes if node.op == "matmul"]
    vpu = [
        node
        for node in request.nodes
        if node.op
        in {"relu", *BF16_UNARY_ATTRS, *BF16_ROW_REDUCTION_ATTRS, *BF16_BINARY_ATTRS}
    ]
    packs = [node for node in request.nodes if node.op == "pack_fp8"]
    fp8_inputs = [
        node
        for node in request.nodes
        if node.op == "input" and node.type.dtype == "fp8_e4m3"
    ]
    if not mxus or not (vpu or packs) or len(fp8_inputs) < 2:
        return False
    fp8 = TensorType((32, 32), "fp8_e4m3", "raw-bit-pattern")
    packed_fp8 = {
        TensorType((32, 32), "fp8_e4m3", PACK_POLICY),
        TensorType((32, 32), "fp8_e4m3", SCALED_PACK_POLICY),
        TensorType((32, 32), "fp8_e4m3", ROUNDED_PACK_POLICY),
    }
    bf16 = TensorType((32, 32), "bf16", "raw-bit-pattern")
    mxu_type = TensorType((32, 32), "bf16", MXU0_POLICY)
    allowed_sources = {
        "input",
        "constant",
        "matmul",
        "relu",
        *BF16_UNARY_ATTRS,
        *BF16_ROW_REDUCTION_ATTRS,
        *BF16_BINARY_ATTRS,
    }
    for node in request.nodes:
        if any(source not in nodes for source in node.inputs):
            return False
        if node.op in {"input", "constant"}:
            if (
                node.effect != node.op
                or node.inputs
                or node.index_maps
                or (node.op == "constant" and node.type != bf16)
            ):
                return False
            if node.type == fp8:
                if node.attrs != (("input_domain", MXU0_INPUT_DOMAIN),):
                    return False
            elif node.type == bf16:
                if node.attrs not in (
                    (),
                    *((attrs[0],) for attrs in BF16_UNARY_ATTRS.values()),
                    *((attrs[0],) for attrs in BF16_ROW_REDUCTION_ATTRS.values()),
                    *((attrs[0],) for attrs in BF16_BINARY_ATTRS.values()),
                ):
                    return False
            else:
                return False
        elif node.op == "matmul":
            if (
                len(node.inputs) != 2
                or node.inputs[0] == node.inputs[1]
                or any(
                    (nodes[source].op != "input" or nodes[source].type != fp8)
                    and (
                        nodes[source].op != "pack_fp8"
                        or nodes[source].type not in packed_fp8
                    )
                    and (
                        nodes[source].op != "transpose"
                        or nodes[source].type not in packed_fp8
                    )
                    for source in node.inputs
                )
                or node.type != mxu_type
                or node.effect != "pure"
                or node.attrs
                or node.index_maps != _mxu0_maps()
            ):
                return False
        elif node.op == "pack_fp8":
            producer = nodes[node.inputs[0]] if len(node.inputs) == 1 else None
            if (
                node.type not in packed_fp8
                or node.effect != "pure"
                or _interlayer_pack_code(node) is None
                or producer is None
                or producer.op
                not in {
                    "matmul",
                    "relu",
                    *BF16_UNARY_ATTRS,
                    *BF16_ROW_REDUCTION_ATTRS,
                    *BF16_BINARY_ATTRS,
                }
                or producer.type not in {mxu_type, bf16}
                or node.index_maps != pointwise_identity_maps(2, 1)
            ):
                return False
        elif node.op == "transpose":
            producer = nodes[node.inputs[0]] if len(node.inputs) == 1 else None
            if (
                producer is None
                or producer.op != "pack_fp8"
                or producer.type not in packed_fp8
                or node.type != producer.type
                or node.effect != "pure"
                or node.attrs
                or node.index_maps != _transpose_maps()
            ):
                return False
        elif node.op == "relu":
            if (
                node.type != bf16
                or node.effect != "pure"
                or node.attrs != BF16_RELU_ATTRS
                or node.index_maps != pointwise_identity_maps(2, 1)
                or len(node.inputs) != 1
                or nodes[node.inputs[0]].type not in {bf16, mxu_type}
            ):
                return False
        elif node.op in BF16_UNARY_ATTRS:
            if (
                node.type != bf16
                or node.effect != "pure"
                or node.attrs != BF16_UNARY_ATTRS[node.op]
                or node.index_maps != pointwise_identity_maps(2, 1)
                or len(node.inputs) != 1
                or nodes[node.inputs[0]].op not in allowed_sources
                or nodes[node.inputs[0]].type not in {bf16, mxu_type}
            ):
                return False
        elif node.op in BF16_ROW_REDUCTION_ATTRS:
            if (
                node.type != bf16
                or node.effect != "pure"
                or node.attrs != BF16_ROW_REDUCTION_ATTRS[node.op]
                or node.index_maps != _row_reduce_maps()
                or len(node.inputs) != 1
                or nodes[node.inputs[0]].op not in allowed_sources
                or nodes[node.inputs[0]].type not in {bf16, mxu_type}
            ):
                return False
        elif node.op in BF16_BINARY_ATTRS:
            attributes = BF16_BINARY_ATTRS[node.op]
            if (
                node.type != bf16
                or node.effect != "pure"
                or node.attrs != attributes
                or node.index_maps != pointwise_identity_maps(2, 2)
                or len(node.inputs) != 2
                or node.inputs[0] == node.inputs[1]
            ):
                return False
            for source_id in node.inputs:
                source = nodes[source_id]
                if source.op not in allowed_sources or source.type not in {
                    bf16,
                    mxu_type,
                }:
                    return False
                if source.op in {"input", "constant"} and source.attrs != (
                    attributes[0],
                ):
                    return False
        else:
            return False
    if any(
        nodes[output].op
        not in {
            "matmul",
            "relu",
            *BF16_UNARY_ATTRS,
            *BF16_ROW_REDUCTION_ATTRS,
            *BF16_BINARY_ATTRS,
        }
        for output in request.outputs
    ):
        return False
    if any(
        node.op != "mul_bf16" or node.attrs != BF16_MUL_ATTRS
        for constant in request.nodes
        if constant.op == "constant"
        for node in request.nodes
        if constant.id in node.inputs
    ):
        return False
    reachable = set(request.outputs)
    pending = list(request.outputs)
    while pending:
        for source in nodes[pending.pop()].inputs:
            if source not in reachable:
                reachable.add(source)
                pending.append(source)
    return reachable == set(nodes)


def _transpose_maps() -> tuple[IndexMap, ...]:
    return (
        IndexMap(2, ((0, 1), (1, 0)), (0, 0)),
        IndexMap(2, ((1, 0), (0, 1)), (0, 0)),
    )


def _row_reduce_maps() -> tuple[IndexMap, ...]:
    # Logical loops (row, output_column, reduction_column) broadcast each row result.
    return (
        IndexMap(3, ((1, 0, 0), (0, 0, 1)), (0, 0)),
        IndexMap(3, ((1, 0, 0), (0, 1, 0)), (0, 0)),
    )


def _mxu0_maps() -> tuple[IndexMap, ...]:
    # A[M,K], W[N,K], C[M,N]; the weight orientation is observable.
    return (
        IndexMap(3, ((1, 0, 0), (0, 0, 1)), (0, 0)),
        IndexMap(3, ((0, 1, 0), (0, 0, 1)), (0, 0)),
        IndexMap(3, ((1, 0, 0), (0, 1, 0)), (0, 0)),
    )


def _mxu0_accumulate_maps() -> tuple[IndexMap, ...]:
    maps = _mxu0_maps()
    return (maps[0], maps[1], maps[2], maps[2])


def _mxu0_k_chain_request(request: KernelRequest) -> bool:
    """An ordered MXU0 reduction bounded by declared external storage."""
    if (
        len(request.nodes) % 3
        or len(request.nodes) // 3 < 2
        or 2 * (len(request.nodes) // 3) + 1 > DRAM_SLOTS
        or request.outputs != (request.nodes[-1].id,)
    ):
        return False
    tile_count = len(request.nodes) // 3
    inputs = request.nodes[: 2 * tile_count]
    reductions = request.nodes[2 * tile_count :]
    return all(
        node.op == "input"
        and node.effect == "input"
        and node.type == TensorType((32, 32), "fp8_e4m3", "raw-bit-pattern")
        and node.attrs == (("input_domain", MXU0_INPUT_DOMAIN),)
        and not node.index_maps
        for node in inputs
    ) and all(
        node.op == ("matmul" if index == 0 else "matmul_accumulate")
        and node.effect == "pure"
        and not node.attrs
        and node.inputs
        == (
            (inputs[0].id, inputs[1].id)
            if index == 0
            else (
                inputs[2 * index].id,
                inputs[2 * index + 1].id,
                reductions[index - 1].id,
            )
        )
        and node.type == TensorType((32, 32), "bf16", MXU0_POLICY)
        and node.index_maps == (_mxu0_maps() if index == 0 else _mxu0_accumulate_maps())
        for index, node in enumerate(reductions)
    )
