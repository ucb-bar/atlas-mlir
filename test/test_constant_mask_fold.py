"""Fold a select only when declared constants prove one uniform mask."""

from copy import deepcopy
from dataclasses import replace
import struct

import numpy as np
import pytest

from merlin.compiler.semantic.constant_masks import fold_constant_masks, verify_constant_masks
from merlin.compiler.semantic.model import ConstantBinding, IndexMap, KernelRequest, SemanticNode, TensorType
from merlin.compiler.semantic.model import pointwise_identity_maps
from merlin.compiler.semantic.source_preparation import (
    preparation_counts, prepare_source, verify_source_preparation,
)

BOOL, F32, I64, BF16 = "boolean", "raw-bit-pattern", "signed-i64", "raw-bit-pattern"
ON, OFF = 0x3fc0, 0xc0a0
# Over the general folder's element budget, so the select reaches the mask stage.
LARGE = 20000


def constant(key, shape, dtype, policy, encoding, data):
    node = SemanticNode(key, "constant", (), TensorType(shape, dtype, policy), effect="constant")
    return node, ConstantBinding(key, encoding, bytes(data).hex())


def scalar_read(rank):
    return IndexMap(rank, tuple((0,) * rank for _ in range(rank)), (0,) * rank)


def mask_source(truth_bits=0x3f800000, threshold=0, flags=None, size=4, runtime_flag=False):
    """A broadcast truth value, an index comparison, and a flag vector gate two scalar broadcasts."""
    shape = (1, size)
    rows = []
    truth, truth_binding = constant("truth", (1, 1), "f32", F32, "f32-le-raw", struct.pack("<I", truth_bits))
    index, index_binding = constant("index", (1, size), "i64", I64, "i64-le", np.arange(size, dtype="<i8").tobytes())
    limit, limit_binding = constant("limit", (1, 1), "i64", I64, "i64-le", struct.pack("<q", threshold))
    flag_values = [1] * size if flags is None else flags
    flag, flag_binding = constant("flag", shape, "i1", BOOL, "i1-raw", flag_values)
    high, high_binding = constant("high", (), "bf16", BF16, "bf16-le-raw", struct.pack("<H", ON))
    low, low_binding = constant("low", (), "bf16", BF16, "bf16-le-raw", struct.pack("<H", OFF))
    unused, unused_binding = constant("unused", (), "i1", BOOL, "i1-raw", [1])
    bindings = [truth_binding, index_binding, limit_binding, high_binding, low_binding, unused_binding]
    inputs = ()
    if runtime_flag:
        flag = SemanticNode("flag", "input", (), flag.type, effect="input")
        inputs = (("flag", "external"),)
    else:
        bindings.append(flag_binding)
    identity = pointwise_identity_maps(2, 1)[1]
    nonzero = SemanticNode("nonzero", "f32_nonzero", ("truth",), TensorType(shape, "i1", BOOL),
                           (("numeric_semantics", "f32-une-zero-v1"),), index_maps=(scalar_read(2), identity))
    compare = SemanticNode("compare", "mask", ("index", "limit"), TensorType(shape, "i1", BOOL),
                           (("operation", "arith.cmpi"), ("predicate", "sge")),
                           index_maps=(identity, scalar_read(2), identity))
    both = SemanticNode("both", "mask", ("nonzero", "compare"), TensorType(shape, "i1", BOOL),
                        (("operation", "arith.andi"), ("predicate", None)), index_maps=pointwise_identity_maps(2, 2))
    gated = SemanticNode("gated", "mask", ("both", "flag"), TensorType(shape, "i1", BOOL),
                         (("operation", "arith.andi"), ("predicate", None)), index_maps=pointwise_identity_maps(2, 2))
    value = TensorType(shape, "bf16", BF16)
    scalar = IndexMap(2, (), ())
    first = SemanticNode("first", "gather", ("high",), value, index_maps=(scalar, identity))
    second = SemanticNode("second", "gather", ("low",), value, index_maps=(scalar, identity))
    chosen = SemanticNode("chosen", "select", ("gated", "first", "second"), value,
                          index_maps=pointwise_identity_maps(2, 3))
    rows = (truth, index, limit, flag, high, low, unused, nonzero, compare, both, gated, first, second, chosen)
    return KernelRequest(rows, ("chosen",), ("external",), inputs, "test", "diagnostic", constants=tuple(bindings))


def reference(truth_bits, threshold, flags, size=4):
    """Evaluate the original graph directly with NumPy, independently of the fold."""
    truth = (truth_bits & 0x7fffffff) != 0
    condition = truth & (np.arange(size) >= threshold) & np.array(flags, dtype=bool)
    return np.where(condition, ON, OFF).astype("<u2").tobytes()


def broadcast_bytes(source, key):
    node = source.node(key)
    assert node.op == "gather" and source.node(node.inputs[0]).type.shape == ()
    binding = next(row for row in source.constants if row.node_id == node.inputs[0])
    return bytes.fromhex(binding.data_hex) * int(np.prod(node.type.shape))


@pytest.mark.parametrize("truth_bits,threshold,flags", [
    (0x3f800000, 0, [1, 1, 1, 1]),  # all true
    (0x7fc00001, -3, [1, 1, 1, 1]),  # a NaN is nonzero
    (0x80000000, 0, [1, 1, 1, 1]),  # negative zero is false
    (0x3f800000, 4, [1, 1, 1, 1]),  # the comparison is false everywhere
    (0x3f800000, 0, [0, 0, 0, 0]),  # the flag is false everywhere
])
def test_uniform_mask_folds_to_the_reference_bytes(truth_bits, threshold, flags):
    source = mask_source(truth_bits, threshold, flags)
    folded, receipt = fold_constant_masks(source)
    assert broadcast_bytes(folded, "chosen") == reference(truth_bits, threshold, flags)
    assert not [node for node in folded.nodes if node.type.dtype == "i1"]
    assert folded.outputs == source.outputs
    assert verify_constant_masks(folded, receipt).digest() == source.digest()


def test_source_preparation_replays_the_mask_stage():
    source = mask_source(size=LARGE)
    prepared, receipt = prepare_source(source)
    assert receipt["schema"] == "merlin.source_preparation.v7"
    assert broadcast_bytes(prepared, "chosen") == reference(0x3f800000, 0, [1] * LARGE, LARGE)
    assert not [node for node in prepared.nodes if node.type.dtype == "i1"]
    assert verify_source_preparation(prepared, receipt).digest() == source.digest()
    assert preparation_counts(receipt)["source_constant_mask_folds"] == 1


@pytest.mark.parametrize("kwargs", [
    dict(threshold=2),  # a constant mask that is not uniform
    dict(flags=[1, 0, 1, 1]),
    dict(runtime_flag=True),  # a mask that is not constant
])
def test_other_masks_are_left_for_normal_admission(kwargs):
    source = mask_source(**kwargs)
    folded, receipt = fold_constant_masks(source)
    assert folded.digest() == source.digest() and not receipt["rewrites"]
    assert folded.node("chosen").op == "select"


def with_branch(source, key, *nodes, bindings=(), inputs=()):
    """Replace one branch of the select with new nodes, keeping the rest of the graph."""
    rows = []
    for node in source.nodes:
        if node.id == key:
            rows.extend(nodes)
        else:
            rows.append(node)
    return replace(source, nodes=tuple(rows), constants=(*source.constants, *bindings),
                   input_storages=(*source.input_storages, *((item, "external") for item in inputs)))


def branch_forms():
    """Each form breaks the both-branches scalar-broadcast limit; the uniform mask chooses `first`."""
    value, identity = TensorType((1, 4), "bf16", BF16), pointwise_identity_maps(2, 1)[1]
    dense, dense_binding = constant("second", (1, 4), "bf16", BF16, "bf16-le-raw", struct.pack("<4H", *[OFF] * 4))
    runtime = SemanticNode("second", "input", (), value, effect="input")
    operand = SemanticNode("operand", "input", (), value, effect="input")
    arithmetic = SemanticNode("second", "bf16_add", ("operand", "operand"), value,
                              (("numeric_semantics", "ieee754-bf16-add-rne-v1"),), index_maps=pointwise_identity_maps(2, 2))
    row, row_binding = constant("row", (1, 4), "bf16", BF16, "bf16-le-raw", struct.pack("<4H", *[ON] * 4))
    gathered = SemanticNode("first", "gather", ("row",), value, index_maps=(identity, identity))
    swapped = IndexMap(2, ((0, 1), (1, 0)), (0, 0))
    remapped = SemanticNode("second", "gather", ("low",), value, index_maps=(IndexMap(2, (), ()), swapped))
    live = SemanticNode("live", "input", (), TensorType((), "bf16", BF16), effect="input")
    broadcast = SemanticNode("first", "gather", ("live",), value, index_maps=(IndexMap(2, (), ()), identity))
    negated = replace(mask_source().node("first"), op="bf16_neg")
    return {
        "unchosen dense constant": ("second", (dense,), (dense_binding,), ()),
        "unchosen runtime tensor": ("second", (runtime,), (), ("second",)),
        "unchosen runtime arithmetic": ("second", (operand, arithmetic), (), ("operand",)),
        "unchosen gather with another output map": ("second", (remapped,), (), ()),
        "chosen dense gather": ("first", (row, gathered), (row_binding,), ()),
        "chosen runtime scalar broadcast": ("first", (live, broadcast), (), ("live",)),
        "chosen unary operation of a scalar": ("first", (negated,), (), ()),
    }


@pytest.mark.parametrize("form", list(branch_forms()))
def test_both_branches_must_be_scalar_broadcasts(form):
    key, nodes, bindings, inputs = branch_forms()[form]
    source = with_branch(mask_source(), key, *nodes, bindings=bindings, inputs=inputs)
    folded, receipt = fold_constant_masks(source)
    assert folded.digest() == source.digest() and not receipt["rewrites"]
    assert folded.node("chosen").op == "select"


def transposed_mask(read, a_bits=(1, 0, 1, 0), b_bits=(1, 1, 0, 0)):
    """xor(gather(a), b) with a 2 by 2 transposing read: uniform only in the source element order."""
    shape, identity = (2, 2), pointwise_identity_maps(2, 1)[1]
    a, a_binding = constant("a", shape, "i1", BOOL, "i1-raw", a_bits)
    b, b_binding = constant("b", shape, "i1", BOOL, "i1-raw", b_bits)
    moved = SemanticNode("moved", "gather", ("a",), a.type, index_maps=(read, identity))
    mask = SemanticNode("mask", "mask", ("moved", "b"), a.type, (("operation", "arith.xori"), ("predicate", None)),
                        index_maps=pointwise_identity_maps(2, 2))
    high, high_binding = constant("high", (), "bf16", BF16, "bf16-le-raw", struct.pack("<H", ON))
    low, low_binding = constant("low", (), "bf16", BF16, "bf16-le-raw", struct.pack("<H", OFF))
    value, scalar = TensorType(shape, "bf16", BF16), IndexMap(2, (), ())
    first = SemanticNode("first", "gather", ("high",), value, index_maps=(scalar, identity))
    second = SemanticNode("second", "gather", ("low",), value, index_maps=(scalar, identity))
    chosen = SemanticNode("chosen", "select", ("mask", "first", "second"), value,
                          index_maps=pointwise_identity_maps(2, 3))
    return KernelRequest((a, b, moved, mask, high, low, first, second, chosen), ("chosen",), ("external",), (),
                         "test", "diagnostic", constants=(a_binding, b_binding, high_binding, low_binding))


def test_dense_gather_reads_the_source_element_order():
    # transpose([[1, 0], [1, 0]]) is [[1, 1], [0, 0]], equal to b, so the xor is false everywhere.
    folded, receipt = fold_constant_masks(transposed_mask(IndexMap(2, ((0, 1), (1, 0)), (0, 0))))
    assert len(receipt["rewrites"]) == 1 and broadcast_bytes(folded, "chosen") == struct.pack("<4H", *[OFF] * 4)


def test_out_of_bounds_mask_read_is_left_unfolded():
    # All-false data: a read past the end must refuse, not look like a uniform false mask.
    source = transposed_mask(IndexMap(2, ((0, 1), (1, 0)), (1, 0)), (0, 0, 0, 0), (0, 0, 0, 0))
    folded, receipt = fold_constant_masks(source)
    assert folded.digest() == source.digest() and not receipt["rewrites"]


def test_condition_that_is_not_boolean_is_left_unfolded():
    source = mask_source()
    ones, binding = constant("ones", (1, 4), "i64", I64, "i64-le", np.ones(4, dtype="<i8").tobytes())
    nodes = []
    for node in source.nodes:
        if node.id == "chosen":
            nodes.append(ones)
            node = replace(node, inputs=("ones", *node.inputs[1:]))
        nodes.append(node)
    wired = replace(source, nodes=tuple(nodes), constants=(*source.constants, binding))
    folded, receipt = fold_constant_masks(wired)
    assert folded.digest() == wired.digest() and not receipt["rewrites"]


def test_select_with_another_output_map_is_left_unfolded():
    source = mask_source()
    maps = (*pointwise_identity_maps(2, 3)[:3], IndexMap(2, ((0, 1), (1, 0)), (0, 0)))
    remapped = replace(source, nodes=tuple(replace(n, index_maps=maps) if n.id == "chosen" else n
                                           for n in source.nodes))
    folded, receipt = fold_constant_masks(remapped)
    assert folded.digest() == remapped.digest() and not receipt["rewrites"]


def test_budget_refuses_a_mask_that_would_otherwise_fold():
    # A uniform mask whose index comparison needs dense work: it folds at the default budget only.
    source = mask_source()
    assert len(fold_constant_masks(source)[1]["rewrites"]) == 1
    folded, receipt = fold_constant_masks(source, element_budget=3)
    assert folded.digest() == source.digest() and not receipt["rewrites"]


def test_uniform_broadcast_of_a_large_mask_needs_no_dense_work():
    source = mask_source(size=1024)
    folded, receipt = fold_constant_masks(source, element_budget=8192)
    assert len(receipt["rewrites"]) == 1 and folded.node("chosen").op == "gather"


def test_shared_nodes_stay_when_a_select_folds():
    source = mask_source()
    keep = SemanticNode("keep", "identity", ("first",), source.node("first").type)
    shared = replace(source, nodes=(*source.nodes, keep), outputs=("chosen", "keep"),
                     output_storages=("external", "external"))
    folded, receipt = fold_constant_masks(shared)
    assert folded.node("first").op == "gather" and folded.node("keep").inputs == ("first",)
    removed = {row["node"]["id"] for row in receipt["removed_nodes"]}
    assert not removed & {"first", "high", "keep"} and {"second", "low", "gated"} <= removed


def test_a_boundary_input_of_the_unchosen_branch_is_kept():
    source = mask_source()
    runtime = SemanticNode("runtime", "input", (), source.node("low").type, effect="input")
    nodes = tuple(runtime if n.id == "low" else replace(n, inputs=("runtime",)) if n.id == "second" else n
                  for n in source.nodes)
    wired = replace(source, nodes=nodes, input_storages=(("runtime", "external"),),
                    constants=tuple(c for c in source.constants if c.node_id != "low"))
    folded, receipt = fold_constant_masks(wired)
    assert folded.node("chosen").inputs == ("high",) and folded.node("runtime").effect == "input"
    assert "second" in {row["node"]["id"] for row in receipt["removed_nodes"]}
    assert verify_constant_masks(folded, receipt).digest() == wired.digest()


@pytest.mark.parametrize("field", ["rewrites", "removed_nodes", "removed_constants", "element_budget"])
def test_changed_receipt_is_refused(field):
    prepared, receipt = prepare_source(mask_source(size=LARGE))
    changed = deepcopy(receipt)
    if field == "element_budget":
        # The original graph still rebuilds; only the exact replay can see the change.
        changed["masks"][field] = 0
    elif field == "rewrites":
        row = changed["masks"][field][0]
        row["inputs"][1], row["inputs"][2] = row["inputs"][2], row["inputs"][1]
    elif field == "removed_nodes":
        row = changed["masks"][field][0]
        row["position"] = len(prepared.nodes) + len(changed["masks"]["removed_nodes"])
    else:
        row = changed["masks"][field][0]
        row["data_hex"] = "00" * (len(row["data_hex"]) // 2)
    with pytest.raises(ValueError):
        verify_source_preparation(prepared, changed)
