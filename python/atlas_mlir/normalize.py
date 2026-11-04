"""Normalize torch-mlir scalar constants and row reductions for the Linalg bridge.

These are source arithmetic preserving rewrites, before any FP8/BF16 policy.
Unrecognized bodies and live argmax indices are left for the importer to reject.
"""

from __future__ import annotations

import io
import struct

from xdsl.context import Context
from xdsl.dialects import affine, arith, builtin, func, linalg, tensor
from xdsl.dialects import math as math_dialect
from xdsl.dialects.linalg.ops import FillOp, GenericOp, ReduceOp, YieldOp
from xdsl.ir import Block, Region
from xdsl.ir.affine import AffineMap, AffineExpr
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.rewriter import Rewriter
from xdsl.transforms.dead_code_elimination import region_dce


def normalize_linalg(source: str) -> str:
    context = Context()
    for dialect in (
        affine.Affine,
        arith.Arith,
        builtin.Builtin,
        func.Func,
        linalg.Linalg,
        tensor.Tensor,
        math_dialect.Math,
    ):
        context.load_dialect(dialect)
    module = Parser(context, source).parse_module()
    for op in list(module.walk()):
        if isinstance(op, arith.TruncFOp) and op.result.type == builtin.f32:
            producer = op.input.owner
            if not isinstance(producer, arith.ConstantOp) or not isinstance(
                producer.value, builtin.FloatAttr
            ):
                continue
            try:
                value = struct.unpack(
                    "<f", struct.pack("<f", producer.value.value.data)
                )[0]
            except OverflowError:
                # Leave out-of-range literals to the compiler's float semantics.
                continue
            constant = arith.ConstantOp(builtin.FloatAttr(value, builtin.f32))
            # Hoist this literal out of the scalar region, as the bridge expects.
            parent = op.parent_op()
            if not isinstance(parent, GenericOp):
                continue
            parent.parent.insert_op_before(constant, parent)
            Rewriter.replace_op(op, [], [constant.result])
        if not isinstance(op, GenericOp) or len(op.inputs) != 1:
            continue
        if [x.data.value for x in op.iterator_types] != ["parallel", "reduction"]:
            continue
        maps = [x.data for x in op.indexing_maps]
        row = AffineMap(2, 0, (AffineExpr.dimension(0),))
        keep = AffineMap(2, 0, (AffineExpr.dimension(0), AffineExpr.constant(0)))
        if maps[0] != AffineMap.identity(2) or any(
            m not in (row, keep) for m in maps[1:]
        ):
            continue
        block = op.body.block
        terminator = block.last_op
        if not isinstance(terminator, YieldOp) or not terminator.operands:
            continue
        scalar = terminator.operands[0].owner
        if not isinstance(scalar, (arith.MaximumfOp, arith.AddfOp)):
            continue
        if tuple(scalar.operands) != tuple(block.args[:2]):
            continue
        # Only drop auxiliary results when nobody consumes them, and all body
        # operations are pure scalar arithmetic (the torch-mlir argmax body).
        if any(result.uses for result in op.results[1:]):
            continue
        allowed = {
            "linalg.index",
            "arith.index_cast",
            "arith.maximumf",
            "arith.addf",
            "arith.cmpf",
            "arith.select",
            "linalg.yield",
        }
        if any(child.name not in allowed for child in block.ops):
            continue
        shape = op.inputs[0].type.get_shape()
        result_type = op.results[0].type
        if (
            len(shape) != 2
            or shape[0] < 1
            or shape[1] < 1
            or result_type.element_type != builtin.f32
        ):
            continue
        keep_dim = result_type.get_shape() == (shape[0], 1)
        if not keep_dim and result_type.get_shape() != (shape[0],):
            continue
        init = op.outputs[0]
        new_ops = []
        if keep_dim:
            fill = init.owner
            if not isinstance(fill, FillOp):
                continue
            empty = tensor.EmptyOp([], builtin.TensorType(builtin.f32, (shape[0],)))
            fill1 = FillOp([fill.inputs[0]], [empty.tensor])
            new_ops.extend([empty, fill1])
            init = fill1.results[0]
        body = Block(arg_types=[builtin.f32, builtin.f32])
        combine = scalar.clone(value_mapper=dict(zip(block.args[:2], body.args)))
        body.add_ops([combine, YieldOp(combine.result)])
        reduced = ReduceOp(
            op.inputs[0],
            init,
            builtin.DenseArrayBase.from_list(builtin.i64, [1]),
            Region(body),
        )
        new_ops.append(reduced)
        result = reduced.results[0]
        if keep_dim:
            expand = tensor.ExpandShapeOp(
                result,
                [],
                builtin.ArrayAttr(
                    [
                        builtin.ArrayAttr(
                            [
                                builtin.IntegerAttr(0, builtin.i64),
                                builtin.IntegerAttr(1, builtin.i64),
                            ]
                        )
                    ]
                ),
                [shape[0], 1],
                result_type,
            )
            new_ops.append(expand)
            result = expand.result
        Rewriter.replace_op(op, new_ops, [result, *([None] * (len(op.results) - 1))])
    # Make scalar broadcasting explicit as a tensor fill and a binary generic.
    # Merlin's graph boundary can then account for the literal as a constant.
    for op in list(module.walk()):
        if not isinstance(op, GenericOp) or len(op.inputs) != 1 or len(op.outputs) != 1:
            continue
        if any(x.data.value != "parallel" for x in op.iterator_types):
            continue
        body_ops = list(op.body.block.ops)
        # torch-mlir captures sigmoid with a closed-over +1. Merlin recognizes
        # the same finite-input expression with a local literal, 1 + exp(-x).
        if len(body_ops) == 5:
            neg, exp, add, div, end = body_ops
            if (
                isinstance(neg, arith.NegfOp)
                and isinstance(exp, math_dialect.ExpOp)
                and isinstance(add, arith.AddfOp)
                and isinstance(div, arith.DivfOp)
                and isinstance(end, YieldOp)
                and neg.operand == op.body.block.args[0]
                and exp.operand == neg.result
                and add.lhs == exp.result
                and div.lhs == add.rhs
                and div.rhs == add.result
                and tuple(end.operands) == (div.result,)
            ):
                literal = add.rhs.owner
                if (
                    isinstance(literal, arith.ConstantOp)
                    and isinstance(literal.value, builtin.FloatAttr)
                    and literal.result.type == builtin.f32
                    and literal.value.value.data == 1.0
                ):
                    local = literal.clone()
                    op.body.block.insert_op_before(local, neg)
                    add.operands = (local.result, exp.result)
                    div.operands = (local.result, add.result)
            continue
        if (
            len(body_ops) != 2
            or not isinstance(body_ops[0], arith.MulfOp)
            or not isinstance(body_ops[1], YieldOp)
        ):
            continue
        mul, end = body_ops
        if tuple(end.operands) != (mul.result,) or mul.lhs != op.body.block.args[0]:
            continue
        constant = mul.rhs.owner
        if (
            not isinstance(constant, arith.ConstantOp)
            or constant.result.type != builtin.f32
        ):
            continue
        if any(m.data != AffineMap.identity(2) for m in op.indexing_maps):
            continue
        fill = FillOp([constant.result], [op.outputs[0]])
        body = Block(arg_types=[builtin.f32] * 3)
        product = mul.clone(value_mapper={mul.lhs: body.args[0], mul.rhs: body.args[1]})
        body.add_ops([product, YieldOp(product.result)])
        generic = GenericOp(
            [op.inputs[0], fill.results[0]],
            list(op.outputs),
            Region(body),
            [op.indexing_maps.data[0]] * 3,
            op.iterator_types,
            [op.results[0].type],
        )
        Rewriter.replace_op(op, [fill, generic])
    for operation in list(module.walk()):
        if isinstance(operation, func.FuncOp):
            region_dce(operation.body)
            # Unused tensor fills cannot affect observable memory, although
            # xDSL conservatively models all linalg.fill operations as writes.
            for child in reversed(list(operation.body.block.ops)):
                if (
                    isinstance(child, (FillOp, tensor.EmptyOp, arith.ConstantOp))
                    and child.results
                    and not any(r.uses for r in child.results)
                ):
                    if isinstance(child, FillOp) and not all(
                        isinstance(v.type, builtin.TensorType) for v in child.outputs
                    ):
                        continue
                    Rewriter.erase_op(child)
    module.verify()
    output = io.StringIO()
    Printer(stream=output).print_op(module)
    return output.getvalue() + "\n"
