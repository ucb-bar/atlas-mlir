"""Atlas source semantics as typed operations, never executed by the importer.

Codes use signless i8 MLIR storage with code_dtype=uint8. Exponents use i32
storage with exponent_dtype=int32; they are data, not integer scale attributes.
"""
from xdsl.dialects.builtin import (BFloat16Type, Float16Type, Float32Type, Float64Type,
                                   IntegerType, StringAttr, TensorType)
from xdsl.ir import Dialect
from xdsl.irdl import (IRDLOperation, irdl_op_definition, operand_def,
                       opt_operand_def, result_def, traits_def)
from xdsl.traits import Pure
from xdsl.utils.exceptions import VerifyException


def _tensor(value, bits, label):
    typ = value.type
    if not (isinstance(typ, TensorType) and isinstance(typ.element_type, IntegerType)
            and typ.element_type.width.data == bits):
        raise VerifyException(f"{label} must be a tensor of i{bits}")
    return typ


def _metadata(op):
    for key, expected in (("code_dtype", "uint8"), ("exponent_dtype", "int32"),
                          ("weight_format", "atlas_e4m3")):
        if op.attributes.get(key) != StringAttr(expected):
            raise VerifyException(f"{op.name}: {key} must be {expected}")


@irdl_op_definition
class QuantizeActOp(IRDLOperation):
    name = "atlas.quantize_act"
    input = operand_def(TensorType)
    codes = result_def(TensorType)
    exponent = result_def(TensorType)
    traits = traits_def(Pure())

    def verify_(self):
        _metadata(self)
        codes = _tensor(self.codes, 8, "codes")
        _tensor(self.exponent, 32, "exponent")
        if codes.get_shape() != self.input.type.get_shape():
            raise VerifyException("quantize_act codes must preserve input shape")


def _verify_contraction(op):
    _metadata(op)
    lhs = _tensor(op.lhs_codes, 8, "lhs_codes")
    rhs = _tensor(op.rhs_codes, 8, "rhs_codes")
    _tensor(op.lhs_exp, 32, "lhs_exp")
    _tensor(op.rhs_exp, 32, "rhs_exp")
    if not lhs.get_shape() or not rhs.get_shape() or lhs.get_shape()[-1] != rhs.get_shape()[-1]:
        raise VerifyException("Atlas contraction reduction dimensions must match")
    if not isinstance(op.result.type.element_type, (BFloat16Type, Float16Type, Float32Type, Float64Type)):
        raise VerifyException("Atlas contraction result must be floating point")


@irdl_op_definition
class QLinearOp(IRDLOperation):
    name = "atlas.qlinear"
    lhs_codes = operand_def(TensorType)
    lhs_exp = operand_def(TensorType)
    rhs_codes = operand_def(TensorType)
    rhs_exp = operand_def(TensorType)
    bias = opt_operand_def(TensorType)
    result = result_def(TensorType)
    traits = traits_def(Pure())
    verify_ = _verify_contraction


@irdl_op_definition
class QKMatmulOp(IRDLOperation):
    name = "atlas.qk_matmul"
    lhs_codes = operand_def(TensorType)
    lhs_exp = operand_def(TensorType)
    rhs_codes = operand_def(TensorType)
    rhs_exp = operand_def(TensorType)
    result = result_def(TensorType)
    traits = traits_def(Pure())
    verify_ = _verify_contraction


@irdl_op_definition
class PVMatmulOp(IRDLOperation):
    name = "atlas.pv_matmul"
    lhs_codes = operand_def(TensorType)
    lhs_exp = operand_def(TensorType)
    rhs_codes = operand_def(TensorType)
    rhs_exp = operand_def(TensorType)
    result = result_def(TensorType)
    traits = traits_def(Pure())
    verify_ = _verify_contraction


Atlas = Dialect("atlas", [QuantizeActOp, QLinearOp, QKMatmulOp, PVMatmulOp], [])
