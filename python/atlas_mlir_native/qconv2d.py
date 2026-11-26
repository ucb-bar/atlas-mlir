"""Lower the native unfolded-code convolution ABI through unchanged qlinear.

The exported prefix owns im2col and activation quantization. This operation
consumes their codes, not image pixels, and restores NCHW after contraction.
"""
from copy import deepcopy
import json

from xdsl.dialects.builtin import ArrayAttr, DenseArrayBase, IntegerAttr, StringAttr, TensorType, i64
from xdsl.dialects.func import CallOp
from xdsl.dialects.linalg.ops import TransposeOp
from xdsl.dialects.tensor import EmptyOp, ExpandShapeOp


def lower(op, call, digest, buffers, typed):
    from .source_normalize import AtlasNormalizeError

    literals = call['literals']
    if type(literals.get('groups')) is int and literals['groups'] > 1:
        raise AtlasNormalizeError('qconv2d grouped convolution is unsupported: groups > 1')
    spatial = tuple(literals.get(key) for key in ('batch', 'out_h', 'out_w'))
    expected = {'batch', 'out_h', 'out_w', 'out_dtype', 'weight_format'}
    if 'bias' in literals:
        expected.add('bias')
    if (set(literals) != expected or any(type(n) is not int or n <= 0 for n in spatial)
            or ('bias' in literals and literals['bias'] is not None)):
        raise AtlasNormalizeError('qconv2d requires positive static batch/out_h/out_w and explicit optional bias')
    roles = [r.get('role') for r in call['operands']]
    names = ['x_codes', 'x_exp', 'w_codes', 'w_exp']
    if roles not in (names, names + ['bias']):
        raise AtlasNormalizeError('qconv2d requires ordered unfolded code/exponent pairs and optional bias')
    x, _, w, *_ = op.operands
    batch, height, width = spatial
    xs, ws = x.type.get_shape(), w.type.get_shape()
    if len(xs) == 2 and len(ws) == 2 and xs[1] != ws[1]:
        raise AtlasNormalizeError('qconv2d grouped reduction mismatch: unfolded and weight K differ')
    if (len(xs) != 2 or len(ws) != 2 or xs[0] != batch * height * width
            or len(op.results) != 1
            or op.results[0].type.get_shape() != (batch, ws[0], height, width)):
        raise AtlasNormalizeError('qconv2d unfolded rows/rank or NCHW result shape differs')
    element = op.results[0].type.element_type
    flat_type = TensorType(element, [xs[0], ws[0]])
    linear_call = deepcopy(call)
    linear_call['target'] = 'atlas.qlinear.default'
    linear_call['schema'] = 'atlas::qlinear(Tensor x_codes, Tensor x_exp, Tensor w_codes, Tensor w_exp, Tensor? bias, str out_dtype, str weight_format) -> Tensor'
    linear_call['literals'] = {k: v for k, v in literals.items() if k not in {'batch', 'out_h', 'out_w'}}
    linear_call['results'][0]['shape'] = [xs[0], ws[0]]
    proxy = CallOp('qlinear', list(op.operands), [flat_type])
    proxy.attributes.update({k: v for k, v in op.attributes.items() if not k.startswith('m2m.literal.')})
    proxy.attributes['m2m.schema'] = StringAttr(linear_call['schema'])
    proxy.attributes.update({'m2m.literal.' + k: StringAttr(json.dumps(v)) for k, v in linear_call['literals'].items()})
    try:
        linear = typed(proxy, linear_call, digest, buffers)[0]
    finally:
        proxy.drop_all_references()
    # The contraction still lowers the source convolution operation.
    linear.attributes['prov.aten'] = StringAttr(call['target'])
    operations = [linear]
    try:
        shape = [batch, height, width, ws[0]]
        groups = ArrayAttr([ArrayAttr([IntegerAttr(i, i64) for i in group]) for group in ((0, 1, 2), (3,))])
        expanded = ExpandShapeOp(linear.results[0], [], groups, shape, TensorType(element, shape))
        operations.append(expanded)
        empty = EmptyOp([], op.results[0].type)
        operations.append(empty)
        transposed = TransposeOp(expanded.results[0], empty.results[0], DenseArrayBase.from_list(i64, [0, 3, 1, 2]))
        operations.append(transposed)
        # The NCHW restoration lowers the same prepared source node. Its quantized
        # origin stays on the contraction alone, which binds the source operation.
        source = {k: v for k, v in op.attributes.items()
                  if k.startswith('prov.') and k != 'prov.origin_node_ids'}
        for item in operations[1:]:
            item.attributes.update(source)
            item.verify()
        return operations
    except Exception:
        for item in reversed(operations):
            item.drop_all_references()
        raise
