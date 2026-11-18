"""Schema-only Atlas fixture: eager execution is deliberately unavailable."""
import torch

_LIB = torch.library.Library("atlas", "FRAGMENT")
_SCHEMAS = (
    'quantize_act(Tensor x, str mode, str reduce, str rule, int? static_exp, str code_dtype, str tag="", int rows=32) -> (Tensor, Tensor)',
    'qlinear(Tensor x_codes, Tensor x_exp, Tensor w_codes, Tensor w_exp, Tensor? bias, str out_dtype, str weight_format) -> Tensor',
    'qconv2d(Tensor x_codes, Tensor x_exp, Tensor w_codes, Tensor w_exp, Tensor? bias, str out_dtype, str weight_format, int batch, int out_h, int out_w) -> Tensor',
    'qk_matmul(Tensor q_codes, Tensor q_exp, Tensor k_codes, Tensor k_exp) -> Tensor',
    'pv_matmul(Tensor p_codes, Tensor p_exp, Tensor vt_codes, Tensor v_exp, int pop_shift=0) -> Tensor',
)
for _schema in _SCHEMAS:
    _name = _schema.split("(")[0]
    if hasattr(torch.ops.atlas, _name):
        continue
    _LIB.define(_schema)
    if _name == "quantize_act":
        def _fake(x, mode, reduce, rule, static_exp, code_dtype, tag="", rows=32):
            shape = [] if mode == "static" or reduce == "tensor" else [*x.shape[:-1], 1]
            return (torch.empty_like(x, dtype=getattr(torch, code_dtype)),
                    x.new_empty(shape, dtype=torch.int32))
    elif _name == "qlinear":
        def _fake(x_codes, x_exp, w_codes, w_exp, bias, out_dtype, weight_format):
            return x_codes.new_empty([*x_codes.shape[:-1], w_codes.shape[0]], dtype=getattr(torch, out_dtype))
    elif _name == "qconv2d":
        def _fake(x_codes, x_exp, w_codes, w_exp, bias, out_dtype, weight_format, batch, out_h, out_w):
            return x_codes.new_empty([batch, w_codes.shape[0], out_h, out_w], dtype=getattr(torch, out_dtype))
    else:
        def _fake(a, ae, b, be, pop_shift=0):
            return a.new_empty([*torch.broadcast_shapes(a.shape[:-2], b.shape[:-2]), a.shape[-2], b.shape[-2]], dtype=torch.float32)
    torch.library.register_fake(f"atlas::{_name}")(_fake)
