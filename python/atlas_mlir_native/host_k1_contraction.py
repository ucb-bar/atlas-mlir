"""Prove dense K=1 source contractions and reuse Merlin's exact f32 multiply.

The source accumulator is positive zero. Its addition is observable for a
negative zero product, so the scalar multiply is followed by a zero-sign fix.
No contraction is relabeled as a pointwise operation in the source graph.
"""

from dataclasses import dataclass
from hashlib import sha256
from math import prod

from merlin.compile.c_float_arithmetic import emit_float_arithmetic_c
from merlin.compile.c_sources import c_source_pair
from merlin.semantic_compiler.model import IndexMap, KernelRequest, SemanticNode, TensorType, pointwise_identity_maps


@dataclass(frozen=True)
class K1Plan:
    source: KernelRequest
    body: KernelRequest
    operands: tuple[str, ...]
    batch_count: int
    m: int
    n: int

    def record(self):
        result = self.source.node(self.source.outputs[0])
        return {
            "schema": "atlas.exact_host_k1_contraction.v1",
            "source_request_digest": self.source.digest(),
            "scalar_request_digest": self.body.digest(),
            "source_operation": result.op,
            "source_inputs": list(result.inputs),
            "source_index_maps": [value.record() for value in result.index_maps],
            "source_types": {node.id: node.type.record() for node in self.source.nodes},
            "source_arithmetic": dict(result.attrs),
            "batch_count": self.batch_count, "m": self.m, "k": 1, "n": self.n,
            "element_order": "canonical-dense-row-major",
            "multiplication": "ieee754-f32-mul-rne-v1",
            "accumulator": "exact-positive-zero-f32",
            "addition": "f32-rne-add-positive-zero; map either product zero sign to positive zero",
            "domain": "non-NaN operands, excluding zero times infinity; reject before output writes",
            "precision_change": "none",
        }


def prepare_source(source):
    pure = [node for node in source.nodes if node.effect == "pure"]
    if len(pure) != 1:
        raise ValueError("K=1 contraction needs one complete source computation")
    result = pure[0]
    arithmetic = dict(result.attrs)
    order = arithmetic.pop("scalar_add_order", None)
    if (result.op != "source_f32_matmul_unqualified" or len(result.inputs) != 3
            or source.outputs != (result.id,) or source.lowering_policy != "diagnostic"
            or source.physical_extents or source.boundary_layouts or source.output_layouts
            or order not in {None, "accumulator-product", "product-accumulator"}
            or arithmetic != {"numeric_semantics": "source-f32-mul-add-unqualified-v1", "reduction_order": "unresolved"}):
        raise ValueError("K=1 contraction source arithmetic or dense boundary differs")
    lhs, rhs, seed = (source.node(key) for key in result.inputs)
    rank = len(result.type.shape)
    if (rank < 2 or any(node.type.dtype != "f32" or node.type.numerical_policy != "raw-bit-pattern"
                        or len(node.type.shape) != rank for node in (lhs, rhs, seed, result))
            or any(node.effect not in {"input", "constant"} or node.inputs or node.attrs or node.index_maps
                   for node in (lhs, rhs))
            or {node.id for node in source.nodes} != {*result.inputs, result.id}
            or seed.id in {lhs.id, rhs.id}):
        raise ValueError("K=1 contraction needs complete dense raw f32 source tensors")
    batch = result.type.shape[:-2]
    m, n = result.type.shape[-2:]
    if lhs.type.shape != (*batch, m, 1) or rhs.type.shape != (*batch, 1, n) or seed.type != result.type:
        raise ValueError("K=1 contraction dimensions or batch prefixes differ")
    loops = rank + 1
    axes = tuple(tuple(int(i == j) for j in range(loops)) for i in range(loops))
    prefix = axes[:rank - 2]
    maps = tuple(IndexMap(loops, (*prefix, *rows), (0,) * rank) for rows in (
        (axes[-3], axes[-1]), (axes[-1], axes[-2]), (axes[-3], axes[-2])))
    if result.index_maps != (*maps, maps[-1]):
        raise ValueError("K=1 contraction maps change canonical matrix or batch indexing")
    binding = next((item for item in source.constants if item.node_id == seed.id), None)
    if (seed.op != "constant" or seed.effect != "constant" or seed.inputs or seed.attrs or seed.index_maps
            or binding is None or binding.encoding != "f32-le-raw" or any(bytes.fromhex(binding.data_hex))):
        raise ValueError("K=1 contraction requires an exact positive-zero seed")
    if max(prod(node.type.shape) * 4 for node in source.nodes) > 2**63 - 1:
        raise ValueError("K=1 contraction exceeds its signed byte-size domain")
    scalar = TensorType((1,), "f32", "raw-bit-pattern")
    body = KernelRequest((
        SemanticNode("lhs", "input", (), scalar, effect="input"),
        SemanticNode("rhs", "input", (), scalar, effect="input"),
        SemanticNode("product", "f32_mul", ("lhs", "rhs"), scalar,
                     (("numeric_semantics", "ieee754-f32-mul-rne-v1"),),
                     index_maps=pointwise_identity_maps(1, 2)),
    ), ("product",), ("external",), (("lhs", "external"), ("rhs", "external")),
        source.target_identity, "diagnostic", source.digest() + ":exact-k1-scalar")
    return K1Plan(source, body, tuple(dict.fromkeys((lhs.id, rhs.id))), prod(batch), m, n)


def emit_c(source, name):
    plan = prepare_source(source)
    # Keep the inner name within Merlin's C identifier length domain.
    inner = "k1_" + (name if len(name) <= 61 else sha256(name.encode()).hexdigest()[:32])
    scalar_record, scalar_files = emit_float_arithmetic_c(plan.body, inner)
    result = source.node(source.outputs[0])
    left, right = (plan.operands.index(key) for key in result.inputs[:2])
    sizes = [prod(source.node(key).type.shape) * 4 for key in plan.operands]
    output_bytes = plan.batch_count * plan.m * plan.n * 4
    checks = " || ".join(
        f"!inputs[{i}] || (uintptr_t)inputs[{i}]>UINTPTR_MAX-{size}ULL || "
        f"((uintptr_t)inputs[{i}]<out+{output_bytes}ULL && out<(uintptr_t)inputs[{i}]+{size}ULL)"
        for i, size in enumerate(sizes))
    wrapper = f"""#include <stddef.h>
#include <stdint.h>
#include <limits.h>
_Static_assert(CHAR_BIT==8 && SIZE_MAX>={output_bytes}ULL,"K=1 host types");
int {name}(void *context,const void *const *inputs,void *const *outputs) {{
  (void)context;
  if (!inputs || !outputs || !outputs[0]) return -1;
  uintptr_t out=(uintptr_t)outputs[0];
  if (out>UINTPTR_MAX-{output_bytes}ULL || {checks}) return -1;
  /* Validate every scalar before publishing any application output. */
  for (unsigned pass=0;pass<2;++pass)
    for (size_t batch=0;batch<{plan.batch_count}ULL;++batch)
      for (size_t m=0;m<{plan.m}ULL;++m)
        for (size_t n=0;n<{plan.n}ULL;++n) {{
          unsigned char scratch[4];
          const void *scalar_inputs[2]={{
            (const unsigned char *)inputs[{left}]+4u*(batch*{plan.m}ULL+m),
            (const unsigned char *)inputs[{right}]+4u*(batch*{plan.n}ULL+n)}};
          void *scalar_outputs[1]={{scratch}};
          int status={inner}(0,scalar_inputs,scalar_outputs);
          if (status) return status;
          if (pass) {{
            /* +0 plus -0 under RNE is +0, including multiply underflow. */
            if (!(scratch[0]|scratch[1]|scratch[2]|(scratch[3]&127u))) scratch[3]=0;
            size_t offset=4u*((batch*{plan.m}ULL+m)*{plan.n}ULL+n);
            for (unsigned byte=0;byte<4;++byte) ((unsigned char *)outputs[0])[offset+byte]=scratch[byte];
          }}
        }}
  return 0;
}}
"""
    files = c_source_pair(name, public_headers={}, private_headers={inner + ".h": scalar_files[inner + ".h"]},
                          sources={inner + ".c": scalar_files[inner + ".c"], "k1_contraction.c": wrapper},
                          declarations=f"int {name}(void *, const void *const *, void *const *);\n")
    return {
        "source_request_digest": source.digest(), "implementation": name,
        "inputs": list(plan.operands), "outputs": list(source.outputs),
        "arithmetic": "exact-source-f32-k1-multiply-add-positive-zero",
        "normalization": plan.record(), "scalar_arithmetic": scalar_record,
    }, files
