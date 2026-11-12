# Source-bound host preparation

The optional `host_preparation_scope` precision-policy field limits selected host
float arithmetic and unary math to preparation. It uses the existing region
selection and alternatives hooks. Supported ancestors must terminate in integer
input boundaries, source constants, or explicitly declared immutable parameter
boundaries; floating activation inputs and unsupported or explicitly scheduled
ancestors do not enter the closure. Supported descendants remain preparation
only while they do not mix unrelated runtime values. Source arithmetic, types,
index maps and broadcast semantics are unchanged.

Without this field, existing global host selection retains its meaning. With
it, arithmetic outside preparation uses the existing device routes. For example,
a broadcast RMSNorm f32 multiply uses the already-selected pointwise mechanism;
it is not passed to the equal-shape host multiply emitter or reinterpreted as a
dense operation. The existing selected unary library/effects policy is required.

An example scope field is:

```json
{
  "host_preparation_scope": {
    "policy": "integer-inputs-and-bound-constants-v1",
    "constant_inputs": [{
      "node_id": "frequencies",
      "type": {"shape": [32], "dtype": "f32", "numerical_policy": "raw-bit-pattern"}
    }]
  }
}
```

Boundary IDs belong to the policy's exact source hash. Planning validates their
IDs and complete types and never infers immutability from an f32 type. Scope
entries declare boundaries; payload authentication remains with immutable
parameter binding. The generated parameter ABI checks payload SHA-256 at bind
and requires immutable lifetime after binding.
