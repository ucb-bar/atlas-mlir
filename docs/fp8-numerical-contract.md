# FP8 numerical contract, first admission milestone

Scope: the generic value/admission boundary in Merlin's refactor and an Atlas
manifest/source-intake adapter, with nonzero tests. No new precision policy, model rule,
weight conversion, compiler pipeline, deployment, or hardware qualification.
The maintained generic contract is in `src/merlin/docs/reference/architecture.md`
under “Scaled FP8 numerical admission”. This note binds its vocabulary to the
read-only Atlas integration source `e41884b67055e68e110824f6813aab106c01806d`.

## Export representation and granularity

`atlas.qlinear`, `atlas.qk_matmul` and `atlas.pv_matmul` consume ordered
`(lhs_codes,lhs_exp,rhs_codes,rhs_exp[,bias])`. The codes are i8 storage of
logical uint8 E4M3FN bytes; exponents are signed i32 values. Logical NT operands
are `[...,M,K]` and `[...,N,K]`, producing `[...,M,N]`. Batch axes broadcast;
scale tensors must broadcast to their respective operand and must not vary
along K. A product's units are restored with `2**(lhs_exp+rhs_exp)`, **not** the
negative sum. A source f32 diagnostic path instead encodes with pre-encode
scales and restores their negative sum; do not conflate the two conventions.

The selected value admission route is unbiased and checks each batch slice as
one rank-two contraction: both operands carry the same batch prefix, or the
right operand is one shared rank-two weight; other broadcasts refuse. It accepts
scalar, per-row, or per-batch exponents, normalizing explicit axes before the
generic value check. It does not claim dynamic quantizer or biased compilation
coverage. Unchanged authored weights and codes remain inputs, not newly encoded
parameters. Export source metadata still requires uint8 code meaning, int32
exponents, NT orientation, zero unfolded pop shift, output dtype and bias order.
See `python/atlas_mlir_native/source_normalize.py`, `source_translation.py`
and `already_quantized.py`. Manifest normalization retains a diagnostic selected
numerical declaration; closed intake compares that declaration and checks every
available immutable constant. Dynamic inputs retain explicit runtime validation
obligations. The original source attributes and byte-bound digest remain separate.
A selected model47 declaration does not prove equivalence to exporter/framework
matmul arithmetic; diagnostic policy is required for this intake route.

Exported codes are decoded in Merlin's `e4m3fn-zero-flush` decoding, which is
atlas-torchao's: exponent field 0 and the codes 0x7f and 0xff are zero with the
code's sign (`fp8_contract.EXPORT_ENCODING`). The name covers code decoding only;
the unpacker's flush and saturation after scaling are covered by the exponent bounds
and the BF16-normal readout rule below. That meaning does not imply MXU admission:
Atlas's selected cap448 route admits only `{0} ∪ [8,126] ∪ [136,254]`;
negative zero, nonzero codes with exponent field 0, 0x7f and 0xff refuse. Operand exponents are
bounded to [-127,127], pair sums to [-133,127]. The sum bound guarantees only a
finite nonzero BF16 scale, not a valid final value. After execution, nonzero
readout and restored result must be finite normal BF16; zero remains allowed.
Result-dependent checks, device status checking and publication of outputs only
on success remain runtime obligations, not discharged by this input helper.

## Activation encoding

Atlas static/dynamic activation packing is distinct from preserving prequantized
weights. f32 activations first round to BF16 using RNE. The packer rounds the
normal significand ties-to-even (including carry), then flushes values below
minimum normal to **positive zero**, and saturates at signed ±448. BF16 zero,
subnormals and NaNs map to positive zero; infinities saturate. This is not an
unrestricted IEEE E4M3 cast and does not preserve zero signs or NaN payloads.

Static quantization returns its authored i32 exponent. Packing clamps only the
shift to [-127,127]; it does not clamp that returned exponent. Consequently a
quantizer with an out-of-range authored exponent is not admitted automatically
by the contraction's stricter exponent domain. Dynamic `ref` uses
`floor(log2(amax))-8` for nonzero finite amax, else zero. `nc` increments that
exponent if the normalized significand exceeds 1.75; subnormal BF16 amax uses
its separately implemented exponent calculation. amax ignores NaN/Inf but
includes finite subnormals. Tensor reduction shares one exponent; token reduces
the last dimension; head reduces the last two; block reduces consecutive rows;
head_block bounds rows within each head. Block exponents repeat per constituent
row. These rules belong to Atlas activation-quantizer families, not to generic
FP8 decoding. See `static_quantize.py`, `dynamic_quantize.py` and
`virtual_quantizer.py`.

## Selected accumulation and ordered epilogue

The selected implementation is MXU1/model47 cap448 anchor arithmetic, not MXU0
and not an exact floating dot. Preserve the implementation's integer anchor,
truncating alignment, 32-bit wrapping accumulation, short-sum normalization,
ascending K32 steps, reset followed by BF16-seeded continuation, positive-zero
inactive lanes, and BF16 RNE readout after **each instruction**. K64 therefore
retains the K32 rounding boundary. Input decoding or nonzero admission tests
alone do not verify that accumulator. See Atlas `binding.py`'s
`MXU1_CAP448_POLICY`, `already_quantized.py` and the selected-model
analysis. The model identity must remain in the admission declaration.

The following equations specify one output lane of the selected model47 route
(`reference.py:mxu1_cap448_lane`, attributed there to unmodified npu_model
`b97b2f5a97e432141c4ef8504bec396651d77a8f:configs/rtl_math.py:ipt_row`).
They describe the selected contract. Round 4's separate scalar comparisons and
device-region model executions are scoped below; no RTL simulation was run.
For each nonzero FP8 pair with exponent fields `ea, eb`, fractions `fa, fb`:

- `p=(8+fa)*(8+fb)`, `c=p>>7`, `E=ea+eb-14+c`,
  `S=128|((p if c else p<<1)&127)`, with sign XOR of the input signs.
- Include the previous BF16 partial's unbiased exponent in the anchor maximum
  if it is nonzero. `A=max(product E, seed exponent)+7`.
  Each product contributes signed `((S<<24)>>(A-E))`: alignment truncates
  magnitude before applying sign. Products with either operand +0 contribute 0.
- For BF16 seed exponent field `pe` and fraction `pm`, let
  `q=A-(pe-127)-24`, `T=(128 if pe else 0)|pm`.
  Its aligned magnitude is 0 if `q>=32` or `q<-31`, `T>>q` if `q>=0`,
  otherwise `(T<<(-q))&0xffffffff`. Apply its sign and add to the products.
  Wrap the sum to signed i32. A zero sum produces +0.
- For nonzero signed sum `z`, `L=floor(log2(abs(z)))`, `d=max(0,L-7)`.
  Keep the top eight bits and the discarded remainder; normalize sums shorter
  than eight bits by left shift `max(0,7-L)`. Increment the kept significand
  when remainder exceeds halfway, or equals halfway and the kept LSB is odd.
  A carry to 256 shifts once and increments `L`. Output BF16 exponent field
  is `L+A-31+127`. Exponents <=0 flush to +0; >=255 saturate to signed
  maximum finite BF16 (0x7f7f/0xff7f); otherwise emit sign, exponent and fraction.

Each ascending K32 instruction uses this result as the next seed; reset uses
+0. Padding contributes +0. This is distinct from summing all decoded products
exactly and rounding once. For example, 1 plus a 1/256 contribution in each of
two instructions rounds at each boundary to 1 (0x3f80), whereas rounding their
combined exact value once gives 1+1/128 (0x3f81). This is a contract witness,
not a result obtained from a device. A source ordinary float matmul must not
be relabeled with this selected arithmetic without explicit authorization.

The ordered sequence is contraction/readout, exponent restoration, output cast,
source bias addition if present, then source activation if present. f32 output
is exact widening of restored BF16, not f32 accumulation. The source BF16 bias
is `RNE_BF16(cast_output(restored_readout)+bias)`. Primitive Atlas VPU addition
rounds in FP32 then chops to BF16. For nonzero operands `1 + 3/512`, RNE yields
BF16 `0x3f81`, chopping yields `0x3f80`. For f32 output the source bias add
uses f32 RNE after the output cast; a BF16 chop is not that contract either.
MXU truncated alignment/wrap and its
per-instruction RNE are a third, distinct numerical contract. Bias as an MXU
seed, pre-scale bias, primitive VPU replacement, or reordered activation cannot
be silently fused. An exact corrected device RNE implementation can be a
separate admitted family with its domain/status checks; a mode name containing
“rne” does not qualify the primitive. Invalid infinity operations and NaNs must
retain source-specific refusal/handling, not inherit quantizer behavior.

## ReLU coordination

The selected capture revision is model2MLIR `d4bcb8411d2fd572f482911f0f4600aae693f5a1`.
The reviewed Merlin contract `fd0b9a6e527c37e98c819498d8076fe3ca45c747` is
ported to the current frontend and semantic source owners.
Its ordered BF16 `select(x < +0, +0, x)` preserves -0 and unordered NaN bit
patterns. The sign-clamp RTL table maps negative zero and negative NaNs to +0.
They differ even if finite positive/negative samples agree. Generic contract
comparison refuses replacing one form with the other; no consumer-based zero
elision proof is assumed. `maximum`/`maxnum` variants also require their own
NaN and zero semantics; they cannot be inferred from the label “ReLU”. Current
Atlas f32 precision/fused paths remain unqualified for unrestricted source
zero/NaN preservation.

## Replayable source bias separation (Round 3)

The adapter now stores the versioned generic contraction record in
`numerical.contraction`. This replaces `atlas.diagnostic_contraction_numerics`;
old records must be regenerated from the unchanged authored source. Unknown or
stale numerical fields are refused, and no old snapshot identity is reused.
The record remains a diagnostic selected route, not source arithmetic equality.

Generic `compiler/semantic/prequantized_bias.py` separates a BF16 column bias
before indexing/constant preparation: a four-operand `.readout`, exact column
`gather` `.bias`, then original-id `bf16_add` declaring
`ieee754-bf16-add-rne-v1`. The readout retains accumulation, restoration and cast
stages and removes only the final source bias stage. The splitter requires
BF16 RNE, no fused activation and a unique final source bias stage when numerical
metadata is present. f32 bias remains unsplit/unqualified. Constants are neither
repacked nor changed. Legacy generic contractions without a numerical record
can retain their structural contract; Atlas intake still requires its complete
record and original metadata.

The split receipt records original nodes and before/after digests. Verification
restores and exactly replays the transformation; structural recognition alone
is not origin authentication. Snapshot source inventory includes the new owner.
Nonzero K32/K64 tests run generic preparation/replay then existing Atlas closed
readout admission. They preserve code, signed exponent and bias bytes and reject
stale biased metadata and altered RNE addition. These are graph/admission tests:
no host emitter, MXU evaluation or application compilation was executed here.

## Application family binding (Round 4)

`compile_semantic_application` now admits fresh nonzero K32/K64 prepared sources
with a replayable bias receipt, through the generic build owner. The selected
sequence is an unbiased ordered MXU readout, exact column gather, original-id
host BF16 RNE addition, then ordered BF16 select. Verification requires the exact
saved preparation receipt and source identities; missing or changed receipts
refuse. This is compilation and source replay, not instruction execution.

The Atlas contraction lowering recipe carries the canonical generic
`numerical.contraction` record in its guards. Changing accumulation, bias or
activation refuses Merlin equality admission. BF16 and f32 output records differ
at the cast/widen boundary. The `host.ordered-select` family of Merlin's host
target is the only host ReLU route; it accepts all BF16 bit patterns and
preserves -0 and NaN payloads with a byte predicate. A sign-clamp declaration
has no route and is refused. It does not authorize device sign clamp.
Host family recipes pin the host target's emitter sources, including the actual
host_ops directory; Atlas recipes pin Atlas's sources and the host target's digest.

The numerical tests execute emitted ordered-select C over all 65,536 BF16
encodings and emitted host RNE addition on a nonzero tie-sensitive witness.
The runtime guard harness injects independently evaluated MXU scalar readouts
through callbacks, checks success and failure publication, and kills deliberate
guard/status/publication mutations. Callback injection does not execute device
instructions. Separate pinned original47 scalar comparisons authenticate imported
model source bytes and compare every K32 partial for the two nonzero applications.
Neither result establishes equivalence to unrestricted framework matmul, native
instruction execution, RTL or hardware. Deployment/placement and policy generation
remain their owning streams' later gates.

Separate device-region instruction simulation now runs the actual emitted
K32/K64 binaries on pinned npu_model b97b2f5 with numerics='rtl', matching the
entire guarded arena on admitted nonzero inputs. The model DRAM window base is
configured to match compiled addresses. This is a functional numerical model,
not fresh RTL simulation or hardware. Executed activation mutations change the
outputs. Individual exponent and restored-overflow refusals set status1 and may
write device staging output. Atomic public publication is the host status guard's
obligation; temporary device output has no unchanged-on-failure promise. The
host callback tests and device-region simulations ran separately; no composed
application/device or deployment qualification is inferred.
