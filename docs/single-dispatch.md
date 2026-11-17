# Toward one dispatch per decoder/vision layer

The current verified full-layer recipes run TinyLlama B1/T64 in **55 launches,
69,857,257 device cycles**, and SmolVLA vision B1/T1024 in **54 launches,
367,502,579 cycles**. These are Rust/RTL-numerics measurements, excluding host
launch latency. Every final and retained intermediate word matches. See
[the measurement history](model-traffic-priorities.md) for settings, artifacts
and the two retained, unresolved outliers on unchanged artifacts.

One dispatch means one actual NPU launch/completion for the layer. A host wrapper
that submits many programs does not meet that goal.

## Why the existing kernels cannot simply be concatenated

The latest recipes contain 35/26 distinct images totaling 187,424/161,816 bytes
for TinyLlama/SmolVLA, even after identical binaries are counted once. Physical
IMEM is 131,072 bytes. Every kernel also has a local device frame and a host ABI
that may pack tiles, transform row order, replicate exponent pages or publish
status/results. The next kernel cannot consume those bytes directly unless its
physical layout and lifetime contract agrees.

The software execution budget is separate from IMEM. Selecting
`--execution-instruction-limit 2100000000` removed 36 SmolVLA launches by letting
its existing counted additions cover whole tensors. All native image/resource
checks remain enabled. This does not connect different operations or solve the
combined-image limit.

## Implemented building block: shared address-parameterized stages

The opt-in Atlas pass spelling is:

```text
--compose-atlas-program=share-bodies=true
```

The Python `compose_regions` API also accepts `share_bodies=True`. The default
remains the existing straight-line composition. The caller must first supply
ordered, connected stages with explicit physical buffers; this API does not
infer source equivalence or insert boundary conversions.

Stages with identical operation/control/data flow can share one body while
receiving different DMA addresses. A counted stage index visits the authored
entries in order and exits through one ECALL. Operation kinds, numeric policies,
operand uses, DMA lengths, loop bounds and other constants must remain identical.
Only aligned address literals carried through pointer additions/CFG edges to
DMA can become parameters. A shape change is not admitted as an address change.

Both the live-IR address analysis and the independent captured-CFG replay prove
every incoming address. An unaligned choice cannot hide between aligned minimum
and maximum addresses. Unknown, cyclic or wrapping address expressions remain
subject to the existing refusal/induction proofs. Allocation, initialized reads,
memory permissions, timing and final image size are still checked normally.

Native/Rust tests cover:

- Distinct input/output addresses, repeated invocations and untouched guards.
- Three dependent copies with all intermediates on device in one launch.
- Two matrix stages with ordered K accumulation and fresh state each invocation.
- Mismatched operations, attributes and value uses; literal loop-bound retention.
- Unaligned interior choices, invalid memory windows and code-size refusals.

The two-matrix test shrinks from **2,320 to 1,288 encoded bytes**. This is a
micro-test, not a full-layer code-size result. All 61 distinct machine programs
from the full-layer recipes pass the updated native checks and re-emit identical
binaries. No full-layer dispatch reduction is attributed to stage sharing yet.
The focused regression run passed **57 tests and 174 subtests**; one existing
Python-simulator test was skipped. The new execution checks use Rust with RTL
numerics. Address replay also rejects forged retained source addresses.

## Remaining implementation order and ownership

1. **Connect device buffers across regions.** Merlin owns source dependencies,
   ownership, lifetime allocation and proof that a logical view preserves bytes.
   Atlas supplies the physical boundary layout and padding of each kernel.
   Alias an edge only when those contracts agree; otherwise emit the required
   device-side relayout. Host packing cannot remain between stages of one launch.
2. **Share shape-parameterized kernel bodies.** Extend the constant-bound proofs
   to finite, authenticated stage descriptors before allowing different loop
   bounds/strides in one body. Atlas owns the device control flow and instruction
   proof. Existing exact rounding, operand reuse and numeric refusal contracts
   must survive; model names or observed input values must not select the code.
3. **Compose and size the complete layer.** Merlin selects the complete stage
   graph and Atlas schedules/allocates/emits it. Reject an image that exceeds
   IMEM; do not report a multi-program host sequence as one dispatch. Preserve
   atomic public outputs when any internal stage rejects its input.
4. **Verify both layers.** Compare every output word and retained intermediate,
   measure actual launches/cycles/traffic in Rust, and preserve prior recipes as
   controls. Push only the tested feature, with its limitations recorded.

Development uses `build-shared-dispatch/` so the reference tools in `build/bin`
remain unchanged. Local receipts are under `out/traffic-audit/`:
`dispatch-budget-comparison.json`, `shared-tool-layer-audit.json`,
`program-sharing-final-tests.log`, and the independent native/Rust test module
`test/test_program_sharing.py`.
