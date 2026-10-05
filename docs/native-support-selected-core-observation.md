# Bounded execution of Merlin-native Atlas programs

The installed OOT `atlas_tensor` support package generated a 95-descriptor
target snapshot through Merlin's native engine at
`1b7517c022727499f3beb9dba64745e445c710b4`. Four fresh strict-native
compilations emitted Atlas instruction streams. The evaluator-only
[`qualify_native_support.py`](../test/qualify_native_support.py) then executed
those streams on a selected-source-linked standalone `AtlasCore`. ACT was not
used by this compilation or execution path.

| Program | Selected instructions | Instruction words | Program SHA-256 | Panels passed |
| --- | ---: | ---: | --- | ---: |
| BF16 LOG2 | 5 | 39 | `f7e998a1cf68091a3e4e6ee773e35845908e7fa7a7042b4b9d1f8a03cd33b6cc` | 2/2 |
| BF16 SQRT | 5 | 39 | `06fa234c4c4d8ff76c23534f6eacf2c4a8919ddf7e05a4723f7afd95affc1663` | 2/2 |
| BF16 EXP2 | 5 | 39 | `27a675738723ee36acd625776f21237967649e1a4beeabb0d444c7ef624e7efa` | 2/2 |
| Two MXU0 contractions → VPU MIN | 17 | 67 | `7a796757511c6201878dafd142aced5adf6fcc510aeb6618c6f20d72ecd26ba3` | 2/2 |

Every panel checked all 2,048 output bytes, all source bytes, a 32-byte guard,
and ECALL halt. The three unary programs each observed 64 DMA reads and 64
writes and halted after 392 cycles. The composed program observed 128 reads,
64 writes, and halted after 902 cycles. EXP2 and SQRT expected bits come from
the hand OOT public reference tests. LOG2 inputs are exact BF16 powers of two;
the expected BF16 results are exact small integers. The composed program uses
exact E4M3 ±1, +2, and +1 weights; the expected MIN results are exact BF16
±32. This is bounded numerical evidence for these panels.

The ARC model was rebuilt from the selected Atlas RTL revision
`0079c0541111197741a231c002e3843fa6f545b2` and the previously recorded
`AtlasRocketConfig` FIRRTL SHA-256
`fefa711dba44498317573ee5af2cfd82edb674cdfff1505318ea93e896fc123d`.
The 78-module `AtlasCore` functional closure omitted five outputless TileLink
monitor instances, which carry assertions and no datapath outputs. The new
state manifest SHA-256 is
`db2d8ae3c8a4ce6a417b0c691be1d446f1c0be95efe5607a251a6946a80de9e3`;
the compiled model SHA-256 is
`196380fca0cb3416a188538b342f8940802ebcf4d42e8d05dfe351b00b2c46fc`.
Both exactly match the earlier selected-source receipts. ARC conversion took
195.41 seconds; Clang shared-library compilation took 283.82 seconds.

The evaluator requires the exact selected model, state, and RTL revision
identities. It checks the native engine label and binary/plan hashes before
running. A one-bit modification of the EXP2 binary and substitution of an
older ARC model were each rejected with a nonzero exit and no success receipt.
The detailed local receipts are under the
Merlin invocation-owned `out/artifacts/targets/atlas/` root:
`phase1-selected-core-build-r1/receipt.json`,
`phase1-native-selected-core-bounded-r2.json`, and
`phase1-native-support-diagnostic-r1.json`.

From the OOT checkout, with those four native compile artifacts available:

```sh
python test/qualify_native_support.py \
  --arc-model "$ARC_MODEL" --arc-state "$ARC_STATE" --modelir "$MODELIR_ROOT" \
  --expected-model-sha256 196380fca0cb3416a188538b342f8940802ebcf4d42e8d05dfe351b00b2c46fc \
  --expected-state-sha256 db2d8ae3c8a4ce6a417b0c691be1d446f1c0be95efe5607a251a6946a80de9e3 \
  --expected-source-revision 0079c0541111197741a231c002e3843fa6f545b2 \
  --program "exp2=$MERLIN_ARTIFACT_ROOT/phase1-95-exp2-compile-r1" \
  --program "sqrt=$MERLIN_ARTIFACT_ROOT/phase1-95-sqrt-compile-r1" \
  --program "log2=$MERLIN_ARTIFACT_ROOT/phase1-95-log2-compile-r1" \
  --program "minmax=$MERLIN_ARTIFACT_ROOT/phase1-95-minmax-compile-r1" \
  --out "$MERLIN_ARTIFACT_ROOT/phase1-native-selected-core-bounded-r2.json"
```

This is standalone-core execution with diagnostic static delays. The FIRRTL
selection used `AtlasRocketConfig`, so it is not an integrated `EE290SimConfig`
execution. These panels do not qualify all 99 decoder modes, full-domain
numerics, temporal scheduling, arbitrary tiles, complete models, a host driver,
or Phase 1 gates D/F/M/N. The required original-model FP8/BF16 quality policy
and post-freeze held-out evaluation remain open.
