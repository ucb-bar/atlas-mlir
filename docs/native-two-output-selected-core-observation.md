# Bounded native two-output execution on selected AtlasCore

Merlin `68f4d2e4d91d4cb10ec462a085e53e6cc661d2dd` and the installed
Atlas support wheel `0.0.5` built a fresh native target snapshot with ACT
packages absent. The compiler received only the public typed request and ABI.
The request has one 2,048-byte BF16 input used by two operations, LOG2 and
SQRT, and two separate 2,048-byte outputs. A second compile reversed the
requested output order. Both binaries were executed by the evaluator-only
[`qualify_native_two_output.py`](../test/qualify_native_two_output.py) on the
previously source-linked standalone `AtlasCore` functional model.

| Output order | Binary SHA-256 | Panels | Cycles each | DMA reads/writes each |
| --- | --- | ---: | ---: | ---: |
| LOG2, SQRT | `d4dc436fdd481c3c1f3bd1334fb16b8c966849000892c1c1e16c156fa9cf651e` | 2/2 | 623 | 64/128 |
| SQRT, LOG2 | `e5c45642613dbbf10e75595bd9e565f36bde082d547c8a38cf8221b1531b1726` | 2/2 | 623 | 64/128 |

The two panels permute the exact BF16 anchors 1, 4, 64, and 256 over all
1,024 elements. Independently specified result bits are LOG2 = 0, 2, 6, 8
and SQRT = 1, 2, 8, 16. Each panel checked every output byte, the preserved
input, a separate 32-byte memory guard, and halt. Both orders passed all
checks. A one-bit binary mutation was rejected before execution by the
program/plan identity check.

The evaluator requires the exact selected model and state hashes:

- model: `196380fca0cb3416a188538b342f8940802ebcf4d42e8d05dfe351b00b2c46fc`
- state: `db2d8ae3c8a4ce6a417b0c691be1d446f1c0be95efe5607a251a6946a80de9e3`
- ModeLIR Atlas driver: `d1897fe1b49adf68cc3d8dc46794f22f1b98418eef6cfea9231aee15cb9c79c2`
- selected Atlas RTL: `0079c0541111197741a231c002e3843fa6f545b2`

The earlier [selected-core build observation](native-support-selected-core-observation.md)
binds the standalone core to the earlier `AtlasRocketConfig` FIRRTL. To rerun
either compiled program in a fresh output location:

```sh
python3 test/qualify_native_two_output.py \
  --arc-model "$ARC_MODEL" --arc-state "$ARC_STATE" \
  --modelir "$MODELLIR_ROOT" \
  --expected-model-sha256 196380fca0cb3416a188538b342f8940802ebcf4d42e8d05dfe351b00b2c46fc \
  --expected-state-sha256 db2d8ae3c8a4ce6a417b0c691be1d446f1c0be95efe5607a251a6946a80de9e3 \
  --expected-modelir-driver-sha256 d1897fe1b49adf68cc3d8dc46794f22f1b98418eef6cfea9231aee15cb9c79c2 \
  --expected-source-revision 0079c0541111197741a231c002e3843fa6f545b2 \
  --program "$NATIVE_PROGRAM" --out "$FRESH_RECEIPT"
```

This result covers these exact BF16 panels, output layouts, and diagnostic
static delays on the standalone core. It does not establish full-domain LUT
semantics, a temporal scheduling contract, execution in unmodified
`EE290SimConfig`, model compilation, or D/F/M/N release gates.
