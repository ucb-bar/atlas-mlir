# Merlin-native Atlas binding

Merlin's native compiler (`merlin-targetgen native-*`) picks the instructions
and assigns registers and DRAM slots. The `atlas_mlir_native` package in
`python/` is the Atlas half: it declares the instruction rules, data layouts and
numerical policies, turns Merlin's choice into Atlas virtual MLIR, lowers and
emits it with `atlas-opt` and `atlas-emit`, and replays the emitted stream
(addresses, register pairs, waits, completion) against that choice. A failed
replay sends Merlin's search to its next candidate. ACT is not involved.

Everything targets the pinned RTL, atlas-npu
`0079c0541111197741a231c002e3843fa6f545b2`; the binding refuses to compile if
`docs/selected-variant-inventory.json` names another revision. An older route,
the [standalone importer](#standalone-captured-linalg-importer), compiles a
captured 32×32 Linalg function without Merlin; the captured-model examples use it.

## Setup

Build `atlas-opt` and `atlas-emit` as in the [root README](../README.md), install
Merlin's Python package, then install this package. Its `atlas` entry point is
what `--support atlas` loads.

```sh
uv pip install --python /path/to/venv/bin/python --no-deps -e .
```

| Variable | Meaning |
| --- | --- |
| `ATLAS_OOT_BIN_DIR` | Directory with `atlas-opt` and `atlas-emit`. Default `<target-source>/build/bin`. |
| `ATLAS_NATIVE_PIPELINE` | `mandatory` (default) or `optimized`. `optimized` ships a rescheduled program that only the compiler verifiers and encoding checks have examined; the native semantic checks ran on the mandatory lowering of the same virtual program, saved beside it as `mandatory-qualification-*`. |
| `MERLIN_NATIVE_ROOT` | Merlin checkout, for tests. Native tests skip without it. |
| `ATLAS_ARC_MODEL`, `ATLAS_ARC_STATE`, `ATLAS_MODELIR_ROOT` | AtlasCore simulator for opt-in core tests. |

## Compile a tile

Build a snapshot once, then compile and verify. Use a fresh output directory
each time; the tools refuse a non-empty one. This compiles a 32×64 by 32×64
MXU0 contraction as two K segments:

```sh
RUN=out/native/k64; SRC=test/examples/native_mxu0_k64_linalg.mlir
mkdir -p "$RUN"
echo '{"fixed_inputs": {"arg0__k0": 3, "arg1__k0": 4, "arg0__k1": 5, "arg1__k1": 6},
       "fixed_outputs": [35]}' > "$RUN/abi.json"
merlin-targetgen native-build --engine merlin_native --support atlas \
  --cargo-target-dir "$RUN/cargo" --source-revision "$(git rev-parse HEAD)" \
  --out "$RUN/snapshot"
merlin-targetgen native-compile --engine merlin_native --support atlas \
  --snapshot "$RUN/snapshot" --linalg "$SRC" --linalg-entry mxu0_k64_tile \
  --ordered-k-tile 32 --abi "$RUN/abi.json" --target-source . \
  --mode diagnostic --out "$RUN/compiled" --status-file "$RUN/status.json"
merlin-targetgen native-verify --engine merlin_native --support atlas \
  --snapshot "$RUN/snapshot" --linalg "$SRC" --linalg-entry mxu0_k64_tile \
  --ordered-k-tile 32 --mode diagnostic \
  --artifact "$RUN/compiled" --out "$RUN/verify.json"
```

`native-build` compiles the rule set into Merlin's e-graph bridge (needs cargo).
`native-select` stops after selection and allocation. `native-verify` rebuilds
the request from the same flags, rechecks every stage file against the
manifest, and reports `verified_artifacts`.

| Flag | Meaning |
| --- | --- |
| `--linalg FILE --linalg-entry NAME` | Linalg source and the function to compile. |
| `--request FILE` | Typed request (`merlin.semantic_kernel.v2`/`v4`) instead of Linalg, e.g. `test/examples/native_bf16_relu_request.json` with ABI `{"fixed_inputs":{"x":3},"fixed_outputs":[35]}`. |
| `--abi FILE` | `fixed_inputs` maps each input (`arg0`, or a tile input like `arg0__k1`) to a DRAM slot; `fixed_outputs` lists output slots in order. Slot *n* < 128 is at `0x90000000 + n·2048`. |
| `--mode diagnostic`, `--target-source .` | The only accepted mode; this checkout (RTL revision, tool directory). |
| `--search-limits FILE` | Bigger search budget (`merlin.native_search_limits.v1`). The attention examples need it. |
| `--ordered-k-tile 32` | Split K into 32-wide MXU0 segments. |
| `--ordered-m-tile 32 --ordered-n-tile 32 --outer-tile-index I` | Compile output tile *I* (row-major) of a larger MXU0 matmul. |
| `--physical-layouts FILE` | Pin input/output layouts for Linalg sources; not with `--request` or outer tiling. Precision-transformed sources need the `merlin.physical_boundary_layouts.v2` form. |
| `--diagnostic-graph-policy FILE` | Lower a whole f32 Linalg graph under a precision policy. |
| `--compile-inputs DIR --diagnostic-region ID [--diagnostic-tile-index I]` | Compile one f32 matmul region (one output window) of a staged model capture. |

Failure statuses: `unsupported_semantics` (no rule), `unqualified_target`
(refused on this target), `resource_limit`, `tool_unavailable`, `compile_error`.

**Typed Linalg** spells out target arithmetic with `merlin.*` policy and
input-domain attributes (see `test/examples/native_*_linalg.mlir`): copies and
transposes of 32×32 tiles; MXU0 contractions as `linalg.generic` with
`A[M,K]`/`W[N,K]` maps, a BF16 +0 fill, an `extf, extf, math.fma` body and
`merlin.reduction_order = "k-ascending"`; ReLU as `arith.maximumf(x, +0)`; add
and subtract as f32 math on widened BF16 keeping the upper 16 bits (plain BF16
`arith.addf` is refused); nearest-even BF16 multiply; and `arith.truncf` packs.
A plain f32 `linalg.matmul` needs a precision policy.

**Whole f32 graph.** For a captured f32 model, add a graph policy:

```sh
merlin-targetgen native-compile --engine merlin_native --support atlas \
  --snapshot "$RUN/snapshot" --linalg-entry forward \
  --linalg test/examples/captured_attention_mlp32_two_outputs_linalg.mlir \
  --diagnostic-graph-policy test/examples/attention_mlp_graph_precision.json \
  --abi test/examples/attention_mlp_graph_abi.json \
  --search-limits test/examples/attention_mlp_graph_search_limits.json \
  --target-source . --mode diagnostic --out "$RUN/graph"
```

The ABI has six FP8 inputs and one generated BF16 constant (`op25r0`, the
score scale). `graph_precision.py` accepts 32×32 f32 inputs, zero-initialized
matmuls (MXU0; transpose only on the weight), multiply by an f32 literal tile,
add, row max and sum with a reshape broadcast, broadcast subtract, exp,
broadcast divide (VRECIP then VMUL) and clamp-at-zero (ReLU), and inserts a pack
before each matmul operand that comes from a BF16 value.

**One model region.** For a capture staged with Merlin's `stage-capture
--precision-policy`, `--diagnostic-region matmul_N` compiles one matmul window
(`--diagnostic-tile-index` counts across N first; MXU0 also splits K). Weights
are packed into `constants/rhs.bin`; one outside the input domain is marked
`blocked`, with no payload. Fixtures: `test/examples/source_bound_*.mlir`.

## Standalone captured-Linalg importer

```sh
python tools/compile_atlas_linalg_tile.py \
  --linalg test/examples/captured_mlp32_linalg.mlir \
  --argument-manifest test/examples/captured_mlp32_arguments.json \
  --weights test/examples/captured_mlp32_weights.safetensors \
  --policy test/examples/atlas_mlp_numeric_policy.json \
  --atlas-bin-dir build/bin --llvm-bin-dir "$LLVM_BIN" --linker "$LLD" \
  --output-dir out/captured-mlp/compiler
```

It parses with xDSL ([requirements](../tools/requirements-captured-mlp.txt)) and
writes Atlas virtual SSA directly. It accepts zero fills, matmuls, weight
transposes, 32-element bias adds, ReLU (`maximumf` or model2mlir's
compare/select form), exp, splat scaling, row max/sum with row broadcasts, and
row subtract/divide, with up to 16 ordered results. Inputs start at
`--input-base` (`0x90000000`, 2 KiB slots), outputs at `--output-base`
(`0x90010000`); weights and biases become constant slots. The LLVM flags are
optional and add LLVM stages and an ELF ([compiler-pipeline.md](compiler-pipeline.md)).
Neither route ever sees sample inputs or expected outputs.

## Supported operations

All tiles are static 32×32. A BF16 tile sits in a register pair and in DRAM as
two 1 KiB halves (columns 0–15, then 16–31). An FP8 tile uses the first 1 KiB of
a 2 KiB slot. The domain is what a rule assumes about its inputs; the device
program does not check it.

| Operation | Atlas instructions | Input formats / domain | Notes |
| --- | --- | --- | --- |
| Copy | DMA, VLOAD, VSTORE | BF16 or FP8, any bits | Outputs can share one load. |
| Transpose | XLU transpose | FP8, raw or packed | Also used on the value tile in attention. |
| Matmul, MXU0 | MXU weight push, matmul (reset or accumulate), readout | E4M3 `A[M,K]`, `W[N,K]`; finite normal or +0 | BF16 rounding after each K step, in order. K chains, tiling, VPU consumers. |
| Matmul, MXU1 | MXU weight push, matmul (reset), readout | v3: {+0, ±1, ±2}. v4, v6, v7: finite normal or +0 | One 32-wide K tile; no K chain, VPU consumer or batching. |
| ReLU | VPU unary | BF16 finite normal or zero | Negative values and −0 become +0. |
| Add | VADD_BF16 | BF16 normal; sum fits f32, normal | f32 add, then BF16 chop. |
| Subtract | VSUB_BF16 | BF16 normal; difference fits f32, normal or zero | Same chop as add. |
| Multiply | VMUL_BF16 | BF16 normal; product normal | Nearest even. One operand may be a BF16 constant tile. |
| Row max | VREDMAX_ROW_BF16 | Any BF16 code | RTL integer order of BF16 bits, not float `max`. Row broadcast. |
| Row sum | VREDSUM_ROW_BF16 | BF16 normal or +0; partial sums stay normal | Five-level f32 pairwise tree (RNE), one BF16 rounding. Row broadcast. |
| Exp | VEXP | BF16 normal in [−80, 80], or ±0 | RTL LUT approximation (`exp_lane.py`), not math `exp`. |
| Reciprocal | VRECIP | Positive normal BF16 in [1, 32] | RTL LUT with truncation (`recip_lane.py`). |
| Pack BF16→FP8 | VPU PACK, then a scalar loop that interleaves the halves into rows | Unit (127): {+0, ±0.5, ±1, ±2}. Scaled (122–128): +0 or values that scale to normal E4M3 without rounding. Rounded (122–128): any BF16 | Code set by SELI. Rounded mode follows `FP8Pack.scala` (`pack_lane.py`). |

Tested compositions run from MXU0→ReLU up to Q/K/V attention followed by a
two-layer ReLU MLP with two or three ordered outputs.

## Tiling

Only MXU0 contractions tile. Merlin partitions the source; the binding checks it.

- **K**: `--ordered-k-tile 32` gives one reset, then accumulate steps, then one
  readout. Tile inputs are named `<arg>__k<i>`. When K is not a multiple of 32,
  the first segment is padded at its start with FP8 +0, so zeros are added
  before any real element. MXU1 refuses K chains.
- **M/N**: `--ordered-m-tile 32 --ordered-n-tile 32 --outer-tile-index I`
  compiles one 32×32 output tile per call. Missing rows and columns in edge
  tiles are FP8 +0; `ordered-mn-tiling.json` keeps each valid extent. Tiles with
  the same shape share one request digest and binary, so the host can launch
  one program for every tile.

## Precision policies

The compiler never chooses precision. A policy file says how f32 arithmetic
becomes FP8/BF16 arithmetic. Each one is a proposal whose effect on model
quality has to be measured.

| Schema | Used by | What it sets |
| --- | --- | --- |
| `atlas.oot.quantized_linalg_policy.v1` | importer | MXU unit per contraction (`mxu_units`); every pack uses code 127. |
| `atlas.oot.quantized_linalg_policy.v2` | importer | Adds `matmul_activation_pack_scale_codes`, one code per contraction. |
| `atlas.oot.quantized_linalg_policy.v3`–`v8` | `--diagnostic-region` | `{mxu_unit, activation_pack_scale_code}` per named region. |
| `atlas.diagnostic_graph_precision.v1` | `--diagnostic-graph-policy` | One `pack_scale_code` for all packs; `interlayer_pack` is `scaled-e8m0-exact-normal-v1` or `scaled-e8m0-rtl-rne-v1`. |

Region versions: v3 is the base (MXU1 domain {+0, ±1, ±2}; MXU1 codes other than
127 refused). v4 adds MXU1 anchor arithmetic on finite normal inputs. v5 maps an
FP8 −0 input to +0 (MXU0 only). v6 is v4 plus that mapping on both units. v7
adds `batched_matmul_regions` (MXU0 only). v8 is v7 plus the v5 mapping.

**Conversions.** f32 inputs round to nearest even onto finite normal E4M3;
values below the smallest normal flush to zero, values above the largest clamp,
NaN and Inf are refused. In the native routes a result of −0 is refused unless
the policy maps it (v5, v6, v8). Biases and literal tiles round to BF16. BF16
outputs widen to f32 without loss.

**Scale codes.** E8M0 code *c* is a factor of 2^(c−127); 122–128 are accepted.
In the importer (v2) and in model regions, the activation is multiplied by
2^(127−c) and the constant weight by 2^(c−127) before E4M3 rounding, so the
factors cancel except for rounding (code 122: activation ×32, weight ÷32). The
importer needs code 127 unless the activation comes from a BF16 value and the
weight is a declared constant. The graph policy does not rescale weights (they
are runtime inputs), so its code changes results: scale 126 differs from the
f32 source by up to 1.875 on the test panels.

**What you supply.** The policy file; f32 inputs that convert inside the input
domain; intermediate values that stay inside each operation's domain
(`execution_plan.json` lists these under `boundary_domain_obligations` and
`producer_domain_obligations` as `caller_premise_unchecked`); and a quality
check of the policy on your model.

## Outputs

| File | Contents |
| --- | --- |
| `atlas-virtual.mlir` | Atlas virtual SSA from the chosen instruction graph. |
| `atlas-machine.mlir` | Physical registers, DMA staging, waits. |
| `physical-program.json` | Every emitted word with its instruction fields. |
| `program.bin` | Device image. |
| `execution_plan.json` | Input/output DRAM addresses and sizes, output layouts, per-output policies, domain obligations. |
| `native-selection.json` | Chosen instruction graph, addresses, issue order, replay fingerprint. |
| `manifest.json` | Hashes of snapshot, request and every stage; `native-verify` checks them. |
| `precision-plan.json` | f32 routes: source-to-target mapping, replacements, conversions. |
| `ordered-k-tiling.json`, `ordered-mn-tiling.json` | Tiling records. |
| `physical-boundary-layouts.json` | Copy of `--physical-layouts`. |
| `constants/` | BF16 constant tiles and packed weights (`rhs.bin`). |

The importer writes the first four plus `program.words.txt`, `word-map.json`,
constants and `import-manifest.json` (hashes, slot map), and with LLVM tools the
`atlas-llvm*.mlir`, `program.ll/.s/.o/.elf` stages.

`tools/check_program_decode.py` decodes every word of `program.bin` against the
RTL decoder.

## Host side

A RISC-V host stages inputs, starts Atlas and reads results ([host-launch.md](host-launch.md)).
The `tools/atlas_*` scripts turn a verified compilation into host data: boot
capsule and embedded words, FP8 pack and BF16 unpack functions, tiled-matrix
launch tables, and MLIR C-interface callbacks for a lowered host model
(`*_model_bridge.py`, `atlas_model_link.py`) that call the trapping
`atlas_model_tile_fail` hook on failure. Host code linked at `0x80000000` needs
`-mcmodel=medany`. Walkthroughs: [native attention+MLP](../examples/handoff/native_attention_mlp/README.md),
[ordered M/N](../examples/handoff/ordered_mn/README.md), [source-bound matrix](../examples/handoff/source_mn/README.md),
[captured attention+MLP](../examples/handoff/captured_attention_mlp/README.md).

## Test status

Unit tests need Merlin and the built tools, no simulator. Compiled attention+MLP
programs were checked with unmodified `npu_model` main
`9e6313bddecb8ba936206eb347f5cd4e5335282c` and `numerics="rtl"`.
"RTL core" is a standalone AtlasCore with a test-side TileLink DRAM responder.

| Example | Unit tests | npu_model | RTL core | FPGA / SoC |
| --- | --- | --- | --- | --- |
| Copy, transpose, ReLU, add, multiply tiles | yes | yes | no (hand-written versions yes) | no |
| Subtract, row max, row sum, exp, reciprocal, BF16-constant multiply | yes | no | yes | no |
| MXU0 tile, K chains up to K=128, K/M/N tails | yes | yes | yes, per tile | no |
| MXU1 tile | yes | no | yes, inside its domain | no |
| MXU0 + VPU (ReLU, bias add, multiply, add→ReLU) | yes | yes | no | no |
| MXU0→ReLU→pack→MXU0, Q/K score, softmax, Q/K/V attention | yes | no | yes | no |
| Native attention+MLP request | yes | no | yes | no; host ELF built, not run |
| f32 attention+MLP graph (incl. rounded pack, scale 126, residual add) | yes | yes (1 panel) | yes | no |
| Ordered M/N matrix host | yes (fake MMIO) | yes | tile program only | no |
| Source-bound f32 matrix (K=72) | yes (fake device) | no | yes, replayed through the host entry | no |
| Captured attention+MLP (importer) | yes | yes | yes, one and two outputs | no |
| Captured MLP to ELF (importer) | yes | no | yes (opt-in end-to-end test) | no |

Main tests: [`test_native_binding.py`](../test/test_native_binding.py),
[`test_graph_precision.py`](../test/test_graph_precision.py),
[`test_precision_abi.py`](../test/test_precision_abi.py),
[`test_native_multi_result_guard.py`](../test/test_native_multi_result_guard.py),
`test_captured_*_import.py`, and the opt-in
[`test_captured_mlp_end_to_end.py`](../test/test_captured_mlp_end_to_end.py), which also
needs `ATLAS_M2M_PYTHON`, `ATLAS_M2M_ROOT`, `ATLAS_MERLIN_CAPTURE_WORKER` and `ATLAS_LLVM_BIN`.

```sh
MERLIN_NATIVE_ROOT=/path/to/merlin ATLAS_OOT_BIN_DIR=/path/to/atlas-build/bin \
  PYTHONPATH="$PWD/python:/path/to/merlin/src:$PWD/test:/path/to/npu-model/source" \
  python -m unittest test_native_binding -v
```

A skipped test means nothing ran.

**Running compiled programs on the RTL core.** `test/rtl/make_*_fixture.py`
turns a compiled directory into a fixture. [`run_compiled_program_atlas_core.py`](../test/rtl/run_compiled_program_atlas_core.py)
rebuilds the pinned Atlas and FP Scala sources, runs
`CompiledProgramAtlasCoreDiagnosticTest.scala` under Verilator, and writes
`result.json` under `--out`. It needs a Chipyard classpath that builds
`AtlasCore` (the Atlas mill target leaves out `diplomatic/`), ScalaTest 3.2.20
and the Chisel 6.7.0 plugin. The three base programs drive 19 built-in cases;
the runner checks their hashes.

```sh
python test/rtl/run_compiled_program_atlas_core.py --rtl-root "$ATLAS_RTL_ROOT" \
  --dependency-classpath-file "$CHIPYARD_CP" --scalatest-classpath-file "$SCALATEST_CP" \
  --scala-compiler "$SCALA_COMPILER_JAR" --chisel-plugin "$CHISEL_PLUGIN_JAR" \
  --verilator "$VERILATOR" --program "$MXU0_BIN" --mxu1-program "$MXU1_BIN" \
  --k64-program "$K64_BIN" --fixture "$FIXTURE_DIR" --out "$FRESH_DIR"
```

`CompiledProgramScalarCoreDiagnosticTest.scala` checks only decode, issue and
control flow of the 38-word MXU0 tile on `ScalarCore` with idle engine
statuses. Copy it and `DmaScalarCoreDiagnosticTest.scala` into
`src/test/scala/atlas/scalar/` of an RTL checkout, then run
`ATLAS_DMA_DECODE_PROGRAM=/abs/program.bin ./mill atlas.test.testOnly
atlas.scalar.CompiledProgramScalarCoreDiagnosticTest`.

## Limits

- The whole path is diagnostic: `strict-native` is refused until an FPGA
  bitstream and host cache behavior are pinned. Nothing has run on an FPGA or
  an integrated Shuttle+Atlas SoC; `runtime/selected-host-map.json` has no
  bitstream identity and assumes coherent DRAM.
- Tiles are static 32×32. Larger shapes only go through MXU0 tiling, one
  output tile per `native-compile` call. No seeded MXU0 accumulators; K chains
  are tested up to K=128. Waits and issue gaps are conservative, not tuned.
- Merlin candidates with multi-result instructions are refused before emission.
- Device programs have no runtime input-domain guard. Generated host pack
  functions refuse bad f32 inputs, but values computed inside the program are
  not checked, and a simulator result on inputs outside a domain means nothing.
- MXU1 is narrow (see the operations table). Under v3, the captured
  attention+MLP's `key.weight` is outside the MXU1 domain and its scale-122
  MXU1 region is refused, so those two regions stay in host code. The v4
  policy compiles all seven regions, but MXU1 is tested on only a few panels.
- Model quality has no target. The 12-case quality suite
  (`test/examples/attention_mlp_graph_quality_suite.json`) has no pass
  threshold; random panels reach about 0.011 maximum absolute error.
- The exp and reciprocal ports follow `sp26-fp-units` (gitlink `9a0cc09`),
  which has no license file.
