# Instruction memory and reduction unrolling

## Capacity and control flow

The selected source configuration provides **128 KiB of IMEM**, or 32,768
four-byte instructions. CMake derives the compiler capacity from
`runtime/selected-host-map.json`; the host tools consume the same selection.
This replaces the former 2,048-word generated-program restriction.

Lowering checks the untimed size. Stream rewriting resolves branches again
when waits, scheduling or delays change layout. An out-of-range conditional
branch becomes an inverted short branch around a long `JAL`; its original
delay slot still executes on both paths. The jump gets its own NOP slot.
Layout expansion repeats until offsets stabilize. Generated correspondence,
physical hazards, and timing remain checked. No indirect return is required.

Final encoding checks capacity after spills, DMA waits, delay instructions,
branch expansion and termination. A smaller deployment budget can be selected
with `--program-size-limit-bytes` on `merlin-targetgen native-compile`, or
`program-size-limit-bytes` on `--lower-atlas-virtual-to-machine`.
An over-capacity result is a compilation failure, not an executable artifact.

Each binary starts at word zero and is linked independently. Program sets use
explicit host dispatch between images, preserving carried BF16 state where the
streaming policy admits it. Their combined byte count is not one resident image.

## K unrolling

M and N are output dimensions; K is the ordered contraction reduction.
Factor 4 groups four continuation tile steps into each K-loop iteration.
The first tile still resets the accumulator. Cleanup handles incomplete groups.
Layer and denoising loops are not unrolled by this setting, and a factor of four
does not imply a fourfold speedup.

Compilation accepts `--reduction-unroll auto|1|2|4|8`. Without a matching measured
profile, `auto` uses factor 1 and records that decision. An explicit factor is
useful for diagnosis; it can fail allocation or another backend check.

## Measure variants

Use the existing native snapshot and a typed whole-contraction request, for
example `test/examples/native_mxu0_unroll_request.json` with (M,K,N)=(2,257,3). Write
`runner.json` as a JSON argv array naming the Python interpreter and
`tools/profile_looped_model.py`. The runner executes generated words through an
independent decoder and requires identical model re-encoding. It checks two
sparse signed input sets, exact outputs, unchanged inputs and surrounding memory.
This runner currently accepts non-streamed, unseeded contractions only.

```sh
merlin-targetgen native-profile --support atlas \
  --snapshot "$SNAPSHOT" --request "$REQUEST" --abi auto \
  --target-source "$ATLAS_ROOT" --runner-command runner.json --out out/unroll-profile

merlin-targetgen native-compile --engine merlin_native --support atlas \
  --snapshot "$SNAPSHOT" --request "$REQUEST" --abi auto \
  --target-source "$ATLAS_ROOT" --mode diagnostic \
  --reduction-unroll auto --tuning-profile out/unroll-profile/profile.json \
  --out out/profiled-compilation
```

The profile accounts for factors 1, 2, 4 and 8, including failed compilations.
Each passing variant gets one warm-up and five measured repetitions. Selection
uses median measured cycles, then smaller final bytes and smaller factor.
The profile includes source, ABI, target, compiler snapshot, tool, pipeline and
search-setting identities; a mismatch uses rolled code. The compiler does not
execute a simulator or access benchmark inputs during normal compilation.

The report retains final bytes, spill/reload counts and execution cycles.
Model cycles exclude CPU dispatch and image loading. Host timings remain
unavailable until a platform runner supplies them. A custom runner receives
`--compiled`, `--out`, `--warmup` and `--repetitions`; it must identify the exact
binary and report checked correctness plus five samples for the selected
`device_cycles` or `warm_host_ticks` objective. Keep model, RTL and FPGA
measurements separate.

Automatic selection only considers successfully checked variants. Failure of
one unroll factor does not make the source operation hardware-unsupported.

The optimized pipeline can currently reject a large unroll group when scalar
spilling turns a loop address into a slot restore that the address proof cannot
reconstruct. The profiling report retains that compilation error and can select
a smaller checked factor. This is a compiler limitation, not an ISA limitation.
