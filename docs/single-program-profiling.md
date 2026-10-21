# Profile one resident program

Measure the final image after allocation, spilling, waits, delays and branch
layout. A sum of separate kernel sizes does not measure a resident program.
Report model cycles, host timings and FPGA measurements separately.

## Captured attention and MLP

Build the [single-image C/H example](baremetal-c-integration.md#one-image-and-one-launch).
There is no CPU computation between attention and the MLP. The existing profile
command accepts Linalg and an explicit precision policy. Write `runner.json` as
an argv array containing the model Python, `tools/profile_graph_model.py`,
`--compile-inputs`, and the absolute staged-capture path, then run:

```sh
merlin-targetgen native-profile --support atlas --snapshot "$SNAPSHOT" \
  --linalg "$COMPILE_INPUTS/program.mlir" --linalg-entry forward \
  --diagnostic-graph-policy test/examples/attention_mlp_graph_finite_zero_arithmetic.json \
  --abi auto --search-limits test/examples/attention_mlp_graph_search_limits.json \
  --target-source "$ATLAS_ROOT" --runner-command runner.json --out out/graph-profile
```

The runner compiles and invokes the C/H against the pinned model. It checks two
runtime inputs, both outputs with the public independent arithmetic reference,
unchanged inputs and guard memory. One warm-up and five measured invocations
must each launch once, with one initial image load. Reference code is used only
by the evaluation tool.

Each 32×32 contraction fits one hardware tile. There is no software K loop:
changing its requested unroll factor has no effect. The report explicitly
records `effective_K_unroll = 1`. Equal bytes and cycles for these variants are
expected and do not establish a K-unrolling speedup.

## Compose looped virtual regions

`--compose-atlas-program` takes an ordered list of virtual functions in the
module attribute `atlas.program_entries`. Each function must return one virtual
state and use explicit DMA with bounded output storage. Their output ranges must
form one contiguous interval; the pass cannot grant write permission to gaps
between buffers. Indexed local ports and calls are rejected. Matching target
attributes are preserved, and the smallest supplied code-size ceiling applies
to the joined program. The pass clones regions and threads state through MLIR block
arguments, leaving one return/ECALL. Run it before virtual verification,
scheduling, spilling and allocation, so those passes see the combined program.
It does not infer fusion, layouts or whether a source model is supported.

The public composed-loop test measures factors 1/2/4/8 on two independent
(M,K,N)=(1,257,1) contractions with two outputs in one image. Select Merlin,
Atlas tools and the pinned model as in [native compilation](native-merlin-binding.md).
Set `ATLAS_RUN_NPU_MODEL=1` and `ATLAS_COMPOSITION_PROFILE_OUT` to a fresh JSON
path, then run:

```sh
python -m unittest discover -s test -p test_program_composition.py -v
```

These measurements use the mandatory pipeline. The optimized pipeline can
reject a large unroll factor after scalar spilling; keep that failure visible
instead of bypassing its address checks.

## Connected graphs with a larger MLP

Capture the Merlin attention/MLP source with `ATTENTION_MLP_HIDDEN=288`, then
stage it using the commands in that source example's README. The frontend
capture contains source floating-point operations. Compilation requires an
explicit precision transformation; it is not a TorchAO-quantized capture.

The resident graph lowerer connects selected tile DAGs, compact matrix loops,
pointwise scaling, ReLU and BF16-to-FP8 pack loops before allocation. Intermediate
tensors stay on Atlas. Weight orientation follows the source transpose maps.
The current resident layout requires tile-aligned matrices, uses one private
DRAM allocation per region result, and refuses host streaming or program
segmentation. A required internal tail layout or an exhausted diagnostic DRAM
window produces an error. Standalone contraction tails retain their separate
checked path.

Use `--abi auto` with `native-compile`. To measure actual reduction unrolling,
use `tools/profile_resident_model.py` in the profile runner instead of the
single-tile C callback runner. Its argv array must include `--compile-inputs`
and `--policy`; the profile command supplies the compiled directory and run
counts. The same `native-profile` command above then measures factors 1/2/4/8.
The runner checks both input panels, ordered outputs, every stored intermediate,
immutable inputs and surrounding memory. By default it measures model cycles.

`attention_mlp_graph_scaled_arithmetic.json` also inserts explicit BF16
power-of-two multiplies before packing and restores contraction scales. These
instructions execute on Atlas and contribute to final size and cycles. Measure
original-model error separately; successful execution does not qualify a
precision policy for SmolVLA or another model.

The [resident C data export](baremetal-c-integration.md#resident-program-data)
provides the program and its buffer descriptions to an external driver.

### Native host runtime timing

Add `--profile-host-runtime` to the resident runner to export the same program as
C/H, build it with `runtime/atlas_host.c`, and time load, residency checks, start
and completion. For example:

```sh
python tools/profile_resident_model.py \
  --compiled "$COMPILED" --compile-inputs "$COMPILE_INPUTS" \
  --policy test/examples/attention_mlp_graph_scaled_arithmetic.json \
  --out "$PROFILE_JSON" --warmup 1 --repetitions 5 --profile-host-runtime
```

The existing Merlin `native-profile` runner command can include that flag too.
Each sample records `load_ns`, `load_or_reuse_ns`, `start_ns`, `wait_ns`, image
loads/reuses, words written and launches. The first sample loads IMEM; later
samples retain the image. The runner checks that exactly one image was loaded
and every invocation launched once. Device cycles and numerical checks remain
unchanged. The generated C and build command stay next to the profile.

These nanoseconds measure native C with Python callbacks, using a virtual MMIO
map and coherent memory. Model execution occurs outside the timed calls. Each
model state is initialized independently; the host's image cache is retained.
The callbacks expose the model's status and retired-instruction CSRs. An initial
poll acknowledges the newly started model before synchronous execution.

This does not measure packing, data transfers, bus/cache delays, the complete CPU
invocation or Shuttle/FPGA latency. In particular, `wait_ns` measures completion
processing after model execution, not how long Atlas takes to run. Keep these
results separate from device cycles and platform measurements.

### RTL fixtures

The resident runner can additionally write selected-AtlasCore fixtures with
`--rtl-fixtures FRESH_DIRECTORY --rtl-max-cycles LIMIT`. Choose the watchdog
from measured program work and the test memory's bounded stalls before running
the test. Pass the generated `input0` and `input1` directories as `--fixture`
arguments to `test/rtl/run_compiled_program_atlas_core.py`.

Fixture v3 compares every byte in the private DRAM arena, including packed
intermediates and untouched padding. It checks DMA addresses against immutable
inputs and private storage and preserves inputs and adjacent guards. Unlike the
older tile fixtures, it permits reads of intermediates and does not prescribe an
exact transaction count. Results report checked 16-bit output words, since the
arena contains both FP8 and BF16 data. This test runs one standalone AtlasCore
through Verilator; it does not qualify a complete EE290 system or FPGA.

## Host clock hooks

After initialization, call `atlas_host_set_clock(host, read_clock, context)`
while idle. The callback returns a monotonic `uint64_t` counter. The platform
owner records its unit and source. For example, the host can supply its permitted
cycle counter; a native harness can use a monotonic nanosecond clock.
Passing a null callback disables timing. Setting the callback resets totals.

`host.timings` accumulates:

- `load_ticks`: IMEM loading, including MMIO callbacks and failed attempts.
- `start_ticks`: pre-launch cache operations and the launch protocol.
- `wait_ticks`: synchronous polling and output-cache synchronization, including timeouts.

Image reuse does not invoke the loader. Use the load/reuse/launch counters to
distinguish cold and resident execution. Standalone `poll` calls are not timed
by `wait_ticks`. Instrument transfers and the full invocation separately.

The model runner reports native C callback-harness nanoseconds with synchronous
simulation time removed. Python callbacks and memory copies remain included.
These are not Shuttle cycles, cache costs or FPGA MMIO latency. Run the same
clock API on the platform before reporting those quantities.

## Full-model experiments

Layer-loop and denoising-loop unrolling need the complete body: normalization,
masks, positional encoding, conversions and state as well as contractions.
Repeated shapes do not supply those implementations. Preserve unavailable
captures and unsupported-operation diagnostics in the experiment population.
The tests above do not establish full SmolVLA or pi0.5 fit or execution.
