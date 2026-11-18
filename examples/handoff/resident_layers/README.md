# Single-image layer examples

These examples export C/H, Atlas instruction words, data and buffer metadata.
Baremetal-IDE supplies the application, startup code, linker script and platform
callbacks, then builds the host ELF. No generated files belong in this directory.

| Input | Computation | C interface |
| --- | --- | --- |
| Captured attention + MLP | 32 tokens, width 32, hidden width 288; returns the hidden activation and final projection | Embedded program/data arrays and physical input/output descriptions |
| Matrix product | A[64,512] × W[128,512]ᵀ → Y[64,128] | Callable wrapper; dense E4M3 inputs and BF16 output |
| Larger matrix product | A[128,512] × W[128,512]ᵀ → Y[128,128] | Same wrapper with larger buffers |

Each case must compile to one image and one launch. The two matrix products
contain 4,194,304 and 8,388,608 MACs, respectively. Their M/N/K loops stay in
the device program; the host does not launch individual tiles.

## Compiler setup

Use `feat/atlas-model-compilation` in both repositories:
[Merlin](https://github.com/ucb-bar/merlin/tree/feat/atlas-model-compilation)
and [Atlas MLIR](https://github.com/ucb-bar/atlas-mlir/tree/feat/atlas-model-compilation).
The Atlas Python backend uses the shared compiler APIs from that Merlin branch.

Build the tools and install the Python packages using the
[native compilation guide](../../../docs/native-merlin-binding.md). Select the
same source checkouts and tool build throughout:

```sh
export MERLIN_ROOT=/path/to/merlin
export ATLAS_ROOT=/path/to/atlas-mlir
export ATLAS_OOT_BIN_DIR="$ATLAS_ROOT/build/bin"
export PYTHONPATH="$MERLIN_ROOT/src:$ATLAS_ROOT/python${PYTHONPATH:+:$PYTHONPATH}"
export CARGO_BUILD_JOBS=2
COMPILER_PYTHON=/path/to/compiler-venv/bin/python
RUN="$ATLAS_ROOT/out/resident-layers"
mkdir -p "$RUN"
"$COMPILER_PYTHON" -m merlin.targetgen.cli native-build \
  --engine merlin_native --support atlas --io-memory-bytes 1048576 \
  --source-revision "$(git -C "$MERLIN_ROOT" rev-parse HEAD)" \
  --cargo-target-dir "$RUN/cargo" --out "$RUN/snapshot"
```

The 1 MiB option selects an external DMA-buffer window, not additional on-chip
memory. Use it consistently for generation, compilation, verification, export
and profiling. Output directories must be fresh.

## Matrix products

From `atlas-mlir`:

```sh
for rows in 64 128; do
  "$COMPILER_PYTHON" examples/handoff/resident_layers/matmul_request.py \
    --m "$rows" --k 512 --n 128 --out "$RUN/matmul-$rows.json"
  "$COMPILER_PYTHON" examples/handoff/resident_layers/build.py \
    --snapshot "$RUN/snapshot" --request "$RUN/matmul-$rows.json" \
    --name "matmul_${rows}_512_128" --target-source "$ATLAS_ROOT" \
    --pipeline mandatory --reduction-unroll 4 \
    --out "$RUN/matmul-$rows"
done
```

`matmul_request.py` authors a typed FP8 matrix product. It is not a PyTorch
capture. It specifies ordered BF16 reduction and the admitted E4M3 input
domain; the generated wrapper checks that domain before device writes.

Each `c/` directory contains the named C/H pair, `.bin`, `.words.txt`, a
manifest, a README and `runtime/atlas_host.c` and `.h`. Compile each named C
source and link the identical runtime once. Allocate the workspace size from
the generated header, initialize `atlas_host`, call the generated `_bind`,
then `_run`. Inputs and outputs are dense row-major encoded bytes. No floating
point quantization is performed by this interface. See the generated header
for sizes and argument order.

## Captured attention + MLP

Provision Model2MLIR using Merlin's capture documentation. The capture
interpreter can differ from the compiler interpreter:

```sh
cd "$MERLIN_ROOT"
ATTENTION_MLP_HIDDEN=288 "$CAPTURE_PYTHON" \
  src/merlin/targetgen/_m2m_capture_worker.py \
  --m2m-dir "$MODEL2MLIR_ROOT" \
  --loader examples/workloads/attention_mlp/loader.py \
  --dtype fp32 --seed 0 --materialize-bundle --out "$RUN/attention-capture"
"$COMPILER_PYTHON" -m merlin.targetgen.cli stage-capture \
  --capture "$RUN/attention-capture" --out "$RUN/attention-inputs"
cd "$ATLAS_ROOT"
"$COMPILER_PYTHON" examples/handoff/resident_layers/build.py \
  --snapshot "$RUN/snapshot" --compile-inputs "$RUN/attention-inputs" \
  --name attention_mlp_32_32_288 --target-source "$ATLAS_ROOT" \
  --pipeline optimized --reduction-unroll 4 \
  --out "$RUN/attention"
```

Capture starts with ordinary floating point PyTorch operations. Compilation
applies the explicit diagnostic E4M3/BF16 policy in
`test/examples/attention_mlp_graph_scaled_arithmetic.json`, including the
intermediate BF16-to-FP8 conversion. This is not a TorchAO-quantized capture or
a model-quality result. Samples and framework outputs remain outside the
compiler inputs.

The explicit `--pipeline optimized` selects virtual and physical scheduling in
addition to required allocation, spills, waits and checks. For a source-order
comparison, repeat the command with `--pipeline mandatory` and a fresh output
directory. `build.py` defaults to `mandatory`; setting an environment variable
does not override its `--pipeline` option.

The attention `c/` directory contains data declarations, not a callable model
wrapper: `attention_mlp_32_32_288.c`, `.h`, `program.bin`, `data/` and
`manifest.json`. The manifest specifies input encoding/scaling, tile layouts,
embedded weights, both ordered outputs and the private DMA arena. The platform
copies the static segments and packed input to their specified addresses,
loads the program, launches once, waits and reads both outputs. The physical
BF16 output is stored in register-pair halves; it is not dense host BF16.

## Check before handing off

`build.py` runs native compilation, stage verification and C export. It records
the exact commands, durations and logs beside the products; it does not execute
the program. The profilers execute the emitted words on the pinned model:

```sh
for rows in 64 128; do
  "$COMPILER_PYTHON" tools/profile_looped_model.py \
    --compiled "$RUN/matmul-$rows/compiler" --io-memory-bytes 1048576 \
    --warmup 1 --repetitions 3 --out "$RUN/matmul-$rows-model.json"
done
"$COMPILER_PYTHON" tools/profile_resident_model.py \
  --compiled "$RUN/attention/compiler" --compile-inputs "$RUN/attention-inputs" \
  --policy test/examples/attention_mlp_graph_scaled_arithmetic.json \
  --io-memory-bytes 1048576 --warmup 1 --repetitions 3 \
  --profile-host-runtime --out "$RUN/attention-model.json"
```

These runs check two public input sets, repeated invocations, outputs and
surrounding memory. Matrix-product inputs are sparse signed witnesses; they do
not exhaust the arithmetic domain. The attention checker also checks stored
intermediates. Optional host timings use native C with Python MMIO callbacks;
they are not Shuttle or FPGA timings.

Cross-compile every exported C source as C11 with the application's `lp64d`
ABI. The [Baremetal-IDE guide](../../../docs/baremetal-c-integration.md) covers
the platform boundary. Do not link multiple copies of `atlas_host.c`.

## Board handoff

Give the team each `c/` directory and its measurement report, plus the selected
source revisions and `runtime/selected-host-map.json`. Keep `compiler/` and
`commands.json` available for debugging; a handoff archive is not required.

The selected EE290 source map has IMEM at `0x20000` (128 KiB), CSRs at
`0x40000`, and launch at CSR offset `0x18`. The actual bitstream ID, reserved
DMA ranges and cache services still need confirmation against the running
board. Relinking host C does not relocate the device's fixed DMA addresses.

The platform must provide synchronous DRAM read/write callbacks and cache
clean/invalidate hooks, or establish that the selected memory is coherent.
A fence alone does not clean a cache. After a timeout, drain or reset the
outstanding launch before reusing buffers. Keep one owner of Atlas IMEM.

For the integrated run, record the build ID, compiler revisions, image hash,
memory map, clock units, cold load time, warm launches, completion status and
output/guard comparisons for two inputs and repeated invocation. Device-model
cycles and standalone AtlasCore results do not qualify the integrated board.

### Return a board result

Before the run, provide the bitstream/build identifier, the generated Chipyard
memory map, and the cache-hook function names with their source revision. The
cache hooks must cover the generated DMA ranges and complete before returning.
Record whether the host linker reserves those ranges against code, stack and
other allocations. These facts cannot be inferred from the EE290 config name.

Run one cold invocation followed by five warm invocations, alternating the two
public input sets. Keep one initialized device context; check outputs and guards
after every call. Report:

| Item | Record |
| --- | --- |
| Program | Image SHA-256, case name, compiler revisions |
| Platform | Build ID, memory-map file, cache-hook source, host ABI |
| Clock | Counter used, unit, frequency if converting cycles to time |
| End-to-end time | Before packing/loading through completed output readback |
| Runtime time | Load, start and wait counter deltas, measured separately |
| Residency | Image-load, image-reuse and launch counter deltas |
| Correctness | Both input sets, outputs, guards, completion/error status |

The existing `atlas_host_set_clock` hook fills cumulative `host.timings` and
residency counters; take before/after differences. It does not include all
wrapper packing and readback work, so also time the complete call externally.
Label Shuttle `rdcycle` measurements as host cycles, not Atlas execution cycles.
Preserve individual samples rather than reporting only the best run.

## Current larger-MLP limitation

The separately authored two-layer MLP input with width 128 and hidden width
512 reaches a backend refusal in both the old and updated tool build: scalar
spilling can hide a DMA pointer's bounds from the verifier. This affects the
32-row and 64-row composed cases tested here. Their failed compilations are
not part of the runnable handoff. Fixing that proof is a backend task; these
examples do not bypass it or replace it with unchecked code.
