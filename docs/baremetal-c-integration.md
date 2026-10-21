# Use generated C with Baremetal-IDE

Merlin compiles the tensor graph and emits host dispatch code. Atlas MLIR emits
Atlas instructions. The deliverable is a C source/header pair, device binaries,
constants, and their manifest. Baremetal-IDE supplies startup, a linker script,
platform callbacks, and the final host ELF.

## Build the compiler

Use the matching `feat/atlas-model-compilation` branches in
[Merlin](https://github.com/ucb-bar/merlin/tree/feat/atlas-model-compilation)
and [Atlas MLIR](https://github.com/ucb-bar/atlas-mlir/tree/feat/atlas-model-compilation).
For layer programs with one image and one launch, start with the
[single-image examples](../examples/handoff/resident_layers/README.md).

Build `atlas-opt` and `atlas-emit` using the LLVM/MLIR SDK described in the
[repository README](../README.md). Install Merlin and this repository's Python
package in the compiler environment. Cargo builds the pinned native selector.
See [native compilation](native-merlin-binding.md) for dependencies.

Use fresh output directories. Generated files belong under `out/`, and are not
source files to commit.

## Compile the attention + MLP fixture

For a captured attention + MLP and larger matrix products, start with the
[single-image layer examples](../examples/handoff/resident_layers/README.md).
They use the same external-memory configuration for compilation, export and
profiling, and distinguish data-only C exports from callable C wrappers.

From this repository:

```sh
ATLAS_GRAPH_PRECISION_POLICY="$PWD/test/examples/attention_mlp_graph_finite_zero_arithmetic.json" \
examples/handoff/captured_attention_mlp/build_c_package.sh \
  "$MERLIN_ROOT" "$COMPILER_PYTHON" "$ATLAS_BIN" out/attention-mlp optimized
```

`ATLAS_BIN` contains `atlas-opt` and `atlas-emit`. `COMPILER_PYTHON` is the
interpreter where both compiler packages are installed. The script builds a
native snapshot, compiles the two-output Linalg fixture, checks the emitted
stages, and exports `out/attention-mlp/c-package/attention_mlp.c` and `.h`.
The fixture exposes its weight tensors as arguments. It is useful for inspecting
the compiler without provisioning a new frontend capture.

The package also contains `attention_mlp.bin`, hexadecimal instruction words,
`constants/`, and `manifest.json`. The binary is an Atlas instruction stream,
not a host executable. The C source embeds the same bytes.

## Compile a captured application

### One image and one launch

For a supported 32×32 attention + MLP capture, use the whole-graph route. It
keeps intermediate conversions and layouts on Atlas and emits one instruction
image. Only input conversion/packing, dispatch and output readback run on the
CPU. Parameters are embedded from the staged capture:

```sh
examples/handoff/captured_attention_mlp/build_c_package.sh \
  "$MERLIN_ROOT" "$COMPILER_PYTHON" "$ATLAS_BIN" out/resident-attention-mlp \
  optimized "$COMPILE_INPUTS"
```

`COMPILE_INPUTS` is Merlin's verified compiler-only capture stage. This route
uses the explicit finite/zero FP8/BF16 whole-graph policy, which differs from the
per-region application's conversion policy. Report original-model error
separately. Unsupported shapes or operations fail without producing several
launches.

Give the platform team `c-package/attention_mlp.c`, `.h`, `.bin`, `constants/`
and `manifest.json`. After `attention_mlp_bind`, call `attention_mlp_model_run`
with the runtime input and two output buffers in the header's order. The first
call loads the image; later calls reuse it while the same device context owns
IMEM. See [single-program profiling](single-program-profiling.md) for measurements.
Compile the pair as C11 with `-DATLAS_DIAGNOSTIC_GRAPH_PRECISION=1`; the generated
README lists its C library requirements and precision limitations.

## Resident program data

For a connected graph with compact matrix loops, export the image and its
buffer descriptions using the same staged source and precision policy used
for compilation:

```sh
python tools/export_resident_c.py \
  --compiled "$COMPILED" --compile-inputs "$COMPILE_INPUTS" \
  --policy test/examples/attention_mlp_graph_scaled_arithmetic.json \
  --name attention_mlp_image --out out/resident-image
```

This produces `attention_mlp_image.c` and `.h`, `program.bin`, `data/`, and
`manifest.json`. The C source embeds the image and static parameters. It exports
instruction words, static data segments, ordered input/output descriptions and
the device buffer list. It contains data declarations; the platform application
supplies its driver and invocation code. The manifest records logical shapes,
physical layouts, exact input conversion and artifact hashes.

The driver sequence is:

1. Reserve the recorded DRAM ranges and load or reuse `attention_mlp_image_program`.
2. Copy each `attention_mlp_image_static_data` segment to its recorded address.
3. Write the runtime inputs to `attention_mlp_image_inputs` in their declared
   packed FP8 representation. The recorded conversion includes any power-of-two
   input scale; copying source f32 bytes would be incorrect.
4. Launch once and wait for completion. With the supplied runtime, use
   `atlas_host_ensure_loaded`, `atlas_host_start` with the generated device
   buffer list, and `atlas_host_wait`. Link `runtime/atlas_host.c` once if using
   those functions, or use the platform's corresponding driver operations.
5. Read the ordered outputs. Each BF16 matrix uses M/N tile order; each 32×32
   tile stores columns 0–15 followed by columns 16–31. The private arena also
   contains intermediate tensors and remains reserved through completion.

The same binary and data files are available separately for inspection or
direct loading. Runtime samples and expected outputs are excluded. Compile the
C source with the platform's `lp64d` ABI. Baremetal-IDE owns startup, linker
placement, cache maintenance, the final ELF and loading it onto the board.
The generated addresses must match the selected system's DMA-visible map.

See [resident profiling](single-program-profiling.md#connected-graphs-with-a-larger-mlp)
for the supported matrix shapes, model checks and RTL fixture generation.

### Explicitly partitioned applications

To embed model parameters and accept only runtime tensors, capture and stage
`examples/workloads/attention_mlp/loader.py` using Merlin's
[workload instructions](https://github.com/ucb-bar/merlin/blob/feat/atlas-model-compilation/examples/workloads/attention_mlp/README.md).
Capture needs a compatible PyTorch/Model2MLIR environment; the native compiler
and the frontend may use separate interpreters.

For the projected attention + MLP application, select its explicit diagnostic
precision settings and run:

```sh
ATLAS_NATIVE_PIPELINE=optimized \
ATLAS_OPERAND_SCALE_EXPONENTS='[0,0]' \
ATLAS_CONTRACTION_EPILOGUES='["relu"]' \
ATLAS_POINTWISE_POLICY=f32-to-bf16-bounded-pointwise-rne-chop-v1 \
MERLIN_APPLICATION_NAME=attention \
examples/handoff/captured_attention_core/build_c_package.sh \
  "$MERLIN_ROOT" "$COMPILER_PYTHON" "$ATLAS_BIN" "$CAPTURE" out/attention-application
```

`CAPTURE` is the completed frontend bundle, not a runtime input directory.
The script stages compiler-only inputs before compilation. The policy selects
FP8 operands, BF16 intermediate arithmetic, and explicit conversions. It does
not establish equivalence to the original floating-point model; report that
error separately. Runtime tensors and expected results are not compiler inputs.

Application precision policy version 12 can additionally select CPU predicates:

```json
"host_predicates": ["f32_nonzero", "mask", "select"]
```

Choose only the needed entries. These permit the matching source forms through
Merlin's predicate C emitter: f32 nonzero, signed-i64 comparisons, Boolean
operations/widening, ordered f32/f64 comparisons, and bit-preserving selection.
Ordered comparisons treat both signed zeros as equal and return false when either
operand is NaN. Unordered predicates remain unsupported. Earlier policy versions do not admit these CPU operations.
A missing emitter is an explicit tool error; device failures never fall back to
CPU computation. Declared constant operands can fold to initialized data, with
hashes retained in the region description. This is a diagnostic application
extension, not full-model coverage or an accuracy qualification.

Policy version 13 retains version 12 fields and additionally accepts an explicit
`host_integer_ops` list. Available entries are `i64_add`, `i64_sub`, `i64_mul`,
`i64_reduce_add`, `i64_cumsum`, `signed_i64_to_f32`, and `signed_i64_to_f64`.
Choose only operations permitted to execute on the CPU. Merlin emits exact
wrapping arithmetic and integer-only round-to-nearest-even conversion in the
application C/H; the platform does not need a Python interpreter. Atlas regions
remain on the accelerator. Earlier policies continue to refuse integer CPU
placement, and a device compiler failure never selects it automatically.

Policy version 14 retains version 13 fields and permits `"bucketize"` in
`host_predicates`. Select it only when CPU boundary counting is allowed. Merlin
emits a checked binary search matching the source lower/upper-bound count,
including duplicate boundaries and NaN queries. Unsorted or NaN boundaries fail
before output writes. This remains CPU computation; it does not establish
device support for the operation. Earlier policies continue to reject it.

Policy version 15 adds `guarded_masked_softmax` with value
`bf16-tiled-masked-row-softmax-empty-zero-v1`. It recognizes a Boolean exclusion
mask followed by stable softmax and a row-maximum guard. Mask bytes are zero
(keep) or one (exclude). Fully masked rows produce positive zero on Atlas.
The device floors the denominator by one before reciprocal; nonempty rows
retain the existing arithmetic. Scores still need the declared bounded BF16
conversion. The generated C/H accepts an opaque input-pointer array and records
the separate f32 score and i1 mask types. Earlier policies retain their existing
mask behavior and do not admit this source form.

Softmax row groups execute in a device-side loop with one image and one launch
per softmax region. The generated C stages all row groups before launch and
collects them after completion. Reserve the entire reported device arena,
including private per-group scratch; the CPU workspace is a separate allocation.
The raw row-softmax export command also accepts earlier separate-row packages.

The minimal directory to give the platform team is
`out/attention-application/compiled/handoff/`. It contains the C/H pair,
`programs/`, `constants/`, `abi.json`, and its own README. The compiler retains
the full stage records separately. The inspectable pair is also available at:

```
out/attention-application/compiled/c-package/combined/attention_combined.c
out/attention-application/compiled/c-package/combined/attention_combined.h
```

Include the header and compile only that C source. Do not also compile the
individual region sources: the combined source already includes their code and
one runtime. `manifest.json` identifies the pair and each device binary under
`segments/`. `application_interface.platform_binding` records the generated
context, initialization, invocation, and recovery functions.

`attention_context` includes region contexts, shared private scratch, and tensor
workspace. Zero-initialize it and prefer static storage on a bare-metal system.
After initializing one `atlas_host` with the board's map, MMIO callbacks, and
cache policy, call:

```c
static attention_context application;
/* host and memory_io are initialized by the platform. */
int status = attention_bind(&application, &host, memory_io, max_polls);
const void *inputs[] = {tokens};
void *outputs[] = {hidden, result};
if (!status)
  status = attention_run(&application, inputs, outputs);
```

The generated header and manifest define tensor order, shapes, types and sizes.
The captured example has one runtime input and two outputs; other graphs need
their own recorded interface. Parameters are embedded separately from runtime
inputs. Outputs are published after the entire dispatch succeeds.

## Platform responsibilities

- Compile the pair as C11 using the application's ABI: **`lp64d`** for Shuttle.
  The generated code requires ordinary memory/library functions; inspect the
  resulting object for the exact unresolved symbols.
- Add the source and include directory to the existing CMake application target.
  Baremetal-IDE chooses its startup, supported `-march`, libraries and linker.
- Provide synchronous device-visible memory read/write callbacks. Their buffers
  must cover the compiled DMA addresses; linking host data elsewhere does not
  relocate addresses embedded in Atlas instructions.
- Use the selected EE290 build's actual IMEM/CSR map and DMA-reachable memory.
  The source-derived example map has no bound bitstream identity.
- Supply cache clean/invalidate hooks unless coherence is established for that
  build. A memory fence alone does not flush a cache. Reserve the device buffers
  against host code, data, stack and other allocations.

The published Baremetal-IDE CMake builds use `lp64d`; the checked upstream main
branches do not provide an Atlas/EE290 board setup. This package does not supply
an FPGA linker script or invent a cache-maintenance sequence.

Check the generated handoff directly against the selected board map:

```sh
python tools/atlas_graph_platform.py \
  --compiled out/attention-application/compiled/handoff \
  --host-map "$HOST_MAP" --soc-memmap "$SOC_MEMMAP" --out out/platform-check.json
```

This checks each serial image separately: different calls may reuse the same
DMA ranges. It verifies packaged bytes and host-visible mapping, not DMA
reachability or cache correctness. The optional `--host-elf` checks the ELF
built by the platform team; an ELF is not required to produce this handoff.
For the required build inputs, see [host launch](host-launch.md).
The platform team can bind the package after the software-tested handoff using
the generated Chipyard memmap, build identity, and board cache implementation.

## Invocation and program reuse

Calls are synchronous. One initialized `atlas_host` owns each physical device;
share it across application contexts. Identical immutable instruction arrays
reuse the resident image after an exact word comparison. Input, constant, state,
and cache operations still run as required. Different images replace each other
only while idle. Device branches are resolved within each image; there are no
cross-image calls or multi-entry IMEM linking.

A timeout leaves device work active. Keep contexts, buffers and scratch alive;
finish the launch or complete a platform reset before reuse. After reset, notify
`atlas_host_reset_completed`, then use `attention_recover` to discard the failed
application dispatch. Before other idle IMEM writes, invalidate residency with
`atlas_host_invalidate_program`.

Model checks, standalone RTL checks, and an `lp64d` object build are separate
from an integrated FPGA run. Record the binary hash, inputs, build identity,
outputs, guards and repeated invocation results when the board is exercised.
