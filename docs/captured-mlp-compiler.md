# Bounded captured MLP compilation

This OOT reference can take one 32×32 PyTorch `Linear → ReLU → Linear`
capture through parsed Linalg, virtual Atlas SSA, physical Atlas MLIR,
structured/final LLVM MLIR, LLVM IR, a RISC-V object, and a linked ELF. It
stages frozen weights and biases in external DRAM slots and declares a separate
runtime input slot. The selected standalone `AtlasCore` executed the LLVM
object words for two runtime inputs after one compilation. This is an
ACT-independent diagnostic path, not Merlin's native semantic selector or a
whole-configuration compiler release.

## Source and precision boundary

The [public loader](../test/models/directed_mlp32.py) is an ordinary PyTorch
model with two biased `Linear` layers and ReLU. The selected Model2MLIR
capture produces `linalg.transpose` for each N×K weight, an explicitly
zero-filled `linalg.matmul`, a column-broadcast `linalg.generic` bias add,
and a `maximumf` ReLU generic. The importer parses the IR with xDSL and
checks indexing maps, iterator types, region bodies, shapes, the zero fill,
operand order, source argument identities, and the function return. It never
chooses code from the function name or provenance label. The
[sanitized Linalg fixture](../test/examples/captured_mlp32_linalg.mlir)
shows this structure without a machine-local weights path. The matching
[public weights fixture](../test/examples/captured_mlp32_weights.safetensors)
and [argument manifest](../test/examples/captured_mlp32_arguments.json)
allow the [numbered compiled bundle](../examples/handoff/captured_mlp/00-linalg.mlir)
to be rebuilt without the capture environment. A fresh capture of the same
loader produced identical linked executable text.

The source is f32. The [proposed numerical policy](../test/examples/atlas_mlp_numeric_policy.json)
explicitly changes it to finite-normal E4M3 operands, BF16 readout, the
selected VPU add behavior, unit-scale E8M0 interlayer pack, and MXU0 then
MXU1. It is a diagnostic precision transform, not an equality rewrite or an
approved model-quality policy. The operand packer rounds admitted normal
values to nearest even, flushes values below the smallest normal to signed
zero, clamps above the largest admitted normal, and rejects NaN/Inf. Biases
round to BF16 at the boundary; selected RTL VPU addition then sums in FP32
and chops to BF16. Internal MXU and pack arithmetic follows the selected
RTL, not PyTorch f32 semantics.

The directed loader uses exact finite values and permutation weights. For
two runtime inputs, all 1,024 output cells per invocation matched the
original PyTorch f32 output bit-for-bit after exact BF16 representation.
That is an original-model quality observation on those two inputs and a
compiled-execution observation on the selected core. It says nothing about
random weights, full FP8/BF16 domains, accuracy on real data, other shapes,
or a complete integrated EE290 SoC.

## Reproduce

Run this from the OOT checkout, with a trace-capable Model2MLIR environment,
Merlin capture worker, this OOT build, and matching LLVM/MLIR tools. Use a
fresh output directory for each invocation. The compiler-side Python packages
used in this run are pinned in
[`requirements-captured-mlp.txt`](../tools/requirements-captured-mlp.txt);
PyTorch and Model2MLIR are needed by capture/evaluation, not imported by the
compiler tool:

```sh
"$M2M_PYTHON" "$MERLIN_ROOT/src/merlin/targetgen/_m2m_capture_worker.py" \
  --m2m-dir "$MODEL2MLIR_ROOT" \
  --loader test/models/directed_mlp32.py \
  --dtype fp32 --seed 0 --out out/captured-mlp/capture

"$M2M_PYTHON" tools/compile_atlas_linalg_tile.py \
  --linalg out/captured-mlp/capture/linalg.mlir \
  --argument-manifest out/captured-mlp/capture/weights.safetensors.manifest.json \
  --weights out/captured-mlp/capture/weights.safetensors \
  --policy test/examples/atlas_mlp_numeric_policy.json \
  --atlas-bin-dir build/bin --llvm-bin-dir "$LLVM_BIN" \
  --linker "$LLD" --output-dir out/captured-mlp/compiler
```

The compiler process receives the parsed source, frozen parameter file,
argument manifest, and policy. It is not passed captured sample inputs or
goldens. `import-manifest.json` binds their hashes, source argument to DRAM
slot mapping, packed constants, emitted program hashes, and selected toolchain
identity. `atlas-virtual.mlir` preserves logical SSA uses; `atlas-machine.mlir`
contains physical registers and a checked generated schedule. The following
files expose `atlas-llvm-structured.mlir`, `atlas-llvm.mlir`, `program.ll`,
`program.s`, `program.o`, `program.elf`, selected words, a word map, and
constant slot payloads. A failed run leaves diagnostics but no success
manifest, and a nonempty output directory is refused.

Each external input slot reserves 2,048 bytes. An FP8 operand uses its first
1,024 bytes. The ABI maps source arguments and frozen parameter keys to
slots; the weight transpose is realized by packing the source N×K weight in
Atlas's N×K physical orientation. A source 32-element bias vector is
broadcast into a 2,048-byte BF16 pair tile before execution. The output is
one 2,048-byte BF16 pair tile. Runtime f32 input quantization belongs to the
launch side and is tested independently from compiler inputs. The selected
standalone core is started through the ModeLIR TileLink driver; the ELF's
void `atlas_program` function is not a C-callable host ABI.

The executable integration test is
[`test_captured_mlp_end_to_end.py`](../test/test_captured_mlp_end_to_end.py).
Set `ATLAS_M2M_PYTHON`, `ATLAS_M2M_ROOT`, `ATLAS_MERLIN_CAPTURE_WORKER`,
`ATLAS_LLVM_BIN`, `ATLAS_OOT_BIN_DIR`, `ATLAS_ARC_MODEL`, `ATLAS_ARC_STATE`,
and `ATLAS_MODELIR_ROOT` to run it. It performs a fresh capture and compile,
checks stale-output and weight-identity rejection, runs two different runtime
inputs through the same linked text, and compares all cells with evaluator-only
PyTorch outputs. The portable
[`test_captured_mlp_import.py`](../test/test_captured_mlp_import.py)
tests structural changes to maps, initialization, ReLU, policy, and metadata.

## Current boundary

Only one static 32×32 tensor output and this admitted structural Linalg
vocabulary are implemented. The compiler does not tile larger layers, handle
tails, allocate arbitrary physical aliases, choose MXUs algorithmically,
implement host partitions, or emit a complete SoC launch artifact. It uses
conservative diagnostic waits whose general timing bound is unqualified.
The selected Model2MLIR checkout used during this diagnostic had uncommitted
changes; the current worker's optional `--materialize-bundle` call was
incompatible with that Model2MLIR API. The successful capture used its normal
Linalg/weights output without the optional bundle. Reproduce against a pinned
compatible frontend before treating the capture as a release qualification.
