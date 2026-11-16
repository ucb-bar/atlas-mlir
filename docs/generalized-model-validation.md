# Shared-pass validation: TinyLlama, SmolVLA and ResNet-50

This extends the [full-width TinyLlama layer integration](generalized-tinyllama-layer.md)
using the same Merlin outlining, upstream host lowering and Atlas counted-K
programs. No OOT generation or optimization-agent loop was run. The complete
change inventory is in the companion Merlin branch's
`examples/atlas/generalization-changelog.md`.

## What was exercised

| Synthetic fixture | Device contractions | Tiles per mode | Source-order device cycles | Optimized device cycles |
|---|---:|---:|---:|---:|
| Scaled whole-Llama PoC: width 128, intermediate 344, two layers, sequence 8 | 19 | 108 | 940,580 | 909,548 |
| Reduced SmolVLA prefix/expert denoising step | 28 | 37 | 165,181 | 160,245 |
| Complete torchvision ResNet-50 at 32×32 | 54 | 924 | 45,974,812 | 44,227,900 |

All 2,138 tiles match independently implemented ordered E4M3/BF16 arithmetic
exactly, and model outputs are bit-identical between scheduling modes. The
original f32 compiled host graph also matches Torch within rtol/atol 2e-4.
Approximate device-path outputs match the same host image using independent
software contraction callbacks bit-for-bit. This checks transport/tiling/device
arithmetic; it is not an independent reconstruction of the entire approximate
graph. Maximum absolute error against original f32 Torch is respectively
0.180702, 0.0962462 and 8.08252. Synthetic weights and these error values do not
qualify pretrained application accuracy.

ResNet-50 at **224×224 compiles in both modes**, with all 54 contractions outlined;
that input size has not been simulator-executed. The 32×32 run retains all 50
layers, batchnorm, residuals, pooling and classifier. Existing model2MLIR
convolution decomposition supplies im2col/matmul; no new convolution lowering
was added. The reduced SmolVLA recipe covers a prefix and an expert step, not
its complete pretrained multimodal policy. The scaled Llama PoC covers integer
embedding IDs, two layers and logits; the earlier full-width single-layer result
remains separate evidence, with its documented FP8 midpoint sensitivity.

All remaining operators execute in the compiled host graph. Cycles sum
npu_model device launches and exclude host, quantization, transfer and dispatch
costs. They are neither end-to-end latency nor a generated-backend comparison.
The model revision is `47b8599d8ffbe5ddd96059ffe2b5be2427a410f5`, selected RTL
identity is `0079c0541111197741a231c002e3843fa6f545b2`, and LLVM revision is
`a47bddccec30255619bb8c37fa59700e661d4e66`.

## Shared changes

- Recognize canonical rank-two/three `linalg.generic` contractions by maps and
  iterators, then check the complete multiply/add body and seed. This moves all
  six SmolVLA attention contractions and four scaled-Llama attention contractions
  through the same path as named matmul. Unknown valid geometry remains host work.
- Accept literal positive-zero `tensor.splat` seeds through the existing reader;
  negative zero, nonzero, fastmath and wrong recurrences still refuse.
- Describe typed static host arguments/results in Merlin, including integer
  IDs, boolean masks and rank-zero counters. Runtime dtype and shape validation
  precedes native invocation. This does not widen the f32 device contract.
- Reuse the existing simulator transport for arbitrary captured models, retaining
  the TinyLlama-specific independent host-partition checker.

The compiler never chooses behavior by model name. Model selection in the
capture script is test fixture selection only. Existing FP4/FP6 storage support
is unchanged; these Atlas programs execute E4M3/BF16.

## Reproduce

Use the environment and private compiler build described in the layer document.
The PoC capture environment additionally needs model2MLIR (`m2m`), safetensors,
Torch and torchvision. The tested capture used Torch 2.11.0 and torchvision
0.26.0+cpu. Set `M2M_ROOT` to its source checkout and `POC_PY` to this environment.
Use fresh destinations. Set `case` to `tinyllama_poc`, `smolvla` or `resnet50`.

```sh
case=resnet50
export POC_DIR="$ATLAS_ROOT/out/build/model-check/$case"
PYTHONPATH="$MERLIN_ROOT/src:$M2M_ROOT" "$POC_PY" \
  test/models/capture_shared_poc.py "$case" --merlin-root "$MERLIN_ROOT" \
  --destination "$POC_DIR/capture" --image-size 32

for pipeline in source-order optimized; do
  PYTHONPATH="$MERLIN_ROOT/src:$ATLAS_ROOT/python" "$COMPILER_PY" \
    -m atlas_mlir_native.shared_model "$POC_DIR/capture/model.mlir" \
    --entry forward --precision f32-e4m3-rne-saturate-flush-ordered-bf16-padded-v1 \
    --destination "$POC_DIR/$pipeline" \
    --atlas-bin-dir out/build/layer/compiler/bin \
    --llc "$LLVM_ROOT/bin/llc" --cc /usr/bin/gcc --pipeline "$pipeline"
done
```

Build the original f32 graph as a separate reference library, using the same
existing Merlin host compiler with no callbacks:

```sh
PYTHONPATH="$MERLIN_ROOT/src" "$COMPILER_PY" - <<'PY'
import os
from pathlib import Path
from xdsl.dialects.builtin import UnitAttr
from merlin.frontends.linalg_text import parse_mlir_text
from merlin.llvmlower.floating_matmul_outline import compile_callback_host
from merlin.xdsl_dialects._common import text
root = Path(os.environ['POC_DIR'])
module = parse_mlir_text((root / 'capture/model.mlir').read_text())
module.body.block.first_op.attributes['llvm.emit_c_interface'] = UnitAttr()
destination = root / 'f32-host'
destination.mkdir()
compile_callback_host(text(module, generic=True), [], destination=destination,
                      llc=os.environ['LLVM_ROOT'] + '/bin/llc', cc='/usr/bin/gcc')
PY
for pipeline in source-order optimized; do
  PYTHONPATH="$MERLIN_ROOT/src:$ATLAS_ROOT/python:$ATLAS_ROOT:$MODEL_ROOT" \
    "$MODEL_PY" test/run_shared_model.py "$POC_DIR/$pipeline" \
    "$POC_DIR/capture" "$POC_DIR/run-$pipeline" \
    --f32-host "$POC_DIR/f32-host" --workers 4
done
```

Use `--image-size 224` in a separate capture for the standard ResNet compile
check. Test scripts consume original Torch outputs only for validation, never
for compilation. Capture records include source and recipe hashes; compiled
manifests bind tools, source, programs and schedules. Local evidence is under
`out/build/model-portfolio`, with the six selected runs enumerated in `review.json`.

The focused Merlin suite passed 176 tests. Atlas's shared dense, counted-loop,
graph and ABI suite passed 47 tests with one explicit optional model replay skip.
The tracked capture command was exercised on the scaled Llama recipe. The final
transport changes were additionally rechecked on SmolVLA and the uneven-width
TinyLlama layer. The repository-wide Merlin wiring gate retains the previously
reproduced 17 baseline failures; no ledger was relaxed.

Production transport, pretrained quality, dynamic shapes, cached decoding and
compiler-authoring admission of the whole-host helper remain unqualified.

The [shared static scheduler integration](shared-static-scheduling.md) separately
validates construction and actual delay emission for fixed matmul and VPU
programs. The full-model results above still use the existing Atlas scheduler.
