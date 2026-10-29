# Task accuracy

The full ResNet50, TinyLlama, and SmolVLA hosts are bit-exact against the Atlas arithmetic: every launch matches a reference that follows the MXU's FP8 E4M3 operands, power-of-two tensor scales, exact products, and BF16 rounding after every multiply-add in ascending K. That is the correctness claim for the compiler, runtime, and host code. It is not a claim that the outputs match the original models in PyTorch.

Because the arithmetic is coarser than PyTorch's, task outputs drift. On one input each:

| Model | PyTorch | Atlas |
| --- | --- | --- |
| TinyLlama, 128-token prompt | Tokens [278, 3711, 5652] | Tokens [278, 14263, 658] |
| SmolVLA policy | Reference actions | 20.4% action relative L2 error |
| ResNet50, church image | Class 497 | Class 562 |

ResNet50 matches PyTorch's top-1 on the three other images checked. An independent CPU emulation of the Atlas arithmetic reproduces all three results, so the gap comes from the number format rather than the compiler or host code. BF16 accumulation over long K dominates for TinyLlama; FP8 weight precision dominates for ResNet50; SmolVLA has both.

## Opt-in precision-mode emulation

`tools/precision_modes/` packages the CPU experiments as one optional tool. It reads the PR #26 captures and local checkpoints, recomputes the model forward pass, and compares the result with native PyTorch. It is offline emulation, not an Atlas run. It does not compile, launch, or modify device programs. No default compiler, runtime, program, or ELF changes, and no production module imports this package.

All recipes retain the BF16 operand front cast, Atlas E4M3 packing with flushed exponent-zero values, ascending-K exact products, and BF16 nearest-even rounding after every MAC. The tool admits normal finite BF16 results and rejects exceptional final stores. `--precision-mode default` reproduces the frozen arithmetic. It is also the CLI default; a different recipe requires an explicit flag.

- `finer-scales` selects one power-of-two scale per activation row and per weight output column, constant across the entire K dimension. Tensor, row, M32-panel, column, and N32-panel scale variants are available through the scale flags. They emulate prepacking and a host output factor, not a native per-lane scale register. Scale-only recipes add no contraction chains.
- `k-split` starts a fresh +0 BF16 accumulator for each K chunk, pads only the final K32 tail, widens the raw partials, applies power-of-two scaling, and adds them in ascending chunk order in FP64. One direct FP64-to-BF16 nearest-even conversion follows the merge, avoiding a float32 double rounding. `--k-chunk N` must be a positive multiple of 32 and defaults to 128. K split adds partial stores, host work, and separate launches unless a future program combines them.
- `two-term` keeps the original tensor-scaled FP8 weight as the high term. It subtracts that decoded, scaled high term from the checkpoint's FP32 weight in FP64, then applies the same BF16-front FP8 encoder to the residual. Each contraction has fresh high and low BF16 chains, added high then low in FP64 before one direct BF16 rounding. The weight-scale override applies to the low term. Applying this everywhere doubles the MXU work and encoded weight storage.

`--finer-scales` combines finer scales with another recipe. `--activation-scale tensor` or `--weight-scale tensor` can retain only one finer axis, which matters for SmolVLA. `--sites` accepts comma-separated exact names or globs, or `@file` containing whitespace-separated names or a JSON list. TinyLlama also accepts `head`, `attention-head`, `mlp`, and `down`. A masked K split with `--finer-scales` keeps finer scales everywhere and splits only selected sites. A masked two-term recipe leaves unselected sites at the frozen default.

## Running the tool

Use a CPU environment with NumPy and PyTorch. TinyLlama also needs safetensors and compatible Transformers code, SmolVLA needs its compatible LeRobot and Transformers code, and ResNet50 needs torchvision. Supply local model files and prepared inputs; the drivers do not download models or tokenize prompts. Native-reference hashes and default gates can reveal a dependency version that differs from the capture environment. `--model-source` supplies an optional LeRobot source root; an existing TinyLlama Transformers overlay can be supplied on `PYTHONPATH`.

Build the single ordered-MAC helper explicitly. It has no numerical thread pool and is kept for model-evaluation speed. This is separate from the compiler build:

```sh
mkdir -p build/precision_modes
c++ -O3 -shared -fPIC -fno-fast-math -ffp-contract=off \
  tools/precision_modes/ordered_mac.cpp \
  -o build/precision_modes/ordered_mac.so
export ATLAS_PRECISION_MAC_LIBRARY="$PWD/build/precision_modes/ordered_mac.so"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
```

The following paths are caller-provided. `TINY_CAPTURE` is the `arithmetic-capture-v2` directory containing `receipt.json`, `SMOL_REFERENCE` and `SMOL_NATIVE_REFERENCE` are the replay and native trajectory NPZ files, and `RESNET_PROVIDER` contains `host-plan.json` and its packed weights and host parameters. `RESNET_REFERENCE` is the arithmetic-reference receipt for the matching image. Each checkpoint must match its run artifacts.

These commands run the full frozen default gates and fresh PyTorch comparisons:

```sh
python tools/precision_modes/cli.py tiny \
  --capture "$TINY_CAPTURE" --checkpoint "$TINY_CHECKPOINT" \
  --output tiny-default.json
python tools/precision_modes/cli.py smol \
  --checkpoint "$SMOL_CHECKPOINT" --backbone "$SMOL_BACKBONE" \
  --observation "$SMOL_OBSERVATION" --reference "$SMOL_REFERENCE" \
  --native-reference "$SMOL_NATIVE_REFERENCE" --output smol-default.json
python tools/precision_modes/cli.py resnet \
  --artifact-dir "$RESNET_PROVIDER" --checkpoint "$RESNET_CHECKPOINT" \
  --input "$RESNET_INPUT" --reference "$RESNET_REFERENCE" \
  --output resnet-default.json
```

The TinyLlama gate compares all three 32,000-element logit vectors and tokens [278, 14263, 658]. The SmolVLA gate compares all 47 trajectory fields, including 300 actions. The ResNet50 gate compares 54 activation payloads, scale bindings, and contraction outputs, plus all 1,000 final logits. ResNet50 retains the archived average-pool and flatten intermediate FP32 division-order difference; those two intermediate hashes are reported, while contractions and final logits must still match exactly. Captured outputs are comparisons only and never supply forward-pass values.

For candidates, use the same arguments and append a mode flag. These are examples of the measured recipes:

```sh
# TinyLlama, scale-only or K128 plus finer scales.
python tools/precision_modes/cli.py tiny \
  --capture "$TINY_CAPTURE" --checkpoint "$TINY_CHECKPOINT" \
  --precision-mode finer-scales --output tiny-finer.json
python tools/precision_modes/cli.py tiny \
  --capture "$TINY_CAPTURE" --checkpoint "$TINY_CHECKPOINT" \
  --precision-mode k-split --k-chunk 128 --finer-scales \
  --output tiny-k128-finer.json

# SmolVLA, K128 with tensor scales, or activation-row scales alone.
python tools/precision_modes/cli.py smol \
  --checkpoint "$SMOL_CHECKPOINT" --backbone "$SMOL_BACKBONE" \
  --observation "$SMOL_OBSERVATION" \
  --precision-mode k-split --k-chunk 128 --output smol-k128.json
python tools/precision_modes/cli.py smol \
  --checkpoint "$SMOL_CHECKPOINT" --backbone "$SMOL_BACKBONE" \
  --observation "$SMOL_OBSERVATION" \
  --precision-mode finer-scales --weight-scale tensor \
  --output smol-row.json

# ResNet50, two terms at every site or the frozen half-work subset.
python tools/precision_modes/cli.py resnet \
  --artifact-dir "$RESNET_PROVIDER" --checkpoint "$RESNET_CHECKPOINT" \
  --input "$RESNET_INPUT" --precision-mode two-term \
  --output resnet-two-term.json
python tools/precision_modes/cli.py resnet \
  --artifact-dir "$RESNET_PROVIDER" --checkpoint "$RESNET_CHECKPOINT" \
  --input "$RESNET_INPUT" --precision-mode two-term \
  --sites @tools/precision_modes/recipes/resnet_half_work.txt \
  --output resnet-half-work.json
```

TinyLlama accepts `--inputs prompts.json --steps 8`, where the JSON is `{"prompts": [{"id": "name", "token_ids": [1, 2, 3]}]}`. It reports free-greedy tokens and forced native-history decisions. Use the same eight frozen 128-token inputs to reproduce the cohort below. Add `--sites attention-head` to K128 plus finer scales for the exploratory attention and head mask. SmolVLA takes one prepared observation NPZ per invocation with `images`, `image_masks`, `language_tokens`, `language_mask`, `state`, and explicit `noise`; repeat the command for each of the five frozen observations. Its adapter targets the frozen SO100 configuration with fifty timesteps and six actions. ResNet50 accepts raw float32 NCHW image bytes, `.npy`, or an NPZ `images` or `inputs` array with shape `[N, 3, 224, 224]`, plus an optional JSON label list through `--labels`. A single reference receipt must match the selected image; `--input-index` selects one image from a batch.

`--output` writes JSON metrics, and `--output-dir` also retains numerical outputs. `python -m tools.precision_modes` is an equivalent entry point. Small tests run through the existing CTest Python discovery or `python -m unittest discover -s test -p test_precision_modes.py`; they use synthetic Fraction controls and one frozen full-K contraction cell per model. Full model evaluations are the explicit commands above, not CTest tests. The repository does not include checkpoints, datasets, or full captures.

## Multi-input results

The packaged tool reruns use one declared FP64 merge for all models. The earlier SmolVLA experiment used an FP32 addition after every scaled partial. With the shared FP64 merge, all 47 saved trajectory fields, including 300 actions per observation, remain byte-identical for every rerun mode. No reported metric changes. These are bounded offline checks, with no device execution, timing, robot-success, or general benchmark claim.

TinyLlama, eight 128-token prompts and eight generated tokens per prompt, compared with native FP32:

| Mode | Complete continuations matching PyTorch | Forced top-1, of 64 | Pooled prefill RMSE |
| --- | ---: | ---: | ---: |
| Default | 0/8 | 43 | 0.423069936 |
| Finer scales | 2/8 | 52 | 0.428153745 |
| K128 plus finer scales | 6/8 | 62 | 0.207920419 |

Finer scales alone restore the frozen prompt's tokens but match only 2 of 8 complete new continuations and raise prefill RMSE on 3 of 8 prompts. K128 plus finer scales matches 6 of 8 and lowers prefill RMSE on all eight. The earlier exploratory attention and head mask also reached 6 of 8, at 5,520 rather than 17,136 estimated separate launches for the frozen three-phase topology, with higher pooled RMSE (0.312 rather than 0.208). That mask was chosen after seeing these prompts and needs an independent input set.

SmolVLA, five related TRAIN observations from episode 0 with the same explicit noise:

| Mode | Mean action relative L2 | Lower error than default |
| --- | ---: | ---: |
| Default | 14.203659% | n/a |
| K128 | 7.263688% | 5/5 |
| K512 | 8.749519% | 4/5 |
| Activation-row scales | 12.555950% | 3/5 |
| K512 plus activation-row scales | 9.525341% | 4/5 |

K128 has the lowest cohort mean among the tested precision modes and beats default on all five observations. It is not the per-observation winner on all five: K512 has lower error on the original observation. That single-input choice did not give the lowest five-input mean. Only the original observation has an external full-run gate. The four additional observations retain the original preparation semantics, including an unresolved frame-0 image-preparation difference of up to 2.38e-7; the frozen original input is used for its gate. Per-column weight scales hurt the original SmolVLA input and were not retested on the other four.

ResNet50, 100 frozen Imagenette images, with native PyTorch label top-1 accuracy of 88%:

| Mode | Top-1 agreement with PyTorch | Label top-1 | Mean logit relative L2 |
| --- | ---: | ---: | ---: |
| Default | 94% | 86% | 0.30102857 |
| Activation-row and weight-column scales | 95% | 86% | 0.29933889 |
| Two terms, quarter-work subset | 96% | 87% | 0.21972690 |
| Two terms, half-work subset | 98% | 88% | 0.21346686 |
| Two terms everywhere | 97% | 89% | 0.20936543 |

The half-work recipe gives 98% agreement and the same 88% label accuracy as PyTorch. Its 35 selected contractions out of 54 add 49.74% MXU work and 38.19% padded weight bytes; half-work refers to instruction cost, not exactly half the site count. Sites were chosen using only four development images and were frozen before the 100-image evaluation. The development defaults reproduce archived contraction and logit bytes; the 100-image cohort has no separate Atlas-run reference. These work estimates exclude packing, scale metadata, host merging, scheduling, and ownership costs, and establish no target speedup.
