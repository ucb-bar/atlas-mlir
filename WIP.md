# WIP: atlas-mlir integration snapshot (wip/atlas-nov22)

## What this branch is

A work-in-progress snapshot (1446d0d) of the integration stack: the board DRAM deploy path, resident weights and the counted layer loop, host f32 math families, and more admitted SmolVLA patterns. A cleaned-up history with nearly the same final tree will replace it on `feat/quantized-model-deployment` (https://github.com/ucb-bar/atlas-mlir/pull/53). Never rewrite or force-push this branch. Put new work in commits on your own branch off `wip/atlas-nov22`, so it moves over later with `git rebase --onto feat/quantized-model-deployment wip/atlas-nov22 <branch>`.

## Setup

Follow README.md "Build". You need Python 3.12 or newer for Atlas and Merlin, a separate Python 3.14 environment for npu_model, MLIR and LLVM from one install (tested at a47bddcc), CMake, Ninja, and a C/C++ compiler. The `capture` extra installs the pinned model2MLIR (d4bcb84) and torch 2.11.0+cpu, which comes from the PyTorch CPU wheel index.

```sh
git clone -b wip/atlas-nov22 https://github.com/ucb-bar/atlas-mlir.git && cd atlas-mlir
python3.12 -m venv .venv && . .venv/bin/activate
python -m pip install -e '.[test,capture]' --extra-index-url https://download.pytorch.org/whl/cpu
python -c "import importlib.metadata as m; print(m.version('z3-solver'))"   # must print 4.16.0.0
cmake -S . -B build -G Ninja -DMLIR_DIR=/path/to/llvm-install/lib/cmake/mlir -DCMAKE_BUILD_TYPE=Release \
  -DPython3_EXECUTABLE="$(command -v python)" -DATLAS_NPU_MODEL_ROOT=/path/to/npu_model
cmake --build build --parallel 2   # builds build/bin/atlas-opt and build/bin/atlas-emit
```

pyproject.toml pins Merlin e563d58ce by its archive URL on https://github.com/reed-nicolas/merlin (the final push pins the ucb-bar archive) with the `semantic-search` extra, which pins z3-solver 4.16.0.0. Keep that version: Merlin counts Z3 work during placement, and 5.x changes results. If your package index lacks 4.16.0.0, install the wheel file directly with `--no-index`. That Merlin snapshot sits on `refactor/repository-layout` 2451b8134 and already includes the fixes from https://github.com/ucb-bar/merlin/pull/66; the final push rebases it onto PR 66 itself. To edit Merlin alongside Atlas, clone https://github.com/reed-nicolas/merlin at `wip/atlas-nov22` (the same commit) and `pip install -e '../merlin[semantic-search]'`.

npu_model is the numerical reference (tested at b97b2f5, the pin in `python/atlas_mlir/provenance.py`). README.md gives the `uv sync --locked ... --python 3.14` step for the npu-model environment. `tools/requirements-virtual-evaluator.txt` pins xdsl 0.65.0 and says it belongs in that environment, so do not install it into the Atlas venv (README.md's combined pip line would conflict with xdsl 0.68.0).

```sh
export ATLAS_OOT_BIN_DIR=$PWD/build/bin MERLIN_NATIVE_ROOT=/path/to/merlin NPU_MODEL_ROOT=/path/to/npu_model
export ATLAS_NPU_MODEL_ROOT=$NPU_MODEL_ROOT ATLAS_LLVM_BIN=/path/to/llvm-install/bin TMPDIR=$PWD/out/work
export PYTHONPATH="$MERLIN_NATIVE_ROOT/src:$MERLIN_NATIVE_ROOT:$NPU_MODEL_ROOT${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$TMPDIR" && ctest --test-dir build --output-on-failure
```

## Rules

Models reach Atlas only through atlas-torchao FX to model2MLIR to Merlin to atlas-mlir, with no hand-written Atlas code and no change to atlas-torchao weights. In-layer loops belong in `compile_region` (`python/atlas_mlir_native/application.py`); runtime and cross-region loops belong in Merlin. Correctness means bit-exact against npu_model (its RTL numerics); never loosen a check, a refusal, or a gate to make something pass. No model names in generic code: derive every choice from shapes, layouts, numeric contracts, and declared hardware facts.

## Suggested work

You may build on any of this branch's work. The layer loop, residency, and DRAM placement are already in progress (see below), so extend them rather than redo them.

### Ideas, roughly smallest to largest

1. Double-buffered weight panels in `compile_region`'s in-layer K or N panel loop: DMA panel i+1 on a second channel while the MXU consumes panel i, each buffer set at most half the staging space. Expected effect: DMA latency hidden behind MXU work, with the same launches and bytes. Start from `python/atlas_mlir_native/quantized_loop.py`, `quantized_panel_geometry.py`, `looped_tiling.py`, `docs/loop-dma-analysis.md`, and `docs/timing-dma-cfg.md`.
2. An IMEM-resident program library per step with Atlas-side chaining: SmolVLA's 68 unique images total 75,636 bytes, under the 131,072-byte IMEM, so load them once and chain consecutive regions by Atlas jumps instead of a host restart each. Expected effect: image loads drop to one set per step and most host restarts disappear.
3. Cost-based plan selection in Merlin: choose among admitted plans by a modelled cost (launches, DMA bytes, and host work). Expected effect: fewer launches and DMA bytes where a better admitted cover exists.
4. One Atlas start per decode step: a resident kernel with mailbox handoff between the host and Atlas. Expected effect: launches per decode step go to one.

The same bar applies to each one: bit-exact against npu_model, measured before and after. Merlin-side work lands on Merlin's `wip/atlas-nov22` and needs an Atlas pin update (see that repository's WIP.md).

### Work in progress

Merlin layout-copy composition (composing private layout chains into one copy) is being built elsewhere, is not in this snapshot, and arrives with the final push. Do not start it.

Counted layer loop with resident weights. Merlin: `src/merlin/compiler/schedule/region_loop.py` finds repeated region windows (analysis only; the flat requests stay authoritative), `src/merlin/compiler/build/regions.py` and `src/merlin/compiler/lower/c/dispatch.py` run them as one counted loop from a static call table, and `src/merlin/compiler/schedule/storage/application_residency.py` with `residency.py` places immutable parameters in a resident arena. Atlas: `python/atlas_mlir_native/resident_loop_image.py`, `loop_body_binding.py`, `resident_deployment.py`, and `quantized_loop_c.py` share loop bodies through checked resident images. Residency is an optimization: an ineligible model declines it, records `application_residency_declined`, and deploys as before; `--jtag` never selects it. Next steps: fold more repeated windows into the loop (the target declares which attributes may differ), and reuse arena space across disjoint lifetimes, since the planner keeps every allocation disjoint today.

DRAM placement. Merlin `src/merlin/compiler/schedule/storage/arena.py` places buffers with the Z3 solver first, under one counted work budget per compile (`ARENA_SOLVER_WORK`), and falls back to checked first fit only when the solver returns unknown. Atlas `python/atlas_mlir/deploy.py`, `deployment_link.py`, `deployment_loader.py`, and `python/atlas_mlir_native/provider_deployment.py` measure the host by a sizing link, start the device window after it, and take every DDR fact from `python/atlas_mlir_native/penguin_sophialake.json`. Next steps: make the plans solver-decided instead of first-fit fallbacks, and replace any wall-clock limit that can change a plan with a counted budget.

## Check and measure

Focused tests (record non-passing test ids before and after your change and compare them; tests gated on the Rust npu model are not part of the gates):

```sh
python -m pytest -q test/test_quantized_loop.py test/test_quantized_loop_dram.py test/test_quantized_loop_tiles.py \
  test/test_quantized_loop_fallback.py test/test_quantized_panel_geometry.py test/test_quantized_panel_reuse.py \
  test/test_shared_weight_panels.py test/test_batch_panel_cache.py test/test_loop_dma_overlap.py test/test_looped_mnk.py \
  test/test_resident_loop_image.py test/test_resident_loop_publication.py test/test_deploy.py test/test_deployment.py \
  test/test_deployment_link.py test/test_deployment_kit.py test/test_deployment_gate.py test/test_chiprunner_composed.py tests
```

Deploys: `atlasc deploy EXPORT --jtag -o OUT` for the K32 and K64 Linear plus ReLU exports and the 112x112x3 CNN, and plain `atlasc deploy EXPORT -o OUT` for a DDR kit, then `atlasc run OUT --dry-run` (`--jtag` kits) or `atlasc run OUT --simulate`, and `ATLAS_DDR_DEPLOYMENT=$PWD/OUT python -m pytest -q test/test_chiprunner_ddr.py` for a DDR kit. `docs/deployment.md` describes the stages.

Not generic: the exports come from atlas-torchao and are in neither repository. The deploy's EXACT gate also needs `ATLAS_MODEL_CHECKER` (with `ATLAS_MODEL_CHECKER_PYTHON`), an importable `functional_runtime`, and `W1_BASELINE`, which neither repository ships or documents. It also needs `ATLAS_BAREMETAL_IDE`, CMake, `riscv64-unknown-elf-gcc`, and Cargo (or a prebuilt `MERLIN_NATIVE_SNAPSHOT`). Composed simulation adds Spike, `dtc`, `SPIKE_INCLUDE`, `COMPOSED_MODEL_PYTHON`, and `COMPOSED_RUNTIME_PATH`. The extra DDR kit checks used so far are a local script, not part of the repository.

For a single-ELF kit (K32, K64, or DDR), read program bytes, launches (each one Atlas start), image loads, and DMA bytes from `OUT/check.json`, at `wip/atlas-nov22` first and then on your branch:

```sh
python -c 'import json,sys; c=json.load(open(sys.argv[1]+"/check.json")); p=c["positive"]; print("result", p["result"], "program bytes", sum(i["bytes"] for i in c["instruction_images"]), "counters", p["host_counters"], "dma bytes", p["dma"]["bytes"])' OUT
```

For a composed `--jtag` kit, report regions completed and referenced values compared from `OUT/composed-check.json` (`positive.completed`, `positive.regions`, `positive.referenced_values`, and `result` EXACT, with the negative MISMATCH), and sum the `launches` of `composed.regions` in `OUT/kit/manifest.json`. Wall time and `elapsed_seconds` are not latency.

## Conventions

Subject-only conventional commits (`feat(loop): ...`, `fix(host): ...`, `test(deploy): ...`), each small and reviewable, with tests in the commit that needs them.
