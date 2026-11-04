# TinyLlama capture, qualification, build, and replay

The repository's D2 capture and `d1-v2-qualify.py` route is the historical 906-launch model: 840 projection panels and 66 attention panels. Its qualifier explicitly requires that inventory. The current kit uses T-on-F3-h0 with 1,038 launches, including 132 norm launches and decode q/k RoPE; the external reproduction route is described below. Running the D2 capture driver does not regenerate the current kit. The maintained `AttentionBundle` and `build_attention.py` admit both inventories, plus the 972-launch post-attention-norm inventory.

Set physical absolute paths in `ATLAS_ROOT`, `MERLIN_ROOT`, `ATLAS_NPU_MODEL_ROOT`, and `TINY_WORK_ROOT`. The model must be unmodified original `47b8599d8ffbe5ddd96059ffe2b5be2427a410f5`. `TINY_WORK_ROOT` holds generated captures and receipts. Initialize a Python environment with the model, NumPy, and capture dependencies; `PYTHON` names that interpreter, and `HOST_CC` names the host C compiler. The input parameter capture, projection scale contract, and attention scale contract are external checksum-bound inputs, not downloaded by these commands. Their schemas and fixed contracts are checked by Merlin's driver. Keep the model and data roots physical, without symlinks. Use at most 16 numerical workers.

```sh
export ATLAS_ROOT=/absolute/path/to/atlas-mlir
export MERLIN_ROOT=/absolute/path/to/merlin
export ATLAS_NPU_MODEL_ROOT=/absolute/path/to/npu-model
export ATLAS_MLIR_BIN=/absolute/path/to/atlas-mlir-build/bin
export TINY_WORK_ROOT=/absolute/path/to/tiny-work
export PYTHON=/absolute/path/to/model-venv/bin/python
export ATLAS_MLIR_BIN="$ATLAS_ROOT/build/bin"
export PYTHONPATH="$ATLAS_NPU_MODEL_ROOT:$ATLAS_ROOT/tools:$MERLIN_ROOT/src"
export HOST_CC=/absolute/path/to/cc
export SOURCE_CAPTURE=/absolute/path/to/parameter-capture.json
export PROJECTION_SCALES=/absolute/path/to/projection-scale-contract.json
export ATTENTION_SCALES=/absolute/path/to/attention-scale-contract.json
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export ATLAS_NPU_MODEL_NATIVE=0
mkdir -p "$TINY_WORK_ROOT/tmp"
export TMPDIR="$TINY_WORK_ROOT/tmp"
ATLAS_LAYOUT_SHA256=$(sha256sum "$ATLAS_ROOT/tools/atlas_host_layout.py" | cut -d ' ' -f 1)
"$PYTHON" -B "$MERLIN_ROOT/examples/atlas/comparisons/d1-capture-c-v2.py" \
  --source-capture "$SOURCE_CAPTURE" \
  --scale-contract "$PROJECTION_SCALES" \
  --attention-scale-contract "$ATTENTION_SCALES" \
  --attention-model-root "$ATLAS_NPU_MODEL_ROOT" \
  --atlas-layout "$ATLAS_ROOT/tools/atlas_host_layout.py" \
  --atlas-layout-sha256 "$ATLAS_LAYOUT_SHA256" \
  --worker-script "$MERLIN_ROOT/examples/atlas/comparisons/d1-worker.py" \
  --worker-python "$PYTHON" \
  --attention-worker-script "$ATLAS_ROOT/tools/tiny_full/attention_v2_worker.py" \
  --attention-worker-python "$PYTHON" \
  --cc "$HOST_CC" --layer-count 22 --decode-count 2 \
  --out "$TINY_WORK_ROOT/d2-capture"
CAPTURE_SHA256=$(sha256sum "$TINY_WORK_ROOT/d2-capture/receipt.json" | cut -d ' ' -f 1)
"$PYTHON" -B "$ATLAS_ROOT/third_party/atlas-kernels/lib/atlas_backend.py" run \
  --backend atlas-mlir --atlas-mlir-bin "$ATLAS_MLIR_BIN" \
  --receipt "$TINY_WORK_ROOT/export-backend.json" -- \
  "$ATLAS_ROOT/tools/tiny_full/export_capture.py" \
  --capture "$TINY_WORK_ROOT/d2-capture/receipt.json" --capture-sha256 "$CAPTURE_SHA256" \
  --model-root "$ATLAS_NPU_MODEL_ROOT" --out "$TINY_WORK_ROOT/d2-assets" \
  --dram-base 0x80000000 --dram-bytes 0x80000000
"$PYTHON" -B "$ATLAS_ROOT/tools/tiny_full/d1-v2-qualify.py" \
  --assets "$TINY_WORK_ROOT/d2-assets" \
  --out "$TINY_WORK_ROOT/d1-v2-qualification/full" --workers 16
```

`--out` for capture and export must be fresh. The 2 GiB DRAM example must match the actual target placement. The public export entry point is `export_capture.py`; `attention_export.py` is its implementation, not a standalone CLI. The C capture driver uses `d1-worker.py` for projections and `tools/tiny_full/attention_v2_worker.py` for attention, with generators imported from the initialized atlas-kernels submodule. It captures positive-zero residual surrogates for o/down; the exporter emits the final plain programs, and the qualifier executes those final programs on original47. The original-model path needs no native proof arguments. `--qualify-launches` on the exporter is an alternative direct per-launch execution gate; the separate qualifier adds its aggregate coverage checks.

The explicit `atlas_backend.py run --backend atlas-mlir` export command selects the machine IR, verifier, and emitter route using `atlas-opt` and `atlas-emit` in `ATLAS_MLIR_BIN`; the backend receipt records the encodes performed in that process. The separate qualifier launches subprocesses, whose encodes are not covered by the parent's export receipt. Require the independent exhaustive program-word comparison gate described in [bringing up a model](../../docs/bringing-up-a-model.md) for every distinct program in the selected default and fast builds before treating the compiler backend as qualified. Numerical exactness, generated source/binary equality, and backend word identity are separate gates.

Without `--reuse-records`, every final program executes on original47. Repeating the qualifier command with `--resume` audits saved rows before reuse. `--audit-only` audits existing results without filling missing executions. Qualification output must be below `$TINY_WORK_ROOT/d1-v2-qualification/`. The stream drivers and workers live beside the Merlin capture driver and retain their interfaces for optional concurrent backfill.

Aggregate qualification receipts encode path pins with @atlas/, @merlin/, @model/, and @data/ roots. build_attention.py takes --model-root and --receipt-data-root to resolve these roots, derives the atlas root from its checkout, and derives the Merlin root from --merlin (the comparisons directory). SHA-256 checks and HEAD blob authentication run after resolution. Legacy receipts can be read with repeatable --receipt-relocate OLD=NEW arguments; relocation never substitutes a digest. Preserve historical source bytes when relocating historical pins.

For the host build, use [the complete build command](../../docs/tiny-attention-build.md) with `--assets "$TINY_WORK_ROOT/d2-assets"`, the aggregate `receipt.json`, its SHA-256, `--merlin "$MERLIN_ROOT/examples/atlas/comparisons"`, `--model-root "$ATLAS_NPU_MODEL_ROOT"`, and `--receipt-data-root "$TINY_WORK_ROOT"`. Supply the external platform root and GNU RISC-V toolchain explicitly. The build output contains `host/`, `sources.cmake`, `build.json`, and `build/atlas-tinyllama.elf`. Captures may require substantial disk space; no storage reuse wrapper is required.

The qualifier admits projection reuse only for the original capture and stream schemas with exact=True and full_arena_exact=True. It also runs one additional original47 execution per unique program. Every byte in the output extent starts as the complement of the expected byte, declared spans must become the expected bytes, holes must retain their second pre-fill, and all other arena bytes must remain unchanged. Coverage reports are included in the aggregate receipt and are audited by --audit-only.

The current 1,038-launch kit route requires the preserved external artifact workspace. It is not self-contained in the public checkout. In the commands below, `V2=/tmp/reednicolas-atlas/oct31/V2`; the exact owner source is `$V2/T/atlas-f3h0`, its original model is `$V2/T/npu-model`, and the Python environment is `$V2/A/.venv/bin/python`. The unchanged baseline assets must exist at `/scratch/reednicolas/atlas-tmp/jobs/F3H0/assets-full`, with `$V2/F/f3h0/qual-full/receipt.json` and all pinned historical source bytes. `T/f3h0/prepare.py` creates a checksum-bound overlay over that asset authority, changing 88 decode q/k launches. Its 950 other launches retain exact source, binary, initial arena, output, and complete final-arena joins. `storage.json` records the inherited asset root, so copying only the overlay directory is insufficient.

The following are the recorded owner interfaces, shown for a fresh isolated copy of that preserved workspace. Their fixed roots, archived compile recipes, external platform tree, and output locations must all be present. Existing receipts and builds must be preserved; these commands are not a request to overwrite them.

```sh
V2=/tmp/reednicolas-atlas/oct31/V2
cd "$V2/T"
PY="$V2/A/.venv/bin/python"
export TMPDIR="$PWD/tmp" PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 ATLAS_NPU_MODEL_NATIVE=0
"$PY" -B f3h0/prepare.py --out "$PWD/bulk/f3h0-assets" --model-root "$PWD/npu-model"
"$PY" -B f3h0/qualify.py --assets "$PWD/bulk/f3h0-assets" --out "$PWD/f3h0/qual/full" --workers 16
"$PY" -B f3h0/package_qualification.py
"$PY" -B f3h0/build.py default
```

The qualifier's `--plan-only` checks the full inventory and writes exact reuse rows without executing the 88 changed launches; `--audit-only` requires existing complete results. The full command resumes by auditing completed rows. `package_qualification.py` archives the 1,038 worker records and logs into `qual/original-receipts.zip`, losslessly compresses the aggregate into `qual/receipt.json.gz`, and records both compressed and uncompressed hashes in `qual/GZIP.json`. `build.py default`, `build.py chip`, and `build.py fast` replay archived compilation/link recipes; their `bulk/f3h0/MODE/build.json` records source bookends, commands, and ELF SHA-256. This is the owner recipe interface, distinct from a new `build_attention.py` build.

For exact final-tip linked ELF identity, the preserved `$V2/PUSH/ORCH-RUN-tiny-resnet.sh` invokes `identity_build.py tiny-default`, `tiny-kit3-fast`, `tiny-m1-fast`, `tiny-m2-fast`, and `resnet`. It pins the source tips, uses explicit external platform roots, reuses frozen generated assets and platform archives, and requires expected SHA-256 plus full `cmp` on 10 rows before writing `PASS.json`. `PUSH/IDENTITY.md` remains the authority for which rows have actually run; its recorded final-tip status is PENDING. This identity gate does not recapture, regenerate programs, qualify arithmetic, or replay the host. Adapt a copy when changing source roots, preserving the original recipe and its evidence.

The actual default QEMU replay runner is external `$V2/T/f3h0/replay/replay_v2_complete.py`, with the 1,038-launch overlay as `--bundle-root`. A fresh replay of an available matching default ELF uses the complete recorded arguments below; `REPLAY_OUT` must name a fresh output directory. The GNU compiler used by the owner recipes is `/bwrcq/C/reednicolas/ee194-sp26-chipyard/.conda-env/riscv-tools/bin/riscv64-unknown-elf-gcc`.

```sh
QEMU=/bwrcq/C/reednicolas/ee194-sp26-chipyard/.conda-env/bin/qemu-system-riscv64
REPLAY_OUT=/absolute/path/to/fresh-tiny-replay
"$PY" -B "$V2/T/f3h0/replay/replay_v2_complete.py" \
  --elf "$V2/T/bulk/f3h0/default/atlas-tinyllama.elf" \
  --bundle-root "$V2/T/bulk/f3h0-assets" --qemu "$QEMU" \
  --out "$REPLAY_OUT" --wall-seconds 7200 \
  --console-write-symbol _write --exit-symbol _exit --icount --gdb-stdio
```

The historical default ELF was retired after hashing, so supply a rebuilt byte-identical default ELF or the matching kit provider at `$V2/E/kit-v2/elfs/atlas-tinyllama-v2-sims.elf` through `--elf`. The external fast runner is `T/f3h0/replay/run_fast.py --out /absolute/path/to/fresh-fast-replay`; it additionally requires the linked fast ELF, generated programs, and `generated/admission.json` at its fixed owner paths. The generic default runner alone does not establish the fast transport admission proof.

Evidence flows from the capture and checksum manifest, through original47 numerical qualification, to build/source pins, linked fast admission when applicable, and the replay receipt. `T/f3h0/replay/default-full.command.json` and `default-full.inputs.json` preserve the measured replay command and input hashes; `default-full/receipt.json` records `PASS_CAPTURED_REPLY_REPLAY`, 1,038 exact launches, 137 exact final buffers, `host_elf_executed=true`, and `numerical_device_execution=false`. The guest tokens are `[278, 3711, 5652]` at position 130. `T/f3h0/GREEN.json` binds qualification, replay, corruption, static regeneration, and memory gates, and `KIT5/candidate/README.md` describes the kit's required terminal checks. Replay executes host instructions with captured Atlas replies, while the separate qualifier establishes exact arithmetic. The 3,528,847,056 cycles in `F/f3h0/results.json` at `derived_Atlas_cycles.total_cycles` are derived numerical-model cycles. None of these artifacts establishes full-model FireSim execution or physical device timing.
