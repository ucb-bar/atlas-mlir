# Bounded model tile runner

`tools/run_model_tile.py` accepts a source-bound contraction, independently checks conversion/packing and MXU0 arithmetic, assembles a conservative fixed schedule with the current model assembler, and executes its decoded instruction objects with explicit `rtl` numerics. It writes raw instruction words as reviewable artifacts; the model does not decode or execute `program.bin`. Results are model evidence only.

The named recipe `atlas_mxu0_fp8_unit_bf16acc_v1` converts source operands to BF16 nearest-even, then applies the VPU FP8Pack unit-scale rule (E8M0 code 127). It flushes underflow, uses the VPU saturation behavior, and differs from MXU FP8-pop conversion and generic Torch FP8 casts. MXU0 rounds every MAC to BF16, resets the first K32 product and continues later products. This is an explicit target derivative; native outputs remain an unchanged diagnostic reference. No model-quality equivalence is asserted.

Use the supplied Python 3.14 model environment with `PYTHONDONTWRITEBYTECODE=1`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, and `OPENBLAS_NUM_THREADS=1`. Supply an isolated model root at `9e6313bddecb8ba936206eb347f5cd4e5335282c`. The runner rejects another revision, tracked model-source changes, and imports outside that root. All path components must be physical paths without symlinks. Select a fresh output destination under the session's configured artifact root.

```sh
python tools/run_model_tile.py --model-root /physical/pinned/npu-model \
  --input /physical/capture/operands.npz --manifest /physical/capture/manifest.json \
  --out /physical/new-artifact-directory
```

For initial loading/layout bringup, replace `--input` and `--manifest` with `--smoke`. This constructs a nonsymmetric K64 fixture, explicitly labeled synthetic. Default runs use two independent reset-initialized model instances with distinct deterministic storage seeds. Completion requires ECALL, no outstanding engines or DMA, exact independent output bytes, unchanged inputs/guards, and observed DMA bytes equal to requested transfers. The runner observes these conditions before any stop/flush helper; it never invokes that helper. Smoke also executes omitted-transpose, final-K-reset and wrong-oracle negative controls; only the expected output-mismatch exception after successful completion/guard/traffic checks satisfies these controls. `--negative-controls` requests them for another K>=64 fixture.

The NPZ must contain finite float32 arrays `A[32,K]`, `B[K,32]`, and `native_C[32,32]`; K is a positive multiple of 32 up to 4096. BF16 sources use exact float32 views of their values. Optional uint16 arrays `A_bf16`, `B_bf16`, `native_C_bf16` authenticate the corresponding exact views. No object arrays, unsupported dimensions or additional keys are accepted. The capture remains responsible for the native operation boundary and complete reduction range.

The JSON manifest requires `schema_version: 1`, `npz_sha256`, `source_dtype` (`float32` or `bfloat16`), and nonempty `source` and `site` objects containing capture identities and semantic/range metadata. It is retained verbatim. Workload names never choose production behavior.

`receipt.json` binds input, runner, oracle, imported model sources, memory curve and output hashes. It reports model cycles from the first fetch edge through ECALL, issue/stall/engine-busy observations, actual DMA A-channel bytes, guards, and native-error diagnostics. `cycles-*.jsonl` retains issue/halt and DMA handshake events; `perfetto-*.json` contains the model trace. Requested transfer bytes and granted DMA VMEM traffic are separate from engine traffic; the runner does not claim measured MREG traffic, host execution, transfer packing cost or device timing. Raw source, encoded operands, initial DRAM, independent expected output, actual output and instruction image are retained. Failure leaves its receipt and available artifacts in place and returns nonzero.

Portable oracle/admission controls run without importing the model:

```sh
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s test -p test_tile_arithmetic.py -v
```

The exact-rational oracle is independent of the execution model. Its finite domain rejects nonfinite source and BF16 overflow; unit-scale packing avoids a new calibration or scale-compensation policy. Bounded decoded-value/product/transition caches preserve exact integer keys and rational arithmetic; cached and uncached recurrences have separate directed and seeded controls. Source NPZ/manifest, imported model files, selected pin/diff, memory curve, runner and oracle hashes must agree before/after qualification. Broader arithmetic, overlapping schedules, integrated RTL, runtime restart semantics and source-quality admission require separate evidence.

`tools/run_model_contraction.py` accepts the same bundle with M and N positive multiples of 32, K a multiple of 32 up to 4096, and at most 1,048,576 output cells. It derives all tile ranges from array shapes, assembles one fixed image once, and executes serialized fresh model instances inside one Python process. Each invocation uses a 256-byte-aligned DRAM working set with guards; output tiles are reconstructed from actual model bytes and compared with the independent full contraction. Its `--smoke` tests a nonsymmetric 64x64x64 contraction and executed negative controls. The reported total is the sum of cold model invocations; host packing/staging, instruction loading and invocation orchestration costs are excluded. It is not a resident tiled kernel or an integrated end-to-end timing result. Every tile retains its guarded payload, actual/expected output, trace and receipt.
