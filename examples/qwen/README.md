# Qwen kernel campaign

For the actual pretrained-model export, graph-derived PyTorch regions and
model-shape library benchmark, use [full_model/README.md](full_model/README.md).
The workflow below is the earlier manually authored microbenchmark campaign.

These are ordinary PyTorch subgraphs of
`deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`, the Qwen2 model in
[Radiance PR #1](https://github.com/ucb-bar/radiance-kernels/pull/1).
See the [results and recommendations](../../docs/qwen-results.md) for measured
cycles, numerical error, and the exact boundaries of each workload.
The follow-up [roofline analysis](../../docs/qwen-roofline.md) includes per-resource
lower bounds, default/faster-memory comparisons, raw reports and reproduction.

For a wheel installation, use Python 3.12 and download the Linux x86-64 wheel
from the **Package** workflow for `feat/qwen-kernels`. Keep a checkout of the
same branch for the experiment scripts; those scripts are not included in the
wheel. From that checkout:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install /path/to/atlas_mlir-0.2.0-py3-none-linux_x86_64.whl
atlasc doctor
atlasc demo
```

The wheel includes the native compiler tools, so this route needs no local
LLVM build. With `uv`, Git and Cargo on PATH, run the campaign:

```sh
python3 examples/qwen/run.py --setup
python3 examples/qwen/run.py
```

Setup downloads pinned Merlin and NPU-model sources and creates three isolated
Python environments under `out/qwen-deps-merlin`: capture, Merlin selection, and
numerical simulation. It does not modify sibling checkouts. Cargo's locked
bridge dependencies are fetched during setup; snapshot construction is offline.
The setup step installs PyTorch and torch-mlir in the capture environment, so
the wheel's optional capture extra is not needed in the launching environment.
The second command creates a fresh `atlas-out/qwen` campaign. Use `-o DIR` for
subsequent runs. Existing results are never overwritten.
Use `--panel signed -o atlas-out/qwen-signed` to repeat with bounded signed
inputs. Both panels use magnitudes in [0.03125,0.15625), within the selected
FP8 input domain; they do not cover arbitrary pretrained activations.

For already prepared environments, override `--capture-python`,
`--merlin-python`, `--model-python`, `--merlin-root`, and `--model-root`.
Dependency checkouts must be clean at the recorded pins. `ATLAS_OOT_BIN_DIR`
selects another compiler build.

The root [README](../../README.md) documents the new `atlasc build` flow through
model2MLIR and Merlin application compilation. It requires an explicit precision
policy and supports authenticated atlas-torchao raw exports. This existing
microbenchmark campaign retains its historical torch-mlir capture for reproducing
the attention/SwiGLU experiments; it is separate from the new application intake.
Its Merlin and numerical model pins now match the package. Existing Markdown
performance reports describe their recorded older revisions and are not new
measurements of this integration. Setup uses a new dependency directory so it
does not overwrite an earlier pinned campaign environment.

The campaign captures all workloads, then compiles supported graphs through
Merlin and Atlas and executes their emitted instructions on the NPU model.
Full-width source graphs that are unsupported, or whose search times out, are
recorded in `summary.json`; they do not get fabricated performance numbers.
Unexpected failures in the supported attention or SwiGLU graph stop the run.
The K=1536 projection has a 60-second search budget in the repeatable campaign;
the report also records an earlier failed 300-second experiment.

Each compiled workload retains:

- The original and normalized Linalg, argument manifest, tensor inputs and f32 reference outputs.
- Precision policy, source hash, Merlin request and ABI, verified native artifacts.
- Original unscheduled, boundary-DMA-only, and scheduled machine programs.
- Re-encoded instruction checks, model traces, bit-exact output/DRAM guard checks,
  engine busy cycles and error against PyTorch in `performance.json`.

The input values and weights are seeded synthetic tensors. The full Qwen
architecture uses hidden width 1536, FFN width 8960, and 128-channel heads. The
attention experiment uses all 128 head channels in a 32-token unmasked cache
window; the SwiGLU experiment is an interior 32-channel partition. The K=128,
256 and 512 projection experiments are reduction partitions, not complete
1536-wide projections. No token throughput, pretrained accuracy, or full-model
execution is claimed.

`kernels.py` also contains full-width RMSNorm/QKV, residual MLP, RoPE and GQA
PyTorch definitions for extending the campaign. The selected Atlas numerical
policy remains diagnostic; in particular its reciprocal domain is [1,32].
