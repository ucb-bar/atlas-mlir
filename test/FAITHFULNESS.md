# Required source joins

`cmake --build build --target atlas-faithfulness` runs `test_model_coverage`, `test_smol_full_plan` including its real-reference case, the actual-capture case in `test_tiny_full_segments.InventoryTest`, and atlas-kernels' `test_build_table` and `test_bench_model.PlanTest`. It fails on missing required files, dependencies, or any skipped check. The ordinary CTest suite retains its optional input-dependent cases.

Initialize the pinned `third_party/atlas-kernels` submodule before configuring CMake, and select a Python interpreter with NumPy and PyYAML using `-DPython3_EXECUTABLE=/absolute/path/to/python`. The export fixtures are the pinned `coverage/fixture/{tinyllama,smolvla}.json`; this target does not rerun `torch.export` or capture models.

Provide matching retained capture inputs as physical paths without symbolic links:

```sh
export SMOL_REPLAY=/absolute/path/to/full-policy-reference
export SMOL_KERNEL_ASSETS=/absolute/path/to/matching-kernel-assets
export TINY_FULL_CAPTURE=/absolute/path/to/tiny-full-capture/receipt.json
cmake --build build --target atlas-faithfulness
```

`SMOL_REPLAY/receipt.json` must be a complete `PASS_ATLAS_ARITHMETIC_FULL_POLICY_REPLAY` reference with 1,421 ordered calls, fresh vision, and matching input/state/RNG bookends. `SMOL_KERNEL_ASSETS/kernels.json` must be its matching finite, source-bound `atlas.full_kernel_assets.v1` manifest with 2,715 panels and the explicit fusion profile. `TINY_FULL_CAPTURE` must contain the complete 465-call Tiny capture, its panels, and target-trace KV/logit metadata admitted by `tools/tiny_full/capture_boundaries.py`. Obtain these authenticated inputs from the retained capture providers described in the [bringup guide](../docs/bringing-up-a-model.md) and [Tiny reproduction notes](../tools/tiny_full/REPRODUCING.md); the target does not prepare or qualify a new kit.

The checks compare ordered calls, shapes, and dtypes with pinned export fixtures, join plans to retained references, and check coverage regeneration and launch planning. They do not derive host operations, fusion boundaries beyond the plan, or scale contracts, and they do not execute device programs or establish full-model equivalence. Numerical execution, encoding, census, and ELF rebuild checks remain separate gates.
