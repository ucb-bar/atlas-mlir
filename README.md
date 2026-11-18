# Atlas MLIR

Atlas MLIR compiles supported PyTorch workloads for the Atlas NPU. The `atlas-mlir`
wheel includes `atlasc`, native `atlas-opt` / `atlas-emit` binaries and their libraries,
the Atlas Merlin provider, the model2MLIR export adapter, and runtime ABI sources.

## Install

Use Linux x86-64 and Python **3.12 or newer**. Download the wheel from the Package
workflow, then install the CPU capture stack:

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install torch==2.11.0+cpu --index-url https://download.pytorch.org/whl/cpu
python -m pip install './atlas_mlir-0.2.0-py3-none-linux_x86_64.whl[capture]'
atlasc doctor --capture
```

Merlin and model2MLIR come from exact Git revisions declared in `pyproject.toml`.
Merlin uses a source archive at its Git SHA to avoid recursively cloning its LLVM
submodule. Git is needed for model2MLIR and `setup-capture`. There is no local
LLVM requirement for an installed wheel.

Before building a new native snapshot, install Cargo and rustc. If you use
rustup, select a toolchain with `rustup default stable`. Run
`atlasc doctor --capture --build` to check the Rust tools and Merlin's locked
crate cache. On a fresh cache, run the `cargo fetch` command printed by doctor
while online. Merlin then builds the bridge offline. If you supply an existing
snapshot with `atlasc build --snapshot DIR`, the Rust build check can be skipped.

For atlas-torchao exports, also run `atlasc setup-capture`. This fetches pinned
exporter and reference sources into `~/.cache/atlas-mlir`. Its exporter modules
are not correctly exposed by its upstream pip metadata, which also pins a
conflicting xDSL version. The capture worker loads the checked source modules;
it does not install that project's unrelated training dependencies. Set
`ATLAS_TORCHAO_ROOT` to use an existing checkout at the same pin. Install additional
model dependencies such as `transformers==5.5.4` if your loader uses them.

Use a fresh environment: `atlas-mlir` owns both plugin registrations. Do not install
`atlas-mlir-native-support` or the separate `atlas-m2m-adapter` beside it.

## Compile a PyTorch reference through Merlin

The default flow is:

```text
live PyTorch loader / atlas-torchao raw export
  → Merlin capture worker → model2MLIR fx_importer (+ Atlas adapter)
  → Merlin stage-capture → native-compile --application → native-verify
```

`fx_importer` is model2MLIR's xDSL importer. This path does not use torch-mlir.
Merlin owns application partitioning, cross-region loops, residency, and host C
composition. Atlas compiles its selected regions and supplies the runtime ABI.

A floating loader defines `get_model_and_inputs()` and returns an evaluated model
and an input tuple. Compile it with an **explicit** application precision policy:

```sh
atlasc build model.py --dtype fp32 \
  --precision-policy precision.py --application-memory-bytes 536870912 \
  -o atlas-out/model
```

`precision.py` defines `get_precision_policy(source_sha256)` and returns the
existing Atlas/Merlin policy dictionary for that source, with the supplied hash.
Use the policy names and domains in Merlin's target `software-spec.yaml` and its
model examples, or the [Atlas binding guide](docs/native-merlin-binding.md).
The CLI does not invent a numerical preset. The arena budget is explicit and
must fit the selected application and deployment.

A source-bound JSON policy is also accepted. For policies that are selected after
inspecting the captured IR, use two steps:

```sh
atlasc capture model.py --dtype fp32 -o atlas-out/capture
# Select precision.json for the exact atlas-out/capture/model.mlir SHA-256.
atlasc build --capture atlas-out/capture --precision-policy precision.json \
  --application-memory-bytes 536870912 -o atlas-out/model
```

### Already-quantized atlas-torchao exports

Follow Merlin's live adapter loader convention. The loader reconstructs the model
with the export's **raw codes, exponents and other tensors**. It exposes
`normalize_capture` and returns the result of `prepare_model`; this is adapter
capture, not weight quantization:

```python
from atlas_mlir_native.capture import normalize_capture, prepare_model


def get_model_and_inputs():
    model, inputs = load_my_raw_export()  # Your architecture and authenticated raw state.
    return prepare_model(
        model, inputs, "export/manifest.json", "export/policy.json",
        expected_contract_sha256=CONTRACT_SHA256,
        expected_policy_sha256=POLICY_SHA256,
    )
```

The full example loader in pinned Merlin is
`examples/atlas/whole-model/unchanged_fx/adapter_loader.py`. It also illustrates
wrapper prefixes and source-bound model boundaries.

```sh
atlasc build model.py --dtype bf16 \
  --contract export/manifest.json --quant-policy export/policy.json \
  --expected-contract-sha256 "$CONTRACT_SHA256" \
  --expected-policy-sha256 "$POLICY_SHA256" \
  --precision-policy precision.py --application-memory-bytes 536870912 \
  -o atlas-out/model
```

Use hashes from the selected export receipt. Adapter validation refuses altered
bytes, unsupported folded exports, and mismatched state. Existing quantized
captures retain their `already_materialized` metadata through `--capture`;
`--already-quantized` is also available explicitly. Saved `.pt2` files are not
qualified intake for this flow.

## Outputs and scope

`application/` contains Merlin's manifest, program set, external parameter blob,
and combined host C. `verification.json` records compiler replay;
`report.json` records dependency identities, wheel/tool/library hashes, source
policy hash and relative artifact paths. `logs/` holds each stage's diagnostics.
`--force` only replaces an earlier `atlasc` output. A timeout aborts compilation.

Compilation and replay are **diagnostic**, not device execution or full-model
accuracy qualification. Link and run the application with Merlin's deployment
flow for your board. Numerical checks use a separate Python 3.14 environment,
CPU PyTorch `2.11.0+cpu`, and npu_model `b97b2f5a`.

`atlasc demo` runs the bundled historical standalone MLP tile example without
PyTorch. `atlasc tile-demo` exposes its direct Linalg importer and optional
diagnostic host ELF. Those demo paths are separate from `atlasc build`.

## Development and documentation

See [setup](docs/setup.md) for source builds and tests and
[package implementation](docs/pip-package-plan.md) for dependency pins and
qualification boundaries. Conda MLIR 23.1.2 is the wheel build toolchain;
equivalence with the backend kit's LLVM revision `a47bddcc` is not established.

| Guide | Covers |
| --- | --- |
| [Compiler pipeline](docs/compiler-pipeline.md) | Virtual SSA, machine IR, scheduling and emission. |
| [Dialect reference](docs/dialect-reference.md) | Operations, verifiers and passes. |
| [Merlin binding](docs/native-merlin-binding.md) | Supported graphs and precision policies. |
| [Host launch](docs/host-launch.md) | Runtime, host ELF and device launch. |
| [Examples](examples/handoff/README.md) | End-to-end compiler flows. |
| [Hardware status](docs/hardware-status.md) | RTL revision, validation and known limits. |
