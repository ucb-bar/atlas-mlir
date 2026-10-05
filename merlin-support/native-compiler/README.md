# Atlas support for Merlin's native selector

This installable package provides Atlas descriptors, selected source checks,
physical materialization, and a standalone-core launch plan through Merlin's
`merlin.native_target_compilers` entry points. Merlin owns rule generation,
e-graph search, buffer-aware extraction, allocation, and selection checking.
The package does not import or run ACT. It also does not call the handwritten
Atlas MLIR tools, so their comparison path remains separate.

This is a bounded diagnostic compiler. The `atlas_tensor` entry point exposes
95 selected instruction descriptors and accepts selected 32×32 FP8/BF16 matrix,
movement, XLU, and VPU tile compositions, including bounded log2, sqrt, and
exp2 paths, plus the explicitly checked tiling forms in the package. Its program
uses conservative delays and the selected standalone AtlasCore conventions.
It does not qualify integrated EE290 timing, a Zephyr driver, complete model
invocations, or all Atlas instructions. An emitted `program.bin` is an Atlas
instruction stream, not a C-callable host ELF.

`src/atlas_native_support/source_import.json` records hashes of the retained
diagnostic wheel used to recover this source. The import changes the package
version and the exact
Merlin revision pin. The checked RTL revision, arithmetic contracts, source
hashes, and solver/Cargo pins remain in the package data. Review a new pin
against actual compiler behavior before changing it.

Install this package into the same environment as the selected Merlin commit:

```sh
uv pip install --python "$MERLIN_PYTHON" --no-deps --no-build-isolation \
  merlin-support/native-compiler
```

Build a fresh target snapshot, then compile a typed request with explicit
strict-native mode. `$MERLIN_ROOT` and `$ATLAS_RTL_ROOT` must refer to the
selected source checkouts; output directories must be fresh. The `--abi` file
provides fixed external addresses, not runtime tensor contents or goldens.

```sh
merlin-targetgen native-build --engine merlin_native --support atlas_tensor \
  --crate "$MERLIN_ROOT/src/merlin/semantic_compiler/egg_bridge" \
  --cargo-target-dir "$ARTIFACT_ROOT/cargo" \
  --source-revision 1b7517c022727499f3beb9dba64745e445c710b4 \
  --out "$ARTIFACT_ROOT/snapshot"

merlin-targetgen native-compile --engine merlin_native \
  --support atlas_tensor --snapshot "$ARTIFACT_ROOT/snapshot" \
  --request "$REQUEST_JSON" --abi "$ABI_JSON" \
  --target-source "$ATLAS_RTL_ROOT" --mode strict-native \
  --out "$ARTIFACT_ROOT/program" \
  --status-file "$ARTIFACT_ROOT/status.json"
```

The manifest binds the request, selected profile, binary, execution plan, and
source revision. The native compiler refuses a mismatched dependency pin or
RTL source. Execute the program only through a separately selected and
qualified Atlas platform; successful compilation alone is not execution.
