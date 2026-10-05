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

## Host build binding

Version 0.0.4 includes `runtime/atlas_host.{c,h}` and
`atlas_native_support.host_build.ee290_baremetal_recipe`. The latter supplies
Atlas/EE290 flags, the C driver, and the selected linker files to Merlin's
shared `HarnessBuildRecipe`. The caller selects the RISC-V compiler,
`htif.ld`, and `htif_nano.specs` by exact path. The C entrypoint
`atlas_ee290_run` loads and reads back IMEM, starts the selected tile, polls
its diagnostic completion marker with a bound, stops it, and reports raw
counters/status. It accepts an already encoded program; it does no instruction
selection, tensor computation, expected-output comparison, or ACT invocation.

The [integrated EE290 diagnostic](../../docs/native-support-ee290-diagnostic.md)
builds a host ELF through that shared recipe. The input-only variant places
public runtime inputs in the host harness and sends three short readback
digests to the evaluator, which holds the expected output separately. This
is a bounded bare-metal interface test, not a qualified Zephyr adapter,
general callable ABI, or complete model runtime.

For a bounded standalone-core check, the evaluator-only
[`test/qualify_native_support.py`](../../test/qualify_native_support.py) takes
four already compiled artifact directories (`exp2`, `sqrt`, `log2`, `minmax`),
an explicit ARC shared library and state manifest, their expected SHA-256
identities, the selected RTL revision, and a ModeLIR checkout.
It verifies each program and plan hash, executes two public input panels per
case, and checks every output byte plus input and guard preservation. The
script refuses a preexisting receipt path and returns nonzero if any panel
fails. Its numerical reference code stays outside the compiler package; a
passing result is limited to the supplied standalone core and panels.
The [bounded selected-core observation](../../docs/native-support-selected-core-observation.md)
records the current eight-panel result and its source and model hashes.
