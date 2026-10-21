# Complete-contraction diagnostic capsule

`tools/export_contraction_capsule.py` packages an existing complete-contraction
model run together with matching raw AtlasCore RTL evidence. It executes neither
simulator nor hardware and emits no ELF. Its status always declares
`diagnostic_only`, `fpga_ready=false`, `full_model=false`,
`launchable_host=false`, and `native_fidelity_claim=false`.

Run from this checkout, with explicit physical paths and externally recorded
receipt hashes. The output directory must not exist:

```sh
python tools/export_contraction_capsule.py \
  --model-receipt "$MODEL_OUT/receipt.json" --model-receipt-sha256 "$MODEL_RECEIPT_SHA" \
  --rtl-receipt "$RTL_OUT/receipt.json" --rtl-receipt-sha256 "$RTL_RECEIPT_SHA" \
  --input "$FROZEN_INPUT" --manifest "$FROZEN_MANIFEST" \
  --producer-root "$PWD" --model-root "$MODEL_ROOT" --out "$FRESH_CAPSULE"
```

The exporter admits the current release producer tuple and its retained source
snapshots, unchanged model/input source bookends, current imported model
sources, every declared model artifact, raw-run source records, runner-build
records and executable identity. It rechecks all admitted files before
publishing. It refuses symbolic-link components, parent traversal and prohibited
path components before filesystem metadata access; ZIP members are admitted
before archive content hashing or copying. Archives use bounded, explicit array
schemas and disable pickle.

The contraction must use complete 32×32 output tiles with the full ordered K
reduction, two initialized model repeats and two raw RTL repeats. The exporter
independently rebuilds encoded input bytes and their guarded DRAM image, checks
every logical/physical output view, reconstructs DMA traffic from model traces,
and replays physical host writes, DMA writes/responses and committed ECALL
status 5 from both raw RTL traces. It requires the final 32 contiguous quiet
cycles, exact outputs, preserved guards and shape-derived traffic counts. This
authenticates recorded execution; it does not rerun the independent arithmetic
oracle or prove source assembly and binary equivalence anew.

The portable payload includes `program.S`, `program.bin`, `initial-dram.bin`,
physical expected/actual output bytes, the original `frozen-input.npz` and
`native-manifest.json`, source/encoded/result arrays, the source manifest, both
receipts, `status.json`, and `SHA256SUMS`. The status retains source file
identities for auditing against the original evidence closure. Traces, build
products and source trees are not duplicated into the capsule; re-export still
needs those original files. Verify the copied payload with `sha256sum --check
SHA256SUMS` after lexically admitting its listed paths.

Physical execution and native accuracy remain separate gates. `native_gate`
recomputes unchanged `native_C` comparison with `rtol=atol=1e-4` and retains the
producer's numerical diagnostic metrics. A physical pass with a native failure
still produces a diagnostic capsule with the failed native gate visible. The
exporter never substitutes raw pre-rescale accumulator words or adjusts
tolerance.

The program's DMA high half is 1, so its DRAM base is `0x100000000`. The
selected RTL can form that address. The raw host runtime accepts full-width
buffer descriptors, but platform integration still requires address
reachability, cache handling and completion validation. Complete graph coverage
and native fidelity, bitstream identity, SoC memory map, loader and integrated
execution remain open gates. Public RTL idle omits VPU busy; the evidence
therefore states its committed ECALL plus quiet-margin observation explicitly.

Run the synthetic admission tests without a model or simulator:

```sh
python -m unittest discover -s test -p test_contraction_capsule.py -v
```
