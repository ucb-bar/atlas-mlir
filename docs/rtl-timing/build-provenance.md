# Checking EE290 build provenance

The [provenance checker](../../tools/check-ee290-provenance.py) connects a selected retention manifest to successful integrated execution receipts and an optional source inventory. It detects changed artifacts and inconsistent links. It does not reconstruct a historical build from current files or run a new build.

This is the next step after the passing [conditional VLS witnesses](selected-evidence.md). The public target remains `EE290SimConfig`; actual producer labels, machine paths, generated reports and local source changes stay in ignored artifacts. A configuration name is not evidence of source equivalence.

## What a successful check establishes

| Relationship | Checked evidence | Remaining limit |
| --- | --- | --- |
| Selected FIRRTL → retained HW/Comb/Seq | Input/snapshot equality, tools, successful recorded lowering and verification, and selected output identity | The checker validates the saved execution records; it does not rerun CIRCT. |
| Final program → emitted words | Executed program snapshot, compiler identity, successful emission log and generated instruction include | The caller must retain the separate final timing/applicability verification. |
| Selected simulator → numerical witness | Simulator and host binary identities, successful commands, three numerical panels, checked guards and matching observations | This finite result does not measure internal issue/access/release timing. |
| Current source/configuration → selected FIRRTL | Optional explicit inventory and recorded revision/local-change metadata | Unestablished without a captured and validated elaboration relationship. |
| Generated simulation sources → executed simulator | Source/file-list and archive identities, plus saved command metadata when available | Unestablished without a captured and validated build relationship, including relevant preprocessing and headers. |

When checking multiple witness arms, the manifest, hardware, elaboration inputs, simulator, source/archive sets, compiler and host tools must agree. The programs may differ. Artifact checks include the referenced files, not just equality between hashes printed in two reports. Malformed identities, missing artifacts, inconsistent command inputs, failed executions, differing shared inputs or mismatched expected digests reject the check.

The resulting `atlas.ee290_provenance_check.v0` receipt has `state = "recorded_links_verified"`, `build_linkage_complete = false` and `scheduling_qualified = false`. A zero exit status means the selected records passed these checks; it does not mean the missing build links or timing assumptions are established. Unsupported future build-receipt fields are rejected rather than accepted as an assertion of completeness.

## Run the checker

Select the exact reviewed input receipts and their expected SHA-256 digests. As with compiler evidence selection, copying a digest from an arbitrary changed report does not establish a trusted selection. No simulator license is needed for this check.

```bash
python3 tools/check-ee290-provenance.py \
  --manifest /path/to/retention/manifest.json \
  --expected-manifest-sha256 MANIFEST_SHA256 \
  --witness-report /path/to/baseline/report.json BASELINE_SHA256 \
  --witness-report /path/to/scheduled/report.json SCHEDULED_SHA256 \
  --output build/rtl-timing/provenance-001
```

The output must be a new directory beneath this checkout's ignored `build/rtl-timing/`. It retains the selected JSON receipts and executed checker alongside `report.json`; referenced large artifacts remain at their checked locations. It does not alter the original receipts or automatically update compiler compatibility bindings.

Optionally supply `--source-inventory /path/to/inventory.json --expected-source-inventory-sha256 INVENTORY_SHA256`. The inventory uses schema `atlas.ee290_source_inventory.v0`, target `EE290SimConfig`, a nonempty `sources` array of `{ "path": ..., "sha256": ..., "bytes": ... }` identities, and optional descriptive `metadata`. Source entries may carry role labels and identities of captured copies. Record current repository revisions and local changes as observations, with an explicit statement of the inventory's coverage. Use descriptive keys such as `repository_directory` for directory metadata; objects containing identity keys are treated as file identities and must have all three fields.

Useful inventory members include selected Atlas sources, configuration/I/O binders, build recipes and existing elaboration/compiler logs. Keep source snapshots, patches, real paths and private producer labels under ignored output. A current repository HEAD, a current source snapshot, modification timestamps and a saved `vcs_rebuild` command do not independently identify the bytes consumed by an earlier build. Additional metadata artifact identities are checked, but their contents do not automatically acquire a verified derivation relationship.

## Closing the remaining links

A fresh FIRRTL-to-Verilog derivation with the selected tool, annotations and lowering policy can establish correspondence with recorded generated modules. Exact module matches are useful evidence, but external SRAM/blackbox implementations require separate coverage, and regenerated sources cannot prove which bytes built an old executable.

Where historical records are insufficient, capture the needed build relationship prospectively: selected inputs and relevant dependencies before execution, exact command/tool/options, successful execution and outputs, and input stability checks. A fresh VCS build can establish the simulation-source edge; it cannot by itself establish the earlier Chisel-to-FIRRTL edge. Keep unreconstructed relationships explicit and define the qualified scope accordingly. Full pinning of unrelated host runtime libraries is not a prerequisite for recording this narrower evidence.

The current checker deliberately does not accept a generic user-authored build-success receipt. A future build-capture producer and its validation rules must be implemented together before such an edge can be promoted. Internal timing capture and boundary checks remain separate work even after build provenance is established.

Run the focused synthetic regressions without rebuilding the compiler or launching VCS:

```bash
python3 -m unittest discover -s test -p test_ee290_provenance.py -v
```
