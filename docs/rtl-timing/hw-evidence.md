# Indexing retained hardware evidence

The [hardware retention step](retained-hw.md) supplies verified HW/Comb/Seq IR. The next step indexes its hierarchy, storage declarations, port connections and candidate event expressions. This is structural evidence for the [timing contract](README.md), not an enabled timing profile or an instruction latency model.

## Select an artifact

Run from the atlas-mlir checkout with the manifest produced by `retain-ee290-hw-ir.py`. The output directory must be new. The indexer uses Python's standard library; it does not build or execute hardware.

```bash
python3 tools/index-retained-hw.py \
  --manifest build/rtl-timing/ee290-run-001/manifest.json \
  --root AtlasCore \
  --output build/rtl-timing/ee290-evidence-001
```

The manifest must describe a verified retention run, and the selected hardware bytes must match its recorded identity. The generated report binds that manifest, hardware IR and executed indexer. This detects a changed extraction input; it is separate from the compiler's [selected-evidence checks](selected-evidence.md). Keep reports and their exact local producer metadata under ignored `build/rtl-timing/`. `EE290SimConfig` retains the public target meaning defined in the contract.

The output contains `hw-index.json`, a `retention-manifest.json` snapshot and `index-retained-hw.executed.py`. The report uses `atlas.retained_hw_index.v0` and records module definitions, instance paths, candidate events and their source line/operation anchors. It retains the selected hardware path and hash without copying the hardware again. Keep that source artifact available; the index alone is not a self-contained hardware bundle.

## What the index supports

The index distinguishes module definitions from instance paths and retains instance port connections. Two instances of one storage module remain separate physical stores. Registers retain next-value, clock/reset and operation anchors; memory declarations and opaque external/generated modules retain their structural boundary. Such declarations alone do not establish arbitration, usable outstanding-command capacity, read-during-write behavior or whole-instruction latency.

Candidate event cones describe backwards SSA dependencies within a module. They stop at registers, module inputs, instance outputs and unsupported operations. Instance wiring supplies the next analysis boundary; the index does not collapse an instance output into an assumed combinational function of every input. A stopped or unsupported cone is visible in the report and is not a timing result.

Candidate names and port shapes are selectors that need source review. In particular, do not assume that scalar `s1_fire` is the compiler's age-zero event for every instruction: vector-engine launch, scalar-memory pipeline requests, CSR effects and terminal control sidebands can follow different paths. A command-valid signal does not itself prove successful acceptance, safe overlap or completion. Deriving those relationships requires the corresponding engine and frontend conditions.

The text reader supports constrained firtool custom assembly: one-line module headers and instance/operation forms, named ports, two-space top-level module indentation and four-space module-body indentation. It is not a general MLIR parser or verifier. Unsupported required structural syntax and unresolved references are errors. Nested-region results remain opaque; top-level OM metadata and blackbox implementation contents are outside the index. Memory geometry declared in OM therefore still needs a separate bound analysis. The retained input's CIRCT verification and the indexer's structural checks serve different purposes.

## Using the result

Choose a candidate event and follow its connections into the relevant engine and storage. Establish capture, grant, first/last access and release conditions, including stalls and competing users, before assigning issue-relative ages. Bind each resulting rule to its source identity, operand domain and unresolved conditions. The [operation coverage table](operation-coverage.md) remains the denominator for instruction coverage; an index spanning every engine does not establish every operation's rules.

The [all-engine admission audit](admission.md) follows those boundaries into capture and software obligations. The [bounded VLOAD/VSTORE replay](vls-timing.md) derives and checks an initial conditional access/release mapping. The [selected-evidence backend](selected-evidence.md) consumes that replay separately with explicit identity selection and conditional opt-in. The index and replay alone establish neither integrated execution nor full instruction coverage; preserve unsupported rules and assumptions explicitly.

The focused indexer regressions run independently of a compiler build:

```bash
python3 test/test_rtl_hw_index.py
```
