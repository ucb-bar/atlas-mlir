# Retaining hardware IR for EE290SimConfig

`EE290SimConfig` means the configuration defined by [bringup-chipyard's AtlasConfigs.scala](https://github.com/ucb-ee194-tapeout/bringup-chipyard/blob/main/generators/chipyard/src/main/scala/config/AtlasConfigs.scala#L12-L25). This is the public target reference, not a substitute for an exact source revision and elaboration identity. The [contract example](examples/contract.json) is an unbound template and does not claim that a build of this reference has been executed or qualified.

## Explicit inputs

The [retention helper](../../tools/retain-ee290-hw-ir.py) takes an existing FIRRTL file, its final firtool annotation sidecar and the selected lowering-options file. It does not elaborate Chisel, invoke Make or execute a simulator. No repository location, local configuration alias or tool-installation path is embedded in the helper. `firtool` and `circt-opt` resolve through PATH unless explicit executable paths are supplied.

Run from the atlas-mlir checkout after selecting and auditing the inputs. The paths below illustrate the public configuration's usual Chipyard output names; replace the input directory with the actual location. The output directory must be new.

```bash
ee290_inputs=/path/to/chipyard/sims/vcs/generated-src/chipyard.harness.TestHarness.EE290SimConfig
ee290_name=chipyard.harness.TestHarness.EE290SimConfig
ee290_sha=$(sha256sum "$ee290_inputs/$ee290_name.fir" | cut -d ' ' -f 1)
python3 tools/retain-ee290-hw-ir.py \
  --firrtl "$ee290_inputs/$ee290_name.fir" \
  --annotations "$ee290_inputs/$ee290_name.appended.anno.json" \
  --chisel-annotations "$ee290_inputs/$ee290_name.anno.json" \
  --lowering-options "$ee290_inputs/.mfc_lowering_options" \
  --source-config EE290SimConfig \
  --expected-firrtl-sha256 "$ee290_sha" \
  --output build/rtl-timing/ee290-run-001
```

`--chisel-annotations` is optional provenance; `--annotations` is the final sidecar passed to firtool. `--source-config` records an optional actual producer label in the generated manifest; it neither selects nor verifies a configuration. Use `--firtool /path/to/firtool --circt-opt /path/to/circt-opt` when the intended tools are outside PATH. An input hash selects bytes and detects a mismatch; it does not establish their source provenance. Pin and record source revisions separately.

## Recorded derivation

The helper snapshots selected inputs under neutral filenames, checks their hashes, and derives prepared annotations and lowering options from the verified snapshots. It preserves inline design annotations, redirects the two supported hierarchy-output filenames into the artifact directory, and rejects unreviewed external blackbox/path annotations. Embedded annotation output paths require an explicit preparation rule. These restrictions define the supported input envelope; an unsupported annotation is not silently dropped.

The lowering follows the Chipyard VCS pipeline's HW-stage policy, including `--repl-seq-mem`, the supplied lowering options and annotation controls, but emits `--ir-hw` instead of `--split-verilog`. Printed debug locations retain source anchors. Audit additional build flags before claiming agreement with another pipeline; explicit input paths alone do not prove equivalent lowering options.

Each artifact directory contains the retained `ee290.hw.mlir`, exact executed helper snapshot, input snapshots, prepared annotations, command logs and `manifest.json`. The manifest records actual input/tool identities, invocation arguments and working directory, annotation-path rewrites, exit codes and output hash. Successful status requires firtool's per-pass verification, a separate `circt-opt <ir> --verify-each -o /dev/null`, expected sequential operations and Atlas module definitions, and unchanged original input hashes. Failure is not timing qualification.

Keep generated artifacts, private launchers/input mappings and bound local contract profiles under ignored `build/rtl-timing/`. The checked-in specification and template remain independent of those local paths. A local launcher can supply its selected inputs to this same public helper while retaining the canonical target reference; it must preserve the actual source metadata rather than asserting that a different producer is byte-identical to the reference.

Debug locations and output attributes can contain absolute paths, so identical designs lowered in different directories can have different MLIR bytes. Record the actual hash of each artifact. The manifest records its derivation; it does not promise directory-independent byte reproducibility.

## Evidence to inspect next

Inspect the AtlasCore transitive hierarchy, distinguishing module definitions from their unit-specific instances. Registers should retain clock/reset/next-value structure. Memories may retain `seq.firmem` operations or external/generated declarations with OM geometry and latency metadata under the selected replacement policy. External/inline blackbox behavior, arbitration and whole-instruction timing require separate analysis; a declared one-cycle memory read latency is not a one-cycle instruction latency.

The [operation coverage table](operation-coverage.md) records the compiler mappings and unresolved rules. Useful first facts include logical register geometry versus physical bank/port sharing, VMEM bank layout, MXU-local storage ownership, scalar issue and engine acceptance predicates, DMA operand capture/configuration, and address-unit transforms. Bind each fact to exact source/IR locators and applicability before enabling a resolver. Register depth, queue capacity and same-operation spacing are not interchangeable with complete instruction timing or admission.

For example, the current [`vectorAddress`](../../lib/AtlasTiming.cpp#L349) convention is `(((base_word + sext12(offset) * 32) >> 3) & 0xffff) * 32`: a word-addressed base, immediate in 32-word units and a 32-byte line projection. A selected-RTL audit must establish that transform, valid-bank admission and truncation/alignment behavior; timed arbitration and row visibility remain separate obligations. This document records the compiler convention without attaching an unpublished build's evidence to the public example.

The finish line still includes an evidence-consuming compiler provider, rejection of incompatible inputs and an emitted program on the selected integrated simulator with output and guard checks, as described by [issue #11](https://github.com/ucb-bar/atlas-mlir/issues/11). IR retention/verification supplies the extraction input; it does not complete those requirements.
