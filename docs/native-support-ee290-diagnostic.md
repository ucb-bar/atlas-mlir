# Integrated EE290 diagnostic for Merlin-native Atlas programs

Four already compiled Merlin-native Atlas programs ran inside an integrated
`EE290SimConfig` Verilator simulator on two public input panels each: **8/8
passed**. A RISC-V evaluator ELF loaded the instruction words through the
Atlas MMIO interface, started the tile, waited for completion, and checked the
outputs, preserved inputs, and a 32-byte guard. This is an evaluator-only host
program: its C source embeds public inputs and expected outputs. The compiler
never received those runtime values or expected outputs.

The result uses a **local diagnostic setup variant**, not the unmodified
`bringup-chipyard` revision or an admitted target configuration. The original
`EE290SimConfig` elaboration failed before FIRRTL because two trace register
windows were only 256 bytes while the selected Saturn frontend required page
granularity. The variant changes those windows to 4 KiB and connects the
previously undriven Shuttle trace context. It leaves the Atlas RTL at the
selected revision. The three source patches are stored in
[`patches/ee290/`](patches/ee290/) for review; they have not been admitted or
published to their upstream repositories.

| Source or tool | Identity for this run |
| --- | --- |
| Original `bringup-chipyard` | `426a862f97f98938660772bd8a5d8f41316d15c3` |
| Local setup variant | `8ad7db6c24e184b1b19dac0b6ed2a7dffbfec8a1` |
| Atlas RTL | `0079c0541111197741a231c002e3843fa6f545b2` |
| Rocket trace window patch | `5a0f71a534c26d8344a116a53237b00f25b4fe1a` |
| Tacit trace window patch | `72bd590cc47cc1fb8654255b9cbad6dc5324d9e1` |
| Shuttle trace context patch | `7eabb45ca8f02a843a2a6bdfb32e8c18a97a5b56` |
| Merlin native compiler | `1b7517c022727499f3beb9dba64745e445c710b4` |
| EE290 simulator SHA-256 | `f68831a388e4abb16ba60420ddcf0f3c14509e7d35513e60a9df0e5dda52b7e1` |
| Atlas host emitter SHA-256 | `fc3db08366d0fcba0c4ca4bd8eeef351a512145acc95ef92b6b299bbdf300bf6` |

The local setup commit records three submodule commits. The patch files here
are their reviewable diffs against the original submodule pins. The selected
Atlas RTL checkout was present separately at the revision above. No change to
Atlas RTL was needed for these tests.

## Results and scope

| Native program | Program words | Public panels | Checked 32-bit words per panel | Retired Atlas instructions | Host-reported core cycles |
| --- | ---: | ---: | ---: | ---: | ---: |
| BF16 LOG2 | 39 | 2/2 | 1,032 | 38 | 60,387 |
| BF16 SQRT | 39 | 2/2 | 1,032 | 38 | 60,387 |
| BF16 EXP2 | 39 | 2/2 | 1,032 | 38 | 60,387 |
| Two MXU0 contractions followed by VPU MIN | 67 | 2/2 | 1,544 | 66 | 73,239 |

The two panels use phases 0 and 5 of the public test fixture. Each case uses
the same native program bytes for both input sets. Checks include the complete
32×32 BF16 output, every source input byte, and the 32-byte guard. The unary
expected values use the public hand OOT references for SQRT and EXP2 and exact
BF16 powers of two for LOG2. The composed program uses bounded E4M3 ±1/+2
operands and exact BF16 MIN results. The cycle counts include the host
diagnostic work, so they are not Atlas kernel latency measurements.

The original source failure, FIRRTL and Verilator build logs, simulator
receipts, and an eight-case summary are retained under the Merlin invocation
root `out/artifacts/targets/atlas/`. In particular:

- `ee290-sim-elaboration-r1/elaboration-address-diagnostic.log`
- `ee290-sim-elaboration-r1/elaboration-page-window-espresso.log`
- `ee290-sim-elaboration-r1/firtool-trace-ctx-candidate.log`
- `ee290-sim-elaboration-r1/verilator-with-dramsim-candidate.log`
- `ee290-native-{log2,sqrt,exp2,minmax}-phase{0,5}-r*/receipt.json`
- `ee290-native-bounded-summary-r1.json`

The stronger identity rerun is
`ee290-native-minmax-phase5-identity-r1/receipt.json`. It pins the local setup
commit, three submodule commits, checker bytes, host emitter, native program,
and simulator bytes. A deliberately wrong simulator hash was rejected before
creating an output directory. The first simulator attempt crashed before
loading the ELF because the generated Verilator initialization exceeded the
default 8 MiB process stack. The evaluator now raises the child simulator's
stack soft limit to its hard limit; the same ELF then passed. The failed
attempt remains in the diagnostic artifact root.

## Reproduction

Use the recorded source revisions and a RISC-V bare-metal compiler with
`htif_nano.specs` and `htif.ld`. The local build used Java 17, a pinned
Espresso binary on `PATH`, `firtool` from CIRCT `267e183e0`, Verilator 5.022,
and the source tree's pinned `fixedpoint` and `DRAMSim2` submodules. With
those provisioned, the simulator build command was:

```sh
cd "$EE290_SOURCE/sims/verilator"
env JAVA_HOME="$JAVA17_HOME" RISCV="$RISCV_TOOLCHAIN" \
  PATH="$JAVA17_HOME/bin:$ESPRESSO_BIN_DIR:$RISCV_TOOLCHAIN/bin:$VERILATOR_BIN_DIR:$PATH" \
  make -j8 CONFIG=EE290SimConfig JAVA_HEAP_SIZE=16G \
  FIRTOOL_BIN="$FIRTOOL_BIN"
```

From this OOT checkout, one panel can be rerun with the existing strict-native
compilation artifact. The command refuses an existing output directory or a
mismatched source, host emitter, or simulator identity:

```sh
python test/qualify_native_ee290.py \
  --case minmax --phase 5 \
  --program "$MERLIN_ARTIFACT_ROOT/phase1-95-minmax-compile-r1" \
  --source-revision 0079c0541111197741a231c002e3843fa6f545b2 \
  --ee290-source "$EE290_SOURCE" \
  --expected-ee290-revision 8ad7db6c24e184b1b19dac0b6ed2a7dffbfec8a1 \
  --host-emitter "$EE290_SOURCE/generators/atlas-npu/baremetal/assembler.py" \
  --expected-host-emitter-sha256 fc3db08366d0fcba0c4ca4bd8eeef351a512145acc95ef92b6b299bbdf300bf6 \
  --cc "$RISCV_TOOLCHAIN/bin/riscv64-unknown-elf-gcc" \
  --simulator "$EE290_SOURCE/sims/verilator/simulator-chipyard.harness-EE290SimConfig" \
  --expected-simulator-sha256 f68831a388e4abb16ba60420ddcf0f3c14509e7d35513e60a9df0e5dda52b7e1 \
  --dramsim-ini-dir "$EE290_SOURCE/generators/testchipip/src/main/resources/dramsim2_ini" \
  --out "$MERLIN_ARTIFACT_ROOT/ee290-native-minmax-phase5-recheck"
```

Other programs use `--case log2`, `sqrt`, or `exp2`, their matching
`phase1-95-*-compile-r1` artifact, and `--phase 0` or `5`.

## Limits

This run exercises actual Merlin-native emitted Atlas words inside a local
integrated EE290 simulator and a compiled RISC-V evaluator host. It does not
establish a deployable Atlas host driver, Zephyr integration, callable kernel
ABI, general DMA scheduling, numerical semantics beyond these panels, tail or
multi-tile behavior, complete models, or the D/F/M/N Phase 1 gates. The
unmodified source revision still fails elaboration. Admitting the diagnostic
setup variant requires a separately reviewed target selection and a new
frozen evaluation campaign. The native ACT comparison was not run here.
