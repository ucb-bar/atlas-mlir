# From Atlas RTL to compiler timing

This walkthrough follows one VLOAD from Chisel to the compiler's timing rule. Nothing in the rule is typed by hand.

## 1. Where in the pipeline

The path is **Chisel → FIRRTL → CIRCT HW/Comb/Seq**, stopped before Verilog emission. [`tools/retain-ee290-hw-ir.py`](../../tools/retain-ee290-hw-ir.py) `--mode debug` runs `firtool --ir-hw -O=debug --preserve-values=named --mlir-print-debuginfo` on the `EE290SimConfig` FIRRTL. `hw` holds modules, instances and wires, `comb` combinational logic, `seq` registers. Debug mode keeps Chisel's value names and a source location per operation.

## 2. Which modules

| Module (instance in `AtlasCore`) | Role for VLOAD |
| --- | --- |
| `ScalarCore` (`scalar`) | Fires the instruction, raising `io_lsuCmd_valid`; its `decoder` maps the word to the LSU op code. |
| `AtlasCore` | Wires `scalar.io_lsuCmd` to `lsu.io_cmd` with no register between. |
| `LSU` (`lsu`) | Captures the command, counts rows, registers responses and writes, drives busy. |
| `Vmem`, `MregFile` (`vmem`, `mreg`) | The memories, whose read latency is measured from the IR. |

The [VLSU spec](../../tools/rtl_extract/specs/vlsu.yaml) names them: module `LSU`, command port `io_cmd`, event bundles `io_vmemVecRead` and `io_mregWriteReq`, busy `io_vecBusy`, decoder instance `scalar/decoder`.

## 3. What the IR looks like

In `ScalarCore`, a fired instruction decoded as an LSU operation becomes the command valid (symbols and attributes omitted):

```mlir
%202 = comb.and bin %s1_fire, %decoder.io_decoded_is_lsu : i1 loc(#loc13834)
%is_lsu_launch = hw.wire %202 : i1 loc(#loc13982)    // ScalarCore.scala:495
%io_lsuCmd_valid = hw.wire %is_lsu_launch : i1
```

In `LSU`, the VLOAD response path is two registers:

```mlir
%vloadRespPending = seq.firreg %vloadIssueRead clock %clock reset sync %reset, %false : i1 loc(#loc14196)
%128 = comb.and bin %io_vmemVecReadData_valid_11, %vloadRespPending : i1 loc(#loc14262)
%vloadWritePending = seq.firreg %128 clock %clock reset sync %reset, %false : i1 loc(#loc14198)
%io_mregWriteReq_valid = hw.wire %vloadWritePending : i1
```

Each `seq.firreg` is one clock edge; `comb.and` and `hw.wire` add none. The row read is requested one cycle after issue, the memory answers a cycle later, and the MREG write is valid the cycle after. Every location resolves to a Chisel line, for example `#loc14262 = loc(".../atlas/lsu/LSU.scala":273:33)`:

```scala
vloadRespPending := vloadIssueRead                              // LSU.scala:268
when(io.vmemVecReadData.valid && vloadRespPending) {            // LSU.scala:273
  vloadWritePending := true.B
  ...
}.otherwise { vloadWritePending := false.B }
io.mregWriteReq.valid := vloadWritePending                      // LSU.scala:281
```

## 4. From IR to timing

- **Export.** [`hw_ir_export`](../../tools/rtl_extract/export/hw_ir_export.cpp), built against CIRCT, writes the spec's modules as JSON (operations, operands, results, instance connections, assertions), structure only.
- **Derive.** [`derive.py`](../../tools/rtl_extract/derive.py) evaluates the decoder on the VLOAD word for the op code and simulates the memory module to measure its read latency.
- **Simulate.** [`control.py`](../../tools/rtl_extract/control.py) executes the control cone exactly, cycle by cycle: the logic behind the spec's valid and busy outputs, cut at register boundaries; data buses lie outside it.
- **Run.** [`runner.py`](../../tools/rtl_extract/runner.py) resets and flushes the LSU, issues the command at age 0, answers memory requests at the derived latency and records each age at which an event bundle is valid; an unknown control value is an error.

The `vlsu.vload` block:

```text
vmem_read:  ages 1–32, 32 rows, step 1
mreg_write: ages 3–34, 32 rows, step 1
first_free_age 35, next_issue_age 35
```

Variants guard the result: other operands leave the ages unchanged; a two-cycle memory response drops a row because the LSU never waits, so the facts record the latency as an assumption.

## 5. From facts to a compiler footprint

[`AtlasRTLEvidence.cpp`](../../lib/AtlasRTLEvidence.cpp) turns the block into the footprint of a VLOAD from VMEM line `L` into MREG `m`:

```text
read  VMEM lines L..L+31 at ages 1..32
write MREG m rows at ages 3..34
hold  the VLOAD path at ages 0..34, and L's VMEM bank at ages 1..32
```

Stream scheduling, delay insertion and the timing verifier use it; [`atlas-emit --rtl-timing-json`](README.md#reproduce) exports it.

## 6. Checking against execution

Pair sweeps ([`pairs.py`](../../tools/rtl_extract/pairs.py)) rerun the control simulation with a second command a gap later, on `LSU` alone and on `AtlasCore` through the RTL decoder, checking all 177 `AtlasCore` assertions; chain sweeps ([`sequences.py`](../../tools/rtl_extract/sequences.py)) do the same for MXU and VPU chains. Gap coverage compares every compiler gap with the sweep bounds. Replay runs scheduled programs on Verilator `AtlasCore` and binds each trace's accesses, ages and responses to the compiler export. See the [verification map](verification.md).

<details>
<summary>Excerpt locations</summary>

Excerpts: regenerated `ee290.debug.hw.mlir` (SHA-256 `249cafcaf97f083eae4282201de65d925f5cb11d12d2809178b7c8613342349a`), the IR the current facts pin. `ScalarCore` starts at line 492944, `LSU` at 495439. SSA numbers and location aliases change with each lowering; search for the named values.

</details>
