# From Atlas RTL to compiler timing

This walkthrough follows one VLOAD from Atlas hardware to the read/write windows used by the compiler. The process has three parts: **index the retained CIRCT structure, review the relevant control and register logic, then check the resulting rule against execution.** The indexer does not infer complete instruction timing automatically.

## 1. Stop before Verilog emission

For this example, the path is **Chisel → FIRRTL → CIRCT HW/Comb/Seq**. The [retention helper](../../tools/retain-ee290-hw-ir.py) runs `firtool --ir-hw` on selected FIRRTL and annotations instead of emitting split Verilog. At this boundary, `hw` describes modules and connections, `comb` describes combinational expressions, and `seq` preserves clocked state. We need all three to distinguish a wire connection from a cycle of latency.

<details>
<summary>Compilation pipeline and extraction boundary</summary>

<img src="figures/chisel_and_sv_circt_convergence.svg" alt="Chisel and other frontends converge on CIRCT core dialects; HW, Comb and Seq are highlighted." width="540">

The highlighted core dialects are our extraction boundary. This walkthrough follows the Chisel/FIRRTL route; the other frontend and backend routes provide context.

</details>

The normal VCS build produces SystemVerilog and a simulator. Retaining hardware IR is a separate step; its [manifest](retained-hw.md) records the selected inputs, lowering command and output identity.

## 2. Follow the command into the LSU

The [indexer](hw-evidence.md) starts at `AtlasCore`, records its instances and port connections, and follows named event expressions backward through their SSA operands. For VLOAD, the relevant path is small:

| Module / instance | What we inspect |
| --- | --- |
| `AtlasCore.scalar` / `ScalarCore` | Whether an instruction fires and launches a vector-memory command. |
| `AtlasCore` | The connection from `scalar.io_lsuCmd` to `lsu.io_cmd`. |
| `AtlasCore.lsu` / `LSU` | Command capture, row counter, response/write registers and busy signal. |
| `AtlasCore.vmem` and `AtlasCore.mreg` | The memory ports, bank selection and shared-access conditions. |

Here are two selected operations from `ScalarCore` in the retained IR:

```mlir
%26 = comb.and bin %pc_ctrl.io_s1_valid, %24, %25
    {sv.namehint = "s1_fire"} : i1 loc(#loc19350)
%200 = comb.and bin %26, %decoder.io_decoded_is_lsu
    {sv.namehint = "is_lsu_launch"} : i1 loc(#loc12156)
```

Following `%24` and `%25` reaches `!stall` and `!halt_now`. Following `%200` forward reaches `ScalarCore`'s command-valid output; the `AtlasCore` instance connects it directly to the LSU, without another register. The LSU captures a VLOAD only when the opcode matches and its load state is idle. For a legal command, **that capture edge is age zero**. Busy-tail and memory conflicts are additional obligations; command-valid alone is not permission to issue. Other engines have their own [admission conditions](admission.md).

## 3. Follow register inputs to recover the stages

Starting from the MREG write-valid output leads back to `vloadWritePending`. The selected Chisel control, with row/data assignments omitted, is:

```scala
vloadRespPending := vloadIssueRead
when(io.vmemVecReadData.valid && vloadRespPending) {
  vloadWritePending := true.B
}.otherwise {
  vloadWritePending := false.B
}
io.mregWriteReq.valid := vloadWritePending
```

The corresponding CIRCT operations make the clock boundaries explicit. These selected lines are reordered for reading; initialization and name-hint attributes are omitted:

```mlir
%vloadRespPending = seq.firreg %114 clock %clock
    reset sync %reset, %false : i1 loc(#loc12500)
%124 = comb.and bin %io_vmemVecReadData_valid, %vloadRespPending
    : i1 loc(#loc12564)
%vloadWritePending = seq.firreg %124 clock %clock
    reset sync %reset, %false : i1 loc(#loc12502)
```

`%114` is the load read-request condition. Each `seq.firreg` samples its input on a clock edge; `comb.and` adds no register stage. The indexer's local expression trace stops at registers and instance boundaries. We inspect each register's next-value expression and follow the recorded port connection to continue the analysis.

Source locations make that review traceable. For example, the response condition above resolves to:

```mlir
#loc12564 = loc("generators/sp26-atlas-acc/src/main/scala/atlas/lsu/LSU.scala":273:33)
```

Locations identify source origins; optimization can merge expressions. Read the location, operands and wiring together. SSA numbers and location aliases can change in another lowering.

The load state machine captures the command, enters Run to issue successive VMEM row reads, then passes through Drain to idle. The registers carry responses to MREG writes. The initial replay supplies a one-cycle memory response because the SRAM bodies are external at this boundary. Busy includes pending responses and writes, so idle state alone does not establish release. The [VLS timing reference](vls-timing.md) gives the exact ages and supported conditions.

## 4. Turn the rule into an operand-specific footprint

For a supported VLOAD targeting MREG `m` from effective VMEM line `L`, the shared provider resolves the reviewed rule into this simplified footprint:

```text
read  VMEM: first=L,    count=32, age=1, step=1
write MREG: first=32*m, count=32, age=3, step=1
hold  VMEM bank: ages 1 through 32
hold  both VLS paths: ages 0 through 34, inclusive
```

The provider checks the address/register domain and conservatively reserves both VLS paths. Scheduling and delay insertion consume the same `Access` and `Hold` facts; final timing verification recomputes them for the emitted stream. [`atlas-emit --rtl-timing-json`](resolved-export.md) exports those resolved facts with the selected evidence and applicability assumptions.

## 5. Check the interpretation against execution

The [LSU replay](vls-timing.md#boundary-checks-and-reproduction) first checks the rule under its explicit memory assumptions. [AtlasCore replay](vls-observation.md#selected-atlascore-replay-with-behavioral-srams) and [full-system VCS capture](vcs-observation.md) then observe the emitted programs: instruction issue, memory requests/responses, destination writes, release, and completion. Numerical checks also verify output and preserved memory.

The baseline and scheduled VLS programs have passed these bounded execution checks. That corroborates the interpreted pipeline for the observed cases; it does not turn a structural index into a general timing proof. The [handoff evidence table](selected-evidence.md#evidence-available-for-the-handoff) records the current validation scope. The provider remains conditional until its remaining operand and environment obligations are accepted.

<details>
<summary>Exact excerpt locations</summary>

The selected `ee290.hw.mlir` has SHA-256 `49fa1794b3b389941dc816bf23a1e16b0190c51dd5277347c1353736a05a3ac6`. Scalar operations are at lines 260781 and 260955; LSU operations at 262814, 262816 and 262881; the source alias at 343371. The Chisel excerpt is `LSU.scala:268–281`, compared with selected sources at revision `2ae0bef209df6db78c3de18e8f651bb43855cce9`. That source snapshot alone does not establish original elaboration provenance; see the [source/build evidence](selected-evidence.md#evidence-available-for-the-handoff).

</details>
