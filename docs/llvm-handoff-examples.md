# Atlas MLIR to LLVM MLIR handoff examples

These two **hand-authored machine-stage diagnostics** show the last MLIR stage
produced by this out-of-tree package. They are not generated from PyTorch,
Linalg, or Merlin's native selector. They do not qualify whole-model MLP or
attention compilation.

| Example | Intended instruction chain | Atlas words | Checked-in LLVM MLIR |
| --- | --- | ---: | --- |
| [MLP tile](../test/examples/handoff_mlp_tile.mlir) | MXU0, BF16 ReLU, E8M0 pack, MXU1 | 57 | [LLVM snapshot](../examples/handoff/handoff_mlp_tile.llvm.mlir) |
| [Attention tile](../test/examples/handoff_attention_tile.mlir) | QKᵀ, row maximum, subtract, multiply by a BF16 approximation to log₂(e)/√32, exp2, row sum, reciprocal, normalize, E8M0 pack, PV | 71 | [LLVM snapshot](../examples/handoff/handoff_attention_tile.llvm.mlir) |

Both inputs use fixed 32×32 FP8 tiles and already oriented weights. The MLP
has no bias; the attention sequence has no mask or causal state. `SELI 127`
selects unit E8M0 scale. The BF16 scale constant is raw `0x3e83`. MXU0 and
MXU1 have different arithmetic. The handwritten delays are diagnostic spacing
based on public instruction examples, **not** general hardware availability
bounds. The selected RTL revision is
`0079c0541111197741a231c002e3843fa6f545b2`.

## Actual last stage

The output has one function and one ordered, side-effecting inline assembly
operation. The MLP snapshot starts as follows; the full checked-in files retain
all 57 or 71 words:

```mlir
module {
  llvm.func @atlas_program() {
    llvm.inline_asm has_side_effects ".word 0x00100e13\0A.word 0x00000293\0A...", "~{memory}" : () -> ()
    llvm.return
  }
}
```

The real LLVM MLIR uses the precise word string shown in each snapshot. The
pass emits one block so LLVM keeps Atlas branch targets and delay-slot words
adjacent. It is a void reset-entry program body, not a C-callable Atlas ABI.
The `.word` stream is not executable on a generic host RISC-V CPU.

To reproduce a complete bundle with the Atlas source, LLVM MLIR, LLVM IR,
RISC-V object, extracted `.text`, words, and hashes:

```sh
python tools/export_llvm_handoff.py \
  --atlas-bin-dir build/bin \
  --llvm-bin-dir /path/to/matching/llvm-install/bin \
  --output-dir out/llvm-handoff-fresh
```

The checked-in snapshots omit the printer's extra final blank line. The output
directory must be empty. The exporter compares the leading object
words against `atlas-emit`, checks that exactly one side-effecting inline
assembly operation contains all source words, and records the source/tool
identity in `manifest.json`. The checked-in LLVM snapshots are regression
fixtures; `test/test_handoff_examples.py` compares them with fresh lowering,
ignoring only trailing whitespace. Rebuild `atlas-opt` from this repo and use
matching LLVM/MLIR tools before reproducing them.

## What the selected-core smoke runs establish

On the source-linked standalone `AtlasCore`, three uniform FP8-one MLP input
tiles produced 1,024 BF16 outputs of `0x4480` (1024). Zero Q/K and FP8-one V
produced 1,024 BF16 attention outputs of `0x3f80` (1). Both streams halted,
preserved their three inputs and a DRAM guard, and performed 96 DMA reads and
64 writes. The MLP run took 897 selected-core cycles; attention took 1,359.
These are two directed smoke inputs, not a timing qualification or general
model result. `test/test_handoff_examples.py` reproduces them when the selected
standalone-core environment is supplied.

The next nonuniform MLP input reveals a required layout conversion. With an
activation of 1 at logical row 0, column 20 and two identity weights, a
logical two-layer MLP would return 1 at row 0, column 20. This stream returns
zero there and 1 at row 16, column 4. The selected BF16-to-FP8 pack joins
successive physical 16-lane rows; the downstream MXU expects a different
32×32 layout. The attention sequence has the same gap before PV. With a
single elevated logit for key 20 and identity V, its elevated result appeared
at row 16, column 4 rather than row 0, column 20 in a diagnostic run.
Neither example should be used as a correct general MLP/attention program
until a checked relayout and numerical policy are added. Both sparse layout
gaps have executable regression tests.

## Proposed delay-analysis handoff

The LLVM snapshot is useful for object generation and exact byte comparison.
It has lost operation names, physical resource roles, and the reason for each
delay. A delay scheduler should consume the paired **Atlas machine MLIR**
before `--convert-atlas-to-llvm`, plus a source-bound timing/effects contract.
It can return a scheduled Atlas stream; this package can then revalidate and
lower it to the same LLVM form.

A small versioned sidecar could map each non-`atlas.start` operation to its
zero-based word index and carry its reads, writes, completion event, and
evidence tier. For example, an MXU launch would state that the accumulator
cannot be popped until its completion condition is met. A DMA launch would
identify the channel and scalar address register lifetime; an architectural
`atlas.dma_wait` would discharge that event. An `atlas.delay` records an
explicit stall count, not proof that all older work has completed. Static
latencies should carry the selected RTL/configuration identity and the test
or proof establishing their bound; unknown latency stays unknown.

For a consumer that only accepts LLVM MLIR, the source-to-word-index sidecar
is necessary for analysis. Editing individual `.word` instructions inside the
inline-assembly string would bypass this dialect's verifiers and emission
checks; schedule edits should return to Atlas MLIR and be lowered again.
The exact adapter to another project remains to be defined against that
project's parser and pass interface.
