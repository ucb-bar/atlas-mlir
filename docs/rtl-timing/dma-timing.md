# DMA capture and completion evidence

DMA transfers capture scalar operands and global base state at launch; their memory lifetime ends through a matching completion wait. This reference records a static trace through the selected retained EE290SimConfig hardware, with a conservative proposed compiler domain. It supplies no executed DMA timing qualification and no fixed external-memory completion bound. See [command admission](admission.md), [retained hardware](retained-hw.md), and [selected evidence](selected-evidence.md) for selection and qualification requirements.

## Selected identities and locators

All HW line numbers below refer to `build/rtl-timing/ee290-portable-helper-check/ee290.hw.mlir`, SHA-256 `49fa1794b3b389941dc816bf23a1e16b0190c51dd5277347c1353736a05a3ac6`. Its `manifest.json` selects FIRRTL SHA-256 `5c6eb7ce8d0005354eb9a2d3f0c0d52bc105c6261ff25b7081825428b93780fa`; the retained artifact has 774 module definitions. `ScalarCore`, `TileLinkAdapter`, `DmaEngine` and `AtlasCore` begin at HW lines 260678, 261265, 261336 and 316694 respectively. These are local evidence locators, not portable file links or sufficient proof of another elaboration's semantics.

Source locators refer to immutable copies selected by `build/rtl-timing/ee290-source-inventory-001/inventory.json`, SHA-256 `939fee9cb515f3f1b62a8c6d6743a312284856efd9e8a51140f83c24b7782a23`. Inventory roles identify these as captured current Scala sources; source snapshots alone do not reconstruct the original Chisel execution edge. The corresponding retained HW equations were independently inspected for the facts below.

| Snapshot | Original source | SHA-256 |
| --- | --- | --- |
| `snapshots/005.scala` | `atlas/common/DmaParams.scala` | `7d335eff289859e6d5a114c76593aa47bbcf24b521645f98268f981c9136d4ad` |
| `snapshots/033.scala` | `atlas/scalar/ScalarCore.scala` | `3ae39e67520d7d684d3aaa57c060ca4f7b191f30be8b783ae3b0f4d8f0aeca3a` |
| `snapshots/067.scala` | `diplomatic/memory/DMA.scala` | `9b802f714f8afa08758ea97d2e92629645e99bf51d7d7af943fb87a93344dfad` |
| `snapshots/070.scala` | `diplomatic/top/AtlasCore.scala` | `cac182a402e37a27f688a1d714b238e7b4ad5189c0b19f3f3e21580e9e60c78b` |

## Launch, configuration and effective operands

`ScalarCore.s1_fire` requires stage-1 valid, no delay/DMA-wait stall and no halt condition (HW 260778–260781). LOAD/STORE launch is `s1_fire && (DMA_LD || DMA_ST)` at HW 261208–261211. CONFIG is `s1_fire && DMA_CONFIG` at HW 261212–261213 and synchronously writes one global `dmaBaseReg`; it asserts no engine command and acquires no transfer slot or channel. WAIT also asserts no engine command. Snapshot 033 lines 489–509 and HW 261220–261223/261262 show the complete assembly and output path.

| Operand | LOAD | STORE | Effective hardware field |
| --- | --- | --- | --- |
| VMEM pointer | `rd_data` | `rs1_data` | Scalar pointer is in 32-bit words; root extracts bits `[18:3]` to a 16-bit line address (HW 316718). Each line contains eight words/32 bytes. |
| DRAM offset | `rs1_data` | `rd_data` | Byte offset, concatenated with the global 32-bit base: `(base << 32) \| offset`, rather than adding the base (HW 261223). |
| Size | `rs2_data` | `rs2_data` | Bytes; root extracts bits `[12:0]` (HW 316719). Engine derives beat count from bits `[12:5]`, discarding partial-beat bits (HW 261505–261506). |
| Channel | decoded `funct3` | decoded `funct3` | Three bits, channels 0–7 (HW 261262 and root command connection at 316709). |

The queue captures direction, channel, VMEM line, assembled 64-bit DRAM address and 13-bit size on the launch clock edge. For slot 0, HW 261755–261761 uses only `command.valid && enqueueIdx==0` to select each captured field; registers are HW 261422–261426. Equivalent slot paths cover all eight slots. Subsequent scalar-register and base writes cannot retarget those captured fields. Snapshot 067 lines 223–229 shows the corresponding Valid-only enqueue.

Each accepted TileLink A beat adds `beatIndex * 32` to the captured DRAM address (HW 261870–261871). `TileLinkAdapter` then extracts address bits `[36:5]` and appends five zero bits (HW 261305–261306): the selected external bus is 37 bits and each request rounds down to 32-byte alignment. Capturing 64 bits does not mean the external interface preserves all 64 bits. Unknown addresses, high-bit aliases and misaligned pointers need rejection or explicit supported semantics.

Reset initializes the global base to zero, and host START also clears it to zero (snapshot 033 lines 67 and 538–540; HW 260738 and 261252). START does not drain or reset the DMA engine. A compiler entry contract must require quiescent DMA. Initial base zero follows a fresh host START or reset, not an arbitrary function/block entry; otherwise require a known incoming base or CONFIG before launch.

## Admission and dynamic completion

There are eight command slots, eight Boolean channel-busy registers, six-bit TileLink source IDs, a 64-beat in-flight limit and a 128-entry store-data queue (snapshot 005 lines 27–37; selected HW 261422–261506/261718/261865–261869). Enqueue advances its three-bit index modulo eight whenever valid is asserted (HW 261859–261860). No command-ready, full, occupied-slot or busy-channel predicate gates capture. Ring slots are assigned by enqueue order, independently of encoded channel. Channel freedom cannot establish slot freedom, and repeated generations on one Boolean channel are not represented independently.

The engine captures per-source direction, slot and VMEM destination line at accepted A requests, then uses the returning D source ID to recover that metadata (snapshot 067 lines 255–260 and 280–289; selected HW source metadata begins at 261508, lookup at 262238–262243). LOAD D acceptance requires VMEM write grant; STORE acknowledgments can retire without a VMEM write (HW 262244–262249). STORE VMEM reads advance only on a read grant, and data is queued before A dispatch (HW 261702–261726). VMEM data is therefore read or written during transfer progress, not atomically at one predicted completion cycle.

Slot retirement requires `slotActive && slotDispatched && willBeZero` (slot 0: HW 262274–262281). `willBeZero` accounts for simultaneous A/D acceptance: increment-only is false, decrement-only requires the previous outstanding count to equal one, and equal increments/decrements require zero. Retirement clears the active/dispatched state and the busy bit selected by the captured command's channel (HW 262281–262290; snapshot 067 lines 299–320). Last A dispatch alone cannot establish completion.

Scalar WAIT stalls while stage-1 contains DMA WAIT and `busy[funct3]` is asserted (HW 260759–260762); successful `s1_fire` occurs after that busy indication clears and ordinary frontend conditions permit retirement. Integrated busy is `DmaEngine.channelBusy[ch] || dmaLaunched[ch]` (HW 316768–316775). Launch sets `dmaLaunched[ch]`, and observing engine busy clears the bridge (HW 316720–316767; snapshot 070 lines 185–191). The bridge covers an immediately following matching WAIT at the launch/busy observation boundary. A wait on a different channel supplies no completion proof for the pending transfer.

External ready/valid traffic, response latency, source-ID availability and VMEM grants regulate progress. No finite DELAY or byte-count latency formula supplies a guaranteed completion event. Completion-established memory visibility remains conditional on correct external memory behavior and the selected VMEM implementation; static engine retirement is not itself a numerical or integrated-system qualification.

## Conservative compiler domain

A minimal domain can admit straight-line LOAD/STORE/CONFIG/WAIT while keeping globally at most one pending transfer. Before another transfer, any VMEM instruction, completion marker or halt, require a successful WAIT on that transfer's captured channel. Treat the captured VMEM/DRAM ranges as live until that wait. Permit ordinary scalar/base overwrite after capture, retaining operand-producing dependencies before launch. CONFIG stays a synchronous global-base write and creates no pending transfer. Unknown operands or unsupported control flow reject.

Require known full scalar values that satisfy the following predicates before truncation: channel 0–7; size 32–4096 bytes inclusive and divisible by 32; VMEM word pointer divisible by eight, with `wordPointer + size/4 <= 393216`; and the full transfer remains in one 65536-word/8192-line bank. The six-bank VMEM range is independently reflected by the selected line-range assertions against unsigned 49151 (HW 261340 and 261721–261729), and bank/line slicing uses bits `[15:13]`/`[12:0]` (HW 261708–261709/262245–262246). One-bank restriction is an intentionally conservative admission predicate, not a newly observed hardware requirement.

Require DRAM offset divisible by 32, global base at most 31 and assembled interval end at most `2^37`; this avoids selected-bus truncation and wrap. An additional rejection of offset-plus-size crossing `2^32` is a conservative policy choice, since the hardware's captured 64-bit addition carries into the upper field. Memory ownership must exclude conflicting host/system traffic during the pending interval, and entry must have no preexisting DMA or VMEM activity.

The 13-bit size encoding alone does not establish safe sizes up to 8191. Nonmultiples of 32 truncate, zero-beat sizes violate the range assertion, and sizes beyond 4096 can exceed the selected 128-entry store-data queue if TileLink blocks: VMEM reads are not gated on its enqueue-ready signal (snapshot 067 lines 196–210; HW 261707/261718). Restricting an isolated store to at most 128 beats avoids needing a proof of progress-dependent queue capacity. It does not qualify arbitrary queued transfers or concurrency.

## Evidence and reproduction

[`replay-ee290-dma.py`](../../tools/replay-ee290-dma.py) builds a selected AtlasCore from retained SystemVerilog and behavioral SRAM snapshots, then executes five finite cases: CONFIG without a transfer, operand changes before launch, operand changes after launch, scalar/global-base changes while pending, and waited channel reuse. The [harness](../../test/ee290-dma-replay.cpp) supplies deterministic TileLink backpressure and delayed responses, checks data and untouched DRAM/VMEM guards, and requires the completion marker and normal halt. The checker independently decodes the VCD against the emitted instructions, recomputes captured operands, and checks accepted A/D requests, VMEM grants, busy protection and successful matching waits.

The `atlas.ee290_dma_replay.v0` receipt binds the selected manifest/HW, RTL/harness/producer snapshots, captured build and execution phases, exact model/program bytes, raw traces and compact observed events. `finite_dma_cases_passed` describes those observations only. The finite cases use 128-byte transfers on channels 0/1 and initial base zero; a real CONFIG to base one while pending tests captured-address stability. They do not qualify every size/bank/channel, CPU/SoC execution, queued concurrency or a finite completion latency. The static domain above remains an explicit conditional extrapolation.

Run from the compiler repository with explicitly selected artifacts and tools:

```sh
python3 tools/replay-ee290-dma.py \
  --selected-core-replay /path/to/selected-core/report.json \
  --expected-replay-sha256 SELECTED_REPORT_SHA256 \
  --output build/rtl-timing/dma-replay-new \
  --verilator /path/to/verilator_bin --verilator-root /path/to/verilator-runtime \
  --cxx /path/to/g++ --make /path/to/make --ar /path/to/ar \
  --atlas-emit /path/to/atlas-emit --jobs 4
python3 -m unittest discover -s test -p test_ee290_dma_replay.py -v
```

An optional `--reuse-compile /path/to/compile/phase.json` reuses a model only after its complete selected input identities, recipe, environment and output identity match. Failed runs remain separate artifacts. Evidence selection and final timing verification must retain pending memory through scalar/base reuse, reject missing/wrong waits and pending publication, and preserve unknown wait duration; see the [consumer contract](compiler-consumers.md).
