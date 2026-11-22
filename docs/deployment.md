# Small on-chip deployment

Use the configured compiler environment and a fresh output directory:

```sh
atlasc deploy EXPORT --jtag -o OUT
```

This compiles the authenticated export with internal onchip deployment, generates
an independent nonzero reference, executes bound EXACT and an output-changing
mutation before linking (bit 7 of the first byte of input zero whose low seven bits are nonzero; when that
reaches no output, one retry flips the sign of input zero's first nonzero element), and packages the Baremetal ELF and runner tools.
Only the qualified small Linear plus ReLU exports at K32 and K64 are covered.
Device storage must fit 786,432 bytes, with a separate 65,536-byte host window.
Instruction images load with the device window, after its bound bytes, and the
host copies them to IMEM from there. A sizing link places the measured host
sections; they and the provider-declared stack must fit the host window
together, with no separate code or data ceiling. Deploy links the compiled host
once right after `atlasc build`; when that host exceeds the declared host memory,
it compiles again with the measured size, and Merlin then stages the plan
(`deployment_host_overflow`) instead of placing it on chip. The generated C is compiled
with `-fstack-usage -fcallgraph-info=su`, and the object stage sums generated
frames from GCC's own data. It refuses malformed or missing usage data, dynamic
frames, a generated function symbol without parsed static usage, any executable
byte outside a function with a parsed frame, a call or taken address that resolves
to code without a parsed frame, and any cycle in its call
model, and a sum above the declared stack. Calls and taken addresses resolve
through symbol addresses, so aliases and weak definitions reach their functions.
In the model an indirect call may reach every generated function whose address
the generated objects take, and any library symbol they take, and library code
may call back any of those generated functions, except from memcpy, memmove,
memset, memcmp, and strlen. Assumed, not checked: no arithmetic on function
pointers; library code reaches generated code only through those taken
addresses, apart from startup's direct call to main; and inline asm in generated
C neither writes sp nor transfers control (the generated asm today is fence and
rdcycle). The sum covers generated frames only.
Prebuilt library and platform frames (newlib, libgcc helpers, `_write`, startup,
trap handler, drivers) are not bounded; their stack is only measured. No heap is
declared, so the stage also refuses code that links an allocator.
The final ELF check refuses any `fence.i`, any `vfmv.f.s`, and any FP-register
write followed by an FP load to the same register, naming the address: the
chip's Shuttle core keeps ucb-bar/shuttle issues 19 and 20 and the bug PR 22
fixes. It decodes the executable section bytes with the core's 16- and 32-bit
instruction lengths, so `.word` and `.insn` encodings, and instructions that
straddle data directives, are refused too. A jump into the middle of an
instruction it already decoded from an earlier start is not covered. No DRAM
loading, paging, or larger-model slicing is qualified here; see the board DDR deployment below.

The resulting kit supports file-only validation:

```sh
python OUT/kit/bench/run_kit.py OUT/kit --dry-run
python OUT/kit/bench/check_rung.py kit --bundle OUT/kit --dry-run
```

Both checks must pass. A dry pass provides no execution evidence or transport permission.

`atlasc run OUT --dry-run` repeats both checks and never opens transport. `atlasc run OUT` is the live run: it runs fresh preflight and one rung (the positive, or the expected-fail negative with `--negative`), checks the complete capture with the kit checker, and prints one verdict line with its log directory (`RUN PASS (negative rejected as expected)` for a negative that fails exactly as predicted). Like `run.sh`, it stops the rung at the same wall-clock bound, and also on SIGTERM or SIGINT, as a transport error (`rc=2`) once the runner's cleanup has stopped OpenOCD; the terminal capture (stdout and stderr) always ends with `rc=`. `--serial PATH` selects the board's UART device (default: the Linux FTDI `if01` path). Each further rung needs its own decision and command. `atlasc run OUT --inputs FILE ...` (or `--inputs -` for a loop on stdin) opens one resident session, runs one step per input file, and returns each step's outputs; a step with new inputs is unchecked unless `--reference DIR` names the independent reference that `atlasc reference OUT --inputs FILE -o DIR` computes. The session first runs the packaged inputs and compares them with the packaged references (`reference-open`), and repeats that after the last step (`reference-close`) only if a step was unchecked. For a single-ELF kit the positive rung still runs first, since its capture checker (`check_rung`) checks more than the session's reference call. For a composed plan, `reference-open` is the positive: the rung is the same call on a fresh session, so it does not run, and `atlasc run OUT --simulate --inputs ...` does not rerun the negative, which deploy ran and `atlasc run OUT --simulate` reruns. So N checked frames of a composed plan take N + 1 passes in one session, or two passes each with one `atlasc run` per frame, where a lost debug link costs only that frame's session. SIGTERM or SIGINT also stops the resident session at any point until its final records are made, including a wait for the next stdin line and its close, as a transport error after its close stops OpenOCD; its record, `OUT/run-*/session-terminal.txt`, lists each step and always ends with `rc=` (`rc=2` on any stop), and for a single kit `session/summary.json` carries the same verdict. A signal after the records are made reaches the previous handler once they are complete. Resident steps have not run on silicon; see the [runner guide](../python/atlas_mlir/chiprunner/README.md). The bench ladder uses the kit's `run.sh`, not `atlasc run`. To run kits from a laptop, see [laptop.md](laptop.md).

## Board DDR deployment

Without `--jtag`, `atlasc deploy EXPORT -o OUT` always places the application in the board's DDR as one ELF for the board loader; there is no automatic choice between the on-chip window and DDR. Every DDR fact comes from the declared SoC (`atlas_mlir_native/penguin_sophialake.json`, region `ddr`: 1 GiB at 0x80000000 on SophiaLake). Admission refuses early when the captured weights plus the fixture's inputs and outputs exceed the declared DDR (`DDR_CAPACITY_EXCEEDED`); final fit still requires linking. The host goes first at the DDR base, so the ELF entry is the DDR base. A measuring compile starts the device window at that base, and its sizing link measures the host's code, data, heap, and stack. The bound compile starts the device window after this measured host extent rounded up to 4 KiB, and ends it at the end of DDR. The final host objects and link must fit the measured host, or deploy refuses. The writable window load holds the bound inputs, outputs, and scratch. The parameters end the window data in their own read-only load (section `.atlas_parameters`, named by the Atlas provider), followed by the instruction images, which the host copies to IMEM. Every ELF load lies in DDR, and nothing loads into VMEM: the program's DMA fills it. A bench-staged plan keeps every tensor on the bench host, so a board deployment refuses it. Reference, objects, bind, the EXACT gate with its executed mutation, link, and packaging are the same stages as above. The kit has the ELF, its negative, the expected output bytes, and `bench/soc.json`, the declaration it was placed against, and no bench runner. `atlasc run OUT --simulate` runs it on the simulated board, and `atlasc run OUT` refuses until the board's SPI loader exists (see `python/atlas_mlir/chiprunner/README.md`). A board program ends the way the declared board host learns that it ended (`board_program` in the declaration): both known DDR hosts, the FireSim tethered image and SophiaLake's UART-TSI host, start the chip over TSI and end the run on a `tohost` write, so the image is built with `PENGUIN26_EXIT_TOHOST=ON` and has `tohost` and `fromhost`; `--jtag` images keep it off. Puffin's host is unconfirmed. A 0 exit code means only that `main` returned 0; the pass rule stays the `ATLAS STATUS`, `HASH`, and `COMPLETE` records. `--clock-mhz` sets the image's clock and UART divisor as above; a clock other than the declared host clock is compiled into the published deployment, so the kit records one clock. An installed `MERLIN_NATIVE_SNAPSHOT` is built for the on-chip window, so this path builds each compile's snapshot from the pinned Merlin, which needs Cargo.

The board kit also contains `memory-plan.json` and `loader-manifest.json`, both covered by `SHA256SUMS`, and both naming the SoC declaration by identity and SHA-256. The memory plan records the measured host exclusion, device window, linker sections, allocations, host symbols, and instruction image addresses. The loader manifest describes every PT_LOAD and the resident weight range as offsets into the same `chip.elf`, with SHA-256 hashes for chunks of at most 1 MiB and separate zero-fill tails. Weights are never duplicated in a second load image. Loading over SPI on Puffin is assumed. `atlasc run` loads from these files; they do not establish loader or board execution. Resume identity is each chunk's destination, length, and hash on one board session, so chunks shared by another ELF, such as its weights, are not loaded again; any write over a chunk's range forgets it. Every chunk must be verified at its destination and all zero-fill tails cleared before entering the ELF. The negative ELF remains a qualification artifact and is not selected by this loader manifest.

## Staged plans

When a plan does not fit the window, `atlasc build` publishes a staged application (`mode: bench_staged`): an ordered schedule of step packages, with the bench host holding every tensor. The same command deploys it per step. Each step package gets measured objects, and its sizing link must place them, its stack, and its heap inside its own host window; a step that does not fit is blocked, named in `deploy-status.json`, and the deployment's status is `PACKAGED_BLOCKED` instead of `PACKAGED`. The composed plan binds application inputs from the capture, step inputs, weights, and outputs as byte ranges of whole values, and references the application outputs with the independent source reference; every output reference must be nonzero. Before any link, the composed plan runs in simulation, each step as its generated application with every launch in the pinned functional model, and so does its declared negative, one byte change of the first application input that must fail at a referenced output. Bind chooses it with the independent reference: the first, middle, and last nonzero elements, each with bit 6 and then bit 7 of its most significant byte flipped, are evaluated in that order on the source inputs, and the first whose reference changes an application output is executed; `bundle/manifest.json` (`staged.negative`) records it and every candidate tried, and bind refuses if none changes an output (a trained model can absorb a single saturated pixel). Its receipt is `OUT/composed-check.json`. Each step is then linked with zero bytes in its input ranges (the runner writes inputs and weights before every start), and `OUT/kit` holds the step ELFs, values, ranges, references, negative, receipt, and runner. Simulation needs Spike, `dtc`, the pinned model, `COMPOSED_MODEL_PYTHON`, and `COMPOSED_RUNTIME_PATH`.

`atlasc run OUT --simulate` reruns the kit with each linked ELF unchanged in Spike and the simulated Atlas devices, then its negative. A `PACKAGED_BLOCKED` kit simulates each blocked step as its application, and `atlasc run OUT` refuses it. Intermediate step boundaries are not referenced: only application outputs are compared. Staged kits carry simulation evidence only; no staged deployment is qualified on silicon.

On refusal or failed checking, preserve the status, receipts, and complete logs.
Stop, diagnose the reported stage, and use a fresh output for a corrected deploy.
Never edit a receipt or substitute reference bytes to obtain a pass. Default
clock metadata is operator-stated 100 MHz and UART divisor 867, from the 115200
baud the SoC description declares for UART0; it is not a measurement of the
board clock.

Inherited Atlas priority grouping and selected fusion lists remain temporary.
Their replacement is declared implementation families with Merlin selecting the
cover. No model-specific policy or schedule is added by deployment packaging.
