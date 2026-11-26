# Small on-chip deployment

Use the configured compiler environment and a fresh output directory:

```sh
atlascc deploy EXPORT --jtag -o OUT
```

This compiles the authenticated export with internal onchip deployment, generates
an independent nonzero reference, executes bound EXACT and an output-changing
mutation before linking (bit 7 of the first byte of input zero whose low seven bits are nonzero; when that
reaches no output, one retry flips the sign of input zero's first nonzero element), and packages the Baremetal ELF and runner tools.
The EXACT checker ships in [`python/atlas_mlir/exactness`](../python/atlas_mlir/exactness/README.md). It runs in
`ATLAS_MODEL_CHECKER_PYTHON`, the interpreter its qualification was recorded on, and `ATLAS_CHECKER_QUALIFICATION`
names the directory with that qualification (`qualification.json` and `memory-qualification.json`).
This repository does not yet ship the command that records them, so a clean clone needs them from an existing qualification.
With `--jtag`, the gated exports are small Linear plus ReLU models at K32 and K64; larger models use the board DDR deployment or staged plans below.
These builds are diagnostic.
Device storage must fit 786,432 bytes, with a separate 65,536-byte host window.
Instruction images load with the device window, after its bound bytes, and the
host copies them to IMEM from there. A sizing link places the measured host
sections; they and the provider-declared stack must fit the host window
together, with no separate code or data ceiling. Deploy links the compiled host
once right after `atlascc build`; when that host exceeds the declared host memory,
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
loading, paging, or larger-model slicing is covered here; see the board DDR deployment below.

The resulting kit supports file-only validation:

```sh
python OUT/kit/bench/run_kit.py OUT/kit --dry-run
python OUT/kit/bench/check_rung.py kit --bundle OUT/kit --dry-run
```

Both checks must pass. A dry pass provides no execution evidence or transport permission.

`atlascc run OUT --dry-run` repeats both checks and never opens transport. `atlascc run OUT` is the live run: it runs fresh preflight and one rung (the positive, or the expected-fail negative with `--negative`), checks the complete capture with the kit checker, and prints one verdict line with its log directory (`RUN PASS (negative rejected as expected)` for a negative that fails exactly as predicted). Like `run.sh`, it stops the rung at the same wall-clock bound, and also on SIGTERM or SIGINT, as a transport error (`rc=2`) once the runner's cleanup has stopped OpenOCD; the terminal capture (stdout and stderr) always ends with `rc=`. `--serial PATH` selects the board's UART device (default: the Linux FTDI `if01` path). Each further rung needs its own decision and command. `atlascc run OUT --inputs FILE ...` (or `--inputs -` for a loop on stdin) opens one resident session, runs one step per input file, and returns each step's outputs; a step with new inputs is unchecked unless `--reference DIR` names the independent reference that `atlascc reference OUT --inputs FILE -o DIR` computes. The session first runs the packaged inputs and compares them with the packaged references (`reference-open`), and repeats that after the last step (`reference-close`) only if a step was unchecked. For a single-ELF kit the positive rung still runs first, since its capture checker (`check_rung`) checks more than the session's reference call. For a composed plan, `reference-open` is the positive: the rung is the same call on a fresh session, so it does not run, and `atlascc run OUT --simulate --inputs ...` does not rerun the negative, which deploy ran and `atlascc run OUT --simulate` reruns. So N checked frames of a composed plan take N + 1 passes in one session, or two passes each with one `atlascc run` per frame, where a lost debug link costs only that frame's session. SIGTERM or SIGINT also stops the resident session at any point until its final records are made, including a wait for the next stdin line and its close, as a transport error after its close stops OpenOCD; its record, `OUT/run-*/session-terminal.txt`, lists each step and always ends with `rc=` (`rc=2` on any stop), and for a single kit `session/summary.json` carries the same verdict. A signal after the records are made reaches the previous handler once they are complete. Resident steps have not run on silicon; see the [runner guide](../python/atlas_mlir/chiprunner/README.md). The bench ladder uses the kit's `run.sh`, not `atlascc run`.

## Board DDR deployment

Without `--jtag`, `atlascc deploy EXPORT -o OUT` always places the application in the board's DDR as one ELF for the board loader; there is no automatic choice between the on-chip window and DDR. Every DDR fact comes from the declared SoC (`atlas_mlir_native/penguin_sophialake.json`, region `ddr`: 1 GiB at 0x80000000 on SophiaLake). Admission refuses early when the captured weights plus the fixture's inputs and outputs exceed the declared DDR (`DDR_CAPACITY_EXCEEDED`); final fit still requires linking. The host goes first at the DDR base, so the ELF entry is the DDR base. Merlin derives the budget (`derive_budget` on region `ddr`, `atlas_mlir_native/memory_budget.py`): the region minus the reservations for code, runtime, stack, weights, state, and KV caches, and its one free range is the device window. A measuring compile takes the whole DDR as its window, with the host after the window data. Merlin's `measure_linked_budget` reads the host's code, runtime, and stack from its sizing link's symbols; weights, state, and KV caches have zero-byte reservations, because Merlin places them inside the window. The bound compile reads that budget with `--memory-budget`, with the host moved to the DDR base, so the window starts at the first 4 KiB boundary after the host and ends at the end of DDR. The final ELF is measured again the same way, and deploy refuses unless the window and resident arena lie in that budget's free range. The stack is the declaration's `host_stack` (8,192 bytes); the link refuses a generated stack bound above it. The writable window load holds the bound inputs, outputs, and scratch. The parameters end the window data in their own read-only load (section `.atlas_parameters`, named by the Atlas provider), followed by the instruction images, which the host copies to IMEM. Every ELF load lies in DDR, and nothing loads into VMEM: the program's DMA fills it. A bench-staged plan keeps every tensor on the bench host, so a board deployment refuses it. Reference, objects, bind, the EXACT gate with its executed mutation, link, and packaging are the same stages as above. The kit has the ELF, its negative, the expected output bytes, and `bench/soc.json`, the declaration it was placed against, and no bench runner. `atlascc run OUT --simulate` runs it on the simulated board, and `atlascc run OUT` refuses until the board's SPI loader exists (see `python/atlas_mlir/chiprunner/README.md`). A board program ends the way the declared board host learns that it ended (`board_program` in the declaration): both known DDR hosts, the FireSim tethered image and SophiaLake's UART-TSI host, start the chip over TSI and end the run on a `tohost` write, so the image is built with `PENGUIN26_EXIT_TOHOST=ON` and has `tohost` and `fromhost`; `--jtag` images keep it off. Puffin's host is unconfirmed. A 0 exit code means only that `main` returned 0; the pass rule stays the `ATLAS STATUS`, `HASH`, and `COMPLETE` records. The image's clock comes from the board it runs on: the declaration's `carriers` list gives each board its placement (`onchip` or `ddr`) and the chip clock it supplies. `--jtag` picks the placement, which picks the carriers, and deploy takes every carrier of that placement (`albatross` at 100 MHz with `--jtag`; `firesim-tethered` and `puffin` at 500 MHz without it). A second carrier of one placement with another clock needs a carrier-named target, which is not built yet. Deploy refuses when no carrier has the placement, when one declares no clock, or when they disagree; there is no clock flag. The clock sets `SYS_CLK_FREQ`, `MTIME_FREQ`, and the UART divisor (clock / baud - 1: 867 at 100 MHz, 4339 at 500 MHz), is compiled into the published deployment, and the kit names its carriers in `bench_facts`. The board budget is 12,000,000,000 chip cycles, recorded as `platform_assumptions.timeout_cycles` and as `timeout_seconds` at the carrier clock (120 s at 100 MHz, 24 s at 500 MHz). The serial-TL link clock belongs to each FPGA bitstream, not to a carrier, so no kit byte depends on it. An installed `MERLIN_NATIVE_SNAPSHOT` is built for the on-chip window, so this path builds each compile's snapshot from the pinned Merlin, which needs Cargo.

A board deployment keeps immutable device operands resident, so both deploy compiles select application residency with no flag. The compiler sizes the resident arena from its residency table, places it at the end of the DDR (within one alignment step of it), and stages the device window in the rest, so the measuring and bound compiles give every resident entry the same address. Each device region that reads a resident operand must pass the provider's source replay (`checked_resident_deployment`) before the deployment is published. The arena is one read-only load (section `.atlas_resident`) after the window, and the link checks every resident entry in it against the packed asset the compiler published; the memory plan and loader manifest list it as `resident`. A parameter with any host reader keeps its host copy in the window's parameters. Residency is an optimization: a model it cannot apply to (one with persistent state inputs, no external parameters, no device operand with a resident layout, or a device region that reads a parameter through an operation with no resident layout) deploys on the board exactly as without residency, and its compiled manifest records why as `application_residency_declined`. So does a model whose resident arena does not fit the DDR after the host or leaves no staging window. Only these kinds decline. Any other failure on the resident path still refuses the deploy: inconsistent resident layouts for one parameter, a packed layout whose size differs from its allocation, a changed parameter payload, a region that does not consume its resident bindings, a failed source replay, a staging window too small for a region, and a link refusal of the resident runtime (the stack bound, a changed resident byte, or a writable arena). A resident region's image is bound to its resident addresses at run time from a read-only template, so the link check exempts it from the linked image symbols; the functional model in the EXACT check and the simulated board refuse any launched image whose bytes differ from the published program, and the core errata scan covers all host code, including each region's image bind. `--jtag` deployments never select residency.

The board kit also contains `memory-plan.json` and `loader-manifest.json`, both covered by `SHA256SUMS`, and both naming the SoC declaration by identity and SHA-256. The memory plan records the measured host exclusion, device window, linker sections, allocations, host symbols, and instruction image addresses. The loader manifest describes every PT_LOAD and the resident weight range as offsets into the same `chip.elf`, with SHA-256 hashes for chunks of at most 1 MiB and separate zero-fill tails. Weights are never duplicated in a second load image. Loading over SPI on Puffin is assumed. `atlascc run` loads from these files; they do not establish loader or board execution. Resume identity is each chunk's destination, length, and hash on one board session, so chunks shared by another ELF, such as its weights, are not loaded again; any write over a chunk's range forgets it. Every chunk must be verified at its destination and all zero-fill tails cleared before entering the ELF. The negative ELF remains a qualification artifact and is not selected by this loader manifest.

## Staged plans

When a plan does not fit the window, `atlascc build` publishes a staged application (`mode: bench_staged`): an ordered schedule of step packages, with the bench host holding every tensor. The same command deploys it per step. Each step package gets measured objects, and its sizing link must place them, its stack, and its heap inside its own host window; a step that does not fit is blocked, named in `deploy-status.json`, and the deployment's status is `PACKAGED_BLOCKED` instead of `PACKAGED`. The composed plan binds application inputs from the capture, step inputs, weights, and outputs as byte ranges of whole values, and references the application outputs with the independent source reference; every output reference must be nonzero. Before any link, the composed plan runs in simulation, each step as its generated application with every launch in the pinned functional model, and so does its declared negative, one byte change of the first application input that must fail at a referenced output. Bind chooses it with the independent reference: the first, middle, and last nonzero elements, each with bit 6 and then bit 7 of its most significant byte flipped, are evaluated in that order on the source inputs, and the first whose reference changes an application output is executed; `bundle/manifest.json` (`staged.negative`) records it and every candidate tried, and bind refuses if none changes an output (a trained model can absorb a single saturated pixel). Its receipt is `OUT/composed-check.json`. Each step is then linked with zero bytes in its input ranges (the runner writes inputs and weights before every start), and `OUT/kit` holds the step ELFs, values, ranges, references, negative, receipt, and runner. Simulation needs Spike, `dtc`, the pinned model, and `COMPOSED_MODEL_PYTHON`; each step runs with the checker's functional runtime in `python/atlas_mlir/exactness`.

`atlascc run OUT --simulate` reruns the kit with each linked ELF unchanged in Spike and the simulated Atlas devices, then its negative. A `PACKAGED_BLOCKED` kit simulates each blocked step as its application, and `atlascc run OUT` refuses it. Intermediate step boundaries are not referenced: only application outputs are compared. Staged kits carry simulation evidence only; no staged deployment is qualified on silicon.

On refusal or failed checking, preserve the status, receipts, and complete logs.
Stop, diagnose the reported stage, and use a fresh output for a corrected deploy.
Never edit a receipt or substitute reference bytes to obtain a pass. Clock
metadata is the declared carrier clock (100 MHz and UART divisor 867 for
`--jtag` kits), from the 115200 baud the SoC description declares for UART0; it
is not a measurement of the board clock.

Inherited Atlas priority grouping and selected fusion lists remain temporary.
Their replacement is declared implementation families with Merlin selecting the
cover. No model-specific policy or schedule is added by deployment packaging.
