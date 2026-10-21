# AtlasCore raw-program checks

`probe.py` accepts an explicit `--rtl-root`, `--provenance` source manifest, Java/firtool executables,
Scala compiler/plugin JARs, and cached dependency JAR/export paths. It records the selected revision, source closure,
file identities and observed tool versions, then checks their identities again after elaboration. Source counts come
from the selected manifest. Dependency JAR source build identities remain unknown.

`build_runner.py` accepts `--core-root` and `--core-sha256` for that completed elaboration, plus explicit
`--verilator-binary`, `--verilator-root`, `--cxx`, `--ar`, `--make`, and `--out` paths. Optional repeatable
`--runtime-library` arguments supply libraries to preload. It records those tools and builds the raw instruction
runner from the emitted interface. These scripts also accept an explicit support helper where indicated; helper
identities are recorded per run. The raw runner is distinct from the retained-core interface used by the ResNet graph
driver.

`run_program.py` authenticates producer images, independently rebuilds physical arithmetic, and loads raw words through
IMEM. Completion requires committed ECALL status 5, public exported engines idle, exact output, unchanged inputs/guards,
requested aggregate DMA traffic, and 32 quiet cycles. Public debug omits VPU busy, so the receipt does not claim that
signal was observed. The next-cycle FIFO responder provides functional evidence; its cycles do not qualify platform
performance.

Per-cycle and DMA traces are off by default. `--trace` explicitly enables diagnostic capture and independent replay of
physical DMA/CSR events. Without it, the receipt labels the evidence as physical output plus aggregate traffic
summaries. Build/replay subprocesses have a 1500-second wall limit. Optional `--cpu` sets simulator affinity; callers
allocate resources externally. These tools contain no CPU reservation or session dispatcher.

By default the oracle rebuilds Fraction arithmetic. `--reuse-model-arithmetic --model-root PATH` instead authenticates
an exact passed model payload, including native inputs, producer sources, imported model sources and memory curve
before/after. It refuses controls or ambiguous old/new output filenames and labels arithmetic reuse explicitly.
Recorded receipts retain their original producer identity. Program operand modes and command metadata must agree, with at
most one 256-KiB resident bank and independently derived DMA counts.

The model revision and registered assembler syntax are pinned. DRAM DMA counts, VMEM traffic and engine occupancy have
separate meanings. These checks establish no native task-quality or platform-timing claim.

`resnet_graph_runner.cpp` is the separate C54 host-graph harness. `../prepare_resnet_graph_rtl.py` prepares its
model-bound images and optionally links against an explicitly supplied generated RTL object directory. Preparation never
runs the driver. The driver defaults to no waveform or DMA log; diagnostic `--vcd PATH` and `--dma-trace` are explicit.
See [the host graph guide](../../docs/resnet-full-host.md).
