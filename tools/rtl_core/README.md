# Current AtlasCore raw-program checks

The accepted route materializes 129 SHA-checked RTL sources, uses pinned real TileLink dependency JARs with unknown source build epochs, and compiles/emits with Scala 2.13.16, Chisel 6.7.0 and firtool 1.62.1. This differs from the component experiments' Chisel 7.0.0-RC4 tuple. The Mill recipe is retained but was unused by the accepted direct compiler route.

`run_program.py` authenticates retained producer images, rebuilds their physical arithmetic independently, loads raw instruction words through IMEM, and replays physical DMA/CSR traces. The next-cycle FIFO memory responder establishes functional evidence. Its cycles do not qualify platform performance. Completion requires committed ECALL status 5, public exported engines idle and 32 quiet cycles; VPU busy is absent from the public debug interface.

`probe.py` remains byte-identical to its accepted build epoch because its original source path/hash is bound by build and execution receipts. Its historical guard requires CPUs 0–2. Post-build `--cpu` replay uses `supervisor.py`, a parameterized extraction of that owned-process guard, with explicit affinity and resource caps. It can also supervise the entire replay command, including Python preparation and physical trace validation. Earlier sources and failure/result packages remain in immutable artifact snapshots.
