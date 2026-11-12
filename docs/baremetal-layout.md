# Selecting a device I/O base for an affine layout region

The narrow `atlas_mlir_native.layout_device.compile_source` and
`prepare_c_package` APIs accept `io_memory_base`, in bytes. Supply the same
value to both calls; C export replays and verifies the compiled placement.
The default remains `binding.DRAM_BASE` (`0x90000000`), with unchanged
publication bytes. This option changes placement only, preserving raw input
and output types and values. Prefix schedules currently reject a custom base.

For example, compile a source-bound region with
`compile_source(snapshot, source, target_source=target, destination=out,
limits=limits, io_memory_base=0x0800e800)` and export with
`prepare_c_package(source, out, name, target_source=target,
io_memory_base=0x0800e800)`. The caller must reserve its complete workspace
in the platform linker layout. The generated execution plan records each
buffer address and extent.

The EE290 target admits DMA spans in its 64 KiB scratchpad at `0x08000000`
and its existing external memory window. Each complete transfer must lie
in one window. Alignment, nonwrapping address proofs, buffer separation,
initialization and final machine contracts continue to apply. A base does
not reserve host memory or establish a new hardware mapping.

Zero-byte DMA spans are rejected. An absent boundary input reserves no span;
this is distinct from issuing a zero-byte transfer. The planner checks the
complete workspace before invoking compiler tools, and the source, boundary
and tile contracts check complete transfer spans, including scalar mailboxes.
