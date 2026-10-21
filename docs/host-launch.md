# Host launch

An Atlas program runs as a reset-entry instruction stream. A bare-metal RV64
host (Shuttle, in the EE290 SoC) copies the program words into Atlas IMEM,
places inputs in DRAM, writes the start CSR, and polls the status CSR until
Atlas halts on ECALL. Atlas reads and writes DRAM through its own DMA. In the
example host ELF, a runner talks to the host through a mailbox in host memory,
the generated graph wrapper converts f32 memrefs to device tiles, and
`runtime/atlas_host.c` drives Atlas over MMIO.

| Piece | Role |
| --- | --- |
| [`runtime/atlas_host.h`](../runtime/atlas_host.h) | Load, start, poll, and wait for one Atlas program. |
| [`runtime/atlas_graph_memref.h`](../runtime/atlas_graph_memref.h) | Convert MLIR f32 memrefs to device tiles, launch, and read results back. |
| `atlas-boot-pack`, `atlas-host-embed` ([`tools/`](../tools/atlas_boot_pack.py)) | Package a checked program as a capsule; turn a capsule into a C header. |
| [`tools/atlas_graph_platform.py`](../tools/atlas_graph_platform.py) | Check a compiled graph and host ELF against an SoC memory map. |
| [`examples/handoff/captured_attention_mlp/`](../examples/handoff/captured_attention_mlp/README.md) | `host_graph_mailbox.c`, `start.S`, `link.ld`, `build_graph_mailbox.sh`. |

## Memory map

The addresses come from the pinned RTL (atlas-npu
`0079c0541111197741a231c002e3843fa6f545b2`) and the `EE290SimConfig` in
bringup-chipyard `245f2a9d203be685176cc01e26e28029c5593202`, which pins that RTL
as a submodule. [`runtime/selected-host-map.json`](../runtime/selected-host-map.json)
records them. Its `bitstream_identity` is `null`: no FPGA build has been bound.

| Region | Address | Notes |
| --- | --- | --- |
| Atlas IMEM | `0x00020000`, 128 KiB | 32-bit words; program starts at word 0. Must be mapped uncached. |
| Atlas CSR window | `0x00040000` | Host checks `0x1c` bytes; must be mapped uncached. |
| Retired-instruction count | CSR + `0x04` | Advances only while Atlas runs; start records it. |
| Status CSR | CSR + `0x08` | Bit 0 = halted. Bits 2:1 = halt reason (`0` none, `2` ECALL). |
| Execution control CSR | CSR + `0x18` | Write `1` to start. Clears the halt reason. |
| Atlas DMA buffers | Fixed per program, below 4 GiB | From the compiled execution plan or capsule layout. Examples use a 1 MiB window at `0x90000000`. |
| Host ELF (example) | `0x80000000`, 256 MiB | From `link.ld`; 16 KiB stack after `.bss`. The mailbox lives in `.bss`. |

## C API

Include `atlas_host.h` and compile `atlas_host.c`. All calls return an
`atlas_host_result`.

| Function | What it does |
| --- | --- |
| `atlas_host_init(host, map, mmio, memory)` | Stores the IMEM/CSR map, MMIO callbacks, and cache policy. Needs a `sync` hook unless `memory.coherent` is set. |
| `atlas_host_load(host, words, count)` | Writes the program into IMEM. Atlas must be halted and idle. A failed load leaves no program loaded. |
| `atlas_host_start(host, buffers, count)` | Checks each DMA buffer (nonzero size, below 4 GiB), runs the cache hook with `before_launch=1` on every buffer, then writes `1` to CSR `0x18`. The `buffers` array must stay valid until completion. |
| `atlas_host_poll(host)` | Reads status once. Returns `BUSY` while running, `OK` on ECALL halt, `BAD_HALT` on any other halt reason. |
| `atlas_host_wait(host, max_polls)` | Calls poll up to `max_polls` times. `TIMEOUT` leaves the launch active; call wait again. |
| `atlas_host_direct_read32` / `_write32` | MMIO callbacks for a directly mapped bare-metal host (volatile access plus a fence). |

Results: `ATLAS_HOST_OK`; `INVALID` (bad argument, map, or buffer); `BUSY` (a
launch is already active, or from poll: still running); `NOT_READY` (no
program, Atlas not halted, or nothing to poll); `TIMEOUT` (launch still active);
`BAD_HALT` (non-ECALL halt); `CACHE_ERROR` (the cache hook returned nonzero).

The host may prepare the next tile between start and wait, as long as it
leaves in-flight buffers alone. After a start, poll accepts an ECALL only after
it has seen Atlas running, the halt reason cleared, or the retired-instruction
count moved, so a stale ECALL from the previous launch never counts as
completion, and a launch that finishes before the first poll still completes.
The program must therefore retire at least one instruction and must not write
CSR `0xC01`.

### Cache hook

```c
int atlas_platform_cache_sync(void *context, const atlas_host_buffer *buffer,
                              int before_launch);
```

On a non-coherent host, the runtime calls this hook and treats a nonzero return
as `ATLAS_HOST_CACHE_ERROR`. Each call must finish its cache work before it
returns. A C fence is not a cache flush.

| `before_launch` | Called for | Must do |
| --- | --- | --- |
| `1` | Every buffer, before the start write | Clean (write back) host lines so Atlas DMA sees host writes, and leave no dirty lines over device-written ranges. |
| `0` | Buffers marked `DEVICE_WRITE` or `DEVICE_READ_WRITE`, after ECALL | Invalidate host lines so the host reads what Atlas wrote. |

The example hosts call this hook when built with `-DATLAS_PLATFORM_CACHE_SYNC=1`.

## Graph mailbox

`host_graph_mailbox.c` initializes the host and graph binding once, then waits
for requests in a static `atlas_graph_mailbox` (64-byte aligned). Each graph
invocation converts the f32 inputs, reloads the program into IMEM, stages inputs
and constants, starts Atlas, waits for ECALL, then reads and decodes outputs.
The build writes the mailbox address and offsets to `host/handoff-status.json`.
All fields are little-endian.

| Offset | Size | Field | Written by |
| --- | --- | --- | --- |
| 0 | 4 | `request_id` | runner |
| 4 | 4 | `response_id` | host |
| 8 | 4 | `status` | host |
| 12 | 4 | reserved | |
| 16 | 4 | `guard_before` = `0x76a31c5b` | host at startup |
| 20 | 6 × 4096 | inputs: six 32×32 f32 tiles, row-major | runner |
| 24596 | N × 4096 | outputs: N 32×32 f32 tiles, row-major (1 ≤ N ≤ 16) | host |
| 24596 + 4096·N | 4 | `guard_after` = `0x76a31c5b` | host at startup |

The mailbox is `24600 + 4096·N` bytes. For the two-output attention + MLP graph
that is 32,792 bytes with `guard_after` at 32,788. For the captured graph, the
first five inputs are weights and the sixth is the token tile. Every request
stages all six, so new weights need no recompile.

Protocol:

1. Start the ELF. Startup clears the mailbox. Wait until `guard_before` reads
   `0x76a31c5b` and `status` is 0.
2. Write all six input tiles, then a new nonzero `request_id` that differs from
   the current `response_id`.
3. The host sets `status = 0xffffffff`, runs the graph, writes `status`, then
   copies `request_id` into `response_id`.
4. When `response_id == request_id`, read `status` and the outputs. Do not
   change inputs before then.

| `status` | Meaning |
| --- | --- |
| `0x00000000` | Success; both guards intact. |
| `0xffffffff` | Request in progress. |
| `0x0001SSRR` | Tile failure: `SS` = `atlas_model_tile_status`, `RR` = `atlas_host_result`. Host stops at WFI. |
| `0x0002000R` | `atlas_host_init` failed with result `R`. Host stops. |
| `0x00030000` | Graph binding setup failed. Host stops. |
| `0x00040000` | A guard word changed during the request. |

The host spins on the mailbox; there is no interrupt. A reused request ID is
ignored and leaves the old outputs in place.

## Reset entry and boot capsule

An Atlas program has no calling convention. It starts at IMEM word 0, must set
every scalar register it uses, must wait for its own DMA transfers, and must end
with ECALL. There is no stack, argument register, or return. ECALL status does
not replace missing DMA waits.

`atlas-boot-pack` packages a program for this entry, and `atlas-host-embed`
turns the capsule into `atlas_program.h` (`atlas_program_words`,
`atlas_program_map`, and `ATLAS_PROGRAM_BITSTREAM_BOUND`, 0 for the source-only
map):

```sh
build/bin/atlas-boot-pack \
  --source out/run/compiled/atlas-machine.mlir --atlas-emit build/bin/atlas-emit \
  --layout examples/handoff/captured_attention_mlp/layout.json \
  --import-manifest out/run/compiled/import-manifest.json --out out/run/capsule
build/bin/atlas-host-embed --capsule out/run/capsule \
  --host-map runtime/selected-host-map.json --out out/run/host/atlas_program.h
```

Without `--object`, the packer runs `atlas-emit` and requires the words to fit
IMEM and end with exactly one ECALL; `--import-manifest` also checks that each
imported tile fits its DRAM region. With `--object x.o --llvm-bin DIR`, the
object's `.text` must be the checked words plus LLVM's final RET (which never
runs), in one symbol at offset 0, with no relocations or compressed words.

The capsule holds `program.bin`, `physical-program.json`, and `manifest.json`.
The layout JSON lists 32-byte-aligned, non-overlapping regions in one 1 MiB DRAM
window. Schema v1 uses `input`, `output`, and `guard` regions. Schema v2 adds a
`mailbox-pointers` call: a mailbox region holds u32 input and output pointers
that must fall in declared pools
([example](../test/examples/vpu_square_mailbox_layout.json)). The packer checks
the layout, not that the program reads those pointers. The tested program loads
them with `LW` and waits `DELAY 8` before use; that is the tested schedule, not
a general load-completion rule.

## Building a board ELF

Default builds pass `-DATLAS_DIAGNOSTIC_ALLOW_UNBOUND=1` and
`-DATLAS_DIAGNOSTIC_ASSUME_COHERENT=1`. Without those, the host sources need a
bound map and a cache hook; they always need `ATLAS_MAX_POLLS`. For a board:

1. Take the generated Chipyard `*.memmap.json` from the SoC build.
2. Copy `runtime/selected-host-map.json`, set `bitstream_identity`, correct
   `imem_base`/`csr_base` if the SoC places Atlas elsewhere, and add
   `soc_memmap_sha256` (SHA-256 of that memmap file).
3. Write a C file that defines `atlas_platform_cache_sync` for the board.
4. Write a linker script for the board's host memory. Host segments must not
   overlap IMEM, the CSRs, or any Atlas DMA buffer.
5. Pick `MAX_POLLS` (a positive integer, at most 9 digits) from a measurement.
6. Run `build_graph_mailbox.sh ... --platform HOST_MAP SOC_MEMMAP CACHE_SYNC_C
   LINKER_SCRIPT MAX_POLLS` (see the
   [example README](../examples/handoff/captured_attention_mlp/README.md)).

`atlas_graph_platform.py` runs before and after the link. It checks the map
names a bitstream and the compiled RTL revision, the memmap hash, start CSR
`0x18`, and IMEM capacity. IMEM and CSRs must be mapped read/write and uncached,
DMA buffers read/write. After the link it checks the ELF entry and load segments
and refuses overlap with Atlas regions. It writes `host/platform-map.c`
(`atlas_selected_host_map`) and two JSON check reports. For capsule-based hosts,
run `atlas-host-embed --soc-memmap MEMMAP --require-bitstream`, then
`--host-elf ELF --check-only` after linking.

## Shape-independent raw launches

`atlas_host_load_raw(host, words, word_count)` loads a reset-entry stream
directly, with no tensor shape or graph wrapper. The original program must have
exactly one ECALL at its end, no EBREAK, no writes to DBG0 or DBG1, and all
control-flow paths must reach that final ECALL after every engine and DMA
operation has completed. Loading invalidates the previous program before
checking a replacement; a failed replacement cannot accidentally launch an old
image. The loader checks the mailbox/trap restrictions but does not prove
control flow, DMA reachability or arithmetic scheduling.

The loader replaces the final ECALL in place with `NOP; CSRRS x31, DBG1, x0;
NOP; CSRRW x0, DBG0, x31; NOP; ECALL`, preserving every original instruction PC
and adding five words. x31 becomes scratch only after the original kernel's work
has completed. The leading NOP separates a preceding DELAY from the epilogue and
the final NOP separates marker publication from ECALL.

Use the normal `atlas_host_start`, `atlas_host_poll` and `atlas_host_wait` calls
after raw loading. Each successful start clears DBG0 and publishes a fresh
nonzero invocation sequence in DBG1 before the release fence and START; the
device alone writes DBG0 while executing. Poll reads DBG0 before status and
succeeds only when the exact expected sequence accompanies halted ECALL. A short
invocation may complete before its first poll without being missed, and a stale
ECALL or previous sequence cannot complete a new launch. Sequences never wrap:
after UINT32_MAX, start refuses another invocation until the caller establishes
quiescence and reinitializes the runtime.

Raw buffer descriptors accept complete `uintptr_t` ranges, including 0x100000000
on RV64; checked inclusive end arithmetic rejects wraparound. This does not
relocate device instructions or infer DMA high halves. The compiled instructions
must set the appropriate DMA base and keep each low32 transfer within its
supported address window. The legacy graph/tile wrappers retain their existing
shape and address admission rules.

The caller must supply known quiescent hardware at initialization. A prelaunch
cache synchronization failure issues no START and leaves the host inactive. A
timeout, post-launch cache synchronization error or raw illegal/EBREAK halt
retains `host->active`, buffer descriptors and device ownership; load/start
remain busy. A host STOP or scalar fault does not drain engines. Preserve
descriptors and all memory until success or externally established quiescence,
then reinitialize as part of a platform-specific recovery policy. MMIO callbacks
must perform ordered synchronous accesses; direct baremetal callbacks and the
runtime publication/acquisition points supply full fences. The runtime is
single-caller and does not provide a queue or allocator.

## Tests

| Behaviour | Test |
| --- | --- |
| Raw shape/address admission, invocation marker, missed-running pulse, retained fault ownership | [`test/raw-host-test.c`](../test/raw-host-test.c) |
| Legacy runtime calls, stale ECALL, timeout, cache hooks | [`test/host-launch-test.c`](../test/host-launch-test.c) |
| Header and memmap checks | [`test/test_host_embed.py`](../test/test_host_embed.py), [`test/test_graph_platform.py`](../test/test_graph_platform.py) |
| Capsule, LLVM object entry, mailbox pointers with two starts (RTL core) | [`test/test_boot_capsule.py`](../test/test_boot_capsule.py), [`test/test_elf_entry_reference.py`](../test/test_elf_entry_reference.py), [`test/test_mailbox_call_reference.py`](../test/test_mailbox_call_reference.py) |
| Dump checkers | [`test/test_graph_mailbox_checker.py`](../test/test_graph_mailbox_checker.py), [`test/test_mailbox_dump_check.py`](../test/test_mailbox_dump_check.py), [`test/test_model_dump_check.py`](../test/test_model_dump_check.py) |

The RTL tests drive the standalone Atlas core through the ModeLIR TileLink
driver (`add52b0a7c96d72e0079b7938e07ca5080871e82`).

## Limits

- Nothing here has run on an FPGA or on the integrated Shuttle + Atlas SoC. The
  host ELFs have been linked and inspected; device programs have run on the
  standalone Atlas core only.
- No bitstream has been bound. The address map comes from RTL and SoC sources.
  The available `AtlasRocketConfig` Verilator build cannot be used: its map has
  no Atlas IMEM or CSR ranges and its DRAM ends at `0x90000000`.
- The memmap checks prove the host can address IMEM, CSRs, and buffers. They do
  not prove Atlas DMA can reach those buffers or that the cache hook is correct.
- Dump checkers ([`tools/check_graph_mailbox_dump.py`](../tools/check_graph_mailbox_dump.py),
  [`tools/atlas_check_mailbox_dump.py`](../tools/atlas_check_mailbox_dump.py),
  [`tools/atlas_check_model_dump.py`](../tools/atlas_check_model_dump.py))
  compare reported bytes. A match reports `reported_match_unqualified`: it does
  not show which bitstream ran or that the run was fresh.
- One synchronous caller; one program in IMEM at a time; fixed DMA addresses
  chosen at compile time.

## ResNet host integration

`atlas_resnet_block_run` packs each live activation, checks its fixed scale
before launch, uses the raw invocation-marker runtime, checks DRAM guards,
decodes BF16 output and applies FP32 affine/ReLU/residual host work with
contraction disabled. Its external entry has shape 64×56×56. A failure retains
live workspace and, for device faults or timeouts, runtime ownership.

For the supported complete-graph build route, see [Complete ResNet50 host
graph](resnet-full-host.md). The graph builder enables `mstatus.FS`, initializes
`fcsr` to nearest-even, reserves a 64 KiB stack and checks ELF assets and load
segments. A linked ELF does not establish host execution, DMA reachability or
cache coherence on a selected SoC.
