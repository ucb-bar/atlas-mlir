# Bounded VLOAD/VSTORE timing and compiler mapping

This slice relates selected LSU hardware events to the current `AtlasTiming` API. The [admission audit](admission.md) covers the other engines and shared obligations; the [operation catalog](operation-catalog.json) remains the full instruction denominator. This document does not enable a contract provider or establish integrated EE290SimConfig execution.

## Event origin and applicable domain

Age zero is the pre-edge scalar `s1_fire && decoded.is_lsu` command presentation, with operands captured at that rising edge. AtlasCore wires this valid directly to LSU; no command register intervenes. Age n describes a request sampled at the nth subsequent rising edge. A write sampled at edge n becomes visible after that edge; reading it at a later edge avoids relying on unspecified same-edge read/write semantics. Scalar-memory commands take a different registered frontend path and are outside this initial replay.

The bounded domain requires reset deasserted, a legal idle-path command, no host restart or unrelated scalar/engine/DMA/host memory traffic, and correct one-cycle synchronous memory responses. A tile consists of 32 rows of 32 bytes. Its effective 16-bit VMEM line address must be a multiple of 32 in `0..49120`, entirely within one of six 8,192-line banks. MREG IDs are `0..63`. Initial overlap cases use distinct logical MREGs and physical banks; same-register dependencies are checked separately, and same-register rolling overwrite remains excluded.

The effective address is `L = (((B + 32 * sext12(I)) mod 2^32) >> 3) & 0xffff`, for scalar word base B and encoded immediate I. The touched byte range is `[32*L, 32*(L+32))`. Record this masking and truncation rather than treating the base as bytes or rejecting/accepting an address before the hardware transform. Unknown addresses cannot establish this bounded domain.

VMEM and MREG SRAM bodies are opaque in the retained IR. Their declared one-cycle read latency and the surrounding response tags motivate the replay environment; that environment is an explicit assumption, not a test of the SRAM implementations. A delayed or missing response is not an elastic stall supported by these LSU streams.

## Derived streams and releases

For both paths, an idle command enters Run, issues rows 0 through 31 on successive cycles, enters Drain for one cycle, and reaches idle state while the final write is still pending. The busy output includes pending stages, so testing the FSM's idle state alone is insufficient for legal launch.

| Event or resource | VLOAD | VSTORE |
| --- | --- | --- |
| Command capture | Edge 0 | Edge 0 |
| Source requests, row i | VMEM read at `1+i` | MREG read at `1+i` |
| Assumed source response | `2+i` | `2+i` |
| Destination requests, row i | MREG write at `3+i` | VMEM write at `3+i` |
| Actual source-port interval | `[1,32]` | `[1,32]` |
| Actual destination-port interval | `[3,34]` | `[3,34]` |
| Busy / active-register report | Ages 1 through 34 | Ages 1 through 34 |
| Last destination write | Edge 34 | Edge 34 |
| First legal subsequent same-path command | Edge 35 | Edge 35 |

For load, `responsePending[n+1]=readRequest[n]` and `writePending[n+1]=responseValid[n] && responsePending[n]`; row/data capture follows that condition. For store, request/address tags pass through two stages, incoming MREG response data is registered, and the VMEM write requires both the delayed response-valid and request tag. These state equations, together with the explicit memory response assumption, establish the streams. The store's last physical source read at edge 32 is distinct from its active-source report and conservative full-register release through age 34.

VMEM has one shared read/write port per bank. A same-bank scalar/vector conflict asserts instead of stalling these streams. Lower-priority DMA/host traffic has separate grants; proving their interaction requires a broader replay. MREG has separate read and write ports per physical bank `id & 31`; registers m and m+32 share ports but store different rows. Frontend 64-bit logical busy masks do not replace physical-port checks.

## Mapping to the existing compiler

The [timing probe](../../test/vls-timing-probe.cpp) compiles the actual [AtlasTiming.cpp](../../lib/AtlasTiming.cpp) and reports its footprints, dependencies and reservation decisions. It contains no asserted expected timing constants. Compare that output with selected hardware observations using the replay below.

| Compiler representation | Mapping / limitation |
| --- | --- |
| `Access {first, count, age, step}` | The two 32-row streams fit directly: source age 1, destination age 3, step 1. VMEM indices are lines and MREG indices are `reg*32+row`. |
| `VloadPath` / `VstorePath` holds | Existing inclusive `[0,34]` also reserves command capture, conservatively covering busy ages 1–34; the next same-path issue is 35. |
| `VmemBank` hold | Load `[1,32]`; store `[3,34]`. These encode physical bank occupancy separately from logical address overlap. |
| `writeRelease=34` | Full load destination is available after the final write; a dependent command at edge 35 also satisfies the frontend logical busy assertion. |
| Store `readRelease=34` | Conservative active-source lifetime; actual last row read is edge 32. Do not shorten it solely from the read stream. |
| `writeDuringRead=true` on a later VLOAD | Existing code permits some row-wise store-to-load overwrite. This requires separate alias/forwarding evidence and is outside the initial passing domain. |
| `dependence` plus `ReservationTable` | Both are needed: data hazards and resource conflicts are different constraints. Neither alone enforces a single scalar issue per cycle or all frontend assertions. |
| Unknown operands | Existing code can return conservative footprints without an error. That does not satisfy a bound profile's required address/admission evidence. |

The current VLS constants agree with the static derivation, so changing their numeric values is not justified by this audit. The missing integration is explicit evidence selection, applicability/admission enforcement and a shared final timed verifier. `verify-atlas-machine-stream` checks structural/encoding requirements; the separate generated-schedule verifier checks its diagnostic-delay policy. Neither is a complete consumer of these selected-RTL timing rules.

## Boundary checks and reproduction

All gaps below are between command/issue edges, not inserted DELAY counts. Additional logical-data hazards can increase them.

| Ordered pair and scope | Rejected boundary | First permitted boundary |
| --- | --- | --- |
| Load/load or store/store | 34: path remains busy | 35 |
| Load/store, same VMEM bank, disjoint ranges and MREGs | 29: one shared-bank access edge overlaps | 30 |
| Store/load, same VMEM bank, disjoint ranges and MREGs | 33: one shared-bank access edge overlaps | 34 |
| Load/store, same logical MREG, different VMEM banks | 34: frontend RAW obligation | 35 |
| Load/store or store/load, distinct VMEM banks and MREGs | 0: one frontend cannot issue both | 1 |

Run the [replay/check driver](../../tools/check-vls-timing.py) from the compiler checkout with explicit inputs and tools. It creates a new artifact directory, retains the selected LSU and macro fragments, exports that slice through CIRCT with assertions preserved, builds a small Verilator model and an isolated compiler timing probe, and saves commands, source/tool hashes, traces and comparisons. It does not rebuild Chipyard or use an existing unbound simulator.

```bash
python3 tools/check-vls-timing.py \
  --manifest build/rtl-timing/ee290-run-001/manifest.json \
  --circt-opt /path/to/circt-opt \
  --verilator /path/to/verilator \
  --cxx /path/to/c++ \
  --make /path/to/make \
  --llvm-source-include /path/to/llvm-project/llvm/include \
  --llvm-build-include /path/to/llvm-build/include \
  --output build/rtl-timing/vls-check-001
```

For relocated tool installations, `--verilator-root` overrides the inferred Verilator package root and `--ar` selects the archive tool. The receipt records the selected and executed tools and these overrides. Generated outputs remain under ignored `build/rtl-timing/`.

The C++ environment supplies one-cycle responses and checks all memory contents against independent whole-tile copies, including untouched locations. It observes per-row addresses and busy signals. The harness reports frontend busy/logical-MREG obligations separately from assertions executed inside the selected LSU. A two-cycle-response negative case tests the materiality of the environment assumption; it does not qualify variable-latency operation. A report must identify which outcomes actually occurred and reject unexpected crashes, missing expected assertions or incorrect outputs.

The suite has 20 hardware replay cases, including 16 paired-command cases. The compiler probe separately reports 22 footprint/domain instances and 1,332 API gap decisions; that larger API matrix is not 1,332 RTL replays. The driver checks exact normalized access streams, release/hold coverage and the replayed pair admissions. Hash and size mismatch checks must reject the input before output creation. A passing report is conditional on the documented environment and keeps rule enablement and integrated qualification false.

The evidence-consumer regressions require no compiler build. They test missing or malformed streams, premature release, insufficient reservations, altered observations and input identity mismatches:

```bash
python3 test/test_vls_timing_evidence.py
```

This validates a conditional module slice and its compiler mapping. It leaves other-engine arbitration, physical alias cases, scalar-memory overlap, selected SRAM implementation behavior, complete frontend execution, reset/restart, contract loading, and integrated emitted-program output/guard validation open.
