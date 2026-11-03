# Full TinyLlama attention host build

`tools/tiny_full/build_attention.py` is the versioned penguin26 attention build adapter. It accepts `atlas.tiny_attention_contract.v1` / `tiny-attention` assets directly. The projection-only `build_host.py`, `KernelAssets`, and `FullBundle` formats retain their existing meaning. See [the export command](tiny-attention-export.md) for capture-to-assets preparation.

The adapter requires the complete 22-layer prefill and two-decode bundle, its checksum-pinned original RTL qualification receipt, a Merlin comparisons source directory, and an explicit RISC-V GNU toolchain bin directory. It checks 906 ordered launches, 840 projection panels, 66 attention panels, the interleaved launch plan, every initial/output/final arena, the receipt's original-model and DMA bindings, and all 137 final cache, logits, and token buffers. `--out` must name a fresh directory. It does not execute the ELF or rerun numerical qualification.

From a fresh checkout, use physical absolute paths for the exported assets, receipt, Merlin source, and toolchain. The build reads the explicitly supplied platform tree, model receipt inputs, source directories, and toolchain.

```sh
python3 -B tools/tiny_full/build_attention.py \
  --assets /absolute/path/to/exported-attention \
  --qualification /absolute/path/to/qualified-receipt.json \
  --qualification-sha256 RECEIPT_SHA256 \
  --merlin /absolute/path/to/merlin/examples/atlas/comparisons \
  --platform-root "$PLATFORM_ROOT" --toolchain-bin /absolute/path/to/riscv-tools/bin \
  --variant sims --out /absolute/path/to/new-sims-build
```

Repeat with `--variant chip` and a different `--out`. The ELF is `OUT/build/atlas-tinyllama.elf`; `OUT/build.json` binds its SHA-256, all compiled source/header bookends, the toolchain libraries, the commands, the assets, the qualification receipt, and the two LOAD ranges. Configure, dependency, and build logs remain in `OUT`. Compilation needs Python with NumPy and the exporter dependencies, CMake 3.19 or later, a host Make, and the selected GNU RISC-V toolchain. Export's model execution dependencies are unnecessary during the host build.

Pass `--platform-root` explicitly to select the board-support tree for the target platform (startup code, drivers, and the penguin26 platform files). The tree supplies the TinyLlama CMake target, linker script, and scalar math support, and is kept outside this repository. The model and projection arithmetic remain in their existing tracked source files; Merlin arithmetic comes from the explicit `--merlin` directory. The unchanged qualified host main is `tools/tiny_full/tinyllama_main.c`. It initializes attention panels/count/cursor and transport, enforces the interleaved plan at each launch, invokes the raw loader and coherent DMA transport, checks output bytes and preservation across the complete arena, and checks final cache/logits/tokens after every step. SIMS uses HTIF; CHIP uses UART at 3686400 baud and tohost completion.

The adapter retains the original source basenames, source and library link order, `rv64gcv_zfh` / `lp64d` / `medany`, Release `-O3 -DNDEBUG`, platform `-O1`, and model `-O2 -ffp-contract=off -fno-fast-math -fno-builtin-expf -fno-builtin-nearbyintf -fno-builtin-scalbnf -fno-builtin-pow`. It does not add debug flags. Existing toolchain library debug sections are preserved. Source paths and assembly `.incbin` paths may move without changing the ELF, with the same source bytes and toolchain. `--expected-elf-sha256` enforces a desired byte identity.

For the reviewed M3 bundle and GNU 13.2.0 toolchain, the complete receipt SHA-256 is `b8348d8186df818b4493925b7f60590611b76a3a56ff6b97cb59b19b39e9eb18`. The expected SIMS ELF SHA-256 is `22d7cd325cd8d2c3aa0db6135273527b2108cc523405ee4f1882b0321bb4328a`, and the expected CHIP ELF SHA-256 is `e7067407cd2a74688791e0f48f88c65309eff0099f9f0461d733bbda883ad3f3`. These identities apply to that exact source/assets/toolchain combination. The explicit export placement must cover both LOAD ranges and the DMA arena; the adapter checks the resulting ELF against that placement. This build is not evidence of full-model F2 execution or measured device performance.

Run focused admission regressions with `python3 -B -m unittest discover -s test -p test_tiny_attention_build.py -v`.
