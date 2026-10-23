# Model ELF handoff to FPGA and FireSim

`tools/atlas_model_handoff.py` copies an unchanged bare-metal ELF and creates a FireSim baremetal workload description and a memory-footprint manifest. It does not execute a model, install a workload, invoke a hardware build or qualify a target. Its manifest is always `PREPARATION_ONLY`; numerical evidence remains the separate live QEMU/model or SoC receipt.

The input ELF must define distinct, aligned, writable eight-byte `tohost` and `fromhost` symbols. FESVR needs both symbols to communicate with the target. The existing host reports successful completion with `tohost=1`; a nonzero exit code uses `(exit_code << 1) | 1`. Declared observation symbols must be writable loaded memory and disjoint from the DMA arena. The arena must avoid executable host memory, IMEM and CSRs. Programs and data are not relocated or changed.

The diagnostic interface is IMEM at `0x20000` for 131072 bytes, CSR at `0x40000` for 28 bytes and a model-specific DMA arena at `0x100000000`. High DMA addresses use the raw runtime's existing 64-bit host pointers and the compiled program's 32-bit DMA window. Each arena must fit a single such window. Every ELF `PT_LOAD` has separate file and memory sizes in the manifest: the loader must copy file bytes and zero the complete remaining memory extent, including NOLOAD DMA segments and BSS. Stack storage is included when the linker reserves it in a loaded RAM segment.

Generate a package from a completed model host build, with all output paths outside this checkout:

```sh
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$ATLAS_SESSION/readiness"
python3 -B tools/atlas_model_handoff.py \
  --elf "$MODEL_ELF" --name "$MODEL_NAME" \
  --dma-base 0x100000000 --dma-bytes "$MODEL_DMA_BYTES" \
  --observe-symbol "$MODEL_MAILBOX" \
  --out "$ATLAS_SESSION/readiness/packages/$MODEL_NAME"
```

The output directory must not exist. The package contains `host.elf`, `handoff.json` and `workload.json`. The manifest records the ELF hash, entry point, each file/zero-fill range, HTIF and observation addresses, and the union of resident host/DMA bytes without counting an overlapping NOLOAD arena twice. It does not include unrelated external capture files because these model ELFs embed their required runtime assets.

For a selected prebuilt FPGA or FireSim target, rerun packaging to a fresh directory with `--soc-memmap "$TARGET_MEMMAP" --target-identity "$TARGET_ID"`. The map is the actual generated Chipyard memory map, not a hand-written approximation. Existing memory-map helpers check every loaded byte with the ELF's required permissions, uncached read/write IMEM and CSR access, and host-visible read/write coverage of the declared DMA arena. Accelerator DMA connectivity needs a separate target binding. The exact map hash and supplied target identity are recorded, and the original map bytes are copied. A supplied identity is a label, not independent proof that it belongs to the map or hardware.

Before a hardware run, bind that map to the selected bitstream or AGFI, driver and Atlas RTL revision. Confirm the host cache policy and Atlas DMA coherence on that target. The existing diagnostic hosts assume their qualified simulation memory behavior; a fence alone does not provide cache maintenance. If a target requires cache hooks or a different address map, rebuild and independently review the platform interface while preserving the admitted program and payload bytes. No such platform change or hardware qualification is implied by a package.

## UART TSI

With the matching prebuilt UART TSI executable, target and serial device selected, run the existing loader directly:

```sh
"$UART_TSI" +tty="$UART_DEVICE" +baudrate=921600 +selfcheck \
  "$MODEL_PACKAGE/host.elf"
```

The 921600 baud value comes from `EE290BringupHostConfig`; select the actual target's baud if it differs. `+selfcheck` reads back the ELF load chunks byte by byte. This loader uses TSI/FESVR ELF loading and HTIF exit; it does not support the simulator's `+loadmem` bypass. A zero loader exit requires target HTIF completion, but does not by itself establish an external bit-exact comparison beyond the checks embedded in the ELF. Preserve the loader output and the model mailbox/results when collecting hardware evidence.

## FireSim

Stage the package into an existing matching FireSim deployment when that hardware run is authorized. These commands are instructions for that later deployment; the packager does not perform them:

```sh
install -D "$MODEL_PACKAGE/host.elf" \
  "$FIRESIM_DEPLOY/workloads/$MODEL_NAME/host.elf"
install -D "$MODEL_PACKAGE/workload.json" \
  "$FIRESIM_DEPLOY/workloads/$MODEL_NAME.json"
cd "$FIRESIM_DEPLOY"
./firesim runworkload -c "$RUNTIME_YAML" -a "$HWDB_YAML" -r "$BUILD_RECIPES_YAML"
```

Set the runtime YAML's `workload.workload_name` to `$MODEL_NAME.json`, select the matching Atlas-capable `target_config.default_hw_config`, and retain `workload.terminate_on_completion: true`. The generated JSON explicitly sets `common_rootfs: null`, because omitting the field can select a derived filesystem image. FireSim resolves the boot binary beneath `workloads/<benchmark_name>/`, generates the driver `+prog0` argument and collects `uartlog`. Its prebuilt driver, bitstream and named memory-region bridge must match the target map, including the high DMA arena. A generic FireSim configuration name does not establish an Atlas target.

## Readiness checklist

- Preserve the original ELF and numerical receipts, and compare the packaged ELF SHA-256 with them.
- Confirm the complete load/zero-fill ranges and DMA arena fit the actual target memory map and loader address width.
- Confirm distinct writable `tohost`/`fromhost`, the model mailbox, completion exit encoding and stack/BSS storage.
- Bind the generated map to the actual prebuilt Atlas target, driver and bitstream or AGFI; qualify cache/DMA behavior.
- Use the existing loader commands above, then retain target completion and every embedded output/preserved-byte check before claiming a hardware pass.

Full TinyLlama and SmolVLA require their full-model generated ELF and actual final footprint. A selected-block package does not qualify a full model. No kernel or product-host performance optimization is part of this handoff.
