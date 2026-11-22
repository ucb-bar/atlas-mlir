Object sizing before final ELF linking
=====================================

Run from the atlas-mlir checkout with `PYTHONPATH=python`:

```sh
python3 -m atlas_mlir_native.elf_build compile generated.c host.c \
  --cc /path/to/riscv64-unknown-elf-gcc -I generated -o fresh-objects
python3 -m atlas_mlir_native.elf_build measure fresh-objects/000-generated.c.o \
  -o sections.json
python3 -m unittest atlas_mlir_native.elf_build.test_sections
```

`compile` uses the bench object flags, writes individual logs, and records
commands, source hashes, and actual allocated section sizes and alignments.
It defaults to 100 MHz and passes integer SYS_CLK_FREQ and MTIME_FREQ
definitions. `--clock-mhz` affects those definitions only at this stage.
Assembly `.incbin` search uses `--assembler-root`. Output directories must be
fresh. Paths through symlinks are refused.

Section sizes include NOBITS and exclude file-only metadata. Their sum excludes
linker padding, archive extraction, garbage collection, and platform runtime.
It is not a final memory footprint. B must consume individual sections and
alignments. Frozen ELF measurements are observations, not a rebuilt replay.

This module deliberately stops at objects. Final Baremetal linking still needs
B's binding and E's authenticated positive EXACT and executed output mutation.
No DDR ELF, checker binding, host-math qualification, or hardware acceptance is
claimed by object compilation.
