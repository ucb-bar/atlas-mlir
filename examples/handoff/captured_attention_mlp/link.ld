ENTRY(_start)
MEMORY { DRAM (rwx) : ORIGIN = 0x80000000, LENGTH = 256M }
SECTIONS {
  .text : { *(.text.init) *(.text .text.*) } > DRAM
  .rodata : { *(.rodata .rodata.*) } > DRAM
  .data : { *(.data .data.*) } > DRAM
  .bss (NOLOAD) : {
    . = ALIGN(8);
    _bss_start = .;
    *(.bss .bss.* COMMON)
    . = ALIGN(8);
    _bss_end = .;
  } > DRAM
  . = ALIGN(16);
  . = . + 0x4000;
  _stack_top = .;
}
