/* Integrated host loader for the bounded EE290SimConfig VLS witness.
 * atlas_program.inc is generated from the supplied final MLIR by atlas-emit.
 * Atlas RTL and existing baremetal infrastructure are read-only references.
 */
#include <stdint.h>
#include <stdio.h>
#include "atlas_program.inc"

#define IMEM ((uintptr_t)0x20000)
#define CSR ((uintptr_t)0x40000)
#define VMEM ((uintptr_t)0x20000000)
#define WORDS 1536U
#define INPUT 256U
#define OUTPUT 768U
#define TILE_WORDS 256U

static uint32_t read32(uintptr_t addr) { return *(volatile uint32_t *)addr; }
static void write32(uintptr_t addr, uint32_t value) { *(volatile uint32_t *)addr = value; }
static void fence(void) { __asm__ volatile ("fence iorw, iorw" ::: "memory"); }

static uint32_t pattern(uint32_t index, uint32_t panel) {
  uint32_t result = 0;
  for (uint32_t byte = 0; byte < 4; ++byte) {
    uint32_t n = 4 * index + byte;
    uint32_t v = panel == 0 ? n : panel == 1 ? 37 * n + 19 :
        ((n & 1) ? 0x80U : 0U) ^ (n / 32);
    result |= (v & 255U) << (8 * byte);
  }
  return result;
}

static uint32_t initial(uint32_t index, uint32_t panel) {
  return index >= INPUT && index < INPUT + TILE_WORDS ?
      pattern(index - INPUT, panel) : 0xa5a55a5aU ^ (index * 0x01010101U);
}

int main(void) {
  write32(CSR + 0x18, 0);
  fence();
  if (!(read32(CSR + 0x08) & 1)) {
    puts("EE290_VLS_FAILED initial halt");
    return 1;
  }
  for (uint32_t i = 0; i < ATLAS_PROGRAM_WORDS; ++i)
    write32(IMEM + 4 * i, atlas_program[i]);
  fence();
  for (uint32_t i = 0; i < ATLAS_PROGRAM_WORDS; ++i) {
    if (read32(IMEM + 4 * i) != atlas_program[i]) {
      printf("EE290_VLS_FAILED instruction readback %u\n", i);
      return 1;
    }
  }
  for (uint32_t panel = 0; panel < 3; ++panel) {
    for (uint32_t i = 0; i < WORDS; ++i) write32(VMEM + 4 * i, initial(i, panel));
    write32(CSR + 0x10, 0);
    write32(CSR + 0x14, 0);
    write32(CSR + 0x00, 0);
    write32(CSR + 0x04, 0);
    fence();
    write32(CSR + 0x18, 1);
    fence();
    uint32_t status = 0, marker = 0, poll = 0;
    for (; poll < 100000U; ++poll) {
      status = read32(CSR + 0x08);
      marker = read32(CSR + 0x10);
      /* Halted reflects halt_now combinationally; its registered reason can
       * settle one cycle later. START cleared that reason to zero, so wait
       * for a nonzero reason before judging ECALL versus illegal/EBREAK.
       * A halted-without-reason state remains bounded by the poll limit.
       */
      if ((status & 1U) && ((status >> 1) & 3U)) break;
    }
    fence();
    uint32_t illegal = read32(CSR + 0x0c);
    uint32_t cycles = read32(CSR + 0x00);
    uint32_t retired = read32(CSR + 0x04);
    printf("EE290_VLS_OBSERVED panel=%u status=%u marker=%u illegal_pc=%u observer_cycles=%u retired=%u polls=%u\n",
           panel, status, marker, illegal, cycles, retired, poll);
    if (status != 5U || marker != 1U || poll == 100000U) {
      puts("EE290_VLS_FAILED completion");
      return 1;
    }
    uint32_t failures = 0;
    for (uint32_t i = 0; i < WORDS; ++i) {
      uint32_t expected = i >= OUTPUT && i < OUTPUT + TILE_WORDS ?
          pattern(i - OUTPUT, panel) : initial(i, panel);
      uint32_t actual = read32(VMEM + 4 * i);
      if (actual != expected) {
        if (failures < 8)
          printf("EE290_VLS_MISMATCH panel=%u word=%u expected=%08x actual=%08x\n",
                 panel, i, expected, actual);
        ++failures;
      }
    }
    if (failures) {
      printf("EE290_VLS_FAILED memory %u\n", failures);
      return 1;
    }
    printf("EE290_VLS_PANEL_PASSED panel=%u output_words=256 preserved_words=1280\n", panel);
  }
  puts("EE290_VLS_PASSED panels=3");
  return 0;
}
