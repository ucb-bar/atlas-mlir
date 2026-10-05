/* Selected Atlas MMIO fields follow atlas-npu baremetal/assembler.py and
 * Chipyard's EE290SimConfig. This is a local attributed binding, not ACT code. */
#include "atlas_host.h"

#define ATLAS_IMEM_BASE 0x00020000UL
#define ATLAS_CSR_BASE 0x00040000UL
#define CSR_CYCLE (ATLAS_CSR_BASE + 0x00UL)
#define CSR_INSTCNT (ATLAS_CSR_BASE + 0x04UL)
#define CSR_STATUS (ATLAS_CSR_BASE + 0x08UL)
#define CSR_ILLEGAL_PC (ATLAS_CSR_BASE + 0x0CUL)
#define CSR_DBG0 (ATLAS_CSR_BASE + 0x10UL)
#define CSR_DBG1 (ATLAS_CSR_BASE + 0x14UL)
#define CSR_EXEC_CONTROL (ATLAS_CSR_BASE + 0x18UL)

static uint32_t read32(uintptr_t address) {
  return *(volatile uint32_t *)address;
}

static void write32(uintptr_t address, uint32_t value) {
  *(volatile uint32_t *)address = value;
}

static void memory_fence(void) { __asm__ volatile("fence" ::: "memory"); }

atlas_ee290_status atlas_ee290_run(const uint32_t *program, size_t words,
                                    uint32_t poll_limit,
                                    atlas_ee290_result *result) {
  if (program == NULL || result == NULL || words == 0 ||
      words > ATLAS_EE290_IMEM_WORDS || poll_limit == 0) {
    return ATLAS_EE290_INVALID_ARGUMENT;
  }
  *result = (atlas_ee290_result){0};

  write32(CSR_EXEC_CONTROL, 0u);
  write32(CSR_DBG0, 0u);
  write32(CSR_DBG1, 0u);
  memory_fence();

  for (size_t i = 0; i < words; ++i) {
    write32(ATLAS_IMEM_BASE + 4u * i, program[i]);
  }
  memory_fence();
  for (size_t i = 0; i < words; ++i) {
    if (read32(ATLAS_IMEM_BASE + 4u * i) != program[i]) {
      return ATLAS_EE290_IMEM_MISMATCH;
    }
  }

  write32(CSR_EXEC_CONTROL, 1u);
  memory_fence();
  for (uint32_t i = 0; i < poll_limit; ++i) {
    result->completion_marker = read32(CSR_DBG0);
    result->polls = i + 1u;
    if (result->completion_marker != 0u) {
      break;
    }
  }
  write32(CSR_EXEC_CONTROL, 0u);
  memory_fence();

  result->cycles = read32(CSR_CYCLE);
  result->instructions = read32(CSR_INSTCNT);
  result->status = read32(CSR_STATUS);
  result->illegal_pc = read32(CSR_ILLEGAL_PC);
  return result->completion_marker ? ATLAS_EE290_OK
                                   : ATLAS_EE290_WAIT_EXHAUSTED;
}
