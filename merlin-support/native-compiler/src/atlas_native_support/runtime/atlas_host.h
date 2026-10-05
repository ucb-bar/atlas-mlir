#ifndef ATLAS_EE290_HOST_H
#define ATLAS_EE290_HOST_H

#include <stddef.h>
#include <stdint.h>

/* Selected EE290SimConfig MMIO binding for the Atlas tile. */
#define ATLAS_EE290_IMEM_WORDS 32768u

typedef enum {
  ATLAS_EE290_OK = 0,
  ATLAS_EE290_INVALID_ARGUMENT = 1,
  ATLAS_EE290_IMEM_MISMATCH = 2,
  ATLAS_EE290_WAIT_EXHAUSTED = 3,
} atlas_ee290_status;

typedef struct {
  uint32_t completion_marker;
  uint32_t cycles;
  uint32_t instructions;
  uint32_t status;
  uint32_t illegal_pc;
  uint32_t polls;
} atlas_ee290_result;

/*
 * Execute one already encoded Atlas program at IMEM word zero.
 * The caller owns DRAM input/output placement, cache visibility, and the
 * program's ABI. A nonzero completion marker is the selected diagnostic
 * protocol; it is not a general interrupt or error classification.
 */
atlas_ee290_status atlas_ee290_run(const uint32_t *program, size_t words,
                                    uint32_t poll_limit,
                                    atlas_ee290_result *result);

#endif
