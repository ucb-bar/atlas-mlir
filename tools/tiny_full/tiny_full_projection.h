#ifndef TINY_FULL_PROJECTION_H
#define TINY_FULL_PROJECTION_H
#include "tiny_projection.h"

typedef struct {
  tiny_projection_context panels;
  size_t calls, step_count;
} tiny_full_projection_context;

int tiny_full_project(void *, unsigned layer, unsigned logical,
                      const uint16_t *, size_t rows, size_t k, size_t width, uint16_t *);
#endif
