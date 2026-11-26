#include "atlas_host.h"
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <assert.h>
#include <stdint.h>
#include <stdio.h>

typedef struct {
  uint32_t words[32], status, retired, marker, sequence;
  unsigned writes, starts, cache_after;
  int cache_fails;
} device;

static uint32_t read32(void *context, uintptr_t address) {
  device *d = context;
  if (address == ATLAS_CSR_BASE + ATLAS_CSR_STATUS) return d->status;
  if (address == ATLAS_CSR_BASE + ATLAS_CSR_RETIRED) return d->retired;
  assert(address == ATLAS_CSR_BASE + ATLAS_CSR_DBG0);
  return d->marker;
}

static void write32(void *context, uintptr_t address, uint32_t value) {
  device *d = context;
  if (address == ATLAS_CSR_BASE + ATLAS_CSR_DBG0) {
    d->marker = value;
  } else if (address == ATLAS_CSR_BASE + ATLAS_CSR_DBG1) {
    d->sequence = value;
  } else if (address == ATLAS_CSR_BASE + ATLAS_CSR_EXEC_CONTROL) {
    assert(value == 1);
    ++d->starts;
  } else {
    assert(address >= ATLAS_IMEM_BASE && address < ATLAS_IMEM_BASE + sizeof d->words);
    assert(!(address & 3));
    d->words[(address - ATLAS_IMEM_BASE) / 4] = value;
    ++d->writes;
  }
}

static int sync_buffer(void *context, const atlas_host_buffer *buffer, int before) {
  device *d = context;
  assert(buffer->size_bytes);
  if (!before) ++d->cache_after;
  return d->cache_fails;
}

static uint64_t tick(void *context) {
  return ++*(uint64_t *)context;
}

static const uint32_t program[] = {0x13, 0x73};
static const uint32_t identical[] = {0x13, 0x73};

/* Every idle-only API must preserve ownership of the same descriptors after
 * timeout, post-launch cache failure, and an observed raw scalar fault. */
static void retained(atlas_host *h, const atlas_host_buffer *buffer, device *d) {
  unsigned writes = d->writes, starts = d->starts;
  uint64_t loads = h->image_loads, reuses = h->image_reuses;
  uint32_t sequence = h->invocation_sequence;
  assert(h->active && h->raw_completion);
  assert(h->active_buffers == buffer && h->active_buffer_count == 1);
  assert(atlas_host_load(h, program, 2) == ATLAS_HOST_BUSY);
  assert(atlas_host_ensure_loaded(h, identical, 2) == ATLAS_HOST_BUSY);
  assert(atlas_host_load_raw(h, program, 2) == ATLAS_HOST_BUSY);
  assert(atlas_host_start(h, buffer, 1) == ATLAS_HOST_BUSY);
  assert(atlas_host_invalidate_program(h) == ATLAS_HOST_BUSY);
  assert(atlas_host_set_clock(h, 0, 0) == ATLAS_HOST_BUSY);
  assert(d->writes == writes && d->starts == starts);
  assert(h->image_loads == loads && h->image_reuses == reuses);
  assert(h->invocation_sequence == sequence && h->program_loaded);
  assert(h->active && h->raw_completion && !h->resident_words);
  assert(h->active_buffers == buffer && h->active_buffer_count == 1);
  assert(h->clock == tick);
}

static void reset_quiescent(atlas_host *h, device *d) {
  /* This fake has no engines; establish its idle state before notification. */
  d->status = 5;
  d->marker = 0;
  uint64_t loads = h->image_loads, launches = h->launches;
  uint32_t sequence = h->invocation_sequence;
  assert(atlas_host_reset_completed(h) == ATLAS_HOST_OK);
  assert(!h->active && !h->program_loaded && !h->resident_words);
  assert(!h->raw_completion && !h->active_buffers && !h->active_buffer_count);
  assert(h->image_loads == loads && h->launches == launches);
  assert(h->invocation_sequence == sequence && h->clock == tick);
}

int main(void) {
  device d = {.status = 5};
  atlas_host h;
  uint64_t ticks = 0;
  atlas_host_buffer buffer = {0x90000000, 32, ATLAS_HOST_DEVICE_WRITE};
  assert(atlas_host_init(&h,
      (atlas_host_map){ATLAS_IMEM_BASE, ATLAS_CSR_BASE, sizeof d.words},
      (atlas_host_mmio){read32, write32, &d},
      (atlas_host_memory){0, sync_buffer, &d}) == ATLAS_HOST_OK);
  assert(atlas_host_set_clock(&h, tick, &ticks) == ATLAS_HOST_OK);

  assert(atlas_host_ensure_loaded(&h, program, 2) == ATLAS_HOST_OK);
  assert(atlas_host_ensure_loaded(&h, identical, 2) == ATLAS_HOST_OK);
  assert(h.image_loads == 1 && h.image_reuses == 1 && d.writes == 2);
  assert(h.timings.load_ticks == 1);
  assert(atlas_host_load_raw(&h, program, 2) == ATLAS_HOST_OK);
  assert(!h.resident_words && !h.resident_word_count && h.raw_completion);
  assert(h.image_loads == 2 && h.imem_words_written == 9 && d.writes == 9);
  assert(h.timings.load_ticks == 2);
  /* Identical source bytes must restore the original legacy stream, since raw
   * loading replaced its final ECALL and extended the program by five words. */
  assert(atlas_host_ensure_loaded(&h, identical, 2) == ATLAS_HOST_OK);
  assert(!h.raw_completion && d.words[1] == 0x73);
  assert(h.image_loads == 3 && h.image_reuses == 1 && d.writes == 11);
  buffer.access = (atlas_host_access)-1;
  assert(atlas_host_start(&h, &buffer, 1) == ATLAS_HOST_INVALID);
  buffer.access = ATLAS_HOST_DEVICE_WRITE;
  assert(atlas_host_load_raw(&h, program, 0) == ATLAS_HOST_INVALID);
  assert(!h.program_loaded && !h.resident_words && !h.raw_completion);
  assert(h.image_loads == 3 && h.imem_words_written == 11);

  assert(atlas_host_load_raw(&h, program, 2) == ATLAS_HOST_OK);
  uint64_t start_ticks = h.timings.start_ticks;
  assert(atlas_host_start(&h, &buffer, 1) == ATLAS_HOST_OK);
  assert(h.timings.start_ticks == start_ticks + 1);
  uint64_t wait_ticks = h.timings.wait_ticks;
  assert(atlas_host_wait(&h, 2) == ATLAS_HOST_TIMEOUT);
  assert(h.timings.wait_ticks == wait_ticks + 1);
  retained(&h, &buffer, &d);
  d.status = 0;
  assert(atlas_host_poll(&h) == ATLAS_HOST_BUSY);
  d.status = 3;
  assert(atlas_host_wait(&h, 1) == ATLAS_HOST_BAD_HALT);
  assert(d.cache_after == 0);
  retained(&h, &buffer, &d);
  reset_quiescent(&h, &d);

  assert(atlas_host_load_raw(&h, program, 2) == ATLAS_HOST_OK);
  assert(atlas_host_start(&h, &buffer, 1) == ATLAS_HOST_OK);
  assert(d.sequence == 2 && h.launches == 2);
  d.marker = d.sequence;
  d.cache_fails = 1;
  assert(atlas_host_wait(&h, 1) == ATLAS_HOST_CACHE_ERROR);
  retained(&h, &buffer, &d);
  d.cache_fails = 0;
  assert(atlas_host_wait(&h, 1) == ATLAS_HOST_OK);
  assert(!h.active && d.cache_after == 2);
  assert(h.image_loads == 5 && h.imem_words_written == 25 && d.writes == 25);
  assert(atlas_host_invalidate_program(&h) == ATLAS_HOST_OK);
  assert(!h.raw_completion && !h.program_loaded && !h.resident_words);
  assert(atlas_host_ensure_loaded(&h, program, 2) == ATLAS_HOST_OK);
  assert(atlas_host_ensure_loaded(&h, identical, 2) == ATLAS_HOST_OK);
  assert(h.image_loads == 6 && h.image_reuses == 2 && h.imem_words_written == 27);
  puts("atlas-host-runtime-merge: PASS");
  return 0;
}
