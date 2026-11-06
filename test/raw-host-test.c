#include "atlas_host.h"
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <assert.h>
#include <stdint.h>
#include <stdio.h>

typedef struct {
  uint32_t words[64], status, done, command;
  unsigned starts, writes, sync_before, sync_after, mode, phase;
  int cache_fails;
  uintptr_t first_poll_address;
} device;

/* Interpret only the adapter's scalar CSR/NOP/ECALL tail. This checks that
 * published instruction words actually copy the current invocation command. */
static void complete(device *d) {
  uint32_t gpr[32] = {0};
  int halted = 0;
  for (size_t pc = 0; pc < 64; ++pc) {
    uint32_t w = d->words[pc];
    if (w == 0x73) {
      halted = 1;
      break;
    }
    if (w == 0x13) continue;
    assert((w & 127) == 0x73);
    unsigned csr = w >> 20, f = (w >> 12) & 7;
    unsigned rd = (w >> 7) & 31, rs = (w >> 15) & 31;
    uint32_t old = csr == 0xc11 ? d->command : d->done;
    if (f == 1) {
      assert(csr == 0xc10 && rd == 0);
      d->done = gpr[rs];
    } else {
      assert(f == 2 && csr == 0xc11 && rs == 0);
    }
    if (rd) gpr[rd] = old;
  }
  assert(halted);
  d->status = 5;
}

static uint32_t read32(void *context, uintptr_t address) {
  device *d = context;
  if (d->starts && !d->first_poll_address) d->first_poll_address = address;
  if (address == ATLAS_CSR_BASE + ATLAS_CSR_DBG0) {
    uint32_t result = d->done;
    if (d->mode == 1 && d->phase++ == 0) complete(d);
    return result;
  }
  assert(address == ATLAS_CSR_BASE + ATLAS_CSR_STATUS);
  return d->status;
}

static void write32(void *context, uintptr_t address, uint32_t value) {
  device *d = context;
  ++d->writes;
  if (address == ATLAS_CSR_BASE + ATLAS_CSR_DBG0) {
    d->done = value;
    return;
  }
  if (address == ATLAS_CSR_BASE + ATLAS_CSR_DBG1) {
    d->command = value;
    return;
  }
  if (address == ATLAS_CSR_BASE + ATLAS_CSR_EXEC_CONTROL) {
    assert(value == 1 && d->command && !d->done);
    ++d->starts;
    d->phase = 0;
    d->first_poll_address = 0;
    if (!d->mode) complete(d);
    return;
  }
  assert(address >= ATLAS_IMEM_BASE && address < ATLAS_IMEM_BASE + sizeof d->words && !(address & 3));
  d->words[(address - ATLAS_IMEM_BASE) / 4] = value;
}

static int sync_buffer(void *context, const atlas_host_buffer *b, int before) {
  device *d = context;
  assert(b->size_bytes);
  if (before)
    ++d->sync_before;
  else
    ++d->sync_after;
  return d->cache_fails;
}

static atlas_host setup(device *d) {
  atlas_host h;
  assert(atlas_host_init(&h, (atlas_host_map){ATLAS_IMEM_BASE, ATLAS_CSR_BASE, sizeof d->words},
                         (atlas_host_mmio){read32, write32, d}, (atlas_host_memory){0, sync_buffer, d}) == ATLAS_HOST_OK);
  return h;
}

int main(void) {
  const uint32_t program[] = {0x13, 0x73};
  device d = {.status = 5};
  atlas_host h = setup(&d);
  atlas_host_buffer b = {ATLAS_DRAM_BASE, 401408, ATLAS_HOST_DEVICE_WRITE};
  assert(atlas_host_load_raw(&h, program, 2) == ATLAS_HOST_OK);
  d.cache_fails = 1;
  assert(atlas_host_start(&h, &b, 1) == ATLAS_HOST_CACHE_ERROR);
  assert(!h.active && !h.invocation_sequence && !d.starts && !d.command);
  d.cache_fails = 0;
  d.sync_before = 0;
  assert(atlas_host_start(&h, &b, 1) == ATLAS_HOST_OK);
  assert(d.command == 1 && d.done == 1 && h.active);
  assert(atlas_host_wait(&h, 1) == ATLAS_HOST_OK);
  assert(d.first_poll_address == ATLAS_CSR_BASE + ATLAS_CSR_DBG0);
  assert(d.sync_before == 1 && d.sync_after == 1);
  assert(atlas_host_start(&h, &b, 1) == ATLAS_HOST_OK);
  assert(d.command == 2 && d.done == 2);
  assert(atlas_host_poll(&h) == ATLAS_HOST_OK);

  /* Old ECALL plus marker changing between separate MMIO reads must not
   * release ownership. The next marker-first observation can complete. */
  d.mode = 1;
  assert(atlas_host_start(&h, &b, 1) == ATLAS_HOST_OK);
  assert(atlas_host_wait(&h, 1) == ATLAS_HOST_TIMEOUT);
  assert(h.active && d.command == 3 && d.done == 3 && d.sync_after == 2);
  assert(d.first_poll_address == ATLAS_CSR_BASE + ATLAS_CSR_DBG0);
  assert(atlas_host_load_raw(&h, program, 2) == ATLAS_HOST_BUSY);
  assert(atlas_host_poll(&h) == ATLAS_HOST_OK);
  assert(d.sync_after == 3);

  /* Wrong or old marker never completes, even with persistent ECALL status. */
  d.mode = 2;
  assert(atlas_host_start(&h, &b, 1) == ATLAS_HOST_OK);
  d.done = 3;
  assert(atlas_host_wait(&h, 3) == ATLAS_HOST_TIMEOUT && h.active);
  d.done = d.command;
  d.status = 0;
  assert(atlas_host_poll(&h) == ATLAS_HOST_BUSY && h.active);
  d.status = 3;
  assert(atlas_host_poll(&h) == ATLAS_HOST_BAD_HALT && h.active);
  assert(atlas_host_poll(&h) == ATLAS_HOST_BAD_HALT && h.active);
  assert(atlas_host_load_raw(&h, program, 2) == ATLAS_HOST_BUSY);
  assert(atlas_host_start(&h, &b, 1) == ATLAS_HOST_BUSY);
  assert(d.sync_after == 3);
  /* The fake device has no engines; the test now explicitly establishes its
   * quiescence before reinitializing the host ownership state. */
  d.status = 5;
  h = setup(&d);
  assert(atlas_host_load_raw(&h, program, 2) == ATLAS_HOST_OK);

  d.mode = 0;
  d.status = 5;
  assert(atlas_host_start(&h, &b, 1) == ATLAS_HOST_OK);
  d.cache_fails = 1;
  assert(atlas_host_poll(&h) == ATLAS_HOST_CACHE_ERROR && h.active);
  d.cache_fails = 0;
  assert(atlas_host_poll(&h) == ATLAS_HOST_OK && !h.active);

  b.address = UINTPTR_MAX - 15;
  b.size_bytes = 32;
  unsigned starts = d.starts;
  assert(atlas_host_start(&h, &b, 1) == ATLAS_HOST_INVALID && d.starts == starts);
  b.size_bytes = 16;
  assert(atlas_host_start(&h, &b, 1) == ATLAS_HOST_OK);
  assert(atlas_host_poll(&h) == ATLAS_HOST_OK);
  b.access = (atlas_host_access)-1;
  assert(atlas_host_start(&h, &b, 1) == ATLAS_HOST_INVALID);
  b.access = ATLAS_HOST_DEVICE_WRITE;
  h.invocation_sequence = UINT32_MAX;
  assert(atlas_host_start(&h, &b, 1) == ATLAS_HOST_NOT_READY && !h.active);

  /* Refusal invalidates any previous program, and no ATLAS_IMEM_BASE write occurs. */
  uint32_t invalid[][3] = {
      {0x73, 0x13, 0x73}, {0x00100073, 0x13, 0x73}, {0xc1001073, 0x13, 0x73}, {0xc110d073, 0x13, 0x73}, {0x13, 0x13, 0x13},
  };
  for (size_t i = 0; i < sizeof invalid / sizeof invalid[0]; ++i) {
    unsigned writes = d.writes;
    assert(atlas_host_load_raw(&h, invalid[i], 3) == ATLAS_HOST_INVALID);
    assert(!h.program_loaded && d.writes == writes);
  }
  assert(atlas_host_load_raw(&h, NULL, 0) == ATLAS_HOST_INVALID);
  assert(atlas_host_load_raw(&h, program, SIZE_MAX) == ATLAS_HOST_INVALID);
  h.map.imem_capacity_bytes = 24;
  assert(atlas_host_load_raw(&h, program, 2) == ATLAS_HOST_INVALID);
  h.map.imem_capacity_bytes = sizeof d.words;
  d.status = 0;
  assert(atlas_host_load_raw(&h, program, 2) == ATLAS_HOST_NOT_READY);
  d.status = 5;
  assert(atlas_host_load(&h, program, 2) == ATLAS_HOST_OK);
  b.address = ATLAS_DRAM_BASE;
  b.size_bytes = 32;
  assert(!h.raw_completion);
  assert(atlas_host_start(&h, &b, 1) == ATLAS_HOST_INVALID);
  puts("atlas-raw-host: PASS");
  return 0;
}
