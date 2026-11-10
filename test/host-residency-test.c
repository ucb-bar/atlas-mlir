#include "atlas_program_slots.h"
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

enum { WORD_CAPACITY = 4096 };
typedef struct {
  uint32_t words[WORD_CAPACITY], status, retired, result;
  size_t capacity, writes, starts;
} device;

static int32_t jump_displacement(uint32_t word) {
  uint32_t raw = ((word >> 31) << 20) | (((word >> 21) & 1023u) << 1) |
                 (((word >> 20) & 1u) << 11) | (((word >> 12) & 255u) << 12);
  return (int32_t)raw - ((raw & 0x100000u) ? 0x200000 : 0);
}

/* Independently execute scalar words, including the selected core's one delay
 * slot. No resident table or expected image selects the instruction stream. */
static void execute(device *d) {
  uint32_t registers[32] = {0};
  size_t pc = 0, pending = 0;
  int has_pending = 0;
  for (unsigned steps = 0; steps < 1000; ++steps) {
    assert(pc < d->capacity);
    uint32_t word = d->words[pc];
    size_t next = pc + 1, redirect = 0;
    int redirects = 0;
    unsigned opcode = word & 127u;
    if (word == 0x73u) {
      d->status = 5;
      d->result = registers[1];
      ++d->retired;
      return;
    } else if (opcode == 0x13u) {
      assert(((word >> 12) & 7u) == 0);
      int32_t immediate = (int32_t)(word >> 20);
      if (immediate & 0x800) immediate -= 0x1000;
      unsigned destination = (word >> 7) & 31u;
      if (destination) registers[destination] = registers[(word >> 15) & 31u] + immediate;
    } else if (opcode == 0x6fu) {
      assert(((word >> 7) & 31u) == 0 && !has_pending);
      redirect = (size_t)((int64_t)pc + jump_displacement(word) / 2);
      redirects = 1;
    } else if (opcode == 0x63u) {
      assert(((word >> 12) & 7u) == 4 && !has_pending);
      uint32_t raw = ((word >> 31) << 12) | (((word >> 25) & 63u) << 5) |
                     (((word >> 8) & 15u) << 1) | (((word >> 7) & 1u) << 11);
      int32_t displacement = (int32_t)raw - ((raw & 0x1000u) ? 0x2000 : 0);
      if ((int32_t)registers[(word >> 15) & 31u] < (int32_t)registers[(word >> 20) & 31u]) {
        redirect = (size_t)((int64_t)pc + displacement / 2);
        redirects = 1;
      }
    } else {
      assert(!"unimplemented test instruction");
    }
    ++d->retired;
    if (has_pending) next = pending;
    has_pending = redirects;
    pending = redirect;
    pc = next;
  }
  assert(!"test stream did not terminate");
}

static uint32_t read32(void *context, uintptr_t address) {
  device *d = context;
  if (address == ATLAS_CSR_BASE + ATLAS_CSR_STATUS) return d->status;
  assert(address == ATLAS_CSR_BASE + ATLAS_CSR_RETIRED);
  return d->retired;
}

static void write32(void *context, uintptr_t address, uint32_t word) {
  device *d = context;
  if (address == ATLAS_CSR_BASE + ATLAS_CSR_EXEC_CONTROL) {
    assert(word == 1);
    ++d->starts;
    execute(d);
    return;
  }
  assert(address >= ATLAS_IMEM_BASE && address < ATLAS_IMEM_BASE + 4 * d->capacity && !(address & 3));
  d->words[(address - ATLAS_IMEM_BASE) / 4] = word;
  ++d->writes;
}

static void launch(atlas_host *h, device *d, uint32_t expected) {
  assert(atlas_host_start(h, 0, 0) == ATLAS_HOST_OK);
  assert(atlas_host_wait(h, 2) == ATLAS_HOST_OK);
  assert(d->result == expected);
}

int main(void) {
  device d = {.capacity = 32, .status = 5};
  atlas_host h;
  atlas_program_slots slots;
  atlas_program_slot metadata[3];
  const uint32_t a[] = {0x02a00093u, 0x73u};
  const uint32_t b[] = {0x00700093u, 0x73u};
  const uint32_t delay[] = {0x00100093u, 0x0060006fu, 0x00108093u, 0x06300093u, 0x73u};
  const uint32_t cfg[] = {0x0040006fu, 0x13u, 0x73u, 0x13u};
  const uint32_t two_halts[] = {0x73u, 0x73u};
  const uint32_t refused[] = {0x97u, 0x8067u, 0x002000efu, 0xc02020f3u,
                             0xffdff06fu, 0x0060006fu, 0xfe000fe3u, 0u, 0x00100073u};
  assert(!atlas_host_init(&h, (atlas_host_map){ATLAS_IMEM_BASE, ATLAS_CSR_BASE, 128},
      (atlas_host_mmio){read32, write32, &d}, (atlas_host_memory){1, 0, 0}));
  assert(!atlas_slots_init(&slots, &h, metadata, 3));
  size_t first, capacity, entries, ah, bh, dh;
  uint64_t generation;
  assert(!atlas_slots_query(&slots, &first, &capacity, &entries, &generation));
  assert(first == 8 && capacity == 128 && entries == 3);
  assert(atlas_slots_eligible(cfg, sizeof cfg));
  assert(!atlas_slots_eligible(two_halts, sizeof two_halts));
  for (size_t i = 0; i < sizeof refused / sizeof refused[0]; ++i) {
    const uint32_t image[] = {refused[i], 0x73u};
    assert(!atlas_slots_eligible(image, sizeof image));
  }
  assert(!atlas_slots_load(&slots, 8, a, sizeof a, &ah));
  assert(!atlas_slots_load(&slots, 16, b, sizeof b, &bh));
  assert(!memcmp(d.words + 2, a, sizeof a) && !memcmp(d.words + 4, b, sizeof b));
  assert(!atlas_slots_select(&slots, ah));
  assert(d.words[1] == 0x13u && jump_displacement(d.words[0]) == 4);
  launch(&h, &d, 42);
  assert(!atlas_slots_select(&slots, bh));
  launch(&h, &d, 7);
  assert(!atlas_slots_select(&slots, ah));
  launch(&h, &d, 42);
  assert(h.image_loads == 2); /* Selection never reloads body words. */
  assert(!atlas_slots_evict(&slots, bh));
  assert(slots.count == 1 && !h.program_loaded);
  assert(!atlas_slots_select(&slots, ah));
  launch(&h, &d, 42);
  assert(!atlas_slots_load(&slots, 16, b, sizeof b, &bh));
  assert(!atlas_slots_load(&slots, 24, delay, sizeof delay, &dh));
  assert(!atlas_slots_select(&slots, dh));
  launch(&h, &d, 2); /* Actual observable selected-core JAL delay slot. */
  size_t writes = d.writes;
  h.active = 1;
  assert(atlas_slots_load(&slots, 48, a, sizeof a, &ah) == ATLAS_HOST_BUSY);
  assert(atlas_slots_select(&slots, bh) == ATLAS_HOST_BUSY);
  assert(atlas_slots_discard(&slots) == ATLAS_HOST_BUSY);
  assert(d.writes == writes && slots.count == 3);
  h.active = 0;
  d.status = 0;
  assert(atlas_slots_select(&slots, bh) == ATLAS_HOST_NOT_READY);
  assert(d.writes == writes);
  d.status = 5;
  assert(!atlas_slots_discard(&slots));
  assert(atlas_slots_select(&slots, bh) == ATLAS_HOST_INVALID);
  assert(!h.program_loaded);
  assert(!atlas_slots_load(&slots, 8, a, sizeof a, &ah));
  assert(ah != bh); /* Tokens do not revive across resets/discards. */
  assert(atlas_slots_load(&slots, 8, b, sizeof b, &bh) == ATLAS_HOST_INVALID);
  assert(!h.program_loaded); /* Overlap refused without writing. */
  assert(atlas_slots_load(&slots, 4, b, sizeof b, &bh) == ATLAS_HOST_INVALID);
  assert(atlas_slots_load(&slots, 128, b, sizeof b, &bh) == ATLAS_HOST_INVALID);
  assert(atlas_slots_load(&slots, SIZE_MAX-3, b, sizeof b, &bh) == ATLAS_HOST_INVALID);
  assert(!atlas_slots_discard(&slots));
  assert(!atlas_slots_load(&slots, 8, a, sizeof a, &ah));
  assert(!atlas_host_load(&h, b, 2)); /* Explicit reset-entry replacement invalidates handles. */
  assert(atlas_slots_select(&slots, ah) == ATLAS_HOST_INVALID);
  assert(!atlas_slots_load(&slots, 8, a, sizeof a, &ah));
  assert(!atlas_slots_reset_completed(&slots));
  assert(atlas_slots_select(&slots, ah) == ATLAS_HOST_INVALID);
  assert(!atlas_slots_load(&slots, 8, a, sizeof a, &ah));
  assert(!atlas_slots_select(&slots, ah));
  assert(!atlas_host_invalidate_program(&h));
  assert(atlas_slots_select(&slots, ah) == ATLAS_HOST_INVALID);
  assert(!atlas_slots_load(&slots, 8, a, sizeof a, &ah));
  assert(!atlas_slots_select(&slots, ah));
  assert(atlas_host_load(&h, 0, 0) == ATLAS_HOST_INVALID);
  assert(atlas_slots_select(&slots, ah) == ATLAS_HOST_INVALID);
  assert(!atlas_slots_load(&slots, 8, a, sizeof a, &ah));
  assert(!atlas_slots_select(&slots, ah));
  h.active = 1;
  assert(!atlas_slots_reset_completed(&slots));
  assert(!h.active && !h.program_loaded);
  assert(atlas_slots_select(&slots, ah) == ATLAS_HOST_INVALID);
  puts("atlas-host-residency-primitives: PASS");
  return 0;
}
