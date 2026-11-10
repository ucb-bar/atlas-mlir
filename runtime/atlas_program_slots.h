#ifndef ATLAS_PROGRAM_SLOTS_H
#define ATLAS_PROGRAM_SLOTS_H
#include "atlas_host.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Physical range metadata only. No source payloads or retention policy.
 * A single owner controls one endpoint. Load/select/discard/evict refuse active
 * or nonhalted hardware. Use discard/reset_completed for invalidation/reset;
 * these and failed replacement stale all handles. Observable external host
 * changes also stale handles. Direct MMIO mutation and host reinitialization
 * violate ownership; they must follow discard. Eviction stales only its handle
 * and clears selection. */
typedef struct { size_t offset, bytes, token; uint64_t generation; } atlas_program_slot;
typedef struct {
  atlas_host *host;
  atlas_program_slot *slots;
  size_t capacity, count, next_token;
  uint64_t generation;
  uint64_t observed_image_loads, observed_words_written;
  const uint32_t *observed_resident_words;
  size_t observed_resident_word_count;
  int observed_program_loaded, observed_raw_completion;
} atlas_program_slots;
atlas_host_result atlas_slots_init(atlas_program_slots *, atlas_host *, atlas_program_slot *, size_t);
atlas_host_result atlas_slots_query(atlas_program_slots *, size_t *, size_t *, size_t *, uint64_t *);
int atlas_slots_eligible(const void *, size_t);
atlas_host_result atlas_slots_load(atlas_program_slots *, size_t, const void *, size_t, size_t *);
atlas_host_result atlas_slots_select(atlas_program_slots *, size_t);
atlas_host_result atlas_slots_discard(atlas_program_slots *);
/* Acknowledge the owner's completed platform reset, including active recovery. */
atlas_host_result atlas_slots_reset_completed(atlas_program_slots *);
atlas_host_result atlas_slots_evict(atlas_program_slots *, size_t);
#ifdef __cplusplus
}
#endif
#endif
