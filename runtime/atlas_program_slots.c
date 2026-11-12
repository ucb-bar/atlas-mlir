#include "atlas_program_slots.h"
#include <limits.h>

static int direct_target_in_body(size_t index, size_t count, uint32_t word, int jump) {
  uint32_t raw;
  int32_t displacement;
  if (jump) {
    raw = ((word >> 31) << 20) | (((word >> 21) & 1023u) << 1) |
          (((word >> 20) & 1u) << 11) | (((word >> 12) & 255u) << 12);
    displacement = (int32_t)raw - ((raw & 0x100000u) ? 0x200000 : 0);
  } else {
    raw = ((word >> 31) << 12) | (((word >> 25) & 63u) << 5) |
          (((word >> 8) & 15u) << 1) | (((word >> 7) & 1u) << 11);
    displacement = (int32_t)raw - ((raw & 0x1000u) ? 0x2000 : 0);
  }
  /* Selected ScalarCore applies the encoded byte displacement / 2 to its
   * instruction-index PC, matching AtlasEncoding.cpp's checked targets. */
  int64_t target = (int64_t)index + displacement / 2;
  return index + 1 < count && target >= 0 && (uint64_t)target < count;
}

static int position_independent_body(const uint32_t *words, size_t count) {
  if (!words || !count || count > INT32_MAX)
    return 0;
  unsigned ecalls = 0;
  int delay_slot = 0;
  for (size_t index = 0; index < count; ++index) {
    uint32_t word = words[index];
    unsigned opcode = word & 0x7fu, funct3 = (word >> 12) & 7u, funct7 = word >> 25;
    int redirects = opcode == 0x63u || opcode == 0x6fu;
    if (delay_slot && redirects) return 0;
    delay_slot = redirects;
    switch (opcode) {
    case 0x03: /* Scalar memory and explicit scale-register accesses. */
      if (funct3 == 3) return 0;
      break;
    case 0x07: /* Vector load/store. */
      if (word & (1u << 14)) return 0;
      break;
    case 0x0f:
      if (word != 0x0000000fu) return 0;
      break;
    case 0x13:
      if ((funct3 == 1 && funct7) || (funct3 == 5 && funct7 != 0 && funct7 != 0x20)) return 0;
      break;
    case 0x23:
      if (funct3 > 2) return 0;
      break;
    case 0x33:
      if (funct7 && (funct7 != 0x20 || (funct3 != 0 && funct3 != 5))) return 0;
      break;
    case 0x37: /* LUI; AUIPC is deliberately absent. */
      break;
    case 0x57:
      if (!(funct7 <= 7 || funct7 == 0x21 || funct7 == 0x24 || funct7 == 0x26 ||
            (funct7 >= 0x40 && funct7 <= 0x4d))) return 0;
      break;
    case 0x5f:
      if (word & (1u << 15)) return 0;
      break;
    case 0x63:
      if (funct3 == 2 || funct3 == 3 || !direct_target_in_body(index, count, word, 0)) return 0;
      break;
    case 0x67: /* Only DELAY; register-indirect JALR is not relocatable. */
      if ((word & 0x000fffffu) != 0x00001067u) return 0;
      break;
    case 0x6b:
      if (word & 0xfff80000u) return 0;
      break;
    case 0x6f:
      if (((word >> 7) & 31u) || !direct_target_in_body(index, count, word, 1)) return 0;
      break;
    case 0x73:
      if (word == 0x00000073u) {
        ++ecalls;
      } else {
        unsigned csr = word >> 20;
        if (!funct3 || funct3 == 4 || (csr != 0xc10u && csr != 0xc11u)) return 0;
      }
      break;
    case 0x77:
      if (funct7 > 13) return 0;
      break;
    case 0x7b:
      if (word & 0xfc000000u) return 0;
      break;
    case 0x7f:
      if (word & ~0x020ff07fu) return 0;
      if ((word & 0x02000000u) && (word & 0x000f8000u)) return 0;
      break;
    default:
      return 0;
    }
  }
  return ecalls == 1;
}

static uint32_t resident_jump(size_t entry_word) {
  uint32_t immediate = (uint32_t)entry_word * 2u;
  return (((immediate >> 20) & 1u) << 31) | (((immediate >> 1) & 1023u) << 21) |
         (((immediate >> 11) & 1u) << 20) | (((immediate >> 12) & 255u) << 12) | 0x6fu;
}

static void observe_host(atlas_program_slots *s) {
  s->observed_image_loads = s->host->image_loads;
  s->observed_words_written = s->host->imem_words_written;
  s->observed_program_loaded = s->host->program_loaded;
  s->observed_resident_words = s->host->resident_words;
  s->observed_resident_word_count = s->host->resident_word_count;
  s->observed_raw_completion = s->host->raw_completion;
}

static atlas_host_result stale_slots(atlas_program_slots *s) {
  s->count = 0;
  observe_host(s);
  if (s->generation == UINT64_MAX) return ATLAS_HOST_INVALID;
  ++s->generation;
  return ATLAS_HOST_OK;
}

static atlas_host_result slot_ready(atlas_program_slots *s) {
  if (!s || !s->host) return ATLAS_HOST_INVALID;
  if (s->host->active) return ATLAS_HOST_BUSY;
  if (s->generation == UINT64_MAX) return ATLAS_HOST_INVALID;
  if (!(s->host->mmio.read32(s->host->mmio.context,
        s->host->map.csr_base+ATLAS_CSR_STATUS)&1u)) return ATLAS_HOST_NOT_READY;
  if (s->observed_image_loads != s->host->image_loads ||
      s->observed_words_written != s->host->imem_words_written ||
      s->observed_program_loaded != s->host->program_loaded ||
      s->observed_resident_words != s->host->resident_words ||
      s->observed_resident_word_count != s->host->resident_word_count ||
      s->observed_raw_completion != s->host->raw_completion) {
    atlas_host_result result = atlas_host_invalidate_program(s->host);
    if (result) return result;
    return stale_slots(s);
  }
  return ATLAS_HOST_OK;
}
atlas_host_result atlas_slots_init(atlas_program_slots *s, atlas_host *host,
                                   atlas_program_slot *slots, size_t capacity) {
  if (!s || !host || !slots || !capacity) return ATLAS_HOST_INVALID;
  *s=(atlas_program_slots){.host=host,.slots=slots,.capacity=capacity,.generation=1};
  observe_host(s);
  return ATLAS_HOST_OK;
}
atlas_host_result atlas_slots_query(atlas_program_slots *s, size_t *first,
                                     size_t *capacity, size_t *entries, uint64_t *generation) {
  atlas_host_result r=slot_ready(s);
  if (r) return r;
  if (!first || !capacity || !entries || !generation) return ATLAS_HOST_INVALID;
  *first=8; *capacity=s->host->map.imem_capacity_bytes;
  /* Direct reset-entry JAL supports these positive word targets. */
  if (*capacity>0x7ffffu*4u) *capacity=0x7ffffu*4u;
  if (*capacity<*first) *first=*capacity; /* Force generic reset-entry fallback. */
  *entries=s->capacity; *generation=s->generation;
  return ATLAS_HOST_OK;
}
int atlas_slots_eligible(const void *payload, size_t bytes) {
  return payload && !((uintptr_t)payload&3u) && bytes && !(bytes&3u) &&
         position_independent_body(payload,bytes/4);
}
static atlas_host_result slot_invalid(atlas_program_slots *s) {
  /* Failed replacement/selection invalidates every retained handle. */
  atlas_host_result result = atlas_host_invalidate_program(s->host);
  if (result) return result;
  stale_slots(s);
  return ATLAS_HOST_INVALID;
}
atlas_host_result atlas_slots_load(atlas_program_slots *s, size_t offset,
    const void *payload, size_t bytes, size_t *handle) {
  atlas_host_result r=slot_ready(s);
  if (r) return r;
  size_t first,capacity,entries; uint64_t generation;
  r=atlas_slots_query(s,&first,&capacity,&entries,&generation);
  if (r) return r;
  /* Any failed replacement leaves no previously selected image launchable. */
  s->host->program_loaded=0;
  if (!handle || offset<first || offset&3u || offset>capacity || bytes>capacity-offset ||
      s->count==s->capacity || s->next_token==SIZE_MAX || !atlas_slots_eligible(payload,bytes)) return slot_invalid(s);
  for (size_t i=0;i<s->count;++i)
    if (offset<s->slots[i].offset+s->slots[i].bytes && s->slots[i].offset<offset+bytes)
      return slot_invalid(s);
  const uint32_t *words=payload;
  for (size_t i=0;i<bytes/4;++i)
    s->host->mmio.write32(s->host->mmio.context,s->host->map.imem_base+offset+i*4,words[i]);
  s->slots[s->count++]=(atlas_program_slot){offset,bytes,++s->next_token,generation};
  *handle=s->next_token;
  ++s->host->image_loads; s->host->imem_words_written+=bytes/4;
  observe_host(s);
  return ATLAS_HOST_OK;
}
atlas_host_result atlas_slots_select(atlas_program_slots *s, size_t handle) {
  atlas_host_result r=slot_ready(s);
  if (r) return r;
  s->host->program_loaded=0;
  size_t index=0;
  while (index<s->count && s->slots[index].token!=handle) ++index;
  if (!handle || index==s->count || s->slots[index].generation!=s->generation)
    return slot_invalid(s);
  size_t entry=s->slots[index].offset/4;
  s->host->mmio.write32(s->host->mmio.context,s->host->map.imem_base+4,0x00000013u);
  s->host->mmio.write32(s->host->mmio.context,s->host->map.imem_base,resident_jump(entry));
  s->host->imem_words_written+=2;
  s->host->resident_words=0; s->host->resident_word_count=0;
  s->host->raw_completion=0; s->host->program_loaded=1;
  observe_host(s);
  return ATLAS_HOST_OK;
}
atlas_host_result atlas_slots_discard(atlas_program_slots *s) {
  atlas_host_result r=slot_ready(s);
  if (r) return r;
  r=atlas_host_invalidate_program(s->host);
  if (!r) r=stale_slots(s);
  return r;
}

atlas_host_result atlas_slots_reset_completed(atlas_program_slots *s) {
  if (!s || !s->host) return ATLAS_HOST_INVALID;
  atlas_host_result result = atlas_host_reset_completed(s->host);
  return result ? result : stale_slots(s);
}

atlas_host_result atlas_slots_evict(atlas_program_slots *s, size_t handle) {
  atlas_host_result result = slot_ready(s);
  if (result) return result;
  size_t index = 0;
  while (index < s->count && s->slots[index].token != handle) ++index;
  if (!handle || index == s->count) return ATLAS_HOST_INVALID;
  /* Any eviction clears selection. Other physical slots and tokens survive. */
  s->host->program_loaded = 0;
  s->host->resident_words = 0;
  s->host->resident_word_count = 0;
  s->slots[index] = s->slots[--s->count];
  observe_host(s);
  return ATLAS_HOST_OK;
}
