#include "atlas_resnet_block.h"
#include "atlas_resnet_host.h"

static int span(const void *pointer, size_t bytes) {
  return pointer && bytes && bytes - 1 <= UINTPTR_MAX - (uintptr_t)pointer;
}

static int overlaps(const void *a, size_t na, const void *b, size_t nb) {
  uintptr_t x = (uintptr_t)a, y = (uintptr_t)b;
  return x <= y ? y - x < na : x - y < nb;
}

static uint32_t cycle(atlas_host *host) {
  return host->mmio.read32(host->mmio.context, host->map.csr_base + ATLAS_CSR_CYCLES);
}

static int valid(const atlas_host *host, const atlas_resnet_block_inputs *in, const atlas_resnet_block_workspace *w) {
  if (!span(host, sizeof(*host)) || !span(in, sizeof(*in)) || !span(w, sizeof(*w)) ||
      !span(in->entry, ATLAS_RESNET_SMALL * sizeof(float)) ||
      !span(in->expected_final_bits, ATLAS_RESNET_LARGE * sizeof(uint32_t)) || !w->max_polls ||
      w->small_words < ATLAS_RESNET_SMALL || w->branch_words < ATLAS_RESNET_LARGE ||
      w->identity_words < ATLAS_RESNET_LARGE || !span(w->small, ATLAS_RESNET_SMALL * sizeof(float)) ||
      !span(w->branch, ATLAS_RESNET_LARGE * sizeof(float)) || !span(w->identity, ATLAS_RESNET_LARGE * sizeof(float)) ||
      !span(w->dram, w->dram_capacity) || w->dram_capacity > UINT32_MAX || w->device_dram_address != ATLAS_DRAM_BASE ||
      ((uintptr_t)in->entry | (uintptr_t)in->expected_final_bits | (uintptr_t)w->small | (uintptr_t)w->branch |
       (uintptr_t)w->identity) &
          3)
    return 0;
  if (overlaps(in, sizeof(*in), w, sizeof(*w)) || overlaps(in, sizeof(*in), host, sizeof(*host)) ||
      overlaps(w, sizeof(*w), host, sizeof(*host)))
    return 0;
  const void *buffers[] = {w->small, w->branch, w->identity, w->dram};
  size_t bytes[] = {ATLAS_RESNET_SMALL * sizeof(float), ATLAS_RESNET_LARGE * sizeof(float),
                    ATLAS_RESNET_LARGE * sizeof(float), w->dram_capacity};
  for (unsigned i = 0; i < 4; ++i) {
    if (overlaps(buffers[i], bytes[i], host, sizeof(*host)) || overlaps(buffers[i], bytes[i], in, sizeof(*in)) ||
        overlaps(buffers[i], bytes[i], w, sizeof(*w)) ||
        overlaps(buffers[i], bytes[i], in->entry, ATLAS_RESNET_SMALL * sizeof(float)) ||
        overlaps(buffers[i], bytes[i], in->expected_final_bits, ATLAS_RESNET_LARGE * sizeof(uint32_t)))
      return 0;
    for (unsigned j = 0; j < i; ++j)
      if (overlaps(buffers[i], bytes[i], buffers[j], bytes[j])) return 0;
  }
  for (unsigned i = 0; i < 4; ++i) {
    const atlas_resnet_block_kernel *k = &in->kernels[i];
    unsigned inner = i == 1 ? 576 : 64, output = i < 2 ? 64 : 256;
    if (!k->program_words || k->program_words > ATLAS_IMEM_BYTES / 4 - 5 ||
        !span(k->program, k->program_words * sizeof(uint32_t)) || !span(k->weights, (size_t)inner * output) ||
        !span(k->affine_scale_bias, 2 * output * sizeof(float)) ||
        ((uintptr_t)k->program | (uintptr_t)k->affine_scale_bias) & 3 || k->kernel != (i == 1 ? 3u : 1u) ||
        k->output_channels != output || k->activation_bytes != (size_t)ATLAS_RESNET_SPATIAL * inner ||
        k->weight_bytes != (size_t)inner * output || k->output_bytes != (size_t)ATLAS_RESNET_SPATIAL * output * 2 ||
        k->output_offset != k->activation_bytes + k->weight_bytes + 256 ||
        k->dram_bytes != k->output_offset + k->output_bytes + 256 || k->dram_bytes > w->dram_capacity ||
        k->activation_scale < -126 || k->activation_scale > 127)
      return 0;
    for (unsigned j = 0; j < 4; ++j)
      if (overlaps(buffers[j], bytes[j], k->program, k->program_words * sizeof(uint32_t)) ||
          overlaps(buffers[j], bytes[j], k->weights, k->weight_bytes) ||
          overlaps(buffers[j], bytes[j], k->affine_scale_bias, 2 * output * sizeof(float)))
        return 0;
  }
  return 1;
}

atlas_resnet_block_result atlas_resnet_block_run(atlas_host *host, const atlas_resnet_block_inputs *in,
                                                 atlas_resnet_block_workspace *w) {
  if (!host || host->active || !valid(host, in, w)) return ATLAS_RESNET_BLOCK_INVALID;
  w->completed_stages = 0;
  w->first_bad_word = UINT32_MAX;
  for (unsigned stage = 0; stage < 4; ++stage) {
    const atlas_resnet_block_kernel *k = &in->kernels[stage];
    w->stage = stage;
    const float *activation = stage == 0 || stage == 3 ? in->entry : w->small;
    if (atlas_resnet_pack_a(activation, 64, 56, 56, k->kernel, w->dram, k->activation_bytes, &w->observed_scale[stage]))
      return ATLAS_RESNET_BLOCK_PACK;
    if (w->observed_scale[stage] != k->activation_scale) return ATLAS_RESNET_BLOCK_SCALE;
    for (size_t i = 0; i < k->weight_bytes; ++i) w->dram[k->activation_bytes + i] = k->weights[i];
    for (size_t i = k->activation_bytes + k->weight_bytes; i < k->output_offset; ++i) w->dram[i] = 0xa5;
    for (size_t i = k->output_offset; i < k->output_offset + k->output_bytes; ++i) w->dram[i] = 0xcd;
    for (size_t i = k->output_offset + k->output_bytes; i < k->dram_bytes; ++i) w->dram[i] = 0x5a;
    w->runtime_result = atlas_host_load_raw(host, k->program, k->program_words);
    if (w->runtime_result != ATLAS_HOST_OK) return ATLAS_RESNET_BLOCK_DEVICE;
    w->active_buffer = (atlas_host_buffer){w->device_dram_address, k->dram_bytes, ATLAS_HOST_DEVICE_READ_WRITE};
    uint32_t before = cycle(host);
    w->runtime_result = atlas_host_start(host, &w->active_buffer, 1);
    if (w->runtime_result == ATLAS_HOST_OK) w->runtime_result = atlas_host_wait(host, w->max_polls);
    w->atlas_cycles[stage] = cycle(host) - before;
    if (w->runtime_result != ATLAS_HOST_OK) return ATLAS_RESNET_BLOCK_DEVICE;
    w->completion_sequence[stage] = host->invocation_sequence;
    for (size_t i = k->activation_bytes + k->weight_bytes; i < k->output_offset; ++i)
      if (((volatile uint8_t *)w->dram)[i] != 0xa5) return ATLAS_RESNET_BLOCK_GUARD;
    for (size_t i = k->output_offset + k->output_bytes; i < k->dram_bytes; ++i)
      if (((volatile uint8_t *)w->dram)[i] != 0x5a) return ATLAS_RESNET_BLOCK_GUARD;
    float *output = stage < 2 ? w->small : stage == 2 ? w->branch : w->identity;
    if (atlas_resnet_decode_output(w->dram + k->output_offset, k->output_bytes, output, k->output_channels, 56, 56) ||
        atlas_resnet_affine(output, output, k->affine_scale_bias, k->affine_scale_bias + k->output_channels,
                            k->output_channels, 56, 56, stage < 2))
      return ATLAS_RESNET_BLOCK_MATH;
    w->completed_stages = stage + 1;
  }
  if (atlas_resnet_residual_relu(w->branch, w->identity, w->branch, ATLAS_RESNET_LARGE)) return ATLAS_RESNET_BLOCK_MATH;
  for (size_t i = 0; i < ATLAS_RESNET_LARGE; ++i) {
    union {
      float f;
      uint32_t u;
    } actual = {.f = w->branch[i]};
    if (actual.u != in->expected_final_bits[i]) {
      w->first_bad_word = (uint32_t)i;
      return ATLAS_RESNET_BLOCK_OUTPUT;
    }
  }
  w->stage = UINT32_MAX;
  return ATLAS_RESNET_BLOCK_OK;
}
