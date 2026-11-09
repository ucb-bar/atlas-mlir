#include "atlas_bf16_call.h"
#include "finite_float_boundary.h"

static int span(const void *p, size_t n) {
  return p && n && (uintptr_t)p <= UINTPTR_MAX - n;
}
static int overlap(const void *a, size_t an, const void *b, size_t bn) {
  return (uintptr_t)a < (uintptr_t)b + bn && (uintptr_t)b < (uintptr_t)a + an;
}
static int device_span(uintptr_t address, size_t bytes) {
  return address && address <= UINT32_MAX && bytes && bytes - 1 <= UINT32_MAX - address;
}
static int program_valid(const atlas_bf16_program *p) {
  if (!p || !p->words || !p->word_count || p->word_count > SIZE_MAX / sizeof(uint32_t) ||
      !span(p->words, p->word_count * sizeof(uint32_t)) || !p->rows || !p->columns ||
      p->rows > SIZE_MAX / p->columns || p->rows * p->columns > SIZE_MAX / sizeof(uint16_t) ||
      p->rows * p->columns * sizeof(uint16_t) != p->payload_bytes ||
      !p->first_columns || p->first_columns >= p->columns || !p->device_alignment ||
      (p->device_alignment & (p->device_alignment - 1)) ||
      p->input_address % p->device_alignment || p->output_address % p->device_alignment ||
      !device_span(p->input_address, p->payload_bytes) || !device_span(p->output_address, p->payload_bytes)) return 0;
  return p->input_address <= p->output_address ?
      p->output_address - p->input_address >= p->payload_bytes :
      p->input_address - p->output_address >= p->payload_bytes;
}
static int separate(atlas_bf16_call *c, const void *p, size_t n) {
  const atlas_bf16_program *program = c->program;
  return !overlap(p, n, c, sizeof(*c)) && !overlap(p, n, program, sizeof(*program)) &&
         !overlap(p, n, program->words, program->word_count * sizeof(uint32_t));
}

atlas_model_tile_status atlas_bf16_call_init(
    atlas_bf16_call *context, const atlas_bf16_program *program,
    atlas_host_map map, atlas_host_mmio mmio, atlas_host_memory memory,
    atlas_tile_memory_io io, size_t max_polls, void *workspace, size_t workspace_bytes) {
  if (!context || (uintptr_t)context % _Alignof(atlas_bf16_call) || !program_valid(program) ||
      !io.write || !io.read || !max_polls || workspace_bytes < program->payload_bytes ||
      !span(workspace, workspace_bytes) || overlap(context, sizeof(*context), workspace, workspace_bytes) ||
      overlap(context, sizeof(*context), program, sizeof(*program)) ||
      overlap(context, sizeof(*context), program->words, program->word_count * sizeof(uint32_t)) ||
      overlap(workspace, workspace_bytes, program, sizeof(*program)) ||
      overlap(workspace, workspace_bytes, program->words, program->word_count * sizeof(uint32_t)) ||
      program->word_count > map.imem_capacity_bytes / sizeof(uint32_t)) return ATLAS_MODEL_TILE_INVALID;
  atlas_host host;
  atlas_host_result status = atlas_host_init(&host, map, mmio, memory);
  if (status != ATLAS_HOST_OK) return ATLAS_MODEL_TILE_INVALID;
  *context = (atlas_bf16_call){.host = host, .io = io, .program = program,
      .workspace = workspace, .workspace_bytes = workspace_bytes, .max_polls = max_polls,
      .device_result = ATLAS_HOST_NOT_READY, .initialized = 1};
  context->active_buffers[0] = (atlas_host_buffer){program->input_address, program->payload_bytes, ATLAS_HOST_DEVICE_READ};
  context->active_buffers[1] = (atlas_host_buffer){program->output_address, program->payload_bytes, ATLAS_HOST_DEVICE_WRITE};
  return ATLAS_MODEL_TILE_OK;
}

int atlas_bf16_call_run(void *opaque, const void *const *inputs, void *const *outputs) {
  atlas_bf16_call *c = opaque;
  if (!c || (uintptr_t)c % _Alignof(atlas_bf16_call) || !c->initialized || !program_valid(c->program) ||
      !inputs || !outputs || !c->io.write || !c->io.read || !c->max_polls ||
      c->workspace_bytes < c->program->payload_bytes || !span(c->workspace, c->workspace_bytes) ||
      !separate(c, c->workspace, c->workspace_bytes)) return ATLAS_MODEL_TILE_INVALID;
  if (c->failed || c->host.active) return ATLAS_MODEL_TILE_DEVICE_ERROR;
  const atlas_bf16_program *p = c->program;
  const void *input = inputs[0];
  void *output = outputs[0];
  size_t bytes = p->payload_bytes;
  if (!span(input, bytes) || !span(output, bytes) ||
      (uintptr_t)input % _Alignof(uint16_t) || (uintptr_t)output % _Alignof(uint16_t) ||
      overlap(input, bytes, output, bytes) ||
      overlap(input, bytes, c->workspace, c->workspace_bytes) ||
      overlap(output, bytes, c->workspace, c->workspace_bytes) ||
      !separate(c, input, bytes) || !separate(c, output, bytes)) return ATLAS_MODEL_TILE_INVALID;
  /* Snapshot every source bit before IO. No BF16-to-float conversion occurs. */
  for (size_t row = 0; row < p->rows; ++row) {
    for (size_t col = 0; col < p->columns; ++col) {
      uint16_t code;
      __builtin_memcpy(&code, (const uint8_t *)input + (row * p->columns + col) * 2, 2);
      if (!merlin_bf16_finite_normal_or_zero(code)) return ATLAS_MODEL_TILE_INPUT_DOMAIN;
      size_t physical = col < p->first_columns ? row * p->first_columns + col :
          p->rows * p->first_columns + row * (p->columns - p->first_columns) + col - p->first_columns;
      c->workspace[2 * physical] = (uint8_t)code;
      c->workspace[2 * physical + 1] = (uint8_t)(code >> 8);
    }
  }
  c->device_result = atlas_host_load(&c->host, p->words, p->word_count);
  if (c->device_result != ATLAS_HOST_OK) goto device_error;
  if (c->io.write(c->io.context, p->input_address, c->workspace, bytes)) goto memory_error;
  c->device_result = atlas_host_start(&c->host, c->active_buffers, 2);
  if (c->device_result == ATLAS_HOST_OK) c->device_result = atlas_host_wait(&c->host, c->max_polls);
  if (c->device_result != ATLAS_HOST_OK) goto device_error;
  if (c->io.read(c->io.context, p->output_address, c->workspace, bytes)) goto memory_error;
  /* Full read and required output visibility completed. Decode cannot fail
   * after the already-checked shape, extent, disjointness and layout contract. */
  for (size_t row = 0; row < p->rows; ++row) {
    for (size_t col = 0; col < p->columns; ++col) {
      size_t physical = col < p->first_columns ? row * p->first_columns + col :
          p->rows * p->first_columns + row * (p->columns - p->first_columns) + col - p->first_columns;
      uint16_t code = (uint16_t)c->workspace[2 * physical] | ((uint16_t)c->workspace[2 * physical + 1] << 8);
      __builtin_memcpy((uint8_t *)output + (row * p->columns + col) * 2, &code, 2);
    }
  }
  return ATLAS_MODEL_TILE_OK;
memory_error:
  c->failed = 1;
  return ATLAS_MODEL_TILE_MEMORY_ERROR;
device_error:
  c->failed = 1;
  return ATLAS_MODEL_TILE_DEVICE_ERROR;
}

atlas_host_result atlas_bf16_call_drain(atlas_bf16_call *c, size_t max_polls) {
  if (!c || !c->initialized || !max_polls) return ATLAS_HOST_INVALID;
  atlas_host_result status = c->host.active ? atlas_host_wait(&c->host, max_polls) : ATLAS_HOST_OK;
  if (status == ATLAS_HOST_OK) c->failed = 0;
  c->device_result = status;
  return status;
}
