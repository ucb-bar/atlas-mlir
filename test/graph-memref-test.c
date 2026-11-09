#include "atlas_graph_memref.h"
#include "atlas_bf16_call.h"

#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

enum { BASE = 0x90000000u, MEMORY_BYTES = 4096 };

typedef struct {
  uint8_t memory[MEMORY_BYTES];
  unsigned state;
  unsigned starts;
  unsigned writes;
  unsigned reads;
  unsigned mmio_reads;
  unsigned imem_writes;
  int fail_read;
  unsigned fail_read_at;
  int stall;
  unsigned cache_before;
  unsigned cache_after;
  int fail_cache_after;
  uint32_t loaded_words[2];
} fake_device;

static uint32_t read32(void *context, uintptr_t address) {
  fake_device *device = context;
  ++device->mmio_reads;
  if (address != 0x2008u) return 0;
  if (device->stall && device->state) return 0;
  if (device->state == 1u) {
    device->state = 2u;
    return 0;
  }
  if (device->state == 2u) device->state = 0u;
  return 5u; /* halted, ECALL */
}

static void write32(void *context, uintptr_t address, uint32_t value) {
  fake_device *device = context;
  if (address >= 0x1000u && address < 0x1008u) {
    ++device->imem_writes;
    device->loaded_words[(address - 0x1000u) / 4] = value;
    return;
  }
  if (address != 0x2018u || value != 1u) return;
  ++device->starts;
  device->state = 1u;
  const uint16_t first[] = {0x3f00u, 0x3fc0u, 0x3f80u, 0x4000u};
  const uint16_t second[] = {0x4040u, 0x40a0u, 0x4080u, 0x40c0u};
  memcpy(device->memory + 0x400, first, sizeof first);
  memcpy(device->memory + 0x500, second, sizeof second);
}

static int write_memory(void *context, uintptr_t address,
                        const uint8_t *source, size_t bytes) {
  fake_device *device = context;
  if (address < BASE || address - BASE > MEMORY_BYTES ||
      bytes > MEMORY_BYTES - (address - BASE)) return 1;
  memcpy(device->memory + address - BASE, source, bytes);
  ++device->writes;
  return 0;
}

static int read_memory(void *context, uintptr_t address,
                       uint8_t *destination, size_t bytes) {
  fake_device *device = context;
  if (device->fail_read || (device->fail_read_at && device->reads + 1 == device->fail_read_at) || address < BASE || address - BASE > MEMORY_BYTES ||
      bytes > MEMORY_BYTES - (address - BASE)) return 1;
  memcpy(destination, device->memory + address - BASE, bytes);
  ++device->reads;
  return 0;
}

static unsigned allocation_calls;
void *__real_malloc(size_t bytes);
void *__wrap_malloc(size_t bytes) {
  ++allocation_calls;
  return __real_malloc(bytes);
}

static int cache_sync(void *context, const atlas_host_buffer *buffer, int before) {
  fake_device *device = context;
  assert(buffer->size_bytes > 0);
  if (before) ++device->cache_before;
  else {
    ++device->cache_after;
    if (device->fail_cache_after) return 1;
  }
  return 0;
}

int main(void) {
  static const uint8_t codes[] = {0x30u, 0x38u};
  static const uint32_t midpoints[] = {0x3f400000u};
  static const merlin_f32_u8_codebook codebook = {
      codes, midpoints, 2u, 0x3e800000u,
      {1u, (1u << 16) | (1u << 24), 0, 0, 0, 0, 0, 0}, 0u};
  static const uint32_t program[] = {0x00000013u, 0x00000073u};
  static const atlas_graph_input_slot input_slots[] = {
      {BASE, 2, 2, 4, &codebook}, {BASE + 0x100, 2, 2, 4, &codebook}};
  static const uint8_t constant[] = {0x80u, 0x3fu};
  static const atlas_graph_constant_slot constant_slots[] = {
      {BASE + 0x200, constant, sizeof constant}};
  static const atlas_graph_output_slot output_slots[] = {
      {BASE + 0x400, 2, 2, 1, 8}, {BASE + 0x500, 2, 2, 1, 8}};
  fake_device device = {0};
  atlas_host host;
  atlas_host_map map = {0x1000u, 0x2000u, 1024u};
  atlas_host_mmio mmio = {read32, write32, &device};
  atlas_host_memory memory = {1, NULL, NULL};
  assert(atlas_host_init(&host, map, mmio, memory) == ATLAS_HOST_OK);
  atlas_host_buffer active[5];
  atlas_graph_memref_binding binding = {
      .host = &host,
      .io = {write_memory, read_memory, &device},
      .program_words = program, .program_word_count = 2,
      .inputs = input_slots, .input_count = 2,
      .constants = constant_slots, .constant_count = 1,
      .outputs = output_slots, .output_count = 2,
      .max_polls = 4, .active_buffers = active, .active_buffer_capacity = 5};
  float first_source[] = {0.5f, 1.0f, 0.5f, 1.0f};
  float second_source[] = {0.0f, 1.0f, 1.0f, 0.0f};
  atlas_f32_memref2 first = {first_source, first_source, 0, {2, 2}, {2, 1}};
  atlas_f32_memref2 second = {second_source, second_source, 0, {2, 2}, {2, 1}};
  const atlas_f32_memref2 *arguments[] = {&first, &second};
  atlas_f32_memref2 results[2] = {0};
  atlas_host_result device_result = ATLAS_HOST_INVALID;
  assert(atlas_graph_run_memrefs(&binding, arguments, 2, results, 2,
                                 &device_result) == ATLAS_MODEL_TILE_OK);
  assert(device_result == ATLAS_HOST_OK && device.starts == 1u);
  assert(device.writes == 3u && device.reads == 2u);
  assert(memcmp(device.memory, "\x30\x38\x30\x38", 4) == 0);
  assert(memcmp(device.memory + 0x100, "\x00\x38\x38\x00", 4) == 0);
  assert(memcmp(device.memory + 0x200, constant, sizeof constant) == 0);
  const float expected_first[] = {0.5f, 1.0f, 1.5f, 2.0f};
  const float expected_second[] = {3.0f, 4.0f, 5.0f, 6.0f};
  assert(memcmp(results[0].aligned, expected_first, sizeof expected_first) == 0);
  assert(memcmp(results[1].aligned, expected_second, sizeof expected_second) == 0);
  free(results[0].allocated);
  free(results[1].allocated);

  atlas_f32_memref2 unchanged[2] = {{0}, {0}};
  unchanged[0].offset = 17;
  unchanged[1].offset = 23;
  unsigned before = device.starts;
  first_source[0] = NAN;
  assert(atlas_graph_run_memrefs(&binding, arguments, 2, unchanged, 2,
                                 &device_result) == ATLAS_MODEL_TILE_INPUT_DOMAIN);
  assert(device.starts == before && unchanged[0].offset == 17 &&
         unchanged[1].offset == 23);
  first_source[0] = 0.5f;

  device.fail_read = 1;
  assert(atlas_graph_run_memrefs(&binding, arguments, 2, unchanged, 2,
                                 &device_result) == ATLAS_MODEL_TILE_MEMORY_ERROR);
  assert(unchanged[0].offset == 17 && unchanged[1].offset == 23);
  device.fail_read = 0;

  atlas_graph_output_slot overlapped[] = {output_slots[0], output_slots[1]};
  overlapped[0].address = input_slots[0].address;
  binding.outputs = overlapped;
  before = device.starts;
  assert(atlas_graph_run_memrefs(&binding, arguments, 2, unchanged, 2,
                                 &device_result) == ATLAS_MODEL_TILE_INVALID);
  assert(device.starts == before);

  /* The bounded pointer route shares the actual graph implementation but
   * uses only caller storage. All sizes/alignment come from this C compiler. */
  binding.outputs = output_slots;
  host.memory = (atlas_host_memory){0, cache_sync, &device};
  size_t required, alignment, state_bytes, state_alignment;
  assert(atlas_graph_pointer_requirements(&binding, &required, &alignment,
                                          &state_bytes, &state_alignment) == ATLAS_MODEL_TILE_OK);
  assert(required == 48 && alignment == _Alignof(float));
  assert(state_bytes == sizeof(atlas_graph_pointer_context) + sizeof(binding) +
                        sizeof(host) + sizeof(active));
  assert(state_alignment >= _Alignof(atlas_graph_pointer_context));
  _Alignas(float) unsigned char workspace[64];
  atlas_graph_pointer_context context = {0};
  assert(atlas_graph_pointer_init(&context, &binding, workspace, required - 1,
                                  state_bytes) == ATLAS_MODEL_TILE_INVALID);
  assert(atlas_graph_pointer_init(&context, &binding, workspace, sizeof(workspace),
                                  state_bytes - 1) == ATLAS_MODEL_TILE_INVALID);
  assert(context.binding == NULL);
  assert(atlas_graph_pointer_init(&context, &binding, workspace, sizeof(workspace),
                                  state_bytes) == ATLAS_MODEL_TILE_OK);
  const void *dense_inputs[] = {first_source, second_source};
  float first_output[4] = {-9, -9, -9, -9};
  float second_output[4] = {-9, -9, -9, -9};
  void *dense_outputs[] = {first_output, second_output};
  unsigned allocations_before = allocation_calls;
  unsigned cache_before = device.cache_before, cache_after = device.cache_after;
  assert(atlas_graph_run_pointers(&context, dense_inputs, dense_outputs) == 0);
  assert(allocation_calls == allocations_before);
  assert(context.device_result == ATLAS_HOST_OK && !host.active);
  assert(device.cache_before == cache_before + 5 && device.cache_after == cache_after + 2);
  assert(memcmp(device.loaded_words, program, sizeof(program)) == 0);
  assert(memcmp(first_output, expected_first, sizeof(first_output)) == 0);
  assert(memcmp(second_output, expected_second, sizeof(second_output)) == 0);
  assert(first_source[0] == 0.5f && second_source[0] == 0.0f);

  /* Input-domain refusal is before any start/write/cache effect. */
  first_source[0] = NAN;
  before = device.starts;
  unsigned writes_before = device.writes;
  cache_before = device.cache_before;
  assert(atlas_graph_run_pointers(&context, dense_inputs, dense_outputs) == ATLAS_MODEL_TILE_INPUT_DOMAIN);
  assert(device.starts == before && device.writes == writes_before && device.cache_before == cache_before);
  first_source[0] = 0.5f;
  const float sentinel[] = {-9, -9, -9, -9};
  memcpy(first_output, sentinel, sizeof sentinel);
  memcpy(second_output, sentinel, sizeof sentinel);

  /* A second read failure cannot publish the already-decoded first output. */
  device.fail_read_at = device.reads + 2;
  assert(atlas_graph_run_pointers(&context, dense_inputs, dense_outputs) == ATLAS_MODEL_TILE_MEMORY_ERROR);
  assert(memcmp(first_output, sentinel, sizeof sentinel) == 0);
  assert(memcmp(second_output, sentinel, sizeof sentinel) == 0);
  device.fail_read_at = 0;

  /* Timeout retains actual active descriptors/context. Reentry cannot mutate
   * them, reuse the workspace or launch again. Explicit drain completes it. */
  device.stall = 1;
  assert(atlas_graph_run_pointers(&context, dense_inputs, dense_outputs) == ATLAS_MODEL_TILE_DEVICE_ERROR);
  assert(context.device_result == ATLAS_HOST_TIMEOUT && host.active);
  assert(host.active_buffers == active && host.active_buffer_count == 5);
  atlas_host_buffer saved_active[5];
  memcpy(saved_active, active, sizeof active);
  unsigned char saved_workspace[sizeof workspace];
  memcpy(saved_workspace, workspace, sizeof workspace);
  before = device.starts;
  assert(atlas_graph_run_pointers(&context, dense_inputs, dense_outputs) == ATLAS_MODEL_TILE_DEVICE_ERROR);
  assert(context.device_result == ATLAS_HOST_BUSY && device.starts == before);
  assert(memcmp(saved_active, active, sizeof active) == 0);
  assert(memcmp(saved_workspace, workspace, sizeof workspace) == 0);
  assert(memcmp(first_output, sentinel, sizeof sentinel) == 0);
  device.stall = 0;
  assert(atlas_host_wait(&host, 4) == ATLAS_HOST_OK && !host.active);

  /* Cache completion failure is also retained and produces no output. */
  device.fail_cache_after = 1;
  assert(atlas_graph_run_pointers(&context, dense_inputs, dense_outputs) == ATLAS_MODEL_TILE_DEVICE_ERROR);
  assert(context.device_result == ATLAS_HOST_CACHE_ERROR && host.active);
  assert(memcmp(first_output, sentinel, sizeof sentinel) == 0);
  device.fail_cache_after = 0;
  assert(atlas_host_wait(&host, 4) == ATLAS_HOST_OK && !host.active);
  void *aliased_outputs[] = {first_source, second_output};
  before = device.starts;
  assert(atlas_graph_run_pointers(&context, dense_inputs, aliased_outputs) == ATLAS_MODEL_TILE_INVALID);
  assert(device.starts == before && allocation_calls == allocations_before);

  /* Raw BF16 transport never widens values to float. This fake endpoint checks
   * codec/domain/lifetime obligations, not compiled kernel arithmetic. */
  atlas_bf16_program raw_program = {program, 2, 2, 2, 1, 8, BASE, BASE + 0x400, 2};
  atlas_bf16_call raw = {0};
  unsigned char raw_workspace[8];
  atlas_host_memory raw_memory = {0, cache_sync, &device};
  atlas_tile_memory_io raw_io = {write_memory, read_memory, &device};
  before = device.mmio_reads;
  assert(atlas_bf16_call_init(&raw, &raw_program, map, mmio, raw_memory, raw_io,
                             4, raw_workspace, sizeof raw_workspace) == ATLAS_MODEL_TILE_OK);
  assert(device.mmio_reads == before);
  uint16_t raw_input[] = {0x8000, 0x0000, 0x3f80, 0xbf80};
  uint16_t raw_output[] = {0x5a5a, 0x5a5a, 0x5a5a, 0x5a5a};
  const uint16_t raw_expected[] = {0x3f00, 0x3f80, 0x3fc0, 0x4000};
  const uint16_t raw_input_snapshot[] = {0x8000, 0x0000, 0x3f80, 0xbf80};
  const void *raw_inputs[] = {raw_input};
  void *raw_outputs[] = {raw_output};
  assert(atlas_bf16_call_run(&raw, raw_inputs, raw_outputs) == 0);
  assert(memcmp(raw_output, raw_expected, sizeof raw_output) == 0);
  assert(memcmp(raw_input, raw_input_snapshot, sizeof raw_input) == 0);
  const unsigned char packed_bits[] = {0x00,0x80,0x80,0x3f,0x00,0x00,0x80,0xbf};
  assert(memcmp(device.memory, packed_bits, sizeof packed_bits) == 0);
  const uint16_t forbidden[] = {0x0001, 0x8001, 0x7f80, 0xff80, 0x7fc1, 0xff81};
  for (size_t i = 0; i < sizeof forbidden / sizeof forbidden[0]; ++i) {
    raw_input[3] = forbidden[i];
    before = device.mmio_reads;
    unsigned saved_writes = device.writes + device.imem_writes + device.cache_before;
    assert(atlas_bf16_call_run(&raw, raw_inputs, raw_outputs) == ATLAS_MODEL_TILE_INPUT_DOMAIN);
    assert(device.mmio_reads == before);
    assert(device.writes + device.imem_writes + device.cache_before == saved_writes);
    assert(memcmp(raw_output, raw_expected, sizeof raw_output) == 0);
  }
  raw_input[3] = raw_input_snapshot[3];
  void *raw_alias[] = {raw_input};
  before = device.mmio_reads;
  assert(atlas_bf16_call_run(&raw, raw_inputs, raw_alias) == ATLAS_MODEL_TILE_INVALID);
  const void *raw_overflow[] = {(void *)(UINTPTR_MAX - 3)};
  assert(atlas_bf16_call_run(&raw, raw_overflow, raw_outputs) == ATLAS_MODEL_TILE_INVALID);
  assert(device.mmio_reads == before);
  device.stall = 1;
  assert(atlas_bf16_call_run(&raw, raw_inputs, raw_outputs) == ATLAS_MODEL_TILE_DEVICE_ERROR);
  assert(raw.failed && raw.host.active && raw.host.active_buffers == raw.active_buffers);
  assert(raw.device_result == ATLAS_HOST_TIMEOUT);
  before = device.mmio_reads;
  assert(atlas_bf16_call_run(&raw, raw_inputs, raw_outputs) == ATLAS_MODEL_TILE_DEVICE_ERROR);
  assert(device.mmio_reads == before && memcmp(raw_output, raw_expected, sizeof raw_output) == 0);
  device.stall = 0;
  assert(atlas_bf16_call_drain(&raw, 4) == ATLAS_HOST_OK && !raw.failed && !raw.host.active);
  device.fail_read = 1;
  assert(atlas_bf16_call_run(&raw, raw_inputs, raw_outputs) == ATLAS_MODEL_TILE_MEMORY_ERROR);
  assert(raw.failed && memcmp(raw_output, raw_expected, sizeof raw_output) == 0);
  device.fail_read = 0;
  assert(atlas_bf16_call_drain(&raw, 4) == ATLAS_HOST_OK);
  device.fail_cache_after = 1;
  assert(atlas_bf16_call_run(&raw, raw_inputs, raw_outputs) == ATLAS_MODEL_TILE_DEVICE_ERROR);
  assert(raw.failed && raw.host.active && raw.device_result == ATLAS_HOST_CACHE_ERROR);
  device.fail_cache_after = 0;
  assert(atlas_bf16_call_drain(&raw, 4) == ATLAS_HOST_OK);
  assert(allocation_calls == allocations_before);
  return 0;
}
