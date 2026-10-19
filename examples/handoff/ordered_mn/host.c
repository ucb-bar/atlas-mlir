/* Diagnostic Shuttle host for one verified Atlas tile program reused over
 * a source M/N output grid. The generated header supplies source geometry. */
#include "atlas_ordered_mn_data.h"

#include <stddef.h>
#include <stdint.h>

void *memset(void *destination, int value, size_t count) {
  unsigned char *bytes = destination;
  for (size_t index = 0; index < count; ++index)
    bytes[index] = (unsigned char)value;
  return destination;
}

enum {
  ACTIVATION_BYTES = ATLAS_ORDERED_MN_ROWS * ATLAS_ORDERED_MN_REDUCTION,
  WEIGHT_BYTES = ATLAS_ORDERED_MN_COLUMNS * ATLAS_ORDERED_MN_REDUCTION,
  OUTPUT_BYTES = 2u * ATLAS_ORDERED_MN_ROWS * ATLAS_ORDERED_MN_COLUMNS,
};

static atlas_host host;
static atlas_ordered_k_tile tile;
static atlas_host_buffer active_buffers[ATLAS_ORDERED_K_WINDOW_COUNT + 1u];
static uint8_t activation[ACTIVATION_BYTES], weight[WEIGHT_BYTES];
static uint8_t output[OUTPUT_BYTES], staging[OUTPUT_BYTES];
static uint8_t physical_tile[2048], scratch[2048];
#ifdef ATLAS_ORDERED_MN_HAS_TAILS
static uint8_t activation_tile[32u * ATLAS_ORDERED_MN_REDUCTION];
static uint8_t weight_tile[32u * ATLAS_ORDERED_MN_REDUCTION];
#endif

volatile uint32_t atlas_matrix_status_word;
volatile uint32_t atlas_matrix_device_word;
volatile uint8_t atlas_matrix_outputs[2][OUTPUT_BYTES];

static int direct_write(void *context, uintptr_t address,
                        const uint8_t *source, size_t bytes) {
  (void)context;
  if (!address || !source || !bytes)
    return 1;
  volatile uint8_t *destination = (volatile uint8_t *)address;
  for (size_t index = 0; index < bytes; ++index)
    destination[index] = source[index];
  __asm__ volatile("fence rw, rw" ::: "memory");
  return 0;
}

static int direct_read(void *context, uintptr_t address,
                       uint8_t *destination, size_t bytes) {
  (void)context;
  if (!address || !destination || !bytes)
    return 1;
  __asm__ volatile("fence rw, rw" ::: "memory");
  volatile const uint8_t *source = (volatile const uint8_t *)address;
  for (size_t index = 0; index < bytes; ++index)
    destination[index] = source[index];
  return 0;
}

/* Public values are finite E4M3 normal numbers or positive zero. They are
 * generated at runtime and are not compiler inputs or expected outputs. */
static void prepare_inputs(unsigned trial) {
  static const uint8_t codes[] = {0x00, 0x30, 0xb0, 0x38, 0xb8, 0x40, 0xc0};
  for (size_t row = 0; row < ATLAS_ORDERED_MN_ROWS; ++row)
    for (size_t k = 0; k < ATLAS_ORDERED_MN_REDUCTION; ++k)
      activation[row * ATLAS_ORDERED_MN_REDUCTION + k] =
          codes[(row + k + trial) % sizeof codes];
  for (size_t row = 0; row < ATLAS_ORDERED_MN_COLUMNS; ++row)
    for (size_t k = 0; k < ATLAS_ORDERED_MN_REDUCTION; ++k)
      weight[row * ATLAS_ORDERED_MN_REDUCTION + k] =
          codes[(2u * row + k + trial) % sizeof codes];
}

int main(void) {
  atlas_host_mmio mmio = {atlas_host_direct_read32, atlas_host_direct_write32, 0};
  atlas_host_memory memory = {1, 0, 0}; /* Diagnostic coherent-memory premise. */
  atlas_host_result device = atlas_host_init(&host, atlas_ordered_k_host_map,
                                             mmio, memory);
  atlas_matrix_device_word = (uint32_t)device;
  if (device != ATLAS_HOST_OK)
    return atlas_matrix_status_word = 0x100u + (uint32_t)device;
  if (atlas_ordered_k_bind(&tile, &host, 0, direct_write, direct_read,
                           10000000u, scratch, sizeof scratch,
                           active_buffers, ATLAS_ORDERED_K_WINDOW_COUNT + 1u))
    return atlas_matrix_status_word = 0x200u;

  for (unsigned trial = 0; trial < 2; ++trial) {
    prepare_inputs(trial);
    atlas_ordered_k_status status;
#ifdef ATLAS_ORDERED_MN_HAS_TAILS
    status = atlas_ordered_mn_run_tails(
        &tile, activation, sizeof activation, weight, sizeof weight,
        ATLAS_ORDERED_MN_ROWS, ATLAS_ORDERED_MN_COLUMNS,
        ATLAS_ORDERED_MN_REDUCTION, output, sizeof output,
        staging, sizeof staging, physical_tile, sizeof physical_tile,
        activation_tile, sizeof activation_tile,
        weight_tile, sizeof weight_tile, &device);
#else
    status = atlas_ordered_mn_run(
        &tile, activation, sizeof activation, weight, sizeof weight,
        ATLAS_ORDERED_MN_ROWS, ATLAS_ORDERED_MN_COLUMNS,
        ATLAS_ORDERED_MN_REDUCTION, output, sizeof output,
        staging, sizeof staging, physical_tile, sizeof physical_tile,
        &device);
#endif
    atlas_matrix_device_word = (uint32_t)device;
    if (status != ATLAS_ORDERED_K_OK)
      return atlas_matrix_status_word = 0x300u + 16u * trial + (uint32_t)status;
    for (size_t byte = 0; byte < OUTPUT_BYTES; ++byte)
      atlas_matrix_outputs[trial][byte] = output[byte];
  }
  return atlas_matrix_status_word = 1u;
}
