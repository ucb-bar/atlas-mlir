#ifndef BF16_TEST_PLATFORM_H
#define BF16_TEST_PLATFORM_H
#include "atlas_bf16_call.h"
int test_platform_initialize(atlas_bf16_call *, const atlas_bf16_program *, void *, size_t);
void test_platform_mode(unsigned mode);
void test_platform_response(const uint8_t *, size_t);
unsigned test_platform_starts(void);
unsigned test_platform_effects(void);
int test_platform_loaded(const uint32_t *, size_t);
const uint8_t *test_platform_input(void);
#endif
