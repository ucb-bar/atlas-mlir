#ifndef ATLAS_RESNET_HOST_H
#define ATLAS_RESNET_HOST_H
#include <stddef.h>
#include <stdint.h>
int atlas_resnet_pack_a(const float *, unsigned, unsigned, unsigned, unsigned, uint8_t *, size_t, int *);
int atlas_resnet_decode_output(const uint8_t *, size_t, float *, unsigned, unsigned, unsigned);
int atlas_resnet_affine(const float *, float *, const float *, const float *, unsigned, unsigned, unsigned, int);
int atlas_resnet_residual_relu(const float *, const float *, float *, size_t);
#endif
