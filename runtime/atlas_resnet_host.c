/* First-bottleneck compatibility API backed by the shared graph arithmetic.
 * Compile native checks and RV64 with -ffp-contract=off.
 */
#include "atlas_resnet_graph_math.h"
#include "atlas_resnet_host.h"

int atlas_resnet_pack_a(const float *input, unsigned channels, unsigned h, unsigned w, unsigned kernel, uint8_t *output,
                        size_t capacity, int *scale) {
  if (kernel != 1 && kernel != 3) return 1;
  return atlas_graph_pack(input, channels, h, w, kernel, 1, kernel / 2, output, capacity, scale);
}

int atlas_resnet_decode_output(const uint8_t *physical, size_t bytes, float *output, unsigned channels, unsigned h,
                               unsigned w) {
  return atlas_graph_decode(physical, bytes, output, channels, h, w);
}

int atlas_resnet_affine(const float *input, float *output, const float *scale, const float *bias, unsigned channels,
                        unsigned h, unsigned w, int relu) {
  if (!h || !w || h > 224 || w > 224) return 1;
  size_t spatial = (size_t)h * w;
  if (atlas_graph_affine(input, output, scale, bias, channels, spatial)) return 1;
  return relu ? atlas_graph_relu(output, output, (size_t)channels * spatial) : 0;
}

int atlas_resnet_residual_relu(const float *branch, const float *identity, float *output, size_t count) {
  if (atlas_graph_add(branch, identity, output, count)) return 1;
  return atlas_graph_relu(output, output, count);
}
