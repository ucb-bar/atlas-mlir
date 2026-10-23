#ifndef TINY_FULL_MODEL_H
#define TINY_FULL_MODEL_H
#include <stddef.h>
#include <stdint.h>

#define TINY_FULL_LAYERS 22
#define TINY_FULL_CACHE_WORDS (4 * 160 * 64)

/* All allocation extents are caller-owned and mutually disjoint. Layer KV
 * entries are compact [4,position,64] within each fixed maximum-size slot.
 * Double buffering preserves the last complete state if a step fails. */
typedef struct {
    const uint16_t *embedding;
    const uint16_t *input_norm[TINY_FULL_LAYERS];
    const uint16_t *post_norm[TINY_FULL_LAYERS];
    const uint16_t *final_norm, *cosine, *sine;
    float epsilon;
} tiny_full_parameters;

typedef struct {
    uint16_t *keys[2], *values[2];
    size_t position;
    unsigned bank;
    uint32_t next_token;
} tiny_full_state;

/* Layers 0..21 use logical projections 0..6 in the original block order.
 * Layer22/logical7 is the complete LM head, width32000. Callback returns
 * every BF16 row-major word produced by the actual preceding host stage. */
typedef int (*tiny_full_projection)(void *, unsigned, unsigned,
                                    const uint16_t *, size_t, size_t, size_t, uint16_t *);
size_t tiny_full_workspace_bytes(void);
int tiny_full_argmax_bf16(const uint16_t *logits, size_t count, uint32_t *token);
int tiny_full_model_step(const tiny_full_parameters *, tiny_full_state *,
                         const uint32_t *tokens, size_t rows,
                         tiny_full_projection, void *, void *workspace, size_t workspace_bytes,
                         uint16_t *logits, size_t logits_words);
#endif
