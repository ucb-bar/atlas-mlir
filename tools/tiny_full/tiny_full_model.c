#include "tiny_full_model.h"
#include "tiny_block.h"
#include "llama_host_attention.h"
#include "smol_host_ops.h"
#include <math.h>
#include <stdint.h>
#include <string.h>

typedef struct {
    uint16_t normalized[128 * 2048], query[128 * 2048];
    uint16_t key[128 * 256], value[128 * 256];
    uint16_t attended[128 * 2048], projected[128 * 2048], residual[128 * 2048];
    uint16_t post_norm[128 * 2048], gate[128 * 5632], up[128 * 5632];
    uint16_t activated[128 * 5632], down[128 * 2048];
    uint16_t attention_scratch[3014656];
    float scores[32 * 128 * 128], softmax[160];
    int64_t positions[128];
} tiny_full_block_workspace;

static size_t block_workspace_bytes(void) {
    return sizeof(tiny_full_block_workspace);
}

static int full_block_run(size_t rows, size_t past_rows, const uint16_t *entry,
                   const uint16_t *iw, const uint16_t *pw,
                   float epsilon, const uint16_t *cosine, const uint16_t *sine,
                   const uint16_t *past_keys, const uint16_t *past_values,
                   tiny_projection project, void *context,
                   void *workspace, size_t workspace_bytes,
                   uint16_t *keys, uint16_t *values, uint16_t *output) {
    if ((rows != 128 && rows != 1) || !entry || !iw || !pw || !cosine || !sine
        || !project || !workspace || workspace_bytes < sizeof(tiny_full_block_workspace)
        || (uintptr_t) workspace % _Alignof(tiny_full_block_workspace)
        || !keys || !values || !output || !isfinite(epsilon) || epsilon <= 0
        || (rows == 1 && (!past_rows || past_rows >= 160 || !past_keys || !past_values))
        || (rows == 128 && (past_rows || past_keys || past_values))) return -1;
    /* Fixed driver owns distinct allocation extents; no dynamic allocation or
     * hidden tensor state. A failed projection/host stage aborts this block. */
    tiny_full_block_workspace *w = workspace;
    for (size_t i = 0; i < rows; ++i) w->positions[i] = (int64_t)(past_rows + i);
    if (smol_host_rmsnorm_bf16(entry, iw, epsilon, w->normalized, rows, 2048)) return -2;
    if (project(context, 0, w->normalized, rows, 2048, 2048, w->query)
        || project(context, 1, w->normalized, rows, 2048, 256, w->key)
        || project(context, 2, w->normalized, rows, 2048, 256, w->value)) return -3;
    llama_host_attention_inputs a = {32, 4, 64, rows, past_rows, 160, 0.125f,
        w->query, w->key, w->value, past_keys, past_values, cosine, sine, w->positions};
    llama_host_attention_buffers b = {w->attention_scratch, 3014656, w->scores, 32 * 128 * 128,
        w->softmax, 160, keys, values, w->attended, 4 * 160 * 64, 128 * 2048};
    if (llama_host_attention_bf16(&a, &b)) return -4;
    if (project(context, 3, w->attended, rows, 2048, 2048, w->projected)) return -5;
    if (smol_host_residual_bf16(entry, w->projected, w->residual, rows * 2048)
        || smol_host_rmsnorm_bf16(w->residual, pw, epsilon, w->post_norm, rows, 2048)) return -6;
    if (project(context, 4, w->post_norm, rows, 2048, 5632, w->gate)
        || project(context, 5, w->post_norm, rows, 2048, 5632, w->up)) return -7;
    if (smol_host_silu_mul_bf16(w->gate, w->up, w->activated, rows * 5632)) return -8;
    if (project(context, 6, w->activated, rows, 5632, 2048, w->down)) return -9;
    return smol_host_residual_bf16(w->residual, w->down, output, rows * 2048) ? -10 : 0;
}

typedef struct {
    tiny_full_block_workspace block;
    uint16_t hidden[2][128 * 2048], normalized[128 * 2048];
} tiny_full_workspace;

typedef struct {
    tiny_full_projection callback;
    void *context;
    unsigned layer;
} projection_bridge;

static int project_layer(void *opaque, unsigned logical, const uint16_t *input,
                         size_t rows, size_t k, size_t width, uint16_t *output) {
    projection_bridge *bridge = opaque;
    return bridge->callback(bridge->context, bridge->layer, logical, input, rows, k, width, output);
}

size_t tiny_full_workspace_bytes(void) {
    return sizeof(tiny_full_workspace);
}

int tiny_full_argmax_bf16(const uint16_t *logits, size_t count, uint32_t *token) {
    if (!logits || !token || count != 32000) return -1;
    float maximum = 0;
    uint32_t best = 0;
    for (size_t i = 0; i < count; ++i) {
        if ((logits[i] & UINT16_C(0x7f80)) == UINT16_C(0x7f80)) return -1;
        uint32_t bits = (uint32_t)logits[i] << 16;
        float value;
        memcpy(&value, &bits, sizeof(value));
        if (!i || value > maximum) {
            maximum = value;
            best = (uint32_t)i;
        }
    }
    *token = best;
    return 0;
}

_Static_assert(sizeof(tiny_full_phase) == 32, "Tiny continuation ABI size");
_Static_assert(sizeof(tiny_full_live_layout) == 64, "Tiny layout ABI size");

int tiny_full_get_live_layout(tiny_full_live_layout *layout) {
    if (!layout) return -1;
    *layout = (tiny_full_live_layout){1, sizeof(tiny_full_workspace),
        {offsetof(tiny_full_workspace, hidden[0]), offsetof(tiny_full_workspace, hidden[1])},
        sizeof(((tiny_full_workspace *)0)->hidden[0]),
        TINY_FULL_LAYERS * TINY_FULL_CACHE_WORDS * sizeof(uint16_t),
        sizeof(tiny_full_phase), sizeof(tiny_full_state)};
    return 0;
}

static int valid_context(const tiny_full_parameters *p, const tiny_full_state *s,
                         void *workspace, size_t workspace_bytes) {
    if (!p || !s || !workspace || workspace_bytes < sizeof(tiny_full_workspace)
        || (uintptr_t)workspace % _Alignof(tiny_full_workspace)
        || s->bank > 1 || s->position > 159 || (s->position && s->position < 128)
        || !p->embedding || !p->final_norm || !p->cosine || !p->sine
        || !isfinite(p->epsilon) || p->epsilon <= 0
        || !s->keys[0] || !s->keys[1] || !s->values[0] || !s->values[1]) return 0;
    for (size_t layer = 0; layer < TINY_FULL_LAYERS; ++layer)
        if (!p->input_norm[layer] || !p->post_norm[layer]) return 0;
    return 1;
}

static int valid_phase(const tiny_full_state *s, const tiny_full_phase *phase) {
    return phase && phase->active == 1 && !phase->reserved
        && phase->position == s->position && phase->bank == s->bank && phase->next_token == s->next_token
        && phase->rows == (s->position ? 1u : 128u) && phase->next_layer <= TINY_FULL_LAYERS;
}

int tiny_full_model_begin(const tiny_full_parameters *p, const tiny_full_state *s, tiny_full_phase *phase,
                         const uint32_t *tokens, size_t rows, void *workspace, size_t workspace_bytes) {
    if (!valid_context(p, s, workspace, workspace_bytes) || !phase || phase->active || !tokens
        || (s->position == 0 ? rows != 128 : rows != 1)) return -1;
    for (size_t i = 0; i < rows; ++i)
        if (tokens[i] >= 32000 || (rows == 1 && tokens[i] != s->next_token)) return -1;
    tiny_full_workspace *w = workspace;
    for (size_t i = 0; i < rows; ++i)
        memcpy(w->hidden[0] + i * 2048, p->embedding + (size_t)tokens[i] * 2048, 2048 * 2);
    *phase = (tiny_full_phase){s->position, s->bank, s->next_token, 0, (uint32_t)rows, 1, 0};
    return 0;
}

int tiny_full_model_layer(const tiny_full_parameters *p, const tiny_full_state *s, tiny_full_phase *phase,
                         unsigned layer, tiny_full_projection project, void *context,
                         void *workspace, size_t workspace_bytes) {
    if (!valid_context(p, s, workspace, workspace_bytes) || !valid_phase(s, phase) || !project
        || layer >= TINY_FULL_LAYERS || phase->next_layer != layer) return -1;
    tiny_full_workspace *w = workspace;
    projection_bridge bridge = {project, context, layer};
    size_t offset = (size_t)layer * TINY_FULL_CACHE_WORDS;
    int status = full_block_run(phase->rows, s->position, w->hidden[layer % 2],
        p->input_norm[layer], p->post_norm[layer], p->epsilon, p->cosine, p->sine,
        s->position ? s->keys[s->bank] + offset : NULL,
        s->position ? s->values[s->bank] + offset : NULL,
        project_layer, &bridge, &w->block, block_workspace_bytes(),
        s->keys[1 - s->bank] + offset, s->values[1 - s->bank] + offset, w->hidden[1 - layer % 2]);
    if (status) { phase->active = 2; return -2; }
    ++phase->next_layer;
    return 0;
}

int tiny_full_model_finish(const tiny_full_parameters *p, tiny_full_state *s, tiny_full_phase *phase,
                          tiny_full_projection project, void *context, void *workspace, size_t workspace_bytes,
                          uint16_t *logits, size_t logits_words) {
    if (!valid_context(p, s, workspace, workspace_bytes) || !valid_phase(s, phase) || !project || !logits
        || phase->next_layer != TINY_FULL_LAYERS || logits_words < (size_t)phase->rows * 32000) return -1;
    tiny_full_workspace *w = workspace;
    size_t rows = phase->rows;
    if (smol_host_rmsnorm_bf16(w->hidden[0], p->final_norm, p->epsilon,
                             w->normalized, rows, 2048)) { phase->active = 2; return -3; }
    if (project(context, 22, 7, w->normalized, rows, 2048, 32000, logits)) { phase->active = 2; return -4; }
    uint32_t next_token;
    if (tiny_full_argmax_bf16(logits + (rows - 1) * 32000, 32000, &next_token)) {
        phase->active = 2; return -5;
    }
    s->bank = 1 - s->bank;
    s->position += rows;
    s->next_token = next_token;
    phase->active = 0;
    return 0;
}

int tiny_full_model_step(const tiny_full_parameters *p, tiny_full_state *s,
                         const uint32_t *tokens, size_t rows, tiny_full_projection project, void *context,
                         void *workspace, size_t workspace_bytes, uint16_t *logits, size_t logits_words) {
    tiny_full_phase phase = {0};
    if (!project || !logits || logits_words < rows * 32000
        || tiny_full_model_begin(p, s, &phase, tokens, rows, workspace, workspace_bytes)) return -1;
    for (unsigned layer = 0; layer < TINY_FULL_LAYERS; ++layer)
        if (tiny_full_model_layer(p, s, &phase, layer, project, context, workspace, workspace_bytes)) return -2;
    return tiny_full_model_finish(p, s, &phase, project, context, workspace, workspace_bytes, logits, logits_words);
}
