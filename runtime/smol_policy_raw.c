#include "smol_policy_raw.h"
#include "host_matrix_fp8pack.h"
#include "smol_host_bits.h"
#include <math.h>

static int reject(smol_policy_raw_context *w, int result) {
    w->failed = 1;
    w->result = result;
    return result;
}

static uint16_t physical(const volatile uint8_t *dram, size_t offset,
                         size_t row, size_t column, size_t columns) {
    size_t tile = (row / 32) * (columns / 32) + column / 32;
    size_t word = tile * 1024 + ((column % 32) / 16) * 512 + (row % 32) * 16 + column % 16;
    return (uint16_t) ((uint16_t) dram[offset + 2 * word]
                     | ((uint16_t) dram[offset + 2 * word + 1] << 8));
}

static float add_bias(uint16_t word, const smol_policy_call *call, size_t column) {
    float value = bf16_to_float(word);
    if (call->bias) value += call->bias_dtype == SMOL_BF16
        ? bf16_to_float(((const uint16_t *) call->bias)[column])
        : ((const float *) call->bias)[column];
    return value;
}

static int invoke(smol_policy_raw_context *w, unsigned site, unsigned layer, unsigned role,
    unsigned image, unsigned dtype, const void *input, size_t rows, size_t k,
    unsigned output_dtype, void *output, size_t n) {
    if (!w) return SMOL_RAW_INVALID;
    if (w->failed) return w->result;
    if (!w->host || w->host->active || !w->panels || !w->calls || !w->dram
        || !w->activation_copy || !w->input_copy || !w->cycle_delta || !w->max_polls
        || !input || !output || dtype > SMOL_F32 || output_dtype > SMOL_F32
        || w->call_cursor >= w->call_count || w->panel_cursor >= w->panel_count
        || !rows || !k || !n || rows > SIZE_MAX / k || rows * k > w->input_capacity
        || w->dram_capacity > 64 * 1024 * 1024) return reject(w, SMOL_RAW_INVALID);
    const smol_policy_call *call = &w->calls[w->call_cursor];
    if (call->rows != rows || call->input_width != k || call->output_width != n
        || call->site != site || call->layer != layer || call->role != role || call->image != image
        || call->input_dtype != dtype || call->output_dtype != output_dtype
        || call->bias_dtype > SMOL_F32 || call->first_panel != w->panel_cursor
        || !call->panel_count || call->panel_count > w->panel_count - w->panel_cursor
        || call->activation_scale < -126 || call->activation_scale > 127) return reject(w, SMOL_RAW_INVALID);
    /* One conversion and scale over the complete actual operator input. */
    for (size_t i = 0; i < rows * k; ++i) {
        float value = dtype == SMOL_BF16 ? bf16_to_float(((const uint16_t *) input)[i]) : ((const float *) input)[i];
        if (!isfinite(value)) return reject(w, SMOL_RAW_INPUT);
        w->input_copy[i] = float_to_bf16(value);
    }
    int scale;
    if (host_matrix_scale_bf16(w->input_copy, rows, k, &scale)) return reject(w, SMOL_RAW_INPUT);
    if (scale != call->activation_scale) return reject(w, SMOL_RAW_SCALE);
    size_t next_row = 0;
    for (size_t panel = 0; panel < call->panel_count; ++panel) {
        const smol_policy_panel *p = &w->panels[w->panel_cursor];
        size_t m = p->physical_rows, pk = p->physical_k, pn = p->physical_n;
        if (!m || !pk || !pn || m % 32 || pk % 32 || pn % 32 || m > 128
            || p->call != w->call_cursor || p->row_start != next_row || p->input_width != k
            || p->output_width != n || !p->logical_rows || p->logical_rows > rows - next_row
            || m != (p->logical_rows + 31) / 32 * 32 || pk != (k + 31) / 32 * 32
            || pn != (n + 31) / 32 * 32 || m > SIZE_MAX / pk || pk > SIZE_MAX / pn
            || m > SIZE_MAX / pn / 2 || p->weight_bytes != pk * pn || !p->weights
            || p->output_offset != m * pk + p->weight_bytes + 256
            || p->dram_bytes != p->output_offset + 2 * m * pn + 256
            || p->dram_bytes > w->dram_capacity || m * pk > w->activation_capacity
            || p->store_bytes != 2 * m * pn || !p->load_bytes
            || !p->program || !p->program_words || p->program_words > 8192
            || p->rescale_word >= p->program_words - 1 || p->activation_scale != scale
            || p->weight_scale < -126 || p->weight_scale > 127) return reject(w, SMOL_RAW_INVALID);
        int exponent = scale + p->weight_scale;
        if (exponent < -126 || exponent > 127
            || p->program[p->rescale_word] != (((uint32_t) (exponent + 127) << 23) | UINT32_C(0x25f)))
            return reject(w, SMOL_RAW_INVALID);
        if (host_matrix_pack_bf16_at_scale(w->input_copy + next_row * k, p->logical_rows, k,
                m, pk, 32, 32, scale, w->activation_copy, w->activation_capacity)) return reject(w, SMOL_RAW_INPUT);
        for (size_t i = 0; i < m * pk; ++i) w->dram[i] = w->activation_copy[i];
        for (size_t i = 0; i < p->weight_bytes; ++i) w->dram[m * pk + i] = p->weights[i];
        for (size_t i = m * pk + p->weight_bytes; i < p->output_offset; ++i) w->dram[i] = 0xa5;
        for (size_t i = p->output_offset; i < p->output_offset + p->store_bytes; ++i) w->dram[i] = 0xcd;
        for (size_t i = p->output_offset + p->store_bytes; i < p->dram_bytes; ++i) w->dram[i] = 0x5a;
        if (atlas_host_load_raw(w->host, p->program, p->program_words) != ATLAS_HOST_OK) return reject(w, SMOL_RAW_RUNTIME);
        w->active_buffer = (atlas_host_buffer) {(uintptr_t) w->dram, p->dram_bytes, ATLAS_HOST_DEVICE_READ_WRITE};
        uint32_t before = w->host->mmio.read32(w->host->mmio.context, w->host->map.csr_base);
        atlas_host_result result = atlas_host_start(w->host, &w->active_buffer, 1);
        if (result == ATLAS_HOST_OK) result = atlas_host_wait(w->host, w->max_polls);
        w->cycle_delta[w->panel_cursor] = w->host->mmio.read32(w->host->mmio.context, w->host->map.csr_base) - before;
        if (result != ATLAS_HOST_OK) return reject(w, SMOL_RAW_RUNTIME);
        for (size_t i = 0; i < m * pk; ++i)
            if (((volatile uint8_t *) w->dram)[i] != w->activation_copy[i]) return reject(w, SMOL_RAW_INPUT);
        for (size_t i = 0; i < p->weight_bytes; ++i)
            if (((volatile uint8_t *) w->dram)[m * pk + i] != p->weights[i]) return reject(w, SMOL_RAW_INPUT);
        for (size_t i = m * pk + p->weight_bytes; i < p->output_offset; ++i)
            if (((volatile uint8_t *) w->dram)[i] != 0xa5) return reject(w, SMOL_RAW_GUARD);
        for (size_t i = p->output_offset + p->store_bytes; i < p->dram_bytes; ++i)
            if (((volatile uint8_t *) w->dram)[i] != 0x5a) return reject(w, SMOL_RAW_GUARD);
        for (size_t row = 0; row < m; ++row)
            for (size_t column = 0; column < pn; ++column) {
                uint16_t value = physical(w->dram, p->output_offset, row, column, pn);
                if ((value & 0x7f80) == 0x7f80
                    || ((row >= p->logical_rows || column >= n) && value != 0)) return reject(w, SMOL_RAW_OUTPUT);
                if (row < p->logical_rows && column < n) {
                    float biased = add_bias(value, call, column);
                    if (!isfinite(biased) || (output_dtype == SMOL_BF16
                        && (float_to_bf16(biased) & 0x7f80) == 0x7f80)) return reject(w, SMOL_RAW_OUTPUT);
                }
            }
        for (size_t row = 0; row < p->logical_rows; ++row)
            for (size_t column = 0; column < n; ++column) {
                float value = add_bias(physical(w->dram, p->output_offset, row, column, pn), call, column);
                size_t index = (next_row + row) * n + column;
                if (output_dtype == SMOL_BF16) ((uint16_t *) output)[index] = float_to_bf16(value);
                else ((float *) output)[index] = value;
            }
        next_row += p->logical_rows;
        ++w->panel_cursor;
    }
    if (next_row != rows) return reject(w, SMOL_RAW_INVALID);
    ++w->call_cursor;
    return SMOL_RAW_OK;
}

int smol_policy_raw_graph(void *opaque, unsigned site, unsigned layer, unsigned role,
    unsigned dtype, const void *input, size_t rows, size_t k, unsigned output_dtype, void *output, size_t n) {
    return invoke(opaque, site, layer, role, 0, dtype, input, rows, k, output_dtype, output, n);
}

int smol_policy_raw_vision(void *opaque, unsigned layer, unsigned role,
    const uint16_t *input, size_t rows, size_t k, uint16_t *output, size_t n) {
    smol_policy_raw_context *w = opaque;
    if (!w) return SMOL_RAW_INVALID;
    return invoke(w, 7, layer, role, w->image, SMOL_BF16, input, rows, k, SMOL_BF16, output, n);
}
