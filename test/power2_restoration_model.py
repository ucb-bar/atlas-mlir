"""Independent selected-model prototype for exact BF16 power-of-two scaling.

This is a diagnostic prototype, not an emitter or a broader intake policy.
"""
import json
import os
from pathlib import Path
import sys

import torch
from npu_model.configs.numerics import RtlNumerics


def normal_codes(low, high):
    return torch.tensor([0, *(sign | ((exponent + 127) << 7) | mantissa
        for sign in (0, 0x8000) for exponent in range(low, high + 1)
        for mantissa in range(128))], dtype=torch.int32)


def check_multiplication(codes):
    values = codes.to(torch.uint16).view(torch.bfloat16)
    comparisons = 0
    for scale in range(-50, 48):
        factor = torch.tensor(2.0 ** scale, dtype=torch.bfloat16)
        actual = RtlNumerics.mul(values, factor).view(torch.uint16).to(torch.int32)
        expected = torch.where(codes == 0, 0, codes + scale * 128)
        assert torch.equal(actual, expected), scale
        comparisons += codes.numel()
    return comparisons


def main():
    receipt_path, atlas_root = map(Path, sys.argv[1:])
    sys.path.insert(0, str(atlas_root.parent / 'merlin/examples/atlas/whole-model/unchanged_fx'))
    from model_runtime import authenticate_model
    model = Path(os.environ['IMPORT_MODEL_ROOT'])
    before = authenticate_model(model)
    torch.set_num_threads(1)

    # Stage one exhausts every raw significand/sign and both scale extrema.
    # Stage two exhausts the complete larger intermediate exponent range, a
    # superset of every stage-one result, independently for every rhs scale.
    raw = normal_codes(-26, 32)
    intermediate = normal_codes(-76, 79)
    first = check_multiplication(raw)
    second = check_multiplication(intermediate)
    for exponent in range(-26, 33):
        for sx in range(-50, 48):
            for sw in range(-50, 48):
                assert -126 <= exponent + sx + sw <= 126
                assert -133 <= sx + sw <= 127

    def bf16(codes):
        return codes.to(torch.uint16).view(torch.bfloat16)

    scales = torch.tensor([-50 + (row * 7) % 98 for row in range(32)], dtype=torch.int32)
    factors = (scales + 127) << 7
    row_source = torch.zeros((32, 32), dtype=torch.int32)
    row_source[:, 0] = factors
    row_broadcast = RtlNumerics.row_sum(bf16(row_source)).expand(32, 32)
    assert torch.equal(row_broadcast.view(torch.uint16).to(torch.int32), factors[:, None].expand(32, 32))

    # A non-row exponent needs only one sparse factor word. Physical
    # col_sum duplicates it into both halves; row_sum adds those two copies.
    for scalar_scale in range(-50,48):
        sparse=torch.zeros((64,16),dtype=torch.int32)
        sparse[0,0]=(scalar_scale+126)<<7
        columns=RtlNumerics.column_sum(bf16(sparse)).expand(64,16)
        logical=torch.cat((columns[:32],columns[32:]),dim=1)
        broadcast=RtlNumerics.row_sum(logical).expand(32,32)
        assert torch.all(broadcast.view(torch.uint16).to(torch.int32)==((scalar_scale+127)<<7))

    # col_sum is a 64x16 physical reduction: register-pair halves are stacked.
    # Two independent sparse broadcasts plus fixed half masks preserve all 32
    # logical column factors. A single broadcast instead merges c and c+16.
    left_source = torch.zeros((64, 16), dtype=torch.int32)
    right_source = torch.zeros((64, 16), dtype=torch.int32)
    left_source[0] = factors[:16]
    right_source[32] = factors[16:]
    left = RtlNumerics.column_sum(bf16(left_source)).expand(64, 16)
    right = RtlNumerics.column_sum(bf16(right_source)).expand(64, 16)
    left_mask = torch.zeros((64, 16), dtype=torch.bfloat16)
    left_mask[:32] = 1
    right_mask = 1 - left_mask
    assembled = RtlNumerics.add(RtlNumerics.mul(left, left_mask), RtlNumerics.mul(right, right_mask))
    expected = torch.cat((factors[:16].expand(32, 16), factors[16:].expand(32, 16)))
    assert torch.equal(assembled.view(torch.uint16).to(torch.int32), expected)
    naive = RtlNumerics.column_sum(bf16(left_source | right_source)).expand(64, 16)
    mismatches = int((naive.view(torch.uint16).to(torch.int32) != expected).sum())
    assert mismatches > 0
    assert before == authenticate_model(model)
    receipt = dict(
        schema='atlas.power2_restoration_prototype.v1', model_sources=before,
        raw_unbiased_exponents=[-26, 32], individual_scales=[-50, 47],
        intermediate_unbiased_exponents=[-76, 79], final_unbiased_exponents=[-126, 126],
        raw_codes=raw.numel(), intermediate_codes=intermediate.numel(),
        first_multiply_exact_cases=first, second_multiply_exact_cases=second,
        proof='All signs/significands and each scale checked against exact integer exponent adjustment. '
              'Stage-two domain contains every stage-one result; all products remain finite normal or +0.',
        row_broadcast_exact_codes=1024, scalar_broadcast_exact_codes=98*1024, two_column_broadcasts_half_mask_exact_codes=1024,
        naive_single_column_broadcast_mismatches=mismatches,
        functional_instruction_cost_measured=False)
    receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')


if __name__ == '__main__':
    main()
