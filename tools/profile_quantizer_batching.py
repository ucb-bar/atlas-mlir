#!/usr/bin/env python3
"""Compare independently admitted token-group batches on the selected Rust model.

This authored quantizer probe checks complete device memory against a separate
host bit implementation. It reports device cycles, not host callback latency.
"""
import argparse
from dataclasses import replace
from hashlib import sha256
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'test')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rows', type=int, default=256)
    parser.add_argument('--columns', type=int, default=256)
    parser.add_argument('--pipeline', choices=('mandatory', 'optimized'), default='optimized')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if min(args.rows, args.columns) <= 0 or args.columns % 32:
        parser.error('positive rows and a positive multiple of 32 columns are required')
    if not all(os.environ.get(key) for key in ('ATLAS_RUST_NPU_MODEL', 'ATLAS_OOT_BIN_DIR', 'MERLIN_NATIVE_ROOT')):
        parser.error('select Rust, Atlas native tools and the Merlin checkout')

    from atlas_mlir_native.activation_quantize_application import prepare_plan
    from atlas_mlir_native.activation_quantize_device import prepare_plan as body_plan, prove_plan
    from atlas_mlir_native.binding import DRAM_BASE, RTL_REVISION
    from atlas_mlir_native.options import codegen_options, current_codegen_options
    from test_activation_quantize_device import dynamic_source, host_reference
    from rust_model import execute
    from tools.profile_quantized_panels import revision

    args.out.mkdir(parents=True, exist_ok=False)
    source = dynamic_source(shape=(args.rows, args.columns), reduce='token', rule='nc')
    selected = replace(current_codegen_options(), io_memory_bytes=1048576,
                       execution_instruction_limit=1000000, native_pipeline=args.pipeline,
                       dynamic_quantizer_group_batching=True)
    (args.out / 'source-request.json').write_text(json.dumps(source.record(), indent=2)+'\n')
    reports = []
    with codegen_options(selected):
        plans = [prepare_plan(source, _batch_refinement=refine) for refine in (False, True)]
        compiled = {}
        for schedule in plans:
            for body in schedule.bodies:
                if body.digest() not in compiled:
                    virtual, machine, physical, bound, _ = prove_plan(body_plan(body))
                    directory = args.out / 'bodies' / body.digest()
                    directory.mkdir(parents=True)
                    for name, payload in (('virtual.mlir', virtual), ('machine.mlir', machine),
                                          ('physical-program.json', json.dumps(physical)),
                                          ('instruction-bound.json', json.dumps(bound))):
                        (directory / name).write_text(payload+'\n')
                    compiled[body.digest()] = physical, bound

        for case in ('finite', 'exceptional'):
            raw = bytearray()
            special = (0, 0x8000, 1, 0x807f, 0x7f80, 0xff80, 0x7fc1, 0xffa3)
            for index in range(args.rows * args.columns):
                row, lane = divmod(index, args.columns)
                word = ((0x3b00 + (row % 11)*128 + (lane*17)%128) | ((lane % 3 == 0) << 15))
                if case == 'exceptional' and row % 7 == 0 and lane < len(special):
                    word = special[lane]
                raw.extend(word.to_bytes(2, 'little'))
            expected_codes, expected_exponents = host_reference(source, bytes(raw))
            (args.out / (case+'-input.bin')).write_bytes(raw)
            for policy, schedule in zip(('halving', 'refined'), plans, strict=True):
                result_codes, result_exponents = bytearray(schedule.count), bytearray(args.rows*4)
                measurements = []
                for index, launch in enumerate(schedule.launches):
                    body = schedule.bodies[launch['body_index']]
                    plan = body_plan(body)
                    physical, bound = compiled[body.digest()]
                    initial = bytearray([0xa5]) * (plan.workspace_bytes+64)
                    first = launch['input_offset_bytes']
                    initial[:launch['input_bytes']] = raw[first:first+launch['input_bytes']]
                    actual, stats = execute(physical, initial, DRAM_BASE, args.out/f'{case}-{policy}-{index}')
                    codes, exps = plan.code_address-DRAM_BASE, plan.exponent_address-DRAM_BASE
                    cb, ce = launch['code_offset_bytes'], launch['exponent_offset_bytes']
                    expected = bytearray(initial)
                    expected[codes:codes+plan.count] = expected_codes[cb:cb+plan.count]
                    # The slot is zero initialized; only logical exponent words
                    # change before the final aligned prefix is written.
                    expected[exps:exps+plan.exponent_bytes] = bytes(plan.exponent_bytes)
                    expected[exps:exps+launch['exponent_bytes']] = expected_exponents[ce:ce+launch['exponent_bytes']]
                    assert actual == expected, 'quantizer bytes or protected memory differ'
                    result_codes[cb:cb+plan.count] = actual[codes:codes+plan.count]
                    result_exponents[ce:ce+launch['exponent_bytes']] = actual[exps:exps+launch['exponent_bytes']]
                    measurements.append({key:stats[key] for key in ('cycles','total_instructions','dram_read_bytes','dram_write_bytes')})
                    assert bound['admitted'] and bound['instruction_bound'] <= 1000000
                assert result_codes == expected_codes and result_exponents == expected_exponents
                row = dict(case=case, policy=policy, dispatches=len(measurements),
                    input_sha256=sha256(raw).hexdigest(), codes_sha256=sha256(result_codes).hexdigest(),
                    exponents_sha256=sha256(result_exponents).hexdigest(),
                    **{key:sum(m[key] for m in measurements) for key in measurements[0]})
                reports.append(row)
                print(json.dumps(row), flush=True)
        for policy, plan in zip(('halving','refined'), plans, strict=True):
            (args.out / (policy+'-schedule.json')).write_text(json.dumps(plan.record(), indent=2)+'\n')

    binary = Path(os.environ['ATLAS_RUST_NPU_MODEL']).resolve()
    native = Path(os.environ['ATLAS_OOT_BIN_DIR'])
    report = dict(scope=__doc__, rows=args.rows, columns=args.columns, pipeline=args.pipeline,
        measurements=reports, exact_complete_memory_match=True, numerics='rtl', pack_layout='register_rows',
        instruction_limit=1000000, rtl_revision=RTL_REVISION,
        provenance=dict(atlas=revision(ROOT), merlin=revision(Path(os.environ['MERLIN_NATIVE_ROOT'])),
            simulator=revision(binary.parent), simulator_binary_sha256=sha256(binary.read_bytes()).hexdigest(),
            compiler_tools={name:sha256((native/name).read_bytes()).hexdigest() for name in ('atlas-opt','atlas-emit')},
            evaluator_sha256=sha256(Path(__file__).read_bytes()).hexdigest()))
    (args.out / 'comparison.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
