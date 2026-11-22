#!/usr/bin/env python3
"""Measure matched authored MXU1 contractions on the explicitly selected Rust model.

This is an evaluator, not model capture or a hardware performance claim. Source
compilation never consumes the randomized execution samples.
"""
import argparse
from dataclasses import replace
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'test')]


def revision(path):
    def git(*arguments):
        return subprocess.check_output(['git', '-C', str(path), *arguments], stderr=subprocess.DEVNULL)
    try:
        return dict(commit=git('rev-parse', 'HEAD').decode().strip(),
                    tracked_diff_sha256=sha256(git('diff', 'HEAD')).hexdigest())
    except subprocess.CalledProcessError:
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for dimension, default in (('m', 128), ('n', 128), ('k', 768)):
        parser.add_argument('--'+dimension, type=int, default=default)
    parser.add_argument('--pipeline', choices=('mandatory', 'optimized'), default='optimized')
    parser.add_argument('--lhs-exponents', choices=('scalar', 'row'), default='scalar')
    parser.add_argument('--rust-model', type=Path, default=os.environ.get('ATLAS_RUST_NPU_MODEL'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if min(args.m, args.n, args.k) <= 0:
        parser.error('matrix extents must be positive')
    if args.rust_model is None or not args.rust_model.is_file() or not os.environ.get('ATLAS_OOT_BIN_DIR'):
        parser.error('select an existing Rust model and ATLAS_OOT_BIN_DIR')
    os.environ['ATLAS_RUST_NPU_MODEL'] = str(args.rust_model.resolve())

    from atlas_mlir_native.already_quantized import prepare_source
    from atlas_mlir_native.binding import DRAM_BASE
    from atlas_mlir_native.options import codegen_options, current_codegen_options
    from atlas_mlir_native.quantized_loop import execution_layout, virtual_source
    from atlas_mlir_native.restoration_budget import instruction_bound
    from test_exponent_page_cache import fixture
    from test_exponent_restoration import compile_fixture
    from test_quantized_operand_reuse import memory_fixture
    from rust_model import execute

    args.out.mkdir(parents=True, exist_ok=False)
    request = fixture(args.m, args.n, args.k, (), args.lhs_exponents, 'row', True)
    plan = prepare_source(request)
    options = replace(current_codegen_options(), io_memory_bytes=67108864,
                      execution_instruction_limit=128000000, exponent_page_cache=True)
    tools = Path(os.environ['ATLAS_OOT_BIN_DIR'])
    provenance = dict(atlas=revision(ROOT), merlin=revision(Path(os.environ['MERLIN_NATIVE_ROOT'])),
                      simulator=revision(args.rust_model.resolve().parent),
                      simulator_binary_sha256=sha256(args.rust_model.read_bytes()).hexdigest(),
                      compiler_tools={name: sha256((tools / name).read_bytes()).hexdigest()
                                      for name in ('atlas-opt', 'atlas-emit')},
                      evaluator_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    (args.out / 'source-request.json').write_text(json.dumps(request.record(), indent=2)+'\n')
    results, outputs = [], []
    with codegen_options(options):
        baseline = execution_layout(plan, operand_reuse='batch-panel-cache')
        memory = memory_fixture(plan, baseline)
        for policy in ('batch-panel-cache', 'interleaved-panel-cache'):
            layout = execution_layout(plan, operand_reuse=policy)
            source = virtual_source(plan, layout)
            machine, program = compile_fixture(source, pipeline=args.pipeline)
            bound = instruction_bound(source, machine, program)
            assert bound['admitted'] and len(program['instructions'])*4 <= 131072
            destination = args.out / policy
            actual, stats = execute(program, memory, DRAM_BASE, destination)
            begin = layout['output_addresses'][0]-DRAM_BASE
            status = layout['status_address']-DRAM_BASE
            end = status+layout['status_bytes']
            assert actual[:begin] == memory[:begin] and actual[end:] == memory[end:]
            assert int.from_bytes(actual[status:status+4], 'little') == 0
            (destination / 'virtual.mlir').write_text(source)
            (destination / 'machine.mlir').write_text(machine)
            (destination / 'output.bin').write_bytes(actual)
            (destination / 'execution-plan.json').write_text(json.dumps(layout, indent=2)+'\n')
            (destination / 'instruction-bound.json').write_text(json.dumps(bound, indent=2)+'\n')
            outputs.append(actual)
            results.append(dict(policy=policy, dispatches=1, encoded_bytes=len(program['instructions'])*4,
                                input_sha256=sha256(memory).hexdigest(), output_sha256=sha256(actual).hexdigest(),
                                virtual_sha256=sha256(source.encode()).hexdigest(),
                                machine_sha256=sha256(machine.encode()).hexdigest(), **stats))
    assert outputs[0] == outputs[1], 'complete output memory differs'
    report = dict(shape=[args.m, args.n, args.k], lhs_exponents=args.lhs_exponents,
                  pipeline=args.pipeline, provenance=provenance,
                  exact_memory_match=True, measurements=results,
                  scope='authored contraction; Rust RTL numerics, measured-curve DMA; device cycles only')
    (args.out / 'comparison.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({**report, 'provenance': 'see comparison.json'}, indent=2))


if __name__ == '__main__':
    main()
