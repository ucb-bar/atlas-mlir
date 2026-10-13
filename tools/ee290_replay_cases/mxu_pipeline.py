"""Commands on one MXU at once: compiler schedules under selected pair facts (replay --pairs), with exact results.

The FP8 tile and accumulate cases of mxu.py rerun with pair facts, so the pop issues while its matmul still writes the
accumulator (age 64 on MXU0, 4 on MXU1) and an accumulating matmul while its accumulator push or weight push runs. Two
tiles run back to back on one unit with both weight slots and both accumulators (the second weight push and matmul
beside the first matmul), and a K=2 accumulation pushes the second weight tile into the other slot while the first
matmul runs, then accumulates into the same accumulator. Every MREG operand of these lies in its own physical bank.
#15's virtual programs (virtual_dma_mxu and the FP8 two-layer MLPs, with and without biases) rerun with pair facts:
their lowerings reuse an MXU source register, or pop onto a register in the source's physical bank, so the
shared-operand pair facts apply. `check` requires two commands of one unit to have been live together (and, for the
MLPs, the relayout loop's moves).
"""
from dataclasses import replace

from . import END, Case, add, filler, put, vload, vstore
from .fp8_mlp import CASES as MLP_CASES
from .mxu import CASES as MXU_CASES, accumulator_codes, fixture, reference
from .virtual_dma_mxu import CASES as VIRTUAL_DMA_CASES


def same_unit(events, engine):
    """Issue gaps of commands on `engine` issued while an earlier command on it was live."""
    commands = sorted((c for c in events['commands'] if c['engine'] == engine), key=lambda c: c['edge'])
    return [b['edge'] - a['edge'] for i, a in enumerate(commands) for b in commands[i + 1:] if b['edge'] < a['release_edge']]


def push(unit, src, slot): return ('mxu_push', f'kind = "weight_fp8", unit = {unit} : i32, src = {src} : i32, slot = {slot} : i32')
def matmul(unit, src, slot, acc, accumulate=False):
    return ('mxu_matmul', f'unit = {unit} : i32, src = {src} : i32, weight_slot = {slot} : i32, acc_slot = {acc} : i32, '
                          f'accumulate = {str(accumulate).lower()}')
def pop(unit, dst, acc): return ('mxu_pop', f'format = "bf16", unit = {unit} : i32, dst = {dst} : i32, slot = {acc} : i32, scale_reg = 0 : i32')


# Lines 0..31 activation 1, 32..63 activation 2, 64..95 weights 1, 96..127 weights 2; results from line 128.
A1, W1 = fixture(0)
A2, W2 = fixture(3)
IMAGE = A1 + A2 + W1 + W2 + filler(4096)
LOADS = (add(6, 0), vload(0, 6), vload(4, 6, 8), vload(8, 6, 16), vload(12, 6, 24))


def stores(*pairs):
    """Store each popped BF16 pair (two registers) to consecutive 64-line panels from line 128."""
    out = []
    for k, dst in enumerate(pairs):
        out += [vstore(dst, 6, 32 + 16 * k), vstore(dst + 1, 6, 40 + 16 * k)]
    return tuple(out)


def two_tiles(unit):
    return (*LOADS, push(unit, 8, 0), push(unit, 12, 1), matmul(unit, 0, 0, 0), matmul(unit, 4, 1, 1),
            pop(unit, 16, 0), pop(unit, 20, 1), *stores(16, 20))


def two_results(vm, dram):
    put(vm, 128, reference(A1, W1))
    put(vm, 192, reference(A2, W2))


def k2(unit):
    return (*LOADS, push(unit, 8, 0), matmul(unit, 0, 0, 0), push(unit, 12, 1), matmul(unit, 4, 1, 0, True),
            pop(unit, 16, 0), *stores(16))


def k2_result(vm, dram): put(vm, 128, reference(A2, W2, accumulator_codes(reference(A1, W1))))


CASES = {}
for unit in (0, 1):
    engine = f'mxu{unit}'
    overlap = lambda events, e=engine: bool(same_unit(events, e))
    for name in ('tile', 'acc_bf16', 'acc_fp8'):
        CASES[f'{engine}_{name}_pipelined'] = replace(MXU_CASES[f'{engine}_{name}'], check=overlap, pairs=True)
    CASES[f'{engine}_two_tiles'] = Case(two_tiles(unit) + END, {0: IMAGE}, two_results, {'vload': 4, engine: 6, 'vstore': 4},
                                        check=overlap, pairs=True)
    CASES[f'{engine}_k2'] = Case(k2(unit) + END, {0: IMAGE}, k2_result, {'vload': 4, engine: 5, 'vstore': 2},
                                 check=overlap, pairs=True)
CASES['virtual_dma_mxu_pipelined'] = replace(VIRTUAL_DMA_CASES['virtual_dma_mxu'], consumers=('delay', 'schedule'), pairs=True,
                                             check=lambda events: bool(same_unit(events, 'mxu0')))
for name in ('virtual_fp8_mlp', 'virtual_fp8_mlp_bias'):
    base = MLP_CASES[name]
    CASES[f'{name}_pipelined'] = replace(base, pairs=True, check=lambda events, c=base.check: c(events) and any(same_unit(events, f'mxu{u}') for u in (0, 1)))
