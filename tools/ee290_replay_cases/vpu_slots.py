"""Two VPU commands at once: compiler schedules under selected pair facts (replay --pairs), checked lane by lane.

A copy (mov) of the exp input feeds exp, and relu overwrites the copy's source, so exp (RAW) and relu (WAR) become
ready together when the copy drains; with pair facts the second issues while the first runs (one cycle after it under
delay insertion), in the other VectorFSM slot. In the chain a two-port add holds back three single-port commands
(relu on its sum; a copy and an immediate fill overwriting its sources) until it retires: two run together and the
third takes the first slot to free up. Operands lie in distinct physical MREG banks, so the pair facts apply. The unary case reruns
vpu_ops' exp, exp2, square, cube and mov stream, which delay insertion overlaps under pair facts.
"""
from dataclasses import replace

from . import END, Case, add, filler, put, tile, vload, vstore
from .vpu_ops import A, B, CASES as VPU_CASES, EXP_IN, RELU_IN, chain, elementwise, unary, vli


def vpu(kind, dst, src): return ('vpu_unary', f'kind = "{kind}", dst = {dst} : i32, src = {src} : i32')


def beside(events):
    """(earlier, later) issue gaps of VPU commands issued while an earlier VPU command was live."""
    vpus = sorted((c for c in events['commands'] if c['engine'] == 'vpu'), key=lambda c: c['edge'])
    return [b['edge'] - a['edge'] for i, a in enumerate(vpus) for b in vpus[i + 1:] if b['edge'] < a['release_edge']]


LOADS = (add(6, 0), *(vload(m, 6, 8 * m) for m in range(4)), vload(6, 6, 32), vload(7, 6, 40))
PAIR = (vpu('mov', 4, 0), vpu('exp', 8, 4), vpu('relu', 0, 2))
STORES = tuple(vstore(m, 6, 128 + 8 * k) for k, m in enumerate((4, 5, 8, 9, 0, 1)))
INPUTS = {0: EXP_IN + RELU_IN + filler(2048), 512: filler(8192)}


def copied(source, destination):
    return lambda vm, dram: put(vm, destination, tile(vm, source, 64))


PAIR_EXPECT = chain(copied(0, 512), unary('vexp.bf16', 0, 576), unary('vrelu.bf16', 64, 640))

# add(m0, m2) -> m4, then relu(m4) -> m8 (RAW), mov(m6) -> m0 and vli.all -> m2 (WAR on the add's sources).
IMMEDIATE = 0x3fc0
CHAIN = (add(6, 0), *(vload(m, 6, 8 * k) for k, m in enumerate((0, 1, 2, 3, 6, 7))),
         ('vpu_binary', 'kind = "add", dst = 4 : i32, lhs = 0 : i32, rhs = 2 : i32'), vpu('relu', 8, 4), vpu('mov', 0, 6),
         ('vli', f'mode = "all", dst = 2 : i32, immediate = {IMMEDIATE} : i32'),
         *(vstore(m, 6, 128 + 8 * k) for k, m in enumerate((4, 5, 8, 9, 0, 1, 2, 3))))


def chained(vm, dram):
    sums = [elementwise('vadd.bf16', vm[r], vm[64 + r]) for r in range(64)]
    put(vm, 512, b''.join(sums))
    put(vm, 576, b''.join(elementwise('vrelu.bf16', row) for row in sums))
    copied(128, 640)(vm, dram)
    put(vm, 704, b''.join(vli('all', IMMEDIATE, r) for r in range(64)))


CASES = {
    'slots_exp_relu': Case(LOADS + PAIR + STORES + END, INPUTS, PAIR_EXPECT, {'vload': 6, 'vpu': 3, 'vstore': 6},
                           check=lambda events: bool(beside(events)), pairs=True),
    'slots_chain': Case(CHAIN + END, {0: A + B + filler(2048), 512: filler(8192)}, chained, {'vload': 6, 'vpu': 4, 'vstore': 8},
                        check=lambda events: len(beside(events)) >= 2, pairs=True),
    'slots_unary': replace(VPU_CASES['vpu_unary'], consumers=('delay',), check=lambda events: bool(beside(events)), pairs=True),
}
