"""Compiler-scheduled streams that exercise the selected overlap rules on hardware.

Each case runs both consumers with full issue-offset binding; `check` requires that the intended overlap
occurred in the decoded events.
"""
from . import END, Case, add, filler, overlapping, put, tile, transpose, upper, vload, vmul, vstore, xlu
from .compute import bf16_pairs, products, raw_tiles

BANK1 = 8192  # first line of VMEM bank 1


def gaps(events, first, second):
    """Issue distances from each `first` command to the next command when that is a `second` command."""
    commands = sorted(events['commands'], key=lambda c: c['edge'])
    return [b['edge'] - a['edge'] for a, b in zip(commands, commands[1:]) if (a['engine'], b['engine']) == (first, second)]


def overlapped(first, second):
    """Check: some `second` command issued while the preceding `first` command was live."""
    def check(events):
        commands = sorted(events['commands'], key=lambda c: c['edge'])
        return any(a['engine'] == first and b['engine'] == second and b['edge'] < a['release_edge']
                   for i, a in enumerate(commands) for b in commands[i + 1:])
    return check


# Two VMULs on the same sources: the second issues at the first's next_issue_age, one edge before the first
# releases.
VPU_PAIR = (add(6, 0), *(vload(m, 6, 8 * m) for m in range(4)), vmul(4, 0, 2), vmul(12, 0, 2),
            *(vstore(m, 6, 32 + 8 * k) for k, m in enumerate((4, 5, 12, 13))))


def vpu_pair(vm, dram): products(vm, dram); put(vm, 192, tile(vm, 128, 64))


# Four tiles copied through m4..m7: VLOAD t+1 overlaps VSTORE t on distinct banks; on one bank the
# VMEM windows order them (load->store 30, store->load 34).
def copy(destination):
    return (add(6, 0), destination, *(op for t in range(4) for op in (vload(4 + t, 6, 8 * t), vstore(4 + t, 7, 8 * t))))


def copied(line): return lambda vm, dram: put(vm, line, tile(vm, 0, 128))


# A VMUL with an unrelated VLOAD from bank 1 issued under it.
VPU_LOAD = (add(6, 0), upper(7, 16), *(vload(m, 6, 8 * m) for m in range(4)), vmul(4, 0, 2), vload(6, 7),
            vstore(4, 6, 32), vstore(5, 6, 40), vstore(6, 6, 48))
SECOND = bytes((29 * i + 5) & 255 for i in range(1024))


def vpu_load(vm, dram): products(vm, dram); put(vm, 192, tile(vm, BANK1))


# Software-pipelined load-transpose-store: tile t's transpose runs under tile t+1's load and tile t's store
# under tile t+1's transpose.
XLU_PIPE = (add(6, 0), upper(7, 16), vload(0, 6, 0), vload(1, 6, 8), xlu(16, 0), vstore(16, 7, 0),
            vload(2, 6, 16), xlu(17, 1), vstore(17, 7, 8), xlu(18, 2), vstore(18, 7, 16))


def transposed(vm, dram):
    for t in range(3): put(vm, BANK1 + 32 * t, transpose(tile(vm, 32 * t)))


CASES = {
    'relax_vpu_under_vload': Case(VPU_LOAD + END, {0: bf16_pairs(), BANK1: SECOND}, vpu_load,
                                  {'vload': 5, 'vpu': 1, 'vstore': 3}, check=lambda events: overlapping(events, 'vpu', 'vload')),
    'relax_xlu_pipeline': Case(XLU_PIPE + END, {0: raw_tiles(), BANK1: filler(3072)}, transposed,
                               {'vload': 3, 'xlu': 3, 'vstore': 3},
                               check=lambda events: overlapping(events, 'xlu', 'vload') and overlapping(events, 'xlu', 'vstore')),
    'relax_copy_banks': Case(copy(upper(7, 16)) + END, {0: raw_tiles(), BANK1: filler(4096)}, copied(BANK1),
                             {'vload': 4, 'vstore': 4}, check=lambda events: overlapping(events, 'vload', 'vstore')),
    'relax_copy_same_bank': Case(copy(add(7, 1024)) + END, {0: raw_tiles()}, copied(128), {'vload': 4, 'vstore': 4},
                                 check=lambda events: overlapping(events, 'vload', 'vstore')),
    'relax_vpu_next_issue': Case(VPU_PAIR + END, {0: bf16_pairs()}, vpu_pair, {'vload': 4, 'vpu': 2, 'vstore': 4},
                                 check=overlapped('vpu', 'vpu')),
}
