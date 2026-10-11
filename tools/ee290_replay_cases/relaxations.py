"""Compiler-scheduled streams that exercise the selected overlap rules on hardware.

Each case runs both consumers with full issue-offset binding; `check` requires that the intended overlap
occurred in the decoded events.
"""
from . import END, Case, add, put, tile, vload, vmul, vstore
from .compute import bf16_pairs, products


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


CASES = {
    'relax_vpu_next_issue': Case(VPU_PAIR + END, {0: bf16_pairs()}, vpu_pair, {'vload': 4, 'vpu': 2, 'vstore': 4},
                                 check=overlapped('vpu', 'vpu')),
}
