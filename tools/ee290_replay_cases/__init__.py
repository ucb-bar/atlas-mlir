"""Replay cases for tools/replay-ee290-compute.py.

Every module in this package that defines ``CASES = {name: Case(...)}`` is discovered.
"""
from dataclasses import dataclass, field
import importlib
import pkgutil


@dataclass(frozen=True)
class Case:
    ops: tuple            # (op, attrs) pairs ending in the terminal; delay-free unless `final`
    vmem: dict            # first VMEM line -> initial bytes (whole 32-byte lines); nothing else is modeled
    expect: object        # expect(vm, dram) edits copies of the initial line/byte maps into the final state
    commands: dict        # dynamic command count per engine (vload, vstore, xlu, vpu, dma)
    dram: dict = field(default_factory=dict)  # DRAM address -> initial bytes
    dma: tuple = (0, 0)   # DRAM read/write beats the testbench must see
    final: bool = False   # ops are a final stream: not scheduled; footprints bind to a serialized reference
    check: object = None  # optional check(events) -> bool on the decoded events
    consumers: tuple = ('delay', 'schedule')

    def mlir(self, ops=None):
        lines = ['module {', '  %s0 = "atlas.start"() : () -> !atlas.state']
        for n, (op, attrs) in enumerate(self.ops if ops is None else ops, 1):
            lines.append(f'  %s{n} = "atlas.{op}"(%s{n-1}) {{{attrs}}} : (!atlas.state) -> !atlas.state')
        return '\n'.join(lines + ['}', ''])


def add(dst, value, src=0): return ('alu_imm', f'kind = "addi", dst = {dst} : i32, src = {src} : i32, immediate = {value} : i32')
def nop(): return add(0, 0)
def upper(dst, value): return ('upper', f'kind = "lui", dst = {dst} : i32, immediate = {value} : i32')
def delay(cycles): return ('delay', f'cycles = {cycles} : i32')
def dma(direction, channel, vmem_reg, dram_reg, size_reg=2): return ('dma', f'direction = "{direction}", channel = {channel} : i32, reg = {vmem_reg} : i32, dram = {dram_reg} : i32, size = {size_reg} : i32')
def dma_config(channel, reg): return ('dma_config', f'channel = {channel} : i32, base_reg = {reg} : i32')
def wait(channel): return ('dma_wait', f'channel = {channel} : i32')
def vload(mreg, base, offset=0): return ('vload', f'dst = {mreg} : i32, base = {base} : i32, offset = {offset} : i32, format = "raw"')
def vstore(mreg, base, offset=0): return ('vstore', f'src = {mreg} : i32, base = {base} : i32, offset = {offset} : i32, format = "raw"')
def xlu(dst, src): return ('xlu_transpose', f'dst = {dst} : i32, src = {src} : i32')
def vmul(dst, lhs, rhs): return ('vpu_binary', f'kind = "mul", dst = {dst} : i32, lhs = {lhs} : i32, rhs = {rhs} : i32')
def branch(kind, lhs, rhs, words): return ('branch', f'kind = "{kind}", lhs = {lhs} : i32, rhs = {rhs} : i32, offset_bytes = {2 * words} : i32')
def jal(words): return ('jump', f'kind = "jal", dst = 0 : i32, base = 0 : i32, offset = {2 * words} : i32')
# Marker publication then ECALL; x1 is clobbered.
END = (add(1, 1), ('csr', 'kind = "rrw", dst = 0 : i32, source = 1 : i32, address = 3088 : i32'), ('trap', 'kind = "ecall"'))


def filler(size, start=0): return bytes((0xa5 ^ (13 * i)) & 255 for i in range(start, start + size))
def tile(vm, line, rows=32): return b''.join(vm[line + r] for r in range(rows))
def put(vm, line, data):
    for r in range(len(data) // 32): vm[line + r] = bytes(data[32 * r:32 * r + 32])
def transpose(data): return bytes(data[32 * c + r] for r in range(32) for c in range(32))


def overlapping(events, first, second):
    """True when commands of two engines were live on a common edge."""
    spans = lambda engine: [(c['edge'], c['release_edge']) for c in events['commands'] if c['engine'] == engine]
    return any(a < d and c < b for a, b in spans(first) for c, d in spans(second))


def load():
    cases = {}
    for info in sorted(pkgutil.iter_modules(__path__), key=lambda i: i.name):
        for name, case in getattr(importlib.import_module(f'{__name__}.{info.name}'), 'CASES', {}).items():
            if name in cases or not isinstance(case, Case): raise ValueError('duplicate or malformed replay case ' + name)
            cases[name] = case
    return cases
