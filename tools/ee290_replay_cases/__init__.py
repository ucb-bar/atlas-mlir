"""Replay cases and exact memory expectations for AtlasCore."""
from dataclasses import dataclass, field

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
    source: str = ''      # a virtual program instead of `ops`; `pipeline` lowers it before facts selection
    pipeline: tuple = ()
    pairs: bool = False   # scheduled with the pair facts selected (replay --pairs)

    def mlir(self, ops=None):
        if self.source and ops is None: return self.source
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

def issue_gaps(events, engine):
    """Issue gaps between commands live together on one engine."""
    commands = sorted((c for c in events['commands'] if c['engine'] == engine), key=lambda c: c['edge'])
    return [b['edge'] - a['edge'] for i, a in enumerate(commands) for b in commands[i + 1:] if b['edge'] < a['release_edge']]

def raw_tiles(): return bytes((37 * i + 11 + 3 * (i // 32)) & 255 for i in range(1024)) + filler(7168, 1024)

def power(i, rhs):
    exponent = i % 3 - 1 if rhs else i % 5 - 2
    negative = i % 11 == 0 if rhs else i % 7 == 0
    return (0x8000 if negative else 0) | ((127 + exponent) << 7)

def product(i):  # closed form of power(i, False) * power(i, True)
    return (0x8000 if (i % 11 == 0) != (i % 7 == 0) else 0) | ((127 + i % 5 - 2 + i % 3 - 1) << 7)

def bf16_pairs():
    return b''.join(power(i, False).to_bytes(2, 'little') for i in range(1024)) + \
           b''.join(power(i, True).to_bytes(2, 'little') for i in range(1024)) + filler(4096, 4096)

def transposed(vm, dram): put(vm, 32, transpose(tile(vm, 0)))
def products(vm, dram): put(vm, 128, b''.join(product(i).to_bytes(2, 'little') for i in range(1024)))

A, B, O = 0x90000000, 0x90001000, 0x90000400
def source(i): return (37 * i + 11) & 255

def composed(vm, dram):
    put(vm, 0, bytes(source(i) for i in range(128)))
    transposed(vm, dram)
    for i, value in enumerate(tile(vm, 32, 4)): dram[O + i] = value

TRANSPOSE = (add(6, 0), vload(0, 6), xlu(1, 0), add(8, 256), vstore(1, 8), add(0, 0))
MUL = tuple(op for bank, words in enumerate([0, 256, 512, 768]) for op in (add(6, words), vload(bank, 6))) + \
      (vmul(4, 0, 2), add(8, 1024), vstore(4, 8), add(8, 1280), vstore(5, 8))
MIXED = (add(5, 0), dma_config(0, 5), add(6, 0), upper(1, 589824), add(2, 128), dma('load', 0, 6, 1), wait(0)) + TRANSPOSE + \
        (upper(3, 589824), add(3, 1024, 3), dma('store', 1, 8, 3), wait(1))

STRAIGHT = {
    'xlu': Case(TRANSPOSE + END, {0: raw_tiles()}, transposed, {'vload': 1, 'xlu': 1, 'vstore': 1}),
    'vmul': Case(MUL + END, {0: bf16_pairs()}, products, {'vload': 4, 'vpu': 1, 'vstore': 2}),
    'dma_xlu': Case(MIXED + END, {0: raw_tiles()}, composed, {'dma': 2, 'vload': 1, 'xlu': 1, 'vstore': 1},
                    dram={A: bytes(source(i) for i in range(128)), B: bytes((53 * i + 79) & 255 for i in range(128)),
                          A + (1 << 32): b'\xe7' * 128, O - 32: b'\xa5' * 192}, dma=(4, 4)),
}

def copy_tile(first, to):
    return lambda vm, dram: put(vm, to, tile(vm, first))

# Three iterations over loop-invariant tiles; only the x13 counter varies.
LOOP = (add(6, 0), add(8, 256), add(13, 0), add(14, 3), vload(0, 6), vstore(0, 8), add(13, 1, 13), branch('blt', 13, 14, -3), nop())

# Each iteration launches and waits a DMA load, so every block instance restarts its epochs.
DMA_LOOP = (add(5, 0), dma_config(0, 5), add(6, 0), upper(1, 589824), add(2, 128), add(8, 256), add(13, 0), add(14, 3),
            dma('load', 0, 6, 1), wait(0), vload(0, 6), vstore(0, 8), add(13, 1, 13), branch('blt', 13, 14, -5), nop())

def loaded_copy(vm, dram): put(vm, 0, bytes(source(i) for i in range(128))); put(vm, 32, tile(vm, 0))

def diamond(condition):
    """x5 == 0 takes the branch to the else arm (tile 1); otherwise the then arm (tile 0) jumps to the join."""
    return (add(6, 0), add(7, 256), add(8, 512), add(5, condition), branch('beq', 5, 0, 5), nop(),
            vload(0, 6), jal(3), nop(), vload(0, 7), vstore(0, 8))

CONTROL = {
    'loop': Case(LOOP + END, {0: raw_tiles()}, copy_tile(0, 32), {'vload': 3, 'vstore': 3}),
    'dma_loop': Case(DMA_LOOP + END, {0: raw_tiles()}, loaded_copy, {'dma': 3, 'vload': 3, 'vstore': 3},
                     dram={A: bytes(source(i) for i in range(128))}, dma=(12, 0)),
    'diamond_taken': Case(diamond(0) + END, {0: raw_tiles()}, copy_tile(32, 64), {'vload': 1, 'vstore': 1}),
    'diamond_fall': Case(diamond(1) + END, {0: raw_tiles()}, copy_tile(0, 64), {'vload': 1, 'vstore': 1}),
}

BANK1 = 8192  # first line of VMEM bank 1; x7 = 16 << 12 words addresses it
SECOND = bytes((29 * i + 5) & 255 for i in range(1024))

# VLOAD into m1 from bank 1, then one cycle later VSTORE of m0 to bank 0.
LOAD_STORE = (add(6, 0), vload(0, 6), delay(34), upper(7, 16), add(8, 256), add(9, 512),
              vload(1, 7), vstore(0, 8), delay(40), vstore(1, 9), delay(40))
# VMUL on m0..m5, then one cycle later VLOAD into m6 from bank 1.
FINAL_VPU_LOAD = tuple(op for bank, words in enumerate([0, 256, 512, 768]) for op in (add(6, words), vload(bank, 6), delay(34))) + \
           (upper(7, 16), add(8, 1024), add(9, 1280), add(10, 1536), vmul(4, 0, 2), vload(6, 7), delay(70),
            vstore(4, 8), delay(34), vstore(5, 9), delay(34), vstore(6, 10), delay(40))

def load_store(vm, dram): put(vm, 32, tile(vm, 0)); put(vm, 64, tile(vm, BANK1))
def vpu_load(vm, dram): products(vm, dram); put(vm, 192, tile(vm, BANK1))

FINAL = {
    'overlap_vls': Case(LOAD_STORE + END, {0: raw_tiles(), BANK1: SECOND}, load_store, {'vload': 2, 'vstore': 2},
                        final=True, consumers=('final',), check=lambda events: overlapping(events, 'vload', 'vstore')),
    'overlap_vpu': Case(FINAL_VPU_LOAD + END, {0: bf16_pairs(), BANK1: SECOND}, vpu_load, {'vload': 5, 'vpu': 1, 'vstore': 3},
                        final=True, consumers=('final',), check=lambda events: overlapping(events, 'vpu', 'vload')),
}

# Two VMULs on the same sources: the second issues at the first's next_issue_age, one edge before the first
# releases.
VPU_PAIR = (add(6, 0), *(vload(m, 6, 8 * m) for m in range(4)), vmul(4, 0, 2), vmul(12, 0, 2),
            *(vstore(m, 6, 32 + 8 * k) for k, m in enumerate((4, 5, 12, 13))))

def vpu_pair(vm, dram): products(vm, dram); put(vm, 192, tile(vm, 128, 64))

# Four tiles copied through m4..m7: VLOAD t+1 overlaps VSTORE t on distinct banks; on one bank the
# VMEM windows order them (load->store 30, store->load 34).
def copy_stream(destination):
    return (add(6, 0), destination, *(op for t in range(4) for op in (vload(4 + t, 6, 8 * t), vstore(4 + t, 7, 8 * t))))

def copied_tiles(line): return lambda vm, dram: put(vm, line, tile(vm, 0, 128))

# A VMUL with an unrelated VLOAD from bank 1 issued under it.
VPU_LOAD = (add(6, 0), upper(7, 16), *(vload(m, 6, 8 * m) for m in range(4)), vmul(4, 0, 2), vload(6, 7),
            vstore(4, 6, 32), vstore(5, 6, 40), vstore(6, 6, 48))
# Software-pipelined load-transpose-store: tile t's transpose runs under tile t+1's load and tile t's store
# under tile t+1's transpose.
XLU_PIPE = (add(6, 0), upper(7, 16), vload(0, 6, 0), vload(1, 6, 8), xlu(16, 0), vstore(16, 7, 0),
            vload(2, 6, 16), xlu(17, 1), vstore(17, 7, 8), xlu(18, 2), vstore(18, 7, 16))

def pipeline_transposed(vm, dram):
    for t in range(3): put(vm, BANK1 + 32 * t, transpose(tile(vm, 32 * t)))

OVERLAP = {
    'relax_vpu_under_vload': Case(VPU_LOAD + END, {0: bf16_pairs(), BANK1: SECOND}, vpu_load,
                                  {'vload': 5, 'vpu': 1, 'vstore': 3}, check=lambda events: overlapping(events, 'vpu', 'vload')),
    'relax_xlu_pipeline': Case(XLU_PIPE + END, {0: raw_tiles(), BANK1: filler(3072)}, pipeline_transposed,
                               {'vload': 3, 'xlu': 3, 'vstore': 3},
                               check=lambda events: overlapping(events, 'xlu', 'vload') and overlapping(events, 'xlu', 'vstore')),
    'relax_copy_banks': Case(copy_stream(upper(7, 16)) + END, {0: raw_tiles(), BANK1: filler(4096)}, copied_tiles(BANK1),
                             {'vload': 4, 'vstore': 4}, check=lambda events: overlapping(events, 'vload', 'vstore')),
    'relax_copy_same_bank': Case(copy_stream(add(7, 1024)) + END, {0: raw_tiles()}, copied_tiles(128), {'vload': 4, 'vstore': 4},
                                 check=lambda events: overlapping(events, 'vload', 'vstore')),
    'relax_vpu_next_issue': Case(VPU_PAIR + END, {0: bf16_pairs()}, vpu_pair, {'vload': 4, 'vpu': 2, 'vstore': 4},
                                 check=lambda events: bool(issue_gaps(events, 'vpu'))),
}

BANK1, BANK3, BANK4 = 8192, 3 * 8192, 4 * 8192          # first VMEM line of each bank
A, B, O1, O2 = 0x90000000, 0x90001000, 0x90002000, 0x90003000
SIZE = 1024
TILE_A = bytes((37 * i + 11 + 3 * (i // 32)) & 255 for i in range(SIZE))
TILE_B = bytes((53 * i + 79 + (i // 32)) & 255 for i in range(SIZE))
SOURCE = bytes((29 * i + 5) & 255 for i in range(SIZE))
GUARD = b'\xa5' * 32

# x2 size; x6/x7 word addresses of bank 3 / bank 4 line 0; x1/x3 DRAM tiles A/B; x8/x9 DRAM outputs O1/O2;
# x10/x11/x12 word addresses of bank 3 line 1024, bank 3 line 2048 and bank 1 line 0.
SETUP = (add(5, 0), dma_config(0, 5), add(2, SIZE), upper(6, BANK3 * 8 >> 12), upper(7, BANK4 * 8 >> 12),
         upper(1, A >> 12), upper(3, B >> 12), upper(8, O1 >> 12), upper(9, O2 >> 12),
         upper(10, (BANK3 + 1024) * 8 >> 12), upper(11, (BANK3 + 2048) * 8 >> 12), upper(12, BANK1 * 8 >> 12))

def guarded(address, data):
    return {address - 32: GUARD, address: data, address + len(data): GUARD}

def concurrent_dma(events):
    """Check: two DMA commands were live on a common edge."""
    spans = [(c['edge'], c['release_edge']) for c in events['commands'] if c['engine'] == 'dma']
    return any(a < d and c < b for i, (a, b) in enumerate(spans) for c, d in spans[i + 1:])

def under_dma(engine):
    """Check: some `engine` command issued while a DMA command was live."""
    def check(events):
        dmas = [(c['edge'], c['release_edge']) for c in events['commands'] if c['engine'] == 'dma']
        return any(a <= c['edge'] < b for c in events['commands'] if c['engine'] == engine for a, b in dmas)
    return check

# Two loads in flight on channels 0 and 1, waited in launch order, then two stores in flight.
TWO_CHANNELS = SETUP + (dma('load', 0, 6, 1), dma('load', 1, 7, 3), wait(0), wait(1),
                        dma('store', 0, 6, 8), dma('store', 1, 7, 9), wait(0), wait(1))

def two_channels(vm, dram):
    put(vm, BANK3, TILE_A)
    put(vm, BANK4, TILE_B)
    for i in range(SIZE):
        dram[O1 + i] = TILE_A[i]
        dram[O2 + i] = TILE_B[i]

# A load into bank 3 lines 0..31 stays pending while a VLOAD reads bank 3 lines 1024..1055 and a VSTORE writes
# bank 3 lines 2048..2079 during the transfer's write phase (the LSU is granted first); a transpose of the loaded
# tile is then stored to bank 1.
LOAD_UNDER_VLS = SETUP + (dma('load', 0, 6, 1), delay(36), vload(4, 10), delay(34), vstore(4, 11), delay(34), xlu(5, 4),
                          delay(66), vstore(5, 12), delay(34), wait(0))

def load_under_vls(vm, dram):
    put(vm, BANK3, TILE_A)
    put(vm, BANK3 + 2048, tile(vm, BANK3 + 1024))
    put(vm, BANK1, transpose(tile(vm, BANK3 + 1024)))

# A store from bank 3 lines 0..31 reads VMEM at once while a VLOAD reads bank 3 lines 1024..1055 (DMA reads wait for
# the LSU) and the loaded tile is stored to bank 3 lines 2048..2079.
STORE_UNDER_VLS = SETUP + (dma('store', 0, 6, 8), vload(4, 10), delay(34), vstore(4, 11), delay(34), wait(0))

def store_under_vls(vm, dram):
    put(vm, BANK3 + 2048, tile(vm, BANK3 + 1024))
    for i in range(SIZE):
        dram[O1 + i] = SOURCE[i]

CONCURRENT = {
    'dma_two_channels': Case(TWO_CHANNELS + END, {BANK3: filler(SIZE), BANK4: filler(SIZE, SIZE)}, two_channels, {'dma': 4},
                             dram={A: TILE_A, B: TILE_B, **guarded(O1, filler(SIZE, 2048)), **guarded(O2, filler(SIZE, 3072))},
                             dma=(64, 64), final=True, consumers=('final',), check=concurrent_dma),
    'dma_load_under_vls': Case(LOAD_UNDER_VLS + END, {BANK3: filler(SIZE), BANK3 + 1024: SOURCE, BANK3 + 2048: filler(SIZE, SIZE),
                                                      BANK1: filler(SIZE, 2048)},
                               load_under_vls, {'dma': 1, 'vload': 1, 'xlu': 1, 'vstore': 2}, dram={A: TILE_A},
                               dma=(32, 0), final=True, consumers=('final',), check=under_dma('vload')),
    'dma_store_under_vls': Case(STORE_UNDER_VLS + END, {BANK3: SOURCE, BANK3 + 1024: TILE_B, BANK3 + 2048: filler(SIZE)},
                                store_under_vls, {'dma': 1, 'vload': 1, 'vstore': 1}, dram=guarded(O1, filler(SIZE, 3072)),
                                dma=(0, 32), final=True, consumers=('final',), check=under_dma('vload')),
}

O = 0x90000400
PAIR_STREAM = (add(5, 0), dma_config(0, 5), upper(1, 589824), upper(3, 589824), add(3, 1024, 3), add(2, 128), add(6, 0), add(7, 256),
        dma('load', 0, 6, 1), dma('store', 1, 7, 3), wait(0), wait(1))
LOADED = bytes((19 * i + 7) & 255 for i in range(128))

def moved(vm, dram):
    stored = b''.join(vm[32 + r] for r in range(4))
    put(vm, 0, LOADED)
    for i, byte in enumerate(stored): dram[O + i] = byte

PAIR_CASES = {
    'dma_pair_in_flight': Case(PAIR_STREAM + END, {0: filler(8192)}, moved, {'dma': 2}, dram={A: LOADED, O - 32: b'\xa5' * 192}, dma=(4, 4),
                               final=True, consumers=('final',), check=concurrent_dma),
    'dma_pair_compiled': Case(PAIR_STREAM + END, {0: filler(8192)}, moved, {'dma': 2}, dram={A: LOADED, O - 32: b'\xa5' * 192}, dma=(4, 4),
                              check=concurrent_dma),
}

def load():
    from . import mxu, vpu_ops
    groups = (STRAIGHT, CONTROL, CONCURRENT, PAIR_CASES, mxu.MLP, mxu.CASES,
              mxu.OVERLAP, mxu.pipelined(), FINAL, OVERLAP, mxu.RELU, mxu.DMA,
              vpu_ops.CASES, vpu_ops.SLOTS)
    cases = {}
    for group in groups:
        for name, case in group.items():
            if name in cases or not isinstance(case, Case): raise ValueError('duplicate or malformed replay case ' + name)
            cases[name] = case
    return cases
