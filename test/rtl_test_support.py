import hashlib
import json
import os
from pathlib import Path
import sys

from test_delay_insertion import OPT, EMIT, ROOT, addi, cycles, nop, program, run, stream, without_delays
from test_virtual_dma import copy
from test_virtual_lowering import EXAMPLES, virtual_chain, virtual_pressure

sys.path.insert(0, str(ROOT / "tools"))
from rtl_extract.spec import read_spec as pair_spec  # generated AtlasCore specs resolve by path

FACTS = os.environ.get("ATLAS_OP_TIMING")
PAIRS = os.environ.get("ATLAS_PAIR_FACTS")  # merlin.op_pairs.v1 derived from FACTS (tools/rtl_extract/pairfacts.py)
PROVIDER = "rtl-op-timing-v1"
CONSUMERS = ("--insert-atlas-delays", "--schedule-atlas-stream")
VERIFY = "--verify-atlas-timing"
HALT = ("trap", 'kind = "ecall"')
MARKER = ("csr", 'kind = "rrw", dst = 0 : i32, source = 1 : i32, address = 3088 : i32')
VLS_COPY = ROOT / "test/examples/ee290_vls_copy.mlir"

def checked(test, source, *options, tool=OPT, diagnostic=None):
    result = run(tool, source, *options)
    test.assertEqual(result.returncode == 0, diagnostic is None, result.stderr)
    if diagnostic:
        test.assertRegex(result.stderr, diagnostic)
    return result.stdout

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def selection(path=None, digest=None, dma=None, pairs=None):
    path = Path(path or FACTS).resolve()
    options = f"op-timing={path} op-timing-sha256={digest or sha256(path)}"
    if pairs:
        options += f" op-pairs={Path(pairs).resolve()} op-pairs-sha256={sha256(pairs)}"
    return "--select-atlas-rtl-evidence=" + options + (f" dma={dma}" if dma else "")

def selected(ops, *passes, path=None, digest=None, dma=None):
    source = ops if isinstance(ops, str) else program(ops)
    return run(OPT, source, selection(path, digest, dma), *passes)

def facts():
    return json.loads(Path(FACTS).read_text())

def block(document, name):
    return next(b for b in document["op_timing"] if b["name"] == name)

def json_variant(source, directory, mutate, name):
    document = json.loads(Path(source).read_text())
    mutate(document)
    path = Path(directory) / name
    path.write_text(json.dumps(document))
    return path

def delay(n):
    return ("delay", f"cycles = {n} : i32")

def lui(dst, immediate):
    return ("upper", f'kind = "lui", dst = {dst} : i32, immediate = {immediate} : i32')

def vls(kind, mreg=4, base=6, offset=0):
    register = "dst" if kind == "vload" else "src"
    return (kind, f'{register} = {mreg} : i32, base = {base} : i32, ' f'offset = {offset} : i32, format = "raw"')

def scalar_load(kind="lw", dst=5, base=6, offset=0):
    return ("scalar_load", f'kind = "{kind}", dst = {dst} : i32, base = {base} : i32, offset = {offset} : i32')

def scalar_store(kind="sw", src=2, base=6, offset=0):
    return ("scalar_store", f'kind = "{kind}", src = {src} : i32, base = {base} : i32, offset = {offset} : i32')

def word_base(dst, words):
    low = words & 4095
    if low >= 2048:
        low -= 4096
    return [lui(dst, ((words - low) >> 12) & 0xfffff), addi(dst, dst, low)]

def branch(kind, lhs, rhs, offset_words):
    return ("branch", f'kind = "{kind}", lhs = {lhs} : i32, rhs = {rhs} : i32, '
                      f'offset_bytes = {2 * offset_words} : i32')

def jal(offset_words):
    return ("jump", f'kind = "jal", dst = 0 : i32, base = 0 : i32, offset = {2 * offset_words} : i32')

VLOAD = vls("vload")
VLS_LOOP = [addi(6, 0, 256), addi(13, 0, 0), addi(14, 0, 3), VLOAD,
            vls("vstore", 4, 6, 32), addi(13, 13, 1), branch("blt", 13, 14, -3), nop(), HALT]

def export(source):
    result = run(EMIT, source, "--rtl-timing-json")
    if result.returncode:
        raise AssertionError(result.stderr)
    return json.loads(result.stdout)

def mxu(operation, unit, reg=2, acc=0, slot=0, scale=0):
    if operation.startswith("push_"):
        kind = {"push_weight": "weight_fp8", "push_acc_fp8": "acc_fp8", "push_acc_bf16": "acc_bf16"}[operation]
        target = slot if operation == "push_weight" else acc
        return ("mxu_push", f'kind = "{kind}", unit = {unit} : i32, src = {reg} : i32, slot = {target} : i32')
    if operation.startswith("pop_"):
        fp8 = operation == "pop_acc_fp8"
        return ("mxu_pop", f'format = "{"fp8" if fp8 else "bf16"}", unit = {unit} : i32, dst = {reg} : i32, '
                           f'slot = {acc} : i32, scale_reg = {scale if fp8 else 0} : i32')
    return ("mxu_matmul", f"unit = {unit} : i32, src = {reg} : i32, weight_slot = {slot} : i32, "
                          f"acc_slot = {acc} : i32, accumulate = {str(operation == 'matmul_acc').lower()}")

def busy(record):
    latency = record["assumptions"]["scratchpad_read_latency"]
    return max(g["last_age"] + (latency if k.startswith("mreg_read") else 0)
               for k, g in record["events"].items() if g["count"])

def config(reg=5, channel=0):
    return ("dma_config", f"channel = {channel} : i32, base_reg = {reg} : i32")

def transfer(direction="load", channel=0, reg=6, dram=1, size=2):
    return ("dma", f'direction = "{direction}", channel = {channel} : i32, reg = {reg} : i32, '
                   f"dram = {dram} : i32, size = {size} : i32")

def wait(channel=0):
    return ("dma_wait", f"channel = {channel} : i32")

def dma_setup(vmem=0, offset=0x90000000, size=128, base=0):
    return [*word_base(5, base), config(), *word_base(6, vmem), *word_base(1, offset), *word_base(2, size)]

WAITED = [*dma_setup(), transfer(), wait(), HALT]
DMA_SECOND = [*word_base(7, 1024), *word_base(3, 0x90001000), *word_base(9, 65536)]
CHANNEL_REUSE = "the RTL does not protect channel reuse"
EIGHT_SLOTS = "the DMA engine has eight command slots"
IN_FLIGHT = "which may still be in flight"
DMA_LOAD = transfer()
DMA_WAIT = wait()

def transpose(src=3, dst=7):
    return ("xlu_transpose", f"dst = {dst} : i32, src = {src} : i32")

def binary(lhs=0, rhs=2, dst=4, kind="mul"):
    return ("vpu_binary", f'kind = "{kind}", dst = {dst} : i32, lhs = {lhs} : i32, rhs = {rhs} : i32')

def unary(kind, src=0, dst=4):
    return ("vpu_unary", f'kind = "{kind}", dst = {dst} : i32, src = {src} : i32')

def reduce(kind, src=0, dst=4):
    return ("vpu_reduce", f'kind = "{kind}", dst = {dst} : i32, src = {src} : i32')

def pack(direction, src=0, dst=4, scale=3):
    return ("vpu_pack", f'direction = "{direction}", dst = {dst} : i32, src = {src} : i32, scale_reg = {scale} : i32')

def vli(mode, dst=4, immediate=16256):
    return ("vli", f'mode = "{mode}", dst = {dst} : i32, immediate = {immediate} : i32')

BANK_WORDS = 8192 * 8
LIMIT = 400  # widest gap probed; every pair here drains well before it
BINARY = {"add": "add", "sub": "sub", "mul": "mul", "pairmax": "max", "pairmin": "min"}
UNARY = {"mov": "mov", "rcp": "recip", "log": "log2",
         **{k: k for k in ("exp", "exp2", "relu", "sin", "cos", "tanh", "sqrt", "square", "cube")}}
REDUCE = {"rsum": "row_sum", "rmax": "row_max", "rmin": "row_min",
          "csum": "col_sum", "cmax": "col_max", "cmin": "col_min"}
VLI = {"vliAll": "all", "vliRow": "row", "vliCol": "col", "vliOne": "one"}
MXU_WORDS = {0: "push_weight", 2: "push_acc_fp8", 4: "push_acc_bf16", 6: "pop_acc_fp8", 8: "pop_acc_bf16",
       10: "matmul", 12: "matmul_acc"}  # funct7 - unit (lib/AtlasEncoding.cpp)

def separated(gap):
    return [] if gap == 1 else [nop()] if gap == 2 else [delay(gap - 2)]

def gapped(setup, first, second, gap, tail=()):
    return program([*setup, first, *separated(gap), second, *tail, delay(LIMIT), nop(), HALT])

def issue_cycles(printed):
    out, cycle = [], 0
    for op, fields, _ in stream(printed):
        out.append((op, sorted(fields.split(", ")), cycle))
        cycle += int(fields.split("cycles = ")[1].split(" ")[0]) + 1 if op == "delay" else 1
    return out

def issue_gap(printed, first, second):
    ops = issue_cycles(printed)
    find = lambda op, skip=-1: next(i for i, (name, fields, _) in enumerate(ops)
                                    if i != skip and (name, fields) == (op[0], sorted(op[1].split(", "))))
    a = find(first)
    return ops[find(second, a)][2] - ops[a][2]

def decode_mxu(word):
    vd, vs1, vs2, f7 = word >> 7 & 63, word >> 13 & 63, word >> 19 & 63, word >> 25
    kind, unit = MXU_WORDS[f7 & ~1], f7 & 1
    if kind == "push_weight":
        return mxu(kind, unit, reg=vs1, slot=vd)
    if kind.startswith("push_acc"):
        return mxu(kind, unit, reg=vs1, acc=vd)
    if kind.startswith("pop"):
        return mxu(kind, unit, reg=vd, acc=vs2, scale=vs1)
    return mxu(kind, unit, reg=vs1, acc=vd, slot=vs2)

def spaced(ops, gap, drain=400):
    out = []
    for i, op in enumerate(ops):
        out += ([] if i == 0 else separated(gap)) + [op]
    return program(out + [delay(drain), nop(), HALT])

def vpu(name, regs):
    vs1, vs2, vd = regs
    if name in BINARY:
        return binary(vs1, vs2, vd, BINARY[name])
    if name in UNARY:
        return unary(UNARY[name], vs1, vd)
    if name in REDUCE:
        return reduce(REDUCE[name], vs1, vd)
    if name in VLI:
        return vli(VLI[name], vd)
    return pack("bf16_to_fp8" if name == "fp8pack" else "fp8_to_bf16", vs1, vd, 3)

def facts_variant(directory, mutate, name="facts.json"):
    return json_variant(FACTS, directory, mutate, name)

def pairs_variant(directory, mutate, name="op-pairs.json"):
    return json_variant(PAIRS, directory, mutate, name)

class RTLTestSupport:
    def check(self, ops, accepted, *passes, diagnostic=None, **options):
        result = selected(ops, *passes, **options)
        self.assertEqual(result.returncode == 0, accepted, result.stderr)
        if diagnostic:
            self.assertIn(diagnostic, result.stderr)
        return result
