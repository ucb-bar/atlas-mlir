"""Round-to-nearest-even BF16 tile add from truncating VPU arithmetic.

The 34-instruction sequence equals the host's FP32 add followed by BF16 RNE
rounding on every operand pair, except that NaN results are canonical 0x7FC0.
s = trunc(a + b) is the BF16 neighbour of the exact sum on the zero side;
after normalizing by sign(s), min/max select the larger operand, the sign of
L + (H - S - U/2) decides between S and its successor, and a parity bias
breaks exact ties to even. The sequence names fixed temporaries m6..m30,
m48 and m50, constants m32..m46, and four renamable pairs: inputs a and b,
the sign sg, and the result. Every binary instruction reads two different
physical MRF banks.
"""

CONSTANTS = ((32, 0x7F00), (34, 0x3C00), (36, 0x3F00), (38, 0x4780),  # 2^127, 2^-7, 0.5, 2^16
             (40, 0x3780), (42, 0x4300), (44, 0x4580), (46, 0x7F80))  # 2^-16, 2^7, 2^12, +inf

# (mnemonic, destination, source, source); "a", "b", "sg", and "r" are renamable.
SEQUENCE = (
    ("vadd", 16, "a", "b"), ("vmul", 18, 16, 32), ("vmul", 18, 18, 32), ("vtanh", "sg", 18, None),
    ("vmul", 6, "a", "sg"), ("vmul", 8, "b", "sg"), ("vmaximum", 10, 6, 8), ("vminimum", 12, 6, 8),
    ("vmul", 20, 16, "sg"), ("vmul", 22, 20, 36), ("vmul", 24, 22, 34), ("vadd", 24, 22, 24),
    ("vsub", 26, 24, 22), ("vmul", 28, 22, 38), ("vmul", 30, 28, 34), ("vadd", 30, 28, 30),
    ("vsub", 30, 30, 28), ("vmul", 28, 30, 40), ("vminimum", 26, 26, 28), ("vsub", 14, 10, 20),
    ("vsub", 14, 14, 26), ("vadd", 48, 12, 14), ("vmul", 28, 26, 42), ("vadd", 30, 22, 28),
    ("vsub", 30, 30, 28), ("vsub", 30, 22, 30), ("vadd", 30, 30, 26), ("vmul", 48, 48, 44),
    ("vminimum", 48, 48, 26), ("vadd", 48, 48, 30), ("vadd", 50, 20, 48), ("vminimum", 50, 50, 46),
    ("vmaximum", 50, 20, 50), ("vmul", "r", 50, "sg"),
)
FIXED = frozenset(register for row in SEQUENCE for register in row[1:] if type(register) is int)


def rne_add(a, b, result, sign=4, *, constants=True):
    """Emit result = RNE(a + b) for BF16 tile pairs, each instruction spaced by delay 66.

    a and b are read only by the first instructions; result is written only
    by the last, so it may alias b. The sign pair must be free scratch.
    """
    names = dict(a=a, b=b, sg=sign, r=result)
    if (any(type(r) is not int or r % 2 or not 0 <= r <= 62 for r in names.values())
            or {a, b, sign, result} & FIXED or sign in (a, b)):
        raise ValueError("RNE add pairs must be even and disjoint from its fixed temporaries")
    lines = [line for register, bits in CONSTANTS for line in (f"vli.all m{register}, {bits}", "delay 66")] if constants else []
    for op, *registers in SEQUENCE:
        registers = [names.get(r, r) for r in registers if r is not None]
        lines += [f"{op}.bf16 " + ", ".join(f"m{r}" for r in registers), "delay 66"]
    return lines
