"""Authenticated Atlas BF16 unary image queries, restricted to qualified domains.

The lookup tables are not monotone. Every ordered input key contributes to the
image hull, including both signed zeros. No mathematical exp/reciprocal premise
is used. Segment trees cache only immutable authenticated arithmetic data.
"""
from functools import lru_cache
import gzip
import hashlib
from pathlib import Path
import struct

DATA = Path(__file__).with_name('data') / 'atlas_unary'
# kind: input ordered interval, compressed SHA256, raw little-endian table SHA256
PINS = {
    'exp': ((0x3c58, 0x8000), 'c868283c0118ac478b2acd44af4e10d5c02bcd236d73e188e1c7ce33f112e566', '2a7c2926e33679d733be4da69c15e9bdd790c9399dcefdbfaa4bc5059a34b59c'),
    'exp2': ((0x3c7f, 0x8000), '88882a702acaf062782d1d8070cff0e7f188fa14fd9bf0b4bb334eef75eeeba5', '5e3f0205d1ea7112de5cb0579c6940c01cba4be054aaf3c407325c81911c90c7'),
    'recip': ((0xbf80, 0xc200), '803a18af1c72a1b6d722b76860d43e3028fd2addaec693b49e76e3957a39e470', 'c61286dc1754fcd6be2cb578042df5162395050ae300f064206863060e2c0109'),
}
TARGET = ('atlas-rtl-unary:provenance982eb886c6c0d5a45682943a0eecd3727f1da630be1c883f624b421e6beb9ed7'
          ':VAtlasCore-d594f72d67c0f8076faab1feae0068c96fcd93b7c31323b5b005a0bb3702119c')


def _ordered(raw):
    return raw ^ (65535 if raw & 32768 else 32768)


def table_bytes(kind):
    """Authenticate on every call, even when an image tree is cached."""
    _, compressed_sha, raw_sha = PINS[kind]
    path = DATA / ({'recip': 'rcp'}.get(kind, kind) + '.bin.gz')
    compressed = path.read_bytes()
    if hashlib.sha256(compressed).hexdigest() != compressed_sha:
        raise ValueError('unary table compressed identity mismatch')
    return _decoded(compressed, raw_sha)


@lru_cache(maxsize=3)
def _decoded(compressed, raw_sha):
    raw = gzip.decompress(compressed)
    if len(raw) != 131072 or hashlib.sha256(raw).hexdigest() != raw_sha:
        raise ValueError('unary table raw identity mismatch')
    return raw


@lru_cache(maxsize=3)
def _tree(kind, dtype, raw_sha, target, data):
    assert dtype == 'bf16' and raw_sha == PINS[kind][2] and target == TARGET
    values = struct.unpack('<65536H', data)
    low, high = [65535] * 131072, [0] * 131072
    for raw, value in enumerate(values):
        index = 65536 + _ordered(raw)
        low[index] = high[index] = _ordered(value)
    for index in range(65535, 0, -1):
        low[index] = min(low[index*2], low[index*2+1])
        high[index] = max(high[index*2], high[index*2+1])
    return tuple(low), tuple(high)


def image(kind, interval):
    """Exact raw ordered image hull; useful for exhaustive independent tests."""
    lo, hi = interval
    if type(lo) is not int or type(hi) is not int or not 0 <= lo <= hi <= 65535:
        raise ValueError('invalid ordered BF16 interval')
    trees = _tree(kind, 'bf16', PINS[kind][2], TARGET, table_bytes(kind))
    lo += 65536
    hi += 65537
    lower, upper = 65535, 0
    while lo < hi:
        if lo & 1:
            lower, upper = min(lower, trees[0][lo]), max(upper, trees[1][lo])
            lo += 1
        if hi & 1:
            hi -= 1
            lower, upper = min(lower, trees[0][hi]), max(upper, trees[1][hi])
        lo //= 2
        hi //= 2
    return lower, upper


def qualified_image(kind, interval):
    """Refuse unknown or partly out-of-domain input; never weaken a guard."""
    if kind not in PINS or interval is None:
        return None
    lo, hi = interval
    safe_lo, safe_hi = PINS[kind][0]
    if not safe_lo <= lo <= hi <= safe_hi:
        return None
    return image(kind, interval), (TARGET, 'unary_table:' + kind + ':' + PINS[kind][2])
