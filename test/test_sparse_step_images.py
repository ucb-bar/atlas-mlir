"""Sparse step images: the zero spans a step ELF clears at startup, and the bytes a loader then leaves out."""
import json
from pathlib import Path
import re
import shutil
import struct
import subprocess
import tempfile
import unittest

from atlas_mlir.chiprunner.composed_sim import CLEARED_TABLE, _elf_loads, cleared_spans, step_loads
from atlas_mlir.chiprunner.composed_tcl import TclTransport
from atlas_mlir.deployment_link import _HARNESS, WINDOW_NOT_CLEARED, window_zero_spans

HOST, WINDOW = 0x08000000, 0x200c0000


def elf(segments, table=None):
    """ELF64 RISC-V image, one PT_LOAD per (address, flags, data), with an optional cleared-span table symbol in the
    first segment's trailing bytes."""
    raw = bytearray(64 + 56 * len(segments)); raw[:6] = b'\x7fELF\x02\x01'
    struct.pack_into('<H', raw, 18, 243); struct.pack_into('<QQ', raw, 24, HOST, 64); struct.pack_into('<HH', raw, 54, 56, len(segments))
    symbol = None
    for i, (address, flags, data) in enumerate(segments):
        if i == 0 and table is not None:
            words = b''.join(struct.pack('<QQ', *span) for span in [*table, (0, 0)])
            symbol = (address + len(data), len(words)); data = data + words
        struct.pack_into('<IIQQQQQQ', raw, 64 + 56 * i, 1, flags, len(raw), address, address, len(data), len(data), 8)
        raw += data
    if symbol is None:
        return bytes(raw)
    strings = b'\0' + CLEARED_TABLE + b'\0'
    symbols = bytes(24) + struct.pack('<IBBHQQ', 1, 0x11, 0, 1, *symbol)
    strings_at = len(raw); raw += strings
    symbols_at = len(raw); raw += symbols
    sections_at = len(raw)
    raw += bytes(64)
    raw += struct.pack('<IIQQQQIIQQ', 0, 2, 0, 0, symbols_at, len(symbols), 2, 1, 8, 24)
    raw += struct.pack('<IIQQQQIIQQ', 0, 3, 0, 0, strings_at, len(strings), 0, 0, 1, 0)
    struct.pack_into('<Q', raw, 40, sections_at); struct.pack_into('<HHH', raw, 58, 64, 3, 0)
    return bytes(raw)


def window_image():
    """Scratch zeros at +0, an input (zeros, the runner writes it) at +256, data at +320, output poison at +512, and
    scratch zeros at +576."""
    return bytes(256) + bytes(64) + b'\x07' * 192 + b'\xa5' * 64 + bytes(448)


def memory(pieces, writes, cleared, background):
    """Memory after the loader's pieces, the runner's writes, and the harness clearing, over stale bytes."""
    mem = dict(background)
    for address, data, *_ in pieces:
        mem.update((address + i, b) for i, b in enumerate(data))
    for address, data in writes:
        mem.update((address + i, b) for i, b in enumerate(data))
    for address, size in cleared:
        mem.update((address + i, 0) for i in range(size))
    return mem


class ClearedSpanTests(unittest.TestCase):
    def setUp(self):
        self.table = [(WINDOW, 256), (WINDOW + 576, 448)]
        self.raw = elf([(HOST, 5, b'\x13' * 32), (WINDOW, 6, window_image())], self.table)

    def test_table_spans_are_read_and_an_elf_without_one_clears_nothing(self):
        self.assertEqual(cleared_spans(self.raw), self.table)
        self.assertEqual(cleared_spans(elf([(HOST, 5, b'\x13' * 32), (WINDOW, 6, window_image())])), [])

    def test_a_span_with_a_nonzero_image_byte_or_out_of_order_refuses(self):
        for table in ([(WINDOW + 512, 64)], [(WINDOW + 576, 448), (WINDOW, 256)], [(WINDOW, 300), (WINDOW + 256, 8)]):
            with self.assertRaises(ValueError):
                cleared_spans(elf([(HOST, 5, b'\x13' * 32), (WINDOW, 6, window_image())], table))

    def test_loader_leaves_out_cleared_spans_and_runner_ranges_and_memory_is_unchanged(self):
        written = [(WINDOW + 256, 64), (WINDOW + 512, 64)]
        pieces, left_out = step_loads(self.raw, written)
        # The zero input and the output poison are both left out: the runner writes each in full before the start.
        self.assertEqual(left_out, 256 + 448 + 64 + 64)
        self.assertEqual([(a, len(d)) for a, d, *_ in pieces if a >= WINDOW], [(WINDOW + 320, 192)])
        full = [(a, d) for a, d, *_ in _elf_loads(self.raw)[1]]
        writes = [(WINDOW + 256, b'\x01\x02' * 32), (WINDOW + 512, b'\xa5' * 64)]
        for byte in (0x00, 0x5a):
            background = {WINDOW + i: byte for i in range(1024)}
            self.assertEqual(memory(pieces, writes, self.table, background), memory(full, writes, [], background))
        # Without the startup clearing or the runner's poison write, stale bytes stay: the check above depends on both.
        stale = {WINDOW + i: 0x5a for i in range(1024)}
        self.assertNotEqual(memory(pieces, writes, [], stale), memory(full, writes, [], stale))
        self.assertNotEqual(memory(pieces, writes[:1], self.table, stale), memory(full, writes[:1], [], stale))

    def test_a_runner_range_not_inside_one_load_is_loaded_as_packaged(self):
        """A range that leaves the file-backed bytes is not the loader's to drop; it loads as packaged."""
        pieces, left_out = step_loads(self.raw, [(HOST + 16, 4096)])
        self.assertEqual(left_out, 256 + 448)
        self.assertEqual([(a, len(d)) for a, d, *_ in pieces if a >= WINDOW], [(WINDOW + 256, 320)])

    def test_a_runner_range_inside_a_cleared_span_refuses(self):
        with self.assertRaises(ValueError):
            step_loads(self.raw, [(WINDOW + 16, 4)])


class LinkTableTests(unittest.TestCase):
    def test_zero_runs_outside_bound_rows_of_at_least_the_alignment(self):
        deployment = dict(window=dict(address=WINDOW), inputs=[dict(address=WINDOW + 256, bytes=64)],
                          outputs=[dict(address=WINDOW + 512, bytes=64)], parameters=dict(address=WINDOW + 960, bytes=64))
        # The 32 zero bytes after the parameters are padding and stay loaded; the 64 after them are cleared.
        image = window_image() + bytes(32) + b'\x01' + bytes(64)
        self.assertEqual(window_zero_spans(deployment, image), [(WINDOW, 256), (WINDOW + 576, 384), (WINDOW + 1057, 64)])

    def test_harness_clears_and_reads_back_the_table_before_binding(self):
        main = _HARNESS[_HARNESS.index('int main(void)'):]
        self.assertIn('const uint64_t atlas_window_zero[][2]={ATLAS_WINDOW_ZERO{0,0}};', _HARNESS)
        self.assertLess(main.index('int result=clear_window(zero);'), main.index('if(!result) result=atlas_host_init('))
        self.assertLess(main.index('atlas_host_init('), main.index('merlin_deployment_bind('))

    def test_a_short_clear_reports_its_status(self):
        """The harness clear runs on the host over stale bytes; a memset that misses one byte must fail closed."""
        start = _HARNESS.index('static int clear_window(')
        function = _HARNESS[start:_HARNESS.index('\n}\n', start) + 3]
        program = '''#include <stdint.h>
#include <stdio.h>
#include <string.h>
#define WINDOW_NOT_CLEARED (%d)
static void fence(void) {}
static size_t short_by;
static void *short_memset(void *to, int value, size_t bytes) { return memset(to, value, bytes - short_by); }
#define memset short_memset
%s
int main(void) {
  static _Alignas(8) unsigned char window[1024];
  /* The short byte is read as part of a word (aligned spans), then as a tail byte (unaligned spans). */
  const uint64_t spans[2][3][2] = {{{(uintptr_t)window, 256}, {(uintptr_t)window + 576, 448}, {0, 0}},
                                   {{(uintptr_t)window + 3, 250}, {(uintptr_t)window + 577, 443}, {0, 0}}};
  for (int table = 0; table < 2; table++)
    for (short_by = 0; short_by < 2; short_by++) {
      memset(window, 0x5a, sizeof(window));
      int status = clear_window(spans[table]);
      size_t last = spans[table][0][0] + spans[table][0][1] - 1 - (uintptr_t)window;
      printf("%%d %%d %%d\\n", status, window[last], window[last + 1]);
    }
  return 0;
}
''' % (WINDOW_NOT_CLEARED, function)
        compiler = shutil.which('gcc') or shutil.which('cc')
        self.assertIsNotNone(compiler, 'a host C compiler is required')
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / 'clear.c').write_text(program)
            subprocess.run([compiler, '-O2', '-Wall', '-Werror', '-o', str(Path(tmp) / 'clear'), str(Path(tmp) / 'clear.c')],
                           check=True)
            lines = subprocess.run([str(Path(tmp) / 'clear')], check=True, capture_output=True, text=True).stdout.split('\n')
        self.assertEqual(lines[:4], ['0 0 90', '%d 90 90' % WINDOW_NOT_CLEARED] * 2)


class TclLoadTests(unittest.TestCase):
    def test_step_loads_its_sparse_image_and_other_images_load_as_packaged(self):
        class RPC:
            timeout, broken = 1, False
            def __init__(self): self.commands = []
            def command(self, text): self.commands.append(text); return ''
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = elf([(HOST, 5, b'\x13' * 32), (WINDOW, 6, window_image())], [(WINDOW, 256), (WINDOW + 576, 448)])
            (root / 'step.elf').write_bytes(raw)
            (root / 'plain.elf').write_bytes(elf([(HOST, 5, b'\x13' * 32), (WINDOW, 6, b'\x01' * 64)]))
            rpc = RPC(); transport = TclTransport(root, root, {}, rpc=rpc)
            region = dict(kind='elf', elf='step.elf', inputs=[dict(address=WINDOW + 256, bytes=64)], weights=[],
                          outputs=[dict(address=WINDOW + 512, bytes=64)])
            transport.prepare(region)
            transport.prepare(dict(region, elf='plain.elf'))
            loaded = [re.fullmatch(r'(load|verify)_image \{([^}]+)\}', text).groups() for text in rpc.commands[1:]]
            image = Path(loaded[0][1])
            self.assertEqual(loaded, [('load', str(image)), ('verify', str(image)),
                                      ('load', str(root / 'plain.elf')), ('verify', str(root / 'plain.elf'))])
            expected = step_loads(raw, [(WINDOW + 256, 64), (WINDOW + 512, 64)])[0]
            self.assertEqual([(a, d) for a, d, *_ in _elf_loads(image.read_bytes())[1]], [(a, d) for a, d, *_ in expected])
            self.assertEqual(_elf_loads(image.read_bytes())[0], HOST)


if __name__ == '__main__':
    unittest.main()
