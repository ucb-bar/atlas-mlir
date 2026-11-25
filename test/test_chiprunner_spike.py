"""A linked step ELF runs unchanged in Spike with the Atlas devices; completion comes from its UART records.

Requires the simulator environment (Spike, dtc, g++, RISC-V GCC, and the pinned model);
a missing tool fails the test.
"""
import contextlib
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from atlas_mlir.chiprunner import composed, preflight, run_kit

BASE = 0x200c0000
PROGRAM = r'''
#define UART ((volatile unsigned int *)0x10020000u)
#define STARTS (*(volatile unsigned int *)0x40004u)
static void put(const char *s) { while (*s) { while (UART[0] & 0x80000000u) {} UART[0] = (unsigned char)*s++; } }
void step(void) {
  volatile unsigned char *in = (volatile unsigned char *)0x%xu, *out = in + 64;
  unsigned starts = STARTS;  /* forwarded to the functional runtime: no launch has started */
  for (int i = 0; i < 4; i++) out[i] = (unsigned char)(in[i] + 1 + starts);
  put("ATLAS STATUS %s CYCLES 1\n");
  put("ATLAS HASH y public_output %s\n");  /* the expected output's digest, fixed at build time */
  if (%d) put("ATLAS COMPLETE\n");
  for (;;) {}
}
'''
START = '.section .text.init\n.global _start\n_start:\n  li sp, 0x0800f000\n  call step\n'
LINK = 'ENTRY(_start) SECTIONS { . = 0x08000000; .text : { *(.text.init) *(.text*) *(.rodata*) *(.srodata*) } /DISCARD/ : { *(.comment) } }'


HOST = r'''
void step(void) {
  volatile unsigned char *in = (volatile unsigned char *)0x%xu, *out = in + 1, *workspace = in + 4096;
  for (int i = 0; i < 4096; i++) workspace[i] = in[0];  /* a window page that holds no declared value */
  out[0] = (unsigned char)(workspace[4095] + 1);
  *(volatile unsigned long *)0x%xu = 1;
  for (;;) {}
}
'''


def build(root, name, digest, status='0', complete=1):
    link(root, name, PROGRAM % (BASE, status, digest, complete))


def link(root, name, source):
    compiler = shutil.which('riscv64-unknown-elf-gcc')
    if compiler is None:
        raise AssertionError('RISC-V GCC is required')
    (root / 'step.c').write_text(source)
    (root / 'start.S').write_text(START)
    (root / 'step.ld').write_text(LINK)
    subprocess.run([compiler, '-march=rv64gc', '-mabi=lp64d', '-O2', '-nostdlib', '-nostartfiles', '-T', str(root / 'step.ld'),
                    '-Wl,--no-warn-rwx-segments', str(root / 'start.S'), str(root / 'step.c'), '-o', str(root / name)], check=True)
    for path in ('step.c', 'start.S', 'step.ld'):
        (root / path).unlink()


def bundle(root, timeout=120, steps=1, **options):
    """`steps` chained step ELFs: each reads 4 bytes at BASE and writes them plus one at BASE + 64."""
    data = [bytes(range(1 + i, 5 + i)) for i in range(steps + 1)]
    for i, value in enumerate(data):
        (root / ('v%d.bin' % i)).write_bytes(value)
    output = dict(id='y', address=BASE + 64, bytes=4)
    (root / 'deployment.json').write_text(json.dumps(dict(hash_ranges=[dict(output, role='public_output')])))
    regions = []
    for i in range(steps):
        build(root, 'step%d.elf' % i, hashlib.sha256(data[i + 1]).hexdigest(), **options)
        entry = int.from_bytes((root / ('step%d.elf' % i)).read_bytes()[24:32], 'little')
        source = dict(file='v0.bin') if i == 0 else dict(source=dict(region='s%d' % (i - 1), output='y'))
        regions.append(dict(id='s%d' % i, kind='elf', elf='step%d.elf' % i, entry=entry, deployment='deployment.json', images=[], launches=0,
                            window=dict(address=BASE, bytes=4096), inputs=[dict(id='x', address=BASE, bytes=4, **source)],
                            outputs=[dict(output, reference='v%d.bin' % (i + 1))]))
    manifest = dict(runner=dict(schema='atlas.kit.v1', record_tag='ATLAS'),
                    composed=dict(regions=regions, timeout_seconds=timeout, receiver_baud=115200))
    (root / 'manifest.json').write_text(json.dumps(manifest))
    (root / 'bench').mkdir(); (root / 'bench/atlas-soc.cfg').write_text('# unused by the Spike peer\n')
    names = sorted(str(p.relative_to(root)) for p in root.rglob('*') if p.is_file())
    (root / 'SHA256SUMS').write_text(''.join(hashlib.sha256((root / n).read_bytes()).hexdigest() + '  ' + n + '\n' for n in names))
    return manifest


def spike_bench(root, work, manifest):
    """OpenOCD Tcl, UART, and process peers over Spike: the step ELF last loaded with `load_image` runs at `resume`.

    The peer serves the runner's own Tcl commands; Spike, the Atlas devices, and the UART records are real.
    """
    from atlas_mlir.chiprunner.composed_sim import SimTransport
    regions = {str(root / r['elf']): r for r in manifest['composed']['regions']}
    state = dict(region=None, uart=b'')
    commands = []
    def sim():
        if 'sim' not in state: state['sim'] = SimTransport(root, work, manifest)
        return state['sim']

    class RPC:
        broken = False
        timeout = 1
        def __init__(self, **kwargs): pass
        def command(self, text):
            commands.append(text)
            match = re.fullmatch(r'(load_image|verify_image|dump_image) \{([^}]+)\}(?: (0x[0-9a-f]+) (bin|[0-9]+))?', text)
            if match and match[1] == 'load_image' and match[3] is None:
                state['region'] = regions[match[2]]; sim().prepare(state['region'])
            elif match and match[1] == 'load_image':
                sim().write(int(match[3], 16), Path(match[2]).read_bytes())
            elif match and match[1] == 'dump_image':
                Path(match[2]).write_bytes(sim().read(int(match[3], 16), int(match[4])))
            elif text.startswith('resume '):
                region = state['region']
                if int(text.split()[1], 16) != region['entry']: raise AssertionError('resume differs from the step entry')
                sim().launch(region); sim().wait(region, manifest['composed']['timeout_seconds'])
                state['uart'] = (sim().work / ('launch-%d-%s' % (sim().counter, region['id'])) / 'uart.log').read_bytes()
            return ''
        def close(self): pass

    class UART:
        def __init__(self, *args): pass
        def clear(self): state['uart'] = b''
        def close(self): pass
        def records(self, timeout, console, tag):
            console.write(state['uart'])
            return state['uart']

    class Process:
        def poll(self): return None
        def terminate(self): pass
        def wait(self, **kwargs): return 0

    patches = [patch.object(preflight, 'bench'), patch.object(run_kit, 'UART', UART),
               patch.object(run_kit, 'TclRPC', RPC), patch.object(run_kit, 'launch_openocd', return_value=Process())]
    return patches, commands


BENCH = SimpleNamespace(openocd='/unused/openocd', serial='/unused/serial', tcl_port=1, rpc_timeout=1, start_timeout=1)


class SpikeStepTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / 'bundle'; self.root.mkdir()

    def test_step_elf_completes_with_device_reads_and_uart_records(self):
        bundle(self.root)
        journal = composed.execute(self.root, Path(self.tmp.name) / 'logs')
        self.assertEqual(journal['completed'][0]['outputs']['y']['after'], hashlib.sha256(b'\x02\x03\x04\x05').hexdigest())
        launch = next((Path(self.tmp.name) / 'logs').glob('launch-1-s0'))
        self.assertEqual((launch / 'uart.log').read_text(),
                         'ATLAS STATUS 0 CYCLES 1\nATLAS HASH y public_output %s\nATLAS COMPLETE\n' % hashlib.sha256(b'\x02\x03\x04\x05').hexdigest())

    def test_failing_status_or_missing_completion_is_a_runtime_failure(self):
        for status, complete in (('3', 0), ('0', 0)):
            with self.subTest(status=status):
                shutil.rmtree(self.root); self.root.mkdir()
                bundle(self.root, timeout=20, status=status, complete=complete)
                logs = Path(self.tmp.name) / ('logs-' + status)
                expected = 'did not report completion' if status != '0' else 'did not complete before its deadline'
                with self.assertRaisesRegex(ValueError, 'step s0 failed: .*' + expected) as caught:
                    composed.execute(self.root, logs)
                self.assertNotIsInstance(caught.exception, composed.BoundaryError)

    def test_uart_hash_must_equal_the_dumped_bytes(self):
        bundle(self.root)
        build(self.root, 'step0.elf', hashlib.sha256(b'\x05\x04\x03\x02').hexdigest())
        names = sorted(str(p.relative_to(self.root)) for p in self.root.rglob('*') if p.is_file() and p.name != 'SHA256SUMS')
        (self.root / 'SHA256SUMS').write_text(''.join(hashlib.sha256((self.root / n).read_bytes()).hexdigest() + '  ' + n + '\n' for n in names))
        with self.assertRaisesRegex(ValueError, 'step s0 failed: .*UART hash differs from dumped bytes: y'):
            composed.execute(self.root, Path(self.tmp.name) / 'logs')

    def test_host_code_may_write_its_whole_declared_window(self):
        status = 0x0800f800
        link(self.root, 'host.elf', HOST % (BASE, status))
        entry = int.from_bytes((self.root / 'host.elf').read_bytes()[24:32], 'little')
        (self.root / 'x.bin').write_bytes(b'\x05'); (self.root / 'y.bin').write_bytes(b'\x06')
        region = dict(id='h', kind='host', elf='host.elf', entry=entry, status=dict(address=status, done_value=1),
                      window=dict(address=BASE, bytes=2 * 4096), inputs=[dict(id='x', address=BASE, bytes=1, file='x.bin')],
                      outputs=[dict(id='y', address=BASE + 1, bytes=1, reference='y.bin')])
        (self.root / 'manifest.json').write_text(json.dumps(dict(composed=dict(regions=[region], timeout_seconds=10))))
        names = ['host.elf', 'manifest.json', 'x.bin', 'y.bin']
        (self.root / 'SHA256SUMS').write_text(''.join(hashlib.sha256((self.root / n).read_bytes()).hexdigest() + '  ' + n + '\n' for n in names))
        journal = composed.execute(self.root, Path(self.tmp.name) / 'logs')
        self.assertEqual(journal['completed'][0]['outputs']['y']['after'], hashlib.sha256(b'\x06').hexdigest())

    def test_live_transport_over_spike_matches_simulation(self):
        manifest = bundle(self.root, steps=2)
        simulated = composed.execute(self.root, Path(self.tmp.name) / 'sim')
        patches, commands = spike_bench(self.root, Path(self.tmp.name) / 'spike', manifest)
        with contextlib.ExitStack() as stack:
            for item in patches: stack.enter_context(item)
            live = composed.execute(self.root, Path(self.tmp.name) / 'live', backend='jtag', bench=BENCH)
        outputs = lambda journal: [(r['id'], r['outputs']['y']['after']) for r in journal['completed']]
        self.assertEqual(outputs(live), outputs(simulated))
        self.assertEqual(outputs(live)[1][1], hashlib.sha256(b'\x03\x04\x05\x06').hexdigest())
        self.assertEqual(sum(text.startswith('resume') for text in commands), 2)
        self.assertEqual(len(list((Path(self.tmp.name) / 'live').glob('uart-*.log'))), 2)


if __name__ == '__main__':
    unittest.main()
