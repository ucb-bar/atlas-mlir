"""Capture one explicitly authorized attempt; never retry or recover."""
import argparse
from datetime import datetime
import os
from pathlib import Path
import subprocess
import sys
import kit_manifest

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('bundle', type=Path)
p.add_argument('stamp')
p.add_argument('rung', nargs='?', choices=['negative'])
a = p.parse_args()
try:
    if datetime.strptime(a.stamp, '%Y%m%dT%H%M%SZ').strftime('%Y%m%dT%H%M%SZ') != a.stamp:
        raise ValueError('fresh UTC stamp required')
    bundle = kit_manifest.safe_path(a.bundle)
    executable = os.environ.get('NOV_OPENOCD', '')
    if not Path(executable).is_absolute():
        raise ValueError('NOV_OPENOCD must be an absolute executable path')
    elf = 'negative/chip.elf' if a.rung else 'chip.elf'
    kit = kit_manifest.load(bundle, elf)
    # Above the runner's own waits: 180 s startup, the UART deadline, 120 s per RPC command, and cleanup.
    bound = 300 + kit['settings']['timeout'] + 120 * (8 + len(kit['dep']['hash_ranges']))
    runs = kit_manifest.safe_path(bundle / 'runs')
    runs.mkdir(exist_ok=True)
    logs = kit_manifest.safe_path(runs / ('kit-' + a.stamp))
    capture = kit_manifest.safe_path(runs / ('kit-' + a.stamp + '-terminal.txt'))
    if logs.exists():
        raise ValueError('attempt logs must be fresh')
    with capture.open('x') as stream:
        # This invocation is the one authorized attempt; run_kit.py reads NOV_OPENOCD itself.
        command = [sys.executable, '-E', '-s', '-B', str(bundle / 'bench/run_kit.py'),
                   str(bundle), '--elf', elf, '--logs', str(logs), '--allow-transport']
        runner = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        try:
            rc = runner.wait(timeout=bound)
        except subprocess.TimeoutExpired:
            runner.terminate()  # run_kit.py closes OpenOCD and the UART on SIGTERM
            try:
                runner.wait(timeout=150)
            except subprocess.TimeoutExpired:
                runner.kill(); runner.wait(timeout=5)
            stream.flush()
            stream.write('KIT TRANSPORT_ERROR runner exceeded %d s and was stopped\n' % bound)
            rc = 2
        stream.write('rc=%d\n' % rc)
    sys.stdout.write(capture.read_text())
    raise SystemExit(rc)
except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
    print('KIT CAPTURE REFUSED ' + str(error))
    raise SystemExit(2)
