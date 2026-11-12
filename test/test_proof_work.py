"""Process-wide proof measurement preserves output and counts repeated work."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import unittest

from test_indexed_slots import source

ROOT = Path(__file__).resolve().parents[1]
BIN = Path(os.environ.get("ATLAS_OOT_BIN_DIR", ROOT / "build/bin"))
CONTRACTS = ("dma", "mxu", "tile", "cfg", "source_memory", "buffer")
COMPILE = ("--spill-atlas-virtual", "--lower-atlas-virtual-to-machine",
           "--insert-atlas-dma-waits", "--insert-atlas-delays",
           "--verify-atlas-generated-schedule", "--verify-atlas-timing")


def run(text, *passes, reporting=False, tool="atlas-opt"):
    env = os.environ.copy()
    env.pop("ATLAS_REPORT_PROOF_WORK", None)
    if reporting:
        env["ATLAS_REPORT_PROOF_WORK"] = "1"
    return subprocess.run([str(BIN / tool), *passes, "-"], input=text,
                          capture_output=True, text=True, env=env)


def report(result):
    reports = [json.loads(line) for line in result.stderr.splitlines()
               if line.startswith('{"schema":"atlas-proof-work-v1"')]
    if len(reports) != 1:
        raise AssertionError(f"expected one process report, got {len(reports)}: {result.stderr}")
    return reports[0]


def larger_source(extra=16):
    text = source()
    load = next(line for line in text.splitlines() if '%s4, %loaded =' in line)
    previous, lines = '%s4', []
    for n in range(extra):
        state = f'%extra_state{n}'
        lines.append(load.replace('%s4', state).replace('%s3', previous)
                     .replace('%loaded', f'%extra_value{n}'))
        previous = state
    text = text.replace('        %zero =', '\n'.join(lines) + '\n        %zero =')
    return text.replace('"atlas.virtual_output_bf16"(%s4,', f'"atlas.virtual_output_bf16"({previous},')


class ProofWorkTests(unittest.TestCase):
    def compile(self, text, reporting=False):
        result = run(text, *COMPILE, reporting=reporting)
        self.assertEqual(result.returncode, 0, result.stderr)
        return result

    def test_reporting_preserves_machine_and_emitted_program(self):
        plain = self.compile(source())
        measured = self.compile(source(), reporting=True)
        self.assertEqual(plain.stdout, measured.stdout)
        self.assertNotIn('atlas-proof-work-v1', plain.stderr)
        counts = report(measured)
        self.assertGreater(counts['six_contract_work'], 0)
        expected = sum(v['record_visits'] + v['transfer_visits']
                       for key, v in counts['checks'].items() if key in CONTRACTS)
        self.assertEqual(counts['six_contract_work'], expected)
        self.assertEqual(counts['total_work'], counts['six_contract_work'] + counts['auxiliary_work'])
        for enabled in (False, True):
            emitted = run(measured.stdout, '--program-json', tool='atlas-emit', reporting=enabled)
            self.assertEqual(emitted.returncode, 0, emitted.stderr)
            if enabled:
                self.assertEqual(emitted.stdout, plain_program)
                report(emitted)
            else:
                plain_program = emitted.stdout

    def test_repeated_passes_accumulate_without_reset(self):
        machine = self.compile(source()).stdout
        once = run(machine, '--pass-pipeline=builtin.module(verify-atlas-generated-schedule)', reporting=True)
        twice = run(machine, '--pass-pipeline=builtin.module(verify-atlas-generated-schedule,verify-atlas-generated-schedule)', reporting=True)
        self.assertEqual(once.returncode, 0, once.stderr)
        self.assertEqual(twice.returncode, 0, twice.stderr)
        one, two = report(once), report(twice)
        for key in CONTRACTS:
            self.assertGreater(one['checks'][key]['invocations'], 0)
            for event in ('invocations', 'record_visits', 'transfer_visits'):
                self.assertEqual(two['checks'][key][event], 2 * one['checks'][key][event], (key, event))
        self.assertEqual(two['total_work'], 2 * one['total_work'])

    def test_whole_program_size_and_independent_processes(self):
        small = report(self.compile(source(), reporting=True))
        large = report(self.compile(larger_source(), reporting=True))
        again = report(self.compile(source(), reporting=True))
        self.assertEqual(small, again)
        self.assertGreater(large['six_contract_work'], small['six_contract_work'])
        for key in CONTRACTS:
            def work(counts):
                row = counts['checks'][key]
                return row['record_visits'] + row['transfer_visits']
            self.assertGreater(work(large), work(small), key)

    def test_reporting_preserves_bound_rejection(self):
        invalid = source(mask=1023)  # word indices exceed the 512-word slot.
        plain = run(invalid, *COMPILE)
        measured = run(invalid, *COMPILE, reporting=True)
        self.assertNotEqual(plain.returncode, 0)
        self.assertEqual(plain.returncode, measured.returncode)
        self.assertEqual(plain.stdout, measured.stdout)
        report(measured)


if __name__ == '__main__':
    unittest.main()
