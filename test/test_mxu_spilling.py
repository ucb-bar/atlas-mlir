"""MXU slot pressure: raw accumulator saves and immutable weight reloads."""

import json
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from test_virtual_spilling import BF16, FP8, ROOT, STATE, compile_program, run, wrap


def mxu_pressure(unit=0, *, weights=True, continuation=True, count=3, reset=False):
    """Independent chains exceed the two physical slots on one owning unit."""
    body = [f'%s0 = "atlas.virtual_start"() : () -> {STATE}']
    serial = 0

    def effect(name, operands, types, result=None, attrs=''):
        nonlocal serial
        previous = f'%s{serial}'
        serial += 1
        state = f'%s{serial}'
        outputs = f'{state}, %{result[0]}' if result else state
        results = f'({STATE}, {result[1]})' if result else STATE
        body.append(f'{outputs} = "atlas.{name}"({previous}'
                    f'{", " if operands else ""}{", ".join(operands)}) '
                    f'{attrs} : ({STATE}{", " if types else ""}{", ".join(types)}) -> {results}')

    acc = f'!atlas.virtual_mxu_acc<{unit}>'
    weight = f'!atlas.virtual_mxu_weight<{unit}>'
    for i in range(count):
        effect('virtual_input_bf16', [], [], (f'b{i}', BF16), f'{{index = {i} : i32}}')
    if continuation:
        effect('virtual_input_fp8', [], [], ('x', FP8), f'{{index = {count} : i32}}')
        for i in range(count if weights else 1):
            effect('virtual_input_fp8', [], [], (f'w{i}', FP8), f'{{index = {count+1+i} : i32}}')
            effect('virtual_mxu_load_weight', [f'%w{i}'], [FP8], (f'weight{i}', weight), f'{{unit = {unit} : i32}}')
    for i in range(count):
        if reset:
            effect('virtual_mxu_reset', ['%x', f'%weight{i if weights else 0}'],
                   [FP8, weight], (f'acc{i}', acc))
        else:
            effect('virtual_mxu_load_acc_bf16', [f'%b{i}'], [BF16], (f'acc{i}', acc), f'{{unit = {unit} : i32}}')
    # Restore the oldest chain last: eviction cannot simply discard it.
    for i in [*range(1, count), 0]:
        handle = f'%acc{i}'
        if continuation:
            effect('virtual_mxu_accumulate', ['%x', f'%weight{i if weights else 0}', handle],
                   [FP8, weight, acc], (f'next{i}', acc))
            handle = f'%next{i}'
        effect('virtual_mxu_readout_bf16', [handle], [acc], (f'y{i}', BF16))
    for i in range(count):
        effect('virtual_output_bf16', [f'%y{i}'], [BF16], attrs=f'{{index = {i} : i32}}')
    body.append(f'return %s{serial} : {STATE}')
    return wrap(body)


def mxu_inputs(continuation=True, count=3, reset=False):
    """Return bytes plus exact integer reference; no compiler packing helper."""
    def packed(cell):
        values = bytearray(2048)
        for row in range(32):
            for col in range(32):
                offset = (col // 16) * 1024 + (row * 16 + col % 16) * 2
                struct.pack_into('<H', values, offset, cell(row, col))
        return bytes(values)

    def bf16(value):
        bits = struct.unpack('<I', struct.pack('<f', value))[0]
        assert bits & 0xffff == 0
        return bits >> 16

    tiles = [packed(lambda r, c, i=i: bf16(i*8 + (r+c) % 4)) for i in range(count)]
    expected = [packed(lambda r, c, i=i: bf16((0 if reset else i*8 + (r+c) % 4) +
                ((2 if reset else 1) * (-1 if c % 2 else 1) * (1 + (r+(c+i) % 32) % 3) if continuation else 0)))
                for i in range(count)]
    if continuation:
        # W[output-column, reduction-index], with one signed unit per row.
        codes = (0x38, 0x40, 0x44)  # exact E4M3 1, 2, 3
        tiles.append(bytes(codes[(r+k) % 3] for r in range(32) for k in range(32)))
        for i in range(count):
            tiles.append(bytes((0xb8 if c % 2 else 0x38) if k == (c+i) % 32 else 0
                               for c in range(32) for k in range(32)))
    return tiles, expected


def mxu_loop(unit=0, iterations=3):
    """Three accumulator/weight chains cross a conditional backedge."""
    source = mxu_pressure(unit)
    lines = source.splitlines()
    cut = next(i for i, line in enumerate(lines) if '"atlas.virtual_mxu_accumulate"' in line)
    acc, weight = f'!atlas.virtual_mxu_acc<{unit}>', f'!atlas.virtual_mxu_weight<{unit}>'
    handles = ', '.join(f'%acc{i}' for i in range(3))
    weights = ', '.join(f'%weight{i}' for i in range(3))
    types = ', '.join([acc]*3 + [weight]*3)
    args = ', '.join([*(f'%a{i}: {acc}' for i in range(3)),
                      *(f'%lw{i}: {weight}' for i in range(3))])
    body = [*lines[:cut], '%zero = arith.constant 0 : i32',
            '%one = arith.constant 1 : i32', f'%limit = arith.constant {iterations} : i32',
            f'cf.br ^loop(%s13, %zero, {handles}, {weights} : {STATE}, i32, {types})',
            f'^loop(%ls: {STATE}, %i: i32, {args}):']
    state = '%ls'
    for i in range(3):
        body.append(f'%ls{i}, %nexta{i} = "atlas.virtual_mxu_accumulate"({state}, %x, %lw{i}, %a{i}) '
                    f': ({STATE}, {FP8}, {weight}, {acc}) -> ({STATE}, {acc})')
        state = f'%ls{i}'
    next_accs = ', '.join(f'%nexta{i}' for i in range(3))
    next_weights = ', '.join(f'%lw{i}' for i in range(3))
    acc_types = ', '.join([acc]*3)
    body += ['%next = arith.addi %i, %one : i32',
             '%more = arith.cmpi slt, %next, %limit : i32',
             (f'cf.cond_br %more, ^loop({state}, %next, {next_accs}, {next_weights} : {STATE}, i32, {types}), '
              f'^done({state}, {next_accs} : {STATE}, {acc_types})'),
             f'^done(%ds: {STATE}, '+', '.join(f'%donea{i}: {acc}' for i in range(3))+'):']
    state = '%ds'
    for i in range(3):
        body.append(f'%ds{i}, %out{i} = "atlas.virtual_mxu_readout_bf16"({state}, %donea{i}) '
                    f': ({STATE}, {acc}) -> ({STATE}, {BF16})')
        state = f'%ds{i}'
    for i in range(3):
        body.append(f'%output{i} = "atlas.virtual_output_bf16"({state}, %out{i}) '
                    f'{{index = {i} : i32}} : ({STATE}, {BF16}) -> {STATE}')
        state = f'%output{i}'
    return '\n'.join([*body, f'return {state} : {STATE}', '}}'])


def mxu_loop_inputs(iterations=3):
    tiles, outputs = mxu_inputs()
    def value(data, offset):
        return struct.unpack('<f', b'\0\0'+data[offset:offset+2])[0]
    # This bounded domain uses exact small integers; every partial accumulator
    # remains BF16-representable, so no rounding assumption enters the oracle.
    outputs = [b''.join(struct.pack('<f', value(seed, j) + iterations*(value(result, j)-value(seed, j)))[2:]
                       for j in range(0, 2048, 2))
               for seed, result in zip(tiles[:3], outputs, strict=True)]
    return tiles, outputs


def mxu_diamond():
    """Both units exceed their slots before a runtime branch and permuted join."""
    acc = [f'!atlas.virtual_mxu_acc<{i//3}>' for i in range(6)]
    weight = [f'!atlas.virtual_mxu_weight<{i//3}>' for i in range(6)]
    lines = [f'%s0 = "atlas.virtual_start"() : () -> {STATE}']
    serial = 0

    def effect(name, operands, types, result, attr=''):
        nonlocal serial
        before = f'%s{serial}'
        serial += 1
        lines.append(f'%s{serial}, %{result[0]} = "atlas.virtual_{name}"({before}'
                     f'{", " if operands else ""}{", ".join(operands)}) {attr} : '
                     f'({STATE}{", " if types else ""}{", ".join(types)}) -> ({STATE}, {result[1]})')

    for i in range(6):
        effect('input_bf16', [], [], (f'b{i}', BF16), f'{{index = {i} : i32}}')
    effect('input_fp8', [], [], ('x', FP8), '{index = 6 : i32}')
    for i in range(6):
        effect('input_fp8', [], [], (f'w{i}', FP8), f'{{index = {i+7} : i32}}')
        effect('mxu_load_weight', [f'%w{i}'], [FP8], (f'weight{i}', weight[i]), f'{{unit = {i//3} : i32}}')
    for i in range(6):
        effect('mxu_load_acc_bf16', [f'%b{i}'], [BF16], (f'acc{i}', acc[i]), f'{{unit = {i//3} : i32}}')
    types = ', '.join([STATE, *acc, *weight])
    values = ', '.join([f'%s{serial}', *(f'%acc{i}' for i in range(6)), *(f'%weight{i}' for i in range(6))])
    lines.append(f'cf.cond_br %take, ^yes({values} : {types}), ^no({values} : {types})')
    for branch in ('yes', 'no'):
        args = ', '.join([f'%{branch}_s: {STATE}',
                          *(f'%{branch}_a{i}: {acc[i]}' for i in range(6)),
                          *(f'%{branch}_w{i}: {weight[i]}' for i in range(6))])
        lines.append(f'^{branch}({args}):')
        state = f'%{branch}_s'
        if branch == 'yes':
            for i in range(6):
                lines.append(f'%yes_s{i}, %next{i} = "atlas.virtual_mxu_accumulate"({state}, %x, %yes_w{i}, %yes_a{i}) '
                             f': ({STATE}, {FP8}, {weight[i]}, {acc[i]}) -> ({STATE}, {acc[i]})')
                state = f'%yes_s{i}'
            carried = [f'%next{i}' for i in range(6)]
        else:
            carried = [f'%no_a{(i//3)*3+(i+1)%3}' for i in range(6)]
        lines.append(f'cf.br ^join({state}, {", ".join(carried)} : {STATE}, {", ".join(acc)})')
    lines.append(f'^join(%joined: {STATE}, '+', '.join(f'%a{i}: {acc[i]}' for i in range(6))+'):')
    state = '%joined'
    for i in range(6):
        lines.append(f'%read_s{i}, %out{i} = "atlas.virtual_mxu_readout_bf16"({state}, %a{i}) '
                     f': ({STATE}, {acc[i]}) -> ({STATE}, {BF16})')
        state = f'%read_s{i}'
    for i in range(6):
        lines.append(f'%out_s{i} = "atlas.virtual_output_bf16"({state}, %out{i}) '
                     f'{{index = {i} : i32}} : ({STATE}, {BF16}) -> {STATE}')
        state = f'%out_s{i}'
    lines.append(f'return {state} : {STATE}')
    return wrap(lines).replace('@test()', '@test(%take: i1)').replace(
        'atlas.input_dram_base =', 'atlas.control_dram_base = 2415984640 : i64, atlas.input_dram_base =')


def mxu_diamond_inputs(take):
    tiles, arithmetic = mxu_inputs(count=6)
    outputs = arithmetic if take & 1 else [tiles[(i//3)*3+(i+1)%3] for i in range(6)]
    return tiles, outputs


class MXUSpillingTest(unittest.TestCase):
    def test_diamond_materializes_both_units_before_scheduling(self):
        source = mxu_diamond()
        self.assertNotEqual(run(source, '--verify-atlas-virtual-stream').returncode, 0)
        for scheduling in ((), ('--schedule-atlas-virtual',)):
            result = run(source, '--spill-atlas-mxu-state', *scheduling,
                         '--spill-atlas-virtual', '--verify-atlas-virtual-stream')
            self.assertEqual(result.returncode, 0, result.stderr)
        _, program = compile_program(source)
        self.assertTrue(program['instructions'])
        for old, new in (
            ('%next0, %next1, %next2', '%yes_a0, %next1, %next2'),
            ('%no_a1, %no_a2, %no_a0', '%no_a1, %no_a2, %no_a1'),
            ('%yes_w0, %yes_a0)', '%yes_w3, %yes_a0)'),
        ):
            mutated = source.replace(old, new, 1)
            self.assertNotEqual(mutated, source)
            rejected = run(mutated, '--spill-atlas-mxu-state')
            self.assertNotEqual(rejected.returncode, 0)

    def test_loop_handles_materialize_before_existing_tile_allocation(self):
        for unit in (0, 1):
            with self.subTest(unit=unit):
                source = mxu_loop(unit)
                self.assertNotEqual(run(source, '--verify-atlas-virtual-stream').returncode, 0)
                rewritten = run(source, '--spill-atlas-virtual', '--verify-atlas-virtual-stream')
                self.assertEqual(rewritten.returncode, 0, rewritten.stderr)
                self.assertIn('atlas.mxu_spill = "save"', rewritten.stdout)
                self.assertIn('atlas.mxu_spill = "restore"', rewritten.stdout)
                scheduled = run(source, '--spill-atlas-mxu-state', '--schedule-atlas-virtual',
                                '--spill-atlas-virtual', '--verify-atlas-virtual-stream')
                self.assertEqual(scheduled.returncode, 0, scheduled.stderr)
                _, program = compile_program(source)
                self.assertTrue(program['instructions'])
                again = run(rewritten.stdout, '--spill-atlas-virtual', '--verify-atlas-virtual-stream')
                self.assertEqual(again.returncode, 0, again.stderr)

    def test_loop_rejects_duplicate_stale_and_captured_accumulators(self):
        source = mxu_loop()
        faults = (
            source.replace('%acc0, %acc1, %acc2,', '%acc0, %acc1, %acc1,', 1),
            source.replace('%nexta0, %nexta1, %nexta2', '%a0, %nexta1, %nexta2', 1),
            source.replace('%x, %lw0, %a0)', '%x, %lw0, %acc0)', 1),
            source.replace('%zero =',
                f'%extra, %dropped = "atlas.virtual_mxu_load_acc_bf16"(%s13, %b0) '
                f'{{unit = 0 : i32}} : ({STATE}, {BF16}) -> ({STATE}, !atlas.virtual_mxu_acc<0>)\n%zero =', 1
            ).replace('cf.br ^loop(%s13,', 'cf.br ^loop(%extra,', 1),
        )
        for mutated in faults:
            with self.subTest(source=mutated):
                self.assertNotEqual(mutated, source)
                result = run(mutated, '--spill-atlas-virtual')
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('MXU', result.stderr)
        result = run(source, '--spill-atlas-virtual=max-spills=0')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('work limit', result.stderr)

    def test_pressure_rewrites_both_units_and_preserves_checks(self):
        for unit in (0, 1):
            for continuation in (False, True):
                with self.subTest(unit=unit, continuation=continuation):
                    source = mxu_pressure(unit, continuation=continuation)
                    rejected = run(source, '--lower-atlas-virtual-to-machine')
                    self.assertNotEqual(rejected.returncode, 0)
                    self.assertIn('live weights' if continuation else 'live accumulators', rejected.stderr)
                    rewritten = run(source, '--spill-atlas-virtual', '--verify-atlas-virtual-stream')
                    self.assertEqual(rewritten.returncode, 0, rewritten.stderr)
                    self.assertIn('atlas.mxu_spill = "save"', rewritten.stdout)
                    self.assertIn('atlas.mxu_spill = "restore"', rewritten.stdout)
                    self.assertIn('atlas.virtual_spill_bf16', rewritten.stdout)
                    if continuation:
                        self.assertIn('atlas.virtual_spill_fp8', rewritten.stdout)
                    machine, program = compile_program(source)
                    self.assertGreater(len(program['instructions']), 0)
                    self.assertIn('kind = "acc_bf16"', machine)
                    # The rewritten form and its physical bindings must both pass.
                    self.assertEqual(run(rewritten.stdout, '--spill-atlas-virtual').returncode, 0)

    def test_capacity_budget_and_stale_handles(self):
        source = mxu_pressure(continuation=False)
        bounded = run(source, '--spill-atlas-virtual=max-spills=0')
        self.assertNotEqual(bounded.returncode, 0)
        self.assertIn('MXU spill rewrite work limit', bounded.stderr)
        too_small = run(source, '--spill-atlas-virtual=vmem-words=256')
        self.assertNotEqual(too_small.returncode, 0)
        self.assertIn('VMEM', too_small.stderr)
        # Consume the same accumulator twice: pressure admission must not hide it.
        stale = source.replace(', %acc0)', ', %acc1)')
        self.assertNotEqual(stale, source)
        rejected = run(stale, '--spill-atlas-virtual')
        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn('stale accumulator handle', rejected.stderr)
        wrong_unit = source.replace('unit = 0 : i32', 'unit = 1 : i32', 1)
        self.assertNotEqual(run(wrong_unit, '--spill-atlas-virtual').returncode, 0)
        fits = mxu_pressure(count=2, continuation=False)
        unchanged = run(fits, '--spill-atlas-virtual')
        self.assertEqual(unchanged.returncode, 0, unchanged.stderr)
        self.assertNotIn('atlas.mxu_spill', unchanged.stdout)

    def test_repeated_eviction(self):
        source = mxu_pressure(count=5, reset=True)
        machine, _ = compile_program(source)
        self.assertIn('atlas.spill_report', machine)
        self.assertIn('mxu_acc_saves', machine)

    def test_stage_driver_exposes_mxu_spills_before_scheduling(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source.mlir'
            source.write_text(mxu_pressure())
            result = subprocess.run([sys.executable, str(ROOT / 'tools/lower_atlas_program.py'),
                                     str(source), '--output-dir', str(root / 'out')],
                                    text=True, capture_output=True, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('atlas.mxu_spill = "save"', (root / 'out/1-virtual.mlir').read_text())
            self.assertTrue(json.loads((root / 'out/3-allocation.json').read_text()))


if __name__ == '__main__':
    unittest.main()
