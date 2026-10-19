"""Pressure-driven VMEM spills, raw copies and checker mutations."""
from __future__ import annotations

import json
import os
import re
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BIN = Path(os.environ.get("ATLAS_OOT_BIN_DIR", ROOT / "build/bin"))
STATE, BF16, FP8, SAVED = (f"!atlas.virtual_{name}" for name in ("state", "bf16", "fp8", "spill"))


def run(source, *passes, tool="atlas-opt"):
    flags = ["--mlir-print-op-on-diagnostic=false"] if tool == "atlas-opt" else []
    return subprocess.run([str(BIN / tool), *flags, *passes, "-"], input=source,
                          text=True, capture_output=True, check=False)


def pressure(count=32, fp8=False):
    tile, suffix = (FP8, "fp8") if fp8 else (BF16, "bf16")
    body = [f'%s0 = "atlas.virtual_start"() : () -> {STATE}']
    for i in range(count):
        body.append(f'%s{i+1}, %v{i} = "atlas.virtual_input_{suffix}"(%s{i}) {{index = {i} : i32}} : ({STATE}) -> ({STATE}, {tile})')
    for i in range(count):
        if fp8:
            body += [f'%a{i} = arith.constant {0x90020000 + 1024*i - 2**32} : i32',
                     f'%n{i} = arith.constant 1024 : i32',
                     f'%s{count+i+1} = "atlas.virtual_dma_store_fp8"(%s{count+i}, %v{i}, %a{i}, %n{i}) : ({STATE}, {tile}, i32, i32) -> {STATE}']
        else:
            body.append(f'%s{count+i+1} = "atlas.virtual_output_bf16"(%s{count+i}, %v{i}) {{index = {i} : i32}} : ({STATE}, {BF16}) -> {STATE}')
    body.append(f'return %s{count*2} : {STATE}')
    return wrap(body)


def wrap(body):
    return ('module { func.func @test() -> '+STATE+' attributes '
            '{atlas.input_dram_base = 2415919104 : i64, atlas.output_dram_base = 2416050176 : i64} {\n'
            +'\n'.join(body)+'\n}}\n')


def raw_roundtrip(fp8=False):
    tile, suffix = (FP8, "fp8") if fp8 else (BF16, "bf16")
    body = [f'%s0 = "atlas.virtual_start"() : () -> {STATE}',
            f'%s1, %v = "atlas.virtual_input_{suffix}"(%s0) {{index = 0 : i32}} : ({STATE}) -> ({STATE}, {tile})',
            f'%s2, %saved = "atlas.virtual_spill_{suffix}"(%s1, %v) {{slot_word = 135168 : i32}} : ({STATE}, {tile}) -> ({STATE}, {SAVED})',
            f'%s3, %again = "atlas.virtual_reload_{suffix}"(%s2, %saved) : ({STATE}, {SAVED}) -> ({STATE}, {tile})']
    if fp8:
        body += ['%addr = arith.constant -1878917120 : i32', '%size = arith.constant 1024 : i32',
                 f'%s4 = "atlas.virtual_dma_store_fp8"(%s3, %again, %addr, %size) : ({STATE}, {FP8}, i32, i32) -> {STATE}']
    else:
        body += [f'%s4 = "atlas.virtual_output_bf16"(%s3, %again) {{index = 0 : i32}} : ({STATE}, {BF16}) -> {STATE}']
    return wrap([*body, f'return %s4 : {STATE}'])


def cfg_pressure(kind="edge", fp8=False, taken=True):
    """Keep input tiles live across control flow without tile block arguments."""
    count = 33 if fp8 else 32
    source = pressure(count, fp8)
    cut = source.index('%a0 =' if fp8 else f'%s{count+1} =')
    prefix, outputs = source[:cut], source[cut:]
    outputs = outputs.replace(f'(%s{count},', '(%incoming,', 1)

    def copies(tag):
        return '\n'.join(
            f'%copy_{tag}_{i} = "atlas.virtual_vpu_unary"(%v{i}) {{kind = "mov"}} : ({BF16}) -> {BF16}'
            for i in range(count))

    if kind == "edge":
        flow = f'cf.br ^use(%s{count} : {STATE})\n^use(%incoming: {STATE}):\n'
    elif kind == "diamond":
        flow = '\n'.join([
            f'%choose = arith.constant {int(taken)} : i1',
            f'cf.cond_br %choose, ^left(%s{count} : {STATE}), ^right(%s{count} : {STATE})',
            f'^left(%left_state: {STATE}):', copies("left"),
            f'cf.br ^use(%left_state : {STATE})',
            f'^right(%right_state: {STATE}):', copies("right"),
            f'cf.br ^use(%right_state : {STATE})',
            f'^use(%incoming: {STATE}):\n'])
    elif kind == "loop":
        flow = '\n'.join([
            '%zero = arith.constant 0 : i32', '%one = arith.constant 1 : i32',
            '%limit = arith.constant 3 : i32',
            f'cf.br ^loop(%s{count}, %zero : {STATE}, i32)',
            f'^loop(%loop_state: {STATE}, %i: i32):', copies("loop"),
            '%next = arith.addi %i, %one : i32',
            '%more = arith.cmpi slt, %next, %limit : i32',
            f'cf.cond_br %more, ^loop(%loop_state, %next : {STATE}, i32), ^use(%loop_state : {STATE})',
            f'^use(%incoming: {STATE}):\n'])
    else:
        raise ValueError(kind)
    return prefix + flow + outputs


def compile_program(source, spill=True, timing="--insert-atlas-delays"):
    passes = (["--spill-atlas-virtual"] if spill else []) + [
        "--lower-atlas-virtual-to-machine", "--insert-atlas-dma-waits", timing,
        "--verify-atlas-generated-schedule", "--verify-atlas-timing"]
    machine = run(source, *passes)
    if machine.returncode:
        raise AssertionError(machine.stderr)
    program = run(machine.stdout, "--program-json", tool="atlas-emit")
    if program.returncode:
        raise AssertionError(program.stderr)
    return machine.stdout, json.loads(program.stdout)


def execute_copies(program, inputs, visits=None, runtime=None):
    """Bounded byte/control check, not a numerical or timing simulator.

    Treat DMA as complete at issue; generated timing checks separately cover
    hazards. No target compiler packing or allocation helper supplies bytes.
    """
    runtime = {} if runtime is None else runtime
    x = runtime.setdefault('x', [0] * 32)
    registers = runtime.setdefault('registers', [bytes(1024) for _ in range(64)])
    vmem, dram = runtime.setdefault('vmem', {}), runtime.setdefault('dram', {})
    for address, data in inputs:
        dram.update((address+i, byte) for i, byte in enumerate(data))
    pc, redirect = 0, None
    instructions = program["instructions"]
    for _ in range(100000):
        row = instructions[pc]
        if visits is not None:
            visits[pc] = visits.get(pc, 0) + 1
        previous, redirect = redirect, None
        op, f = row["operation"], row["fields"]
        if op == "atlas.upper":
            x[f["dst"]] = f["immediate"] << 12
        elif op == "atlas.alu_imm":
            source, imm = x[f["src"]], f["immediate"]
            if f["kind"] == "addi":
                value = source + imm
            elif f["kind"] == "xori":
                value = source ^ imm
            elif f["kind"] == "andi":
                value = source & imm
            elif f["kind"] == "sltiu":
                value = int(source < (imm & 0xffffffff))
            else:
                raise AssertionError(f)
            x[f["dst"]] = value & 0xffffffff
        elif op == "atlas.alu_reg":
            a, b = x[f["lhs"]], x[f["rhs"]]
            signed = lambda v: v if v < 2**31 else v - 2**32
            if f["kind"] == "add":
                value = a + b
            elif f["kind"] == "xor":
                value = a ^ b
            elif f["kind"] == "slt":
                value = int(signed(a) < signed(b))
            elif f["kind"] == "sltu":
                value = int(a < b)
            else:
                raise AssertionError(f)
            x[f["dst"]] = value & 0xffffffff
        elif op == "atlas.branch":
            assert f["kind"] == "bne"
            if x[f["lhs"]] != x[f["rhs"]]:
                redirect = pc + f["offset_bytes"] // 2
        elif op == "atlas.jump":
            assert f["dst"] == 0
            redirect = pc + f["offset"] // 2
        elif op == "atlas.dma":
            address, external, size = x[f["reg"]]*4, x[f["dram"]], x[f["size"]]
            src, dst, start, end = ((dram, vmem, external, address) if f["direction"] == "load"
                                   else (vmem, dram, address, external))
            for i in range(size):
                dst[end+i] = src[start+i]
        elif op == "atlas.vstore":
            address = x[f["base"]]*4 + f["offset"]
            vmem.update((address+i, byte) for i, byte in enumerate(registers[f["src"]]))
        elif op == "atlas.vload":
            address = x[f["base"]]*4 + f["offset"]
            registers[f["dst"]] = bytes(vmem[address+i] for i in range(1024))
        elif op == "atlas.scalar_store":
            assert f["kind"] == "sw"
            address = (x[f["base"]] + f["offset"]) & 0xffffffff
            vmem.update((address+i, byte) for i, byte in enumerate(x[f["src"]].to_bytes(4, 'little')))
        elif op == "atlas.scalar_load":
            assert f["kind"] == "lw"
            address = (x[f["base"]] + f["offset"]) & 0xffffffff
            x[f["dst"]] = int.from_bytes(bytes(vmem[address+i] for i in range(4)), 'little')
        elif op == "atlas.vpu_unary":
            assert f["kind"] == "mov"
            registers[f["dst"]:f["dst"]+2] = registers[f["src"]:f["src"]+2]
        elif op == "atlas.trap":
            assert f["kind"] == "ecall" and previous is None
            return dram
        else:
            assert op in {"atlas.delay", "atlas.dma_wait", "atlas.dma_config", "atlas.csr", "atlas.trap"}, op
        x[0] = 0
        assert previous is None or redirect is None, "control transfer in delay slot"
        pc = previous if previous is not None else pc + 1
    raise AssertionError("byte/control execution work limit exceeded")


class VirtualSpillingTest(unittest.TestCase):
    def test_mixed_fp8_bf16_pressure_preserves_the_shared_register_file(self):
        source = pressure(16)
        start = f'%s0 = "atlas.virtual_start"() : () -> {STATE}'
        source = source.replace(start, start + '\n' +
            f'%extra_state, %extra = "atlas.virtual_input_fp8"(%s0) {{index = 63 : i32}} : ({STATE}) -> ({STATE}, {FP8})')
        source = source.replace('"atlas.virtual_input_bf16"(%s0)', '"atlas.virtual_input_bf16"(%extra_state)')
        source = source.replace(f'return %s32 : {STATE}', '\n'.join([
            '%address = arith.constant -1878851584 : i32',
            '%bytes = arith.constant 1024 : i32',
            f'%done = "atlas.virtual_dma_store_fp8"(%s32, %extra, %address, %bytes) : ({STATE}, {FP8}, i32, i32) -> {STATE}',
            f'return %done : {STATE}']))
        self.assertNotEqual(run(source, "--lower-atlas-virtual-to-machine").returncode, 0)
        _, program = compile_program(source)
        inputs = [(0x90000000+i*2048, bytes((j*7+i*31) % 256 for j in range(2048))) for i in range(16)]
        inputs.append((0x90000000+63*2048, bytes(range(256))*4))
        memory = execute_copies(program, inputs)
        for i, (_, data) in enumerate(inputs[:16]):
            self.assertEqual(bytes(memory[0x90020000+i*2048+j] for j in range(2048)), data)
        self.assertEqual(bytes(memory[0x90030000+j] for j in range(1024)), inputs[-1][1])

    def test_fp8_pressure_recovers_and_preserves_all_inputs(self):
        source = pressure(33, fp8=True)
        self.assertNotEqual(run(source, "--lower-atlas-virtual-to-machine").returncode, 0)
        _, program = compile_program(source)
        self.assertIn('"atlas.virtual_spill_fp8"', run(source, "--spill-atlas-virtual").stdout)
        inputs = [(0x90000000+i*2048, bytes((j*19+i*13) % 256 for j in range(1024))) for i in range(33)]
        memory = execute_copies(program, inputs)
        for i, (address, data) in enumerate(inputs):
            self.assertEqual(bytes(memory[0x90020000+i*1024+j] for j in range(1024)), data)
            self.assertEqual(bytes(memory[address+j] for j in range(1024)), data)

    def test_pressure_failure_recovers_with_real_spills_and_exact_output_bytes(self):
        failed = run(pressure(), "--lower-atlas-virtual-to-machine")
        self.assertNotEqual(failed.returncode, 0)
        self.assertIn("exceeds 31 physical pairs", failed.stderr)
        machine, program = compile_program(pressure())
        self.assertIn("stores = 2", machine)
        inputs = [(0x90000000+i*2048, bytes((j*19+i*13) % 256 for j in range(2048))) for i in range(32)]
        memory = execute_copies(program, inputs)
        for i, (address, data) in enumerate(inputs):
            self.assertEqual(bytes(memory[0x90020000+i*2048+j] for j in range(2048)), data)
            self.assertEqual(bytes(memory[address+j] for j in range(2048)), data)

    def test_a_spilled_loaded_tile_is_saved_after_its_first_use(self):
        # A loaded tile stays in its staging window until its first use, so a
        # save at its launch would wait for the transfer there.
        body = [f'%s0 = "atlas.virtual_start"() : () -> {STATE}',
                '%a_addr = arith.constant -1879048192 : i32', '%f_addr = arith.constant -1879046144 : i32',
                '%size = arith.constant 2048 : i32', '%half = arith.constant 1024 : i32',
                f'%s1, %a = "atlas.virtual_dma_load_bf16"(%s0, %a_addr, %size) : ({STATE}, i32, i32) -> ({STATE}, {BF16})',
                f'%s2, %f = "atlas.virtual_dma_load_fp8"(%s1, %f_addr, %half) : ({STATE}, i32, i32) -> ({STATE}, {FP8})',
                f'%u = "atlas.virtual_vpu_unary"(%a) {{kind = "exp"}} : ({BF16}) -> {BF16}']
        body += [f'%v{i} = "atlas.virtual_vpu_unary"(%u) {{kind = "relu"}} : ({BF16}) -> {BF16}' for i in range(15)]
        state = "s2"
        for i, value in enumerate([f"v{i}" for i in range(15)] + ["z"]):
            if value == "z":
                body.append(f'%z = "atlas.virtual_vpu_binary"(%a, %u) {{kind = "add"}} : ({BF16}, {BF16}) -> {BF16}')
            body += [f'%o{i} = arith.constant {0x90010000 + 2048 * i - 2**32} : i32',
                     f'%t{i} = "atlas.virtual_dma_store_bf16"(%{state}, %{value}, %o{i}, %size) : ({STATE}, {BF16}, i32, i32) -> {STATE}']
            state = f"t{i}"
        body.append(f'return %{state} : {STATE}')
        result = run(wrap(body), '--spill-atlas-virtual')
        self.assertEqual(result.returncode, 0, result.stderr)
        ops = [line.strip() for line in result.stdout.splitlines() if ' = ' in line]
        load = next(n for n, line in enumerate(ops) if '"atlas.virtual_dma_load_bf16"' in line)
        tile = ops[load].split(' = ')[0].split(', ')[1]
        uses = [n for n, line in enumerate(ops) if n > load and re.search(rf'[(, ]{tile}[,)]', line.split(' = ', 1)[1])]
        saves = [n for n in uses if '"atlas.virtual_spill_bf16"' in ops[n]]
        direct = sorted(set(uses) - set(saves))
        self.assertEqual(len(saves), 1)
        self.assertTrue(direct and direct[0] < saves[0], "the first use reads the staged tile, then it is saved")
        lowered = run(result.stdout, '--lower-atlas-virtual-to-machine', '--insert-atlas-dma-waits')
        self.assertEqual(lowered.returncode, 0, lowered.stderr)

    def test_loaded_tiles_staged_for_their_windows_spill(self):
        # Every load launches before any store, so each launch past the eighth
        # stages an earlier tile to take its window, long before its one use.
        source = run(pressure(40), "--lower-atlas-virtual-boundary-to-dma").stdout
        _, program = compile_program(source)
        inputs = [(0x90000000+i*2048, bytes((j*19+i*13) % 256 for j in range(2048))) for i in range(40)]
        memory = execute_copies(program, inputs)
        for i, (_, data) in enumerate(inputs):
            self.assertEqual(bytes(memory[0x90020000+i*2048+j] for j in range(2048)), data)

    def test_no_pressure_does_not_insert_spills(self):
        result = run(pressure(2), "--spill-atlas-virtual")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("stores = 0", result.stdout)
        self.assertNotIn('"atlas.virtual_spill_bf16"', result.stdout)

    def test_raw_fp8_and_both_bf16_halves_survive(self):
        for fp8 in (False, True):
            for timing in ("--insert-atlas-delays", "--schedule-atlas-stream"):
                with self.subTest(fp8=fp8, timing=timing):
                    _, program = compile_program(raw_roundtrip(fp8), timing=timing)
                    data = bytes((i*17+i//256) % 256 for i in range(1024 if fp8 else 2048))
                    memory = execute_copies(program, [(0x90000000, data)])
                    self.assertEqual(bytes(memory[0x90020000+i] for i in range(len(data))), data)

    def test_spill_budget_exhaustion_and_work_limit_are_errors(self):
        for option, diagnostic in (("--spill-atlas-virtual=vmem-words=256", "frame exhausted"),
                                   ("--spill-atlas-virtual=max-spills=0", "work limit")):
            result = run(pressure(), option)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(diagnostic, result.stderr)
        result = run(raw_roundtrip(), "--spill-atlas-virtual=vmem-words=256")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("frame exhausted", result.stderr)

    def test_spills_are_deterministic_idempotent_and_remain_checked_after_scheduling(self):
        source = pressure()
        first = run(source, "--spill-atlas-virtual")
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertEqual(run(source, "--spill-atlas-virtual").stdout, first.stdout)
        self.assertEqual(run(first.stdout, "--spill-atlas-virtual").stdout, first.stdout)
        scheduled = run(first.stdout, "--schedule-atlas-virtual", "--verify-atlas-virtual-stream")
        self.assertEqual(scheduled.returncode, 0, scheduled.stderr)
        _, program = compile_program(scheduled.stdout)
        inputs = [(0x90000000+i*2048, bytes((j*31+i*7) % 256 for j in range(2048))) for i in range(32)]
        memory = execute_copies(program, inputs)
        for i, (_, data) in enumerate(inputs):
            self.assertEqual(bytes(memory[0x90020000+i*2048+j] for j in range(2048)), data)

    def test_cfg_capture_spills_preserve_bytes_across_edges(self):
        for fp8 in (False, True):
            with self.subTest(fp8=fp8):
                source = cfg_pressure(fp8=fp8)
                self.assertNotEqual(run(source, "--lower-atlas-virtual-to-machine").returncode, 0)
                first = run(source, "--spill-atlas-virtual")
                self.assertEqual(first.returncode, 0, first.stderr)
                self.assertEqual(run(first.stdout, "--spill-atlas-virtual").stdout, first.stdout)
                _, program = compile_program(first.stdout)
                size, count = (1024, 33) if fp8 else (2048, 32)
                inputs = [(0x90000000+i*2048, bytes((j*19+i*13) % 256 for j in range(size))) for i in range(count)]
                memory = execute_copies(program, inputs)
                for i, (address, data) in enumerate(inputs):
                    self.assertEqual(bytes(memory[0x90020000+i*size+j] for j in range(size)), data)
                    self.assertEqual(bytes(memory[address+j] for j in range(size)), data)

    def test_spills_reload_on_both_diamond_paths_and_loop_iterations(self):
        for kind, taken in (("diamond", False), ("diamond", True), ("loop", True)):
            with self.subTest(kind=kind, taken=taken):
                source = cfg_pressure(kind, taken=taken)
                self.assertNotEqual(run(source, "--lower-atlas-virtual-to-machine").returncode, 0)
                spilled = run(source, "--spill-atlas-virtual", "--schedule-atlas-virtual", "--verify-atlas-virtual-stream")
                self.assertEqual(spilled.returncode, 0, spilled.stderr)
                _, program = compile_program(spilled.stdout)
                inputs = [(0x90000000+i*2048, bytes((j*31+i*7) % 256 for j in range(2048))) for i in range(32)]
                visits = {}
                memory = execute_copies(program, inputs, visits)
                if kind == "loop":
                    self.assertTrue(any(visits.get(pc) == 3 and row['operation'] == 'atlas.vload'
                                        for pc, row in enumerate(program['instructions'])))
                for i, (address, data) in enumerate(inputs):
                    self.assertEqual(bytes(memory[0x90020000+i*2048+j] for j in range(2048)), data)
                    self.assertEqual(bytes(memory[address+j] for j in range(2048)), data)

    def test_spills_do_not_bypass_the_final_program_size_limit(self):
        from test_virtual_scheduling import wide_return_block

        result = run(wide_return_block(32), "--spill-atlas-virtual")
        self.assertEqual(result.returncode, 0, result.stderr)
        lowered = run(result.stdout, "--lower-atlas-virtual-to-machine")
        self.assertNotEqual(lowered.returncode, 0)
        self.assertIn("2048-word policy limit", lowered.stderr)
        self.assertEqual(lowered.stdout, "")

    def test_cfg_spill_writer_layout_and_dominance(self):
        source = raw_roundtrip()
        start = source.index('%s2, %saved =')
        end = source.index('%s3, %again =')
        store = source[start:end].replace('(%s1,', '(%store_state,')
        source = (source[:start] + f'cf.br ^save(%s1 : {STATE})\n'
                  + f'^use(%load_state: {STATE}):\n'
                  + source[end:].replace('(%s2,', '(%load_state,', 1))
        source = source.replace('\n}}', '\n' + f'^save(%store_state: {STATE}):\n'
                                + store + f'cf.br ^use(%s2 : {STATE})\n' + '}}')
        spilled = run(source, "--spill-atlas-virtual")
        self.assertEqual(spilled.returncode, 0, spilled.stderr)
        self.assertLess(spilled.stdout.index('"atlas.virtual_spill_bf16"'),
                        spilled.stdout.index('"atlas.virtual_reload_bf16"'))
        self.assertEqual(run(spilled.stdout, "--spill-atlas-virtual").stdout, spilled.stdout)
        _, program = compile_program(source)
        data = bytes(range(256))*8
        memory = execute_copies(program, [(0x90000000, data)])
        self.assertEqual(bytes(memory[0x90020000+j] for j in range(2048)), data)
        invalid = source.replace(f'cf.br ^save(%s1 : {STATE})',
            '%choose = arith.constant 1 : i1\n' +
            f'cf.cond_br %choose, ^save(%s1 : {STATE}), ^use(%s1 : {STATE})')
        rejected = run(invalid, "--verify-atlas-virtual-stream")
        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn('dominat', rejected.stderr)

    def test_cfg_slots_cannot_reuse_textually_completed_storage(self):
        source = raw_roundtrip().replace('%s4 =', '\n'.join([
            f'cf.br ^use(%s3 : {STATE})', f'^use(%incoming: {STATE}):',
            f'%s5, %saved2 = "atlas.virtual_spill_bf16"(%incoming, %again) {{slot_word = 135680 : i32}} : ({STATE}, {BF16}) -> ({STATE}, {SAVED})',
            f'%s6, %again2 = "atlas.virtual_reload_bf16"(%s5, %saved2) : ({STATE}, {SAVED}) -> ({STATE}, {BF16})',
            '%s4 ='])).replace('(%s3, %again)', '(%s6, %again2)')
        compile_program(source)
        rejected = run(source.replace('135680', '135168'), "--verify-atlas-virtual-stream")
        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn('live spill slots overlap', rejected.stderr)

    def test_entry_control_count_uses_the_memory_abi_when_registers_do_not_fit(self):
        source = pressure(1).replace('@test()', '@test(' + ', '.join(f'%arg{i}: i32' for i in range(18)) + ')')
        result = run(source, "--spill-atlas-virtual")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("atlas.memory_controls", result.stdout)

    def test_spill_slots_cannot_alias_staging_or_escape_the_reserved_frame(self):
        for word in (131072, 135169, 196608):
            result = run(raw_roundtrip().replace("135168", str(word)), "--verify-atlas-virtual-stream")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("reserved VMEM frame", result.stderr)

    def test_overlapping_live_slots_and_wrong_format_are_rejected(self):
        spilled = run(pressure(), "--spill-atlas-virtual")
        self.assertEqual(spilled.returncode, 0, spilled.stderr)
        changed = spilled.stdout.replace("slot_word = 135680", "slot_word = 135168")
        self.assertNotEqual(changed, spilled.stdout)
        result = run(changed, "--verify-atlas-virtual-stream")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("live spill slots overlap", result.stderr)
        wrong = raw_roundtrip().replace("atlas.virtual_reload_bf16", "atlas.virtual_reload_fp8")
        self.assertNotEqual(run(wrong, "--verify-atlas-virtual-stream").returncode, 0)

    def test_modified_physical_spill_address_is_caught(self):
        machine, _ = compile_program(raw_roundtrip())
        changed = machine.replace("immediate = 33 : i32", "immediate = 34 : i32", 1)
        self.assertNotEqual(changed, machine)
        result = run(changed, "--verify-atlas-generated-schedule")
        self.assertNotEqual(result.returncode, 0)

    def test_completed_spill_slots_can_be_reused(self):
        source = raw_roundtrip()
        source = source.replace(f'return %s4 : {STATE}', '\n'.join([
            f'%s5, %saved2 = "atlas.virtual_spill_bf16"(%s4, %again) {{slot_word = 135168 : i32}} : ({STATE}, {BF16}) -> ({STATE}, {SAVED})',
            f'%s6, %again2 = "atlas.virtual_reload_bf16"(%s5, %saved2) : ({STATE}, {SAVED}) -> ({STATE}, {BF16})',
            f'return %s6 : {STATE}']))
        self.assertEqual(run(source, "--verify-atlas-virtual-stream").returncode, 0)
        compile_program(source)


if __name__ == "__main__":
    unittest.main()
