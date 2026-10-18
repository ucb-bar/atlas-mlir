"""Pressure-driven VMEM spills, raw copies and checker mutations."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import unittest

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
        body += [f'%addr = arith.constant -1878917120 : i32', '%size = arith.constant 1024 : i32',
                 f'%s4 = "atlas.virtual_dma_store_fp8"(%s3, %again, %addr, %size) : ({STATE}, {FP8}, i32, i32) -> {STATE}']
    else:
        body += [f'%s4 = "atlas.virtual_output_bf16"(%s3, %again) {{index = 0 : i32}} : ({STATE}, {BF16}) -> {STATE}']
    return wrap([*body, f'return %s4 : {STATE}'])


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


def execute_copies(program, inputs):
    """Independent sequential byte transport check, not a timing simulator.

    Treat DMA as complete at issue; generated timing checks separately cover
    hazards. No target compiler packing or allocation helper supplies bytes.
    """
    x = [0] * 32
    registers = [bytes(1024) for _ in range(64)]
    vmem, dram = {}, {}
    for address, data in inputs:
        dram.update((address+i, byte) for i, byte in enumerate(data))
    for row in program["instructions"]:
        op, f = row["operation"], row["fields"]
        if op == "atlas.upper":
            x[f["dst"]] = f["immediate"] << 12
        elif op == "atlas.alu_imm":
            assert f["kind"] == "addi"
            x[f["dst"]] = (x[f["src"]] + f["immediate"]) & 0xffffffff
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
        else:
            assert op in {"atlas.delay", "atlas.dma_wait", "atlas.dma_config", "atlas.csr", "atlas.trap"}, op
        x[0] = 0
    return dram


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
        machine, program = compile_program(source)
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

    def test_cfg_pressure_is_an_explicit_failure(self):
        from test_virtual_scheduling import wide_return_block

        result = run(wide_return_block(32), "--spill-atlas-virtual")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("across CFG edges is not supported", result.stderr)
        self.assertEqual(result.stdout, "")

    def test_scalar_pressure_is_not_relabelled_as_tensor_spilling(self):
        source = pressure(1).replace('@test()', '@test(' + ', '.join(f'%arg{i}: i32' for i in range(18)) + ')')
        result = run(source, "--spill-atlas-virtual")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("scalar register spilling is not supported", result.stderr)
        self.assertEqual(result.stdout, "")

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
