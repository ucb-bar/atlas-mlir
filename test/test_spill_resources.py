"""CFG parallel copies, scalar spills, and shared VMEM placement."""

import struct
import unittest

from test_virtual_spilling import (
    BF16,
    FP8,
    SAVED,
    STATE,
    compile_program,
    execute_copies,
    pressure,
    raw_roundtrip,
    run,
    wrap,
)


def block_pressure(loop=False):
    count = 16
    source = pressure(count)
    start = f'%s0 = "atlas.virtual_start"() : () -> {STATE}'
    source = source.replace(start, start + '\n' +
        f'%extra_state, %extra = "atlas.virtual_input_fp8"(%s0) {{index = 63 : i32}} : ({STATE}) -> ({STATE}, {FP8})')
    source = source.replace('"atlas.virtual_input_bf16"(%s0)', '"atlas.virtual_input_bf16"(%extra_state)')
    cut = source.index('%s17 =')
    prefix, outputs = source[:cut], source[cut:]
    outputs = outputs.replace('(%s16,', '(%incoming,')
    for i in range(count):
        outputs = outputs.replace(f'%v{i})', f'%a{i})')
    arguments = ', '.join(f'%a{i}: {BF16}' for i in range(count))
    values = ', '.join(f'%v{i}' for i in range(count))
    types = ', '.join([STATE] + [BF16]*count)
    if not loop:
        flow = f'cf.br ^use(%s16, {values} : {types})\n^use(%incoming: {STATE}, {arguments}):\n'
    else:
        loop_arguments = ', '.join(f'%b{i}: {BF16}' for i in range(count))
        rotated = ', '.join(f'%b{(i+1)%count}' for i in range(count))
        current = ', '.join(f'%b{i}' for i in range(count))
        flow = '\n'.join([
            '%zero = arith.constant 0 : i32', '%one = arith.constant 1 : i32',
            '%three = arith.constant 3 : i32',
            f'cf.br ^loop(%s16, {values}, %zero : {types}, i32)',
            f'^loop(%loop_state: {STATE}, {loop_arguments}, %i: i32):',
            '%more = arith.cmpi slt, %i, %three : i32',
            '%next = arith.addi %i, %one : i32',
            f'cf.cond_br %more, ^loop(%loop_state, {rotated}, %next : {types}, i32), ^use(%loop_state, {current} : {types})',
            f'^use(%incoming: {STATE}, {arguments}):\n'])
    return prefix + flow + outputs


def scalar_pressure(loop=False):
    body = [f'%s0 = "atlas.virtual_start"() : () -> {STATE}',
            f'%s1, %good = "atlas.virtual_input_bf16"(%s0) {{index = 0 : i32}} : ({STATE}) -> ({STATE}, {BF16})',
            f'%s2, %bad = "atlas.virtual_input_bf16"(%s1) {{index = 1 : i32}} : ({STATE}) -> ({STATE}, {BF16})']
    for i in range(18):
        body.extend([f'%c{i} = arith.constant {i} : i32', f'%v{i} = arith.addi %seed, %c{i} : i32'])
    if loop:
        body.extend([
            'cf.br ^loop(%s2, ' + ', '.join(f'%v{i}' for i in range(18)) + f', %c0 : {STATE}, ' + ', '.join(['i32']*19) + ')',
            f'^loop(%loop_state: {STATE}, ' + ', '.join(f'%a{i}: i32' for i in range(18)) + ', %i: i32):',
            '%more = arith.cmpi slt, %i, %c2 : i32',
            '%next = arith.addi %i, %c1 : i32',
            'cf.cond_br %more, ^loop(%loop_state, ' + ', '.join(f'%a{(i+1)%18}' for i in range(18)) + f', %next : {STATE}, ' + ', '.join(['i32']*19) + f'), ^check(%loop_state, %a0 : {STATE}, i32)',
            f'^check(%check_state: {STATE}, %selected: i32):',
            '%expected = arith.addi %seed, %c2 : i32',
            '%ok = arith.cmpi eq, %selected, %expected : i32'])
        state = '%check_state'
    else:
        value = '%v0'
        for i in range(1, 18):
            body.append(f'%sum{i} = arith.addi {value}, %v{i} : i32')
            value = f'%sum{i}'
        # For the test seed 1: sum(1+i for i in range(18)) = 171.
        body.extend(['%expected = arith.constant 171 : i32', f'%ok = arith.cmpi eq, {value}, %expected : i32'])
        state = '%s2'
    body.extend([
        f'cf.cond_br %ok, ^yes({state} : {STATE}), ^no({state} : {STATE})',
        f'^yes(%yes_state: {STATE}):', f'cf.br ^exit(%yes_state, %good : {STATE}, {BF16})',
        f'^no(%no_state: {STATE}):', f'cf.br ^exit(%no_state, %bad : {STATE}, {BF16})',
        f'^exit(%done: {STATE}, %result: {BF16}):',
        f'%out = "atlas.virtual_output_bf16"(%done, %result) {{index = 0 : i32}} : ({STATE}, {BF16}) -> {STATE}',
        f'return %out : {STATE}'])
    return wrap(body).replace('@test()', '@test(%seed: i32)').replace(
        'atlas.input_dram_base =', 'atlas.control_dram_base = 2415984640 : i64, atlas.input_dram_base =')


def scratch(source, *, word=-1, words=512, pinned=False):
    start = f'%s0 = "atlas.virtual_start"() : () -> {STATE}'
    declaration = (f'%scratch = "atlas.virtual_slot"() {{kind = "raw", words = {words} : i32, '
                   f'slot_word = {word} : i32, purpose = "scratch", pinned = {str(pinned).lower()}}} : () -> {SAVED}')
    return source.replace(start, start + '\n' + declaration)


def entry_pressure(boolean=False):
    source = scalar_pressure().replace('@test(%seed: i32)', '@test(' + ', '.join(f'%arg{i}: i32' for i in range(18)) + ')')
    for i in range(18):
        source = source.replace(f'%v{i} = arith.addi %seed, %c{i}', f'%v{i} = arith.addi %arg{i}, %c0')
    # Horner accumulation makes argument order observable, unlike a plain sum.
    for i in range(1, 18):
        previous = '%v0' if i == 1 else f'%sum{i-1}'
        source = source.replace(f'%sum{i} = arith.addi {previous}, %v{i}',
            f'%twice{i} = arith.addi {previous}, {previous} : i32\n%sum{i} = arith.addi %twice{i}, %v{i}')
    expected = 0
    for argument in range(1, 19):
        expected = (2*expected + argument) & 0xffffffff
    source = source.replace('arith.constant 171', f'arith.constant {expected}')
    if boolean:
        source = source.replace('%arg17: i32)', '%arg17: i32, %flag: i1)')
        source = source.replace('cf.cond_br %ok, ^yes(', 'cf.cond_br %ok, ^flag(')
        source = source.replace(f'^yes(%yes_state: {STATE}):', '\n'.join([
            f'^flag(%flag_state: {STATE}):',
            f'cf.cond_br %flag, ^yes(%flag_state : {STATE}), ^no(%flag_state : {STATE})',
            f'^yes(%yes_state: {STATE}):']))
    return source


def fp8_block_pressure():
    count = 33
    source = pressure(count, fp8=True)
    cut = source.index('%a0 =')
    prefix, outputs = source[:cut], source[cut:]
    outputs = outputs.replace('(%s33,', '(%incoming,')
    for i in range(count):
        outputs = outputs.replace(f'%v{i},', f'%tile{i},')
    values = ', '.join(f'%v{i}' for i in range(count))
    current = ', '.join(f'%carry{i}' for i in range(count))
    rotated = ', '.join(f'%carry{(i+1)%count}' for i in range(count))
    types = ', '.join([STATE] + [FP8]*count)
    flow = '\n'.join([
        '%zero = arith.constant 0 : i32', '%one = arith.constant 1 : i32',
        '%three = arith.constant 3 : i32',
        f'cf.br ^loop(%s33, {values}, %zero : {types}, i32)',
        f'^loop(%loop_state: {STATE}, ' + ', '.join(f'%carry{i}: {FP8}' for i in range(count)) + ', %iteration: i32):',
        '%more = arith.cmpi slt, %iteration, %three : i32',
        '%next = arith.addi %iteration, %one : i32',
        f'cf.cond_br %more, ^loop(%loop_state, {rotated}, %next : {types}, i32), ^use(%loop_state, {current} : {types})',
        f'^use(%incoming: {STATE}, ' + ', '.join(f'%tile{i}: {FP8}' for i in range(count)) + '):\n'])
    return prefix + flow + outputs


class SpillResourcesTest(unittest.TestCase):
    def test_fp8_loop_arguments_preserve_raw_bytes(self):
        source = fp8_block_pressure()
        _, program = compile_program(source)
        runtime = {}
        for invocation in range(2):
            tiles = [bytes((j*17+i*29+invocation*71)%256 for j in range(1024)) for i in range(33)]
            inputs = [(0x90000000+i*2048, tile) for i, tile in enumerate(tiles)]
            memory = execute_copies(program, inputs, runtime=runtime)
            for i in range(33):
                self.assertEqual(bytes(memory[0x90020000+i*1024+j] for j in range(1024)), tiles[(i+3)%33])

    def test_entry_controls_spill_without_changing_the_input_order(self):
        for boolean in (False, True):
            rewritten = run(entry_pressure(boolean), '--spill-atlas-virtual')
            self.assertEqual(rewritten.returncode, 0, rewritten.stderr)
            self.assertIn('"atlas.virtual_control"', rewritten.stdout)
            machine, program = compile_program(rewritten.stdout)
            self.assertIn('atlas.control_argument_types = array<i32: 32, 32', machine)
            runtime = {}
            for invocation in range(2):
                arguments = list(range(1, 19))
                if boolean:
                    arguments.append(3 if invocation == 0 else 2)
                elif invocation:
                    arguments.reverse()
                good, bad = bytes([19])*2048, bytes([33])*2048
                controls = struct.pack('<'+'I'*len(arguments), *arguments) + bytes(1024-len(arguments)*4)
                memory = execute_copies(program, [(0x90000000, good), (0x90000800, bad),
                    (0x90010000, controls)], runtime=runtime)
                self.assertEqual(bytes(memory[0x90020000+j] for j in range(2048)), good if invocation == 0 else bad)

    def test_block_arguments_and_rotating_loop_preserve_repeated_invocations(self):
        for loop in (False, True):
            with self.subTest(loop=loop):
                source = block_pressure(loop)
                self.assertNotEqual(run(source, '--lower-atlas-virtual-to-machine').returncode, 0)
                rewritten = run(source, '--spill-atlas-virtual')
                self.assertEqual(rewritten.returncode, 0, rewritten.stderr)
                self.assertEqual(run(rewritten.stdout, '--spill-atlas-virtual').stdout, rewritten.stdout)
                self.assertIn('"atlas.virtual_slot"', rewritten.stdout)
                if loop:
                    self.assertIn('purpose = "edge"', rewritten.stdout)
                    self.assertIn('"atlas.virtual_slot_copy"', rewritten.stdout)
                scheduled = run(rewritten.stdout, '--schedule-atlas-virtual')
                self.assertEqual(scheduled.returncode, 0, scheduled.stderr)
                _, program = compile_program(scheduled.stdout)
                runtime = {}
                for invocation in range(2):
                    tiles = [bytes((j*17+i*29+invocation*71)%256 for j in range(2048)) for i in range(16)]
                    inputs = [(0x90000000+i*2048, tile) for i, tile in enumerate(tiles)]
                    inputs.append((0x9001f800, bytes([73])*1024))
                    inputs.append((0x90028000, bytes([91])*64))
                    memory = execute_copies(program, inputs, runtime=runtime)
                    for i in range(16):
                        self.assertEqual(bytes(memory[0x90020000+i*2048+j] for j in range(2048)), tiles[(i+3)%16] if loop else tiles[i])
                        self.assertEqual(bytes(memory[0x90000000+i*2048+j] for j in range(2048)), tiles[i])
                    self.assertEqual(bytes(memory[0x90028000+j] for j in range(64)), bytes([91])*64)

    def test_scalar_results_and_loop_arguments_use_exact_stores_and_reloads(self):
        for loop in (False, True):
            with self.subTest(loop=loop):
                source = scalar_pressure(loop)
                self.assertNotEqual(run(source, '--lower-atlas-virtual-to-machine').returncode, 0)
                rewritten = run(source, '--spill-atlas-virtual')
                self.assertEqual(rewritten.returncode, 0, rewritten.stderr)
                self.assertIn('kind = "i32"', rewritten.stdout)
                machine, program = compile_program(rewritten.stdout)
                self.assertIn('"atlas.scalar_store"', machine)
                self.assertIn('"atlas.scalar_load"', machine)
                runtime = {}
                seeds = (1, 2, 0x7fffffff, 0xffffffff) if loop else (1, 1)
                for invocation, seed in enumerate(seeds):
                    good, bad = bytes([19+invocation])*2048, bytes([33])*2048
                    memory = execute_copies(program, [(0x90000000, good), (0x90000800, bad),
                        (0x90010000, struct.pack('<I', seed)+bytes(1020))], runtime=runtime)
                    self.assertEqual(bytes(memory[0x90020000+j] for j in range(2048)), good)

    def test_spills_share_placement_with_pinned_and_relocatable_scratch(self):
        for pinned in (False, True):
            source = scratch(pressure(), word=135168 if pinned else -1, pinned=pinned)
            result = run(source, '--spill-atlas-virtual')
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('slot_word = 135680', result.stdout)
            _, program = compile_program(result.stdout)
            data = [bytes((j+i)%256 for j in range(2048)) for i in range(32)]
            memory = execute_copies(program, [(0x90000000+i*2048, tile) for i, tile in enumerate(data)])
            for i, tile in enumerate(data):
                self.assertEqual(bytes(memory[0x90020000+i*2048+j] for j in range(2048)), tile)
        exhausted = run(scratch(pressure(), words=61440), '--spill-atlas-virtual')
        self.assertNotEqual(exhausted.returncode, 0)
        self.assertIn('shared temporary pool', exhausted.stderr)

    def test_slot_formats_initialization_and_physical_mutations_are_checked(self):
        source = scratch(raw_roundtrip(), words=512)
        source = source.replace('kind = "raw"', 'kind = "bf16"')
        source = source.replace('%s2, %saved =', f'%bad_state, %uninitialized = "atlas.virtual_restore"(%s1, %scratch) : ({STATE}, {SAVED}) -> ({STATE}, {BF16})\n%s2, %saved =')
        source = source.replace('"atlas.virtual_spill_bf16"(%s1,', '"atlas.virtual_spill_bf16"(%bad_state,')
        refused = run(source, '--verify-atlas-virtual-stream')
        self.assertNotEqual(refused.returncode, 0)
        self.assertIn('not initialized', refused.stderr)
        rewritten = run(scalar_pressure(), '--spill-atlas-virtual')
        self.assertEqual(rewritten.returncode, 0, rewritten.stderr)
        wrong = rewritten.stdout.replace('kind = "i32"', 'kind = "i1"', 1)
        self.assertNotEqual(run(wrong, '--verify-atlas-virtual-stream').returncode, 0)
        machine, _ = compile_program(rewritten.stdout)
        lines = machine.splitlines()
        for i, line in enumerate(lines):
            if '"atlas.scalar_store"' in line:
                lines[i] = line.replace('offset = 0', 'offset = 4', 1)
                break
        refused = run('\n'.join(lines), '--verify-atlas-generated-schedule')
        self.assertNotEqual(refused.returncode, 0)

    def test_dma_constants_and_shared_scalar_uses_preserve_static_proofs(self):
        addresses = [0x90020000 + i*2048 for i in range(18)]
        data = bytes(range(256))*8
        for scalar_uses in (False, True):
            with self.subTest(scalar_uses=scalar_uses):
                body = [f'%s0 = "atlas.virtual_start"() : () -> {STATE}',
                        f'%s1, %tile = "atlas.virtual_input_bf16"(%s0) {{index = 0 : i32}} : ({STATE}) -> ({STATE}, {BF16})']
                if scalar_uses:
                    body.append(f'%s2, %bad = "atlas.virtual_input_bf16"(%s1) {{index = 1 : i32}} : ({STATE}) -> ({STATE}, {BF16})')
                body += [f'%address{i} = arith.constant {address-2**32} : i32'
                         for i, address in enumerate(addresses)]
                body += ['%size = arith.constant 2048 : i32']
                state, tile = '%s1', '%tile'
                if scalar_uses:
                    # A checked sum makes non-DMA uses of every address observable.
                    value = '%address0'
                    for i in range(1, 18):
                        body.append(f'%sum{i} = arith.addi {value}, %address{i} : i32')
                        value = f'%sum{i}'
                    expected = (sum(addresses) + 2**31) % 2**32 - 2**31
                    body.extend([
                        f'%expected = arith.constant {expected} : i32',
                        f'%ok = arith.cmpi eq, {value}, %expected : i32',
                        f'cf.cond_br %ok, ^exit(%s2, %tile : {STATE}, {BF16}), ^exit(%s2, %bad : {STATE}, {BF16})',
                        f'^exit(%incoming: {STATE}, %selected: {BF16}):'])
                    state, tile = '%incoming', '%selected'
                for i in range(18):
                    body.append(f'%stored{i} = "atlas.virtual_dma_store_bf16"({state}, {tile}, %address{i}, %size) : ({STATE}, {BF16}, i32, i32) -> {STATE}')
                    state = f'%stored{i}'
                body.append(f'return {state} : {STATE}')
                source = wrap(body)
                direct = run(source, '--lower-atlas-virtual-to-machine')
                # DMA-only constants are launch-time immediates, not register pressure.
                self.assertEqual(direct.returncode == 0, not scalar_uses, direct.stderr)
                rewritten = run(source, '--spill-atlas-virtual')
                self.assertEqual(rewritten.returncode, 0, rewritten.stderr)
                self.assertEqual('atlas.spill_rematerialized' in rewritten.stdout, scalar_uses)
                _, program = compile_program(rewritten.stdout)
                memory = execute_copies(program, [(0x90000000, data), (0x90000800, bytes([73])*2048)])
                for address in addresses:
                    self.assertEqual(bytes(memory[address+j] for j in range(2048)), data)

    def test_saved_boolean_conditions_preserve_both_branch_outcomes(self):
        source = scalar_pressure()
        start = f'%s0 = "atlas.virtual_start"() : () -> {STATE}'
        declaration = f'%boolean = "atlas.virtual_slot"() {{kind = "i1", words = 1 : i32, slot_word = -1 : i32, purpose = "spill"}} : () -> {SAVED}'
        source = source.replace(start, start + '\n' + declaration)
        source = source.replace('cf.cond_br %ok,', '\n'.join([
            f'%saved_bool = "atlas.virtual_save"(%s2, %boolean, %ok) : ({STATE}, {SAVED}, i1) -> {STATE}',
            f'%bool_state, %again = "atlas.virtual_restore"(%saved_bool, %boolean) : ({STATE}, {SAVED}) -> ({STATE}, i1)',
            'cf.cond_br %again,'])).replace('^yes(%s2', '^yes(%bool_state').replace('^no(%s2', '^no(%bool_state')
        _, program = compile_program(source)
        runtime = {}
        for seed in (1, 2):
            good, bad = bytes([19])*2048, bytes([33])*2048
            memory = execute_copies(program, [(0x90000000, good), (0x90000800, bad),
                (0x90010000, struct.pack('<I', seed)+bytes(1020))], runtime=runtime)
            self.assertEqual(bytes(memory[0x90020000+j] for j in range(2048)), good if seed == 1 else bad)



if __name__ == '__main__':
    unittest.main()
