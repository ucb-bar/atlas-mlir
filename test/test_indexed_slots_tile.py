"""TILE word-view regressions use unified indexed slots and explicit wrapping."""
from __future__ import annotations

import re
import unittest

from test_virtual_spilling import BF16, SAVED, STATE, compile_program, execute_copies, run, wrap


def indexed_source(*, initialized=True, store=True, index=513):
    body = [
        f'%s0 = "atlas.virtual_start"() : () -> {STATE}',
        f'%slot = "atlas.virtual_slot"() {{kind="bf16",words=512:i32,slot_word=135168:i32,purpose="scratch",pinned=true}} : () -> {SAVED}',
        f'%s1, %tile = "atlas.virtual_input_bf16"(%s0) {{index=0:i32}} : ({STATE}) -> ({STATE},{BF16})',
        f'%index = arith.constant {index} : i32',
        '%mask511 = arith.constant 511 : i32',
        '%wrapped = arith.andi %index,%mask511 : i32',
        '%one = arith.constant 1 : i32',
        '%shift = arith.constant 16 : i32',
    ]
    state = '%s1'
    if initialized:
        body.append(f'%saved = "atlas.virtual_save"({state},%slot,%tile) : ({STATE},{SAVED},{BF16}) -> {STATE}')
        state = '%saved'
    body += [
        f'%loaded, %word = "atlas.virtual_slot_load"({state},%slot,%wrapped) {{kind="lw",byte_offset=0:i32}} : ({STATE},{SAVED},i32) -> ({STATE},i32)',
        '%high = arith.shrui %word,%shift : i32',
        '%low = arith.shli %high,%shift : i32',
        '%difference = arith.subi %word,%low : i32',
        '%mask = arith.andi %difference,%one : i32',
        '%xor = arith.xori %mask,%one : i32',
        '%combined = arith.ori %xor,%mask : i32',
        '%flag = arith.cmpi eq,%combined,%one : i32',
        '%other = arith.xori %flag,%flag : i1',
        '%flags = arith.ori %flag,%other : i1',
        '%both = arith.andi %flag,%flags : i1',
        '%extended = arith.extui %both : i1 to i32',
        '%replacement = arith.ori %word,%extended : i32',
    ]
    state = '%loaded'
    if store:
        body.append(f'%stored = "atlas.virtual_slot_store"({state},%slot,%replacement,%wrapped) {{kind="sw",byte_offset=0:i32}} : ({STATE},{SAVED},i32,i32) -> {STATE}')
        state = '%stored'
    body += [
        f'%restored, %result = "atlas.virtual_restore"({state},%slot) : ({STATE},{SAVED}) -> ({STATE},{BF16})',
        f'%done = "atlas.virtual_output_bf16"(%restored,%result) {{index=0:i32}} : ({STATE},{BF16}) -> {STATE}',
        f'return %done : {STATE}',
    ]
    return wrap(body)


class IndexedSlotsTest(unittest.TestCase):
    def test_scalar_pressure_preserves_guarded_dma_address_inductions(self):
        count = 20
        source = wrap([
            f'%s = "atlas.virtual_start"() : () -> {STATE}',
            '%zero = arith.constant 0 : i32', '%one = arith.constant 1 : i32',
            '%base = arith.constant -1879048192 : i32',
            '%output = arith.constant -1878917120 : i32',
            '%bytes = arith.constant 2048 : i32',
            *(f'%initial{i} = arith.constant {i} : i32' for i in range(count)),
            f'cf.br ^loop(%s,%zero,%base,' + ','.join(f'%initial{i}' for i in range(count))
                + f' : {STATE},i32,i32,' + ','.join('i32' for _ in range(count)) + ')',
            f'^loop(%state: {STATE},%counter: i32,%pointer: i32,'
                + ','.join(f'%value{i}: i32' for i in range(count)) + '):',
            '%limit = arith.constant 2 : i32',
            '%more = arith.cmpi ult,%counter,%limit : i32',
            f'cf.cond_br %more,^body(%state : {STATE}),^done(%state : {STATE})',
            f'^body(%body_state: {STATE}):',
            f'%loaded,%tile = "atlas.virtual_dma_load_bf16"(%body_state,%pointer,%bytes) : ({STATE},i32,i32) -> ({STATE},{BF16})',
            f'%stored = "atlas.virtual_dma_store_bf16"(%loaded,%tile,%output,%bytes) : ({STATE},{BF16},i32,i32) -> {STATE}',
            '%next_counter = arith.addi %counter,%one : i32',
            '%next_pointer = arith.addi %pointer,%bytes : i32',
            *(f'%updated{i} = arith.addi %value{i},%one : i32' for i in range(count)),
            f'cf.br ^loop(%stored,%next_counter,%next_pointer,' + ','.join(f'%updated{i}' for i in range(count))
                + f' : {STATE},i32,i32,' + ','.join('i32' for _ in range(count)) + ')',
            f'^done(%end: {STATE}):', f'return %end : {STATE}',
        ])
        source = source.replace('attributes {', 'attributes {atlas.output_dram_bytes = 2048 : i64, ')
        spilled = run(source, '--spill-atlas-virtual')
        self.assertEqual(spilled.returncode, 0, spilled.stderr)
        self.assertIn('"atlas.virtual_restore"', spilled.stdout)
        _, program = compile_program(spilled.stdout)
        first, second = bytes([13]) * 2048, bytes([29]) * 2048
        memory = execute_copies(program, [(0x90000000, first), (0x90000800, second)])
        self.assertEqual(bytes(memory[0x90020000 + i] for i in range(2048)), second)

    def test_counted_loop_retains_index_origin_and_slot_initialization(self):
        source = indexed_source(index=0).replace('%index = arith.constant 0', '%zero = arith.constant 0')
        source = source.replace('%mask511 = arith.constant 511 : i32\n%wrapped = arith.andi %index,%mask511 : i32', '')
        source = source.replace('%loaded, %word =',
            f'cf.br ^words(%saved,%zero : {STATE},i32)\n'
            f'^words(%word_state: {STATE},%index: i32):\n'
            '%mask511 = arith.constant 511 : i32\n%wrapped = arith.andi %index,%mask511 : i32\n%loaded, %word =')
        source = source.replace('"atlas.virtual_slot_load"(%saved,',
                                '"atlas.virtual_slot_load"(%word_state,')
        source = source.replace('%restored, %result =',
            '%limit = arith.constant 513 : i32\n'
            '%next = arith.addi %index,%one : i32\n'
            '%more = arith.cmpi ult,%next,%limit : i32\n'
            f'cf.cond_br %more,^words(%stored,%next : {STATE},i32),^finish(%stored : {STATE})\n'
            f'^finish(%last_state: {STATE}):\n%restored, %result =')
        source = source.replace('"atlas.virtual_restore"(%stored,', '"atlas.virtual_restore"(%last_state,')
        _, program = compile_program(source)
        payload = bytes((i * 17 + 2) % 256 for i in range(2048))
        memory = execute_copies(program, [(0x90000000, payload)])
        expected = bytearray(payload)
        for index in range(512):
            expected[index * 4] |= 1
        self.assertEqual(bytes(memory[0x90020000 + i] for i in range(2048)), expected)

    def test_word_view_supports_wrapping_indices_and_scalar_bit_operations(self):
        for index in (-1, 0, 511, 512, 513, 2147483647):
            with self.subTest(index=index):
                machine, program = compile_program(indexed_source(index=index))
                self.assertIn('slot_index_load', machine)
                self.assertIn('slot_index_store', machine)
                masks = [row for row in program['instructions'] if row['operation'] == 'atlas.alu_imm'
                         and row['fields']['kind'] == 'andi' and row['fields']['immediate'] == 511]
                self.assertGreaterEqual(len(masks), 1)
                payload = bytes((i * 17 + 2) % 256 for i in range(2048))
                memory = execute_copies(program, [(0x90000000, payload)])
                expected = bytearray(payload)
                expected[(index % 512) * 4] |= 1
                self.assertEqual(bytes(memory[0x90020000 + i] for i in range(2048)), expected)

    def test_indexed_access_cannot_initialize_an_unwritten_slot(self):
        for store in (False, True):
            rejected = run(indexed_source(initialized=False, store=store), '--verify-atlas-virtual-stream')
            self.assertNotEqual(rejected.returncode, 0)
            self.assertRegex(rejected.stderr, r'uninitialized|not initialized')
        source = indexed_source(initialized=False)
        source = re.sub(r'%loaded, %word = "atlas.virtual_slot_load"[^\n]*',
                        f'%word = arith.constant 0 : i32', source).replace('(%loaded,%slot,%replacement,%wrapped)', '(%s1,%slot,%replacement,%wrapped)')
        rejected = run(source, '--verify-atlas-virtual-stream')
        self.assertNotEqual(rejected.returncode, 0)
        self.assertRegex(rejected.stderr, r'uninitialized|not initialized')

    def test_wrapping_mask_does_not_authorize_an_undersized_slot(self):
        changed = indexed_source().replace('kind="bf16",words=512', 'kind="bf16",words=256')
        rejected = run(changed, '--verify-atlas-virtual-stream')
        self.assertNotEqual(rejected.returncode, 0)

    def test_poisonous_shift_counts_and_nonwrapping_flags_are_rejected(self):
        for changed in (indexed_source().replace('%shift = arith.constant 16', '%shift = arith.constant 32'),
                        indexed_source().replace('arith.subi %word,%low', 'arith.subi %word,%low overflow<nsw>')):
            rejected = run(changed, '--verify-atlas-virtual-stream')
            self.assertNotEqual(rejected.returncode, 0)

    def test_tampered_mask_address_shift_and_store_value_are_rejected(self):
        machine, _ = compile_program(indexed_source())
        changes = [machine.replace('immediate = 511 : i32', 'immediate = 1023 : i32', 1),
                   machine.replace('immediate = 2 : i32', 'immediate = 3 : i32', 1)]
        # Changing an indexed store's data register keeps instruction geometry
        # valid but must violate its authenticated source scalar origin.
        lines = machine.splitlines()
        for n, line in enumerate(lines):
            if '"atlas.scalar_store"' in line and 'atlas.virtual_tile_command' in line:
                altered = re.sub(r'src = \d+ : i32', 'src = 1 : i32', line)
                if altered != line:
                    changes.append('\n'.join([*lines[:n], altered, *lines[n+1:]]))
                    break
        self.assertEqual(len(changes), 3)
        for changed in changes:
            self.assertNotEqual(changed, machine)
            rejected = run(changed, '--verify-atlas-generated-schedule')
            self.assertNotEqual(rejected.returncode, 0)


if __name__ == '__main__':
    unittest.main()
