"""Loader descriptors are checked against actual ELF payload bytes, without transport."""
import hashlib
import struct
import tempfile
import unittest
from pathlib import Path

from atlas_mlir.deploy import Refusal
from atlas_mlir.deployment_loader import CHUNK_BYTES, describe


class LoaderTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.path = Path(temp.name) / 'chip.elf'
        from atlas_mlir_native.soc import SOC
        self.ddr = SOC.memory_region('ddr')
        self.base, self.window = self.ddr.base, self.ddr.base + 8192
        self.weight_bytes = CHUNK_BYTES + 7
        raw = bytearray(512 + self.weight_bytes)
        raw[:6] = b'\x7fELF\x02\x01'
        struct.pack_into('<H', raw, 18, 243)
        struct.pack_into('<QQ', raw, 24, self.base, 64)
        struct.pack_into('<HH', raw, 54, 56, 3)
        self.loads = [dict(address=self.base, file_bytes=4, memory_bytes=16, flags=7),
                      dict(address=self.window, file_bytes=16, memory_bytes=16, flags=6),
                      dict(address=self.window + 16, file_bytes=self.weight_bytes,
                           memory_bytes=self.weight_bytes, flags=4)]
        for i, (load, offset) in enumerate(zip(self.loads, [256, 272, 512])):
            struct.pack_into('<II6Q', raw, 64 + 56 * i, 1, load['flags'], offset, load['address'],
                             load['address'], load['file_bytes'], load['memory_bytes'], 1)
        raw[512:] = b'W' * self.weight_bytes
        self.path.write_bytes(raw)
        self.deployment = dict(host_allocation=dict(address=self.base, bytes=8192, usable_bytes=8192),
                               window=dict(address=self.window, bytes=self.ddr.capacity_bytes - 8192), allocations=[],
                               parameters=dict(address=self.window + 16, bytes=self.weight_bytes,
                                               sha256=hashlib.sha256(raw[512:]).hexdigest()))
        self.record = dict(entry=self.base, sha256=hashlib.sha256(raw).hexdigest(), loads=self.loads,
                           host_reservations=dict(__stack_end=self.base + 8192),
                           allocated_sections=[], instruction_images={})

    def test_chunks_cover_single_elf_weights_and_zero_fill_is_separate(self):
        plan, manifest = describe(self.path, self.deployment, self.record)
        from atlas_mlir_native import soc
        self.assertEqual(plan['ddr'], dict(address=self.ddr.base, bytes=self.ddr.capacity_bytes))
        for record in (plan, manifest):
            self.assertEqual(record['soc'], dict(identity=soc.IDENTITY, sha256=soc.DECLARATION_SHA256,
                                                 kit_member='bench/soc.json'))
            self.assertNotIn('board', record)
        self.assertEqual(plan['host_reservation']['bytes'], 8192)
        self.assertEqual(manifest['segments'][0]['zero_fill'], dict(address=self.base + 4, bytes=12))
        chunks = manifest['weights']['chunks']
        self.assertEqual([r['bytes'] for r in chunks], [CHUNK_BYTES, 7])
        self.assertEqual(chunks[1]['file_offset'], 512 + CHUNK_BYTES)
        self.assertEqual(chunks[1]['address'], self.window + 16 + CHUNK_BYTES)
        self.assertEqual(chunks[1]['sha256'], hashlib.sha256(b'W' * 7).hexdigest())
        self.assertFalse(manifest['transport_executed'])

    def test_corrupted_linked_weight_refuses(self):
        with self.path.open('r+b') as stream:
            stream.seek(512)
            stream.write(b'X')
        with self.assertRaisesRegex(Refusal, 'weights differ'):
            describe(self.path, self.deployment, self.record)

    def test_changed_elf_outside_weights_refuses(self):
        with self.path.open('r+b') as stream:
            stream.seek(256)
            stream.write(b'X')
        with self.assertRaisesRegex(Refusal, 'ELF differs'):
            describe(self.path, self.deployment, self.record)

    def test_out_of_board_load_refuses_even_if_record_agrees(self):
        with self.path.open('r+b') as stream:
            stream.seek(64 + 56 + 16)
            stream.write(struct.pack('<QQ', self.ddr.end, self.ddr.end))
        with self.assertRaisesRegex(Refusal, 'outside DDR'):
            describe(self.path, self.deployment, self.record)

    def test_writable_weights_and_wrong_reservation_refuse(self):
        with self.path.open('r+b') as stream:
            stream.seek(64 + 112 + 4)
            stream.write(struct.pack('<I', 6))
        self.loads[2]['flags'] = 6
        with self.assertRaisesRegex(Refusal, 'read-only ELF backing'):
            describe(self.path, self.deployment, self.record)
        self.deployment['host_allocation']['bytes'] = 4096
        with self.assertRaisesRegex(Refusal, 'measured host placement'):
            describe(self.path, self.deployment, self.record)
        self.deployment['host_allocation'].update(bytes=8192, usable_bytes=4096)
        with self.assertRaisesRegex(Refusal, 'exceeds the measured host exclusion'):
            describe(self.path, self.deployment, self.record)

    def test_truncated_table_and_load_identity_drift_refuse(self):
        self.record['loads'] = []
        with self.assertRaisesRegex(Refusal, 'loads differ'):
            describe(self.path, self.deployment, self.record)
        self.path.write_bytes(self.path.read_bytes()[:70])
        with self.assertRaisesRegex(Refusal, 'load table is truncated'):
            describe(self.path, self.deployment, self.record)

    def test_a_resident_window_ends_at_the_arena_and_the_arena_is_listed(self):
        arena_bytes, weights_end = 4096, 512 + self.weight_bytes
        arena = self.ddr.end - arena_bytes
        raw = bytearray(self.path.read_bytes()) + b'R' * arena_bytes
        struct.pack_into('<H', raw, 56, 4)
        loads = [*self.loads, dict(address=arena, file_bytes=arena_bytes, memory_bytes=arena_bytes, flags=4)]
        for i, (load, offset) in enumerate(zip(loads, [320, 336, 512, weights_end])):
            struct.pack_into('<II6Q', raw, 64 + 56 * i, 1, load['flags'], offset, load['address'],
                             load['address'], load['file_bytes'], load['memory_bytes'], 1)
        raw[320:340] = bytes(20)
        self.path.write_bytes(raw)
        self.record.update(sha256=hashlib.sha256(raw).hexdigest(), loads=loads)
        self.deployment.update(window=dict(address=self.window, bytes=arena - self.window),
                               resident=dict(address=arena, bytes=arena_bytes, loads=[],
                                             sha256=hashlib.sha256(b'R' * arena_bytes).hexdigest()))
        plan, manifest = describe(self.path, self.deployment, self.record)
        self.assertEqual(plan['resident'], self.deployment['resident'])
        self.assertEqual((manifest['resident']['file_offset'], manifest['resident']['address']), (weights_end, arena))
        self.assertEqual([c['bytes'] for c in manifest['resident']['chunks']], [arena_bytes])
        self.deployment['window']['bytes'] = self.ddr.end - self.window
        with self.assertRaisesRegex(Refusal, 'measured host placement'):
            describe(self.path, self.deployment, self.record)
        self.deployment['window']['bytes'] = arena - self.window
        self.deployment['resident']['sha256'] = '0' * 64
        with self.assertRaisesRegex(Refusal, 'resident arena differs'):
            describe(self.path, self.deployment, self.record)
