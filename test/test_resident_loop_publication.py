"""Public resident runtime replay trusts native/source truth, never JSON hashes."""
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from atlas_mlir_native.options import codegen_options
from atlas_mlir_native.quantized_loop_c import _resident_runtime_receipt
from atlas_mlir_native.resident_loop_publication import verify_resident_loop_runtime
import test_resident_loop_image as fixture
from test_resident_loop_c_application import renamed


def encoded(record):
    return json.dumps(record, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def rehash(record):
    for row in record['trip_bindings']:
        binding = row['binding']
        binding['execution_plan_sha256'] = sha256(encoded(row['execution_plan'])).hexdigest()
        binding.pop('binding_sha256', None)
        binding['binding_sha256'] = sha256(encoded(binding)).hexdigest()
    record['table_sha256'] = sha256(encoded(record['trip_bindings'])).hexdigest()
    return record


class ResidentLoopPublicationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        directory = os.environ.get('ATLAS_RESIDENT_IMAGE_RECEIPTS')
        if directory:
            with patch.dict(os.environ, ATLAS_RESIDENT_IMAGE_RECEIPTS=str(Path(directory)/'publication')):
                fixture.ResidentLoopImageTest.setUpClass.__func__(cls)
        else:
            fixture.ResidentLoopImageTest.setUpClass.__func__(cls)

    tearDownClass = classmethod(fixture.ResidentLoopImageTest.tearDownClass.__func__)

    def case(self, trips):
        sources, allocations = [self.source], deepcopy(self.selected.resident_bindings)
        for trip in range(1, trips):
            source, names = renamed(self.source, f'current_{trip}_')
            source = replace(source, source_identity=f'extracted current source {trip}')
            sources.append(source)
            allocations[source.digest()] = {
                names[key]:dict(row, address=row['address']+trip*8192)
                for key, row in self.selected.resident_bindings[self.source.digest()].items()}
        selected = replace(self.selected, resident_bindings=allocations)
        with codegen_options(selected):
            bound = [self.image.bind(source) for source in sources]
            state = dict(image=self.image, bindings=bound,
                         source_receipt=self.image.emit_c('published_resident_image').receipt)
            published = _resident_runtime_receipt(state)
        return tuple(sources), selected, published

    def verify(self, sources, selected, published, compiled=None):
        with codegen_options(selected):
            return verify_resident_loop_runtime(self.source, self.compiled if compiled is None else compiled,
                sources, published, name='published', target_source=fixture.ROOT)

    def test_valid_2_and_22_trip_publications_replay_into_a_frozen_checked_result(self):
        for trips in (2, 22):
            with self.subTest(trips=trips):
                sources, selected, published = self.case(trips)
                checked = self.verify(sources, selected, published)
                self.assertEqual(len(checked.bound), trips)
                self.assertEqual(checked.table_sha256, published['table_sha256'])
                self.assertEqual(checked.runtime_sha256, sha256(encoded(published)).hexdigest())
                self.assertIs(checked.image_receipt['forced_reload'], True)
                (self.root/f'published-runtime-{trips}.json').write_text(json.dumps(published, indent=2)+'\n')
                for current, source, row in zip(checked.bound, sources, published['trip_bindings'], strict=True):
                    self.assertEqual(current.receipt['source_digest'], source.digest())
                    self.assertEqual(current.execution_plan, row['execution_plan'])
                with self.assertRaises(FrozenInstanceError):
                    checked.table_sha256 = 'changed'
                with self.assertRaises(TypeError):
                    checked.image_receipt['forced_reload'] = False
                # Caller-owned publication mutation has no alias into replay results.
                published['trip_bindings'][0]['execution_plan']['status_address'] += 2048
                self.assertNotEqual(checked.bound[0].execution_plan['status_address'],
                                    published['trip_bindings'][0]['execution_plan']['status_address'])

    def test_rehashed_table_binary_bytes_storage_load_and_template_receipts_refuse(self):
        sources, selected, original = self.case(2)
        for kind in ('address', 'bytes', 'binary', 'template', 'storage', 'load', 'extra_field', 'boolean'):
            with self.subTest(tamper=kind):
                published = deepcopy(original)
                row = published['trip_bindings'][-1]
                if kind == 'address':
                    row['addresses'][0] += 8192
                elif kind == 'bytes':
                    row['execution_plan']['resident_bindings'][0]['bytes'] += 2048
                elif kind == 'binary':
                    row['binding']['binary_sha256'] = '0'*64
                elif kind == 'template':
                    published['image']['template_sha256'] = '0'*64
                elif kind == 'storage':
                    published['mutable_image_storage'] = 'aliased_external_words'
                elif kind == 'load':
                    published['load_convention'] = 'ensure_loaded'
                elif kind == 'extra_field':
                    published['unverified_claim'] = True
                else:
                    published['image']['forced_reload'] = 1
                rehash(published)
                with self.assertRaisesRegex(ValueError, 'differs from current source replay'):
                    self.verify(sources, selected, published)

    def test_genuinely_checked_alternate_allocation_and_changed_source_contracts_refuse(self):
        sources, selected, published = self.case(2)
        allocations = deepcopy(selected.resident_bindings)
        last = sources[-1]
        allocations[last.digest()][last.nodes[2].id]['address'] += 2*1024*1024
        with codegen_options(replace(selected, resident_bindings=allocations)):
            alternate = self.image.bind(last)
            row = dict(binding=alternate.receipt, execution_plan=alternate.execution_plan,
                       addresses=list(self.image.binding_addresses(alternate)))
        altered = deepcopy(published)
        altered['trip_bindings'][-1] = row
        rehash(altered)
        with self.assertRaisesRegex(ValueError, 'differs from current source replay'):
            self.verify(sources, selected, altered)
        for current in (sources[:-1], sources[::-1]):
            with self.subTest(source_order=[source.digest() for source in current]):
                with self.assertRaisesRegex(ValueError, 'differs from current source replay'):
                    self.verify(current, selected, published)
        calibrated = replace(last, nodes=(*last.nodes[:-1], replace(last.nodes[-1],
            attrs=(*last.nodes[-1].attrs, ('calibration_id', 'different numerical source policy')))))
        allocations = deepcopy(selected.resident_bindings)
        allocations[calibrated.digest()] = allocations.pop(last.digest())
        with self.assertRaisesRegex(ValueError, 'source structure differs'):
            self.verify((*sources[:-1], calibrated), replace(selected, resident_bindings=allocations), published)
        with self.assertRaisesRegex(ValueError, 'compilation pipeline differs'):
            self.verify(sources, replace(selected, native_pipeline='optimized'), published)

    def test_original_template_mutation_refuses_even_with_the_same_published_table(self):
        sources, selected, published = self.case(2)
        with tempfile.TemporaryDirectory(prefix='resident-publication-tamper-') as directory:
            root = Path(directory)/'compiled'
            shutil.copytree(self.compiled, root)
            binary = bytearray((root/'program.bin').read_bytes())
            binary[0] ^= 1
            (root/'program.bin').write_bytes(binary)
            with self.assertRaises(ValueError):
                self.verify(sources, selected, published, root)


if __name__ == '__main__':
    unittest.main()
