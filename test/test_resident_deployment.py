"""Checked normal-deployment components from real compiler residency scopes."""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from atlas_mlir_native.application import AtlasApplicationAdapter
from atlas_mlir_native.binding import DRAM_BASE
from atlas_mlir_native.options import codegen_options, current_codegen_options
from atlas_mlir_native import layout_device, quantized_loop
from atlas_mlir_native.resident_deployment import checked_resident_deployment
from merlin.compiler.schedule.storage import application_residency as residency
from merlin.compiler.lower.c.parameters import ApplicationParameters
from merlin.compiler.schedule.region_loop import correspondence
from merlin.compiler.partition.semantic_regions import outline_semantic_regions
from merlin.compiler.lower.c.weights import WeightEntry
from merlin.compiler.semantic.model import SemanticNode
from merlin.compiler.semantic.search.exploration import SearchLimits
from atlas_test_fixtures.semantic_regions import plan_for
from test_vmem_window import source_request
from test_layout_device import cases as layout_cases
from test_looped_mnk import build_snapshot
from test_resident_loop_image import ROOT


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':')).encode()


def set_record(files, record):
    files['residency.json'] = (json.dumps(record, sort_keys=True, indent=2)+'\n').encode()


def weights(root, source, tensors):
    payload, header, entries = bytearray(), {}, []
    for key, dtype, values in tensors:
        width = {'I8': 1, 'I32': 4, 'BF16': 2}[dtype]
        payload.extend(bytes((-len(payload)) % width))
        start, data = len(payload), values.tobytes()
        payload.extend(data)
        name = 'stored_'+key
        header[name] = dict(dtype=dtype, shape=list(values.shape), data_offsets=[start, len(payload)])
        entries.append(WeightEntry(int(key[3:]), name, dtype, list(values.shape), start, len(data)))
    metadata = json.dumps(header).encode()
    archive = struct.pack('<Q', len(metadata))+metadata+payload
    path = root/'weights.safetensors'
    path.write_bytes(archive)
    parameters = ApplicationParameters(source.digest(), 'a'*64, sha256(archive).hexdigest(),
        sha256(payload).hexdigest(), len(payload), tuple(entries), path)
    parameters.check(source)
    return parameters


class Adapter(AtlasApplicationAdapter):
    def __init__(self, plan):
        self.selected_plan = plan

    def plan(self, source):
        return self.selected_plan


@contextmanager
def active(case, *, files=None, options=None):
    token = residency._PUBLICATION.set(case.files if files is None else files)
    try:
        with codegen_options(case.options if options is None else options):
            yield
    finally:
        residency._PUBLICATION.reset(token)


class ResidentDeploymentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix='atlas-resident-deployment-')
        cls.root = Path(os.environ.get('ATLAS_RESIDENT_DEPLOYMENT_RECEIPTS', cls.temporary.name))
        cls.root.mkdir(parents=True, exist_ok=True)
        cls.options = replace(current_codegen_options(), application_residency=True,
            io_memory_base=DRAM_BASE, io_memory_bytes=262144, io_memory_alignment=2048,
            io_memory_capacity_base=0x80000000, io_memory_capacity_bytes=0x40000000)
        with codegen_options(cls.options):
            cls.snapshot = build_snapshot(cls.root/'native')
        cls.cases = [cls.build_quantized(trips) for trips in (2, 22)]
        cls.layout = cls.build_layout()

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    @classmethod
    def finish_case(cls, source, plan, root, tensors, kind):
        adapter = Adapter(plan)
        parameters = weights(root, source, tensors)
        outlined = outline_semantic_regions(source, entry='resident_components', plan=plan)
        sources = outlined.regions
        segments = root/'segments'
        segments.mkdir()
        with codegen_options(cls.options), residency.residency_scope(source, adapter, parameters, None):
            selected, packages = current_codegen_options(), []
            choices = {choice.id: choice for choice in plan.selected}
            for index, (key, region) in enumerate(sources):
                choice = choices[key]
                if choice.placement == 'host':
                    packages.append(adapter.export_region(region, choice, f'host_{index}', segments, ROOT))
                elif index == 0:
                    module = quantized_loop if kind == 'already_quantized' else layout_device
                    module.compile_source(cls.snapshot, region, target_source=ROOT,
                        destination=segments/'body', limits=SearchLimits())
                    packages.append(adapter.export_region(region, choice, 'body', segments, ROOT))
                else:
                    package = adapter.bind_loop_region(packages[0], region, choice,
                        correspondence(sources[0][1].record(), region.record()), segments, ROOT)
                    if package is None:
                        raise AssertionError('synthetic resident region did not bind its template')
                    packages.append(replace(package, body_index=0))
            packages = adapter.finalize_loop_regions(tuple(packages), target_source=ROOT)
            rows = [dict(id=key, placement=choices[key].placement, context_index=index,
                source_request_digest=region.digest(), description=package.description,
                native_programs=list(package.programs), inputs=list(package.inputs), outputs=list(package.outputs),
                **({'compiled_body_index': package.body_index} if package.body_index is not None else {}))
                for index, ((key, region), package) in enumerate(zip(sources, packages, strict=True))]
            case = SimpleNamespace(source=source, sources=sources, rows=rows, root=root,
                                   options=selected, files=deepcopy(residency.publication()))
            checked_resident_deployment(rows, sources, root, ROOT, source_request_digest=source.digest())
        return case

    @classmethod
    def build_quantized(cls, trips):
        root = cls.root/f'counted-{trips}'
        root.mkdir()
        # Each packed operand/exponent asset occupies a whole 4 KiB page,
        # keeping this fixture within the admitted LUI-only relocation form.
        base, requests, tensors = source_request(3, 35, 33, batch=()), [], []
        for trip in range(trips):
            names = {key: f'arg{trip*4+index}' for index, key in enumerate(('a', 'sx', 'w', 'sw'))}
            names['y'] = f'result_{trip}'
            request = replace(base, nodes=tuple(replace(node, id=names[node.id],
                inputs=tuple(names[key] for key in node.inputs)) for node in base.nodes),
                outputs=(names['y'],), input_storages=tuple((names[key], value) for key, value in base.input_storages))
            requests.append(request)
            weight = np.full((35, 33), 0x38, dtype=np.uint8)
            weight[trip % 35, :] = 0x40
            scale = (np.arange(35, dtype='<i4') % 3-1).reshape(35, 1)
            tensors.extend(((names['w'], 'I8', weight), (names['sw'], 'I32', scale)))
        host = SemanticNode('host_result', 'reshape', (requests[0].nodes[0].id,), requests[0].nodes[0].type)
        source = replace(base,
            nodes=tuple(node for request in requests for node in request.nodes if node.effect == 'input')+
                  tuple(request.nodes[-1] for request in requests)+(host,),
            outputs=tuple(request.outputs[0] for request in requests)+(host.id,),
            output_storages=('external',)*(trips+1),
            input_storages=tuple(row for request in requests for row in request.input_storages))
        plan = plan_for(source, tuple((index,) for index in range(trips+1)))
        plan = replace(plan, selected=tuple(replace(choice,
            placement='host' if index == trips else 'accelerator',
            implementation='synthetic_host' if index == trips else 'synthetic_counted',
            metadata=(('kind', 'layout' if index == trips else 'already_quantized'),))
            for index, choice in enumerate(plan.selected)))
        return cls.finish_case(source, plan, root, tensors, 'already_quantized')

    @classmethod
    def build_layout(cls):
        root = cls.root/'layout'
        root.mkdir()
        base = layout_cases()['dense_dma']
        source = replace(base, nodes=(replace(base.nodes[0], id='arg0'),
            replace(base.nodes[-1], inputs=('arg0',))), input_storages=(('arg0', 'external'),))
        plan = plan_for(source, ((0,),))
        plan = replace(plan, selected=tuple(replace(choice, placement='accelerator',
            implementation='synthetic_layout', metadata=(('kind', 'layout_device'),)) for choice in plan.selected))
        return cls.finish_case(source, plan, root,
            [('arg0', 'BF16', np.arange(96, dtype='<u2').reshape(3, 32))], 'layout_device')

    def check(self, case, *, rows=None, sources=None, files=None, options=None, digest=None, root=None):
        with active(case, files=files, options=options):
            return checked_resident_deployment(case.rows if rows is None else rows,
                case.sources if sources is None else sources, case.root if root is None else root, ROOT,
                source_request_digest=case.source.digest() if digest is None else digest)

    def test_actual_publications_return_one_owned_load_per_asset_and_verified_contexts(self):
        for case in (*self.cases, self.layout):
            with self.subTest(contexts=len(case.rows)):
                with patch.object(AtlasApplicationAdapter, 'pack_resident_input',
                        side_effect=AssertionError('deployment checker duplicated source packing')):
                    result = self.check(case)
                loads = json.loads(case.files['residency.json'])['loads']
                self.assertEqual(len(result.assets), len(loads))
                self.assertEqual(len({asset.name for asset in result.assets}), len(loads))
                self.assertEqual(len(result.contexts), len(case.rows))
                self.assertEqual(result.residency.digest(), result.residency_plan_sha256)
                self.assertEqual([context.context_index for context in result.contexts], list(range(len(case.rows))))
                for asset in result.assets:
                    self.assertEqual(asset.data, case.files[asset.path])
                    self.assertEqual(asset.sha256, sha256(asset.data).hexdigest())
                    self.assertTrue(asset.address+asset.bytes <= result.staging_base
                                    or result.staging_base+result.staging_bytes <= asset.address)
                if case is not self.layout:
                    counted = result.contexts[:-1]
                    self.assertEqual([context.source_binding_index for context in counted], list(range(len(counted))))
                    self.assertTrue(all(context.compiled_body_index == 0 for context in counted))
                    self.assertTrue(all(context.runtime is counted[0].runtime for context in counted))
                    self.assertEqual(result.contexts[-1].verification, 'source_identity_only')
                else:
                    self.assertEqual(result.contexts[0].verification, 'ordinary_layout_source_replay')
                with self.assertRaises(FrozenInstanceError):
                    result.staging_base = 0
                with self.assertRaises(TypeError):
                    result.assets[0].layout['forged'] = True

    def test_context_source_index_order_table_and_native_paths_refuse(self):
        case = self.cases[0]
        for tamper in ('host', 'context_index', 'source_digest', 'binding_index', 'table', 'native_path', 'ports', 'port_order', 'precision'):
            with self.subTest(tamper=tamper):
                rows = deepcopy(case.rows)
                if tamper == 'host':
                    rows[0]['placement'] = 'host'
                elif tamper == 'context_index':
                    rows[0]['context_index'] = 1
                elif tamper == 'source_digest':
                    rows[0]['source_request_digest'] = 'f'*64
                elif tamper == 'binding_index':
                    rows[1]['description']['source_binding_index'] = 0
                elif tamper == 'table':
                    for row in rows[:2]:
                        runtime = deepcopy(row['description']['resident_loop_runtime'])
                        runtime['trip_bindings'] = runtime['trip_bindings'][::-1]
                        runtime['table_sha256'] = sha256(encoded(runtime['trip_bindings'])).hexdigest()
                        row['description']['resident_loop_runtime'] = runtime
                elif tamper == 'native_path':
                    rows[0]['native_programs'] = ['../outside-native-root']
                elif tamper == 'ports':
                    rows[0]['inputs'] = rows[0]['inputs'][:-1]
                elif tamper == 'port_order':
                    rows[0]['inputs'].reverse()
                else:
                    rows[0]['description']['precision_plan']['shape'][0] += 1
                with self.assertRaises(ValueError):
                    self.check(case, rows=rows)
        for sources in (case.sources[:-1], case.sources[::-1]):
            with self.assertRaises(ValueError):
                self.check(case, sources=sources)
        with self.assertRaises(ValueError):
            self.check(case, digest='f'*64)
        with self.assertRaises(ValueError):
            checked_resident_deployment(case.rows, case.sources, case.root, ROOT,
                                        source_request_digest=case.source.digest())
        link = case.root/'segments/linked'
        link.symlink_to('body', target_is_directory=True)
        linked = deepcopy(case.rows)
        for row in linked[:2]:
            row['native_programs'] = ['segments/linked']
        with self.assertRaisesRegex(ValueError, 'linked'):
            self.check(case, rows=linked)

    def test_missing_extra_reordered_loads_assets_and_unsafe_asset_paths_refuse(self):
        case = self.cases[0]
        for tamper in ('missing_load', 'extra_load', 'reordered_loads', 'bytes', 'missing_file', 'extra_file', 'load_path'):
            with self.subTest(tamper=tamper):
                files = deepcopy(case.files)
                record = json.loads(files['residency.json'])
                path = record['loads'][0]['path']
                if tamper == 'missing_load':
                    record['loads'].pop()
                elif tamper == 'extra_load':
                    record['loads'].append(deepcopy(record['loads'][0]))
                elif tamper == 'reordered_loads':
                    record['loads'].reverse()
                elif tamper == 'bytes':
                    files[path] = bytes([files[path][0]^1])+files[path][1:]
                elif tamper == 'missing_file':
                    files.pop(path)
                elif tamper == 'extra_file':
                    files['residency/extra.bin'] = b'unclaimed'
                else:
                    record['loads'][0]['path'] = '../../outside-publication.bin'
                set_record(files, record)
                with self.assertRaises(ValueError):
                    self.check(case, files=files)

    def test_layout_owners_are_unique(self):
        layout = self.layout
        rows = deepcopy(layout.rows)
        duplicate = deepcopy(rows[0])
        duplicate.update(id='another_layout_context', context_index=1)
        rows.append(duplicate)
        sources = (*layout.sources, ('another_layout_context', layout.sources[0][1]))
        with self.assertRaisesRegex(ValueError, 'reused without a checked shared body'):
            self.check(layout, rows=rows, sources=sources)

    def test_wrong_current_allocation_staging_and_self_rehashed_layout_native_refuse(self):
        case = self.cases[0]
        with self.assertRaises(ValueError):
            self.check(case, options=replace(case.options, io_memory_base=case.options.io_memory_base+2048))
        files, rows = deepcopy(case.files), deepcopy(case.rows)
        record = json.loads(files['residency.json'])
        source_digest = case.sources[0][1].digest()
        node = next(iter(case.options.resident_bindings[source_digest]))
        address = case.options.resident_bindings[source_digest][node]['address']+1048576
        bindings = deepcopy(case.options.resident_bindings)
        bindings[source_digest][node]['address'] = address
        record['bindings'] = bindings
        for binding in rows[0]['description']['resident_bindings']:
            if binding['node_id'] == node:
                binding['address'] = address
        name = next(load['name'] for load in record['loads'] if load['source_node'] == node)
        for entry in record['entries']:
            if entry['name'] == name:
                entry['address'] = address
        fields = ('schema', 'arena', 'dma_aperture', 'required_bytes', 'reserved_names', 'entries')
        record['plan_sha256'] = sha256(encoded({key: record[key] for key in fields})).hexdigest()
        set_record(files, record)
        with self.assertRaisesRegex(ValueError, 'allocation plan differs'):
            self.check(case, rows=rows, files=files, options=replace(case.options, resident_bindings=bindings))
        layout = self.layout
        altered = deepcopy(layout.rows)
        altered[0]['description']['layout_device_plan']['input_addresses'][0] += 32
        altered[0]['description']['native_manifest_sha256'] = 'f'*64
        with self.assertRaises(ValueError):
            self.check(layout, rows=altered)
        with tempfile.TemporaryDirectory(prefix='resident-layout-native-forgery-') as directory:
            altered = deepcopy(layout.rows)
            root = Path(directory)/'native'
            shutil.copytree(layout.root, root)
            folder = root/layout.rows[0]['native_programs'][0]
            binary = bytearray((folder/'program.bin').read_bytes())
            binary[0] ^= 0x80
            (folder/'program.bin').write_bytes(binary)
            manifest = json.loads((folder/'manifest.json').read_bytes())
            manifest['binary_sha256'] = sha256(binary).hexdigest()
            manifest['stage_artifacts']['program.bin'] = sha256(binary).hexdigest()
            (folder/'manifest.json').write_text(json.dumps(manifest))
            altered[0]['description']['native_manifest_sha256'] = sha256((folder/'manifest.json').read_bytes()).hexdigest()
            with self.assertRaises(ValueError):
                self.check(layout, rows=altered, root=root)


if __name__ == '__main__':
    unittest.main()
