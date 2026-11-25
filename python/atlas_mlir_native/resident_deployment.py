"""Check resident deployment inputs without admitting a deployment or transport.

Current source tuples and the invocation-owned Merlin residency publication are
the roots of trust. Published region summaries supply candidates, not proof.
Packed bytes are taken only from that live publication; source archives are not
reopened or packed again here.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import re
from types import MappingProxyType, SimpleNamespace
from collections.abc import Mapping

from merlin.compiler.schedule.storage.application_residency import publication, validate_region_bindings
from merlin.compiler.schedule.storage.residency import ResidencyPlan, ResidencyRequest, ReservedInterval, plan_residency
from merlin.compiler.semantic.model import KernelRequest

from .binding import validate_io_window
from .memory_windows import dma_span_in_window
from .options import current_codegen_options
from .resident_loop_publication import verify_resident_loop_runtime


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _same(left, right, message):
    if _encoded(left) != _encoded(right):
        raise ValueError(message)


def _checked_source_ports(source, inputs, outputs):
    runtime = {key for key, _ in source.input_storages}
    effects = {node.id: node.effect for node in source.nodes}
    if (not isinstance(inputs, list) or not all(isinstance(port, str) for port in inputs)
            or not runtime <= set(inputs)
            or any(port not in runtime and effects.get(port) != 'constant' for port in inputs)
            or not isinstance(outputs, list) or outputs != list(source.outputs)):
        raise ValueError('resident deployment context ports differ from current source')
    # C wrappers may repeat source operands and receive source constants, or
    # embed those constants themselves. Only runtime boundary coverage is
    # mandatory here; resident providers prove their ordered C ABI below.


def _relative(value):
    if not isinstance(value, str):
        raise ValueError('resident deployment path must be a canonical relative string')
    path = PurePosixPath(value)
    if (path.is_absolute() or path.as_posix() != value or '..' in path.parts
            or '\\' in value or value in {'', '.'}):
        raise ValueError('resident deployment path must be a canonical relative string')
    return path


def _root(value):
    path = Path(value).absolute()
    if '..' in path.parts or any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError('resident deployment root is linked or noncanonical')
    path = path.resolve()
    if not path.is_dir():
        raise ValueError('resident deployment root is absent')
    return path


def _folder(root, relative):
    path = _relative(relative)
    for count in range(1, len(path.parts)+1):
        if (root/Path(*path.parts[:count])).is_symlink():
            raise ValueError('resident native publication path is linked')
    folder = root/Path(*path.parts)
    if not folder.is_dir() or not folder.resolve().is_relative_to(root):
        raise ValueError('resident native publication is absent or outside its root')
    return folder


def _replayed_choice(source, kind, family_claims):
    """The choice an export recorded: its kind and, under a generated policy, the family claimed for every member."""
    metadata = (('kind', kind),)
    if family_claims is not None:
        claims = dict(family_claims)
        families = {claims.get(node.id) for node in source.nodes if node.effect == 'pure'}
        if len(families) != 1 or None in families:
            raise ValueError('resident context differs from its generated family cover')
        metadata += (('numerical_family', families.pop()),)
    return SimpleNamespace(metadata=metadata)


@dataclass(frozen=True)
class CheckedResidentAsset:
    name: str
    address: int
    bytes: int
    alignment: int
    source_node: str
    source_name: str
    path: str
    sha256: str
    data: bytes
    layout: Mapping[str, object]


@dataclass(frozen=True)
class CheckedResidentContext:
    context_index: int
    region_id: str
    source_digest: str
    native_programs: tuple[str, ...]
    verification: str
    compiled_body_index: int | None = None
    source_binding_index: int | None = None
    execution_plan: Mapping[str, object] | None = None
    binding: Mapping[str, object] | None = None
    binary_sha256: str | None = None
    runtime_sha256: str | None = None
    runtime: Mapping[str, object] | None = None


@dataclass(frozen=True)
class CheckedResidentDeployment:
    assets: tuple[CheckedResidentAsset, ...]
    contexts: tuple[CheckedResidentContext, ...]
    source_request_digest: str
    residency_plan_sha256: str
    staging_base: int
    staging_bytes: int
    capacity_base: int
    capacity_bytes: int
    dma_base: int
    dma_capacity: int
    residency: ResidencyPlan


def checked_resident_deployment(regions, source_regions, native_root, target_source, *, source_request_digest,
                                family_claims=None):
    """Return checked assets/contexts; this does not enable normal deployment.

    ``source_regions`` is the trusted ordered ``(region_id, KernelRequest)``
    sequence from current outlining. The mandatory full source digest comes
    from the caller's authenticated intake, not a region summary or JSON hash.
    ``family_claims`` is the adapter's generated cover (node id, family), or
    None without a generated policy; replayed recipes select from it.
    Nonresident contexts carry source identity only and still need ordinary
    provider deployment validation. Capacity is the active declared physical
    window; board admission and host placement remain the caller's work.
    """
    # Check every supplied path lexically before probing either root, including
    # summaries whose source or allocation checks would otherwise fail first.
    regions, source_regions = tuple(regions), tuple(source_regions)
    native_paths = []
    for row in regions:
        paths = row.get('native_programs', []) if isinstance(row, dict) else None
        if (not isinstance(paths, list) or not all(isinstance(path, str) for path in paths)
                or len(paths) != len(set(paths))):
            raise ValueError('resident deployment native paths are repeated or invalid')
        native_paths.append(tuple(str(_relative(path)) for path in paths))
    options = current_codegen_options()
    files = publication()
    if (not options.application_residency or not isinstance(files, dict) or 'residency.json' not in files
            or not isinstance(options.resident_bindings, dict) or not options.resident_bindings):
        raise ValueError('resident deployment requires an active compiler residency publication and allocations')
    if (not isinstance(source_request_digest, str) or len(source_request_digest) != 64
            or any(char not in '0123456789abcdef' for char in source_request_digest)):
        raise ValueError('resident deployment requires the trusted full source digest')
    if not regions or len(regions) != len(source_regions):
        raise ValueError('resident deployment contexts do not cover current sources')
    record = json.loads(files['residency.json'])
    if (not isinstance(record, dict) or record.get('schema') != 'merlin.residency.v1'
            or record.get('source_request_digest') != source_request_digest):
        raise ValueError('resident deployment publication differs from the trusted full source')
    root, target_source = _root(native_root), _root(target_source)
    seen_ids, requested, inventories, asset_for_port = set(), options.resident_bindings, {}, {}
    from .application import AtlasApplicationAdapter

    for index, (row, current) in enumerate(zip(regions, source_regions, strict=True)):
        if (not isinstance(current, (tuple, list)) or len(current) != 2
                or not isinstance(current[0], str) or not isinstance(current[1], KernelRequest)):
            raise ValueError('resident deployment sources must be ordered source tuples')
        key, source = current
        if (not isinstance(row, dict) or row.get('id') != key or key in seen_ids
                or type(row.get('context_index')) is not int or row['context_index'] != index
                or row.get('source_request_digest') != source.digest()
                or row.get('placement') not in {'host', 'accelerator'}
                or not isinstance(row.get('description'), dict)):
            raise ValueError('resident deployment context identity or source order differs')
        seen_ids.add(key)
        _checked_source_ports(source, row.get('inputs'), row.get('outputs'))
        paths = row.get('native_programs', [])
        if (not isinstance(paths, list) or not all(isinstance(path, str) for path in paths)
                or len(paths) != len(set(paths))):
            raise ValueError('resident deployment native paths are repeated or invalid')
        description, selected = row['description'], requested.get(source.digest(), {})
        if not selected:
            if description.get('resident_bindings') or description.get('resident_loop_runtime'):
                raise ValueError('nonresident context claims a resident publication')
            continue
        if row['placement'] != 'accelerator' or not paths:
            raise ValueError('resident source ports require an accelerator native publication')
        alternative = SimpleNamespace(metadata=(('kind', description.get('kind')),))
        module = AtlasApplicationAdapter._resident_input_module(source, alternative)
        if module is None:
            raise ValueError('resident context has no supported provider input inventory')
        inventory = {item['node_id']: item for item in module.inventory_resident_inputs(source)}
        if not set(selected) <= set(inventory):
            raise ValueError('resident allocation contains an unsupported source input')
        for node_id in selected:
            item = inventory[node_id]
            name = 'parameter_'+sha256(json.dumps([node_id, item['layout']], sort_keys=True).encode()).hexdigest()
            expected = dict(item=item, node_id=node_id)
            if name in inventories:
                _same(inventories[name], expected, 'one resident asset has conflicting current layouts')
            inventories[name] = expected
            asset_for_port[source.digest(), node_id] = name
    validate_region_bindings(source_regions, tuple(SimpleNamespace(description=row['description']) for row in regions))
    staging_base, staging_bytes = validate_io_window()
    capacity_base, capacity_bytes = options.io_memory_capacity_base, options.io_memory_capacity_bytes
    if (type(capacity_base) is not int or type(capacity_bytes) is not int or capacity_bytes <= 0
            or not dma_span_in_window(capacity_base, capacity_bytes)
            or not capacity_base <= staging_base < staging_base+staging_bytes <= capacity_base+capacity_bytes):
        raise ValueError('resident deployment staging or declared capacity is invalid')
    geometry = AtlasApplicationAdapter.resident_memory_geometry()
    plan = plan_residency([ResidencyRequest(name, 'immutable_parameter', value['item']['bytes'],
        value['item']['alignment'], 'application', 'once_at_load') for name, value in inventories.items()],
        base=capacity_base, capacity=capacity_bytes, dma_base=geometry['dma_base'], dma_capacity=geometry['dma_capacity'],
        reserved=[ReservedInterval('regional_io', staging_base, staging_bytes,
            options.io_memory_alignment or geometry['alignment'], 'staging', 'region invocation', 'per_call')])
    _same({key: record.get(key) for key in plan.record()}, plan.record(),
          'resident allocation plan differs from current inventories or geometry')
    if record.get('plan_sha256') != plan.digest():
        raise ValueError('resident allocation plan hash differs from current replay')
    entries = {entry.name: entry for entry in plan.entries}
    expected_bindings = {}
    for _, source in source_regions:
        for node_id in requested.get(source.digest(), {}):
            name = asset_for_port[source.digest(), node_id]
            entry = entries[name]
            expected_bindings.setdefault(source.digest(), {})[node_id] = dict(address=entry.address, bytes=entry.bytes, immutable=True)
    _same(requested, expected_bindings, 'current resident allocations differ from the replayed asset plan')
    loads = record.get('loads')
    names = sorted(inventories)
    if (not isinstance(loads, list) or not all(isinstance(row, dict) for row in loads)
            or [row.get('name') for row in loads] != names):
        raise ValueError('resident initial loads are missing, extra, repeated or reordered')
    ports = record.get('host_parameters', [])
    if (not isinstance(ports, list) or not all(isinstance(port, dict) for port in ports)
            or len({port.get('node_id') for port in ports}) != len(ports)):
        raise ValueError('resident source parameter inventory is invalid')
    ports = {port['node_id']: port for port in ports}
    assets, payload_paths = [], set()
    for load in loads:
        name, value = load['name'], inventories[load['name']]
        entry, item, node_id = entries[name], value['item'], value['node_id']
        path = 'residency/'+name+'.bin'
        if (load.get('path') != path or load.get('source_node') != node_id
                or node_id not in ports or not isinstance(load.get('source_name'), str)
                or load.get('source_name') != ports[node_id].get('name')):
            raise ValueError('resident initial load source or path differs')
        _same(load.get('layout'), item['layout'], 'resident initial load layout differs')
        payload = files.get(path)
        if (type(payload) is not bytes or len(payload) != entry.bytes
                or load.get('sha256') != sha256(payload).hexdigest()
                or not dma_span_in_window(entry.address, entry.bytes)
                or entry.address < staging_base+staging_bytes and staging_base < entry.end):
            raise ValueError('resident packed asset bytes, hash or span differs')
        payload_paths.add(path)
        assets.append(CheckedResidentAsset(name, entry.address, entry.bytes, entry.alignment, node_id,
            load['source_name'], path, load['sha256'], payload, _freeze(item['layout'])))
    if set(files) != {'residency.json', *payload_paths}:
        raise ValueError('resident publication has missing or extra packed asset files')
    contexts, groups, owners = [None]*len(regions), {}, {}
    for index, (row, (_, source)) in enumerate(zip(regions, source_regions, strict=True)):
        description = row['description']
        selected = requested.get(source.digest(), {})
        if not selected:
            contexts[index] = CheckedResidentContext(index, row['id'], source.digest(), native_paths[index], 'source_identity_only')
            continue
        paths = native_paths[index]
        if len(paths) != 1:
            raise ValueError('resident deployment requires one native template per resident context')
        folder = _folder(root, paths[0])
        if description.get('resident_loop_runtime') is not None:
            body_index = row.get('compiled_body_index')
            representative = index if body_index is None else body_index
            if (type(representative) is not int or not 0 <= representative <= index
                    or representative != index and regions[representative].get('compiled_body_index') is not None):
                raise ValueError('resident shared body does not name an original preceding context')
            if paths != native_paths[representative]:
                raise ValueError('resident shared body template paths differ')
            if paths[0] in owners and owners[paths[0]] != representative:
                raise ValueError('resident template path is reused without a checked shared body')
            owners[paths[0]] = representative
            groups.setdefault(representative, []).append(index)
            continue
        if description.get('kind') != 'layout_device' or row.get('compiled_body_index') is not None:
            raise ValueError('resident context lacks a supported checked template publication')
        if paths[0] in owners:
            raise ValueError('resident template path is reused without a checked shared body')
        owners[paths[0]] = index
        from .layout_device import prepare_c_package

        function = description.get('binding_function', '')
        name = function[:-5] if isinstance(function, str) and function.endswith('_bind') else ''
        expected, emitted = prepare_c_package(source, folder, name, target_source=target_source)
        # The adapter adds its compiler recipe after the C package, so the replay recomputes it from the same source.
        from .compiler_interface import region_compiler_recipe

        recipe = region_compiler_recipe(source, _replayed_choice(source, description.get('kind'), family_claims))
        if recipe is not None:
            expected = dict(expected, compiler_recipe=recipe)
        _same(description, expected, 'resident layout description differs from strong ordinary source replay')
        if row['inputs'] != expected['input_node_ids'] or row['outputs'] != expected['output_node_ids']:
            raise ValueError('resident layout C port order differs from source replay')
        contexts[index] = CheckedResidentContext(index, row['id'], source.digest(), paths, 'ordinary_layout_source_replay',
            execution_plan=_freeze(expected['layout_device_plan']), binary_sha256=sha256(emitted['program.bin']).hexdigest())
    for representative, members in groups.items():
        original = regions[representative]
        if members[0] != representative or original.get('compiled_body_index') is not None:
            raise ValueError('resident shared body omits its original representative')
        descriptor = json.loads(_encoded(original['description']['resident_loop_runtime']))
        name = original['description'].get('package_name')
        if not isinstance(name, str) or re.fullmatch(r'[A-Za-z][A-Za-z0-9_]{0,63}', name) is None:
            raise ValueError('resident counted template package name is invalid')
        for binding_index, index in enumerate(members):
            description = regions[index]['description']
            if (description.get('kind') != 'already_quantized'
                    or description.get('execution_route') != 'checked_counted_image'
                    or type(description.get('source_binding_index')) is not int
                    or description['source_binding_index'] != binding_index):
                raise ValueError('resident context binding indices or counted route differ')
            from .already_quantized import prepare_source

            source = source_regions[index][1]
            precision = prepare_source(source)
            _same(description.get('precision_plan'), precision.record(), 'resident precision plan differs from current source')
            if (description.get('input_node_ids') != list(precision.input_ids)
                    or description.get('output_node_ids') != list(source.outputs)
                    or regions[index]['inputs'] != list(precision.input_ids)
                    or regions[index]['outputs'] != list(source.outputs)):
                raise ValueError('resident counted C port order differs from current source')
            _same(description.get('resident_loop_runtime'), descriptor, 'resident group runtime descriptors differ')
        checked = verify_resident_loop_runtime(source_regions[representative][1],
            _folder(root, native_paths[representative][0]), [source_regions[index][1] for index in members],
            descriptor, name=name, target_source=target_source)
        runtime = _freeze(descriptor)
        for binding_index, (index, bound) in enumerate(zip(members, checked.bound, strict=True)):
            source = source_regions[index][1]
            contexts[index] = CheckedResidentContext(index, regions[index]['id'], source.digest(), native_paths[index],
                'resident_counted_source_replay', representative, binding_index, _freeze(bound.execution_plan),
                _freeze(bound.receipt), bound.receipt['binary_sha256'], checked.runtime_sha256, runtime)
    return CheckedResidentDeployment(tuple(assets), tuple(contexts), source_request_digest, plan.digest(),
        staging_base, staging_bytes, capacity_base, capacity_bytes, geometry['dma_base'], geometry['dma_capacity'], plan)
