"""Source-bound row blocks of fully checked counted quantized contractions.

Only M and the batch instance are partitioned. Each child retains the complete
ordered K chain, unchanged code bytes, and signed-exponent device restoration.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
import json
from math import prod
from pathlib import Path

from merlin.semantic_compiler.target_binding import NativeCompilationError, verify_native_publication

from .already_quantized import QuantizedPlan, prepare_source
from .binding import AtlasNativeBinding, TILE_BYTES, dram_slots
from .quantized_loop_sizing import size_counted, selected_route


def _json(value):
    return (json.dumps(value, sort_keys=True, indent=2) + '\n').encode()


class NoFittingRowBlock(NativeCompilationError):
    """Every row candidate, including one logical row, was explicitly refused."""

    def __init__(self, source, full, reasons):
        self.source = source
        self.full = full
        self.reasons = tuple(reasons)
        super().__init__('resource_limit', 'no counted quantized row block fits, including one logical row')

    def record(self):
        return dict(schema='atlas.quantized_row_block_refusal.v1',
            source_request_digest=self.source.digest(), target_identity=self.source.target_identity,
            full_counted_sizing=self.full, attempts=list(self.reasons),
            minimum_rows=1, fallback='k32_only_after_explicit_resource_refusal')


@dataclass(frozen=True)
class RowBlockSchedule:
    source: object
    plan: QuantizedPlan
    block_rows: int
    row_partition: tuple[int, ...]
    bodies: tuple
    launches: tuple[dict, ...]
    io_memory_bytes: int
    profile_sha256: str

    def record(self):
        return dict(schema='atlas.quantized_row_block_schedule.v1',
            source_request_digest=self.source.digest(), target_identity=self.source.target_identity,
            input_node_ids=list(self.plan.input_ids), output_node_ids=list(self.source.outputs),
            shape=[self.plan.m, self.plan.n, self.plan.k], batch_shape=list(self.plan.batch_shape),
            block_rows=self.block_rows, row_partition=list(self.row_partition), io_memory_bytes=self.io_memory_bytes,
            profile_sha256=self.profile_sha256, functional_instruction_limit=1000000,
            bodies=[dict(index=i, path=f'segments/body_{i:03d}', request_digest=body.digest(),
                         shape=list(body.node(self.plan.output_id).type.shape))
                    for i, body in enumerate(self.bodies)],
            launches=list(self.launches), launches_per_invocation=len(self.launches),
            k_order='complete unchanged ordered K32 chain within each counted image',
            restoration_placement='device', weight_conversion='none; raw byte slices only',
            runtime_arithmetic='none; raw copies, padding, and counted device dispatch only')


def _child(plan, batch, first, rows):
    """Return the exact dense child and its parent byte views, without sizing."""
    source = plan.source
    specifications = {}

    def view(node_id, shape, offset, count):
        candidate = (shape, offset, count)
        if node_id in specifications and specifications[node_id] != candidate:
            raise NativeCompilationError('unsupported_semantics',
                'aliased quantized row operands need incompatible byte views')
        specifications[node_id] = candidate

    for role, key in enumerate(plan.operand_ids):
        operand = source.node(key)
        prefix = operand.type.shape[:-2]
        selected_batch = batch if prefix else 0
        count = plan.m if role == 0 else plan.n
        child_rows = rows if role == 0 else count
        start = first if role == 0 else 0
        shape = (*((1,) * len(prefix)), child_rows, plan.k)
        view(key, shape, (selected_batch * count + start) * plan.k, child_rows * plan.k)
        exponent = source.node(plan.exponent_ids[role])
        if not exponent.type.shape:
            view(exponent.id, (), 0, 4)
        elif exponent.type.shape[-2:] == (1, 1):
            view(exponent.id, (*((1,) * len(prefix)), 1, 1), selected_batch * 4, 4)
        else:
            view(exponent.id, (*((1,) * len(prefix)), child_rows, 1),
                 (selected_batch * count + start) * 4, child_rows * 4)
    nodes = []
    for node in source.nodes:
        shape = ((*((1,) * len(plan.batch_shape)), rows, plan.n)
                 if node.id == plan.output_id else specifications[node.id][0])
        nodes.append(replace(node, type=replace(node.type, shape=shape)))
    constants = []
    for binding in source.constants:
        _, offset, count = specifications[binding.node_id]
        constants.append(replace(binding, data_hex=binding.data_hex[offset * 2:(offset + count) * 2]))
    # Position is an invocation property. Equal shapes and constant bytes share
    # a child request and image across all original batches and row positions.
    identity = sha256(_json(dict(parent=source.digest(), rows=rows))).hexdigest()
    child = replace(source, nodes=tuple(nodes), constants=tuple(constants), source_identity=identity)
    prepare_source(child)
    slices = [dict(node_id=key, offset_bytes=specifications[key][1], bytes=specifications[key][2])
              for key in plan.input_ids]
    for item in slices:
        node = source.node(item['node_id'])
        width = 4 if node.type.dtype in ('i32', 'si32') else 1
        if item['offset_bytes'] + item['bytes'] > prod(node.type.shape) * width:
            raise ValueError('quantized row view exceeds its original operand')
    return child, slices


def schedule_for_rows(source, block_rows, *, row_partition=None):
    """Rebuild a recorded partition for export, without compiler invocations."""
    plan = prepare_source(source)
    if plan.epilogue_precision is not None:
        raise NativeCompilationError('unsupported_semantics',
            'row-blocked contraction must retain its separate epilogue')
    if (type(block_rows) is not int or not 0 < block_rows <= plan.m
            or (block_rows != plan.m and block_rows >= 32 and block_rows % 32)):
        raise ValueError('row block must be a positive tile-row boundary or the complete M extent')
    if row_partition is None:
        row_partition = tuple(min(block_rows, plan.m - first) for first in range(0, plan.m, block_rows))
    row_partition = tuple(row_partition)
    remaining = plan.m
    for length in row_partition:
        if (type(length) is not int or not 0 < length <= min(block_rows, remaining)
                or (length != remaining and length >= 32 and length % 32)):
            raise ValueError('invalid exact row partition')
        remaining -= length
    if not row_partition or row_partition[0] != block_rows or remaining:
        raise ValueError('row partition must cover every original row exactly')
    bodies, indices, launches = [], {}, []
    for batch in range(prod(plan.batch_shape)):
        first = 0
        for rows in row_partition:
            child, slices = _child(plan, batch, first, rows)
            key = child.digest()
            if key not in indices:
                indices[key] = len(bodies)
                bodies.append(child)
            width = 2 if plan.output_dtype == 'bf16' else 4
            launches.append(dict(body_index=indices[key], batch_index=batch, row_start=first, rows=rows,
                input_slices=slices, output_offset_bytes=(batch * plan.m + first) * plan.n * width,
                output_bytes=rows * plan.n * width))
            first += rows
    return RowBlockSchedule(source, plan, block_rows, row_partition, tuple(bodies), tuple(launches),
                            dram_slots() * TILE_BYTES, AtlasNativeBinding.profile().digest())


def _counted(sizing):
    record = sizing.record()
    if record.get('route') == 'counted':
        return True
    if (record.get('route') != 'k32_tiles' or record.get('reason') not in
            {'selected_dram_window', 'encoded_instruction_memory',
             'untimed_instruction_bound', 'physical_instruction_bound'}):
        raise ValueError('row scheduling requires an explicit counted resource decision')
    return False


def prepare_schedule(source, *, target_source=None, limits=None):
    """Choose the largest admitted M block; errors/timeouts never select K32."""
    plan = prepare_source(source)
    full = size_counted(plan, target_source=target_source, limits=limits)
    if _counted(full):
        return None
    if plan.epilogue_precision is not None:
        raise NativeCompilationError('unsupported_semantics',
            'row-blocked contraction must retain its separate epilogue')
    # The sequence is bounded by source M. Do not binary-search: admission of
    # a different scalar-restoration budget need not be monotone in M.
    refusals = []
    checked = {}

    def candidates(maximum):
        return (maximum, *range(((maximum - 1) // 32) * 32, 0, -32),
                *range(min(maximum - 1, 31), 0, -1))

    def fits(length, starts):
        for batch in range(prod(plan.batch_shape)):
            for first in starts:
                child, _ = _child(plan, batch, first, length)
                key = child.digest()
                if key not in checked:
                    checked[key] = size_counted(prepare_source(child), target_source=target_source, limits=limits)
                decision = checked[key]
                if not _counted(decision):
                    refusals.append(dict(block_rows=length, batch_index=batch, row_start=first,
                                         child_request_digest=key, sizing=decision.record()))
                    return False
        return True

    for main_rows in candidates(plan.m):
        if fits(main_rows, range(0, plan.m - main_rows + 1, main_rows)):
            break
    else:
        raise NoFittingRowBlock(source, full.record(), refusals)
    partition = [main_rows] * (plan.m // main_rows)
    first = sum(partition)
    while first < plan.m:
        for tail in candidates(plan.m - first):
            if fits(tail, (first,)):
                partition.append(tail)
                first += tail
                break
        else:
            raise NoFittingRowBlock(source, full.record(), refusals)
    return schedule_for_rows(source, main_rows, row_partition=partition)


def compile_source(snapshot, source, schedule, *, target_source, destination, limits):
    """Publish each distinct counted child through the existing native path."""
    from .quantized_loop import compile_source as compile_counted

    rebuilt = schedule_for_rows(source, schedule.block_rows, row_partition=schedule.row_partition)
    if schedule.record() != rebuilt.record():
        raise ValueError('row schedule differs from source or selected target profile')
    # Do not silently publish a K32 child if settings changed after planning.
    for child in schedule.bodies:
        if not _counted(size_counted(prepare_source(child), target_source=target_source, limits=limits)):
            raise NativeCompilationError('resource_limit', 'selected counted row block no longer fits')
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    (destination / 'segments').mkdir()
    records = {'source-request.json': source.record(), 'already-quantized-plan.json': schedule.plan.record(),
               'row-block-schedule.json': schedule.record()}
    for name, record in records.items():
        (destination / name).write_bytes(_json(record))
    segments = []
    for index, child in enumerate(schedule.bodies):
        relative = f'segments/body_{index:03d}'
        leaf = destination / relative
        compile_counted(snapshot, child, target_source=target_source, destination=leaf, limits=limits)
        manifest = json.loads((leaf / 'manifest.json').read_text())
        if selected_route(prepare_source(child), leaf) != 'counted':
            raise ValueError('row child publication is not a counted image')
        segments.append(dict(index=index, path=relative, request_digest=child.digest(),
            binary_sha256=manifest['binary_sha256'],
            manifest_sha256=sha256((leaf / 'manifest.json').read_bytes()).hexdigest()))
    manifest = dict(schema='atlas.quantized_row_block_application.v1', artifact_kind='program_set',
        engine='merlin_native', compile_mode='diagnostic', qualification_scope='diagnostic_only',
        request_digest=source.digest(), target_identity=source.target_identity, segments=segments,
        launches_per_invocation=len(schedule.launches), execution_requires='ordered_exact_quantized_row_slices',
        stage_artifacts={name: sha256((destination / name).read_bytes()).hexdigest() for name in records})
    (destination / 'row-block-manifest.json').write_bytes(_json(manifest))
    verify_publication(source, destination, target_source=target_source)
    return manifest


def verify_publication(source, compiled, *, target_source):
    """Authenticate the original source, selected partition, and counted leaves."""
    root = Path(compiled)
    expected = {'source-request.json', 'already-quantized-plan.json', 'row-block-schedule.json',
                'row-block-manifest.json', 'segments'}
    if (root.is_symlink() or not root.is_dir() or {p.name for p in root.iterdir()} != expected
            or any((root / name).is_symlink() for name in expected)):
        raise ValueError('row-block publication inventory differs')
    record = json.loads((root / 'row-block-schedule.json').read_text())
    schedule = schedule_for_rows(source, record['block_rows'], row_partition=record['row_partition'])
    if (record != schedule.record()
            or json.loads((root / 'source-request.json').read_text()) != source.record()
            or json.loads((root / 'already-quantized-plan.json').read_text()) != schedule.plan.record()):
        raise ValueError('row-block source, schedule, or target profile differs')
    manifest = json.loads((root / 'row-block-manifest.json').read_text())
    expected_manifest = dict(schema='atlas.quantized_row_block_application.v1', artifact_kind='program_set',
        engine='merlin_native', compile_mode='diagnostic', qualification_scope='diagnostic_only',
        request_digest=source.digest(), target_identity=source.target_identity,
        launches_per_invocation=len(schedule.launches), execution_requires='ordered_exact_quantized_row_slices',
        stage_artifacts={name: sha256((root / name).read_bytes()).hexdigest()
                         for name in expected - {'row-block-manifest.json', 'segments'}})
    if {k: v for k, v in manifest.items() if k != 'segments'} != expected_manifest:
        raise ValueError('row-block scheduling envelope differs')
    names = {f'body_{i:03d}' for i in range(len(schedule.bodies))}
    if {p.name for p in (root / 'segments').iterdir()} != names:
        raise ValueError('row-block child inventory differs')
    segments = []
    inventory_sha = sha256((Path(target_source) / 'docs/selected-variant-inventory.json').read_bytes()).hexdigest()
    for index, child in enumerate(schedule.bodies):
        relative = f'segments/body_{index:03d}'
        leaf = root / relative
        if leaf.is_symlink() or not leaf.is_dir() or (leaf / 'manifest.json').is_symlink():
            raise ValueError('row-block child is absent or linked')
        detail = json.loads((leaf / 'manifest.json').read_text())
        verify_native_publication(leaf, detail, engine='merlin_native',
                                 request_digest=child.digest(), target_identity=source.target_identity)
        if (json.loads((leaf / 'source-request.json').read_text()) != child.record()
                or detail.get('target_inventory_sha256') != inventory_sha
                or selected_route(prepare_source(child), leaf) != 'counted'):
            raise ValueError('row-block child source, inventory, or counted route differs')
        segments.append(dict(index=index, path=relative, request_digest=child.digest(),
            binary_sha256=detail['binary_sha256'],
            manifest_sha256=sha256((leaf / 'manifest.json').read_bytes()).hexdigest()))
    if manifest.get('segments') != segments:
        raise ValueError('row-block leaf publication identity differs')
    return schedule, manifest
