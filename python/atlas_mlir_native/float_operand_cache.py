"""Retain a proved repeated BF16 operand in invocation-local scratch tiles."""
from dataclasses import dataclass, replace
from math import gcd, prod

from merlin.compile.layout_c import repeated_dense_read_period
from .binding import TILE_BYTES
from .private_staging import PRIVATE_STAGING_TILE_LIMIT

TILE_ELEMENTS = TILE_BYTES // 2


@dataclass(frozen=True)
class RepeatedInput:
    port: int
    period: int

    @property
    def tiles(self):
        return self.period // gcd(self.period, TILE_ELEMENTS)

    def record(self):
        return dict(input_index=self.port, period_elements=self.period)


def with_repeated_input(plan, input_layout):
    """Derive repetition from the original view, never from sampled values."""
    if (input_layout is None or not plan.private_output
            or plan.op not in ('bf16_add','bf16_sub') or len(plan.operands) != 2):
        return plan
    physical, view = input_layout
    keys = [node.id for node in plan.operands]
    if (view.id not in keys or physical.id in keys or physical.effect != 'input' or physical.op != 'input'
            or physical.inputs or physical.attrs or physical.index_maps
            or view.type.dtype != 'bf16' or view.type.numerical_policy != 'raw-bit-pattern'
            or plan.count > prod(view.type.shape)):
        return plan
    period = repeated_dense_read_period(physical, view)
    if period is None:
        return plan
    repeated = RepeatedInput(keys.index(view.id), period)
    # Reuse the existing target staging budget, leaving ordinary spill space.
    # A complete cycle plus its starting phase must fit in the authored view.
    if (repeated.tiles > PRIVATE_STAGING_TILE_LIMIT
            or prod(view.type.shape) < period + repeated.tiles*TILE_ELEMENTS):
        return plan
    return replace(plan, repeated_input=repeated)


def cached_tiles(plan):
    """Only full leaves that actually reuse a tile use compact transport."""
    repeated = plan.repeated_input
    if (repeated is None or plan.tile_lanes != TILE_ELEMENTS
            or plan.count % TILE_ELEMENTS or plan.count // TILE_ELEMENTS <= repeated.tiles):
        return 0
    return repeated.tiles


def published_input_sizes(record, dense_sizes):
    """Validate the compact input footprint for an authenticated runtime plan.

    This checks the target transport contract, not the source indexing proof;
    compilation and package replay establish the latter from the source IR.
    """
    cache = record.get('input_cache')
    if not isinstance(cache, dict):
        raise AssertionError('repeated input needs its cache descriptor')
    port, period, tiles = (cache.get(key) for key in
                           ('input_index', 'period_elements', 'tiles'))
    logical = record.get('logical_output_bytes')
    if (record.get('schema') != 'atlas.exact_float.v1'
            or record.get('operation') not in ('bf16_add', 'bf16_sub')
            or record.get('private_output') is not True
            or record.get('publication_contract') !=
                'device result is private until zero status; discard all result bytes on failure'
            or len(dense_sizes) != 2
            or type(port) is not int or port not in (0, 1)
            or type(period) is not int or not 0 < period < 1 << 27
            or type(tiles) is not int or not 0 < tiles <= PRIVATE_STAGING_TILE_LIMIT
            or tiles != period // gcd(period, TILE_ELEMENTS)
            or type(logical) is not int or logical % TILE_BYTES
            or logical <= tiles*TILE_BYTES
            or dense_sizes != [logical, logical]
            or cache.get('initialization') != 'each invocation'
            or cache.get('transport') != 'complete repeated dense tile cycle'
            or record.get('boundary_transport_layout') !=
                'compact-repeated-input-cycle-and-dense-private-output'):
        raise AssertionError('invalid repeated input transport geometry')
    sizes = list(dense_sizes)
    sizes[port] = tiles*TILE_BYTES
    return sizes


def initialize_cache(builder, plan):
    """Load immutable cache slots afresh for this device invocation."""
    for index in range(plan.cached_tiles):
        name = f'input_cache_{index}'
        builder.slot(name)
        address = plan.input_addresses[plan.repeated_input.port] + index*TILE_BYTES
        builder.save(name, builder.dma_load(builder.constant(address), TILE_BYTES))


def restore_cached_input(builder, plan, slot, index):
    """Select a constant scratch slot while retaining the ordinary state edges."""
    if plan.cached_tiles == 1:
        builder.save(slot, builder.restore('input_cache_0'))
        return
    state_type = '!atlas.virtual_state'
    for tile in range(plan.cached_tiles):
        if tile + 1 < plan.cached_tiles:
            equal = builder.value(f'arith.cmpi eq, {index}, {builder.constant(tile)} : i32')
            builder.lines.append(f'    cf.cond_br {equal}, ^cached_input_{tile}({builder.state} : {state_type}), ^cache_next_{tile}({builder.state} : {state_type})')
            builder.lines.append(f'  ^cached_input_{tile}(%cached_input_state_{tile}: {state_type}):')
            builder.state = f'%cached_input_state_{tile}'
        builder.save(slot, builder.restore(f'input_cache_{tile}'))
        builder.lines.append(f'    cf.br ^cached_input_ready({builder.state} : {state_type})')
        if tile + 1 < plan.cached_tiles:
            builder.lines.append(f'  ^cache_next_{tile}(%cache_next_state_{tile}: {state_type}):')
            builder.state = f'%cache_next_state_{tile}'
    builder.lines.append(f'  ^cached_input_ready(%cached_input_ready_state: {state_type}):')
    builder.state = '%cached_input_ready_state'
