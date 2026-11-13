"""Keep tensor- and batch-invariant exponent pages in their existing VMEM slots."""
import os

from merlin.semantic_compiler.target_binding import NativeCompilationError


def enabled():
    value = os.environ.get('ATLAS_EXPONENT_PAGE_CACHE', '0')
    if value not in ('0', '1'):
        raise NativeCompilationError('unsupported_semantics', 'ATLAS_EXPONENT_PAGE_CACHE must be 0 or 1')
    return value == '1'


def recorded_enabled(record):
    if 'exponent_page_cache' not in record:
        return False
    if record['exponent_page_cache'] is not True:
        raise ValueError('recorded exponent page caching must be explicitly enabled')
    return True


def record(plan, *, selected=None):
    from .exponent_restoration import exponent_geometry

    selected = enabled() if selected is None else selected
    if type(selected) is not bool:
        raise ValueError('exponent page cache selection must be boolean')
    if not selected:
        return {}
    scopes = ['tile' if geometry['mode'] == 'row' else
              'invocation' if geometry['pages'] == 1 else 'batch'
              for geometry in exponent_geometry(plan)]
    return dict(exponent_page_cache=True, exponent_page_cache_scopes=scopes)


class ExponentPageCache:
    """Describe immutable page lifetimes without changing factor admission."""

    def __init__(self, plan, layout):
        expected = record(plan, selected=recorded_enabled(layout))
        if any(layout.get(key) != expected.get(key) for key in
               ('exponent_page_cache', 'exponent_page_cache_scopes')):
            raise ValueError('exponent page cache scope differs from its source')
        self.scopes = tuple(expected.get('exponent_page_cache_scopes', ('tile', 'tile')))

    @property
    def resident(self):
        return tuple(index for index, scope in enumerate(self.scopes) if scope != 'tile')

    def prefetch(self, scope, addresses, *, input_state, output_state, prefix):
        from .exponent_restoration import Emitter, BF16, SLOT, STATE

        e = Emitter(prefix)
        e.state = input_state
        for index, (which, address) in enumerate(zip(('lhs', 'rhs'), addresses, strict=True)):
            if self.scopes[index] != scope:
                continue
            if isinstance(address, int):
                address = e.const(address)
            page = e.effect('dma_load_bf16', [address, e.const(2048)], ['i32', 'i32'], BF16)
            e.effect('save', ['restore_' + which, page], [SLOT, BF16])
        e.emit(f'cf.br ^{prefix}_ready(%{e.state} : {STATE})')
        e.lines.append(f'  ^{prefix}_ready(%{output_state}: {STATE}):')
        return e.lines
