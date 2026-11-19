"""Keep tensor- and batch-invariant exponent pages in their existing VMEM slots."""


def enabled():
    from atlas_mlir_native.options import current_codegen_options
    return current_codegen_options().exponent_page_cache

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
        from .exponent_restoration import Emitter, STATE, load_pages

        e = Emitter(prefix)
        e.state = input_state
        load_pages(e, addresses, [index for index, selected in enumerate(self.scopes) if selected != scope])
        e.emit(f'cf.br ^{prefix}_ready(%{e.state} : {STATE})')
        e.lines.append(f'  ^{prefix}_ready(%{output_state}: {STATE}):')
        return e.lines
