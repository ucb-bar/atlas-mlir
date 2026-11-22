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


def panel_record(plan, layout):
    """Bound row-page reuse to the explicitly selected inline panel traversal."""
    from .exponent_restoration import exponent_geometry
    from .quantized_panel_reuse import inline_readout_admitted

    rows = layout.get('operand_reuse_panel_m_tiles', 0)
    if (not recorded_enabled(layout) or layout.get('operand_reuse_panel_order') != 'k-major'
            or not inline_readout_admitted(rows, plan.k)):
        return {}
    roles = [role for role, geometry in zip(('lhs', 'rhs'), exponent_geometry(plan), strict=True)
             if geometry['mode'] == 'row']
    return dict(exponent_panel_cache_roles=roles) if roles else {}


class ExponentPageCache:
    """Describe immutable page lifetimes without changing factor admission."""

    def __init__(self, plan, layout):
        expected = record(plan, selected=recorded_enabled(layout))
        if any(layout.get(key) != expected.get(key) for key in
               ('exponent_page_cache', 'exponent_page_cache_scopes')):
            raise ValueError('exponent page cache scope differs from its source')
        self.scopes = tuple(expected.get('exponent_page_cache_scopes', ('tile', 'tile')))
        # Absence retains previously published panel schedules. New publications
        # record the additional lifetimes explicitly, independently of defaults.
        panel = panel_record(plan, layout)
        if ('exponent_panel_cache_roles' in layout
                and ('exponent_panel_cache_roles' not in panel
                     or layout['exponent_panel_cache_roles'] != panel['exponent_panel_cache_roles'])):
            raise ValueError('exponent panel cache roles differ from the selected traversal')
        self.panel_roles = tuple(layout.get('exponent_panel_cache_roles', ()))

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
