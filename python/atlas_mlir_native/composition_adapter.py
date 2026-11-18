"""Expose proved adjacent device groups to Merlin's region selector."""
from __future__ import annotations

from dataclasses import replace

from merlin.compile.application_plan import select_application_plan
from merlin.compile.semantic_regions import extract_semantic_region

# Recognizers retain ownership of their source/domain proofs. The wrapper owns
# their state and dispatch, so ordinary AtlasApplicationAdapter callers retain
# the existing smaller regions unless they explicitly opt into enlargement.
_ADJACENT_PATTERNS = (
    ("adjacent_pointwise", "_adjacent_groups"),
    ("adjacent_square_reduce", "_square_groups"),
    ("adjacent_reduction_scale", "_reduction_scale_groups"),
    ("adjacent_pointwise_chain", "_pointwise_chain_groups"),
    ("quantized_epilogue", "_quantized_epilogue_groups"),
    ("quantized_epilogue_narrowing", "_quantized_epilogue_narrowing_groups"),
    ("quantized_narrowing", "_quantized_narrowing_groups"),
    ("pointwise_narrowing", "_pointwise_narrowing_groups"),
    ("softmax_narrowing", "_softmax_narrowing_groups"),
    ("softmax_input_widening", "_softmax_input_widening_groups"),
    ("pointwise_input_widening", "_pointwise_input_widening_groups"),
    ("expanded_silu", "_expanded_silu_groups"),
    ("quantized_operand_permutation", "_quantized_operand_permutation_groups"),
    ("quantizer_input_permutation", "_quantizer_input_permutation_groups"),
    ("float_input_layout", "_float_input_layout_groups"),
)


def kind_of(row):
    return dict(row.metadata).get('kind')


def contiguous(row):
    return row.nodes == tuple(range(min(row.nodes), max(row.nodes) + 1))


def device_casts(choices):
    """Single-node exact-float device choices by node index."""
    return {row.nodes[0]: row for row in choices if len(row.nodes) == 1
            and row.placement == 'accelerator' and kind_of(row) == 'exact_float_device'}


def device_quantizers(adapter, choices):
    """Static activation quantizer choices selected for the device, by first node index."""
    return {min(row.nodes): row for row in choices if kind_of(row) == 'activation_quantize_device'
            or kind_of(row) == 'static_quantize' and adapter.activation_quantizer_route == 'device'}


def neighbor_pairs(choices, neighbors, accepts, *, before=False):
    """Pair each accepted device choice with the neighbor at its preceding or following node."""
    for row in choices:
        if row.placement == 'accelerator' and accepts(row):
            other = neighbors.get(min(row.nodes) - 1 if before else max(row.nodes) + 1)
            if other is not None:
                yield (other, row) if before else (row, other)


def device_runs(choices, width):
    """Windows of single-node accelerator choices over consecutive nodes."""
    singles = sorted((row for row in choices if len(row.nodes) == 1), key=lambda row: row.nodes)
    for start in range(len(singles) - width + 1):
        rows = singles[start:start + width]
        if all(row.placement == 'accelerator' and row.nodes[0] == rows[0].nodes[0] + index
               for index, row in enumerate(rows)):
            yield rows


def merged_choice(base, rows, region, kind, cycles, *, inputs=None, prefix=None, **fields):
    """Widen `base` over `rows`, taking the region's boundary ports from their buffers."""
    from merlin.xdsl_dialects.lowering.global_plan import CycleInterval

    ports = {port.buffer: port for row in rows for port in (*row.inputs, *row.outputs)}
    if inputs is None:
        inputs = tuple(node.id for node in region.nodes if node.effect in ('input', 'constant'))
    return replace(base, id=(prefix or kind) + '_' + base.id, nodes=tuple(node for row in rows for node in row.nodes),
        implementation='source_' + kind, inputs=tuple(ports[key] for key in inputs),
        outputs=tuple(ports[key] for key in region.outputs), cycles=CycleInterval.unknown(cycles),
        metadata=(('kind', kind),), **fields)


def fuse(adapter, source, choices, state, kind, candidates, *, prepare, cycles, base=0, inputs=None,
         errors=ValueError, admit=None, **fields):
    """Append one `kind` choice per candidate row group that `prepare(region, rows)` accepts.

    `prepare` declines by raising `errors`. `admit(group)` runs outside that
    guard, so its failures stay visible; it returns extra choice fields, or None
    to decline. Groups are kept on the adapter by region digest.
    """
    result, groups = list(choices), {}
    setattr(adapter, state, groups)
    pure = [node for node in source.nodes if node.effect == 'pure']
    for rows in candidates:
        try:
            region = extract_semantic_region(source, tuple(pure[index].id for row in rows for index in row.nodes))
            group = prepare(region, rows)
        except errors:
            continue
        extra = {} if admit is None else admit(group)
        if extra is None:
            continue
        groups[region.digest()] = group
        result.append(merged_choice(rows[base], rows, region, kind, cycles,
                                    inputs=None if inputs is None else inputs(group), **fields, **extra))
    return tuple(result)


def fold_operand_views(adapter, state, kind, candidates, choices, *, accepts, prepare, cycles,
                       errors=ValueError, admit=None):
    """Append one `kind` choice per private operand view that its device consumer absorbs."""
    from merlin.xdsl_dialects.lowering.global_plan import CycleInterval

    result, groups = list(choices), {}
    setattr(adapter, state, groups)
    for candidate in candidates:
        consumer = candidate.consumer
        if consumer.placement != 'accelerator' or not accepts(kind_of(consumer)):
            continue
        try:
            group = prepare(candidate.region, kind_of(consumer))
        except errors:
            continue
        extra = {} if admit is None else admit(group)
        if extra is None:
            continue
        groups[group.source.digest()] = group
        nodes = tuple(sorted((*candidate.producer.nodes, *consumer.nodes)))
        result.append(replace(consumer, id=kind + '_' + '_'.join(map(str, nodes)), nodes=nodes,
            implementation='source_' + kind, inputs=candidate.inputs, outputs=candidate.outputs,
            cycles=CycleInterval.unknown(cycles), metadata=(('kind', kind),), **extra))
    return tuple(result)


def source_names(name):
    """C symbols of one exported source-dispatch region."""
    return dict(context_type=name + '_context', source_context_type=name + '_source_context',
                binding_function=name + '_bind', source_binding_function=name + '_source_bind')


def c_region(name, inputs, outputs, files, programs, description):
    from merlin.compile.semantic_application import CompiledCRegion
    from .c_runtime import RUNTIME_SHARED_FILES

    return CompiledCRegion(name + '_source_dispatch', name + '.h', inputs, outputs, files,
                           RUNTIME_SHARED_FILES, programs, description)


def write_source_request(destination, source, indent=None):
    """Create `destination` and bind the exact source request that its compilation used."""
    import json
    from pathlib import Path

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    (destination / 'source-request.json').write_text(json.dumps(source.record(), sort_keys=True, indent=indent) + '\n')
    return destination


def check_source_request(compiled, source, message, entries=None):
    """Refuse a compiled directory whose bound source request differs from `source`."""
    import json
    from pathlib import Path

    compiled = Path(compiled)
    if (compiled.is_symlink() or entries is not None and {path.name for path in compiled.iterdir()} != entries
            or (compiled / 'source-request.json').is_symlink()
            or json.loads((compiled / 'source-request.json').read_text()) != source.record()):
        raise ValueError(message)
    return compiled


class AdjacentDeviceAdapter:
    """Opt into proved adjacent device groups while retaining smaller choices."""

    def __init__(self, base, *, patterns=None):
        self.base = base
        if patterns is None:
            self.patterns = _ADJACENT_PATTERNS
        else:
            names = tuple(patterns)
            supported = dict(_ADJACENT_PATTERNS)
            if len(set(names)) != len(names) or any(name not in supported for name in names):
                raise ValueError("unknown or duplicate adjacent device pattern")
            self.patterns = tuple((name, supported[name]) for name in names)
        for _, state in self.patterns:
            setattr(self, state, {})

    def __getattr__(self, name):
        return getattr(self.base, name)

    def application_region_alternatives(self, source):
        from importlib import import_module

        choices = tuple(self.base.plan(source).selected)
        for module, _ in self.patterns:
            choices = import_module("." + module, __package__).alternatives(self, source, choices)
        return choices

    def plan(self, source):
        return select_application_plan(source, self.application_region_alternatives(source))

    def _group(self, region, alternative):
        from importlib import import_module

        kind = dict(alternative.metadata)["kind"]
        for module, state in self.patterns:
            if module == kind:
                return import_module("." + module, __package__), getattr(self, state)[region.digest()]
        return None

    def compile_region(self, snapshot, region, alternative, name, segments, target_source, limits):
        selected = self._group(region, alternative)
        if selected is None:
            return self.base.compile_region(snapshot, region, alternative, name, segments, target_source, limits)
        module, group = selected
        from .binding import validate_io_window
        validate_io_window()
        module.compile_group(group, snapshot, target_source=target_source, destination=segments / name, limits=limits)
        return module.export_group(group, segments / name, name, target_source=target_source)

    def export_region(self, region, alternative, name, segments, target_source):
        selected = self._group(region, alternative)
        if selected is None:
            return self.base.export_region(region, alternative, name, segments, target_source)
        module, group = selected
        return module.export_group(group, segments / name, name, target_source=target_source)
