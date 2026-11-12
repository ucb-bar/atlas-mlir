"""Select existing host implementations around explicitly selected unary math.

This is a placement proof over the source DAG, with no arithmetic or type
rewrite. Unsupported ancestors become boundaries. Descendants stop when they
mix another runtime value into the unary preparation.
"""

POLICY = "selected-unary-math-host-preparation-v1"
SCOPE_POLICY = "integer-inputs-and-bound-constants-v1"


def select(source, roots, host_kind, *, excluded=(), input_boundaries=None):
    """Return computation IDs and checked host kinds in the preparation closure.

    ``host_kind`` checks the existing emitter for one unchanged source slice.
    Forward edges may reuse preparation ancestors and exact constant-derived
    source values, but may not introduce unrelated runtime boundary values or
    computations. Explicit device schedules are supplied as exclusions. With
    ``input_boundaries``, every root must have a fully supported ancestry ending
    at constants or those explicitly admitted inputs; unsupported or excluded
    computations cannot silently become preparation boundaries.
    """
    excluded = set(excluded)
    selected = {}
    kinds = {}

    def kind(key):
        if key not in kinds:
            node = source.node(key)
            kinds[key] = (host_kind(node) if node.effect == "pure" and key not in excluded else None)
        return kinds[key]

    if input_boundaries is not None:
        input_boundaries = set(input_boundaries)
        provenance = {}
        pending = [(key, False) for key in roots]
        while pending:
            key, checked_inputs = pending.pop()
            if key in provenance:
                continue
            node = source.node(key)
            if node.effect == "constant":
                provenance[key] = True
            elif node.effect == "input":
                provenance[key] = key in input_boundaries
            elif not node.inputs or kind(key) is None:
                provenance[key] = False
            elif checked_inputs:
                provenance[key] = all(provenance[value] for value in node.inputs)
            else:
                pending.append((key, True))
                pending.extend((value, False) for value in node.inputs)
        roots = tuple(key for key in roots if provenance[key])

    pending = list(roots)
    while pending:
        key = pending.pop()
        if key in selected:
            continue
        selected_kind = kind(key)
        if selected_kind is not None:
            selected[key] = selected_kind
            pending.extend(source.node(key).inputs)

    # Constant provenance is structural: no tensor data is read or synthesized.
    constants = set()
    for node in source.nodes:
        if node.effect == "constant" or (node.effect == "pure" and node.inputs
                                        and all(key in constants for key in node.inputs)):
            constants.add(node.id)

    def constant_support(key):
        support = {}
        pending = [key]
        while pending:
            key = pending.pop()
            node = source.node(key)
            if key in support or node.effect == "constant":
                continue
            selected_kind = kind(key)
            if key not in constants or selected_kind is None:
                return None
            support[key] = selected_kind
            pending.extend(node.inputs)
        return support

    dependent = set(roots) & selected.keys()
    for node in source.nodes:
        if node.effect != "pure" or node.id in dependent or not any(key in dependent for key in node.inputs):
            continue
        selected_kind = kind(node.id)
        if selected_kind is None:
            continue
        support = {}
        for key in node.inputs:
            if key in selected or source.node(key).effect == "constant":
                continue
            constant = constant_support(key)
            if constant is None:
                break
            support.update(constant)
        else:
            selected.update(support)
            selected[node.id] = selected_kind
            dependent.add(node.id)
    return selected
