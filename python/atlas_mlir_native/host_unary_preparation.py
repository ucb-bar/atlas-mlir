"""Select existing host implementations around explicitly selected unary math.

This is a placement proof over the source DAG, with no arithmetic or type
rewrite. Unsupported ancestors become boundaries. Descendants stop when they
mix another runtime value into the unary preparation.
"""

POLICY = "selected-unary-math-host-preparation-v1"


def select(source, roots, host_kind, *, excluded=()):
    """Return computation IDs and checked host kinds in the preparation closure.

    ``host_kind`` checks the existing emitter for one unchanged source slice.
    Forward edges may reuse preparation ancestors and exact constant-derived
    source values, but may not introduce unrelated runtime boundary values or
    computations. Explicit device schedules are supplied as exclusions.
    """
    excluded = set(excluded)
    selected = {}
    kinds = {}

    def kind(key):
        if key not in kinds:
            node = source.node(key)
            kinds[key] = (host_kind(node) if node.effect == "pure" and key not in excluded else None)
        return kinds[key]

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
