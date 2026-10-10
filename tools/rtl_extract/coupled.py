"""Two control circuits clocked together, with same-cycle wiring between them resolved by fixed-point iteration.

Spec keys (recipe ``coupled``): the primary circuit is ``module`` with the usual ``cuts``/``probes``;
``partner: {module, inputs: [names], cuts?, probes?, unreset?}`` is the second circuit, where ``inputs`` lists
which top-level ``inputs`` (stimulus, responses, partner cut names) belong to it; ``links: {input: output}``
drives an input port of one circuit with the same-cycle value of an output (port or probe) of the other.
Linked inputs start unknown each cycle and are refined until stable, so a combinational loop between the
circuits leaves its values unknown and fails the run instead of picking an order.
"""
from .control import ControlCircuit, require
from .spec import keys

PARTNER_KEYS = ({"module", "inputs"}, {"cuts", "probes", "unreset"})


def ports(document, module, direction):
    found = [m for m in document["modules"] if m["name"] == module]
    require(found, f"Missing module: {module}")
    return {p["name"] for p in found[0]["ports"] if p["direction"] == direction}


def reach(circuit, names):
    """Registers and input names that the named outputs depend on (state feedback included)."""
    seen, registers, inputs, stack = set(), set(), set(), [circuit.outputs[n] for n in names]
    while stack:
        key = stack.pop()
        if key in seen:
            continue
        seen.add(key)
        kind, operands, attrs = circuit.nodes[key]
        if kind == "input":
            inputs.add(attrs)
        if key in circuit.registers:
            registers.add(key)
        stack.extend(operands)
    return registers, inputs


class CoupledCircuit:
    """Same interface as ``ControlCircuit`` (``registers``, ``state``, ``cone``, ``cycle``) over two circuits."""

    def __init__(self, circuits, links):
        self.circuits, self.links = circuits, links
        self.registers = {(i, k): v for i, c in enumerate(circuits) for k, v in c.registers.items()}
        self.unreset = {(i, k) for i, c in enumerate(circuits) for k in c.unreset}

    @property
    def state(self):
        return {(i, k): v for i, c in enumerate(self.circuits) for k, v in c.state.items()}

    @state.setter
    def state(self, value):
        for i, c in enumerate(self.circuits):
            c.state = {k: value[i, k] for k in c.registers}

    def owner(self, name):
        return next(i for i, c in enumerate(self.circuits) if name in c.outputs)

    def cone(self, names):
        """Registers behind ``names``, following links into the circuit that drives each reached linked input."""
        found, pending, done = set(), list(names), set()
        while pending:
            name = pending.pop()
            if name in done:
                continue
            done.add(name)
            index = self.owner(name)
            registers, inputs = reach(self.circuits[index], [name])
            found |= {(index, k) for k in registers}
            pending += [self.links[i] for i in inputs if i in self.links]
        return found

    def cycle(self, inputs):
        values = {name: None for name in self.links}
        for _ in range(len(self.links) + 2):
            saved = [c.state for c in self.circuits]
            outputs = {}
            for circuit in self.circuits:
                outputs.update(circuit.cycle({**inputs, **values}))
            refined = {target: outputs[source] for target, source in self.links.items()}
            if refined == values:
                return outputs
            for circuit, state in zip(self.circuits, saved):
                circuit.state = state
            values = refined
        raise ValueError("Linked inputs did not settle")


def build(document, spec, signals):
    """``signals`` are the observed names; each must be an output port or probe of exactly one circuit."""
    partner, links = spec["partner"], spec["links"]
    keys(partner, *PARTNER_KEYS, "partner")
    require(isinstance(links, dict) and links, "links: expected a non-empty mapping")
    require(set(partner["inputs"]) <= set(spec["inputs"]), "partner.inputs must name top-level inputs")
    require(set(partner.get("cuts", {})) <= set(partner["inputs"]), "partner cut inputs must be listed in partner.inputs")
    modules = [spec["module"], partner["module"]]
    probes = [spec["probes"], partner.get("probes", {})]
    cuts = [spec["cuts"], partner.get("cuts", {})]
    unreset = [spec["unreset"], partner.get("unreset", [])]
    outputs = [ports(document, m, "output") | set(p) for m, p in zip(modules, probes)]
    targets = [ports(document, m, "input") & set(links) for m in modules]
    require(not targets[0] & targets[1], f"Linked inputs exist on both circuits: {sorted(targets[0] & targets[1])}")
    require(targets[0] | targets[1] == set(links), f"Linked inputs are not input ports: {sorted(set(links) - targets[0] - targets[1])}")
    require(not set(links) & set(spec["inputs"]), "Linked inputs must not also be top-level inputs")
    wanted = [[], []]
    for name in dict.fromkeys([*signals, *links.values()]):
        owners = [i for i in (0, 1) if name in outputs[i]]
        require(len(owners) == 1, f"Signal {name!r} must be an output of exactly one circuit")
        wanted[owners[0]].append(name)
    for target, source in links.items():
        require(target in targets[0] if source in wanted[1] else target in targets[1], f"Link {target} <- {source} must cross circuits")
    own = [set(spec["inputs"]) - set(partner["inputs"]), set(partner["inputs"])]
    circuits = []
    for i in (0, 1):
        circuit_probes = {n: s for n, s in probes[i].items() if n in wanted[i]}
        circuit_outputs = [n for n in wanted[i] if n not in circuit_probes]
        approved = {spec["clock"], spec["reset"], *own[i], *targets[i]}
        circuits.append(ControlCircuit(document, modules[i], circuit_outputs, approved, cuts[i], circuit_probes,
                                       spec["clock"], spec["reset"], unreset[i]))
    return CoupledCircuit(circuits, links)
