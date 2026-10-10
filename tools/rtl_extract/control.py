"""Exact finite-width execution of control cones sliced at register boundaries."""
from functools import cache, reduce
import json
import operator


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(text):
    return int(text == "true") if text in ("true", "false") else int(text.split(" : ")[0])


def width(typ):
    require(typ.startswith("i") and typ[1:].isdigit(), f"Unsupported integer type: {typ}")
    return int(typ[1:])


class ControlCircuit:
    """Slice hierarchy at register boundaries, then execute exact finite-width control.

    Only requested outputs and their transitive state updates are included. A
    dependency on an unapproved input or unsupported operation fails construction.
    Unreset control must be flushed by the caller before age zero. ``probes`` expose
    values inside the hierarchy as extra outputs and ``cuts`` replace values with
    named inputs; both take selectors ``{"instance": "a/b", "result": name}`` or
    ``{"value": name, "path": "a/b"}`` (matched on ``name``, then ``sv.namehint``).
    ``unreset`` lists registers allowed to stay unknown after reset/flush as ``{"path": "a/b", "name": reg}``
    (path ``""`` is the top); each must name exactly one register of the circuit.
    """
    allowed = {"hw.constant", "hw.wire", "seq.firreg", "comb.mux", "comb.and",
               "comb.or", "comb.xor", "comb.add", "comb.sub", "comb.icmp",
               "comb.extract", "comb.concat", "comb.shru", "comb.shl",
               "comb.replicate", "hw.array_create", "hw.array_get"}

    def __init__(self, document, top, outputs, inputs, cuts=None, probes=None, clock="clock", reset="reset", unreset=()):
        self.modules = {m["name"]: m for m in document["modules"]}
        require(top in self.modules, f"Missing module: {top}")
        self.clock, self.reset = clock, reset
        self.contexts, self.nodes, self.registers, self.types = {}, {}, {}, {}
        self.context("", top, {})
        ports = self.contexts[""][2]
        require(clock in ports, f"Missing clock port: {clock}")
        self.cuts = {self.resolve(selector): name for name, selector in (cuts or {}).items()}
        self.inputs = {p["value"]: p["name"] for p in self.modules[top]["ports"] if p["direction"] == "input"}
        self.allowed_inputs = set(inputs) | set(cuts or {})
        self.outputs = {}
        for name in outputs:
            require(name in ports and ports[name]["direction"] == "output", f"Missing output port: {name}")
            self.outputs[name] = self.visit(("", ports[name]["value"]))
        for name, selector in (probes or {}).items():
            self.outputs[name] = self.visit(self.resolve(selector))
        self.unreset = {self.register(selector) for selector in unreset}
        self.state = {node: None for node in self.registers}

    def register(self, selector):
        require(isinstance(selector, dict) and set(selector) == {"path", "name"}, f"Unsupported register selector: {selector}")
        path = "".join("/" + n for n in selector["path"].split("/") if n)
        found = [k for k in self.registers if k[0] == path and selector["name"] in (self.nodes[k][2].get("name"), self.nodes[k][2].get("sv.namehint"))]
        require(len(found) == 1, f"Register selector {selector} matches {len(found)} registers")
        return found[0]

    def context(self, path, name, bindings):
        require(name in self.modules, f"Missing module: {name}")
        module = self.modules[name]
        definitions = {v: op for op in module["operations"] for v in op["results"]}
        ports = {p["name"]: p for p in module["ports"]}
        self.contexts[path] = (definitions, bindings, ports)
        self.types.update(((path, a["id"]), a["type"]) for a in module["arguments"])
        self.types.update(((path, v), t) for op in module["operations"] for v, t in zip(op["results"], op["result_types"]))

    def enter(self, path, op):
        attrs = op["attributes"]
        child = path + "/" + attrs["instanceName"]
        if child not in self.contexts:
            module = self.modules.get(attrs["moduleName"].lstrip("@"))
            require(module is not None, f"Missing module: {attrs['moduleName'].lstrip('@')}")
            child_ports = {p["name"]: p for p in module["ports"]}
            args = json.loads(attrs["argNames"])
            self.context(child, module["name"], {child_ports[n]["value"]: (path, v) for n, v in zip(args, op["operands"])})
        return child

    def instance(self, path, name):
        definitions = self.contexts[path][0]
        found = {id(op): op for op in definitions.values()
                 if op["kind"] == "hw.instance" and op["attributes"]["instanceName"] == name}
        require(len(found) == 1, f"Expected one instance {name!r} under {path or '/'}")
        return next(iter(found.values()))

    def resolve(self, selector):
        require(isinstance(selector, dict) and set(selector) in ({"instance", "result"}, {"value"}, {"value", "path"}),
                f"Unsupported selector: {selector}")
        names = [n for n in (selector.get("instance") or selector.get("path") or "").split("/") if n]
        path = ""
        last = names.pop() if "instance" in selector else None
        for name in names:
            path = self.enter(path, self.instance(path, name))
        if last is not None:
            op = self.instance(path, last)
            results = json.loads(op["attributes"]["resultNames"])
            require(selector["result"] in results, f"Missing instance result: {last}.{selector['result']}")
            return path, op["results"][results.index(selector["result"])]
        definitions = self.contexts[path][0]
        for attr in ("name", "sv.namehint"):
            found = sorted({v for v, op in definitions.items() if len(op["results"]) == 1 and op["attributes"].get(attr) == selector["value"]})
            require(len(found) <= 1, f"Ambiguous value name: {selector['value']}")
            if found:
                return path, found[0]
        raise ValueError(f"Missing named value: {selector['value']}")

    def clock_root(self, key):
        path, value = key
        definitions, bindings, _ = self.contexts[path]
        if value in bindings:
            return self.clock_root(bindings[value])
        if value in definitions and definitions[value]["kind"] == "hw.wire":
            return self.clock_root((path, definitions[value]["operands"][0]))
        return key

    def visit(self, key):
        if key in self.nodes:
            return key
        path, value = key
        if key in self.cuts:
            self.nodes[key] = ("input", (), self.cuts[key])
            return key
        definitions, bindings, ports = self.contexts[path]
        if value in bindings:
            return self.visit(bindings[value])
        if not path and value in self.inputs:
            name = self.inputs[value]
            require(name in self.allowed_inputs, f"Timing depends on unapproved input: {name}")
            self.nodes[key] = ("input", (), name)
            return key
        require(value in definitions, f"Unresolved value: {key}")
        op = definitions[value]
        if op["kind"] == "hw.instance":
            child = self.enter(path, op)
            result_name = json.loads(op["attributes"]["resultNames"])[op["results"].index(value)]
            return self.visit((child, self.contexts[child][2][result_name]["value"]))
        kind = op["kind"]
        require(kind in self.allowed and not op.get("has_regions"), f"Unsupported timing operation: {kind}")
        require(len(op["results"]) == 1, "Multiple operation results are unsupported")
        self.nodes[key] = (kind, (), op["attributes"])
        if kind == "seq.firreg":
            require(len(op["operands"]) in (2, 4), "Enabled register is unsupported")
            require(self.clock in ports and op["operands"][1] == ports[self.clock]["value"] and
                    self.clock_root((path, op["operands"][1])) == ("", self.contexts[""][2][self.clock]["value"]),
                    "Changed register clock")
            require(set(op["attributes"]) <= {"firrtl.random_init_start", "inner_sym", "name", "sv.namehint"},
                    "Unsupported register attribute")
            if len(op["operands"]) == 4:
                require(self.reset in ports and op["operands"][2] == ports[self.reset]["value"], "Changed register reset")
            args = tuple(self.visit((path, v)) for i, v in enumerate(op["operands"]) if i != 1)
            self.registers[key] = args
        else:
            args = tuple(self.visit((path, v)) for v in op["operands"])
        self.nodes[key] = (kind, args, op["attributes"])
        return key

    def cone(self, names):
        """Registers that can influence the named outputs, including state feedback."""
        seen, found, stack = set(), set(), [self.outputs[n] for n in names]
        while stack:
            key = stack.pop()
            if key in seen:
                continue
            seen.add(key)
            if key in self.registers:
                found.add(key)
            stack.extend(self.nodes[key][1])
        return found

    def cycle(self, inputs):
        @cache
        def get(key):
            kind, operands, attrs = self.nodes[key]
            if kind == "input":
                require(attrs in inputs, f"Missing input {attrs}")
                return inputs[attrs]
            if kind == "seq.firreg":
                return self.state[key]
            if kind == "comb.mux":
                select = get(operands[0])
                if select is None:
                    yes, no = get(operands[1]), get(operands[2])
                    return yes if yes == no else None
                return get(operands[1 if select else 2])
            args = [get(v) for v in operands]
            if kind == "hw.array_create":
                return tuple(reversed(args))
            if kind == "hw.array_get":
                if args[0] is None or args[1] is None:
                    return None
                require(0 <= args[1] < len(args[0]), "Out-of-range array access")
                return args[0][args[1]]
            n = width(self.types[key])
            mask = (1 << n) - 1
            if any(arg is None for arg in args):
                if kind == "comb.and" and 0 in args:
                    return 0
                if kind == "comb.or" and mask in args:
                    return mask
                return None
            if kind == "hw.constant":
                result = integer(attrs["value"])
            elif kind == "hw.wire":
                result = args[0]
            elif kind == "comb.extract":
                result = args[0] >> integer(attrs["lowBit"])
            elif kind == "comb.concat":
                result = 0
                for operand, arg in zip(operands, args):
                    result = (result << width(self.types[operand])) | arg
            elif kind == "comb.replicate":
                step = width(self.types[operands[0]])
                require(n % step == 0, "Non-integral replicate")
                result = sum(args[0] << i for i in range(0, n, step))
            elif kind == "comb.icmp":
                pred = integer(attrs["predicate"])
                require(0 <= pred <= 9, "Unsupported comparison predicate")
                if 2 <= pred <= 5:
                    w = width(self.types[operands[0]])
                    args = [v - (1 << w) if v >> (w - 1) else v for v in args]
                index = pred if pred <= 5 else pred - 4
                result = int((operator.eq, operator.ne, operator.lt, operator.le, operator.gt, operator.ge)[index](*args))
            else:
                operations = {"comb.and": operator.and_, "comb.or": operator.or_, "comb.xor": operator.xor,
                              "comb.add": operator.add, "comb.sub": operator.sub,
                              "comb.shru": operator.rshift, "comb.shl": operator.lshift}
                require(kind in operations, f"Unsupported operation {kind}")
                result = reduce(operations[kind], args)
            return result & mask
        outputs = {name: get(key) for name, key in self.outputs.items()}
        self.state = {key: get(args[2] if len(args) == 3 and get(args[1]) else args[0]) for key, args in self.registers.items()}
        return outputs
