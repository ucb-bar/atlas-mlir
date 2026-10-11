"""Exact finite-width execution of control cones sliced at register boundaries."""
from functools import cache, reduce
import json
import operator
import re


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(text):
    return int(text == "true") if text in ("true", "false") else int(text.split(" : ")[0])


def width(typ):
    require(typ.startswith("i") and typ[1:].isdigit(), f"Unsupported integer type: {typ}")
    return int(typ[1:])


FIRES_WHEN = {"sv.error": None, "sv.fatal": None, "sv.assert": 0}  # sv.assert fires when its operand is 0


def assertions(module):
    """Assertions of one exported module (``hw_ir_export --assertions``), one per guard.

    Macros are read as in a simulation build: every ``ifdef`` macro is undefined and every ``sv.macro.ref``
    condition (``ASSERT_VERBOSE_COND_``, ``STOP_COND_``) is 1, so an assertion is ``live`` unless it sits in an
    ``ifdef`` then-branch or a macro else-branch. It fires when every ``(value, level)`` term holds: the enclosing
    ``sv.if`` conditions (1 in then-, 0 in else-branches) and, for ``sv.assert``, its operand at 0. ``clock`` is the
    enclosing single-edge ``sv.always posedge``; anything else is recorded as ``problem``. ``sv.error`` and
    ``sv.fatal`` under the same guard are one assertion.
    """
    ops = {op["id"]: op for op in module["operations"]}
    definitions = {v: op for op in module["operations"] for v in op["results"]}
    found = {}
    for site in module.get("assertions", []):
        op = ops[site["op"]]
        terms, clock, live, problem = [], None, True, None
        if op["kind"] not in FIRES_WHEN:
            problem = f"unsupported assertion {op['kind']}"
        for parent in site["parents"]:
            region, owner = parent["region"], ops[parent["op"]]
            kind = owner["kind"]
            if kind in ("sv.ifdef", "sv.ifdef.procedural"):
                live &= region == 1
            elif kind == "sv.if" and definitions.get(owner["operands"][0], {}).get("kind") == "sv.macro.ref":
                live &= region == 0
            elif kind == "sv.if":
                terms.append((owner["operands"][0], int(region == 0)))
            elif kind == "sv.always" and clock is None and owner["attributes"].get("events") == "[0 : i32]" and len(owner["operands"]) == 1:
                clock = owner["operands"][0]
            else:
                problem = problem or f"assertion inside {kind}"
        if FIRES_WHEN.get(op["kind"]) is not None:
            terms.append((op["operands"][0], FIRES_WHEN[op["kind"]]))
        if clock is None:
            problem = problem or "assertion outside an always posedge block"
        entry = found.setdefault((tuple(terms), clock, live, problem), {
            "id": op["id"], "kinds": [], "message": None, "location": op["location"],
            "live": live, "terms": terms, "clock": clock, "problem": problem})
        entry["kinds"].append(op["kind"])
        text = op["attributes"].get("message") or op["attributes"].get("label")
        entry["message"] = entry["message"] or text
    return list(found.values())


class ControlCircuit:
    """Slice hierarchy at register boundaries, then execute exact finite-width control.

    Only requested outputs and their transitive state updates are included; a dependency on an
    unapproved input or unsupported operation fails construction. ``probes`` expose inner values as
    extra outputs and ``cuts`` replace them with named inputs; selectors are ``{"instance": "a/b",
    "result": name}`` or ``{"value": name, "path": "a/b"}`` (matched on ``name``, then ``sv.namehint``).
    ``unreset`` lists registers allowed to stay unknown after reset/flush as ``{"path": "a/b", "name": reg}``
    (path ``""`` is the top); each must name exactly one register. ``invariants`` (see ``select``) are assertions in
    the hierarchy whose firing conditions become extra outputs named in ``self.invariants``.
    """
    allowed = {"hw.constant", "hw.wire", "seq.firreg", "comb.mux", "comb.and",
               "comb.or", "comb.xor", "comb.add", "comb.sub", "comb.icmp",
               "comb.extract", "comb.concat", "comb.shru", "comb.shl",
               "comb.replicate", "hw.array_create", "hw.array_get"}

    def __init__(self, document, top, outputs, inputs, cuts=None, probes=None, clock="clock", reset="reset", unreset=(),
                 invariants=None):
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
        self.top, self.invariants, self.excluded, self.matched = top, {}, [], set()
        if invariants is not None:
            selected, self.excluded, self.matched = select(self.modules, top, invariants)
            for path, item in selected:
                self.watch(path, item)
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

    def watch(self, path, item):
        """Add the firing condition of assertion ``item`` at instance ``path`` as an output."""
        name = f"{self.top}{path}#{item['id']}"
        where = f"invariant {name} ({first_line(item['message'])})"
        require(item["problem"] is None, f"{where}: {item['problem']}")
        context = ""
        for part in [p for p in path.split("/") if p]:
            context = self.enter(context, self.instance(context, part))
        definitions, _, ports = self.contexts[context]
        clock = definitions.get(item["clock"])
        require(clock is not None and clock["kind"] == "seq.from_clock" and self.clock in ports and
                clock["operands"][0] == ports[self.clock]["value"] and
                self.clock_root((context, clock["operands"][0])) == ("", self.contexts[""][2][self.clock]["value"]),
                f"{where}: not clocked by {self.clock}")
        key = ("fire", context, item["id"])
        try:
            self.nodes[key] = ("fire", tuple(self.visit((context, v)) for v, _ in item["terms"]), tuple(l for _, l in item["terms"]))
        except (ValueError, RecursionError) as error:
            raise ValueError(f"{where}: {error}") from None
        self.outputs[name] = key
        self.invariants[name] = {"path": path, "message": first_line(item["message"]), "location": item["location"]}

    def reach(self, names, stop=()):
        """Registers and input names the named outputs depend on, including state feedback; traversal does not enter
        nodes under the instance paths in ``stop``."""
        seen, registers, inputs, stack = set(), set(), set(), [self.outputs[n] for n in names]
        while stack:
            key = stack.pop()
            if key in seen or any(key[0] == p or key[0].startswith(p + "/") for p in stop):
                continue
            seen.add(key)
            kind, operands, attrs = self.nodes[key]
            if kind == "input":
                inputs.add(attrs)
            if key in self.registers:
                registers.add(key)
            stack.extend(operands)
        return registers, inputs

    def cone(self, names):
        return self.reach(names)[0]

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
            if kind == "fire":
                values = [get(v) for v in operands]
                if any(v is not None and v != level for v, level in zip(values, attrs)):
                    return 0
                return None if None in values else 1
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


def first_line(text):
    return (text or "").split("\n")[0].removeprefix("Assertion failed: ")


def sites(modules, top):
    """``(path, module, assertion)`` for every assertion in the instance hierarchy under ``top``."""
    @cache
    def below(name):
        module = modules[name]
        require("assertions" in module, f"{name} was exported without assertions (hw_ir_export --assertions)")
        found = [("", name, item) for item in assertions(module)]
        for op in module["operations"]:
            child = op["attributes"].get("moduleName", "").lstrip("@") if op["kind"] == "hw.instance" else None
            if child:
                require(child in modules, f"Assertions of {child} are not visible (extern module)")
                found += [(f"/{op['attributes']['instanceName']}{p}", m, a) for p, m, a in below(child)]
        return tuple(found)
    return list(below(top))


def select(modules, top, rules):
    """``(selected, excluded)``: every live assertion under ``top`` except those matched by an ``exclude`` rule
    (``{path?, module?, message?, reason}``; ``path`` is an instance path prefix, ``message`` a regular expression),
    and the indices of the rules that matched. Disabled assertions are excluded with the macro reading as reason."""
    selected, excluded, used = [], [], set()
    for path, module, item in sites(modules, top):
        if not item["live"]:
            excluded.append({"path": path, "message": first_line(item["message"]), "reason": "not compiled in a simulation build"})
            continue
        hits = [i for i, rule in enumerate(rules.get("exclude", [])) if matches(rule, path, module, item)]
        used.update(hits)
        if hits:
            excluded.append({"path": path, "message": first_line(item["message"]), "reason": rules["exclude"][hits[0]]["reason"]})
        else:
            selected.append((path, item))
    return selected, excluded, used


def unmatched(rules, matched):
    unused = [rule for i, rule in enumerate(rules.get("exclude", [])) if i not in matched]
    require(not unused, f"Invariant exclusions match no assertion: {unused}")


def matches(rule, path, module, item):
    prefix = "".join("/" + n for n in rule.get("path", "").split("/") if n)
    return ((path == prefix or path.startswith(prefix + "/") or not prefix) and rule.get("module", module) == module
            and re.search(rule.get("message", ""), first_line(item["message"])) is not None)
