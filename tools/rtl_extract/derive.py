"""Spec inputs derived from the IR: bundle ports, memory response latency and decoded command codes.

``resolve(document, spec)`` returns a concrete spec (every input, field, latency and code known) plus
the derivation notes reported in record assumptions. Every derivation fails closed.
"""
import copy
import json

from .control import ControlCircuit, integer, require

LIMIT = 64


def modules(spec):
    """Extra modules a spec needs in the document: the system instantiating engine, memories and decoder."""
    return [spec["system"]] if spec.get("system") else []


def table(document):
    return {m["name"]: m for m in document["modules"]}


def ports(document, name):
    found = table(document).get(name)
    require(found is not None, f"Missing module: {name}")
    return {p["name"]: p for p in found["ports"]}


def instances(document, top, name):
    """Instance paths under ``top`` (``a/b``) whose module is ``name``."""
    mods, found, stack = table(document), [], [("", top)]
    while stack:
        path, current = stack.pop()
        for op in mods[current]["operations"]:
            if op["kind"] == "hw.instance":
                child, path_ = op["attributes"]["moduleName"].lstrip("@"), f"{path}/{op['attributes']['instanceName']}".lstrip("/")
                if child == name:
                    found.append(path_)
                elif child in mods:
                    stack.append((path_, child))
    return found


def unique(document, top, name):
    require(top in table(document), f"Missing module: {top}")
    found = instances(document, top, name)
    require(len(found) == 1, f"{top} instantiates {name} {len(found)} times; expected exactly once")
    return found[0]


def locate(document, top, path):
    """``(parent module, parent path, instance op)`` of the instance at ``path`` under ``top``."""
    mods, module, parent = table(document), table(document)[top], ""
    names = path.split("/")
    for i, name in enumerate(names):
        ops = [op for op in module["operations"] if op["kind"] == "hw.instance" and op["attributes"]["instanceName"] == name]
        require(len(ops) == 1, f"Expected one instance {name!r} in {module['name']}")
        if i == len(names) - 1:
            return module, parent, ops[0]
        parent = f"{parent}/{name}".lstrip("/")
        module = mods[ops[0]["attributes"]["moduleName"].lstrip("@")]


def pins(op):
    attrs = op["attributes"]
    return dict(zip(json.loads(attrs["argNames"]), op["operands"])), dict(zip(json.loads(attrs["resultNames"]), op["results"]))


def driver(document, top, path, port):
    """Selector (and value id) of the value driving input ``port`` of the instance at ``path``."""
    module, parent, op = locate(document, top, path)
    value = pins(op)[0].get(port)
    require(value is not None, f"Missing instance input: {path}.{port}")
    source = next((d for d in module["operations"] if value in d["results"]), None)
    require(source is not None, f"{path}.{port} is driven by a port of {module['name']}; no nameable driver")
    if source["kind"] == "hw.instance":
        name = json.loads(source["attributes"]["resultNames"])[source["results"].index(value)]
        return {"instance": f"{parent}/{source['attributes']['instanceName']}".lstrip("/"), "result": name}, value
    name = source["attributes"].get("name") or source["attributes"].get("sv.namehint")
    require(name, f"{path}.{port} is driven by an unnamed {source['kind']}")
    return {"path": parent, "value": name}, value


def prefix(selector, path):
    if "instance" in selector:
        return {"instance": f"{path}/{selector['instance']}", "result": selector["result"]}
    return {"path": f"{path}/{selector.get('path', '')}".strip("/"), "value": selector["value"]}


def control_field(document, spec, owner, port):
    """True when ``port`` depends only on approved inputs through supported operations (not a payload)."""
    approved = {spec["clock"], spec["reset"], *spec["inputs"], *spec.get("links", {})}
    cuts = spec["cuts"] if owner == spec["module"] else spec.get("partner", {}).get("cuts", {})
    try:
        ControlCircuit(document, owner, [port], approved, cuts, None, spec["clock"], spec["reset"])
        return True
    except (ValueError, RecursionError):
        return False


def expand_ports(document, spec):
    """Inputs of the ``command`` bundle and valid/fields of event ``bundle``s, read from the module ports."""
    if spec.get("command"):
        bundle, found = spec["command"], ports(document, spec["module"])
        names = [n for n, p in found.items() if p["direction"] == "input" and (n == f"{bundle}_valid" or n.startswith(f"{bundle}_bits_"))]
        require(f"{bundle}_valid" in names, f"Missing command port {bundle}_valid on {spec['module']}")
        spec["inputs"] = {**{n: 0 for n in names}, **spec["inputs"]}
    owners = [spec["module"]] + ([spec["partner"]["module"]] if "partner" in spec else [])
    for group, event in spec["events"].items():
        if "bundle" not in event:
            continue
        bundle = event["bundle"]
        found = [m for m in owners if ports(document, m).get(f"{bundle}_valid", {}).get("direction") == "output"]
        require(len(found) == 1, f"events.{group}: {bundle}_valid must be an output of exactly one module")
        owned = ports(document, found[0])
        if owned.get(f"{bundle}_ready", {}).get("direction") == "input":
            spec["inputs"].setdefault(f"{bundle}_ready", 0)
        if "fields" not in event:
            event["fields"] = {n[len(bundle) + 6:]: n for n, p in owned.items() if p["direction"] == "output"
                               and n.startswith(f"{bundle}_bits_") and control_field(document, spec, found[0], n)}
        require(all(event.get(k) in (None, *event["fields"]) for k in ("row", "split_by")), f"events.{group} names an unknown field")
    for name, op in spec["operations"].items():
        for command in op["commands"].values():
            require(set(command) <= set(spec["inputs"]), f"operations.{name} drives undeclared inputs {sorted(set(command) - set(spec['inputs']))}")


def measure(document, memory, request, response, clock, reset, reset_cycles, flush_cycles):
    """Ages from a one-cycle ``request`` pulse to ``response`` valid, all other memory inputs idle at 0."""
    found = ports(document, memory)
    inputs = {n for n, p in found.items() if p["direction"] == "input"}
    circuit = ControlCircuit(document, memory, [response], inputs, None, None, clock, reset)
    idle = {n: 0 for n in inputs}
    for i in range(reset_cycles + flush_cycles):
        circuit.cycle({**idle, reset: int(i < reset_cycles)})
    for age in range(LIMIT):
        out = circuit.cycle({**idle, reset: 0, request: int(age == 0)})[response]
        require(out is not None, f"{memory}.{response} is unknown at age {age}")
        if out:
            require(age >= 1, f"{memory}.{response} answers combinationally")
            return age
    raise ValueError(f"{memory}.{response} did not answer {request} within {LIMIT} ages")


def read_latency(document, memory):
    """The ``readLatency`` attribute shared by every ``seq.firmem`` in the memory hierarchy (None without one)."""
    mods, seen, stack, found = table(document), set(), [memory], set()
    while stack:
        name = stack.pop()
        if name in seen or name not in mods:
            continue
        seen.add(name)
        for op in mods[name]["operations"]:
            if op["kind"] == "seq.firmem":
                found.add(integer(op["attributes"]["readLatency"]))
            elif op["kind"] == "hw.instance":
                stack.append(op["attributes"]["moduleName"].lstrip("@"))
    require(len(found) <= 1, f"{memory} mixes memory read latencies {sorted(found)}")
    return next(iter(found), None)


def response(document, spec, group, cache):
    """Latency observed by ``group``: the memory port wired to its valid and response input, simulated."""
    event, owners = spec["events"][group], [spec["module"]] + ([spec["partner"]["module"]] if "partner" in spec else [])
    owner = [m for m in owners if ports(document, m).get(event["valid"], {}).get("direction") == "output"]
    require(len(owner) == 1, f"events.{group}: {event['valid']} must be an output of exactly one module")
    system, memory = spec["system"], event["response"]["memory"]
    engine, store = unique(document, system, owner[0]), unique(document, system, memory)
    _, engine_parent, engine_op = locate(document, system, engine)
    _, store_parent, store_op = locate(document, system, store)
    require(engine_parent == store_parent, f"{engine} and {store} are not siblings; the response path is not direct")
    engine_in, engine_out = pins(engine_op)
    store_in, store_out = pins(store_op)
    requests = [n for n, v in store_in.items() if v == engine_out[event["valid"]]]
    responses = [n for n, v in store_out.items() if v == engine_in.get(event["response"]["input"])]
    require(len(requests) == 1, f"events.{group}: {event['valid']} drives {len(requests)} {memory} inputs; expected one")
    require(len(responses) == 1, f"events.{group}: {event['response']['input']} is driven by {len(responses)} {memory} outputs; expected one")
    key = (memory, requests[0], responses[0])
    if key not in cache:
        measured = measure(document, memory, requests[0], responses[0], spec["clock"], spec["reset"],
                           spec["reset_cycles"], spec["flush_cycles"])
        attribute = read_latency(document, memory)
        require(attribute is None or measured >= attribute, f"{memory} response valid ({measured}) precedes its read data ({attribute})")
        cache[key] = {"memory": f"{system}/{store}", "request": requests[0], "response": responses[0],
                      "measured": measured, "memory_read_latency": attribute}
    return cache[key]


def decode(document, spec, cache):
    """Codes the system's decoder delivers to the engine input ``decode.input`` for each declared instruction word."""
    key = ("decode", spec["module"], json.dumps(spec["decode"], sort_keys=True))
    if key in cache:
        return cache[key]
    rule, system = spec["decode"], spec["system"]
    engine = unique(document, system, spec["module"])
    source, source_value = driver(document, system, rule["instruction"]["instance"], rule["instruction"]["port"])
    if rule["input"] in spec["cuts"]:
        target = prefix(spec["cuts"][rule["input"]], engine)
    else:
        require(rule["input"] in ports(document, spec["module"]), f"decode.input {rule['input']} is neither a cut nor a port")
        target = driver(document, system, engine, rule["input"])[0]
    circuit = ControlCircuit(document, system, [], {spec["clock"], spec["reset"]}, {"instruction": source},
                             {"code": target}, spec["clock"], spec["reset"])
    require(circuit.resolve(source)[1] == source_value, f"Selector {source} does not name the decoder input")
    codes = {}
    for name, word in rule["words"].items():
        codes[name] = circuit.cycle({"instruction": word})["code"]
        require(codes[name] is not None, f"decode: the code for {name} depends on state or unknown values")
    clash = {c for c in codes.values() if list(codes.values()).count(c) > 1}
    require(not clash, f"decode: instructions {sorted(n for n, c in codes.items() if c in clash)} decode to the same code")
    idle = circuit.cycle({"instruction": 0})["code"]
    require(idle not in codes.values(), f"decode: {sorted(n for n, c in codes.items() if c == idle)} decode like the all-zero word")
    cache[key] = codes
    return codes


def substitute(value, codes):
    if isinstance(value, dict):
        return {k: substitute(v, codes) for k, v in value.items()}
    if isinstance(value, str) and value.startswith("$"):
        require(value[1:] in codes, f"Unknown command code {value}")
        return codes[value[1:]]
    return value


def resolve(document, spec, cache=None):
    """``(spec, notes)``: the spec with derived ports, latencies and codes; notes go into record assumptions."""
    cache = {} if cache is None else cache
    spec = copy.deepcopy(spec)
    expand_ports(document, spec)
    notes = {}
    for group, event in spec["events"].items():
        if "memory" in event.get("response", {}):
            derived = response(document, spec, group, cache)
            if event["response"].get("latency") is None:
                event["response"]["latency"] = derived["measured"]
            notes[group] = {**derived, "used": event["response"]["latency"]}
    if "decode" in spec:
        spec["operations"] = substitute(spec["operations"], decode(document, spec, cache))
    else:
        spec["operations"] = substitute(spec["operations"], {})
    return spec, ({"scratchpad_read_derivation": notes} if notes else {})
