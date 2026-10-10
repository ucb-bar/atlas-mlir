"""Load and validate per-engine YAML specs; unknown or missing keys fail closed."""
import copy
from pathlib import Path

import yaml

from .control import require

TARGETS = Path(__file__).resolve().parent / "targets"
REQUIRED = {"engine", "module", "events"}
OPTIONAL = {"description", "recipe", "clock", "reset", "reset_cycles", "flush_cycles", "limit", "tail", "inputs",
            "busy", "probes", "cuts", "operations", "operation_table", "variants", "checks", "expect", "unreset",
            "system", "command", "decode"}
DEFAULTS = {"recipe": "single", "clock": "clock", "reset": "reset", "reset_cycles": 16, "flush_cycles": 16,
            "limit": 320, "tail": 2, "busy": None, "probes": {}, "cuts": {}, "operations": {}, "inputs": {},
            "variants": {}, "checks": [], "expect": {}, "unreset": []}
RECIPE_KEYS = {"single": set(), "coupled": {"partner", "links"}}
EVENT_KEYS = (set(), {"valid", "bundle", "fields", "row", "split_by", "response"})
OPERATION_KEYS = ({"commands"}, {"description", "next_issue"})
VARIANT_FIXED = {"engine", "module", "variants", "checks", "expect", "operation_table", "system", "command", "decode"}


def keys(value, required, optional, where):
    require(isinstance(value, dict), f"{where}: expected a mapping")
    require(not required - set(value), f"{where}: missing keys {sorted(required - set(value))}")
    require(not set(value) - required - optional, f"{where}: unknown keys {sorted(set(value) - required - optional)}")


def ages(key):
    if isinstance(key, int):
        return [key]
    require(isinstance(key, str) and ".." in key, f"Unsupported command age: {key!r}")
    first, last = (int(v) for v in key.split(".."))
    require(0 <= first <= last, f"Unsupported command age range: {key!r}")
    return list(range(first, last + 1))


def substitute(value, code):
    if isinstance(value, dict):
        return {k: substitute(v, code) for k, v in value.items()}
    return code if value == "$code" else value


def symbols(value):
    if isinstance(value, dict):
        return {s for v in value.values() for s in symbols(v)}
    return {value[1:]} if isinstance(value, str) and value.startswith("$") else set()


def merge(base, overlay):
    result = copy.deepcopy(base)
    for key, value in overlay.items():
        result[key] = merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else copy.deepcopy(value)
    return result


def validate(spec, where):
    recipe = spec.get("recipe", "single")
    require(recipe in RECIPE_KEYS, f"{where}: unknown recipe {recipe!r}")
    keys(spec, REQUIRED, OPTIONAL | RECIPE_KEYS[recipe] | {"only"}, where)
    spec = copy.deepcopy({**DEFAULTS, **spec})
    require(all(isinstance(v, int) for v in spec["inputs"].values()), f"{where}: input idle values must be integers")
    words = {}
    if "decode" in spec:
        keys(spec["decode"], {"input", "instruction", "words"}, set(), f"{where}: decode")
        keys(spec["decode"]["instruction"], {"instance", "port"}, set(), f"{where}: decode.instruction")
        words = spec["decode"]["words"]
        require(isinstance(words, dict) and words and all(isinstance(w, int) for w in words.values()),
                f"{where}: decode.words must map names to instruction words")
    require(spec.get("system") or not ("decode" in spec or any("memory" in e.get("response", {}) for e in spec["events"].values())),
            f"{where}: decode and response.memory need a system module")
    prefix = f"{spec['command']}_" if spec.get("command") else None
    require(isinstance(spec["unreset"], list), f"{where}: unreset must be a list of {{path, name}} selectors")
    for selector in spec["unreset"]:
        keys(selector, {"path", "name"}, set(), f"{where}: unreset")
    for name, event in spec["events"].items():
        keys(event, *EVENT_KEYS, f"{where}: events.{name}")
        require(("valid" in event) != ("bundle" in event), f"{where}: events.{name} needs exactly one of valid and bundle")
        if "bundle" in event:
            event["valid"] = f"{event['bundle']}_valid"
        if "fields" in event or "bundle" not in event:
            fields = event.get("fields", {})
            require(all(event.get(k) in (None, *fields) for k in ("row", "split_by")), f"{where}: events.{name} names an unknown field")
        if "response" in event:
            response = event["response"]
            keys(response, {"input"}, {"latency", "memory"}, f"{where}: events.{name}.response")
            require("latency" in response or "memory" in response, f"{where}: events.{name}.response needs latency or memory")
            response.setdefault("latency", None)
            require(response["latency"] is None or isinstance(response["latency"], int) and response["latency"] >= 1,
                    f"{where}: events.{name}.response.latency must be a positive integer")
            spec["inputs"].setdefault(response["input"], 0)
    require(set(spec["cuts"]) <= set(spec["inputs"]), f"{where}: cut inputs need idle values in inputs")
    if "operation_table" in spec:
        table = spec["operation_table"]
        keys(table, {"template"}, {"codes"}, f"{where}: operation_table")
        require("codes" in table or words, f"{where}: operation_table needs codes or decode.words")
        codes = table.get("codes", {name: f"${name}" for name in words})
        generated = {name: substitute(table["template"], code) for name, code in codes.items()}
        require(not set(generated) & set(spec["operations"]), f"{where}: operation_table redefines operations")
        spec["operations"] = {**generated, **spec["operations"]}
    require(spec["operations"], f"{where}: no operations")
    for name, op in spec["operations"].items():
        require("/" not in name and "." not in name, f"{where}: invalid operation name {name!r}")
        keys(op, *OPERATION_KEYS, f"{where}: operations.{name}")
        for age, command in op["commands"].items():
            ages(age)
            undeclared = {i for i in set(command) - set(spec["inputs"]) if not (prefix and i.startswith(prefix))}
            require(not undeclared, f"{where}: operations.{name} drives undeclared inputs {sorted(undeclared)}")
        require(symbols(op) <= set(words), f"{where}: operations.{name} uses codes absent from decode.words: {sorted(symbols(op) - set(words))}")
        if "next_issue" in op:
            keys(op["next_issue"], {"signal"}, {"bit"}, f"{where}: operations.{name}.next_issue")
    for check in spec["checks"]:
        keys(check, {"equal"}, {"reason"}, f"{where}: checks")
    require(set(spec.get("only", spec["operations"])) <= set(spec["operations"]), f"{where}: only names unknown operations")
    return spec


def load_spec(path):
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    spec = validate(raw, path.name)
    for name, overlay in spec["variants"].items():
        require("/" not in name and "." not in name, f"{path.name}: invalid variant name {name!r}")
        require(isinstance(overlay, dict) and not set(overlay) & VARIANT_FIXED, f"{path.name}: variants.{name} overrides a fixed key")
        validate(merge({k: v for k, v in raw.items() if k not in ("variants",)}, overlay), f"{path.name}: variants.{name}")
    spec["raw"] = raw
    return spec


def variant(spec, name):
    """The spec with variant ``name`` applied (``None`` is the base spec)."""
    if name is None:
        return spec
    merged = validate(merge({k: v for k, v in spec["raw"].items() if k != "variants"}, spec["variants"][name]), name)
    merged["raw"] = spec["raw"]
    return merged


def available_engines(target):
    return sorted(p.stem for p in (TARGETS / target).glob("*.yaml"))


def load_target(target, engines=None):
    names = available_engines(target)
    require(names, f"Unknown target: {target}")
    selected = names if engines is None else list(engines)
    require(set(selected) <= set(names), f"Unknown engines for {target}: {sorted(set(selected) - set(names))}")
    specs = {name: load_spec(TARGETS / target / f"{name}.yaml") for name in selected}
    require(all(s["engine"] == n for n, s in specs.items()), "Spec engine names must match their file names")
    return specs
