"""Sweep pair gaps against isolated events, issue guards, drain and assertions."""
from collections import Counter
import copy
import multiprocessing
from pathlib import Path

from . import derive
from .control import require
from .runner import RECIPES, failure, failures, group_valid, observed, schedule
from .spec import ages, invariants, keys, load_sweeps, read_spec, spec_names

FILE_KEYS = ({"engine", "pairs"}, {"description", "equivalent", "ignore", "expect", "invariants", "shared", "relation"})
PAIR_KEYS = ({"first", "second", "gaps"}, {"description", "first_inputs", "second_inputs", "guards", "as", "relation"})
SCHEMA = "rtl_extract.op_pairs.v1"

def groups(raw, spec, where):
    equivalent = raw.setdefault("equivalent", {})
    members = [g for names in equivalent.values() for g in names]
    require(set(members) <= set(spec["events"]) and len(members) == len(set(members)),
            f"{where}: equivalent must name each event group at most once")
    raw.setdefault("ignore", [])
    require(isinstance(raw["ignore"], list) and set(raw["ignore"]) <= set(spec["events"]) and not set(raw["ignore"]) & set(members),
            f"{where}: ignore must name event groups outside equivalent")

def overlay_inputs(spec, overlay, where):
    prefix = f"{spec['command']}_" if spec.get("command") else None
    require(isinstance(overlay, dict) and all(isinstance(v, int) for v in overlay.values()),
            f"{where}: must map inputs to integers")
    undeclared = {i for i in overlay if i not in spec["inputs"] and not (prefix and i.startswith(prefix))}
    require(not undeclared, f"{where}: drives undeclared inputs {sorted(undeclared)}")

def names(value, allowed, where):
    require(isinstance(value, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in value.items()),
            f"{where}: must map names to strings")
    require(allowed is None or set(value) <= allowed, f"{where}: unknown keys {sorted(set(value) - (allowed or set()))}")

def validate(raw, spec, where):
    keys(raw, *FILE_KEYS, where)
    names(raw.get("relation", {}), None, f"{where}: relation")
    require(raw["engine"] == spec["engine"], f"{where}: engine {raw['engine']!r} does not match its spec")
    require(isinstance(raw["pairs"], dict) and raw["pairs"], f"{where}: no pairs")
    groups(raw, spec, where)
    if "invariants" in raw:
        invariants(raw["invariants"], f"{where}: invariants")
    raw.setdefault("shared", [])
    require(isinstance(raw["shared"], list), f"{where}: shared must be a list")
    for entry in raw["shared"]:
        keys(entry, {"reason"}, {"input", "path"}, f"{where}: shared")
        require(("input" in entry) != ("path" in entry), f"{where}: shared entries name one input or one path")
    signals = set(observed(spec))
    for name, pair in raw["pairs"].items():
        at = f"{where}: pairs.{name}"
        require("/" not in name and "." not in name, f"{at}: invalid pair name")
        keys(pair, *PAIR_KEYS, at)
        for side in ("first", "second"):
            require(pair[side] in spec["operations"], f"{at}: unknown operation {pair[side]!r}")
            overlay_inputs(spec, pair.get(f"{side}_inputs", {}), f"{at}: {side}_inputs")
        require(ages(pair["gaps"])[0] >= 1, f"{at}: gaps start at 1")
        names(pair.get("as", {}), {"first", "second"}, f"{at}: as")
        names(pair.get("relation", {}), None, f"{at}: relation")
        for guard in guards(spec, pair):
            keys(guard, {"signal"}, {"bit"}, f"{at}: guards")
            require(guard["signal"] in signals, f"{at}: guard {guard['signal']!r} is not an observed control signal")
    return raw

def guards(spec, pair):
    if "guards" in pair:
        return pair["guards"]
    second = spec["operations"][pair["second"]]
    return [second["next_issue"]] if "next_issue" in second else []

def available():
    return spec_names("pairs")

def load_file(path, spec):
    path = Path(path)
    return validate(read_spec(path), spec, path.name)

def load(names=None, specs=None):
    return load_sweeps("pairs", names, specs, validate)

def prepared(raw, spec):
    return {**spec, "invariants": raw["invariants"]} if "invariants" in raw else spec

def needs_assertions(raw, spec):
    return "invariants" in prepared(raw, spec)

def commands(spec, op, overlay, shift=0):
    return {age + shift: {**command, **overlay} for age, command in schedule(spec["operations"][op]).items()}

def run(circuit, spec, timeline):
    trial = copy.deepcopy(spec)
    trial["operations"]["\0pair"] = {"commands": timeline}
    trial["limit"] = spec["limit"] + max(timeline) - min(timeline)  # the later command gets the full window too
    trace = RECIPES[spec["recipe"]][1](circuit, trial, "\0pair")
    drained = not spec["busy"] or trace[-1]["busy"] == 0
    return trace, None if drained else "busy did not clear"

def events(trace, equivalent, shift=0, ignore=()):
    alias = {g: name for name, groups in equivalent.items() for g in groups}
    return sorted((alias.get(group, group), e["age"] + shift, tuple(sorted(fields.items())))
                  for e in trace for group, fields in e["events"].items() if group not in ignore)

def difference(expected, actual):
    missing = list((Counter(expected) - Counter(actual)).elements())
    extra = list((Counter(actual) - Counter(expected)).elements())
    first = min(missing + extra, key=lambda x: x[1])
    return f"{'missing' if first in missing else 'unexpected'} {first[0]} events"

def clear(value, bit):
    return value is not None and not (value >> bit & 1 if bit is not None else value)

def merge(schedules):
    merged = {}
    for schedule in schedules:
        if set(merged) & set(schedule):
            return None
        merged.update(schedule)
    return merged

def judge(circuit, spec, raw, timeline, issues, expected, explain=difference):
    if timeline is None:
        return {"accepted": False, "error": "command ages collide"}
    try:
        trace, problem = run(circuit, spec, timeline)
    except ValueError as error:
        return {"accepted": False, "error": str(error)}
    blocked = [f"guard {g['signal']}" + (f"[{g['bit']}]" if "bit" in g else "") + f" set{suffix}"
               for age, gs, suffix in issues for g in gs if not clear(trace[age]["signals"][g["signal"]], g.get("bit"))]
    actual = events(trace, raw["equivalent"], ignore=raw["ignore"])
    outcome = {"guards": blocked, "invariants": failures(circuit, trace),
               "events": None if actual == expected else explain(expected, actual), "error": problem}
    outcome = {k: v for k, v in outcome.items() if v}
    return {**outcome, "accepted": not outcome}

def isolate(circuit, spec, raw, label, op, overlay):
    trace, problem = run(circuit, spec, commands(spec, op, overlay))
    require(problem is None, f"{label}{op} alone: {problem}")
    fired = failures(circuit, trace)
    if fired:
        raise ValueError(f"{label}{op} alone: {failure(fired[0])}")
    return events(trace, raw["equivalent"], ignore=raw["ignore"]), {g for e in trace for g in e["events"]} - set(raw["ignore"])

def check(circuit, spec, raw, pair, alone, gap):
    first = commands(spec, pair["first"], pair.get("first_inputs", {}))
    second = commands(spec, pair["second"], pair.get("second_inputs", {}), gap)
    expected = sorted(alone[0] + [(g, a + gap, f) for g, a, f in alone[1]])
    return {"gap": gap, **judge(circuit, spec, raw, merge([first, second]), [(min(second), guards(spec, pair), "")], expected)}

def describe(outcome):
    if outcome["accepted"]:
        return None
    parts = outcome.get("guards", []) + [failure(f) for f in outcome.get("invariants", [])]
    parts += [outcome[k] for k in ("error", "events") if outcome.get(k)]
    if outcome.get("guards") and not set(outcome) - {"gap", "accepted", "guards"}:
        parts.append("events unchanged")
    return "; ".join(parts)

def trial(circuit, spec, raw, pair, alone, gap):
    return describe(check(circuit, spec, raw, pair, alone, gap))

def limits(results):
    stable = None
    for gap, reason in reversed(results):
        if reason is not None:
            break
        stable = gap
    return min((g for g, r in results if r is None), default=None), stable

def runs(results):
    out = []
    for gap, reason in results:
        result = reason or "accepted"
        if out and out[-1]["result"] == result and out[-1]["to"] == gap - 1:
            out[-1]["to"] = gap
        else:
            out.append({"from": gap, "to": gap, "result": result})
    return out

def shared_control(circuit, spec, raw, groups):
    require(hasattr(circuit, "reach"), "shared-control check needs a single circuit")
    stop = ["".join("/" + n for n in e["path"].split("/") if n) for e in raw["shared"] if "path" in e]
    declared = {e["input"] for e in raw["shared"] if "input" in e} | {spec["clock"], spec["reset"]}
    cones = [circuit.reach(sorted({group_valid(spec, g) for g in side}), stop) for side in groups]

    def label(key):
        attrs = circuit.nodes[key][2]
        return f"{key[0]}/{attrs.get('name') or attrs.get('sv.namehint') or key[1]}"
    registers = sorted(label(k) for k in cones[0][0] & cones[1][0])
    inputs = sorted((cones[0][1] & cones[1][1]) - declared)
    return {"groups": [sorted(g) for g in groups], "registers": [len(c[0]) for c in cones],
            "shared_registers": registers, "shared_inputs": inputs, "disjoint": not registers and not inputs}

TASK = None

def parallel(function, tasks, jobs):
    if jobs <= 1 or len(tasks) < 2:
        return [function(t) for t in tasks]
    with multiprocessing.get_context("fork").Pool(min(jobs, len(tasks))) as pool:
        return pool.map(function, tasks, chunksize=1)

def isolate_task(task):
    circuit, spec, raw = TASK[:3]
    return isolate(circuit, spec, raw, *task)

def check_task(task):
    circuit, spec, raw, alone = TASK
    name, gap = task
    return check(circuit, spec, raw, raw["pairs"][name], alone[name], gap)

def side_key(pair, side):
    return pair[side], tuple(sorted(pair.get(f"{side}_inputs", {}).items()))

def setup(raw, spec, pair):
    return {**{f"{side}_inputs": dict(pair.get(f"{side}_inputs", {})) for side in ("first", "second")},
            "as": {side: pair.get("as", {}).get(side, f"{spec['engine']}.{pair[side]}") for side in ("first", "second")},
            "relation": {**raw.get("relation", {}), **pair.get("relation", {})}}

def extract(document, raw, spec, jobs=1):
    global TASK
    concrete, _ = derive.resolve(document, prepared(raw, spec), {})
    circuit = RECIPES[concrete["recipe"]][0](document, concrete)
    labels = {}
    for name, pair in raw["pairs"].items():
        for side in ("first", "second"):
            labels.setdefault(side_key(pair, side), f"{name}: ")
    try:
        TASK = (circuit, concrete, raw)
        isolated = dict(zip(labels, parallel(isolate_task, [(label, op, dict(overlay)) for (op, overlay), label in labels.items()], jobs)))
        alone = {name: [isolated[side_key(pair, side)][0] for side in ("first", "second")] for name, pair in raw["pairs"].items()}
        tasks = [(name, gap) for name, pair in raw["pairs"].items() for gap in ages(pair["gaps"])]
        TASK = (circuit, concrete, raw, alone)
        outcomes = parallel(check_task, tasks, jobs)
    finally:
        TASK = None
    records = []
    for name, pair in raw["pairs"].items():
        mine = [o for (owner, _), o in zip(tasks, outcomes) if owner == name]
        groups = [isolated[side_key(pair, side)][1] for side in ("first", "second")]
        results = [(o["gap"], describe(o)) for o in mine]
        low, high = limits(results)
        records.append({"name": f"{spec['engine']}.{name}", "engine": spec["engine"], "module": spec["module"],
                        "first": pair["first"], "second": pair["second"], **setup(raw, spec, pair),
                        "gaps": [results[0][0], results[-1][0]],
                        "min_gap": low, "stable_gap": high, "guards": guards(concrete, pair), "results": runs(results),
                        "rejected": [o for o in mine if not o["accepted"]],
                        "evidence": f"{concrete['recipe']} control simulation of {spec['module']}: {len(results)} gaps, "
                                    "event multisets compared with isolated runs"})
        if "invariants" in concrete:
            records[-1]["invariants"] = {"checked": len(circuit.invariants), "excluded": circuit.excluded}
        if not groups[0] & groups[1]:
            records[-1]["shared_control"] = shared_control(circuit, concrete, raw, groups)
    return records

def sweep_main(kind, load, extract, schema, argv=None):
    import argparse
    import json
    from . import facts
    from .ir import load_document
    from .runner import modules
    parser = argparse.ArgumentParser(description=f"Sweep declared operation {kind}.")
    parser.add_argument("--hw-ir", type=Path, required=True)
    parser.add_argument("--exporter", type=Path, required=True)
    parser.add_argument(f"--{kind}", nargs="+", help="spec names (default: all)")
    if kind == "sequences":
        parser.add_argument("--only", nargs="+", help="sequence names (default: all)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=1, help="worker processes")
    args = parser.parse_args(argv)
    loaded = load(getattr(args, kind))
    only = getattr(args, "only", None)
    if kind == "sequences":
        unknown = set(only or ()) - {n for raw, _ in loaded.values() for n in raw[kind]}
        require(not unknown, f"Unknown sequences: {sorted(unknown)}")
    document = load_document(args.hw_ir, sorted({m for _, s in loaded.values() for m in modules(s)}), args.exporter,
                             any(needs_assertions(r, s) for r, s in loaded.values()))
    extra = (only and set(only),) if kind == "sequences" else ()
    records = [r for raw, spec in loaded.values() for r in extract(document, raw, spec, args.jobs, *extra)]
    facts.write(args.output, {"schema": schema, "hw_ir": {"path": str(args.hw_ir.resolve()), "sha256": facts.sha256(args.hw_ir)},
                             f"op_{kind}": records})
    fields = ("min_gap", "stable_gap", "results") if kind == "pairs" else ("min_gap", "stable_gap")
    print(json.dumps({r["name"]: {k: r[k] for k in fields} for r in records}, indent=1))
    return 0

def main(argv=None):
    return sweep_main("pairs", load, extract, SCHEMA, argv)

if __name__ == "__main__":
    raise SystemExit(main())
