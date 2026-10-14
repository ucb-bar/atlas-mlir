"""Sweep uniformly spaced chains with the pair acceptance rules."""
from collections import Counter
from pathlib import Path

from . import derive, pairs
from .control import require
from .runner import RECIPES, observed
from .spec import ages, invariants, keys, load_sweeps, read_spec, spec_names

FILE_KEYS = ({"engine", "sequences"}, {"description", "equivalent", "ignore", "expect", "invariants"})
SEQUENCE_KEYS = ({"ops", "lengths", "gaps"}, {"description"})
OP_KEYS = ({"op"}, {"inputs", "guards", "as"})
SCHEMA = "rtl_extract.op_sequences.v1"

def validate(raw, spec, where):
    keys(raw, *FILE_KEYS, where)
    require(raw["engine"] == spec["engine"], f"{where}: engine {raw['engine']!r} does not match its spec")
    require(isinstance(raw["sequences"], dict) and raw["sequences"], f"{where}: no sequences")
    pairs.groups(raw, spec, where)
    if "invariants" in raw:
        invariants(raw["invariants"], f"{where}: invariants")
    signals = set(observed(spec))
    for name, sequence in raw["sequences"].items():
        at = f"{where}: sequences.{name}"
        require("/" not in name and "." not in name, f"{at}: invalid sequence name")
        keys(sequence, *SEQUENCE_KEYS, at)
        require(isinstance(sequence["ops"], list) and sequence["ops"], f"{at}: ops must be a non-empty list")
        for n, entry in enumerate(sequence["ops"]):
            keys(entry, *OP_KEYS, f"{at}.ops[{n}]")
            require(entry["op"] in spec["operations"], f"{at}.ops[{n}]: unknown operation {entry['op']!r}")
            pairs.overlay_inputs(spec, entry.get("inputs", {}), f"{at}.ops[{n}]: inputs")
            require(isinstance(entry.get("as", ""), str), f"{at}.ops[{n}]: as must name an op_timing block")
            for guard in pairs.guards(spec, step(entry)):
                keys(guard, {"signal"}, {"bit"}, f"{at}.ops[{n}]: guards")
                require(guard["signal"] in signals, f"{at}.ops[{n}]: guard {guard['signal']!r} is not an observed control signal")
        require(ages(sequence["lengths"])[0] >= 2, f"{at}: lengths start at 2")
        require(ages(sequence["gaps"])[0] >= 1, f"{at}: gaps start at 1")
    expect = raw.setdefault("expect", {})
    require(isinstance(expect, dict) and set(expect) <= set(raw["sequences"]), f"{where}: expect names unknown sequences")
    for name, lengths in expect.items():
        require(isinstance(lengths, dict) and set(lengths) <= set(ages(raw["sequences"][name]["lengths"])) and all(isinstance(g, int) for g in lengths.values()),
                f"{where}: expect.{name} must map chain lengths of the sequence to stable gaps")
    return raw

def step(entry):
    return {"second": entry["op"], **({"guards": entry["guards"]} if "guards" in entry else {})}

def available():
    return spec_names("sequences")

def load_file(path, spec):
    path = Path(path)
    return validate(read_spec(path), spec, path.name)

def load(names=None, specs=None):
    return load_sweeps("sequences", names, specs, validate)

def chain(sequence, length):
    return [sequence["ops"][i % len(sequence["ops"])] for i in range(length)]

def timeline(spec, steps, gap):
    schedules = [pairs.commands(spec, s["op"], s.get("inputs", {}), i * gap) for i, s in enumerate(steps)]
    return pairs.merge(schedules), schedules

def lost(expected, actual):
    remaining, positions = Counter(actual), []
    for i, events in enumerate(expected):
        wanted = Counter(events)
        if wanted - remaining:
            positions.append(i)
        remaining -= wanted & remaining
    return positions

def difference(expected, actual):
    parts = []
    for label, counter in (("missing", Counter(expected) - Counter(actual)), ("unexpected", Counter(actual) - Counter(expected))):
        groups = Counter(g for (g, _, _), n in counter.items() for _ in range(n))
        if groups:
            parts.append(f"{label} " + ", ".join(f"{g} x{n}" for g, n in sorted(groups.items())))
    return "; ".join(parts)

def trial(circuit, spec, raw, steps, alone, gap):
    merged, schedules = timeline(spec, steps, gap)
    expected = [[(g, a + i * gap, f) for g, a, f in events] for i, events in enumerate(alone)]
    flat = sorted(e for events in expected for e in events)
    issues = [(min(schedule), pairs.guards(spec, step(s)), f" at op {i}") for i, (s, schedule) in enumerate(zip(steps, schedules)) if i]
    explain = lambda flat, actual: f"{difference(flat, actual)} (ops {lost(expected, actual) or 'none'})"
    return pairs.describe(pairs.judge(circuit, spec, raw, merged, issues, flat, explain))

def isolated(circuit, spec, raw, entry):
    return pairs.isolate(circuit, spec, raw, "", entry["op"], entry.get("inputs", {}))[0]

def key(entry):
    return entry["op"], tuple(sorted(entry.get("inputs", {}).items()))

TASK = None

def cell(task):
    case, gap = task
    circuit, spec, raw, cases = TASK
    return trial(circuit, spec, raw, cases[case][1], cases[case][2], gap)

def extract(document, raw, spec, jobs=1, only=None):
    global TASK
    concrete, _ = derive.resolve(document, pairs.prepared(raw, spec), {})
    circuit = RECIPES[concrete["recipe"]][0](document, concrete)
    selected = [n for n in raw["sequences"] if only is None or n in only]
    chains = [(name, chain(raw["sequences"][name], length), length)
              for name in selected for length in ages(raw["sequences"][name]["lengths"])]
    entries = {key(s): s for _, steps, _ in chains for s in steps}
    pairs.TASK = (circuit, concrete, raw)
    try:
        alone = dict(zip(entries, (events for events, _ in pairs.parallel(
            pairs.isolate_task, [("", s["op"], s.get("inputs", {})) for s in entries.values()], jobs))))
    finally:
        pairs.TASK = None
    cases = [(name, steps, [alone[key(s)] for s in steps], length) for name, steps, length in chains]
    tasks = [(c, gap) for c, case in enumerate(cases) for gap in ages(raw["sequences"][case[0]]["gaps"])]
    TASK = (circuit, concrete, raw, cases)
    try:
        reasons = pairs.parallel(cell, tasks, jobs)
    finally:
        TASK = None
    records = []
    for c, (name, steps, _, length) in enumerate(cases):
        results = [(gap, reason) for (case, gap), reason in zip(tasks, reasons) if case == c]
        low, high = pairs.limits(results)
        records.append({"name": f"{spec['engine']}.{name}/{length}", "engine": spec["engine"], "module": spec["module"],
                        "sequence": name, "length": length,
                        "ops": [{"op": s["op"], **({"inputs": s["inputs"]} if "inputs" in s else {})} for s in steps],
                        "gaps": [results[0][0], results[-1][0]], "min_gap": low, "stable_gap": high,
                        "results": pairs.runs(results),
                        "evidence": f"{concrete['recipe']} control simulation of {spec['module']}: {len(results)} spacings, "
                                    "event multisets compared with isolated runs"})
        if "invariants" in concrete:
            records[-1]["invariants"] = {"checked": len(circuit.invariants), "excluded": circuit.excluded}
    return records

def main(argv=None):
    return pairs.sweep_main("sequences", load, extract, SCHEMA, argv)

if __name__ == "__main__":
    raise SystemExit(main())
