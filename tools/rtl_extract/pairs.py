"""Pairwise overlap: issue operation A, then operation B ``g`` ages later, and find the gaps at which both behave as alone.

A pair file ``targets/<target>/pairs/<name>.yaml`` names an engine spec of the same target and declares pairs:

    engine: <spec stem>
    equivalent: {<name>: [group, ...]} # optional; groups compared as one resource (e.g. interchangeable ports)
    ignore: [group, ...]               # optional; level-type groups (busy flags) left out of the comparison
    pairs:
      <pair>:
        first: <operation>             # issued at its own command ages
        second: <operation>            # issued at its own command ages + g
        gaps: "a..b"                   # swept gaps (inclusive)
        first_inputs: {input: value}   # optional overlay on every command of first (distinct operands)
        second_inputs: {input: value}  # optional overlay on every command of second
        guards: [{signal, bit?}]       # optional; default: second's next_issue. Must be clear at age g.
    expect: {<pair>: {...}}            # regression values read only by tests

A gap is accepted when (1) no command ages collide, (2) every control output stays known and, with a ``busy``
signal, the engine drains within ``limit``, (3) every guard is clear at the second's issue age (the issue condition a
frontend asserts but the engine does not enforce), and (4) the per-age event multiset (group or its ``equivalent``
name, age and fields; ``ignore`` groups excluded) equals the first's isolated events plus the second's isolated
events shifted by ``g``. A rejected or ignored launch, a dropped, merged or delayed event, or a changed field fails
(4). A gap blocked by a guard still reports whether the events changed ("events unchanged" marks a guard stricter
than the engine needs).
``stable_gap`` is the smallest gap from which every swept gap is accepted; that is the value a scheduling rule may
use. Python stays target-agnostic: names, operands and guards live in YAML.
"""
from collections import Counter
import copy
from pathlib import Path

import yaml

from . import derive
from .control import require
from .runner import RECIPES, observed, schedule
from .spec import TARGETS, ages, keys, load_target

FILE_KEYS = ({"engine", "pairs"}, {"description", "equivalent", "ignore", "expect"})
PAIR_KEYS = ({"first", "second", "gaps"}, {"description", "first_inputs", "second_inputs", "guards"})
SCHEMA = "rtl_extract.op_pairs.v1"


def validate(raw, spec, where):
    keys(raw, *FILE_KEYS, where)
    require(raw["engine"] == spec["engine"], f"{where}: engine {raw['engine']!r} does not match its spec")
    require(isinstance(raw["pairs"], dict) and raw["pairs"], f"{where}: no pairs")
    equivalent = raw.setdefault("equivalent", {})
    members = [g for groups in equivalent.values() for g in groups]
    require(set(members) <= set(spec["events"]) and len(members) == len(set(members)),
            f"{where}: equivalent must name each event group at most once")
    raw.setdefault("ignore", [])
    require(isinstance(raw["ignore"], list) and set(raw["ignore"]) <= set(spec["events"]) and not set(raw["ignore"]) & set(members),
            f"{where}: ignore must name event groups outside equivalent")
    prefix = f"{spec['command']}_" if spec.get("command") else None
    signals = set(observed(spec))
    for name, pair in raw["pairs"].items():
        at = f"{where}: pairs.{name}"
        require("/" not in name and "." not in name, f"{at}: invalid pair name")
        keys(pair, *PAIR_KEYS, at)
        for side in ("first", "second"):
            require(pair[side] in spec["operations"], f"{at}: unknown operation {pair[side]!r}")
            overlay = pair.get(f"{side}_inputs", {})
            require(isinstance(overlay, dict) and all(isinstance(v, int) for v in overlay.values()),
                    f"{at}: {side}_inputs must map inputs to integers")
            undeclared = {i for i in overlay if i not in spec["inputs"] and not (prefix and i.startswith(prefix))}
            require(not undeclared, f"{at}: {side}_inputs drives undeclared inputs {sorted(undeclared)}")
        require(ages(pair["gaps"])[0] >= 1, f"{at}: gaps start at 1")
        for guard in guards(spec, pair):
            keys(guard, {"signal"}, {"bit"}, f"{at}: guards")
            require(guard["signal"] in signals, f"{at}: guard {guard['signal']!r} is not an observed control signal")
    return raw


def guards(spec, pair):
    if "guards" in pair:
        return pair["guards"]
    second = spec["operations"][pair["second"]]
    return [second["next_issue"]] if "next_issue" in second else []


def available(target):
    return sorted(p.stem for p in (TARGETS / target / "pairs").glob("*.yaml"))


def load_file(path, spec):
    path = Path(path)
    return validate(yaml.safe_load(path.read_text()), spec, path.name)


def load(target, names=None, specs=None):
    """``{name: (pair file, engine spec)}`` for the pair files of ``target``."""
    selected = available(target) if names is None else list(names)
    require(set(selected) <= set(available(target)), f"Unknown pair files: {sorted(set(selected) - set(available(target)))}")
    raws = {n: yaml.safe_load((TARGETS / target / "pairs" / f"{n}.yaml").read_text()) for n in selected}
    specs = dict(specs or {})
    missing = sorted({r["engine"] for r in raws.values()} - set(specs))
    specs.update(load_target(target, missing) if missing else {})
    return {n: (validate(r, specs[r["engine"]], f"pairs/{n}.yaml"), specs[r["engine"]]) for n, r in raws.items()}


def commands(spec, op, overlay, shift=0):
    return {age + shift: {**command, **overlay} for age, command in schedule(spec["operations"][op]).items()}


def run(circuit, spec, timeline):
    """``(trace, problem)`` for ``timeline`` ({age: command}) under the recipe's runner; ``problem`` is a drain failure."""
    trial = copy.deepcopy(spec)
    trial["operations"]["\0pair"] = {"commands": timeline}
    trial["limit"] = spec["limit"] + max(timeline) - min(timeline)  # the later command gets the full window too
    trace = RECIPES[spec["recipe"]][1](circuit, trial, "\0pair")
    drained = not spec["busy"] or trace[-1]["busy"] == 0
    return trace, None if drained else f"busy did not clear within {trial['limit']} ages"


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


def trial(circuit, spec, raw, pair, alone, gap):
    """``None`` if the second may issue at ``gap``, otherwise the reason it may not."""
    first = commands(spec, pair["first"], pair.get("first_inputs", {}))
    second = commands(spec, pair["second"], pair.get("second_inputs", {}), gap)
    if set(first) & set(second):
        return "command ages collide"
    try:
        trace, problem = run(circuit, spec, {**first, **second})
    except ValueError as error:
        return str(error)
    issue = min(second)
    blocked = [f"guard {g['signal']}" + (f"[{g['bit']}]" if "bit" in g else "") + " set"
               for g in guards(spec, pair) if not clear(trace[issue]["signals"][g["signal"]], g.get("bit"))]
    expected = sorted(alone[0] + [(g, a + gap, f) for g, a, f in alone[1]])
    actual = events(trace, raw["equivalent"], ignore=raw["ignore"])
    changed = None if actual == expected else difference(expected, actual)
    reasons = blocked + [r for r in (problem, changed) if r]
    if blocked and not changed and not problem:
        reasons.append("events unchanged")
    return "; ".join(reasons) or None


def runs(results):
    """Compress ``[(gap, reason)]`` to ``[{from, to, result}]`` runs (``result`` is ``accepted`` or the reason)."""
    out = []
    for gap, reason in results:
        result = reason or "accepted"
        if out and out[-1]["result"] == result and out[-1]["to"] == gap - 1:
            out[-1]["to"] = gap
        else:
            out.append({"from": gap, "to": gap, "result": result})
    return out


def extract(document, raw, spec):
    """One record per pair of one pair file: per-gap results, the minimum accepted gap and the stable gap."""
    concrete, _ = derive.resolve(document, spec, {})
    circuit = RECIPES[concrete["recipe"]][0](document, concrete)
    records = []
    for name, pair in raw["pairs"].items():
        alone = []
        for side in ("first", "second"):
            trace, problem = run(circuit, concrete, commands(concrete, pair[side], pair.get(f"{side}_inputs", {})))
            require(problem is None, f"{name}: {pair[side]} alone: {problem}")
            alone.append(events(trace, raw["equivalent"], ignore=raw["ignore"]))
        results = [(gap, trial(circuit, concrete, raw, pair, alone, gap)) for gap in ages(pair["gaps"])]
        stable = None
        for gap, reason in reversed(results):
            if reason is not None:
                break
            stable = gap
        records.append({"name": f"{spec['engine']}.{name}", "engine": spec["engine"], "module": spec["module"],
                        "first": pair["first"], "second": pair["second"], "gaps": [results[0][0], results[-1][0]],
                        "min_gap": min((g for g, r in results if r is None), default=None), "stable_gap": stable,
                        "guards": guards(concrete, pair), "results": runs(results),
                        "evidence": f"{concrete['recipe']} control simulation of {spec['module']}: {len(results)} gaps, "
                                    "event multisets compared with isolated runs"})
    return records


def main(argv=None):
    import argparse
    import json
    from . import facts
    from .ir import load_document
    from .runner import modules
    parser = argparse.ArgumentParser(description="Sweep declared operation pairs (see tools/rtl_extract/pairs.py).")
    parser.add_argument("--hw-ir", type=Path, required=True)
    parser.add_argument("--exporter", type=Path, required=True)
    parser.add_argument("--target", default="atlas")
    parser.add_argument("--pairs", nargs="+", help="pair files (default: every file under targets/<target>/pairs/)")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    loaded = load(args.target, args.pairs)
    document = load_document(args.hw_ir, sorted({m for _, s in loaded.values() for m in modules(s)}), args.exporter)
    records = [r for raw, spec in loaded.values() for r in extract(document, raw, spec)]
    facts.write(args.output, {"schema": SCHEMA, "hw_ir": {"path": str(args.hw_ir.resolve()), "sha256": facts.sha256(args.hw_ir)},
                              "op_pairs": records})
    print(json.dumps({r["name"]: {"min_gap": r["min_gap"], "stable_gap": r["stable_gap"], "results": r["results"]}
                      for r in records}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
