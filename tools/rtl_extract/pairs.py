"""Pairwise overlap: issue operation A, then operation B ``g`` ages later, and find the gaps at which both behave as alone.

A pair file ``targets/<target>/pairs/<name>.yaml`` names an engine spec of the same target and declares pairs:

    engine: <spec stem>                # targets/<target>/<stem>.yaml or targets/<target>/system/<stem>.yaml
    equivalent: {<name>: [group, ...]} # optional; groups compared as one resource (e.g. interchangeable ports)
    ignore: [group, ...]               # optional; level-type groups (busy flags) left out of the comparison
    invariants: {exclude: [...]}       # optional; replaces the spec's invariants (see spec.invariants)
    shared: [{input|path, reason}]     # optional; control the two operations may share (cone check below)
    relation: {name: value}            # optional; operand relation every pair's overlays satisfy (for pair facts)
    pairs:
      <pair>:
        first: <operation>             # issued at its own command ages
        second: <operation>            # issued at its own command ages + g
        gaps: "a..b"                   # swept gaps (inclusive)
        first_inputs: {input: value}   # optional overlay on every command of first (distinct operands)
        second_inputs: {input: value}  # optional overlay on every command of second
        guards: [{signal, bit?}]       # optional; default: second's next_issue. Must be clear at age g.
        as: {first: name, second: name} # optional; the op_timing blocks the sides stand for (default <engine>.<op>)
        relation: {name: value}        # optional; merged over the file's relation
    expect: {<pair>: {...}}            # regression values read only by tests

A gap is accepted when (1) no command ages collide, (2) every control output stays known and, with a ``busy``
signal, the engine drains within ``limit``, (3) every guard is clear at the second's issue age (the issue condition a
frontend asserts but the engine does not enforce), (4) every invariant stays 0 at every age (unknown fails), and (5)
the per-age event multiset (group or its ``equivalent`` name, age and fields; ``ignore`` groups excluded) equals the
first's isolated events plus the second's isolated events shifted by ``g``. A rejected or ignored launch, a dropped,
merged or delayed event, or a changed field fails (5). A gap blocked by a guard still reports whether the events
changed ("events unchanged" marks a guard stricter than the engine needs); ``rejected`` lists each failed part per gap.
When the two operations' event groups are disjoint, ``shared_control`` reports the registers and inputs both groups'
valid signals depend on, other than clock, reset and the declared ``shared`` entries.
``stable_gap`` is the smallest gap from which every swept gap is accepted; that is the value a scheduling rule may
use. Python stays target-agnostic: names, operands and guards live in YAML.
"""
from collections import Counter
import copy
import multiprocessing
from pathlib import Path

import yaml

from . import derive
from .control import require
from .runner import RECIPES, failure, failures, group_valid, observed, schedule
from .spec import TARGETS, ages, available_engines, invariants, keys, load_system, load_target

FILE_KEYS = ({"engine", "pairs"}, {"description", "equivalent", "ignore", "expect", "invariants", "shared", "relation"})
PAIR_KEYS = ({"first", "second", "gaps"}, {"description", "first_inputs", "second_inputs", "guards", "as", "relation"})
SCHEMA = "rtl_extract.op_pairs.v1"


def groups(raw, spec, where):
    """Validates the ``equivalent`` and ``ignore`` groups shared by pair and sequence files."""
    equivalent = raw.setdefault("equivalent", {})
    members = [g for names in equivalent.values() for g in names]
    require(set(members) <= set(spec["events"]) and len(members) == len(set(members)),
            f"{where}: equivalent must name each event group at most once")
    raw.setdefault("ignore", [])
    require(isinstance(raw["ignore"], list) and set(raw["ignore"]) <= set(spec["events"]) and not set(raw["ignore"]) & set(members),
            f"{where}: ignore must name event groups outside equivalent")


def overlay_inputs(spec, overlay, where):
    """Validates an operand overlay: integer values on declared (or command-prefixed) inputs."""
    prefix = f"{spec['command']}_" if spec.get("command") else None
    require(isinstance(overlay, dict) and all(isinstance(v, int) for v in overlay.values()),
            f"{where}: must map inputs to integers")
    undeclared = {i for i in overlay if i not in spec["inputs"] and not (prefix and i.startswith(prefix))}
    require(not undeclared, f"{where}: drives undeclared inputs {sorted(undeclared)}")


def names(value, allowed, where):
    """Validates a mapping of names to strings (``as``, ``relation``); ``allowed`` limits the keys when given."""
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
    engines = [m for m in missing if m in available_engines(target)]
    specs.update(load_target(target, engines) if engines else {})
    specs.update({m: load_system(target, m) for m in missing if m not in engines})
    return {n: (validate(r, specs[r["engine"]], f"pairs/{n}.yaml"), specs[r["engine"]]) for n, r in raws.items()}


def prepared(raw, spec):
    """The spec a pair file runs: its own ``invariants`` replace the spec's."""
    return {**spec, "invariants": raw["invariants"]} if "invariants" in raw else spec


def needs_assertions(raw, spec):
    return "invariants" in prepared(raw, spec)


def commands(spec, op, overlay, shift=0):
    return {age + shift: {**command, **overlay} for age, command in schedule(spec["operations"][op]).items()}


def run(circuit, spec, timeline):
    """``(trace, problem)`` for ``timeline`` ({age: command}) under the recipe's runner; ``problem`` is a drain failure."""
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
    """One timeline from per-operation schedules, or ``None`` when two commands share an age."""
    merged = {}
    for schedule in schedules:
        if set(merged) & set(schedule):
            return None
        merged.update(schedule)
    return merged


def judge(circuit, spec, raw, timeline, issues, expected, explain=difference):
    """Outcome of one run, shared by pairs and chains: ``{guards, invariants, events, error, accepted}`` (failed parts
    first). ``issues`` lists ``(age, guards, suffix)``: each guard must be clear at that issue age (``suffix`` ends its
    message); ``expected`` is the sorted event multiset; ``explain(expected, actual)`` words an event mismatch."""
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
    """``(events, groups)`` of ``op`` issued alone; a drain failure or a firing invariant is an error."""
    trace, problem = run(circuit, spec, commands(spec, op, overlay))
    require(problem is None, f"{label}{op} alone: {problem}")
    fired = failures(circuit, trace)
    if fired:
        raise ValueError(f"{label}{op} alone: {failure(fired[0])}")
    return events(trace, raw["equivalent"], ignore=raw["ignore"]), {g for e in trace for g in e["events"]} - set(raw["ignore"])


def check(circuit, spec, raw, pair, alone, gap):
    """Per-gap outcome: ``{gap, accepted, guards, invariants, events, error}`` (failed parts only)."""
    first = commands(spec, pair["first"], pair.get("first_inputs", {}))
    second = commands(spec, pair["second"], pair.get("second_inputs", {}), gap)
    expected = sorted(alone[0] + [(g, a + gap, f) for g, a, f in alone[1]])
    return {"gap": gap, **judge(circuit, spec, raw, merge([first, second]), [(min(second), guards(spec, pair), "")], expected)}


def describe(outcome):
    """One line for a per-gap outcome, ``None`` when accepted."""
    if outcome["accepted"]:
        return None
    parts = outcome.get("guards", []) + [failure(f) for f in outcome.get("invariants", [])]
    parts += [outcome[k] for k in ("error", "events") if outcome.get(k)]
    if outcome.get("guards") and not set(outcome) - {"gap", "accepted", "guards"}:
        parts.append("events unchanged")
    return "; ".join(parts)


def trial(circuit, spec, raw, pair, alone, gap):
    """``None`` if the second may issue at ``gap``, otherwise the reason it may not."""
    return describe(check(circuit, spec, raw, pair, alone, gap))


def limits(results):
    """``(min_gap, stable_gap)`` of ``[(gap, reason)]``: the smallest accepted gap and the smallest from which all pass."""
    stable = None
    for gap, reason in reversed(results):
        if reason is not None:
            break
        stable = gap
    return min((g for g, r in results if r is None), default=None), stable


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


def shared_control(circuit, spec, raw, groups):
    """Registers and inputs reached from both operations' event valids, other than clock, reset and the declared
    ``shared`` inputs; traversal stops at declared ``shared`` instance paths, whose conflicts invariants govern."""
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


def check_gap(gap):
    return check(*TASK, gap)


def sweep(circuit, spec, raw, pair, alone, gaps, jobs=1):
    """``check`` for every gap, in ``jobs`` forked processes when more than one."""
    global TASK
    if jobs <= 1 or len(gaps) < 2:
        return [check(circuit, spec, raw, pair, alone, gap) for gap in gaps]
    TASK = (circuit, spec, raw, pair, alone)
    with multiprocessing.get_context("fork").Pool(min(jobs, len(gaps))) as pool:
        return pool.map(check_gap, gaps, chunksize=1)


def extract(document, raw, spec, jobs=1):
    """One record per pair of one pair file: per-gap results, the minimum accepted gap and the stable gap."""
    concrete, _ = derive.resolve(document, prepared(raw, spec), {})
    circuit = RECIPES[concrete["recipe"]][0](document, concrete)
    records, isolated = [], {}
    for name, pair in raw["pairs"].items():
        alone, groups = [], []
        for side in ("first", "second"):
            key = (pair[side], tuple(sorted(pair.get(f"{side}_inputs", {}).items())))
            if key not in isolated:
                isolated[key] = isolate(circuit, concrete, raw, f"{name}: ", pair[side], dict(key[1]))
            alone.append(isolated[key][0])
            groups.append(isolated[key][1])
        outcomes = sweep(circuit, concrete, raw, pair, alone, ages(pair["gaps"]), jobs)
        results = [(o["gap"], describe(o)) for o in outcomes]
        low, high = limits(results)
        records.append({"name": f"{spec['engine']}.{name}", "engine": spec["engine"], "module": spec["module"],
                        "first": pair["first"], "second": pair["second"], "gaps": [results[0][0], results[-1][0]],
                        "min_gap": low, "stable_gap": high,
                        "guards": guards(concrete, pair), "results": runs(results),
                        "rejected": [o for o in outcomes if not o["accepted"]],
                        "evidence": f"{concrete['recipe']} control simulation of {spec['module']}: {len(results)} gaps, "
                                    "event multisets compared with isolated runs"})
        if "invariants" in concrete:
            records[-1]["invariants"] = {"checked": len(circuit.invariants), "excluded": circuit.excluded}
        if not groups[0] & groups[1]:
            records[-1]["shared_control"] = shared_control(circuit, concrete, raw, groups)
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
    parser.add_argument("--jobs", type=int, default=1, help="processes per gap sweep")
    args = parser.parse_args(argv)
    loaded = load(args.target, args.pairs)
    document = load_document(args.hw_ir, sorted({m for _, s in loaded.values() for m in modules(s)}), args.exporter,
                             any(needs_assertions(r, s) for r, s in loaded.values()))
    records = [r for raw, spec in loaded.values() for r in extract(document, raw, spec, args.jobs)]
    facts.write(args.output, {"schema": SCHEMA, "hw_ir": {"path": str(args.hw_ir.resolve()), "sha256": facts.sha256(args.hw_ir)},
                              "op_pairs": records})
    print(json.dumps({r["name"]: {"min_gap": r["min_gap"], "stable_gap": r["stable_gap"], "results": r["results"]}
                      for r in records}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
