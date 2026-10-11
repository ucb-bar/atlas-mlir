"""Sequences of N operations at uniform spacing: operation i is issued ``i * gap`` ages after the first.

A sequence file ``targets/<target>/sequences/<name>.yaml`` names an engine spec of the same target and declares chains:

    engine: <spec stem>
    equivalent: {<name>: [group, ...]} # optional; as in pair files
    ignore: [group, ...]               # optional; as in pair files
    sequences:
      <sequence>:
        ops:                           # a cycle; a chain of length n takes its first n entries, repeating the cycle
          - op: <operation>            # issued at its own command ages + i * gap
            inputs: {input: value}     # optional operand overlay on every command of this entry
            guards: [{signal, bit?}]   # optional; default: the operation's next_issue. Must be clear at its issue age.
        lengths: "a..b"                # chain lengths (at least 2)
        gaps: "a..b"                   # swept spacings (inclusive, from 1)
    expect: {<sequence>: {<length>: <stable_gap>}}   # regression values read only by tests

A spacing is accepted for a chain when the pair rules hold for all operations at once (see ``pairs.py``): no command
ages collide, control stays known and busy drains, every guard is clear at each issue age after the first, and the
event multiset equals the union of the isolated runs, each shifted by its issue offset. A rejection names the missing
or unexpected events and the chain positions that lost events. ``stable_gap`` is, per chain length, the smallest
spacing from which every swept spacing is accepted. Python stays target-agnostic: names and operands live in YAML.
"""
from collections import Counter
import json
from pathlib import Path

import yaml

from . import derive, pairs
from .control import require
from .runner import RECIPES, observed
from .spec import TARGETS, ages, keys, load_target

FILE_KEYS = ({"engine", "sequences"}, {"description", "equivalent", "ignore", "expect"})
SEQUENCE_KEYS = ({"ops", "lengths", "gaps"}, {"description"})
OP_KEYS = ({"op"}, {"inputs", "guards"})
SCHEMA = "rtl_extract.op_sequences.v1"


def validate(raw, spec, where):
    keys(raw, *FILE_KEYS, where)
    require(raw["engine"] == spec["engine"], f"{where}: engine {raw['engine']!r} does not match its spec")
    require(isinstance(raw["sequences"], dict) and raw["sequences"], f"{where}: no sequences")
    equivalent = raw.setdefault("equivalent", {})
    members = [g for groups in equivalent.values() for g in groups]
    require(set(members) <= set(spec["events"]) and len(members) == len(set(members)),
            f"{where}: equivalent must name each event group at most once")
    raw.setdefault("ignore", [])
    require(isinstance(raw["ignore"], list) and set(raw["ignore"]) <= set(spec["events"]) and not set(raw["ignore"]) & set(members),
            f"{where}: ignore must name event groups outside equivalent")
    prefix = f"{spec['command']}_" if spec.get("command") else None
    signals = set(observed(spec))
    for name, sequence in raw["sequences"].items():
        at = f"{where}: sequences.{name}"
        require("/" not in name and "." not in name, f"{at}: invalid sequence name")
        keys(sequence, *SEQUENCE_KEYS, at)
        require(isinstance(sequence["ops"], list) and sequence["ops"], f"{at}: ops must be a non-empty list")
        for n, entry in enumerate(sequence["ops"]):
            keys(entry, *OP_KEYS, f"{at}.ops[{n}]")
            require(entry["op"] in spec["operations"], f"{at}.ops[{n}]: unknown operation {entry['op']!r}")
            overlay = entry.get("inputs", {})
            require(isinstance(overlay, dict) and all(isinstance(v, int) for v in overlay.values()),
                    f"{at}.ops[{n}]: inputs must map inputs to integers")
            undeclared = {i for i in overlay if i not in spec["inputs"] and not (prefix and i.startswith(prefix))}
            require(not undeclared, f"{at}.ops[{n}]: inputs drives undeclared inputs {sorted(undeclared)}")
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
    """The entry as a pair-style ``second`` (operation and optional guards) for ``pairs.guards``."""
    return {"second": entry["op"], **({"guards": entry["guards"]} if "guards" in entry else {})}


def available(target):
    return sorted(p.stem for p in (TARGETS / target / "sequences").glob("*.yaml"))


def load_file(path, spec):
    path = Path(path)
    return validate(yaml.safe_load(path.read_text()), spec, path.name)


def load(target, names=None, specs=None):
    """``{name: (sequence file, engine spec)}`` for the sequence files of ``target``."""
    selected = available(target) if names is None else list(names)
    require(set(selected) <= set(available(target)), f"Unknown sequence files: {sorted(set(selected) - set(available(target)))}")
    raws = {n: yaml.safe_load((TARGETS / target / "sequences" / f"{n}.yaml").read_text()) for n in selected}
    specs = dict(specs or {})
    missing = sorted({r["engine"] for r in raws.values()} - set(specs))
    specs.update(load_target(target, missing) if missing else {})
    return {n: (validate(r, specs[r["engine"]], f"sequences/{n}.yaml"), specs[r["engine"]]) for n, r in raws.items()}


def chain(sequence, length):
    return [sequence["ops"][i % len(sequence["ops"])] for i in range(length)]


def timeline(spec, steps, gap):
    """``(timeline, schedules)`` with operation ``i`` shifted by ``i * gap``; ``timeline`` is ``None`` on a collision."""
    schedules = [pairs.commands(spec, s["op"], s.get("inputs", {}), i * gap) for i, s in enumerate(steps)]
    merged = {}
    for schedule in schedules:
        if set(merged) & set(schedule):
            return None, schedules
        merged.update(schedule)
    return merged, schedules


def lost(expected, actual):
    """Chain positions whose expected events are not all present, in order of the actual events they claim."""
    remaining, positions = Counter(actual), []
    for i, events in enumerate(expected):
        wanted = Counter(events)
        if wanted - remaining:
            positions.append(i)
        remaining -= wanted & remaining
    return positions


def difference(expected, actual):
    """Per-group counts of missing and unexpected events, e.g. ``missing compute x32, acc_write x32``."""
    parts = []
    for label, counter in (("missing", Counter(expected) - Counter(actual)), ("unexpected", Counter(actual) - Counter(expected))):
        groups = Counter(g for (g, _, _), n in counter.items() for _ in range(n))
        if groups:
            parts.append(f"{label} " + ", ".join(f"{g} x{n}" for g, n in sorted(groups.items())))
    return "; ".join(parts)


def trial(circuit, spec, raw, steps, alone, gap):
    """``None`` if the chain is accepted at spacing ``gap``, otherwise the reason it is not."""
    merged, schedules = timeline(spec, steps, gap)
    if merged is None:
        return "command ages collide"
    try:
        trace, problem = pairs.run(circuit, spec, merged)
    except ValueError as error:
        return str(error)
    blocked = [f"guard {g['signal']}" + (f"[{g['bit']}]" if "bit" in g else "") + f" set at op {i}"
               for i, (s, schedule) in enumerate(zip(steps, schedules)) if i
               for g in pairs.guards(spec, step(s)) if not pairs.clear(trace[min(schedule)]["signals"][g["signal"]], g.get("bit"))]
    expected = [[(g, a + i * gap, f) for g, a, f in events] for i, events in enumerate(alone)]
    actual = pairs.events(trace, raw["equivalent"], ignore=raw["ignore"])
    flat = sorted(e for events in expected for e in events)
    changed = None if actual == flat else f"{difference(flat, actual)} (ops {lost(expected, actual) or 'none'})"
    reasons = blocked + [r for r in (problem, changed) if r]
    if blocked and not changed and not problem:
        reasons.append("events unchanged")
    return "; ".join(reasons) or None


def isolated(circuit, spec, raw, entry):
    """The events of one chain entry issued alone."""
    trace, problem = pairs.run(circuit, spec, pairs.commands(spec, entry["op"], entry.get("inputs", {})))
    require(problem is None, f"{entry['op']} alone: {problem}")
    return pairs.events(trace, raw["equivalent"], ignore=raw["ignore"])


def key(entry):
    return json.dumps(entry, sort_keys=True)


CONTEXT = {}


def cell(task):
    """One (case, gap) trial; ``CONTEXT`` is inherited by forked workers."""
    case, gap = task
    context = CONTEXT["context"]
    return trial(context["circuit"], context["spec"], context["raw"], context["cases"][case][1], context["cases"][case][2], gap)


def extract(document, raw, spec, jobs=1, only=None):
    """One record per (sequence, length): per-spacing results, the minimum accepted spacing and the stable spacing."""
    concrete, _ = derive.resolve(document, spec, {})
    circuit = RECIPES[concrete["recipe"]][0](document, concrete)
    selected = [n for n in raw["sequences"] if only is None or n in only]
    cases, alone = [], {}
    for name in selected:
        sequence = raw["sequences"][name]
        for length in ages(sequence["lengths"]):
            steps = chain(sequence, length)
            for s in steps:
                if key(s) not in alone:
                    alone[key(s)] = isolated(circuit, concrete, raw, s)
            cases.append((name, steps, [alone[key(s)] for s in steps], length))
    tasks = [(c, gap) for c, case in enumerate(cases) for gap in ages(raw["sequences"][case[0]]["gaps"])]
    CONTEXT["context"] = {"circuit": circuit, "spec": concrete, "raw": raw, "cases": cases}
    if jobs > 1:
        import multiprocessing
        with multiprocessing.get_context("fork").Pool(jobs) as pool:
            reasons = pool.map(cell, tasks, chunksize=1)
    else:
        reasons = [cell(t) for t in tasks]
    CONTEXT.clear()
    records = []
    for c, (name, steps, _, length) in enumerate(cases):
        results = [(gap, reason) for (case, gap), reason in zip(tasks, reasons) if case == c]
        stable = None
        for gap, reason in reversed(results):
            if reason is not None:
                break
            stable = gap
        records.append({"name": f"{spec['engine']}.{name}/{length}", "engine": spec["engine"], "module": spec["module"],
                        "sequence": name, "length": length,
                        "ops": [{"op": s["op"], **({"inputs": s["inputs"]} if "inputs" in s else {})} for s in steps],
                        "gaps": [results[0][0], results[-1][0]],
                        "min_gap": min((g for g, r in results if r is None), default=None), "stable_gap": stable,
                        "results": pairs.runs(results),
                        "evidence": f"{concrete['recipe']} control simulation of {spec['module']}: {len(results)} spacings, "
                                    "event multisets compared with isolated runs"})
    return records


def main(argv=None):
    import argparse
    from . import facts
    from .ir import load_document
    from .runner import modules
    parser = argparse.ArgumentParser(description="Sweep declared operation chains (see tools/rtl_extract/sequences.py).")
    parser.add_argument("--hw-ir", type=Path, required=True)
    parser.add_argument("--exporter", type=Path, required=True)
    parser.add_argument("--target", default="atlas")
    parser.add_argument("--sequences", nargs="+", help="sequence files (default: every file under targets/<target>/sequences/)")
    parser.add_argument("--only", nargs="+", help="sequence names within the files (default: all)")
    parser.add_argument("--jobs", type=int, default=1, help="worker processes for the (chain, spacing) trials")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    loaded = load(args.target, args.sequences)
    unknown = set(args.only or ()) - {n for raw, _ in loaded.values() for n in raw["sequences"]}
    require(not unknown, f"Unknown sequences: {sorted(unknown)}")
    document = load_document(args.hw_ir, sorted({m for _, s in loaded.values() for m in modules(s)}), args.exporter)
    records = [r for raw, spec in loaded.values() for r in extract(document, raw, spec, args.jobs, args.only and set(args.only))]
    facts.write(args.output, {"schema": SCHEMA, "hw_ir": {"path": str(args.hw_ir.resolve()), "sha256": facts.sha256(args.hw_ir)},
                              "op_sequences": records})
    print(json.dumps({r["name"]: {"min_gap": r["min_gap"], "stable_gap": r["stable_gap"]} for r in records}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
