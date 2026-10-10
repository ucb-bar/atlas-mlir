"""Generic spec-driven runs of control circuits and their reduction to op_timing records."""
import json

from . import coupled, derive, facts, summaries
from .control import ControlCircuit, require
from .spec import ages, variant

RUN_ONLY = {"operations", "variants", "checks", "expect", "raw", "only", "limit", "tail",
            "reset_cycles", "flush_cycles", "description"}


def modules(spec):
    """Modules a spec needs: the simulated tops (recipes with several circuits extend this) and derivation sources."""
    return [spec["module"]] + ([spec["partner"]["module"]] if "partner" in spec else []) + derive.modules(spec)


def control_signals(spec):
    """Signals whose values decide timing; they must be known after reset and flush."""
    signals = [spec["busy"]] if spec["busy"] else []
    signals += [e["valid"] for e in spec["events"].values()]
    signals += [op["next_issue"]["signal"] for op in spec["operations"].values() if "next_issue" in op]
    return list(dict.fromkeys(signals))


def observed(spec):
    signals = control_signals(spec) + [s for e in spec["events"].values() for s in e.get("fields", {}).values()]
    return list(dict.fromkeys(signals))


def build_single(document, spec):
    outputs = [s for s in observed(spec) if s not in spec["probes"]]
    inputs = {spec["clock"], spec["reset"], *spec["inputs"]}
    return ControlCircuit(document, spec["module"], outputs, inputs, spec["cuts"], spec["probes"], spec["clock"], spec["reset"], spec["unreset"])


def schedule(operation):
    commands = {}
    for key, command in operation["commands"].items():
        for age in ages(key):
            require(age not in commands, f"Two commands at age {age}")
            commands[age] = command
    return commands


def reset_and_flush(circuit, spec, control):
    idle = {**spec["inputs"], spec["clock"]: 0}
    circuit.state = {key: None for key in circuit.registers}
    for _ in range(spec["reset_cycles"]):
        circuit.cycle({**idle, spec["reset"]: 1})
    for _ in range(spec["flush_cycles"]):
        circuit.cycle({**idle, spec["reset"]: 0})
    cone = circuit.cone(control)
    require(circuit.unreset <= cone, f"{len(circuit.unreset - cone)} unreset registers are not in the control cone")
    known = [key for key in circuit.unreset if circuit.state[key] is not None]
    require(not known, f"{len(known)} unreset registers are known after reset/flush; remove them from unreset")
    unknown = [key for key in cone - circuit.unreset if circuit.state[key] is None]
    require(not unknown, f"{len(unknown)} control registers remain unknown after reset/flush")


def run_single(circuit, spec, name):
    """Drive one operation's commands by age, feed responses back and record per-age events."""
    commands, control = schedule(spec["operations"][name]), control_signals(spec)
    reset_and_flush(circuit, spec, control)
    pending, trace, free = set(), [], None
    for age in range(spec["limit"]):
        values = {**spec["inputs"], spec["clock"]: 0, spec["reset"]: 0, **commands.get(age, {})}
        for group, event in spec["events"].items():
            if "response" in event:
                values[event["response"]["input"]] = int((age, group) in pending)
        out = circuit.cycle(values)
        require(all(out[s] is not None for s in control), f"Control output unknown at age {age}")
        events = {}
        for group, event in spec["events"].items():
            if out[event["valid"]]:
                fields = {k: out[s] for k, s in event.get("fields", {}).items()}
                require(None not in fields.values(), f"Event field unknown while {group} is valid at age {age}")
                events[group] = fields
                if "response" in event:
                    pending.add((age + event["response"]["latency"], group))
        trace.append({"age": age, "busy": out[spec["busy"]] if spec["busy"] else None, "signals": out, "events": events})
        if spec["busy"] and free is None and age > max(commands) and not out[spec["busy"]] and all(a <= age for a, _ in pending):
            free = age
        if free is not None and age >= free + spec["tail"]:
            break
    return trace


RECIPES = {"single": (build_single, run_single),
           "coupled": (lambda document, spec: coupled.build(document, spec, observed(spec)), run_single)}


def summarize(trace, spec, name):
    operation = spec["operations"][name]
    start = min(schedule(operation))
    result = {"events": summaries.events(trace, spec["events"])}
    if spec["busy"]:
        free = summaries.first_free_age(trace, start)
        require(free is not None, f"Busy did not clear within {spec['limit']} ages")
        require(summaries.busy_contiguous(trace, start, free), "Busy interval is not contiguous")
        result["first_free_age"] = free
    if "next_issue" in operation:
        issue = summaries.next_issue_age(trace, start, operation["next_issue"]["signal"], operation["next_issue"].get("bit"))
        require(issue is not None, f"Next issue not reached within {spec['limit']} ages")
        result["next_issue_age"] = issue
    return result


def signature(spec):
    events = {n: {k: v for k, v in e.items() if k != "response"} | ({"response": e["response"]["input"]} if "response" in e else {})
              for n, e in spec["events"].items()}
    return json.dumps({**{k: v for k, v in spec.items() if k not in RUN_ONLY}, "events": events,
                       "inputs": sorted(spec["inputs"]), "control": control_signals(spec)}, sort_keys=True)


def lookup(records, path):
    name, _, fields = path.partition(".")
    require(name in records, f"Check names unknown record {name!r}")
    value = records[name]
    for field in fields.split(".") if fields else []:
        value = value[field] if isinstance(value, dict) and field in value else None
    return value


def apply_checks(spec, records):
    """Spec-level equalities between records; a failure makes every named record unresolved."""
    local = {r["name"].split(".", 1)[1]: r for r in records}
    for check in spec["checks"]:
        names = [p.partition(".")[0] for p in check["equal"]]
        if any(local[n]["status"] != "computed" for n in names if n in local):
            continue
        values = [lookup(local, p) for p in check["equal"]]
        if any(v != values[0] for v in values):
            reason = f"Check failed: {' == '.join(check['equal'])}" + (f" ({check['reason']})" if "reason" in check else "")
            for name in names:
                facts.unresolve(local[name], reason)
    return records


def extract(document, spec):
    """Records for every operation of the base spec and of each variant."""
    records, circuits, derived = [], {}, {}
    for name in [None, *spec["variants"]]:
        current = variant(spec, name)
        build, run = RECIPES[current["recipe"]]
        try:
            current, notes = derive.resolve(document, current, derived)
            key = signature(current)
            if key not in circuits:
                circuits[key] = build(document, current)
            circuit = circuits[key]
        except ValueError as error:
            records += [facts.record(current, op, name, reason=f"Circuit construction failed: {error}")
                        for op in current.get("only", current["operations"])]
            continue
        for op in current.get("only", current["operations"]):
            try:
                trace = run(circuit, current, op)
                result = summarize(trace, current, op)
            except ValueError as error:
                records.append(facts.record(current, op, name, reason=str(error)))
                continue
            registers = len(getattr(circuit, "registers", ()))
            evidence = (f"{current['recipe']} control simulation of {current['module']}: {registers} registers, "
                        f"{len(trace)} ages after {current['reset_cycles']} reset + {current['flush_cycles']} flush cycles")
            records.append(facts.record(current, op, name, result, evidence=evidence))
            records[-1]["assumptions"].update(notes)
    return apply_checks(spec, records)
