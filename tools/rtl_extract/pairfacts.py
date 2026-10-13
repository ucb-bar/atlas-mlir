"""Pair facts: the pair and chain sweep results a compiler may select, as a compact ``merlin.op_pairs.v1`` document.

Inputs are an ``op_timing`` document (``tools/extract-rtl-timing.py``), pair sweeps (``rtl_extract.op_pairs.v1``) and
optional chain sweeps (``rtl_extract.op_sequences.v1``) of one hardware IR. Each sweep record is joined with the pair or
sequence file that produced it (``pairs-<file>.json`` names the file, else the record's engine) for the names its sides
stand for (``as``, default ``<engine>.<op>``) and the operand relation its overlays satisfy (``relation``). Every
record must check assertions and name resolved ``op_timing`` blocks.

    {schema: "merlin.op_pairs.v1", hw_ir: {path, sha256}, op_timing: {path, sha256},
     sources: [{path, sha256, schema}],
     op_pairs: [{first, second, engine, relation, gaps, min_gap, stable_gap, assertions: {checked, excluded}, source}],
     capacity: [{engine, live, window, witness: {source, gap}}]}

``first``/``second`` are ``op_timing`` block names and ``engine`` their common engine (``null`` across engines). A
record applies when the operands satisfy ``relation``; the second may then issue at any gap from ``stable_gap`` through
``gaps[1]``. ``capacity`` is, per engine, the most operations observed live at once in any accepted run of the inputs
(pairs at an accepted gap, chains at an accepted spacing); it is what the runs show, never more. ``window`` names when
an operation is live: from its issue until its ``next_issue_age`` where its engine's blocks have one
(``next_issue_age``), otherwise from its issue through the last age of any of its events (``last_event_age``). Every
block of one engine must take the same window. Per-gap results stay in the sweep files.
"""
import json
from pathlib import Path

from . import pairs, sequences
from .control import require
from .facts import sha256
from .spec import ages

SCHEMA = "merlin.op_pairs.v1"


def window(block):
    """``(name, length)``: an operation is live for ``length`` ages from its issue."""
    if block.get("next_issue_age") is not None:
        return "next_issue_age", block["next_issue_age"]
    return "last_event_age", 1 + max(g["last_age"] for g in block["events"].values() if g["count"])


def accepted(record):
    """The accepted gaps (or spacings) of a sweep record."""
    return [g for run in record["results"] if run["result"] == "accepted" for g in range(run["from"], run["to"] + 1)]


def assertions(record):
    checked = record.get("invariants")
    require(checked and checked["checked"] > 0, f"{record['name']}: pair facts need assertion-checked sweeps")
    return {"checked": checked["checked"], "excluded": len(checked["excluded"])}


def stem(path, prefix, available, engine):
    name = Path(path).stem.removeprefix(prefix)
    return name if name in available else engine


class Builder:
    def __init__(self, op_timing, target):
        self.blocks = {b["name"]: b for b in op_timing["op_timing"]}
        self.hw_ir = op_timing["hw_ir"]
        self.target = target
        self.records, self.capacity, self.keys, self.windows = [], {}, set(), {}

    def block(self, name, where):
        found = self.blocks.get(name)
        require(found is not None and found["events"] is not None, f"{where}: {name!r} is not a resolved op_timing block")
        return found

    def same_ir(self, document, path):
        require(document["hw_ir"]["sha256"] == self.hw_ir["sha256"], f"{path}: describes a different hardware IR than the op_timing facts")

    def window(self, block):
        """The block's window, which every resolved block of its engine must share."""
        engine = block["engine"]
        if engine not in self.windows:
            kinds = {window(b)[0] for b in self.blocks.values() if b["engine"] == engine and b["events"] is not None}
            require(len(kinds) == 1, f"blocks of engine {engine!r} take different live windows: {sorted(kinds)}")
            self.windows[engine] = kinds.pop()
        return window(block)

    def observe(self, issues, source, gap):
        """Counts the live operations of one accepted run: ``issues`` lists ``(block, issue age)``."""
        windows, names = {}, {}
        for block, at in issues:
            names[block["engine"]], length = self.window(block)
            windows.setdefault(block["engine"], []).append((at, at + length))
        for engine, spans in windows.items():
            live = max(sum(start <= at < end for start, end in spans) for at, _ in spans)
            if live > self.capacity.get(engine, {"live": 0})["live"]:
                self.capacity[engine] = {"engine": engine, "live": live, "window": names[engine],
                                         "witness": {"source": source, "gap": gap}}

    def pair_sweep(self, path, document):
        require(document.get("schema") == pairs.SCHEMA, f"{path}: expected schema {pairs.SCHEMA}")
        self.same_ir(document, path)
        available = pairs.available(self.target)
        loaded = {}
        for record in document["op_pairs"]:
            name = stem(path, "pairs-", available, record["engine"])
            if name not in loaded:
                loaded.update(pairs.load(self.target, [name]))
            raw, spec = loaded[name]
            pair = raw["pairs"].get(record["name"].split(".", 1)[1])
            require(pair is not None and (pair["first"], pair["second"]) == (record["first"], record["second"])
                    and [ages(pair["gaps"])[0], ages(pair["gaps"])[-1]] == record["gaps"],
                    f"{record['name']}: does not match pairs/{name}.yaml")
            first, second = (pair.get("as", {}).get(side, f"{spec['engine']}.{pair[side]}") for side in ("first", "second"))
            a, b = self.block(first, record["name"]), self.block(second, record["name"])
            relation = {**raw.get("relation", {}), **pair.get("relation", {})}
            key = (first, second, json.dumps(relation, sort_keys=True))
            require(key not in self.keys, f"{record['name']}: a second record for {first} -> {second} under {relation}")
            self.keys.add(key)
            self.records.append({"first": first, "second": second, "engine": a["engine"] if a["engine"] == b["engine"] else None,
                                 "relation": relation, "gaps": record["gaps"], "min_gap": record["min_gap"],
                                 "stable_gap": record["stable_gap"], "assertions": assertions(record), "source": record["name"]})
            for gap in accepted(record):
                self.observe([(a, 0), (b, gap)], record["name"], gap)

    def sequence_sweep(self, path, document):
        require(document.get("schema") == sequences.SCHEMA, f"{path}: expected schema {sequences.SCHEMA}")
        self.same_ir(document, path)
        available = sequences.available(self.target)
        loaded = {}
        for record in document["op_sequences"]:
            assertions(record)
            name = stem(path, "sequences-", available, record["engine"])
            if name not in loaded:
                loaded.update(sequences.load(self.target, [name]))
            raw, spec = loaded[name]
            declared = raw["sequences"].get(record["sequence"])
            steps = sequences.chain(declared, record["length"]) if declared else None
            require(steps and [{"op": s["op"], **({"inputs": s["inputs"]} if "inputs" in s else {})} for s in steps] == record["ops"],
                    f"{record['name']}: does not match sequences/{name}.yaml")
            chain = [self.block(s.get("as", f"{spec['engine']}.{s['op']}"), record["name"]) for s in steps]
            for gap in accepted(record):
                self.observe([(block, i * gap) for i, block in enumerate(chain)], record["name"], gap)

    def document(self, op_timing_path, sources):
        return {"schema": SCHEMA, "hw_ir": self.hw_ir,
                "op_timing": {"path": str(Path(op_timing_path).resolve()), "sha256": sha256(op_timing_path)},
                "sources": sources, "op_pairs": self.records,
                "capacity": [self.capacity[e] for e in sorted(self.capacity)]}


def build(op_timing_path, pair_paths, sequence_paths=(), target="atlas"):
    builder = Builder(json.loads(Path(op_timing_path).read_text()), target)
    sources = []
    for paths, add in ((pair_paths, builder.pair_sweep), (sequence_paths, builder.sequence_sweep)):
        for path in paths:
            document = json.loads(Path(path).read_text())
            add(path, document)
            sources.append({"path": str(Path(path).resolve()), "sha256": sha256(path), "schema": document["schema"]})
    return builder.document(op_timing_path, sources)


def write(path, document):
    """One line per list element, so the document stays small and diffable."""
    line = lambda value: json.dumps(value, separators=(", ", ": "))
    parts = [f'  "{k}": ' + ("[\n" + ",\n".join("    " + line(v) for v in value) + "\n  ]" if isinstance(value, list) and value
                            else line(value)) for k, value in document.items()]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("{\n" + ",\n".join(parts) + "\n}\n")


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="Collect pair and chain sweep results as pair facts (see tools/rtl_extract/pairfacts.py).")
    parser.add_argument("--op-timing", type=Path, required=True, help="merlin.op_timing.v1 facts of the same hardware IR")
    parser.add_argument("--pairs", type=Path, nargs="+", required=True, help="rtl_extract.op_pairs.v1 sweep files")
    parser.add_argument("--sequences", type=Path, nargs="*", default=[], help="rtl_extract.op_sequences.v1 sweep files")
    parser.add_argument("--target", default="atlas")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    document = build(args.op_timing, args.pairs, args.sequences, args.target)
    write(args.output, document)
    print(json.dumps({"op_pairs": len(document["op_pairs"]), "capacity": document["capacity"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
