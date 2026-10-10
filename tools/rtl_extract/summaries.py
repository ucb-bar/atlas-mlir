"""Generic summaries of per-age traces: event streams, release and next-issue ages."""


def stream(ages):
    ages = sorted(ages)
    steps = {b - a for a, b in zip(ages, ages[1:])}
    return {"first_age": ages[0] if ages else None, "last_age": ages[-1] if ages else None, "count": len(ages),
            "step": steps.pop() if len(steps) == 1 else None}


def contiguous(rows):
    return rows == list(range(len(rows)))


def group(entries, row=None, split_by=None):
    """Summarize ``[(age, fields)]`` for one event group."""
    summary = stream([age for age, _ in entries])
    names = sorted({k for _, fields in entries for k in fields} - {row})
    summary["values"] = {k: sorted({fields[k] for _, fields in entries}) for k in names}
    if row:
        summary["row_contiguous"] = contiguous([fields[row] for _, fields in entries])
    if split_by:
        streams = []
        for key in sorted({fields[split_by] for _, fields in entries}):
            chunk = []
            for age, fields in entries:
                if fields[split_by] != key:
                    continue
                if row and chunk and fields[row] == 0:
                    streams.append((key, chunk))
                    chunk = []
                chunk.append((age, fields))
            streams.append((key, chunk))
        summary["streams"] = [{split_by: key, **stream([a for a, _ in chunk]),
                               **({"row_contiguous": contiguous([f[row] for _, f in chunk])} if row else {})}
                              for key, chunk in streams]
    return summary


def events(trace, groups):
    return {name: group([(e["age"], e["events"][name]) for e in trace if name in e["events"]],
                        spec.get("row"), spec.get("split_by")) for name, spec in groups.items()}


def first_free_age(trace, start=0):
    """First age after ``start`` at which the busy signal is low."""
    return next((e["age"] for e in trace if e["age"] > start and e["busy"] == 0), None)


def busy_contiguous(trace, start, free):
    """Busy is high on every age in (start, free) and low from ``free`` to the end of the trace."""
    return all(bool(e["busy"]) == (start < e["age"] < free) for e in trace if e["age"] > start)


def next_issue_age(trace, start, signal, bit=None):
    """First age after ``start`` at which ``signal`` (or its ``bit``) is clear."""
    def clear(value):
        return value is not None and not (value >> bit & 1 if bit is not None else value)
    return next((e["age"] for e in trace if e["age"] > start and clear(e["signals"][signal])), None)
