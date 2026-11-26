"""Prepare pinned exporter imports, then delegate capture entirely to Merlin."""
import json
from pathlib import Path
import sys

from .provenance import environment_record


def main():
    record = environment_record(capture=True)
    if "--quant-adapter" in sys.argv:
        from .torchao import activate
        record["atlas-torchao"] = activate()
    from merlin.capture.preparation.worker import main as capture
    status = capture(sys.argv[1:])
    if status == 0:
        out = Path(sys.argv[sys.argv.index("--out") + 1])
        (out / "atlascc-environment.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
