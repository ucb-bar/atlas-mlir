"""HW IR ingestion; the only module that knows how the IR becomes a JSON document."""
import json
from pathlib import Path
import subprocess

from .control import require


def export(hw_ir, modules, exporter):
    raw = subprocess.check_output([str(Path(exporter).resolve()), str(Path(hw_ir).resolve()), *sorted(set(modules))])
    return json.loads(raw)


def load_document(hw_ir, modules, exporter):
    """Export ``modules`` and every hw.module they instantiate (extern modules stay declarations)."""
    names, document = set(modules), None
    while True:
        document = export(hw_ir, names, exporter)
        require(not set(document["missing_modules"]) & set(modules),
                f"Requested modules absent from hardware IR: {sorted(set(document['missing_modules']) & set(modules))}")
        available = set(document["available_modules"])
        children = {op["attributes"]["moduleName"].lstrip("@") for m in document["modules"]
                    for op in m["operations"] if op["kind"] == "hw.instance"}
        missing = (children & available) - names
        if not missing:
            return document
        names |= missing
