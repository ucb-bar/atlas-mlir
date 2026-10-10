#!/usr/bin/env python3
"""Compute engine operation timing by control simulation of CIRCT HW IR (see tools/rtl_extract/README.md)."""
import argparse
import json
from pathlib import Path

from rtl_extract import facts
from rtl_extract.ir import load_document
from rtl_extract.runner import extract, modules
from rtl_extract.spec import available_engines, load_target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hw-ir", type=Path, required=True, help="CIRCT HW dialect IR (.mlir)")
    parser.add_argument("--exporter", type=Path, required=True, help="hw_ir_export built from tools/rtl_extract/export")
    parser.add_argument("--target", default="atlas")
    parser.add_argument("--engines", nargs="+", help="default: every spec under targets/<target>/")
    parser.add_argument("--output", type=Path, required=True, help="op_timing JSON")
    args = parser.parse_args()
    specs = load_target(args.target, args.engines or available_engines(args.target))
    document = load_document(args.hw_ir, sorted({m for s in specs.values() for m in modules(s)}), args.exporter)
    records = [r for spec in specs.values() for r in extract(document, spec)]
    facts.write(args.output, facts.document(args.target, args.hw_ir, args.exporter, records))
    print(json.dumps({r["name"]: {"status": r["status"], "first_free_age": r["first_free_age"],
                                  **{g: e["first_age"] for g, e in (r["events"] or {}).items()}} for r in records}, indent=1))
    return 0 if all(r["status"] == "computed" for r in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
