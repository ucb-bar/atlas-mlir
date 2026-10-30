"""Explicit entry point for offline precision-mode comparisons."""

import argparse
import importlib
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "precision_modes"


def parser(model_name=None):
    result = argparse.ArgumentParser(description=__doc__)
    models = result.add_subparsers(dest="model", required=True)
    for name in ("tiny", "smol", "resnet"):
        model = models.add_parser(name)
        model.add_argument("--precision-mode", default="default", choices=(
            "default", "finer-scales", "k-split", "two-term"))
        model.add_argument("--k-chunk", type=int, default=128,
                           help="K split size, a positive multiple of 32")
        model.add_argument("--finer-scales", action="store_true",
                           help="Combine finer scales with another recipe")
        model.add_argument("--activation-scale",
                           choices=("tensor", "row"))
        model.add_argument("--weight-scale",
                           choices=("tensor", "column"))
        model.add_argument("--sites", default="all",
                           help="Comma-separated site names/globs, @file, or "
                                "Tiny mask head, attention-head, mlp, or down")
        model.add_argument("--output", type=Path, help="Write JSON metrics")
        model.add_argument("--output-dir", type=Path,
                           help="Also save model output arrays")
        if name == model_name:
            driver = importlib.import_module(
                f".models.{name}", __package__)
            driver.add_arguments(model)
    return result


def main(argv=None):
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument("model", nargs="?",
                          choices=("tiny", "smol", "resnet"))
    selected, _ = selector.parse_known_args(argv)
    command = parser(selected.model)
    args = command.parse_args(argv)
    if args.k_chunk <= 0 or args.k_chunk % 32:
        command.error("--k-chunk must be a positive multiple of 32")
    if args.precision_mode == "default" and (
            args.finer_scales or args.activation_scale or args.weight_scale or
            args.sites != "all"):
        command.error("scale and site overrides require an opt-in mode")
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    driver = importlib.import_module(f".models.{args.model}", __package__)
    with torch.inference_mode():
        report = driver.run(args)
    report["recipe"] = {
        name: getattr(args, name) for name in (
            "precision_mode", "k_chunk", "finer_scales",
            "activation_scale", "weight_scale", "sites")
    }
    payload = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
