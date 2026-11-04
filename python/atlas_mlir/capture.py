"""Capture a PyTorch model as the Linalg module, argument manifest and weights the importer reads.

This does what model2mlir's fx_importer backend does for the importer: torch.export, then
torch-mlir's FX importer with parameters kept as function arguments, lowered to
linalg-on-tensors.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

FUNCTION = "forward"


class CaptureError(ValueError):
    """The model cannot be loaded or exported."""


@dataclass(frozen=True)
class Capture:
    linalg: Path
    arguments: Path
    weights: Path
    inputs: Path | None  # Example inputs by graph input name, for the host program.


def _torch() -> tuple[Any, Any]:
    try:
        import torch
        from torch_mlir import fx
    except ImportError as error:
        raise CaptureError(
            f"capturing a PyTorch model needs torch and torch-mlir ({error}); "
            "install atlas-mlir[capture] in a supported Python environment"
        ) from error
    return torch, fx


def load(source: Path) -> tuple[Any, tuple[Any, ...]]:
    """A model and its example inputs from a loader .py or a torch.export .pt2 file."""
    torch, _ = _torch()
    if not source.is_file():
        raise CaptureError(f"model source does not exist: {source}")
    if source.suffix == ".pt2":
        program = torch.export.load(str(source))
        example = program.example_inputs
        if not example or example[1]:
            raise CaptureError(f"{source} has no positional example inputs; "
                               "export it with torch.export.export(model, args) and save that")
        return program, tuple(example[0])
    if source.suffix != ".py":
        raise CaptureError(f"{source}: expected a loader .py or a .pt2 file")
    spec = importlib.util.spec_from_file_location(f"atlas_model_{source.stem}", source)
    module = importlib.util.module_from_spec(spec)
    # Dataclasses and other runtime annotation helpers consult sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    loader = getattr(module, "get_model_and_inputs", None)
    if loader is None:
        raise CaptureError(f"{source} defines no get_model_and_inputs() returning (model, inputs)")
    model, inputs = loader()
    return model, tuple(inputs)


def capture(model: Any, inputs: tuple[Any, ...], out: Path) -> Capture:
    torch, fx = _torch()
    from safetensors.numpy import save_file

    if isinstance(model, torch.export.ExportedProgram):
        program = model
    else:
        program = torch.export.export(model.eval(), tuple(inputs))
    # No decomposition table: torch-mlir then lowers aten.linear to a zero-filled
    # matmul and one bias-add generic, the forms the importer admits.
    parameters = {**program.state_dict, **program.constants}
    arguments: dict[str, dict[str, Any]] = {}
    weights: dict[str, Any] = {}
    names = []
    for index, spec in enumerate(program.graph_signature.input_specs):
        if spec.kind.name == "USER_INPUT":
            arguments[str(index)] = {"kind": "input", "name": spec.arg.name}
            names.append(spec.arg.name)
            continue
        if spec.kind.name not in ("PARAMETER", "BUFFER", "CONSTANT_TENSOR"):
            raise CaptureError(f"graph input {spec.arg.name} has unsupported kind {spec.kind.name}")
        tensor = parameters[spec.target].detach().contiguous()
        if tensor.dtype != torch.float32:
            raise CaptureError(f"parameter {spec.target} is {tensor.dtype}; the importer reads f32")
        weights[spec.target] = tensor.numpy()
        arguments[str(index)] = {"weight": spec.target, "kind": "param",
                                 "dtype": "float32", "shape": list(tensor.shape)}
    if len(names) != len(inputs) or not all(isinstance(item, torch.Tensor) for item in inputs):
        raise CaptureError("example inputs must be one tensor per graph input")
    module = fx.stateless_fx_import(program.graph_module, output_type="linalg-on-tensors",
                                    model_name=FUNCTION)
    # The importer checks this against the weights it is given.
    with module.context:
        module.operation.attributes["prov.weights_file"] = fx.ir.StringAttr.get("weights.safetensors")

    out.mkdir(parents=True)
    result = Capture(out / "linalg.mlir", out / "arguments.json",
                     out / "weights.safetensors", out / "inputs.safetensors")
    result.linalg.write_text(str(module))
    result.arguments.write_text(json.dumps(arguments, indent=2) + "\n")
    save_file(weights, str(result.weights))
    save_file({name: tensor.detach().to(torch.float32).contiguous().numpy()
               for name, tensor in zip(names, inputs)}, str(result.inputs))
    return result
