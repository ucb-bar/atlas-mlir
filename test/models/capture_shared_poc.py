"""Recapture existing Merlin proofs of concept with explicit synthetic weights.

Model selection belongs only to this test fixture. The production compiler sees
ordinary source IR and never branches on these names. SmolVLA is the retained
reduced prefix/expert capsule; the Llama fixture is Merlin's scaled whole-Llama
recipe. ResNet-50 uses the complete torchvision architecture at the chosen input
resolution. None of these fixtures establishes pretrained application quality.
"""

import argparse
import importlib.metadata
import importlib.util
import json
import subprocess
import sys
from hashlib import sha256
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file


def capture(model_name, destination, merlin_root, image_size=224):
    import m2m
    from merlin.frontends.linalg_text import parse_mlir_text

    destination.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(0)
    torch.set_num_threads(1)
    if model_name == "resnet50":
        from torchvision.models import resnet50

        model = resnet50(weights=None).eval()
        inputs = (torch.randn(1, 3, image_size, image_size),)
        source_file = Path(sys.modules[resnet50.__module__].__file__)
        provenance = {"torchvision_version": importlib.metadata.version("torchvision")}
    else:
        recipe = {
            "smolvla": "model/M4_smolvla_denoise_gemmini",
            "tinyllama_poc": "radiance/model/M0_small_llama_fp32",
        }[model_name]
        source_file = (
            merlin_root / "merlin/contract/capsules" / recipe / "capsule.pytorch.py"
        )
        spec = importlib.util.spec_from_file_location("poc_model", source_file)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        model, inputs = module.get_model_and_inputs()
        provenance = {
            "merlin_revision": subprocess.check_output(
                ["git", "-C", str(merlin_root), "rev-parse", "HEAD"], text=True
            ).strip(),
            "recipe": recipe,
        }
    result = m2m.convert(
        model,
        tuple(inputs),
        backend="fx_importer",
        quantization=None,
        level="linalg-on-tensors",
        weights_path=str(destination / "weights.safetensors"),
    )
    if not result.ok:
        raise ValueError(result.diagnostics)
    (destination / "model.mlir").write_text(result.mlir_text)
    parse_mlir_text(result.mlir_text).verify()
    with torch.no_grad():
        expected = model(*inputs)
    if not isinstance(expected, (tuple, list)):
        expected = (expected,)
    np.savez(
        destination / "torch-f32.npz",
        **{f"output{i}": v.detach().numpy() for i, v in enumerate(expected)},
    )
    manifest = json.loads(
        (destination / "weights.safetensors.manifest.json").read_text()
    )
    weights = load_file(str(destination / "weights.safetensors"))
    flat = []
    user = iter(inputs)
    for i in range(len(manifest)):
        row = manifest[str(i)]
        if row.get("stub") or "error" in row:
            raise ValueError("incomplete parameter capture")
        value = (
            weights[row["weight"]] if row["kind"] in ("param", "buffer") else next(user)
        )
        flat.append(value.detach().numpy())
    if next(user, None) is not None:
        raise ValueError("unbound user input")
    np.savez(destination / "inputs.npz", **{f"arg{i}": v for i, v in enumerate(flat)})
    record = {
        "model": model_name,
        "weights": "synthetic",
        "seed": 0,
        "input_shapes": [list(x.shape) for x in inputs],
        "recipe_sha256": sha256(source_file.read_bytes()).hexdigest(),
        "torch_version": torch.__version__,
        "source_sha256": sha256(result.mlir_text.encode()).hexdigest(),
        **provenance,
    }
    (destination / "capture.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=["tinyllama_poc", "smolvla", "resnet50"])
    parser.add_argument("--merlin-root", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()
    capture(
        args.model,
        args.destination.resolve(),
        args.merlin_root.resolve(),
        args.image_size,
    )
