"""Diagnostic model compilation using shared outlining and ordinary host passes.

Every recognized f32 matmul/batch_matmul goes through an explicitly selected
E4M3-input/BF16-accumulator policy. Other operations retain ordinary f32 host
lowering. Programs cover one output tile and the complete padded K reduction;
host dispatch varies batch/M/N without reassociating K or splitting its state.
This is synchronous host/device partitioning, not all-device model execution.
"""

import json
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

from merlin.frontends.linalg_text import parse_mlir_text
from merlin.llvmlower.floating_matmul_outline import (
    compile_callback_host,
    outline_floating_matmuls,
)
from merlin.transforms.ordered_float_dense import OrderedFloatDense
from xdsl.dialects.builtin import TensorType, f32

from .shared_counted_dense import counted_dense_program
from .shared_dense import (
    INPUT_BASE,
    MODEL_REVISION,
    POLICY,
    SLOT_BYTES,
    compile_virtual,
    schedule_dense,
)

PRECISION = "f32-e4m3-rne-saturate-flush-ordered-bf16-padded-v1"


def compile_model(
    source,
    *,
    entry,
    precision,
    destination,
    atlas_bin_dir,
    llc,
    cc,
    pipeline="source-order",
):
    if precision != PRECISION:
        raise ValueError("explicit diagnostic model precision required")
    destination = Path(destination)
    host, calls = outline_floating_matmuls(
        source, entry=entry, numerical_policy=precision
    )
    if not calls:
        raise ValueError("model has no supported contractions")
    module = parse_mlir_text(source)
    fn = module.body.block.first_op
    if any(
        type(t) is not TensorType
        or t.element_type != f32
        or any(d <= 0 for d in t.get_shape())
        for t in (*fn.function_type.inputs, *fn.function_type.outputs)
    ):
        raise ValueError(
            "model transport requires positive static f32 tensor inputs and outputs"
        )
    signature = {
        "inputs": [list(t.get_shape()) for t in fn.function_type.inputs],
        "outputs": [list(t.get_shape()) for t in fn.function_type.outputs],
    }
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "source.mlir").write_text(source)
    (destination / "host.mlir").write_text(host)
    kernels = {}
    for call in calls:
        k = call["input_shapes"][0][-1]
        padded_k = (k + 31) // 32 * 32
        if padded_k > 8192:
            raise ValueError("diagnostic K expansion budget exceeded")
        key = str(padded_k)
        if key not in kernels:
            selected = OrderedFloatDense(
                sha256(source.encode()).hexdigest(),
                "tile",
                32,
                padded_k,
                32,
                False,
                POLICY,
            )
            schedule = schedule_dense(selected, max_tile_products=256)
            output_base = INPUT_BASE + (2 * (padded_k // 32) + 1) * SLOT_BYTES
            virtual = counted_dense_program(selected, schedule, output_base=output_base)
            manifest = compile_virtual(
                call["source"],
                virtual,
                destination=destination / key,
                atlas_bin_dir=atlas_bin_dir,
                pipeline=pipeline,
            )
            (destination / key / "schedule.mk").write_text(schedule.kernel.text())
            kernels[key] = {
                "selected": asdict(selected),
                "output_base": output_base,
                "words": manifest["words"],
                "kernel_sha256": schedule.kernel.digest(),
                "artifacts": {
                    name: sha256((destination / key / name).read_bytes()).hexdigest()
                    for name in (
                        "manifest.json",
                        "program.bin",
                        "program.json",
                        "source.mlir",
                        "virtual.mlir",
                        "machine.mlir",
                        "schedule.mk",
                    )
                },
            }
        call["kernel"] = key
    compile_callback_host(host, calls, destination=destination, llc=llc, cc=cc)
    result = {
        "schema": "atlas.shared-model-diagnostic.v1",
        "entry": entry,
        "precision": precision,
        "source_sha256": sha256(source.encode()).hexdigest(),
        "calls": calls,
        "kernels": kernels,
        "signature": signature,
        "pipeline": pipeline,
        "model_revision": MODEL_REVISION,
        "host_tools": {
            name: sha256(Path(path).read_bytes()).hexdigest()
            for name, path in (("llc", llc), ("cc", cc))
        },
        "scope": "compiled f32 host graph with synchronous approximate device contractions",
        "numeric_obligations": [
            "finite contraction inputs",
            "application quality unqualified",
            "zero-padded K order and boundary quantization are explicit policy choices",
        ],
        "artifacts": {
            name: sha256((destination / name).read_bytes()).hexdigest()
            for name in (
                "source.mlir",
                "host.mlir",
                "host.ll",
                "host.o",
                "dispatch.c",
                "mlir_runtime.c",
                "model.so",
            )
        },
    }
    (destination / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--entry", required=True)
    parser.add_argument("--precision", required=True, choices=[PRECISION])
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--atlas-bin-dir", type=Path, required=True)
    parser.add_argument("--llc", type=Path, required=True)
    parser.add_argument("--cc", type=Path, required=True)
    parser.add_argument(
        "--pipeline", choices=["source-order", "optimized"], default="source-order"
    )
    args = parser.parse_args()
    print(
        json.dumps(
            compile_model(
                args.source.read_text(),
                entry=args.entry,
                precision=args.precision,
                destination=args.destination,
                atlas_bin_dir=args.atlas_bin_dir,
                llc=args.llc,
                cc=args.cc,
                pipeline=args.pipeline,
            ),
            indent=2,
        )
    )
