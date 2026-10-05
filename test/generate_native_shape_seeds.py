"""Generate the public E3 MXU0 shape requests without runtime inputs or goldens.

These requests fix the source indexing and numerical contract before compilation.
The independent execution panels live in ``qualify_native_shapes.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from merlin.semantic_compiler.model import IndexMap, KernelRequest, SemanticNode, TensorType

from atlas_native_support.mxu0 import Mxu0Contract


SHAPES = (
    (32, 32, 32), (64, 32, 96), (32, 64, 32), (64, 64, 64),
    (31, 32, 33), (33, 31, 65), (1, 65, 7), (7, 1, 33),
)


def make_case(shape: tuple[int, int, int], contract: Mxu0Contract) -> tuple[KernelRequest, dict]:
    m, k, n = shape
    rows = contract.record["geometry"]["array_rows"]
    columns = contract.record["geometry"]["array_cols"]
    if rows != columns:
        raise ValueError("seed ABI requires separately planned rectangular geometry")
    numerical = contract.record["numerical_policy"]
    operand = numerical["operand_dtype"]
    operand_policy = numerical["operand_policy"]
    result = numerical["result_dtype"]
    result_policy = numerical["result_policy"]
    maps = (
        IndexMap(3, ((1, 0, 0), (0, 0, 1)), (0, 0)),
        IndexMap(3, ((0, 1, 0), (0, 0, 1)), (0, 0)),
        IndexMap(3, ((1, 0, 0), (0, 1, 0)), (0, 0)),
    )
    input_storage = (
        "external_fp8" if (m, k, n) == (rows, columns, columns)
        else "external_fp8_row_major"
    )
    request = KernelRequest(
        nodes=(
            SemanticNode("activation", "input", (), TensorType((m, k), operand, operand_policy), effect="input"),
            SemanticNode("weight", "input", (), TensorType((n, k), operand, operand_policy), effect="input"),
            SemanticNode(
                "result", "contraction", ("activation", "weight"),
                TensorType((m, n), result, result_policy),
                attrs=(("reduction_order", "k_ascending"),), index_maps=maps,
            ),
        ),
        outputs=("result",), output_storages=("external_bf16",),
        input_storages=(("activation", input_storage), ("weight", input_storage)),
        target_identity=contract.identity,
        source_identity=f"public-shape-{m}-{k}-{n}",
    )
    activation_panels = ((m + rows - 1) // rows) * ((k + columns - 1) // columns)
    weight_panels = ((n + rows - 1) // rows) * ((k + columns - 1) // columns)
    abi = {
        "fixed_inputs": {"activation": 0, "weight": activation_panels},
        "fixed_outputs": [max(8, activation_panels + weight_panels)],
    }
    return request, abi


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("output path must be fresh")
    contract = Mxu0Contract.load()
    args.out.mkdir(parents=True)
    for shape in SHAPES:
        request, abi = make_case(shape, contract)
        directory = args.out / "x".join(map(str, shape))
        directory.mkdir()
        (directory / "request.json").write_text(json.dumps(request.record(), indent=2, sort_keys=True) + "\n")
        (directory / "abi.json").write_text(json.dumps(abi, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"cases": len(SHAPES), "out": str(args.out)}, sort_keys=True))


if __name__ == "__main__":
    main()
