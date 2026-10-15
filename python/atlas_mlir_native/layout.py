"""Checked logical view of one selected Atlas BF16 output tile."""

from __future__ import annotations

from collections.abc import Mapping


def unpack_bf16_pair_halves(physical: bytes) -> bytes:
    """Return a 32×32 row-major BF16 tile from the two physical 16-column halves.

    The caller must use this boundary conversion before Merlin's row-major
    M/N tile assembler. It does not change BF16 bits or discard tail lanes.
    """
    if type(physical) is not bytes or len(physical) != 2048:
        raise ValueError("Atlas BF16 tile must contain 2048 physical bytes")
    logical = bytearray(2048)
    for row in range(32):
        for column in range(32):
            source = (column // 16) * 1024 + (row * 16 + column % 16) * 2
            destination = (row * 32 + column) * 2
            logical[destination:destination + 2] = physical[source:source + 2]
    return bytes(logical)


def pack_bf16_pair_halves(logical: bytes) -> bytes:
    """Place a row-major 32×32 BF16 constant into the selected register-pair layout."""
    if type(logical) is not bytes or len(logical) != 2048:
        raise ValueError("Atlas BF16 tile must contain 2048 logical bytes")
    physical = bytearray(2048)
    for row in range(32):
        for column in range(32):
            source = (row * 32 + column) * 2
            destination = (column // 16) * 1024 + (row * 16 + column % 16) * 2
            physical[destination:destination + 2] = logical[source:source + 2]
    return bytes(physical)


def unpack_native_bf16_output(plan: Mapping[str, object], physical: bytes,
                              output_index: int = 0) -> bytes:
    """Decode a native output only when its recorded physical layout agrees."""
    addresses = plan.get("output_addresses")
    layouts = plan.get("output_physical_layouts")
    if (plan.get("schema") != "atlas.native_tile_execution_plan.v8"
            or plan.get("shape") != [32, 32]
            or plan.get("dtype") != "bf16"
            or plan.get("output_bytes") != 2048
            or not isinstance(addresses, list) or not addresses
            or not isinstance(layouts, list) or len(layouts) != len(addresses)
            or type(output_index) is not int or not 0 <= output_index < len(layouts)
            or layouts[output_index] != "bf16_pair_halves"):
        raise ValueError("native output is not a declared BF16 register-pair tile")
    return unpack_bf16_pair_halves(physical)
