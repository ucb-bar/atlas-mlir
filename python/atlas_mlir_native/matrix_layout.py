"""Device byte transport between Atlas matrix and dense BF16 storage.

This is a physical encoding conversion, not a source reshape or a numerical
cast. Merlin retains ownership of source edges and allocation. The caller
connects the returned stage only after proving that both ports represent the
same logical matrix, dtype and numerical policy.
"""

from .binding import io_memory_base as selected_io_base, dram_slots, TILE_BYTES
from .layout_device import _segment, virtual_transfers

DENSE_BF16 = "dense_bf16"
MATRIX_BF16 = "mn_tiles_bf16_pair_halves"


def bf16_matrix_transition(*, rows, columns, source_address, destination_address,
                           source_layout, destination_layout, io_memory_base=None,
                           size_limit=None):
    """Emit an exact on-device permutation of complete 32x32 BF16 tiles.

    Each DMA run copies one 16-column, 32-byte row half. Every element is read
    and written once. No value is decoded, rounded or canonicalized; NaN payloads
    and signed zeros are just bytes. Ragged matrices require a separate padding
    contract and are deliberately refused here. This stage has no host work.
    """
    if any(type(n) is not int or n <= 0 or n % 32 for n in (rows, columns)):
        raise ValueError("matrix layout transfer requires complete positive 32x32 tiles")
    if (source_layout, destination_layout) not in (
        (DENSE_BF16, MATRIX_BF16), (MATRIX_BF16, DENSE_BF16)
    ):
        raise ValueError("matrix layout transfer requires opposite dense/pair-half layouts")

    from .binding import validate_explicit_io_base, validate_io_window

    base = selected_io_base() if io_memory_base is None else io_memory_base
    validate_explicit_io_base(base)
    capacity = dram_slots() * TILE_BYTES
    validate_io_window(base, capacity)
    size = rows * columns * 2
    for address in (source_address, destination_address):
        if (type(address) is not int or address % 32 or address < base
                or address + size > base + capacity):
            raise ValueError("matrix layout buffer exceeds the aligned selected I/O window")
    if max(source_address, destination_address) < min(source_address, destination_address) + size:
        raise ValueError("matrix layout input and output buffers must not overlap")

    # Coordinates: tile row, tile column, register half, row, column-in-half.
    # The last 16 BF16 codes are contiguous in both encodings and form one DMA.
    shape = (rows // 32, columns // 32, 2, 32, 16)
    dense = (32 * columns * 2, 64, 32, columns * 2, 2)
    matrix = ((columns // 32) * 2048, 2048, 1024, 32, 2)
    source, destination = (dense, matrix) if source_layout == DENSE_BF16 else (matrix, dense)
    segment = _segment(0, shape, source, destination, 0, 0, 2)
    return virtual_transfers((segment,), input_addresses=(source_address,),
        output_address=destination_address, output_bytes=size, io_memory_base=base,
        size_limit=size_limit)
