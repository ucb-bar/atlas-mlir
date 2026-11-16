"""Atlas tile byte layout and finite source-domain guards, without compiler imports."""

SLOT_BYTES = 2048


def pack_inputs(selected, lhs: bytes, rhs: bytes):
    """Enforce the source domain and map row-major inputs to Atlas tile slots."""
    if len(lhs) != selected.m * selected.k or len(rhs) != selected.k * selected.n:
        raise ValueError("input byte extent differs from source")
    if any(v and (v & 0x78 == 0 or v & 0x7F == 0x7F) for v in lhs + rhs):
        raise ValueError("input outside finite-normal E4M3 or positive-zero domain")
    payloads = []
    for m in range(0, selected.m, 32):
        for k in range(0, selected.k, 32):
            payloads.append(
                b"".join(
                    lhs[(m + r) * selected.k + k : (m + r) * selected.k + k + 32]
                    for r in range(32)
                )
            )
    for k in range(0, selected.k, 32):
        for n in range(0, selected.n, 32):
            payloads.append(
                bytes(
                    rhs[(n + c) * selected.k + k + r]
                    if selected.rhs_transposed
                    else rhs[(k + r) * selected.n + n + c]
                    for c in range(32)
                    for r in range(32)
                )
            )
    return tuple((i * SLOT_BYTES, payload) for i, payload in enumerate(payloads))
