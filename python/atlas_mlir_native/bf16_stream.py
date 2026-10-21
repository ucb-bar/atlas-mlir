"""Track BF16 value identities through concrete DMA and register accesses."""

from __future__ import annotations

from dataclasses import dataclass

from merlin.semantic_compiler.target_binding import NativeCompilationError


@dataclass(frozen=True)
class BF16DataTrace:
    loads: tuple[tuple[int, int], ...]
    computations: tuple[tuple, ...]
    stores: tuple[tuple, ...]


def trace_bf16_data(effects, *, unary_modes, binary_modes) -> BF16DataTrace:
    """Replay data, separately from the schedule's timing and overlap checks.

    A loaded half is identified by its DRAM byte address. Each computation
    creates a new value version, including when it reuses a physical pair.
    This lets independent loads move without confusing their consumers or
    accepting a store of an older value in the same register.
    """
    scratch, registers, pending = {}, {}, {}
    loads, computations = [], []
    stores = {}

    def read(mapping, key):
        if key not in mapping:
            raise NativeCompilationError(
                "unqualified_target",
                "emitted BF16 operation reads different source pairs: uninitialized register"
                if mapping is registers else "BF16 stream reads uninitialized data",
            )
        return mapping[key]

    for effect in effects:
        kind = effect[0]
        if kind == "dma":
            direction, dram, word, size, channel = effect[1:]
            if size not in {1024, 2048} or word % 256:
                raise NativeCompilationError(
                    "unqualified_target", "BF16 stream has misaligned DMA data"
                )
            words = tuple(word + offset // 4 for offset in range(0, size, 1024))
            if direction == "load":
                pending.setdefault(channel, []).append((dram, words))
            elif direction == "store":
                stores.setdefault((dram, size), []).append(
                    tuple(read(scratch, address) for address in words)
                )
            else:
                raise NativeCompilationError("unqualified_target", "invalid BF16 DMA direction")
        elif kind == "wait":
            for dram, words in pending.pop(effect[1], []):
                for half, word in enumerate(words):
                    scratch[word] = ("input", dram + half * 1024)
        elif kind == "vload":
            value = read(scratch, effect[2])
            if value[0] != "input":
                raise NativeCompilationError(
                    "unqualified_target", "BF16 boundary load reads non-input data"
                )
            registers[effect[1]] = value
            loads.append((effect[1], value[1]))
        elif kind == "vstore":
            scratch[effect[2]] = read(registers, effect[1])
        elif kind in unary_modes or kind in binary_modes:
            sources = effect[1:3] if kind in binary_modes else effect[1:2]
            values = tuple(read(registers, source + half)
                           for source in sources for half in (0, 1))
            computations.append((kind, values))
            for half in (0, 1):
                registers[effect[-1] + half] = ("result", len(computations) - 1, half)
        elif kind not in {"config", "complete", "trap"}:
            raise NativeCompilationError(
                "unqualified_target", "BF16 data replay has an unmodeled operation"
            )
    return BF16DataTrace(
        tuple(loads), tuple(computations),
        tuple((key, tuple(values)) for key, values in sorted(stores.items())),
    )


def loaded_bf16_pairs(trace: BF16DataTrace) -> dict[int, list[int]]:
    """Index adjacent raw register halves by the source tile's byte address."""
    if len(trace.loads) % 2:
        raise NativeCompilationError("unqualified_target", "incomplete BF16 load pair")
    pairs = {}
    for first, second in zip(trace.loads[::2], trace.loads[1::2]):
        if second != (first[0] + 1, first[1] + 1024):
            raise NativeCompilationError("unqualified_target", "BF16 load pair changed")
        pairs.setdefault(first[1], []).append(first[0])
    return pairs
