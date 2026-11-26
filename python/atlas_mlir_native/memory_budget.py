"""Atlas's application budgets, derived by Merlin from a memory region of the SoC declaration.

Merlin's derive_budget subtracts every reservation from one declared region, and the free range it leaves is the
device window. The host's code, runtime, and stack come from a link. Weights, state, and KV caches have no
reservation of their own, because Merlin places them inside that window.
"""
from dataclasses import asdict

from merlin.compiler.build.memory_budget import (REQUIRED_KINDS, LinkedReservation, Reservation, budget_record,
                                                 derive_budget, measure_linked_budget)

from .soc import SOC

# Page alignment of the window: the window base is the first 4 KiB boundary after the host.
ALIGNMENT_BYTES = 4096
HOST_BOUNDS = (LinkedReservation("host_code", "code", "__text_start", "__data_start"),
               LinkedReservation("host_runtime", "runtime", "__data_start", "__stack_start"),
               LinkedReservation("host_stack", "stack", "__stack_start", "__stack_end"))
HOST_KINDS = frozenset(row.kind for row in HOST_BOUNDS)


def _absent(region, kinds, evidence):
    return tuple(Reservation(kind, kind, region.id, region.base, 0, evidence) for kind in sorted(kinds))


def _placed(region):
    return _absent(region, REQUIRED_KINDS - HOST_KINDS, "placed by Merlin inside the device window")


def unhosted_budget(region_id):
    """The whole region as one window: its host links elsewhere, or after the window data in a measuring compile."""
    region = SOC.memory_region(region_id)
    return derive_budget(region, _absent(region, HOST_KINDS, "no host is linked into this region before the window")
                         + _placed(region), alignment_bytes=ALIGNMENT_BYTES)


def hosted_ddr_budget(elf, host_address, *, nm_tool, timeout_s):
    """The DDR budget once the host a measuring link placed at host_address moves to the DDR base.

    Merlin measures the host's code, runtime, and stack from the ELF's own symbols; each range then moves by the
    same offset, so the window starts at the first page after the host the bound compile links first.
    """
    ddr = SOC.memory_region("ddr")
    measured = measure_linked_budget(ddr, elf, HOST_BOUNDS, reservations=_placed(ddr),
                                     alignment_bytes=ALIGNMENT_BYTES, nm_tool=nm_tool, timeout_s=timeout_s)
    linked = [row for row in measured.budget.reservations if row.kind in HOST_KINDS]
    if min(row.base for row in linked) != host_address:
        raise ValueError("the measured host does not start at its allocation")
    shift = ddr.base - host_address
    host = tuple(Reservation(row.id, row.kind, ddr.id, row.base + shift, row.size_bytes,
                             row.evidence + "; moved to the DDR base") for row in linked)
    budget = derive_budget(ddr, host + _placed(ddr), alignment_bytes=ALIGNMENT_BYTES)
    if len(budget.free_ranges) != 1:
        raise ValueError("the DDR budget must leave one window after the host")
    return budget


def final_ddr_budget(elf, *, nm_tool, timeout_s):
    """Merlin's budget of a final board ELF, measured from its linked host."""
    ddr = SOC.memory_region("ddr")
    return measure_linked_budget(ddr, elf, HOST_BOUNDS, reservations=_placed(ddr), alignment_bytes=ALIGNMENT_BYTES,
                                 nm_tool=nm_tool, timeout_s=timeout_s).budget


def window_record(budget):
    """The record Merlin's compile reads with --memory-budget: the budget and its one window."""
    if len(budget.free_ranges) != 1:
        raise ValueError("a deployment window budget has exactly one free range")
    return budget_record(budget, range_index=0)


def declared_region(document):
    """The declared region a budget record subtracts from; refuses a region the SoC declaration does not hold."""
    region = SOC.memory_region(document["region"]["id"])
    if asdict(region) != document["region"]:
        raise ValueError("memory budget region differs from the SoC declaration")
    return region
