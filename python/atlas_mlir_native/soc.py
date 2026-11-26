"""The SoC Atlas targets: one declaration owned by this target, validated by Merlin's generic schema."""
from hashlib import sha256
import json
from pathlib import Path

from merlin.target.soc import soc_description

DECLARATION = Path(__file__).with_name("penguin_sophialake.json")
# Records that name the declaration a deployment was placed against use this identity and digest.
IDENTITY = DECLARATION.stem
DECLARATION_SHA256 = sha256(DECLARATION.read_bytes()).hexdigest()
_DOCUMENT = json.loads(DECLARATION.read_text())
SOC = soc_description(_DOCUMENT)
# Upper bound on captured read-only state read for identity: no deployment of this SoC can hold more than its
# declared off-chip memory. Bench-staged kits keep the state on the bench host; a resident deploy checks its own placement.
CAPTURED_STATE_READ_BYTES = SOC.memory_region("ddr").capacity_bytes


def board_program(document):
    """How the board's host starts a DDR program and learns that it ended: an Atlas-owned fact of the declaration."""
    row = document.get("board_program")
    if (not isinstance(row, dict) or set(row) != {"start", "completion", "evidence"}
            or any(type(row[key]) is not str or not row[key] for key in row)
            or row["completion"] not in ("tohost", "return")):
        raise ValueError("SoC declaration requires board_program start, completion (tohost or return), and evidence")
    return dict(row)


BOARD_PROGRAM = board_program(_DOCUMENT)


def host_stack(document):
    """The stack every deployment links for the host core: an Atlas-owned fact of the declaration."""
    row = document.get("host_stack")
    if (not isinstance(row, dict) or set(row) != {"bytes", "evidence"} or type(row["bytes"]) is not int
            or row["bytes"] <= 0 or row["bytes"] % 16 or type(row["evidence"]) is not str or not row["evidence"]):
        raise ValueError("SoC declaration requires host_stack bytes (positive, 16-byte aligned) and evidence")
    return row["bytes"]


HOST_STACK_BYTES = host_stack(_DOCUMENT)
PLACEMENTS = ("onchip", "ddr")


def carriers(document):
    """The boards the chip runs on: an Atlas-owned fact of the declaration, one row per board.

    A row's clock is what the board supplies to the chip, or null while unknown. The serial-TL link clock is a fact of
    each FPGA bitstream, recorded with a run, so no row may carry one and no kit byte depends on it.
    """
    rows = document.get("carriers")
    if not isinstance(rows, list) or not rows:
        raise ValueError("SoC declaration requires carriers")
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"id", "placement", "clock_hz", "evidence"}:
            raise ValueError("each carrier declares exactly id, placement, clock_hz, and evidence; "
                             "a link clock belongs to the bitstream, not the carrier")
        if (type(row["id"]) is not str or not row["id"] or row["placement"] not in PLACEMENTS
                or type(row["evidence"]) is not str or not row["evidence"]
                or row["clock_hz"] is not None and (type(row["clock_hz"]) is not int or row["clock_hz"] <= 0)):
            raise ValueError(f"carrier {row.get('id')!r} needs a name, a placement ({' or '.join(PLACEMENTS)}), "
                             "a positive clock_hz or null, and evidence")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("carrier ids must be unique")
    return tuple(dict(row) for row in rows)


CARRIERS = carriers(_DOCUMENT)


def carrier_clock(placement, rows=None):
    """The chip clock every declared carrier of this placement supplies, and those carriers; refuses rather than guess."""
    rows = [row for row in (CARRIERS if rows is None else rows) if row["placement"] == placement]
    if not rows:
        raise ValueError(f"no carrier in the SoC declaration runs {placement} deployments")
    for row in rows:
        if row["clock_hz"] is None:
            raise ValueError(f"carrier {row['id']} declares no chip clock; state it in the SoC declaration's carriers")
    if len({row["clock_hz"] for row in rows}) != 1:
        raise ValueError(f"{placement} carriers supply different chip clocks: "
                         + ", ".join(f"{row['id']} {row['clock_hz']} Hz" for row in rows))
    return rows[0]["clock_hz"], tuple(row["id"] for row in rows)
