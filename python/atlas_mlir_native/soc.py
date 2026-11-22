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
