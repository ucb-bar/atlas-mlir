"""Static register checks shared by every contraction assembly entry point."""

from functools import wraps
from dataclasses import replace
from inspect import signature
import re


_MATRIX = re.compile(r"\bm(\d+)\b")
_SCALAR = re.compile(r"\bx(\d+)\b")
_BINARY = {"vadd.bf16", "vsub.bf16", "vmul.bf16", "vminimum.bf16", "vmaximum.bf16"}
_UNARY = {"vmov", "vmov.bf16"} | {
    name + ".bf16" for name in ("vrecip", "vexp", "vexp2", "vrelu", "vsin", "vcos", "vtanh", "vlog2",
                                "vsqrt", "vsquare", "vcube", "vredsum", "vredmin", "vredmax",
                                "vredsum.row", "vredmin.row", "vredmax.row")
}


def _instructions(source):
    return [line.split("#", 1)[0].split("//", 1)[0].strip() for line in source.splitlines()]


def check_matrix_registers(source):
    """Enforce RTL BF16 pairs and the two-source physical-bank restriction.

    VLOAD/VSTORE, XLU, FP8 MXU traffic, and weight pushes use single registers.
    Mixed pack/unpack use a pair only on their BF16 side. Equal binary sources
    are legal; distinct sources must not share any bank modulo 32.
    """
    for line_number, line in enumerate(_instructions(source), 1):
        if not line or line.endswith(":"):
            continue
        op = line.split(None, 1)[0].lower()
        registers = [int(match) for match in _MATRIX.findall(line)]
        if any(register >= 64 for register in registers):
            raise ValueError(f"matrix register outside m0..m63 at line {line_number}: {line}")
        pairs = []
        if op.startswith("vpack.bf16.fp8"):
            pairs = registers[1:]
        elif op.startswith("vunpack.fp8.bf16"):
            pairs = registers[:1]
        elif (op in ("vmov", "vli.all", "vli.row") or op.endswith(".bf16")
              or (op.startswith(("vmatpush.", "vmatpop.")) and ".bf16." in op)):
            pairs = registers
        if any(register % 2 or register > 62 for register in pairs):
            raise ValueError(f"BF16 pair requires an even register at line {line_number}: {line}")
        if op in _UNARY and len(registers) != 2:
            raise ValueError(f"unary BF16 operation requires two registers at line {line_number}: {line}")
        if op in _BINARY:
            if len(registers) != 3:
                raise ValueError(f"binary BF16 operation requires three registers at line {line_number}: {line}")
            left, right = registers[1:]
            left_banks = {left % 32, (left + 1) % 32}
            right_banks = {right % 32, (right + 1) % 32}
            if left != right and left_banks & right_banks:
                raise ValueError(f"binary BF16 sources share a physical bank at line {line_number}: {line}")


def check_scalar_scratch(source, scratch_registers):
    """Require hook scratch to be unused by the recipe without its hook.

    The prefix carveout is `li x30, 1` followed by DMA configuration,
    and those `dma.config x30` instructions. They finish before any hook can
    execute and preserve C's default x30/x31 scratch choice. The assembly
    wrapper also removes direct_input from this baseline: its gather uses
    x26..x31 as call-local temporaries, all dead before the epilogue executes.
    Later recipe uses of scratch registers remain subject to the check.
    """
    lines = [line for line in _instructions(source) if line]
    used = set()
    prefix = True
    for index, line in enumerate(lines):
        if line.endswith(":"):
            prefix = False
        if prefix:
            setup = re.fullmatch(r"li\s+x30\s*,\s*1", line)
            next_config = index + 1 < len(lines) and re.fullmatch(r"dma\.config\s+x30\s*,\s*ch[01]", lines[index + 1])
            if setup and next_config:
                continue
            if re.fullmatch(r"dma\.config\s+x30\s*,\s*ch[01]", line):
                continue
        used.update(int(match) for match in _SCALAR.findall(line))
    collisions = sorted(used & set(scratch_registers))
    if collisions:
        names = ", ".join(f"x{register}" for register in collisions)
        raise ValueError(f"epilogue scalar scratch collides with recipe registers: {names}")


def check_hook_footprint(source, hook):
    """Require hook MRF writes to avoid every register the recipe names, except the stored tile m2/m3."""
    named = {int(match) for line in _instructions(source) for match in _MATRIX.findall(line)}
    written = set()
    for line in _instructions("\n".join(hook)):
        registers = [int(match) for match in _MATRIX.findall(line)]
        op = line.split(None, 1)[0].lower() if registers else ""
        if registers and not op.startswith(("vstore", "vmatpush")):
            pair = op.endswith(".bf16") or op.startswith("vli")
            written.update((registers[0], registers[0] + 1) if pair else registers[:1])
    collisions = sorted((named & written) - {2, 3})
    if collisions:
        raise ValueError("epilogue overwrites recipe matrix registers: " + ", ".join(f"m{r}" for r in collisions))


def checked_assembly(*, scratch_registers=None, dispatch=False):
    """Check direct recipe calls as well as calls through atlas_program.

    Reuse recipes select their own free pair. The default x30/x31 pair is a
    sentinel requesting that selection; an explicit pair must match it.
    The dispatcher leaves scratch
    admission to the selected recipe when traffic reuse is enabled, avoiding
    a second check using the caller's pre-remap scratch pair.
    """
    def decorate(generator):
        parameters = signature(generator)

        @wraps(generator)
        def checked(*args, **kwargs):
            bound = parameters.bind(*args, **kwargs)
            bound.apply_defaults()
            epilogue = bound.arguments.get("epilogue")
            if epilogue is not None and not (dispatch and bound.arguments.get("traffic_reuse") is True):
                if scratch_registers is not None:
                    if epilogue.scratch_registers not in ((30, 31), scratch_registers):
                        raise ValueError(f"epilogue scalar scratch must be default (30, 31) or recipe pair {scratch_registers}")
                    if epilogue.scratch_registers != scratch_registers:
                        epilogue = replace(epilogue, scratch_registers=scratch_registers)
                        bound.arguments["epilogue"] = epilogue
                baseline = parameters.bind(*args, **kwargs)
                baseline.arguments["epilogue"] = None
                if "direct_input" in parameters.parameters:
                    baseline.arguments["direct_input"] = None
                source = generator(*baseline.args, **baseline.kwargs)
                check_matrix_registers(source)
                check_scalar_scratch(source, scratch_registers or epilogue.scratch_registers)
                if any(step.op == "add_rne" for step in epilogue.steps):
                    # The output pointer only changes scalar lines of the hook.
                    output = min(set(range(1, 32)) - {3, 8, 9, *epilogue.scratch_registers})
                    check_hook_footprint(source, epilogue.apply(output))
            source = generator(*bound.args, **bound.kwargs)
            check_matrix_registers(source)
            return source

        return checked

    return decorate
