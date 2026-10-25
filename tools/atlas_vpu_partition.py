"""Complete operand partitions for the guarded 64-register/32-bank geometry."""
from __future__ import annotations

from functools import lru_cache


def _shape(roles):
    groups = {}
    directions = []
    for role in roles:
        operand = role.get("operand")
        count, offset = role.get("registers"), role.get("offset")
        if (operand not in {"rs1", "rs2", "rd"} or type(count) is not int
                or count not in {1, 2} or type(offset) is not int or offset not in {0, 1}
                or type(role.get("write")) is not bool):
            raise ValueError("unsupported operand partition role")
        groups.setdefault(operand, []).extend(range(offset, offset + count))
        directions.extend([role["write"]] * count)
    if not groups or len(groups) > 3:
        raise ValueError("unsupported operand partition arity")
    for offsets in groups.values():
        if offsets not in ([0], [0, 1]):
            raise ValueError("operand partition requires one register or an aligned adjacent pair")
    return tuple((key, len(offsets)) for key, offsets in groups.items()), tuple(directions)


def _footprints(registers, directions):
    if len(registers) != len(directions):
        raise ValueError("operand roles differ from source register footprints")
    result = []
    for reg, write in zip(registers, directions):
        if reg.write != write or not reg.requests:
            raise ValueError("unsupported operand partition footprint direction")
        requests = {}
        for age, row in reg.requests:
            if (type(age) is not int or not 0 <= age < 4096 or type(row) is not int
                    or not 0 <= row < 32 or age in requests):
                raise ValueError("unsupported operand partition row requests")
            requests[age] = row
        result.append((write, tuple(sorted(requests.items()))))
    return tuple(result)


def _constraints(footprints):
    """Forbidden equal-logical and different-logical bank aliases per operand."""
    result = []
    for i, (aw, a) in enumerate(footprints):
        left = dict(a)
        for j, (bw, b) in enumerate(footprints[i + 1:], i + 1):
            common = [(left[age], row) for age, row in b if age in left]
            if aw == bw:
                # Native VPU read cache coalesces precisely (logical reg,row).
                equal = bool(common) if aw else any(x != y for x, y in common)
                alias = bool(common)
            else:
                equal, alias = any(x == y for x, y in common), False
            if equal or alias:
                result.append((i, j, equal, alias))
    return tuple(result)


def _legal(registers, constraints):
    for i, j, equal, alias in constraints:
        a, b = registers[i], registers[j]
        if a % 32 == b % 32 and (equal if a == b else alias):
            return False
    return True


def relation(registers_a, registers_b):
    return ",".join(f"{i}{'=' if a == b else '~'}{j}"
                    for i, a in enumerate(registers_a)
                    for j, b in enumerate(registers_b) if a % 32 == b % 32) or "disjoint"


def _assignments(widths):
    """All bank-pair partitions, with logical-half and singleton-lane choices."""
    def visit(values, used):
        if len(values) == len(widths):
            yield tuple(values)
            return
        width = widths[len(values)]
        for bank_pair in range(used + 1):
            # A fresh bank-pair's first logical half can always be named zero.
            for half in ((0,) if bank_pair == used else (0, 1)):
                for lane in ((0,) if width == 2 else (0, 1)):
                    values.append(2 * bank_pair + 32 * half + lane)
                    yield from visit(values, max(used, bank_pair + 1))
                    values.pop()
    yield from visit([], 0)


@lru_cache(maxsize=128)
def _domain(shape_a, shape_b, footprints_a, footprints_b):
    shapes = shape_a + shape_b
    widths = tuple(width for _, width in shapes)
    if len(widths) > 6:
        raise ValueError("unsupported operand partition size")
    constraints_a, constraints_b = _constraints(footprints_a), _constraints(footprints_b)
    result = {}
    for bases in _assignments(widths):
        registers_a = tuple(base + index for base, (_, width) in zip(bases, shape_a)
                            for index in range(width))
        registers_b = tuple(base + index for base, (_, width) in zip(bases[len(shape_a):], shape_b)
                            for index in range(width))
        if not _legal(registers_a, constraints_a) or not _legal(registers_b, constraints_b):
            continue
        key = relation(registers_a, registers_b)
        # Retain a deterministic legal witness for every exact cross relation.
        result.setdefault(key, (bases[:len(shape_a)], bases[len(shape_a):]))
    return tuple((key, *value) for key, value in sorted(result.items()))


def operand_domain(first_roles, second_roles, first_registers, second_registers):
    """Return exact cross relations and concrete, independently legal witnesses.

    Footprints must come from guarded source-derived facts. Missing/unknown
    structures fail closed; legality includes duplicate-source coalescing,
    distinct-logical same-bank reads, writes and same-row read/write collisions.
    This also supports engines with unrestricted singleton register operands.
    """
    shape_a, directions_a = _shape(first_roles)
    shape_b, directions_b = _shape(second_roles)
    footprints_a = _footprints(first_registers, directions_a)
    footprints_b = _footprints(second_registers, directions_b)
    return [(key, dict(zip((key for key, _ in shape_a), a)),
             dict(zip((key for key, _ in shape_b), b)))
            for key, a, b in _domain(shape_a, shape_b, footprints_a, footprints_b)]


def standalone_domain(roles, registers):
    """All legal internal equality/bank-alias classes for one instruction."""
    shape, directions = _shape(roles)
    footprints = _footprints(registers, directions)
    constraints = _constraints(footprints)
    result = {}
    for bases in _assignments(tuple(width for _, width in shape)):
        regs = tuple(base + index for base, (_, width) in zip(bases, shape)
                     for index in range(width))
        if _legal(regs, constraints):
            result.setdefault(relation(regs, regs), bases)
    return [(key, dict(zip((key for key, _ in shape), values)))
            for key, values in sorted(result.items())]


def preserves_inputs(roles, registers, values):
    """Prove no internal write precedes the first read of an input row."""
    _shape(roles)
    concrete = [values[role["operand"]] + role["offset"] + index
                for role in roles for index in range(role["registers"])]
    reads, writes = {}, {}
    for reg, plan in zip(concrete, registers):
        target = writes if plan.write else reads
        for age, row in plan.requests:
            key = (reg, row)
            target[key] = min(target.get(key, age), age)
    return all(key not in writes or writes[key] >= age for key, age in reads.items())


def require_preserved_inputs(roles, registers, values):
    if not preserves_inputs(roles, registers, values):
        raise ValueError("source operand alias writes an input row before its first sampled read")
    return True
