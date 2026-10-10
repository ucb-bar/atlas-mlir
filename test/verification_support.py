"""Shared vocabulary, boundary checks and text mutations for generated-artifact tests."""

from __future__ import annotations

import re

from test_virtual_lowering import run


STATE = "!atlas.state"
MARKER = 'atlas.generated_from_virtual = "resource-contract-v5"'
TIMED = 'atlas.timing_state = "timed"'
UNTIMED_STATE = 'atlas.timing_state = "untimed"'
PROVIDER = 'atlas.timing_provider = "npu-model-rtl-match-v1"'
TRANSFER = "atlas.virtual_dma_transfer"
NOP = ("alu_imm", 'kind = "addi", dst = 0 : i32, src = 0 : i32, immediate = 0 : i32')
DELAY = ("delay", 'cycles = 256 : i32, atlas.delay_reason = "tensor_completion"')
TRAP = ("trap", 'kind = "ecall"')
MASK = 0xffffffff
# Classification diagnostics (lib/AtlasGeneratedArtifact.cpp).
UNSUPPORTED = "unsupported Atlas virtual-to-machine artifact marker"
UNMARKED = "generated resource metadata requires an Atlas virtual-to-machine artifact marked"
INCOMPLETE = "resource-contract-v5 artifact requires"
CLASSIFICATION = (UNSUPPORTED, UNMARKED, INCOMPLETE)

VERIFY = ("atlas-opt", ("--verify-atlas-generated-schedule",))
LLVM = (("atlas-opt", ("--convert-atlas-to-llvm",)), ("atlas-opt", ("--convert-atlas-to-llvm-calls",)))
TIMED_FINAL = (VERIFY, ("atlas-emit", ()), *LLVM)
UNTIMED = (VERIFY, ("atlas-emit", ("--allow-untimed",)))
STREAM_REWRITES = (("atlas-opt", ("--insert-atlas-delays",)), ("atlas-opt", ("--schedule-atlas-stream",)))

FIELD_RE = re.compile(r'([\w.]+) = (-?\d+ : i32|"[^"]*"|true|false|array<i32[^>]*>)')
WORD_RE = re.compile(r"atlas\.word = (-?\d+) : i32")


def assert_boundaries(test, text: str, boundaries=None, *, rejects: str | None = None) -> None:
    """Accept at every boundary, or reject with `rejects` (any diagnostic when empty) and no unintended classification failure."""
    for tool, options in boundaries or (TIMED_FINAL if TIMED in text else UNTIMED):
        with test.subTest(tool=tool, options=options):
            result = run(tool, text, *options)
            if rejects is None:
                test.assertEqual(result.returncode, 0, result.stderr)
                test.assertTrue(result.stdout)
                continue
            test.assertNotEqual(result.returncode, 0, result.stdout)
            test.assertEqual(result.stdout, "")
            test.assertTrue(result.stderr)
            test.assertIn(rejects, result.stderr)
            for unintended in CLASSIFICATION:
                if unintended not in rejects:
                    test.assertNotIn(unintended, result.stderr)


def checked(test, text: str, *options: str, tool: str = "atlas-opt") -> str:
    result = run(tool, text, *options)
    test.assertEqual(result.returncode, 0, result.stderr)
    return result.stdout


def finalize_rejects(test, structured: str, diagnostic: str = "") -> None:
    assert_boundaries(test, structured, (("atlas-opt", ("--finalize-atlas-llvm-calls",)),), rejects=diagnostic)


def value(raw: str):
    if raw.startswith("array<i32"):
        inner = raw[len("array<i32"):-1].lstrip(":").strip()
        return tuple(int(item) for item in inner.split(",")) if inner else ()
    if raw.startswith('"'):
        return raw[1:-1]
    if raw in ("true", "false"):
        return raw == "true"
    return int(raw.split(" : ")[0])


def fields(text: str) -> dict:
    return {name: value(raw) for name, raw in FIELD_RE.findall(text)}


def attribute_value(text: str, key: str) -> str:
    """The value of the first attribute `key` in `text`, which may nest brackets."""
    begin = text.index(key + " = ") + len(key) + 3
    depth, quoted = 0, False
    for end in range(begin, len(text)):
        char = text[end]
        if char == '"':
            quoted = not quoted
        elif quoted:
            continue
        elif char in "[{<(":
            depth += 1
        elif char in "]}>)" and depth:
            depth -= 1
        elif not depth and char in ",}":
            return text[begin:end]
    raise AssertionError(f"unterminated {key}")


def contract_text(machine: str, name: str) -> str:
    if f"atlas.virtual_{name}_contract = " not in machine:
        raise AssertionError(f"artifact must retain the source {name} contract")
    return attribute_value(machine, f"atlas.virtual_{name}_contract")


def records(machine: str, name: str) -> list[dict]:
    """Records of a flat contract array; *_byte addresses are unsigned."""
    result = []
    for text in re.findall(r"\{([^{}]*)\}", contract_text(machine, name)):
        entry = fields(text)
        result.append({key: item & MASK if key.endswith("_byte") else item for key, item in entry.items()})
    return result


def replace_contract(machine: str, name: str, text: str) -> str:
    old = f"atlas.virtual_{name}_contract = {contract_text(machine, name)}"
    return machine.replace(old, f"atlas.virtual_{name}_contract = {text}", 1)


def drop_attribute(machine: str, name: str) -> str:
    """Remove one top-level module attribute."""
    header, rest = machine.split("\n", 1)
    old = f"{name} = {attribute_value(header, name)}" if name + " = " in header else name
    start = header.index(old)
    end = start + len(old)
    if header.startswith(", ", end):
        end += 2
    elif header[start - 2:start] == ", ":
        start -= 2
    return header[:start] + header[end:] + "\n" + rest


def lines_of(machine: str, name: str, *needles: str) -> list[int]:
    return [i for i, line in enumerate(machine.splitlines()) if f'"atlas.{name}"' in line and all(n in line for n in needles)]


def line_index(machine: str, *needles: str) -> int:
    return next(i for i, line in enumerate(machine.splitlines()) if all(n in line for n in needles))


def rewrite_line(machine: str, where, change) -> str:
    """Apply `change` to line `where` (an index, or the first line satisfying a predicate); the line must change."""
    lines = machine.splitlines()
    index = where if isinstance(where, int) else next(i for i, line in enumerate(lines) if where(line))
    changed = change(lines[index])
    if changed == lines[index]:
        raise AssertionError(f"mutation left line {index} unchanged: {lines[index]}")
    lines[index] = changed
    return "\n".join(lines) + "\n"


def replace_line(machine: str, index: int, old: str, new: str) -> str:
    def change(line: str) -> str:
        if old not in line:
            raise AssertionError((old, line))
        return line.replace(old, new, 1)
    return rewrite_line(machine, index, change)


def last_constant_write(machine: str, before: int, reg: int) -> int:
    """Line index of the last LUI or ADDI writing x`reg` before line `before`."""
    lines = machine.splitlines()
    return max(i for i in range(before) if f"dst = {reg} : i32" in lines[i]
               and ('"atlas.alu_imm"' in lines[i] or '"atlas.upper"' in lines[i]))


def shift_constant(line: str, addi_step: int) -> str:
    """The same constant write moved by one aligned step: `addi_step` through
    ADDI, or one 4096 unit through LUI."""
    step = 1 if '"atlas.upper"' in line else addi_step
    return re.sub(r"immediate = (-?\d+)", lambda m: f"immediate = {int(m.group(1)) + step}", line, count=1)


def constant(reg: int, value: int) -> list[tuple[str, str]]:
    low = ((value + 2048) & 4095) - 2048
    return [("upper", f'kind = "lui", dst = {reg} : i32, immediate = {((value - low) >> 12) & 0xFFFFF} : i32'),
            ("alu_imm", f'kind = "addi", dst = {reg} : i32, src = {reg} : i32, immediate = {low} : i32')]


def operations(machine: str) -> list[str]:
    return [line for line in machine.splitlines() if ": (!atlas.state)" in line]


def reorder(machine: str, ordered: list[str]) -> str:
    """Replace the issued operations with `ordered`, rethreading their state."""
    lines = machine.splitlines()
    first = next(n for n, line in enumerate(lines) if ": (!atlas.state)" in line)
    last = max(n for n, line in enumerate(lines) if ": (!atlas.state)" in line)
    previous = re.search(r"%(\w+) =", next(line for line in lines if '"atlas.start"' in line))[1]
    rebuilt = []
    for index, line in enumerate(ordered):
        line = re.sub(r"^\s*%\w+ =", f"  %r{index} =", line)
        rebuilt.append(re.sub(r'("atlas\.[^"]+"\()%\w+(\))', rf"\g<1>%{previous}\g<2>", line))
        previous = f"r{index}"
    return "\n".join(lines[:first] + rebuilt + lines[last + 1:]) + "\n"


def moved(machine: str, chosen, anchor) -> str:
    """Move the operations satisfying `chosen`, in order, to just before the first other one satisfying `anchor`."""
    ops = operations(machine)
    group = [line for line in ops if chosen(line)]
    rest = [line for line in ops if not chosen(line)]
    at = next(n for n, line in enumerate(rest) if anchor(line))
    return reorder(machine, rest[:at] + group + rest[at:])


def insert_after(machine: str, index: int, inserted) -> str:
    """Insert (name, fields) operations after line `index`, owned by that line's source block."""
    line = machine.splitlines()[index]
    block = re.search(r"atlas\.virtual_cfg_block = (\d+) : i32", line)[1]
    ops = operations(machine)
    at = ops.index(line) + 1
    new = [f'%x = "atlas.{name}"(%x) {{{text}, atlas.virtual_cfg_block = {block} : i32}} : ({STATE}) -> {STATE}' for name, text in inserted]
    return reorder(machine, ops[:at] + new + ops[at:])


def remove_line(machine: str, index: int) -> str:
    ops = operations(machine)
    ops.remove(machine.splitlines()[index])
    return reorder(machine, ops)


def structured_word(line: str, old: str, new: str, shift: int, width: int, field: int) -> str:
    """Change a structured LLVM call's field and the same bits of its encoded atlas.word."""
    if old not in line:
        raise AssertionError((old, line))
    line = line.replace(old, new, 1)
    word = int(WORD_RE.search(line)[1]) & MASK
    word = word & ~(((1 << width) - 1) << shift) | field << shift
    return WORD_RE.sub(f"atlas.word = {word} : i32", line, count=1)


def shorten_all_delays(machine: str) -> str:
    changed = re.sub(r"cycles = \d+ : i32", "cycles = 1 : i32", machine)
    if changed == machine:
        raise AssertionError("expected a delay to shorten")
    return changed


def dma_wait(channel: int = 0, identity: int | None = 0) -> tuple[str, str]:
    tag = f", {TRANSFER} = {identity} : i32" if identity is not None else ""
    return "dma_wait", f"channel = {channel} : i32{tag}"


def handoff_chain(test, untimed: str, *, reorder: bool = True) -> dict[str, str]:
    """Checked stages from an untimed artifact to finalized LLVM; finalization must equal direct conversion."""
    test.assertIn(UNTIMED_STATE, untimed)
    checked(test, untimed, "--verify-atlas-machine-stream", "--verify-atlas-generated-schedule")
    stages = {"untimed": untimed}
    if reorder:
        stages["ordered"] = checked(test, untimed, "--schedule-atlas-stream=insert-delays=false",
                                    "--verify-atlas-machine-stream", "--verify-atlas-generated-schedule")
        test.assertIn(UNTIMED_STATE, stages["ordered"])
        test.assertNotIn('"atlas.delay"', stages["ordered"])
    stages["timed"] = checked(test, stages.get("ordered", untimed), "--insert-atlas-delays", "--verify-atlas-timing",
                              "--verify-atlas-generated-schedule")
    test.assertNotIn("atlas.delay_reason", stages["timed"])
    test.assertTrue(checked(test, stages["timed"], tool="atlas-emit"))
    stages["structured"] = checked(test, stages["timed"], "--convert-atlas-to-llvm-calls")
    for stage in ("timed", "structured"):
        test.assertIn(TIMED, stages[stage])
        test.assertIn(PROVIDER, stages[stage])
    stages["direct"] = checked(test, stages["timed"], "--convert-atlas-to-llvm")
    test.assertEqual(checked(test, stages["structured"], "--finalize-atlas-llvm-calls"), stages["direct"])
    return stages
