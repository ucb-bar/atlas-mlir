"""model wait insertion completes launch n-8 before launch n reuses its DMA command slot."""

from __future__ import annotations

import json
import os
import pathlib
import re
import subprocess
import tempfile
import unittest

from test_delay_insertion import OPT, run
from model_evidence import requires_model_evidence
from test_virtual_scheduling import F, S, T, add, const, dma_load, dma_store, function, inp, outp, start


def fixture_evidence(directory: pathlib.Path) -> str:
    """Select generated facts through the production loader."""
    facts = pathlib.Path(os.environ["ATLAS_TEST_MODEL_FACTS"])
    return f"--select-atlas-model-evidence=facts={facts}"


BASE = -2147483648


def chain(pairs: int, helper: bool = False) -> str:
    """Load x0, then `pairs` loads each waited by its use, store their sum, and use x0 last."""
    lines = [start("s0"), const("size", 2048), *(const(f"a{i}", BASE + 2048 * i) for i in range(pairs + 3)),
             dma_load("s0", "s1", "x0", "a0", "size", "bf16")]
    state = 1
    if helper:
        lines.append(inp("s1", "s2", "h", 0))
        state = 2
    for i in range(1, pairs + 1):
        lines.append(dma_load(f"s{state}", f"s{state + 1}", f"x{i}", f"a{i}", "size", "bf16"))
        state += 1
        if i > 1:
            lines.append(add(f"y{i}", f"y{i - 1}" if i > 2 else "x1", f"x{i}"))
    lines += [dma_store(f"s{state}", f"s{state + 1}", f"y{pairs}", f"a{pairs + 1}"), add("t", "x0", f"y{pairs}"),
              dma_store(f"s{state + 1}", f"s{state + 2}", "t", f"a{pairs + 2}")]
    state += 2
    if helper:
        lines += [add("u", "h", "t"), outp(f"s{state}", f"s{state + 1}", "u", 0)]
        state += 1
    return function("ring", lines + [f"return %s{state} : {S}"])


def dma_sequence(text: str) -> list[tuple[str, int | None]]:
    """Each launch and wait in pc order with its explicit transfer, None for a helper."""
    sequence = []
    for line in text.splitlines():
        if kind := re.search(r'"atlas\.(dma|dma_wait)"', line):
            transfer = re.search(r"atlas\.virtual_dma_transfer = (\d+)", line)
            sequence.append((kind.group(1), int(transfer.group(1)) if transfer else None))
    return sequence


@requires_model_evidence
class DMARingWaitTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = pathlib.Path(temporary.name)
        self.evidence = fixture_evidence(self.directory)

    def lower(self, source: str, *, order: bool = True) -> str:
        """Prepare, replay the source order and lower for model, leaving the waits pending."""
        prepared = run(OPT, source, self.evidence, "--atlas-compile-prepare=boundary=none")
        self.assertEqual(prepared.returncode, 0, prepared.stderr)
        graph, replay = self.directory / "graph.json", self.directory / "order.json"
        written = run(OPT, prepared.stdout, f"--schedule-atlas-virtual=provider=model graph-output={graph}")
        self.assertEqual(written.returncode, 0, written.stderr)
        replay.write_text(json.dumps([list(range(len(block["preds"]))) for block in json.loads(graph.read_text())]))
        lowered = run(OPT, prepared.stdout, "--atlas-compile-schedule=provider=model boundary=none" +
                      (f" order-file={replay}" if order else ""), "--atlas-compile-spill",
                      "--atlas-compile-lower=provider=model")
        self.assertEqual(lowered.returncode, 0, lowered.stderr)
        return lowered.stdout

    def insert(self, lowered: str, provider: str) -> str:
        inserted = run(OPT, lowered, f"--insert-atlas-dma-waits=provider={provider}")
        self.assertEqual(inserted.returncode, 0, inserted.stderr)
        return inserted.stdout

    def schedule(self, inserted: str) -> subprocess.CompletedProcess[str]:
        return run(OPT, inserted, "--schedule-atlas-stream=provider=model", "--verify-atlas-machine-stream",
                   "--verify-atlas-timing", "--verify-atlas-generated-schedule")

    def test_ninth_launch_waits_for_the_first_transfer(self) -> None:
        for pairs in (6, 7):
            with self.subTest(pairs=pairs):
                lowered = self.lower(chain(pairs))
                model, builtin = self.insert(lowered, "model"), self.insert(lowered, "builtin")
                sequence = dma_sequence(model)
                self.assertEqual(sequence.count(("dma_wait", 0)), 1)
                scheduled = self.schedule(model)
                self.assertEqual(scheduled.returncode, 0, scheduled.stderr)
                if pairs == 6:
                    # Launch 8 is the second store, which follows x0's use and its wait.
                    self.assertEqual(model, builtin)
                    continue
                launches = [index for index, (kind, _) in enumerate(sequence) if kind == "dma"]
                self.assertEqual(sequence[launches[8] - 1], ("dma_wait", 0))
                self.assertEqual(dma_sequence(builtin).count(("dma_wait", 0)), 1)
                self.assertNotEqual(model, builtin)
                refused = self.schedule(builtin)
                self.assertNotEqual(refused.returncode, 0)
                self.assertIn("DMA command slot is still pending: launch 8 reuses slot 0 of unwaited launch 0",
                              refused.stderr)

    def test_helper_launch_completes_the_reused_slot(self) -> None:
        lowered = self.lower(chain(7, helper=True))
        model = self.insert(lowered, "model")
        self.assertIn(("dma", None), dma_sequence(model))
        self.assertEqual(model, self.insert(lowered, "builtin"))
        scheduled = self.schedule(model)
        self.assertEqual(scheduled.returncode, 0, scheduled.stderr)

    def test_stream_reader_invariant_rejects_launch_inside_a_loop(self) -> None:
        source = function("loop", [
            start("s0"), const("size", 2048), *(const(f"a{i}", BASE + 2048 * i) for i in range(3)),
            dma_load("s0", "s1", "x", "a0", "size", "bf16"),
            f'%p = "atlas.virtual_pack_fp8"(%x) {{scale_code = 127 : i32}} : ({T}) -> {F}',
            f'%y = "atlas.virtual_mxu_matmul"(%p, %p) {{unit = 0 : i32}} : ({F}, {F}) -> {T}',
            dma_load("s1", "s2", "z", "a1", "size", "bf16"), add("w", "y", "z"),
            dma_store("s2", "s3", "w", "a2"), f"return %s3 : {S}"])
        lines = self.lower(source, order=False).splitlines()
        pack = next(re.search(r"atlas\.virtual_cfg_source = (\d+)", line).group(1)
                    for line in lines if "atlas.virtual_cfg_helper" in line and "atlas.virtual_cfg_source" in line)
        launch = next(index for index, line in enumerate(lines)
                      if '"atlas.dma"' in line and "atlas.virtual_dma_transfer = 1 " in line)
        state = lines[launch].split("=")[0].strip()
        owner = ('atlas.virtual_cfg_block = 0 : i32, atlas.virtual_cfg_helper = "pack", '
                 f'atlas.virtual_cfg_source = {pack} : i32')
        # A PACK-owned branch back to the launch, with its scalar delay slot.
        lines[launch + 1] = lines[launch + 1].replace(f"({state})", "(%slot)", 1)
        lines[launch + 1:launch + 1] = [
            f'  %loop = "atlas.branch"({state}) <{{kind = "blt", '
            'lhs = 0 : i32, rhs = 0 : i32, offset_bytes = -2 : i32}> '
            f"{{{owner}}} : (!atlas.state) -> !atlas.state",
            f'  %slot = "atlas.alu_imm"(%loop) <{{dst = 0 : i32, immediate = 0 : i32, kind = "addi", src = 0 : i32}}> '
            f"{{{owner}}} : (!atlas.state) -> !atlas.state"]
        # The source memory contract refuses the repeated launch before slot counting.
        rejected = run(OPT, "\n".join(lines) + "\n", "--insert-atlas-dma-waits=provider=model")
        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn("launched twice", rejected.stderr)


if __name__ == "__main__":
    unittest.main()
