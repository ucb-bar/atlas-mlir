"""Selected dma=wait policy beside RTL-computed facts (ATLAS_OOT_BIN_DIR, ATLAS_OP_TIMING)."""
import json
import re
import unittest

from test_delay_insertion import OPT, EMIT, addi, run
from test_rtl_timing import CONSUMERS, FACTS, HALT, MARKER, RESOLVER, VERIFY, delay, selected, vls, word_base


def config(reg=5, channel=0):
    return ("dma_config", f"channel = {channel} : i32, base_reg = {reg} : i32")


def transfer(direction="load", channel=0, reg=6, dram=1, size=2):
    return ("dma", f'direction = "{direction}", channel = {channel} : i32, reg = {reg} : i32, '
                   f"dram = {dram} : i32, size = {size} : i32")


def wait(channel=0):
    return ("dma_wait", f"channel = {channel} : i32")


def setup(vmem=0, offset=0x90000000, size=128, base=0):
    return [*word_base(5, base), config(), *word_base(6, vmem), *word_base(1, offset), *word_base(2, size)]


WAITED = [*setup(), transfer(), wait(), HALT]


@unittest.skipUnless(FACTS and OPT.is_file() and EMIT.is_file(),
                     "requires explicit RTL timing facts and built tools")
class SelectedDMATimingTest(unittest.TestCase):
    def check_consumers(self, ops, accepted):
        for consumer in CONSUMERS:
            with self.subTest(consumer=consumer):
                checked = selected(ops, consumer, VERIFY, dma="wait")
                self.assertEqual(checked.returncode == 0, accepted, checked.stderr)
                if accepted:
                    self.assertEqual(run(EMIT, checked.stdout).returncode, 0, checked.stderr)
        return checked

    def test_both_consumers_and_final_export(self):
        for consumer in CONSUMERS:
            checked = selected(WAITED, consumer, VERIFY, dma="wait")
            self.assertEqual(checked.returncode, 0, checked.stderr)
            self.assertIn('dma = "wait"', checked.stdout)
            exported = run(EMIT, checked.stdout, "--rtl-timing-json")
            self.assertEqual(exported.returncode, 0, exported.stderr)
            data = json.loads(exported.stdout)
            self.assertEqual(data["schema"], "atlas.resolved_rtl_timing.v1")
            self.assertEqual(data["resolver"], {"id": RESOLVER, "version": 1, "dma_policy": "wait"})
            self.assertIn("dma.load.ch0..7", data["applicability"]["supported_operations"])
            self.assertIsNone(data["applicability"]["dma_domain"]["completion_latency"])
            launch = next(i for i in data["instructions"] if i["mnemonic"] == "dma.load.ch0")
            retired = next(i for i in data["instructions"] if i["mnemonic"] == "dma.wait.ch0")
            self.assertIsNone(launch["footprint"]["done_age"])
            self.assertIsNone(launch["footprint"]["dma_cycles"])
            memory = [a for a in launch["footprint"]["accesses"] if a["resource"] == "vmem"]
            self.assertTrue(memory)
            for access in memory:
                self.assertTrue(access["at_completion"])
                self.assertIsNone(access["age"])
                self.assertIsNone(access["step"])
            self.assertGreater(retired["issue_epoch"], launch["issue_epoch"])
            self.assertEqual(retired["epoch_offset"], 0)
            self.assertNotIn("logical_issue_cycle", launch)
            self.assertIn("minimum_issue_cycle", launch)

    def test_accepted_sequences(self):
        cases = [[addi(5, 0, 31), config(), HALT],  # config is synchronous, not a transfer
                 # Captured scalar operands and the global base may change while pending.
                 [*setup(), transfer(), *word_base(1, 0x90001000), addi(6, 0, 32), addi(2, 0, 32),
                  addi(5, 0, 1), config(), wait(), addi(5, 0, 0), config(), HALT]]
        cases += [[*setup(), transfer(channel=c), wait(c), transfer("store", c), wait(c), HALT] for c in range(8)]
        for ops in cases:
            with self.subTest(ops=ops):
                self.check_consumers(ops, True)

    def test_pending_is_released_only_by_its_wait(self):
        prefix = [*setup(), transfer()]
        for ops in ([*prefix, wait(1), HALT], [*prefix, transfer(channel=1), wait(1), HALT],
                    [*prefix, addi(5, 0, 1), config(), vls("vload"), wait(), HALT],
                    [*prefix, MARKER, wait(), HALT], [*prefix, HALT], [*prefix, delay(100), HALT],
                    [wait(), HALT], [*prefix, wait(), wait(), HALT]):
            with self.subTest(ops=ops):
                self.check_consumers(ops, False)

    def test_operand_domain_boundaries_and_unknowns(self):
        for vmem, offset, size, base in ((0, 0, 32, 0), (392192, 0xffffffe0, 32, 31), (392192, 0x90000000, 4096, 31)):
            self.check_consumers([*setup(vmem, offset, size, base), transfer(), wait(), HALT], True)
        for vmem, offset, size, base in ((1, 0, 32, 0), (393216, 0, 32, 0), (65528, 0, 64, 0),
                                         (0, 1, 32, 0), (0, 0xffffffe0, 64, 0), (0, 0, 0, 0),
                                         (0, 0, 31, 0), (0, 0, 33, 0), (0, 0, 4128, 0), (0, 0, 32, 32)):
            with self.subTest(vmem=vmem, offset=offset, size=size, base=base):
                self.check_consumers([*setup(vmem, offset, size, base), transfer(), wait(), HALT], False)
        self.check_consumers([addi(6, 0, 0), transfer(), wait(), HALT], False)

    def test_final_boundaries_recheck_the_wait(self):
        checked = selected(WAITED, VERIFY, dma="wait")
        self.assertEqual(checked.returncode, 0, checked.stderr)
        # Replace the wait with a NOP, keeping the state chain intact.
        unsafe = re.sub(r'(%\w+) = "atlas.dma_wait"\((%\w+)\)[^\n]*',
                        lambda m: m[1] + ' = "atlas.alu_imm"(' + m[2] + ') <{dst = 0 : i32, immediate = 0 : i32, '
                        'kind = "addi", src = 0 : i32}> : (!atlas.state) -> !atlas.state', checked.stdout)
        self.assertNotEqual(unsafe, checked.stdout)
        for tool, args in ((OPT, [VERIFY]), (EMIT, []), (OPT, ["--convert-atlas-to-llvm"]),
                           (OPT, ["--convert-atlas-to-llvm-calls"])):
            with self.subTest(args=args):
                self.assertNotEqual(run(tool, unsafe, *args).returncode, 0)
        staged = run(OPT, checked.stdout, "--convert-atlas-to-llvm-calls")
        self.assertEqual(staged.returncode, 0, staged.stderr)
        self.assertEqual(run(OPT, staged.stdout, "--finalize-atlas-llvm-calls").returncode, 0)
        # Drop the staged wait call and renumber the remaining words.
        lines, index = [], 0
        for line in staged.stdout.splitlines():
            if 'atlas.source_op = "atlas.dma_wait"' in line:
                continue
            if "atlas.word_index =" in line:
                line = re.sub(r"atlas.word_index = [0-9]+ : i32", f"atlas.word_index = {index} : i32", line)
                index += 1
            lines.append(line)
        self.assertEqual(len(lines), len(staged.stdout.splitlines()) - 1)
        finalized = run(OPT, "\n".join(lines), "--finalize-atlas-llvm-calls")
        self.assertNotEqual(finalized.returncode, 0)
        self.assertIn("DMA", finalized.stderr)

    def test_dma_policy_must_be_explicit_and_known(self):
        unselected = selected(WAITED, "--insert-atlas-delays")
        self.assertNotEqual(unselected.returncode, 0)
        self.assertIn("dma=wait", unselected.stderr)
        for policy in ("latency", "none", "true"):
            with self.subTest(policy=policy):
                result = selected(WAITED, VERIFY, dma=policy)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("unsupported DMA policy", result.stderr)
        plain = selected([HALT], VERIFY)
        self.assertEqual(plain.returncode, 0, plain.stderr)
        self.assertNotIn("dma =", plain.stdout)
        self.assertEqual(json.loads(run(EMIT, plain.stdout, "--rtl-timing-json").stdout)["resolver"]["dma_policy"],
                         "none")


if __name__ == "__main__":
    unittest.main()
