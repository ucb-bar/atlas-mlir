"""Original virtual semantics versus actual compiler schedules, including DMA."""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
import os
import subprocess
import unittest

from test_virtual_evaluator_core import compare_result, selected_core, tile_bytes
from atlas_virtual_evaluator import EvaluationResult, Tile, compare_results, evaluate, operation_name, parse_program
from test_virtual_evaluator_cfg_core import cases as cfg_cases, runtime_inputs as cfg_inputs
from test_virtual_evaluator_dma_core import SOURCES as DMA_SOURCES, case as dma_case, mapped, patterned, READY_CODES, READY_SUM, TWICE_BF16
from test_virtual_evaluator_mxu_core import case as mxu_case, source_for_unit, inputs_with_guards, panel, identity_weight
from test_virtual_evaluator_vpu_core import SOURCE as VPU_SOURCE, vpu_inputs
from test_virtual_dma import STATE, dma_load, dma_await, dma_store, dma_wait, wrap
from test_virtual_lowering import BIN, ROOT, emitted, lower, object_words, run


OUTPUT_BASE = 0x90004000
RANDOM_SEEDS = range(4)
DUAL = (ROOT / "test/examples/virtual_mxu_accumulation.mlir").read_text().replace("atlas.output_dram_base = 2415923200", "atlas.output_dram_base = 2415935488")
DUAL = DUAL.replace("%io2, %w =", "%io2, %w0 =").replace("(%io2, %w)", "(%inputs_ready, %w0)").replace("(%io3, %w)", "(%io3, %w1)")
DUAL = DUAL.replace('%io3, %weight0 =', '%inputs_ready, %w1 = "atlas.virtual_input_fp8"(%io2) {index = 2 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_fp8)\n    %io3, %weight0 =')
TWO_DMA = wrap([
    f'%io0 = "atlas.virtual_start"() : () -> {STATE}',
    "%a_addr = arith.constant -2147483648 : i32",
    "%b_addr = arith.constant -2147481600 : i32",
    "%addr = arith.constant -2147475456 : i32",
    "%size = arith.constant 2048 : i32",
    dma_load("bf16", addr="a_addr", handle="a_event"),
    dma_load("bf16", before="io1", after="io2", addr="b_addr", handle="b_event"),
    dma_await("bf16", before="io2", after="io3", handle="b_event", result="b"),
    dma_await("bf16", before="io3", after="io4", handle="a_event", result="a"),
    '%positive = "atlas.virtual_vpu_unary"(%a) {kind = "relu"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16',
    '%sum = "atlas.virtual_vpu_binary"(%positive, %b) {kind = "add"} : (!atlas.virtual_bf16, !atlas.virtual_bf16) -> !atlas.virtual_bf16',
    dma_store("bf16", before="io4", after="io5", value="sum"),
    dma_wait(before="io5", after="io6"),
], final="io6")


@lru_cache(maxsize=None)
def schedules(source: str) -> tuple[tuple[str, str], ...]:
    variants = []
    for seed in (None, *RANDOM_SEEDS):
        option = "--schedule-atlas-virtual" + (f"=random-seed={seed}" if seed is not None else "")
        result = run("atlas-opt", source, option, "--verify-atlas-virtual-stream")
        if result.returncode:
            raise AssertionError(result.stderr)
        variants.append(("default" if seed is None else f"seed{seed}", result.stdout))
    return tuple(variants)


def order(program) -> tuple:
    return tuple((operation_name(op), tuple(sorted((name, str(value)) for name, value in (*op.attributes.items(), *op.properties.items()) if name != "op_name__"))) for op in program.operations)


def shape(program) -> tuple:
    positions = {block: index for index, block in enumerate(program.blocks)}
    return tuple((tuple(str(arg.type) for arg in block.args), operation_name(tuple(block.ops)[-1]),
                  tuple(positions[successor] for successor in tuple(block.ops)[-1].successors)) for block in program.blocks)


def workloads(phase: int):
    inputs, summed, positive = vpu_inputs(phase)
    yield "vpu", VPU_SOURCE.read_text(), inputs, EvaluationResult({0: inputs.tiles[0], 1: inputs.tiles[0], 2: summed, 3: positive}, inputs.memory)
    for unit in (0, 1):
        for name in ("legacy_pack", "seed_bf16" if unit == 0 else "seed_fp8"):
            inputs, outputs, _ = mxu_case(name, phase, unit)
            yield f"mxu{unit}_{name}", source_for_unit(name, unit), inputs, EvaluationResult(outputs, inputs.memory)
    for name in ("pending_work", "mxu_fp8_0", "mxu_bf16_1"):
        inputs, memory, _ = dma_case(name, phase)
        yield f"dma_{name}", DMA_SOURCES[name], inputs, EvaluationResult({}, memory)
    for same_unit in (False, True):
        source = DUAL
        if same_unit:
            source = source.replace("unit = 1 : i32", "unit = 0 : i32").replace("virtual_mxu_weight<1>", "virtual_mxu_weight<0>").replace("virtual_mxu_acc<1>", "virtual_mxu_acc<0>")
        x = panel(phase)
        inputs = inputs_with_guards({0: x, 1: identity_weight(), 2: identity_weight(3)}, 2)
        first = Tile("bf16", tuple(TWICE_BF16[code] for code in x.bits))
        second = Tile("bf16", tuple(TWICE_BF16[x.bits[row * 32 + (col + 3) % 32]] for row in range(32) for col in range(32)))
        yield f"two_chains_{'same_unit' if same_unit else 'both_units'}", source, inputs, EvaluationResult({0: first, 1: second}, inputs.memory)
    ready = patterned(READY_CODES, phase)
    payload = tile_bytes(Tile("bf16", tuple(READY_SUM[bits] for bits in ready.bits)))
    inputs, memory = mapped({0x80000000: tile_bytes(ready), 0x80000800: tile_bytes(Tile("bf16", (0x3F00,) * 1024)),
                             0x80002000: b"\xA5" * 2048}, {0x80002000: payload})
    yield "two_dma_reverse_await", TWO_DMA, inputs, EvaluationResult({}, memory)
    for case in cfg_cases():
        if case.name not in ("runtime_i1", "tile_swap", "relu_loop"):
            continue
        for controls, indices in case.variants:
            inputs = cfg_inputs(case.widths, controls, phase)
            tiles = dict(inputs.tiles)
            tiles[2] = Tile("bf16", tuple(0 if bits & 0x8000 else bits for bits in tiles[0].bits))
            literal = EvaluationResult({index: tiles[selected] for index, selected in enumerate(indices)}, inputs.memory)
            yield f"cfg_{case.name}_{controls}", case.source, inputs, literal


class VirtualEvaluatorSchedulingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            probe = subprocess.run([str(BIN / "atlas-opt"), "--help"], capture_output=True, text=True, timeout=30)
            available = probe.returncode == 0 and "--schedule-atlas-virtual" in probe.stdout
        except (OSError, subprocess.TimeoutExpired):
            available = False
        if not available:
            message = "ATLAS_OOT_BIN_DIR must select a compiler with --schedule-atlas-virtual"
            if os.environ.get("ATLAS_REQUIRE_VIRTUAL_SCHEDULER") == "1":
                raise AssertionError(message)
            raise unittest.SkipTest(message)

    def test_default_and_random_schedules_preserve_complete_reference_results(self) -> None:
        for phase in (0, 3):
            for name, source, inputs, literal in workloads(phase):
                original = evaluate(parse_program(source), inputs)
                compare_results(literal, original)
                for variant, scheduled in schedules(source):
                    with self.subTest(workload=name, phase=phase, schedule=variant):
                        compare_results(original, evaluate(parse_program(scheduled), inputs))

    def test_schedules_preserve_structure_and_change_operation_order(self) -> None:
        changed = diverse = False
        seen = set()
        for _, source, _, _ in workloads(0):
            if source in seen:
                continue
            seen.add(source)
            original = parse_program(source)
            orders = set()
            for variant, scheduled in schedules(source):
                with self.subTest(schedule=variant, source=original.function.sym_name.data):
                    parsed = parse_program(scheduled)
                    self.assertEqual(shape(parsed), shape(original))
                    self.assertEqual(Counter(order(parsed)), Counter(order(original)))
                    changed |= order(parsed) != order(original)
                    orders.add(order(parsed))
            diverse |= len(orders) > 1
        self.assertTrue(changed, "all compiler schedules retained source order")
        self.assertTrue(diverse, "random seeds produced no distinct operation orders")

    def test_scheduled_machine_words_match_llvm_objects(self) -> None:
        seen = set()
        for name, source, _, _ in workloads(0):
            if source in seen:
                continue
            seen.add(source)
            for variant, scheduled in schedules(source)[:2]:
                with self.subTest(workload=name, schedule=variant):
                    machine = lower(scheduled)
                    self.assertEqual(object_words(machine), emitted(machine))

    def test_selected_core_original_and_scheduled_programs_match_original_reference(self) -> None:
        with selected_core(max_cycles=100000) as execute:
            for name, source, inputs, literal in workloads(3):
                if name not in ("vpu", "mxu0_legacy_pack", "dma_pending_work", "cfg_tile_swap_(3,)"):
                    continue
                original = evaluate(parse_program(source), inputs)
                compare_results(literal, original)
                for variant, text in (("original", source), *schedules(source)[:2]):
                    with self.subTest(workload=name, schedule=variant):
                        machine = lower(text)
                        words = object_words(machine)
                        self.assertEqual(words, emitted(machine))
                        preload = [(region.address, region.data) for region in inputs.memory]
                        if original.outputs:
                            preload.append((OUTPUT_BASE, b"\xA5" * ((max(original.outputs) + 1) * 2048)))
                        actual = execute(words, preload)
                        self.assertTrue(actual.halted, "selected core did not halt")
                        compare_result(original, actual.slave.captured, output_base=OUTPUT_BASE)


if __name__ == "__main__":
    unittest.main()
