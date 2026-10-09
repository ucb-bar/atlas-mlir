"""Logical tile DMA events lower at their launch and completion points."""

from __future__ import annotations

import importlib.util
from itertools import product
import os
from pathlib import Path
import re
import unittest

from test_virtual_dma import (
    STATE, constants, copy, dma_load, dma_await, dma_store, dma_wait, tile, wrap,
)
from test_virtual_lowering import BIN, ROOT, emitted, lower, object_words, run
from test_virtual_mxu_handles import program, load, reset, readout
from test_virtual_mxu_lowering import instructions


MARKER = "atlas.virtual_dma_transfer"
STAGING_WORD = 131072
MXU_EXAMPLES = {
    "fp8": ROOT / "test/examples/virtual_dma_mxu.mlir",
    "bf16": ROOT / "test/examples/virtual_dma_mxu_bf16.mlir",
}


def observed(entries: list[dict]) -> list[tuple[dict, dict[int, int]]]:
    registers = {index: 0 for index in range(32)}
    result = []
    for entry in entries:
        result.append((entry, registers.copy()))
        fields = entry["fields"]
        operation = entry["operation"]
        if operation == "atlas.upper" and fields["kind"] == "lui":
            registers[fields["dst"]] = fields["immediate"] << 12
        elif operation == "atlas.alu_imm" and fields["kind"] == "addi":
            registers[fields["dst"]] = registers[fields["src"]] + fields["immediate"]
        elif operation == "atlas.alu_reg" and fields["kind"] == "add":
            registers[fields["dst"]] = registers[fields["lhs"]] + registers[fields["rhs"]]
        registers = {key: value & 0xffffffff for key, value in registers.items()}
        registers[0] = 0
    return result


def independent_work() -> str:
    pure = (f'    %independent = "atlas.virtual_vpu_unary"(%seed) '
            f'{{kind = "relu"}} : ({tile("bf16")}) -> {tile("bf16")}')
    source = program(
        *constants("bf16"), load("io2", "s0", "weight"),
        dma_load("bf16", before="s0", after="s1"),
        "    %later_addr = arith.constant -2147481600 : i32",
        "    %later_size = arith.constant 2048 : i32",
        reset("s1", "s2", "acc0"),
        dma_await("bf16", before="s2", after="s3"),
        readout("s3", "s4", "product", "acc0"),
        dma_store("bf16", before="s4", after="s5").replace(
            "%addr, %size", "%later_addr, %later_size"),
        pure, dma_wait(before="s5", after="s6"), final_state="s6",
    )
    return source.replace("atlas.output_dram_base = 2415923200",
                          "atlas.output_dram_base = 2415927296")


def replace_operation(line: str, name: str, fields: str) -> str:
    return re.sub(r'"atlas\.[^"]+"(\([^)]*\)) .*? : \(',
                  lambda match: f'"atlas.{name}"{match[1]} {{{fields}}} : (', line)


class VirtualDMALoweringTest(unittest.TestCase):
    def setUp(self) -> None:
        self.assertTrue((BIN / "atlas-opt").is_file(), "build atlas-opt first")
        self.assertTrue((BIN / "atlas-emit").is_file(), "build atlas-emit first")

    def checked(self, source: str) -> tuple[str, list[dict]]:
        machine = lower(source)
        self.assertIn("atlas.generated_from_virtual", machine)
        self.assertNotIn('"atlas.virtual_', machine)
        for option in ("--verify-atlas-machine-stream", "--verify-atlas-generated-schedule"):
            result = run("atlas-opt", machine, option)
            self.assertEqual(result.returncode, 0, result.stderr)
        entries = instructions(machine)
        self.assertEqual(tuple(entry["word_u32"] for entry in entries), emitted(machine))
        return machine, entries

    def test_full_tiles_use_private_staging_and_matching_channels(self) -> None:
        for fmt, halves, size in (("fp8", 1, 1024), ("bf16", 2, 2048)):
            with self.subTest(fmt=fmt):
                machine, entries = self.checked(copy(fmt))
                launches = []
                vector_addresses = {"atlas.vload": [], "atlas.vstore": []}
                for index, (entry, registers) in enumerate(observed(entries)):
                    fields = entry["fields"]
                    operation = entry["operation"]
                    if operation == "atlas.dma":
                        launches.append((fields["direction"], fields["channel"],
                                         registers[fields["reg"]], registers[fields["dram"]],
                                         registers[fields["size"]], fields[MARKER]))
                        self.assertEqual((fields["reg"], fields["dram"], fields["size"]), (4, 7, 9))
                    elif operation in vector_addresses:
                        vector_addresses[operation].append(registers[fields["base"]] + fields["offset"] * 32)
                        wait = entries[index + 1]
                        self.assertEqual(wait["operation"], "atlas.delay")
                        self.assertEqual(wait["fields"]["cycles"], 256)
                self.assertEqual([launch[:5] for launch in launches],
                                 [("load", 0, STAGING_WORD, 0x80000000, size),
                                  ("store", 1, STAGING_WORD, 0x80000000, size)])
                self.assertNotEqual(launches[0][5], launches[1][5])
                waits = [entry["fields"] for entry in entries if entry["operation"] == "atlas.dma_wait"]
                self.assertEqual([(wait["channel"], wait[MARKER]) for wait in waits],
                                 [(launch[1], launch[5]) for launch in launches])
                expected = [STAGING_WORD + 256 * half for half in range(halves)]
                self.assertEqual(vector_addresses["atlas.vload"], expected)
                self.assertEqual(vector_addresses["atlas.vstore"], expected)
                reparsed = run("atlas-opt", machine)
                self.assertEqual(reparsed.returncode, 0, reparsed.stderr)
                self.assertEqual(reparsed.stdout, machine)

    def test_launch_completion_separation_preserves_mxu_and_scalar_register_reuse(self) -> None:
        _, entries = self.checked(independent_work())
        launch = next(i for i, entry in enumerate(entries)
                      if entry["operation"] == "atlas.dma" and MARKER in entry["fields"])
        wait = next(i for i in range(launch + 1, len(entries))
                    if entries[i]["operation"] == "atlas.dma_wait")
        interval = entries[launch + 1:wait]
        self.assertTrue(any(entry["operation"] == "atlas.mxu_matmul" for entry in interval))
        captures = [entry["fields"] for entry in entries[max(0, launch - 5):launch]
                    if entry["operation"] == "atlas.alu_imm"
                    and entry["fields"]["dst"] in (7, 9)
                    and entry["fields"]["src"] != 0]
        self.assertEqual({fields["dst"] for fields in captures}, {7, 9})
        sources = {fields["src"] for fields in captures}
        self.assertTrue(any(entry["fields"].get("dst") in sources for entry in interval))
        store = next(i for i, entry in enumerate(entries)
                     if entry["operation"] == "atlas.dma"
                     and MARKER in entry["fields"] and entry["fields"]["direction"] == "store")
        store_wait = next(i for i in range(store + 1, len(entries))
                          if entries[i]["operation"] == "atlas.dma_wait")
        self.assertTrue(any(entry["operation"] == "atlas.vpu_unary"
                            for entry in entries[store + 1:store_wait]))
        snapshots = observed(entries)
        self.assertEqual(snapshots[launch][1][7], 0x80000000)
        self.assertEqual(snapshots[wait][1][7], 0x80000000)
        self.assertEqual(snapshots[store][1][7], 0x80000800)

    def test_explicit_dma_mxu_chain_preserves_dataflow_on_both_units(self) -> None:
        for fmt, unit in product(MXU_EXAMPLES, (0, 1)):
            with self.subTest(fmt=fmt, unit=unit):
                source = MXU_EXAMPLES[fmt].read_text()
                halves = 1 if fmt == "fp8" else 2
                virtual = source.replace("unit = 0 : i32", f"unit = {unit} : i32")
                for handle in ("weight", "acc"):
                    virtual = virtual.replace(f"virtual_mxu_{handle}<0>",
                                              f"virtual_mxu_{handle}<{unit}>")
                _, entries = self.checked(virtual)
                snapshots = observed(entries)
                dma = [(entry["fields"], registers)
                       for entry, registers in snapshots
                       if entry["operation"] == "atlas.dma"]
                self.assertEqual([(fields["direction"], fields["channel"],
                                   registers[fields["reg"]], registers[fields["dram"]],
                                   registers[fields["size"]])
                                  for fields, registers in dma],
                                 [("load", 0, STAGING_WORD, 0x90000000, 1024),
                                  ("load", 0, STAGING_WORD, 0x90000400, 1024),
                                  ("store", 1, STAGING_WORD, 0x90001000, 1024 * halves)])
                waits = [entry["fields"] for entry in entries
                         if entry["operation"] == "atlas.dma_wait"]
                self.assertEqual([(fields["channel"], fields[MARKER]) for fields in waits],
                                 [(fields["channel"], fields[MARKER]) for fields, _ in dma])
                self.assertEqual(len({fields[MARKER] for fields, _ in dma}), 3)
                flow = [entry["operation"] for entry in entries
                        if entry["operation"] in ("atlas.dma", "atlas.dma_wait", "atlas.vload", "atlas.vstore")
                        or entry["operation"].startswith("atlas.mxu_")]
                self.assertEqual(flow, ["atlas.dma", "atlas.dma_wait", "atlas.vload"] * 2
                                 + ["atlas.mxu_push", "atlas.mxu_matmul", "atlas.mxu_matmul",
                                    "atlas.mxu_pop"] + ["atlas.vstore"] * halves
                                 + ["atlas.dma", "atlas.dma_wait"])
                loads = [entry["fields"] for entry in entries
                         if entry["operation"] == "atlas.vload"]
                self.assertNotEqual(loads[0]["dst"], loads[1]["dst"])
                mxu = [entry["fields"] for entry in entries
                       if entry["operation"].startswith("atlas.mxu_")]
                self.assertTrue(all(fields["unit"] == unit for fields in mxu))
                self.assertEqual(mxu[0]["kind"], "weight_fp8")
                self.assertEqual(mxu[0]["src"], loads[1]["dst"])
                self.assertEqual([fields["src"] for fields in mxu[1:3]], [loads[0]["dst"]] * 2)
                self.assertEqual([fields["accumulate"] for fields in mxu[1:3]],
                                 [False, True])
                pop = mxu[-1]
                self.assertEqual(pop["format"], fmt)
                if fmt == "bf16":
                    self.assertGreaterEqual(pop["dst"], 32)
                    self.assertEqual(pop["dst"] % 2, 0)
                    self.assertEqual(pop["scale_reg"], 0)
                pop_index = next(index for index, entry in enumerate(entries)
                                 if entry["operation"] == "atlas.mxu_pop")
                scale = [(index, entry["fields"]) for index, entry in enumerate(entries)
                         if entry["operation"] == "atlas.scalar_load"
                         and entry["fields"]["kind"] == "seli"]
                self.assertEqual([(fields["dst"], fields["offset"]) for _, fields in scale],
                                 [(pop["scale_reg"], 129)] if fmt == "fp8" else [])
                self.assertTrue(all(index < pop_index for index, _ in scale))
                stores = [entry["fields"] for entry in entries
                          if entry["operation"] == "atlas.vstore"]
                self.assertEqual([fields["src"] for fields in stores],
                                 [pop["dst"] + half for half in range(halves)])
                addresses = {"atlas.vload": [], "atlas.vstore": []}
                for entry, registers in snapshots:
                    if entry["operation"] in addresses:
                        fields = entry["fields"]
                        addresses[entry["operation"]].append(registers[fields["base"]] + fields["offset"] * 32)
                self.assertEqual(addresses["atlas.vload"], [STAGING_WORD] * 2)
                self.assertEqual(addresses["atlas.vstore"],
                                 [STAGING_WORD + 256 * half for half in range(halves)])

    def test_later_implicit_boundary_transfers_keep_their_half_tile_size(self) -> None:
        source = copy().replace(dma_store("bf16"),
            f'    %hidden_io, %hidden = "atlas.virtual_input_fp8"(%io2) '
            f'{{index = 0 : i32}} : ({STATE}) -> ({STATE}, {tile("fp8")})\n'
            + dma_store("bf16", before="hidden_io"))
        _, entries = self.checked(source)
        legacy = [(entry, registers) for entry, registers in observed(entries)
                  if entry["operation"] == "atlas.dma" and MARKER not in entry["fields"]]
        self.assertEqual(len(legacy), 1)
        self.assertEqual(legacy[0][0]["fields"]["size"], 2)
        self.assertEqual(legacy[0][1][2], 1024)

    def test_staging_reuse_starts_after_each_completion(self) -> None:
        source = copy("fp8").replace(dma_store("fp8"),
            dma_load("fp8", before="io2", after="next_io", handle="next") + "\n"
            + dma_await("fp8", before="next_io", after="ready_io", handle="next", result="next_tile") + "\n"
            + dma_store("fp8", before="ready_io", value="next_tile"))
        _, entries = self.checked(source)
        transfers = [entry for entry in entries if entry["operation"] in ("atlas.dma", "atlas.dma_wait")]
        self.assertEqual([entry["operation"] for entry in transfers], ["atlas.dma", "atlas.dma_wait"] * 3)
        ids = [entry["fields"][MARKER] for entry in transfers]
        self.assertEqual(ids[::2], ids[1::2])
        self.assertEqual(len(set(ids)), 3)
        self.assertTrue(all(registers[entry["fields"]["reg"]] == STAGING_WORD
                            for entry, registers in observed(entries) if entry["operation"] == "atlas.dma"))

    def test_explicit_loads_and_stores_cannot_overlap_the_control_mailbox(self) -> None:
        for fmt in ("fp8", "bf16"):
            source = copy(fmt, address=0x80000800).replace("@dma()", "@dma(%unused: i1)")
            source = source.replace("atlas.output_dram_base = 2415923200 : i64",
                                    "atlas.output_dram_base = 2415923200 : i64, "
                                    "atlas.control_dram_base = 2147483648 : i64")
            self.checked(source)
            load_overlap = source.replace("arith.constant 2147485696 : i32",
                                          "arith.constant -2147483648 : i32")
            store_overlap = source.replace(dma_store(fmt),
                "    %mailbox_addr = arith.constant -2147483648 : i32\n"
                + dma_store(fmt).replace("%addr, %size", "%mailbox_addr, %size"))
            for changed in (load_overlap, store_overlap):
                with self.subTest(fmt=fmt, store=changed == store_overlap):
                    result = run("atlas-opt", changed, "--lower-atlas-virtual-to-machine")
                    self.assertNotEqual(result.returncode, 0, result.stdout)
                    self.assertEqual(result.stdout, "")
                    self.assertIn("explicit DMA span overlaps the control mailbox", result.stderr)

    def test_branch_and_loop_complete_dma_before_cfg_edges(self) -> None:
        branch = wrap([
            f'    %io0 = "atlas.virtual_start"() : () -> {STATE}', *constants("bf16"),
            "    %choose = arith.constant true",
            f"    cf.cond_br %choose, ^left(%io0 : {STATE}), ^right(%io0 : {STATE})",
            f"  ^left(%left_io: {STATE}):",
            dma_load("bf16", before="left_io", after="left_pending", handle="left_event"),
            dma_await("bf16", before="left_pending", after="left_ready", handle="left_event", result="left_tile"),
            f"    cf.br ^join(%left_ready, %left_tile : {STATE}, {tile('bf16')})",
            f"  ^right(%right_io: {STATE}):",
            dma_load("bf16", before="right_io", after="right_pending", handle="right_event"),
            dma_await("bf16", before="right_pending", after="right_ready", handle="right_event", result="right_tile"),
            f"    cf.br ^join(%right_ready, %right_tile : {STATE}, {tile('bf16')})",
            f"  ^join(%joined_io: {STATE}, %joined_tile: {tile('bf16')}):",
            dma_store("bf16", before="joined_io", value="joined_tile"), dma_wait(),
        ])
        loop = wrap([
            f'    %io0 = "atlas.virtual_start"() : () -> {STATE}', *constants("bf16"),
            "    %zero = arith.constant 0 : i32", "    %one = arith.constant 1 : i32",
            f"    cf.br ^loop(%io0, %zero : {STATE}, i32)",
            f"  ^loop(%loop_io: {STATE}, %n: i32):",
            dma_load("bf16", before="loop_io"), dma_await("bf16"),
            "    %next = arith.addi %n, %one : i32",
            "    %again = arith.cmpi slt, %next, %one : i32",
            f"    cf.cond_br %again, ^loop(%io2, %next : {STATE}, i32), "
            f"^exit(%io2, %tile : {STATE}, {tile('bf16')})",
            f"  ^exit(%ready: {STATE}, %value: {tile('bf16')}):",
            dma_store("bf16", before="ready", value="value"), dma_wait(),
        ])
        for source in (branch, loop):
            with self.subTest(loop=source == loop):
                machine, entries = self.checked(source)
                self.assertTrue(any(entry["operation"] == "atlas.branch" for entry in entries))
                self.assertTrue(any(entry["operation"] == "atlas.jump" for entry in entries))
                self.assertIn(MARKER, machine)

    def test_generated_pending_intervals_reject_missing_wrong_wait_and_unsafe_work(self) -> None:
        machine, _ = self.checked(independent_work())
        lines = machine.splitlines()
        launch = next(i for i, line in enumerate(lines) if '"atlas.dma"' in line and MARKER in line)
        wait = next(i for i in range(launch + 1, len(lines)) if '"atlas.dma_wait"' in lines[i])
        scalar = next(i for i in range(launch + 1, wait) if '"atlas.alu_imm"' in lines[i])
        mxu = next(i for i in range(launch + 1, wait) if '"atlas.mxu_matmul"' in lines[i])
        replacements = [
            ("missing wait", wait, replace_operation(lines[wait], "alu_imm", 'dst = 0 : i32, src = 0 : i32, immediate = 0 : i32, kind = "addi"')),
            ("wrong channel", wait, lines[wait].replace("channel = 0 : i32", "channel = 1 : i32")),
            ("wrong identity", wait, re.sub(rf"{MARKER} = \d+ : i32", f"{MARKER} = 999 : i32", lines[wait])),
            ("VMEM access", mxu, replace_operation(lines[mxu], "vload", 'dst = 0 : i32, base = 4 : i32, offset = 0 : i32, format = "raw"')),
            ("extra DMA", scalar, replace_operation(lines[scalar], "dma", 'direction = "load", channel = 0 : i32, reg = 4 : i32, dram = 7 : i32, size = 9 : i32')),
            ("DMA configuration", scalar, replace_operation(lines[scalar], "dma_config", 'channel = 0 : i32, base_reg = 5 : i32')),
        ]
        for index in (launch, wait):
            replacement, removed = re.subn(
                rf"{re.escape(MARKER)} = \d+ : i32, ", "", lines[index], count=1)
            self.assertEqual(removed, 1)
            self.assertNotIn(MARKER, replacement)
            self.assertIn("atlas.virtual_tile_command =", replacement)
            replacements.append((f"missing marker at {index}", index, replacement))
        for marker in ('-1 : i32', '0 : i64', '"bad"'):
            replacements.append((f"malformed marker {marker}", launch,
                                 re.sub(rf"{MARKER} = \d+ : i32", f"{MARKER} = {marker}", lines[launch])))
        orphan = next(i for i in range(launch) if '"atlas.dma_wait"' in lines[i])
        replacements.append(("orphan marker", orphan,
                             lines[orphan].replace("{atlas.virtual_tile_command",
                                                   f"{{{MARKER} = 999 : i32, atlas.virtual_tile_command", 1)))
        changes = []
        for name, index, replacement in replacements:
            changed = lines.copy()
            changed[index] = replacement
            self.assertNotEqual(changed, lines)
            changes.append((name, "\n".join(changed)))
        changes.append(("duplicate transfer identity", machine.replace(
            f"{MARKER} = 1 : i32", f"{MARKER} = 0 : i32")))
        first = next(i for i, line in enumerate(lines) if '"atlas.alu_imm"' in line)
        redirect = lines.copy()
        redirect[first] = replace_operation(lines[first], "branch",
            f'kind = "beq", lhs = 0 : i32, rhs = 0 : i32, offset_bytes = {(scalar - first) * 2} : i32')
        redirect[first + 1] = replace_operation(lines[first + 1], "alu_imm",
            'kind = "addi", dst = 0 : i32, src = 0 : i32, immediate = 0 : i32')
        changes.append(("redirect into pending interval", "\n".join(redirect)))
        for name, changed in changes:
            self.assertNotEqual(changed, machine)
            for tool, options in (("atlas-emit", ()),
                                  ("atlas-opt", ("--verify-atlas-generated-schedule",)),
                                  ("atlas-opt", ("--convert-atlas-to-llvm",))):
                with self.subTest(case=name, tool=tool):
                    result = run(tool, changed, *options)
                    self.assertNotEqual(result.returncode, 0, result.stdout)
                    self.assertTrue(result.stderr)
                    self.assertEqual(result.stdout, "")
                    if name == "orphan marker":
                        self.assertIn("DMA.WAIT must match the pending DMA channel and transfer ID", result.stderr)

    def test_pending_seli_writes_use_the_separate_extended_register_file(self) -> None:
        machine, _ = self.checked(independent_work())
        lines = machine.splitlines()
        launch = next(i for i, line in enumerate(lines) if '"atlas.dma"' in line and MARKER in line)
        scalar = next(i for i in range(launch + 1, len(lines)) if '"atlas.alu_imm"' in lines[i])
        for reg in (4, 7, 9):
            changed = lines.copy()
            state = re.search(r'"atlas.alu_imm"\((%\w+)\)', lines[scalar])[1]
            changed[scalar] = lines[scalar].replace(f"({state})", "(%seli_probe)", 1)
            changed.insert(scalar,
                f'  %seli_probe = "atlas.scalar_load"({state}) '
                f'{{kind = "seli", dst = {reg} : i32, base = 0 : i32, offset = 1 : i32}} '
                ': (!atlas.state) -> !atlas.state')
            self.assertNotEqual(changed[scalar], lines[scalar])
            for tool, options in (("atlas-emit", ()),
                                  ("atlas-opt", ("--verify-atlas-generated-schedule",))):
                with self.subTest(reg=reg, tool=tool):
                    result = run(tool, "\n".join(changed), *options)
                    self.assertEqual(result.returncode, 0, result.stderr)

    def test_structured_llvm_preserves_both_formats(self) -> None:
        for fmt in ("fp8", "bf16"):
            with self.subTest(fmt=fmt):
                machine, _ = self.checked(copy(fmt))
                structured = run("atlas-opt", machine, "--convert-atlas-to-llvm-calls")
                self.assertEqual(structured.returncode, 0, structured.stderr)
                self.assertIn(MARKER, structured.stdout)
                finalized = run("atlas-opt", structured.stdout, "--finalize-atlas-llvm-calls")
                self.assertEqual(finalized.returncode, 0, finalized.stderr)
                direct = run("atlas-opt", machine, "--convert-atlas-to-llvm")
                self.assertEqual(direct.returncode, 0, direct.stderr)
                self.assertEqual(finalized.stdout, direct.stdout)

    def test_llvm_objects_preserve_both_formats(self) -> None:
        for fmt in ("fp8", "bf16"):
            with self.subTest(fmt=fmt):
                machine = lower(copy(fmt))
                self.assertEqual(object_words(machine), emitted(machine))

    def test_selected_assembler_matches_dma_and_vector_words(self) -> None:
        root = os.environ.get("ATLAS_ASSEMBLER_ROOT")
        if not root:
            self.skipTest("set ATLAS_ASSEMBLER_ROOT for selected assembler")
        spec = importlib.util.spec_from_file_location("selected_virtual_dma_assembler", Path(root) / "assembler.py")
        assert spec and spec.loader
        assembler = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(assembler)
        for fmt in ("fp8", "bf16"):
            for entry in instructions(lower(copy(fmt))):
                operation, fields = entry["operation"], entry["fields"]
                if operation == "atlas.dma":
                    if fields["direction"] == "load":
                        name, args = "DMA_LOAD", (fields["reg"], fields["dram"], fields["size"], fields["channel"])
                    else:
                        name, args = "DMA_STORE", (fields["dram"], fields["reg"], fields["size"], fields["channel"])
                elif operation == "atlas.dma_wait":
                    name, args = "DMA_WAIT", (fields["channel"],)
                elif operation == "atlas.dma_config":
                    name, args = "DMA_CONFIG", (fields["base_reg"], fields["channel"])
                elif operation in ("atlas.vload", "atlas.vstore"):
                    name = "VLOAD" if operation == "atlas.vload" else "VSTORE"
                    args = (fields["dst" if operation == "atlas.vload" else "src"], fields["base"], fields["offset"])
                else:
                    continue
                with self.subTest(fmt=fmt, operation=name):
                    self.assertEqual(entry["word_u32"], getattr(assembler, name)(*args))


if __name__ == "__main__":
    unittest.main()
