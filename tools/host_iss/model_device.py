"""Diagnostic MMIO bridge: real host buffers, admitted instruction objects.

Only the hash-bound 54-kernel ResNet development bundle is supported.
Derived from the validated diagnostic bridge; this portable adaptation has not
executed the whole ELF. No TileLink or SoC timing claim.
"""

from atlas_tooling import write_json, sha as digest
from collections import Counter
from dataclasses import replace
import json
import struct
import time

from atlas_host_layout import (ATLAS_CSR_BASE, ATLAS_CSR_CYCLES, ATLAS_CSR_RETIRED, ATLAS_CSR_STATUS, ATLAS_CSR_PC,
    ATLAS_CSR_DBG0, ATLAS_CSR_DBG1, ATLAS_CSR_EXEC_CONTROL, ATLAS_DRAM_BASE, ATLAS_IMEM_BASE, ATLAS_IMEM_BYTES,
    recheck as recheck_layout)
from atlas_path_guard import checked
from run_model_tile import configure_model, model_binding
from bundle import runtime_program


class Device:
    def __init__(self, remote, out, wall_deadline, bundle, model_root, cycle_cap=20000000):
        self.remote, self.out = remote, checked(out)
        self.deadline, self.cycle_cap = wall_deadline, cycle_cap
        self.bundle = bundle
        self.pins = {}
        self.model_root = checked(model_root)
        _, self.torch, self.assemble, _ = configure_model(self.model_root)
        self.binding = model_binding(self.model_root)
        from npu_model.configs.hardware.default import DefaultHardwareConfig
        from npu_model.hardware import Core
        from npu_model.hardware.dma import DmaExecutionUnit
        from npu_model.logging.logger import Logger, LoggerConfig

        config = DefaultHardwareConfig()
        config.arch_state_config = replace(config.arch_state_config, numerics="rtl", randomize_init=True, init_seed=42)
        self.logger = Logger(LoggerConfig(filename="/dev/null"))
        self.core = Core(config, self.logger)
        self.core.arch_state.halted = True
        self.dma = next(exu for exu in self.core.exus if isinstance(exu, DmaExecutionUnit))
        self.imem = bytearray(ATLAS_IMEM_BYTES)
        self.written = set()
        self.completed = []
        self.programs = []
        for index in range(54):
            receipt = json.loads(self.payload(f"kernels/{index:02d}/model/receipt.json"))
            assert receipt["status"] == "passed_model_only"
            source = self.payload(f"kernels/{index:02d}/model/program.S").decode()
            original = self.payload(f"kernels/{index:02d}/model/program.bin")
            program, words = runtime_program(source, original, self.assemble)
            self.programs.append((program, words, receipt))

    def payload(self, name):
        raw = self.bundle.read(name)
        self.pins[name] = digest(raw)
        return raw

    def read32(self, address):
        assert address % 4 == 0
        if ATLAS_IMEM_BASE <= address < ATLAS_IMEM_BASE + ATLAS_IMEM_BYTES:
            return struct.unpack_from("<I", self.imem, address - ATLAS_IMEM_BASE)[0]
        table = {ATLAS_CSR_CYCLES: 0xC00, ATLAS_CSR_RETIRED: 0xC01, ATLAS_CSR_STATUS: 0xC02, ATLAS_CSR_PC: 0xC03,
            ATLAS_CSR_DBG0: 0xC10, ATLAS_CSR_DBG1: 0xC11}
        offset = address - ATLAS_CSR_BASE
        assert offset in table, ("unadmitted CSR read", hex(address))
        return self.core.arch_state.read_csrf(table[offset])

    def write32(self, address, value):
        assert address % 4 == 0 and 0 <= value < 1 << 32
        if ATLAS_IMEM_BASE <= address < ATLAS_IMEM_BASE + ATLAS_IMEM_BYTES:
            assert self.core.arch_state.halted
            offset = address - ATLAS_IMEM_BASE
            struct.pack_into("<I", self.imem, offset, value)
            self.written.add(offset // 4)
            return
        offset = address - ATLAS_CSR_BASE
        if offset == ATLAS_CSR_EXEC_CONTROL:
            assert value == 1, "only drained START is admitted"
            self.start()
            return
        table = {ATLAS_CSR_CYCLES: 0xC00, ATLAS_CSR_RETIRED: 0xC01, ATLAS_CSR_DBG0: 0xC10, ATLAS_CSR_DBG1: 0xC11}
        assert offset in table, ("unadmitted CSR write", hex(address))
        self.core.arch_state.write_csrf(table[offset], value)

    def start(self):
        index = len(self.completed)
        assert index < 54
        core, state = self.core, self.core.arch_state
        assert state.halted and not any(exu.has_in_flight for exu in core.exus)
        assert not any(state.check_flag(channel) for channel in range(8))
        program, words, receipt = self.programs[index]
        assert self.written == set(range(len(words) // 4))
        assert self.imem[: len(words)] == words, "actual host IMEM does not match admitted program"
        assert state.read_csrf(0xC10) == 0 and state.read_csrf(0xC11) == index + 1
        initial = self.payload(f"kernels/{index:02d}/model/initial-dram.bin")
        expected = self.payload(f"kernels/{index:02d}/model/expected-bf16-tiles.bin")
        actual_host_input = self.remote.read(ATLAS_DRAM_BASE, len(initial))
        assert actual_host_input == initial, "host staging differs from frozen native/model input"
        state.write_dram(ATLAS_DRAM_BASE,
            self.torch.frombuffer(bytearray(actual_host_input), dtype=self.torch.uint8).clone())
        # ScalarCore hostStart restarts PC/hold/delay and clears dmaBaseReg;
        # it preserves registers, VMEM, accumulators, DRAM and CSR counters.
        core.load_program(program)
        core.ifu.reset()
        core._reset_s1()
        state.pc = state.npc = state.execute_pc = 0
        state.redirect_requested = state.in_delay_slot = False
        state.current_uop = None
        state.base = 0
        state.halt_reason = None
        state._csr_values[0xC03] = 0
        state.halted = False
        before = core.cycle_count
        events = Counter()
        while not core.is_finished() and core.cycle_count - before < self.cycle_cap:
            if core.cycle_count % 1024 == 0 and time.monotonic() >= self.deadline:
                raise TimeoutError("model bridge wall cap")
            core.tick()
            event = self.dma.last_cycle
            if event.a:
                events["store_bytes" if event.a[2] else "load_bytes"] += 32
            if event.d is not None:
                events["response_beats"] += 1
        assert state.halted and state.halt_reason == "ecall"
        assert not any(exu.has_in_flight for exu in core.exus)
        assert not any(state.check_flag(channel) for channel in range(8))
        assert state.read_csrf(0xC10) == index + 1
        for key, count in receipt["requested_dma"].items():
            assert events[key] == count
        actual = state.read_dram(ATLAS_DRAM_BASE, len(initial)).numpy().tobytes()
        offset = len(initial) - len(expected) - 256
        assert (actual[:offset] == initial[:offset]
            and actual[offset + len(expected) :] == initial[offset + len(expected) :])
        assert actual[offset : offset + len(expected)] == expected
        self.remote.write(ATLAS_DRAM_BASE, actual)
        result = dict(index=index, sequence=index + 1, cycles=core.cycle_count - before, program_sha256=digest(words),
            host_input_sha256=digest(actual_host_input),
            actual_output_sha256=digest(actual[offset : offset + len(expected)]), expected_output_sha256=digest(expected),
            exact=True, drained=True, observed_dma=dict(events), reset_count=1)
        write_json(self.out / f"kernel-{index:02d}.json", result)
        self.completed.append(result)
        self.written.clear()

    def close(self):
        self.logger.close()
        self.core.close()

    def verify_bookends(self):
        recheck_layout()
        assert model_binding(self.model_root) == self.binding
        for path, expected in self.pins.items():
            assert digest(self.bundle.read(path)) == expected
