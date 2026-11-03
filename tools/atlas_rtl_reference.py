"""Execute scheduled physical Atlas instructions with explicit RTL numerics.

This reference executes instruction semantics atomically, including scalar
branch delay slots. The timed persistent-device QEMU replay remains the final
launch gate; this helper makes no cycle or physical timing claim.
"""

from collections import Counter
from dataclasses import replace
import io


class RtlReference:
    def __init__(self, arena_base):
        import torch
        from npu_model.configs.hardware.default import DefaultHardwareConfig
        from npu_model.hardware.arch_state import ArchState
        from npu_model.util.converter import input_to_program
        self.torch, self.assemble = torch, input_to_program
        config = replace(DefaultHardwareConfig().arch_state_config, numerics="rtl", randomize_init=True, init_seed=42)
        self.state = ArchState(config)
        self.arena_base = arena_base

    def run(self, source, initial, output_offset, output_bytes, requested_dma, *, fp8_scale=None):
        program = self.assemble(io.StringIO(source))
        state, torch = self.state, self.torch
        state.write_dram(self.arena_base, torch.frombuffer(bytearray(initial), dtype=torch.uint8).clone())
        state.halted, state.halt_reason, state.base = False, None, 0
        pc = count = 0
        pending = None
        events = Counter()
        saturation = nonfinite = 0
        while not state.halted:
            if not 0 <= pc < len(program) or count >= 50_000_000:
                raise RuntimeError(f"reference instruction bound at PC {pc}")
            instruction = program[pc]
            state.execute_pc, state.npc, state.redirect_requested = pc, pc + 1, False
            if instruction.mnemonic in ("dma.load", "dma.store"):
                events["load_bytes" if instruction.mnemonic == "dma.load" else "store_bytes"] += state.read_xrf(instruction.rs2)
            if fp8_scale is not None and instruction.mnemonic == "vmaximum.bf16" and int(instruction.vd) == 2:
                value = state.read_mrf_bf16_tile(2).float()
                saturation += int((value.abs() > 448 * 2.0**fp8_scale).sum())
                nonfinite += int((~torch.isfinite(value)).sum())
            instruction.exec(state)
            next_target = state.npc if state.redirect_requested else None
            pc = pending if pending is not None else pc + 1
            pending = next_target
            count += 1
        if state.halt_reason != "ecall":
            raise AssertionError("reference did not reach ECALL")
        for key, value in requested_dma.items():
            if events[key] != value:
                raise AssertionError(("reference DMA mismatch", key, events[key], value))
        actual = state.read_dram(self.arena_base, len(initial)).numpy().tobytes()
        end = output_offset + output_bytes
        if actual[:output_offset] != initial[:output_offset] or actual[end:] != initial[end:]:
            raise AssertionError("reference changed preserved arena bytes")
        return actual[output_offset:end], dict(physical_instructions=count, observed_dma=dict(events),
            numerics="rtl", timing_mode=False, preserved=True, saturation_bytes=saturation, nonfinite_values=nonfinite)

    def close(self):
        self.state.close()
