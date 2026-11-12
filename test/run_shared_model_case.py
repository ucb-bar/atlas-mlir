"""Execute independently decoded final bytes and compare explicit raw witnesses.

Run with the selected npu_model Python environment. The compiler never receives
the expected outputs in the case file. This harness is deliberately test-only.
"""

from dataclasses import replace
import json
from pathlib import Path
import sys

import torch

from npu_model.configs import isa_definition as isa
from npu_model.configs.hardware.default import DefaultHardwareConfig
from npu_model.isa_metadata import _fields, _format_type
from npu_model.logging import LoggerConfig
from npu_model.simulation import Simulation
from npu_model.software.program import InstantiableProgram

from tools.check_program_decode import audit_program, decode_binary
from tools.check_variant_inventory import load_inventory


def instructions(binary):
    inventory = load_inventory()
    modes = {row["id"]: row for row in inventory["variants"]}
    decoded = decode_binary(inventory, binary)
    result = []
    def signed(value, bits):
        return value - (1 << bits) if value & (1 << (bits-1)) else value
    for row in decoded["instructions"]:
        word = row["word_u32"]
        name = modes[row["mode_id"]]["model_classes"][0].partition("_CH")[0]
        cls = getattr(isa, name)
        fmt = _format_type(cls).__name__
        values = dict(rd=(word >> 7) & 31, rs1=(word >> 15) & 31, rs2=(word >> 20) & 31,
                      vd=(word >> 7) & 63, vs1=(word >> 13) & 63, vs2=(word >> 19) & 63,
                      es1=(word >> 13) & 63, ch=(word >> 12) & 7, imm=signed(word >> 20, 12))
        if fmt == "UType":
            values["imm"] = word >> 12
        elif fmt == "VIType":
            values["imm"] = word >> 16
        elif fmt == "SType":
            values["imm"] = signed(((word >> 25) << 5) | ((word >> 7) & 31), 12)
        elif fmt == "SBType":
            values["imm"] = signed((((word >> 31)&1)<<12) | (((word >> 7)&1)<<11) |
                                   (((word >> 25)&63)<<5) | (((word >> 8)&15)<<1), 13)
        instruction = cls(**{f.repr: f.inner(values[f.repr]) for f in _fields(cls)})
        if instruction.to_bytecode() != word:
            raise ValueError(f"decoded instruction differs at word {row['word_index']}")
        result.append(instruction)
    return result


def execute(directory, case_path):
    case = json.loads(case_path.read_text())
    binary = (directory / "program.bin").read_bytes()
    audit = audit_program(load_inventory(), json.loads((directory / "program.json").read_text()), binary)
    assert audit["status"] == "PASS", audit
    initial = [(offset, bytes.fromhex(data)) for offset, data in case["initial"]]
    expected = [(offset, bytes.fromhex(data)) for offset, data in case["expected"]]
    preserved = [(offset, bytes.fromhex(data)) for offset, data in case["preserved"]]
    program = InstantiableProgram(instructions(binary))
    program.dram_base = case["dram_base"]
    program.memory_regions = [(offset, torch.tensor(list(data), dtype=torch.uint8)) for offset, data in initial]
    config = DefaultHardwareConfig()
    config.arch_state_config = replace(config.arch_state_config, numerics="rtl")
    torch.set_num_threads(1)
    sim = Simulation(config, LoggerConfig(filename=str(case_path.with_suffix(".trace.json"))), program, verbose=False)
    try:
        sim.run(max_cycles=case.get("max_cycles", 100000))
        assert sim.core.is_finished(), "program did not halt within cycle budget"
        stats = sim.get_stats()
        assert not stats.runtime_errors, stats.runtime_errors
        for role, regions in (("output", expected), ("preserved", preserved)):
            for offset, data in regions:
                observed = bytes(sim.core.arch_state.read_dram(program.dram_base + offset, len(data)).tolist())
                assert observed == data, (role, hex(offset), sum(a != b for a, b in zip(observed, data)))
        result = {"correct": True, "words": len(binary)//4, "cycles": stats.cycles,
                  "output_bytes": sum(len(data) for _, data in expected), "guards_preserved": True}
        case_path.with_suffix(".result.json").write_text(json.dumps(result, indent=2) + "\n")
        return result
    finally:
        sim.close()


if __name__ == "__main__":
    print(json.dumps(execute(Path(sys.argv[1]), Path(sys.argv[2]))))
