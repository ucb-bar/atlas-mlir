"""Checked conversion from independently decoded words to model instructions."""

import inspect


def model_instruction(row, inventory, isa):
    """Bind a decoded instruction and require an identical re-encoded word.

    npu_model executes instruction objects. Decode uses the existing RTL
    inventory, independently of the emitter and its JSON operand annotations.
    """
    from npu_model.isa import SBType, SType, UJType, UType, VIType

    mode = row["mode_id"]
    variant = next(v for v in inventory["variants"] if v["id"] == mode)
    name = mode.removesuffix("_ANY") if mode.startswith("DMA_") else variant["model_classes"][0]
    cls, word = getattr(isa, name), row["word_u32"]
    operands = {
        "rd": (word >> 7) & 31,
        "rs1": (word >> 15) & 31,
        "rs2": (word >> 20) & 31,
        "vd": (word >> 7) & 63,
        "vs1": (word >> 13) & 63,
        "es1": (word >> 13) & 63,
        "vs2": (word >> 19) & 63,
        "ch": (word >> 12) & 7,
    }
    if issubclass(cls, SBType):
        immediate = row["fields"]["offset_bytes"]
    elif issubclass(cls, UJType):
        immediate = row["fields"]["offset"]
    elif issubclass(cls, SType):
        immediate = ((word >> 25) << 5) | ((word >> 7) & 31)
    elif issubclass(cls, VIType):
        immediate = word >> 16
    elif issubclass(cls, UType):
        immediate = word >> 12
    else:
        immediate = word >> 20
    operands["imm"] = immediate
    instruction = cls(**{name: operands[name] for name in inspect.signature(cls).parameters})
    if instruction.to_bytecode() != word:
        raise ValueError(f"npu_model instruction differs from word {row['word_index']}")
    return instruction
