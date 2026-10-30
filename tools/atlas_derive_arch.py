"""Derive model-owned storage geometry and address mappings without simulation."""
from __future__ import annotations

import ast

from atlas_model_source import (UnsupportedModelTiming, assignment, method as _method, module_tree,
                                named, one as _one, positive_int)


def _same(node, expression):
    expected = ast.parse(expression, mode="eval").body
    if isinstance(node, ast.Name) and isinstance(expected, ast.Name):
        return node.id == expected.id
    return ast.dump(node) == ast.dump(expected)


def _mreg_mapping(tree):
    # The bank_conflict.py digest pins the mapping around these source literals.
    method = _method(named(tree, "BankConflictChecker"), "access_mreg")
    bank = assignment(method, "bank", single_write=True).value
    modulus = positive_int(bank.right, "MREG bank modulus")
    physical = assignment(method, "physical_row", single_write=True).value
    stride = positive_int(physical.left.right, "MREG physical-row stride")
    reserve = _method(named(tree, "BankConflictChecker"), "reserve_mreg")
    generator = _one((n for n in ast.walk(reserve) if isinstance(n, ast.GeneratorExp)), "MREG range guard")
    limit = positive_int(generator.elt.values[1].comparators[0], "MREG logical-register upper bound")
    return modulus, stride, limit


def _bf16_mapping(bank_tree, arch_tree):
    # The ArchState and bank checker guards pin the read/write pair mapping.
    pair = _one((n for n in bank_tree.body if isinstance(n, ast.FunctionDef) and n.name == "_pair"),
                "BF16 register-pair helper")
    addition = next(n for n in ast.walk(pair) if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Add))
    read = _method(named(arch_tree, "ArchState"), "read_mrf_bf16_tile")
    count = sum(isinstance(n, ast.Call) and _same(n.func, "self.read_mrf_bf16") for n in ast.walk(read))
    return count, positive_int(addition.right, "BF16 pair stride")


def _fp8_mapping(tree):
    read = _method(named(tree, "ArchState"), "read_mrf_fp8")
    return sum(isinstance(node, ast.Subscript) and _same(node.value, "self.mrf") for node in ast.walk(read))


def _scalar_bytes(tree):
    method = _method(named(tree, "ArchState"), "write_xrf")
    masks = [n for n in method.body if isinstance(n, ast.AugAssign) and _same(n.target, "value")]
    mask = _one(masks, "scalar-register width mask")
    if not isinstance(mask.op, ast.BitAnd):
        raise UnsupportedModelTiming("unsupported scalar-register width mask")
    value = positive_int(mask.value, "scalar-register width mask")
    bits = value.bit_length()
    if value != (1 << bits) - 1 or bits % 8:
        raise UnsupportedModelTiming("scalar-register mask must cover a whole number of bytes")
    return bits // 8


def instruction_memory_words(tree):
    imem = named(tree, "InstructionMemory")
    return positive_int(assignment(imem, "WORDS", single_write=True).value,
                        "InstructionMemory.WORDS", maximum=2147483648)


def derive_arch(cfg):
    """Return source-derived architecture facts for the actual selected config."""
    from npu_model.hardware import arch_state, bank_conflict, ifu
    import torch

    bank_tree, arch_tree = module_tree(bank_conflict), module_tree(arch_state)
    # The central ArchState contract pins allocation, scalar width, and tile access shapes.
    banks, row_stride, logical_limit = _mreg_mapping(bank_tree)
    pair_count, pair_stride = _bf16_mapping(bank_tree, arch_tree)
    fp8_count = _fp8_mapping(arch_tree)
    instruction_words = instruction_memory_words(module_tree(ifu))
    arch = cfg.arch_state_config
    fields = {
        "mreg_count": (arch, "num_m_registers"), "mreg_rows": (arch, "mrf_depth"),
        "mreg_row_bytes": (arch, "mrf_width"), "vmem_bytes": (arch, "vmem_size"),
        "vmem_bank_bytes": (cfg, "vmem_bank_bytes"),
        "dram_bytes": (arch, "dram_size"), "dma_channels": (cfg, "dma_num_channels"),
    }
    facts = {name: positive_int(getattr(owner, attribute, None), "architecture configuration " + attribute)
             for name, (owner, attribute) in fields.items()}
    base = getattr(arch, "dram_base", None)
    if type(base) is not int or base < 0:
        raise UnsupportedModelTiming("architecture configuration dram_base must be a nonnegative integer")
    if facts["mreg_count"] != logical_limit:
        raise UnsupportedModelTiming("configured MREG count differs from bank checker's logical-register upper bound")
    if facts["mreg_rows"] != row_stride:
        raise UnsupportedModelTiming("configured MREG rows differ from physical-row stride")
    if facts["mreg_count"] % banks:
        raise UnsupportedModelTiming("MREG count is not divisible by physical-bank modulus")
    if facts["vmem_bytes"] % facts["vmem_bank_bytes"]:
        raise UnsupportedModelTiming("VMEM capacity is not divisible by physical-bank bytes")
    element_bytes = torch.bfloat16.itemsize
    if type(element_bytes) is not int or element_bytes <= 0 or facts["mreg_row_bytes"] % element_bytes:
        raise UnsupportedModelTiming("MREG row bytes do not contain whole BF16 elements")
    if pair_stride >= facts["mreg_count"]:
        raise UnsupportedModelTiming("BF16 pair stride exceeds the configured MREG range")
    fp8_element_bytes = torch.float8_e4m3fn.itemsize
    if type(fp8_element_bytes) is not int or fp8_element_bytes <= 0 or facts["mreg_row_bytes"] % fp8_element_bytes:
        raise UnsupportedModelTiming("MREG row bytes do not contain whole FP8 elements")
    register_bytes = facts["mreg_rows"] * facts["mreg_row_bytes"]
    facts.update(dram_base=base, mreg_bytes=register_bytes, mreg_physical_bank_count=banks,
                 instruction_memory_words=instruction_words,
                 bf16_registers_per_tile=pair_count, bf16_element_bytes=element_bytes,
                 fp8_element_bytes=fp8_element_bytes, fp8_registers_per_tile=fp8_count,
                 vmem_bank_count=facts["vmem_bytes"] // facts["vmem_bank_bytes"],
                 scalar_word_bytes=_scalar_bytes(arch_tree))
    return facts
