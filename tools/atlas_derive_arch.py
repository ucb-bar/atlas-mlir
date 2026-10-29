"""Derive model-owned storage geometry and address mappings without simulation."""
from __future__ import annotations

import ast
import inspect

from pathlib import Path
from atlas_model_diagnostics import UnsupportedModelTiming


def _source(module):
    value = inspect.getsourcefile(module)
    if value is None:
        raise UnsupportedModelTiming("architecture source file is unavailable")
    path = Path(value)
    return ast.parse(path.read_text())


def _one(nodes, description):
    items = list(nodes)
    if len(items) != 1:
        raise UnsupportedModelTiming("expected one " + description)
    return items[0]


def _method(tree, cls, name):
    owner = _one((n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls), cls)
    return _one((n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == name), cls + "." + name)


def _same(node, expression):
    expected = ast.parse(expression, mode="eval").body
    if isinstance(node, ast.Name) and isinstance(expected, ast.Name):
        return node.id == expected.id
    return ast.dump(node) == ast.dump(expected)


def _int(node, description):
    if not isinstance(node, ast.Constant) or type(node.value) is not int or node.value <= 0:
        raise UnsupportedModelTiming(description + " must be a positive integer literal")
    return node.value


def _positive(owner, name):
    value = getattr(owner, name, None)
    if type(value) is not int or value <= 0:
        raise UnsupportedModelTiming("architecture configuration " + name + " must be a positive integer")
    return value


def _assignment(method, name):
    node = _one((n for n in ast.walk(method) if isinstance(n, ast.Assign)
                 and len(n.targets) == 1 and _same(n.targets[0], name)), method.name + " assignment to " + name)
    # A second or nested write would invalidate the extracted expression.
    stores = [n for n in ast.walk(method) if isinstance(n, ast.Name) and n.id == name
              and isinstance(n.ctx, ast.Store)]
    if len(stores) != 1:
        raise UnsupportedModelTiming("unsupported reassignment of " + name + " in " + method.name)
    return node.value


def _mreg_mapping(tree):
    # The bank_conflict.py digest pins the mapping around these source literals.
    method = _method(tree, "BankConflictChecker", "access_mreg")
    modulus = _int(_assignment(method, "bank").right, "MREG bank modulus")
    physical = _assignment(method, "physical_row")
    stride = _int(physical.left.right, "MREG physical-row stride")
    reserve = _method(tree, "BankConflictChecker", "reserve_mreg")
    generator = _one((n for n in ast.walk(reserve) if isinstance(n, ast.GeneratorExp)), "MREG range guard")
    limit = _int(generator.elt.values[1].comparators[0], "MREG logical-register upper bound")
    return modulus, stride, limit


def _bf16_mapping(bank_tree, arch_tree):
    # The ArchState and bank checker guards pin the read/write pair mapping.
    pair = _one((n for n in bank_tree.body if isinstance(n, ast.FunctionDef) and n.name == "_pair"),
                "BF16 register-pair helper")
    addition = next(n for n in ast.walk(pair) if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Add))
    read = _method(arch_tree, "ArchState", "read_mrf_bf16_tile")
    count = sum(isinstance(n, ast.Call) and _same(n.func, "self.read_mrf_bf16") for n in ast.walk(read))
    return count, _int(addition.right, "BF16 pair stride")


def _fp8_mapping(tree):
    read = _method(tree, "ArchState", "read_mrf_fp8")
    return sum(isinstance(node, ast.Subscript) and _same(node.value, "self.mrf") for node in ast.walk(read))


def _scalar_bytes(tree):
    method = _method(tree, "ArchState", "write_xrf")
    masks = [n for n in method.body if isinstance(n, ast.AugAssign) and _same(n.target, "value")]
    mask = _one(masks, "scalar-register width mask")
    if not isinstance(mask.op, ast.BitAnd):
        raise UnsupportedModelTiming("unsupported scalar-register width mask")
    value = _int(mask.value, "scalar-register width mask")
    bits = value.bit_length()
    if value != (1 << bits) - 1 or bits % 8:
        raise UnsupportedModelTiming("scalar-register mask must cover a whole number of bytes")
    return bits // 8


def derive_arch(cfg):
    """Return source-derived architecture facts for the actual selected config."""
    from npu_model.hardware import arch_state, bank_conflict
    import torch

    bank_tree, arch_tree = _source(bank_conflict), _source(arch_state)
    # The central ArchState contract pins allocation, scalar width, and tile access shapes.
    banks, row_stride, logical_limit = _mreg_mapping(bank_tree)
    pair_count, pair_stride = _bf16_mapping(bank_tree, arch_tree)
    fp8_count = _fp8_mapping(arch_tree)
    arch = cfg.arch_state_config
    fields = {
        "mreg_count": (arch, "num_m_registers"), "mreg_rows": (arch, "mrf_depth"),
        "mreg_row_bytes": (arch, "mrf_width"), "vmem_bytes": (arch, "vmem_size"),
        "vmem_bank_bytes": (cfg, "vmem_bank_bytes"), "weight_buffer_count": (arch, "num_wb_registers"),
        "weight_buffer_bytes": (arch, "wb_width"), "xreg_count": (arch, "num_x_registers"),
        "ereg_count": (arch, "num_e_registers"), "csr_count": (arch, "num_csrs"),
        "dram_bytes": (arch, "dram_size"), "dma_channels": (cfg, "dma_num_channels"),
        "dma_beat_bytes": (cfg, "dma_beat_bytes"), "dma_max_in_flight": (cfg, "dma_max_in_flight"),
        "dma_max_transfer_bytes": (cfg, "dma_max_transfer_bytes"), "fetch_width": (cfg, "fetch_width"),
    }
    facts = {name: _positive(owner, attribute) for name, (owner, attribute) in fields.items()}
    # Dormant legacy estimates are recorded, not fixed-layout capabilities.
    for name in ("offchip_link_width_bits", "offchip_link_core_cycles_per_beat", "dma_offchip_command_words",
                 "vmem_bus_width_bits", "vmem_bus_core_cycles_per_beat", "vmem_bytes_per_cycle"):
        value = getattr(cfg, name, None)
        if type(value) is not int:
            raise UnsupportedModelTiming("architecture configuration " + name + " must be an integer")
        facts[name] = value
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
    facts.update(schema="atlas.architecture.v1", dram_base=base,
                 mreg_bytes=register_bytes, mreg_physical_bank_count=banks,
                 mreg_bank_modulus=banks, mreg_registers_per_bank=facts["mreg_count"] // banks,
                 mreg_physical_row_stride=row_stride, mreg_logical_limit=logical_limit,
                 bf16_registers_per_tile=pair_count, bf16_pair_stride=pair_stride,
                 bf16_element_bytes=element_bytes, bf16_tile_bytes=pair_count * register_bytes,
                 fp8_element_bytes=fp8_element_bytes, fp8_registers_per_tile=fp8_count,
                 fp8_tile_bytes=fp8_count * register_bytes,
                 vmem_bank_count=facts["vmem_bytes"] // facts["vmem_bank_bytes"],
                 scalar_word_bytes=_scalar_bytes(arch_tree))
    return facts
