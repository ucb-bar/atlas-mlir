"""Derive model-owned storage geometry and address mappings without simulation."""
from __future__ import annotations

import ast
import inspect

from atlas_derive_scratchpad import source_path
from atlas_model_diagnostics import UnsupportedModelTiming


class UnsupportedArchitecture(UnsupportedModelTiming):
    """The selected model does not expose an understood architecture mapping."""


def _source(module):
    value = inspect.getsourcefile(module)
    if value is None:
        raise UnsupportedArchitecture("architecture source file is unavailable")
    path = source_path(value, UnsupportedArchitecture, "architecture source path contains a symlink", absolute=True)
    return ast.parse(path.read_text())


def _one(nodes, description):
    items = list(nodes)
    if len(items) != 1:
        raise UnsupportedArchitecture("expected one " + description)
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
        raise UnsupportedArchitecture(description + " must be a positive integer literal")
    return node.value


def _positive(owner, name):
    value = getattr(owner, name, None)
    if type(value) is not int or value <= 0:
        raise UnsupportedArchitecture("architecture configuration " + name + " must be a positive integer")
    return value


def _assignment(method, name):
    node = _one((n for n in ast.walk(method) if isinstance(n, ast.Assign)
                 and len(n.targets) == 1 and _same(n.targets[0], name)), method.name + " assignment to " + name)
    # A second or nested write would invalidate the extracted expression.
    stores = [n for n in ast.walk(method) if isinstance(n, ast.Name) and n.id == name
              and isinstance(n.ctx, ast.Store)]
    if len(stores) != 1:
        raise UnsupportedArchitecture("unsupported reassignment of " + name + " in " + method.name)
    return node.value


def _member_value(method, target):
    values = []
    for node in ast.walk(method):
        targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
        if any(ast.unparse(item) == target for item in targets):
            values.append(node.value)
    return _one(values, method.name + " assignment to " + target)


def _storage_contract(tree):
    method = _method(tree, "ArchState", "initialize_buffers")
    expressions = {
        "self.mrf": "[torch.zeros(self.cfg.mrf_depth * self.cfg.mrf_width, dtype=torch.uint8) for _ in range(self.cfg.num_m_registers)]",
        "self.vmem": "torch.zeros(self.cfg.vmem_size, dtype=torch.uint8)",
        "self.dram": "SparseMemory(self.cfg.dram_size)",
        "self.xrf": "[0] * self.cfg.num_x_registers",
        "self.erf": "[0] * self.cfg.num_e_registers",
        "self.csrf": "[0] * self.cfg.num_csrs",
    }
    for engine in ("mxu0", "mxu1"):
        expressions[f"self.wb['{engine}']"] = "[torch.zeros(self.cfg.wb_width, dtype=torch.uint8) for _ in range(self.cfg.num_wb_registers)]"
    for target, expression in expressions.items():
        if not _same(_member_value(method, target), expression):
            raise UnsupportedArchitecture("unsupported architectural storage allocation for " + target)


def _vmem_mapping(lsu_tree, dma_tree):
    ports = _method(lsu_tree, "LoadStoreUnit", "_check_vmem_ports")
    if not _same(_assignment(ports, "bank"), "address // self.config.vmem_bank_bytes"):
        raise UnsupportedArchitecture("unsupported LSU physical VMEM bank mapping")
    announce = _method(lsu_tree, "LoadStoreUnit", "_announce_vmem_ports")
    if not _same(_assignment(announce, "banks"), "{address // self.config.vmem_bank_bytes: mnemonic for mnemonic, address in self._vmem_accesses_at(self.cycle + 1)}"):
        raise UnsupportedArchitecture("unsupported LSU announced VMEM bank mapping")
    _body_contract(_method(dma_tree, "DmaExecutionUnit", "_bank"),
                   "return line_address // self.bank_bytes, (line_address % self.bank_bytes) // self.beat_bytes",
                   "DMA physical VMEM bank mapping")
    initialize = _method(dma_tree, "DmaExecutionUnit", "__init__")
    for member, field in (("bank_bytes", "vmem_bank_bytes"), ("beat_bytes", "dma_beat_bytes"),
                          ("num_channels", "dma_num_channels"), ("max_in_flight", "dma_max_in_flight"),
                          ("max_transfer_bytes", "dma_max_transfer_bytes")):
        if not _same(_member_value(initialize, "self." + member), "self.config." + field):
            raise UnsupportedArchitecture("unsupported DMA configuration mapping for " + member)


def _body_contract(method, body, description):
    actual = [n for n in method.body if not (isinstance(n, ast.Expr)
              and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str))]
    expected = ast.parse(body).body
    if [ast.dump(n) for n in actual] != [ast.dump(n) for n in expected]:
        raise UnsupportedArchitecture("unsupported " + description)


def _mreg_mapping(tree):
    method = _method(tree, "BankConflictChecker", "access_mreg")
    bank = _assignment(method, "bank")
    if not (isinstance(bank, ast.BinOp) and isinstance(bank.op, ast.Mod) and _same(bank.left, "reg")):
        raise UnsupportedArchitecture("unsupported MREG bank mapping: expected reg % bank_count")
    modulus = _int(bank.right, "MREG bank modulus")
    physical = _assignment(method, "physical_row")
    # Recognize (reg // bank_count) * rows_per_register + row; extract both
    # constants and check their relation instead of re-emitting 32.
    if not (isinstance(physical, ast.BinOp) and isinstance(physical.op, ast.Add)
            and _same(physical.right, "row") and isinstance(physical.left, ast.BinOp)
            and isinstance(physical.left.op, ast.Mult) and isinstance(physical.left.left, ast.BinOp)
            and isinstance(physical.left.left.op, ast.FloorDiv) and _same(physical.left.left.left, "reg")):
        raise UnsupportedArchitecture("unsupported MREG physical-row mapping")
    divisor = _int(physical.left.left.right, "MREG row-group divisor")
    stride = _int(physical.left.right, "MREG physical-row stride")
    if divisor != modulus:
        raise UnsupportedArchitecture("MREG row-group divisor differs from bank modulus")
    prefix = ast.parse(f"self.begin_cycle(cycle)\nbank = reg % {modulus}\n"
                       f"physical_row = (reg // {divisor}) * {stride} + row\nkey = (write, bank)").body
    if [ast.dump(n) for n in method.body[:4]] != [ast.dump(n) for n in prefix]:
        # The native method carries a docstring, which is not executable.
        body = [n for n in method.body if not (isinstance(n, ast.Expr)
                and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str))]
        if [ast.dump(n) for n in body[:4]] != [ast.dump(n) for n in prefix]:
            raise UnsupportedArchitecture("unsupported MREG bank mapping control flow")
    if any(isinstance(n, ast.Name) and n.id in {"reg", "row"} and isinstance(n.ctx, ast.Store)
           for n in ast.walk(method)):
        raise UnsupportedArchitecture("unsupported MREG input reassignment")
    reserve = _method(tree, "BankConflictChecker", "reserve_mreg")
    generator = _one((n for n in ast.walk(reserve) if isinstance(n, ast.GeneratorExp)), "MREG range guard")
    condition = generator.elt
    if not (isinstance(condition, ast.BoolOp) and isinstance(condition.op, ast.Or)
            and len(condition.values) == 2 and _same(condition.values[0], "reg < 0")
            and isinstance(condition.values[1], ast.Compare)):
        raise UnsupportedArchitecture("unsupported MREG logical-register range guard")
    upper = condition.values[1]
    if not (len(upper.comparators) == 1 and len(upper.ops) == 1
            and isinstance(upper.ops[0], ast.GtE) and _same(upper.left, "reg")):
        raise UnsupportedArchitecture("unsupported MREG logical-register upper bound")
    limit = _int(upper.comparators[0], "MREG logical-register upper bound")
    if not _same(generator, f"(reg < 0 or reg >= {limit} for reg in reads | writes)"):
        raise UnsupportedArchitecture("unsupported MREG logical-register guard iteration")
    return modulus, stride, limit


def _bf16_mapping(bank_tree, arch_tree):
    pair = _one((n for n in bank_tree.body if isinstance(n, ast.FunctionDef) and n.name == "_pair"), "BF16 register-pair helper")
    additions = [n for n in ast.walk(pair) if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Add)]
    if len(additions) != 2 or not all(_same(n.left, "reg") for n in additions):
        raise UnsupportedArchitecture("unsupported BF16 register-pair mapping")
    offsets = [_int(n.right, "BF16 pair stride") for n in additions]
    if offsets[0] != offsets[1]:
        raise UnsupportedArchitecture("BF16 register-pair helper has inconsistent strides")
    stride = offsets[0]
    _body_contract(pair, f"if isinstance(reg, MatrixReg):\n    return frozenset({{reg, MatrixReg(reg + {stride})}})\n"
                         f"return frozenset({{reg, reg + {stride}}})", "BF16 register-pair helper")
    read = _method(arch_tree, "ArchState", "read_mrf_bf16_tile")
    _body_contract(read, f"return torch.cat((self.read_mrf_bf16(vs), self.read_mrf_bf16(vs + {stride})), dim=1)",
                   "BF16 tile read mapping")
    # The number of registers comes from actual read calls, checked against the
    # write shape and split. This is independent of BF16's bytes per element.
    count = len([n for n in ast.walk(read) if isinstance(n, ast.Call)
                 and _same(n.func, "self.read_mrf_bf16")])
    write = _method(arch_tree, "ArchState", "write_mrf_bf16_tile")
    _body_contract(write, "cols_per_register = self.cfg.mrf_width // torch.bfloat16.itemsize\n"
                   "assert value.dtype == torch.bfloat16\n"
                   f"assert value.shape == (self.cfg.mrf_depth, cols_per_register * {count})\n"
                   "self.write_mrf_bf16(vd, value[:, :cols_per_register].contiguous())\n"
                   f"self.write_mrf_bf16(vd + {stride}, value[:, cols_per_register:].contiguous())",
                   "BF16 tile write mapping")
    return count, stride


def _fp8_mapping(tree):
    read = _method(tree, "ArchState", "read_mrf_fp8")
    _body_contract(read, "return self.mrf[vs].view(torch.float8_e4m3fn).reshape("
                   "self.cfg.mrf_depth, self.cfg.mrf_width // torch.float8_e4m3fn.itemsize)",
                   "FP8 register read mapping")
    write = _method(tree, "ArchState", "write_mrf_fp8")
    _body_contract(write, "assert value.dtype == torch.float8_e4m3fn\n"
                   "assert value.numel() == self.cfg.mrf_depth * self.cfg.mrf_width // torch.float8_e4m3fn.itemsize\n"
                   "self.mrf[vd].view(torch.float8_e4m3fn)[:] = value.flatten()",
                   "FP8 register write mapping")
    return len([node for node in ast.walk(read) if isinstance(node, ast.Subscript)
                and _same(node.value, "self.mrf")])


def _scalar_bytes(tree):
    method = _method(tree, "ArchState", "write_xrf")
    masks = [n for n in method.body if isinstance(n, ast.AugAssign) and _same(n.target, "value")]
    mask = _one(masks, "scalar-register width mask")
    if not isinstance(mask.op, ast.BitAnd):
        raise UnsupportedArchitecture("unsupported scalar-register width mask")
    value = _int(mask.value, "scalar-register width mask")
    bits = value.bit_length()
    if value != (1 << bits) - 1 or bits % 8:
        raise UnsupportedArchitecture("scalar-register mask must cover a whole number of bytes")
    _body_contract(method, f"value &= {value}\n"
                   "if rd == 0:\n    return\n"
                   "if rd < len(self.xrf) and self.xrf[rd] == value:\n    return\n"
                   "self.xrf[rd] = value\n"
                   "if self.logger:\n    self.logger.log_arch_value('xrf', rd, value)",
                   "scalar storage semantics in ArchState.write_xrf")
    return bits // 8


def derive_arch(cfg):
    """Return source-derived architecture facts for the actual selected config."""
    from npu_model.hardware import arch_state, bank_conflict, dma, lsu
    import torch

    bank_tree, arch_tree = _source(bank_conflict), _source(arch_state)
    _storage_contract(arch_tree)
    _vmem_mapping(_source(lsu), _source(dma))
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
            raise UnsupportedArchitecture("architecture configuration " + name + " must be an integer")
        facts[name] = value
    base = getattr(arch, "dram_base", None)
    if type(base) is not int or base < 0:
        raise UnsupportedArchitecture("architecture configuration dram_base must be a nonnegative integer")
    if facts["mreg_count"] != logical_limit:
        raise UnsupportedArchitecture("configured MREG count differs from bank checker's logical-register upper bound")
    if facts["mreg_rows"] != row_stride:
        raise UnsupportedArchitecture("configured MREG rows differ from physical-row stride")
    if facts["mreg_count"] % banks:
        raise UnsupportedArchitecture("MREG count is not divisible by physical-bank modulus")
    if facts["vmem_bytes"] % facts["vmem_bank_bytes"]:
        raise UnsupportedArchitecture("VMEM capacity is not divisible by physical-bank bytes")
    element_bytes = torch.bfloat16.itemsize
    if type(element_bytes) is not int or element_bytes <= 0 or facts["mreg_row_bytes"] % element_bytes:
        raise UnsupportedArchitecture("MREG row bytes do not contain whole BF16 elements")
    if pair_stride >= facts["mreg_count"]:
        raise UnsupportedArchitecture("BF16 pair stride exceeds the configured MREG range")
    fp8_element_bytes = torch.float8_e4m3fn.itemsize
    if type(fp8_element_bytes) is not int or fp8_element_bytes <= 0 or facts["mreg_row_bytes"] % fp8_element_bytes:
        raise UnsupportedArchitecture("MREG row bytes do not contain whole FP8 elements")
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
