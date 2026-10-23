"""Architecture capacities and model-owned mapping mutations propagate or refuse."""
import ast
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import atlas_derive_arch as derive


BANK_SOURCE = """
def _pair(reg):
    if isinstance(reg, MatrixReg):
        return frozenset({reg, MatrixReg(reg + 1)})
    return frozenset({reg, reg + 1})

class BankConflictChecker:
    def reserve_mreg(self, owner, reads, writes):
        if any(reg < 0 or reg >= 64 for reg in reads | writes):
            raise BankConflictError('outside range')

    def access_mreg(self, cycle, reg, row, write, owner):
        self.begin_cycle(cycle)
        bank = reg % 32
        physical_row = (reg // 32) * 32 + row
        key = (write, bank)
        self._mreg_ports[key] = (reg, physical_row, owner)
"""

ARCH_SOURCE = """
class ArchState:
    def initialize_buffers(self):
        self.vmem = torch.zeros(self.cfg.vmem_size, dtype=torch.uint8)
        self.dram = SparseMemory(self.cfg.dram_size)
        self.mrf = [torch.zeros(self.cfg.mrf_depth * self.cfg.mrf_width, dtype=torch.uint8) for _ in range(self.cfg.num_m_registers)]
        self.xrf = [0] * self.cfg.num_x_registers
        self.erf = [0] * self.cfg.num_e_registers
        self.csrf = [0] * self.cfg.num_csrs
        self.wb['mxu0'] = [torch.zeros(self.cfg.wb_width, dtype=torch.uint8) for _ in range(self.cfg.num_wb_registers)]
        self.wb['mxu1'] = [torch.zeros(self.cfg.wb_width, dtype=torch.uint8) for _ in range(self.cfg.num_wb_registers)]

    def write_mrf_bf16_tile(self, vd, value):
        cols_per_register = self.cfg.mrf_width // torch.bfloat16.itemsize
        assert value.dtype == torch.bfloat16
        assert value.shape == (self.cfg.mrf_depth, cols_per_register * 2)
        self.write_mrf_bf16(vd, value[:, :cols_per_register].contiguous())
        self.write_mrf_bf16(vd + 1, value[:, cols_per_register:].contiguous())

    def read_mrf_bf16_tile(self, vs):
        return torch.cat((self.read_mrf_bf16(vs), self.read_mrf_bf16(vs + 1)), dim=1)

    def write_xrf(self, rd, value):
        value &= 0xFFFFFFFF
        self.xrf[rd] = value
"""


LSU_SOURCE = """
class LoadStoreUnit:
    def _check_vmem_ports(self):
        for mnemonic, address in self._vmem_accesses_at(self.cycle):
            bank = address // self.config.vmem_bank_bytes

    def _announce_vmem_ports(self):
        banks = {address // self.config.vmem_bank_bytes: mnemonic for mnemonic, address in self._vmem_accesses_at(self.cycle + 1)}
"""

DMA_SOURCE = """
class DmaExecutionUnit:
    def __init__(self):
        self.bank_bytes = self.config.vmem_bank_bytes
        self.beat_bytes = self.config.dma_beat_bytes
        self.num_channels = self.config.dma_num_channels
        self.max_in_flight = self.config.dma_max_in_flight
        self.max_transfer_bytes = self.config.dma_max_transfer_bytes

    def _bank(self, line_address):
        return line_address // self.bank_bytes, (line_address % self.bank_bytes) // self.beat_bytes
"""


def configuration():
    return SimpleNamespace(
        arch_state_config=SimpleNamespace(mrf_depth=32, mrf_width=32, num_m_registers=64,
            vmem_size=1572864, num_wb_registers=2, wb_width=1024, num_x_registers=32,
            num_e_registers=32, num_csrs=4096, dram_size=0x1000000000, dram_base=0x80000000),
        vmem_bank_bytes=262144, dma_num_channels=8, dma_beat_bytes=32, dma_max_in_flight=64,
        dma_max_transfer_bytes=4096, fetch_width=1, offchip_link_width_bits=32,
        offchip_link_core_cycles_per_beat=2, dma_offchip_command_words=2,
        vmem_bus_width_bits=512, vmem_bus_core_cycles_per_beat=1, vmem_bytes_per_cycle=64)


class ArchitectureDerivationTest(unittest.TestCase):
    def derive(self, cfg=None, bank=BANK_SOURCE, arch=ARCH_SOURCE, lsu=LSU_SOURCE, dma=DMA_SOURCE):
        bank_module, arch_module, lsu_module, dma_module = object(), object(), object(), object()
        modules = {"npu_model": SimpleNamespace(), "npu_model.hardware":
                   SimpleNamespace(bank_conflict=bank_module, arch_state=arch_module, lsu=lsu_module, dma=dma_module),
                   "torch": SimpleNamespace(bfloat16=SimpleNamespace(itemsize=2))}
        def source(module):
            return ast.parse({bank_module: bank, arch_module: arch, lsu_module: lsu, dma_module: dma}[module])
        with patch.dict(sys.modules, modules), patch.object(derive, "_source", side_effect=source):
            return derive.derive_arch(cfg or configuration())

    def test_baseline_architecture_has_source_mapping_and_no_rtl_ring(self):
        facts = self.derive()
        self.assertEqual(facts["schema"], "atlas.architecture.v1")
        self.assertEqual((facts["mreg_count"], facts["mreg_rows"], facts["mreg_row_bytes"],
                          facts["mreg_bytes"]), (64, 32, 32, 1024))
        self.assertEqual((facts["bf16_registers_per_tile"], facts["bf16_pair_stride"],
                          facts["bf16_tile_bytes"], facts["bf16_element_bytes"]), (2, 1, 2048, 2))
        self.assertEqual((facts["mreg_bank_modulus"], facts["mreg_registers_per_bank"],
                          facts["mreg_physical_row_stride"]), (32, 2, 32))
        self.assertEqual((facts["vmem_bytes"], facts["vmem_bank_bytes"], facts["vmem_bank_count"]),
                         (1572864, 262144, 6))
        self.assertEqual((facts["dma_channels"], facts["scalar_word_bytes"]), (8, 4))
        self.assertFalse(any("ring" in key or "command_slots" in key for key in facts))

    def test_config_capacity_mutations_propagate(self):
        cfg = configuration()
        cfg.arch_state_config.vmem_size *= 2
        cfg.vmem_bank_bytes //= 2
        cfg.arch_state_config.mrf_width *= 2
        cfg.arch_state_config.num_wb_registers = 4
        cfg.arch_state_config.wb_width *= 2
        cfg.arch_state_config.num_x_registers = 64
        cfg.dma_num_channels = 16
        cfg.dma_max_transfer_bytes *= 2
        changed = self.derive(cfg)
        self.assertEqual(changed["vmem_bank_count"], 24)
        self.assertEqual(changed["mreg_bytes"], 2048)
        self.assertEqual(changed["bf16_tile_bytes"], 4096)
        self.assertEqual(changed["weight_buffer_count"], 4)
        self.assertEqual(changed["weight_buffer_bytes"], 2048)
        self.assertEqual(changed["xreg_count"], 64)
        self.assertEqual(changed["dma_channels"], 16)
        self.assertEqual(changed["dma_max_transfer_bytes"], 8192)

    def test_source_bank_modulus_changes_propagate(self):
        bank = BANK_SOURCE.replace("reg % 32", "reg % 16").replace("reg // 32", "reg // 16")
        changed = self.derive(bank=bank)
        self.assertEqual(changed["mreg_physical_bank_count"], 16)
        self.assertEqual(changed["mreg_bank_modulus"], 16)
        self.assertEqual(changed["mreg_registers_per_bank"], 4)
        self.assertEqual(changed["mreg_physical_row_stride"], 32)

    def test_source_logical_bound_and_geometry_changes_propagate_together(self):
        cfg = configuration()
        cfg.arch_state_config.num_m_registers = 128
        cfg.arch_state_config.mrf_depth = 16
        bank = BANK_SOURCE.replace("reg >= 64", "reg >= 128").replace("* 32 + row", "* 16 + row")
        changed = self.derive(cfg, bank=bank)
        self.assertEqual((changed["mreg_count"], changed["mreg_logical_limit"]), (128, 128))
        self.assertEqual((changed["mreg_rows"], changed["mreg_physical_row_stride"]), (16, 16))
        self.assertEqual(changed["mreg_bytes"], 512)

    def test_bf16_pair_mapping_changes_require_read_write_and_reservations_to_agree(self):
        bank = BANK_SOURCE.replace("reg + 1", "reg + 2")
        arch = ARCH_SOURCE.replace("vd + 1", "vd + 2").replace("vs + 1", "vs + 2")
        self.assertEqual(self.derive(bank=bank, arch=arch)["bf16_pair_stride"], 2)
        with self.assertRaisesRegex(derive.UnsupportedArchitecture, "BF16 tile read mapping"):
            self.derive(bank=bank)
        with self.assertRaisesRegex(derive.UnsupportedArchitecture, "BF16 tile write mapping"):
            self.derive(arch=ARCH_SOURCE.replace("vd + 1", "vd + 2"))

    def test_scalar_source_mask_controls_word_bytes(self):
        self.assertEqual(self.derive(arch=ARCH_SOURCE.replace("0xFFFFFFFF", "0xFFFF"))["scalar_word_bytes"], 2)
        with self.assertRaisesRegex(derive.UnsupportedArchitecture, "whole number of bytes"):
            self.derive(arch=ARCH_SOURCE.replace("0xFFFFFFFF", "0xFFFE"))

    def test_storage_and_vmem_mapping_mutations_fail_precisely(self):
        with self.assertRaisesRegex(derive.UnsupportedArchitecture, "storage allocation for self.mrf"):
            self.derive(arch=ARCH_SOURCE.replace("self.cfg.mrf_depth * self.cfg.mrf_width,", "self.cfg.mrf_depth * self.cfg.mrf_width * 2,"))
        with self.assertRaisesRegex(derive.UnsupportedArchitecture, "LSU physical VMEM bank mapping"):
            self.derive(lsu=LSU_SOURCE.replace("address // self.config.vmem_bank_bytes", "address % self.config.vmem_bank_bytes"))
        with self.assertRaisesRegex(derive.UnsupportedArchitecture, "DMA physical VMEM bank mapping"):
            self.derive(dma=DMA_SOURCE.replace("line_address // self.bank_bytes", "line_address % self.bank_bytes"))
        with self.assertRaisesRegex(derive.UnsupportedArchitecture, "DMA configuration mapping for num_channels"):
            self.derive(dma=DMA_SOURCE.replace("self.config.dma_num_channels", "self.config.dma_num_channels * 2"))

    def test_unknown_mapping_changes_fail_precisely(self):
        cases = [
            (BANK_SOURCE.replace("reg % 32", "reg & 31"), "MREG bank mapping"),
            (BANK_SOURCE.replace("reg // 32", "reg // 16"), "divisor differs"),
            (BANK_SOURCE.replace("+ row", "+ row + 1"), "physical-row mapping"),
            (BANK_SOURCE.replace("key = (write, bank)", "bank += 1\n        key = (write, bank)"), "reassignment"),
        ]
        for bank, diagnostic in cases:
            with self.subTest(diagnostic=diagnostic), self.assertRaisesRegex(derive.UnsupportedArchitecture, diagnostic):
                self.derive(bank=bank)

    def test_inconsistent_geometry_fails_precisely(self):
        cases = [("mrf_depth", 16, "physical-row stride"),
                 ("num_m_registers", 32, "logical-register upper bound"),
                 ("vmem_size", 1572865, "not divisible"),
                 ("mrf_width", 31, "whole BF16 elements"),
                 ("num_x_registers", 0, "num_x_registers must be a positive integer")]
        for name, value, diagnostic in cases:
            cfg = configuration()
            setattr(cfg.arch_state_config, name, value)
            with self.subTest(name=name), self.assertRaisesRegex(derive.UnsupportedArchitecture, diagnostic):
                self.derive(cfg)

    def test_source_path_guards_run_before_reading(self):
        with patch.object(derive.inspect, "getsourcefile", return_value="/tmp/prohibited_VLSI/source.py"), \
                patch.object(Path, "read_text", side_effect=AssertionError("read source")):
            with self.assertRaisesRegex(derive.UnsupportedArchitecture, "prohibited architecture source path"):
                derive._source(object())


@unittest.skipUnless(os.environ.get("ATLAS_DERIVE_MODEL_ROOT"), "set ATLAS_DERIVE_MODEL_ROOT for model source contracts")
class NativeArchitectureTest(unittest.TestCase):
    def test_selected_model_source_matches_derived_architecture(self):
        sys.path.insert(0, os.environ["ATLAS_DERIVE_MODEL_ROOT"])
        from atlas_model_timing import configuration as selected_configuration
        from npu_model.hardware import arch_state, bank_conflict
        cfg, _ = selected_configuration()
        with patch.object(arch_state.ArchState, "__init__", side_effect=AssertionError("allocated architecture state")):
            facts = derive.derive_arch(cfg)
        self.assertEqual((facts["mreg_count"], facts["mreg_bytes"], facts["bf16_tile_bytes"]), (64, 1024, 2048))
        checker = bank_conflict.BankConflictChecker()
        for reg in range(facts["mreg_count"]):
            for row in (0, facts["mreg_rows"] - 1):
                checker.access_mreg(reg * facts["mreg_rows"] + row, reg, row, False, "mapping-check")
                bank = reg % facts["mreg_bank_modulus"]
                self.assertEqual(checker._mreg_ports[(False, bank)][:2],
                                 (reg, (reg // facts["mreg_bank_modulus"]) * facts["mreg_physical_row_stride"] + row))
        self.assertEqual(facts["vmem_bank_count"], 6)

    def test_selected_model_source_mutations_propagate_or_refuse(self):
        sys.path.insert(0, os.environ["ATLAS_DERIVE_MODEL_ROOT"])
        from atlas_model_timing import configuration as selected_configuration
        from npu_model.hardware import arch_state, bank_conflict
        cfg, _ = selected_configuration()
        original_source = derive._source

        def changed_mapping(module):
            tree = original_source(module)
            if module is bank_conflict:
                method = derive._method(tree, "BankConflictChecker", "access_mreg")
                for node in ast.walk(method):
                    if (isinstance(node, ast.BinOp) and isinstance(node.left, ast.Name)
                            and node.left.id == "reg" and isinstance(node.op, (ast.Mod, ast.FloorDiv))):
                        node.right = ast.Constant(16)
            return tree

        with patch.object(derive, "_source", side_effect=changed_mapping):
            self.assertEqual(derive.derive_arch(cfg)["mreg_physical_bank_count"], 16)

        def changed_storage(module):
            tree = original_source(module)
            if module is arch_state:
                method = derive._method(tree, "ArchState", "initialize_buffers")
                allocation = derive._member_value(method, "self.mrf")
                allocation.elt.args[0].op = ast.Add()
            return tree

        with patch.object(derive, "_source", side_effect=changed_storage):
            with self.assertRaisesRegex(derive.UnsupportedArchitecture, "storage allocation for self.mrf"):
                derive.derive_arch(cfg)


if __name__ == "__main__":
    unittest.main()
