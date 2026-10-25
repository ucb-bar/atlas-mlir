"""Selected-source architecture mapping and capacity regressions."""
import ast
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from model_evidence import isolate_model_imports

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

    def read_mrf_fp8(self, vs):
        return self.mrf[vs].view(torch.float8_e4m3fn).reshape(self.cfg.mrf_depth, self.cfg.mrf_width // torch.float8_e4m3fn.itemsize)

    def write_mrf_fp8(self, vd, value):
        assert value.dtype == torch.float8_e4m3fn
        assert value.numel() == self.cfg.mrf_depth * self.cfg.mrf_width // torch.float8_e4m3fn.itemsize
        self.mrf[vd].view(torch.float8_e4m3fn)[:] = value.flatten()

    def write_xrf(self, rd, value):
        value &= 0xFFFFFFFF
        if rd == 0:
            return
        if rd < len(self.xrf) and self.xrf[rd] == value:
            return
        self.xrf[rd] = value
        if self.logger:
            self.logger.log_arch_value('xrf', rd, value)
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
                   "torch": SimpleNamespace(bfloat16=SimpleNamespace(itemsize=2), float8_e4m3fn=SimpleNamespace(itemsize=1))}
        def source(module):
            return ast.parse({bank_module: bank, arch_module: arch, lsu_module: lsu, dma_module: dma}[module])
        with patch.dict(sys.modules, modules), patch.object(derive, "_source", side_effect=source):
            return derive.derive_arch(cfg or configuration())

    def test_source_geometry_and_mapping_propagate_without_rtl_ring(self):
        facts = self.derive()
        expected = dict(schema="atlas.architecture.v1", mreg_count=64, mreg_rows=32,
                        mreg_row_bytes=32, mreg_bytes=1024, bf16_registers_per_tile=2,
                        bf16_pair_stride=1, bf16_tile_bytes=2048, bf16_element_bytes=2,
                        fp8_registers_per_tile=1, fp8_tile_bytes=1024, fp8_element_bytes=1,
                        mreg_bank_modulus=32, mreg_registers_per_bank=2,
                        mreg_physical_row_stride=32, vmem_bytes=1572864,
                        vmem_bank_bytes=262144, vmem_bank_count=6, dma_channels=8,
                        scalar_word_bytes=4)
        self.assertEqual({key: facts[key] for key in expected}, expected)
        self.assertFalse(any("ring" in key or "command_slots" in key for key in facts))
        self.assertEqual(self.derive(arch=ARCH_SOURCE.replace("0xFFFFFFFF", "0xFFFF"))["scalar_word_bytes"], 2)
        bank = BANK_SOURCE.replace("reg + 1", "reg + 2")
        arch = ARCH_SOURCE.replace("vd + 1", "vd + 2").replace("vs + 1", "vs + 2")
        self.assertEqual(self.derive(bank=bank, arch=arch)["bf16_pair_stride"], 2)
        bank = BANK_SOURCE.replace("reg % 32", "reg % 16").replace("reg // 32", "reg // 16")
        changed = self.derive(bank=bank)
        self.assertEqual((changed["mreg_bank_modulus"], changed["mreg_registers_per_bank"]), (16, 4))
        cfg = configuration()
        cfg.arch_state_config.num_m_registers, cfg.arch_state_config.mrf_depth = 128, 16
        bank = BANK_SOURCE.replace("reg >= 64", "reg >= 128").replace("* 32 + row", "* 16 + row")
        changed = self.derive(cfg, bank=bank)
        self.assertEqual((changed["mreg_logical_limit"], changed["mreg_physical_row_stride"], changed["mreg_bytes"]),
                         (128, 16, 512))

    def test_unreviewed_source_mappings_fail_with_precise_diagnostics(self):
        cases = [
            ("bank", "reg + 1", "reg + 2", "BF16 tile read mapping"),
            ("arch", "vd + 1", "vd + 2", "BF16 tile write mapping"),
            ("arch", "self.mrf[vs].view(torch.float8_e4m3fn)", "self.mrf[vs + 1].view(torch.float8_e4m3fn)", "FP8 register read mapping"),
            ("arch", "value.numel() == self.cfg.mrf_depth", "value.numel() == 2 * self.cfg.mrf_depth", "FP8 register write mapping"),
            ("arch", "self.xrf[rd] = value", "self.xrf[rd] = value & 0xFFFF", "scalar storage semantics"),
            ("arch", "if rd == 0:", "if rd == 1:", "scalar storage semantics"),
            ("arch", "0xFFFFFFFF", "0xFFFE", "whole number of bytes"),
            ("arch", "self.cfg.mrf_depth * self.cfg.mrf_width,", "self.cfg.mrf_depth * self.cfg.mrf_width * 2,", "storage allocation for self.mrf"),
            ("lsu", "address // self.config.vmem_bank_bytes", "address % self.config.vmem_bank_bytes", "LSU physical VMEM bank mapping"),
            ("dma", "line_address // self.bank_bytes", "line_address % self.bank_bytes", "DMA physical VMEM bank mapping"),
            ("dma", "self.config.dma_num_channels", "self.config.dma_num_channels * 2", "DMA configuration mapping for num_channels"),
            ("bank", "reg % 32", "reg & 31", "MREG bank mapping"),
            ("bank", "reg // 32", "reg // 16", "divisor differs"),
            ("bank", "+ row", "+ row + 1", "physical-row mapping"),
            ("bank", "key = (write, bank)", "bank += 1\n        key = (write, bank)", "reassignment"),
        ]
        sources = dict(bank=BANK_SOURCE, arch=ARCH_SOURCE, lsu=LSU_SOURCE, dma=DMA_SOURCE)
        for source, before, after, diagnostic in cases:
            with self.subTest(source=source, diagnostic=diagnostic):
                self.assertIn(before, sources[source])
                with self.assertRaisesRegex(derive.UnsupportedArchitecture, diagnostic):
                    self.derive(**{source: sources[source].replace(before, after)})

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
        facts = self.derive(cfg)
        for field, expected in dict(vmem_bank_count=24, mreg_bytes=2048, bf16_tile_bytes=4096,
                                    weight_buffer_count=4, weight_buffer_bytes=2048,
                                    xreg_count=64, dma_channels=16, dma_max_transfer_bytes=8192).items():
            self.assertEqual(facts[field], expected, field)


    def test_inconsistent_geometry_fails_precisely(self):
        for name, value, diagnostic in (("mrf_depth", 16, "physical-row stride"),
                ("num_m_registers", 32, "logical-register upper bound"),
                ("vmem_size", 1572865, "not divisible"), ("mrf_width", 31, "whole BF16 elements"),
                ("num_x_registers", 0, "num_x_registers must be a positive integer")):
            cfg = configuration()
            setattr(cfg.arch_state_config, name, value)
            with self.subTest(name=name), self.assertRaisesRegex(derive.UnsupportedArchitecture, diagnostic):
                self.derive(cfg)


@unittest.skipUnless(os.environ.get("ATLAS_DERIVE_MODEL_ROOT"), "set ATLAS_DERIVE_MODEL_ROOT for model source contracts")
class NativeArchitectureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        isolate_model_imports(cls, os.environ["ATLAS_DERIVE_MODEL_ROOT"])

    def test_selected_model_source_matches_derived_architecture(self):
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
