"""Configuration admission follows compiler capabilities, before native state IO."""
from dataclasses import make_dataclass
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from atlas_model_capabilities import check_configuration
from atlas_model_diagnostics import UnsupportedModelTiming
from atlas_model_timing import configuration_values


def configuration():
    arch = dict(mrf_depth=32, mrf_width=32, num_m_registers=64, num_wb_registers=2,
                wb_width=1024, num_x_registers=32, num_e_registers=32, num_csrs=4096,
                vmem_size=1536 * 1024, dram_base=0x80000000, dram_size=0x1000000000,
                numerics="rtl", randomize_init=False)
    arch_type = make_dataclass("Arch", [(key, type(value)) for key, value in arch.items()])
    return SimpleNamespace(
        arch_state_config=arch_type(**arch), fetch_width=1, dma_num_channels=8,
        dma_beat_bytes=32, dma_max_transfer_bytes=4096, vmem_bank_bytes=256 * 1024,
        dma_max_in_flight=64, dma_memory_backend="curve",
        dma_memory_params=dict(curves="selected.json", frequency_ghz=0.75, queue_depth=13, window_beats=7),
        isa=type("IsaSpec", (), {"__module__": "npu_model.isa"}),
        execution_units=dict(Scalar0="ScalarExecutionUnit", Matrix0="MatrixExecutionUnitSystolic",
                             Matrix1="MatrixExecutionUnitInner", Vector0="VectorExecutionUnit",
                             XLU="CrossLaneExecutionUnit", DMA0="DmaExecutionUnit", LSU="LoadStoreUnit"))


class CapabilitiesTest(unittest.TestCase):
    def test_dispatch_membership_and_order_are_closed(self):
        for mutation in (lambda units: dict(reversed(list(units.items()))),
                         lambda units: dict(units, Extra="ScalarExecutionUnit")):
            cfg = configuration()
            cfg.execution_units = mutation(cfg.execution_units)
            with self.assertRaisesRegex(UnsupportedModelTiming, "dispatch order/membership"):
                check_configuration(cfg)

    def test_supported_config_and_independent_in_flight(self):
        cfg = configuration()
        check_configuration(cfg)
        for outstanding in (1, 17, 129):
            cfg.dma_max_in_flight = outstanding
            check_configuration(cfg)

    def test_each_architecture_limit_names_the_unsupported_capability(self):
        cfg = configuration()
        for field, value in vars(cfg.arch_state_config).items():
            changed = configuration()
            setattr(changed.arch_state_config, field, not value if isinstance(value, bool)
                    else value + 1 if isinstance(value, int) else "other")
            with self.subTest(field=field), self.assertRaisesRegex(
                    UnsupportedModelTiming, "unsupported capability: arch_state_config\\." + field):
                check_configuration(changed)
        for field in ("fetch_width", "dma_num_channels", "dma_beat_bytes", "dma_max_transfer_bytes", "vmem_bank_bytes"):
            changed = configuration()
            setattr(changed, field, getattr(changed, field) + 1)
            with self.subTest(field=field), self.assertRaisesRegex(
                    UnsupportedModelTiming, "unsupported capability: " + field):
                check_configuration(changed)

    def test_dormant_curve_path_settings_do_not_restrict_admission(self):
        cfg = configuration()
        for field in ("mxu0_matmul_latency_cycles", "mxu1_matmul_latency_cycles", "vpu_simple_op_latency_cycles",
                      "vpu_non_pipelineable_op_latency_cycles", "xlu_transform_latency_cycles",
                      "offchip_link_width_bits", "offchip_link_core_cycles_per_beat", "dma_offchip_command_words",
                      "vmem_bus_width_bits", "vmem_bus_core_cycles_per_beat", "vmem_bytes_per_cycle",
                      "dma_memory_latency_cycles", "dma_memory_cycles_per_beat"):
            setattr(cfg, field, 123)
        check_configuration(cfg)

    def test_float_configuration_fields_are_preserved_for_identity(self):
        cfg = configuration()
        cfg.legacy_float = 0.75
        values = configuration_values(cfg)
        self.assertEqual(values["legacy_float"], 0.75)
        self.assertEqual(values["dma_memory_params"]["frequency_ghz"], 0.75)


if __name__ == "__main__":
    unittest.main()
