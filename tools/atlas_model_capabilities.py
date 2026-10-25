"""Check representation limits before allocation; DMA curve parameters are derived."""
from atlas_model_diagnostics import UnsupportedModelTiming


def unsupported(field):
    raise UnsupportedModelTiming("unsupported capability: " + field)


def check_configuration(cfg):
    arch = getattr(cfg, "arch_state_config", None)
    if arch is None:
        unsupported("arch_state_config")
    expected_arch = dict(mrf_depth=32, mrf_width=32, num_m_registers=64,
                         num_wb_registers=2, wb_width=1024, num_x_registers=32,
                         num_e_registers=32, num_csrs=4096, vmem_size=1536 * 1024,
                         dram_base=0x80000000, dram_size=0x1000000000, numerics="rtl", randomize_init=False)
    for field, expected in expected_arch.items():
        value = getattr(arch, field, None)
        if type(value) is not type(expected) or value != expected:
            unsupported("arch_state_config." + field)
    for field, expected in dict(fetch_width=1, dma_num_channels=8, dma_beat_bytes=32,
                                dma_max_transfer_bytes=4096, vmem_bank_bytes=256 * 1024).items():
        value = getattr(cfg, field, None)
        if type(value) is not int or value != expected:
            unsupported(field)
    value = getattr(cfg, "dma_max_in_flight", None)
    if type(value) is not int or not 0 < value <= 2147483647:
        unsupported("dma_max_in_flight")
    if getattr(cfg, "dma_memory_backend", None) != "curve":
        unsupported("dma_memory_backend")
    units = getattr(cfg, "execution_units", None)
    expected_units = dict(Scalar0="ScalarExecutionUnit", Matrix0="MatrixExecutionUnitSystolic",
                          Matrix1="MatrixExecutionUnitInner", Vector0="VectorExecutionUnit",
                          XLU="CrossLaneExecutionUnit", DMA0="DmaExecutionUnit", LSU="LoadStoreUnit")
    if not isinstance(units, dict):
        unsupported("execution_units")
    if tuple(units) != tuple(expected_units):
        unsupported("execution_units dispatch order/membership")
    for field, expected in expected_units.items():
        if units.get(field) != expected:
            unsupported("execution_units." + field)
    isa = getattr(cfg, "isa", None)
    if not isinstance(isa, type) or isa.__module__ + "." + isa.__qualname__ != "npu_model.isa.IsaSpec":
        unsupported("isa")
