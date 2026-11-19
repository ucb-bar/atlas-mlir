"""Completed panels retain exact output and avoid the final DRAM round trip."""
import re

import pytest

from atlas_mlir_native.already_quantized import prepare_source
from atlas_mlir_native.exponent_restoration import guard_record
from atlas_mlir_native.quantized_loop import execution_layout, virtual_source
from atlas_mlir_native.options import CodegenOptions, codegen_options
from test_already_quantized import source_request


@pytest.mark.parametrize('m,n,k,inline', [
    (64, 65, 2048, True), (128, 65, 768, True),
    (225, 35, 3073, False), (128, 65, 5632, False),
    (128, 35, 4096, True), (128, 35, 4097, False)])
def test_guard_receipt_names_every_emitted_factor_loop(m, n, k, inline):
    with codegen_options(CodegenOptions(io_memory_bytes=67108864)):
        plan = prepare_source(source_request(m=m, n=n, k=k, batch=()))
        layout = execution_layout(plan, operand_reuse='panel-cache')
        virtual = virtual_source(plan, layout)
        record = guard_record(plan, layout)
        rows = layout['operand_reuse_panel_m_tiles']
        if inline:
            assert record['schema'] == 'atlas.exponent_restoration_guards.v2'
            guards = record['guards']
            assert len(guards) == rows
            assert 'restore_m_header' not in virtual
        else:
            assert record['schema'] == 'atlas.exponent_restoration_guard.v1'
            guards = [record]
            assert 'restore_m_header' in virtual
        assert len(guards) == virtual.count('atlas.exponent_restoration_guard =')
        labels = set(re.findall(r'^\s*\^(\w+)\(', virtual, re.MULTILINE))
        for guard in guards:
            for key in ('guard_block', 'vector_block', 'scalar_block', 'factor_body',
                        'factor_done', 'join_block', 'factor_header'):
                assert guard[key] in labels
