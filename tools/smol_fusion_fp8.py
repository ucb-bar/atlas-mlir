"""Opt-in vision fc1 bias/GELU producer with natural FP8 fc2 input tiles."""
from atlas_epilogue import Epilogue, Step, TileInput
from atlas_program import assembly, dma_load_bytes, _reuse_module
from smol_fusion_program import DRAM_BASE, FusionProgram, channel_tiles, stage_panel, _sha
from chipyard.arena import relocate_assembly


def stage_fc1_fp8(row, initial, *, bias_bf16, output_scale_exponent, weight_source=None):
    """Preserve A/B and guards, append bias after the complete FP8 output extent.

    The explicit output exponent is the captured fc2 consumer A scale. This
    operation never rescans device values or substitutes a new consumer scale.
    """
    baseline = stage_panel(row, initial)
    if (row.get('recipe', row.get('program_policy')) != 'traffic-reuse-v1'
            or not row.get('module', '').endswith('.mlp.fc1') or '.vision_model.' not in row['module']):
        raise ValueError('FP8 GELU requires a qualified full-width vision fc1 producer')
    m, k, n = row['physical_mkn']
    offset = row['output_offset']
    extra = channel_tiles(bias_bf16, n)
    extra_offset = offset + m*n + 256
    scratch = (27, 28) if _reuse_module(m, k, n) == 'atlas_program_reuse_resident' else (10, 11)
    epilogue = Epilogue(m, n, offset, (Step('add', TileInput(extra_offset, 'channel')), Step('gelu_tanh')),
                        output_dtype='fp8', output_scale_exponent=output_scale_exponent,
                        scratch_registers=scratch)
    arena = initial[:offset] + b'\xcd'*(m*n) + b'Z'*256 + extra + b'Z'*256
    source = assembly(m, k, n, offset, row['output_rescale_exponent'], traffic_reuse=True, epilogue=epilogue, weight_source=weight_source)
    if weight_source is not None:
        from smol_weight_relocate import relocate_assembly as bound_relocate
        source = bound_relocate(source, 0x100000000, DRAM_BASE, weight_source=weight_source)
    else:
        source = relocate_assembly(source, 0x100000000, DRAM_BASE)
    metadata = dict(baseline.metadata, operations=['bias', 'gelu_tanh'], host_ops_on_success=['bias', 'gelu_tanh'],
                    output_dtype='fp8', output_layout='tiles', output_scale_exponent=output_scale_exponent,
                    output_bytes=m*n, extra_offset=extra_offset, extra_bytes=len(extra), extra_sha256=_sha(extra),
                    total_dram_bytes=len(arena), dram_bytes=len(arena),
                    requested_dma=dict(load_bytes=dma_load_bytes(m, k, n, traffic_reuse=True, epilogue=epilogue),
                                       store_bytes=m*n),
                    scratch_registers=list(scratch), source_sha256=_sha(source.encode()), initial_sha256=_sha(arena),
                    input_regions=[dict(name='bias', offset=extra_offset, size_bytes=len(extra), dtype='bf16',
                                        layout='channel', scale_exponent=0, channel_bf16_sha256=_sha(bias_bf16),
                                        sha256=_sha(extra))])
    return FusionProgram(source, arena, metadata)
