"""Fixed captured SmolVLA policy graph and existing128-row kernel panels."""
VISION = 'vlm_with_expert.vlm.model.vision_model.'
TEXT = 'vlm_with_expert.vlm.model.text_model.'
EXPERT = 'vlm_with_expert.lm_expert.'
ROLES = ('self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj',
         'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj')


def calls():
    result = []

    def add(module, shape, dtype, site, layer=0, role=0, image=0):
        rows, k, n = shape
        result.append(dict(index=len(result), module=module, rows=rows, input_width=k,
            output_width=n, input_dtype=dtype, output_dtype=dtype, site=site, layer=layer,
            role=role, image=image))

    for image in range(2):
        add(VISION+'embeddings.patch_embedding', (1024, 768, 768), 0, 7, image=image)
        for layer in range(12):
            prefix = VISION+f'encoder.layers.{layer}.'
            for role, name in enumerate(('q_proj', 'k_proj', 'v_proj', 'out_proj'), 1):
                add(prefix+'self_attn.'+name, (1024, 768, 768), 0, 7, layer, role, image)
            add(prefix+'mlp.fc1', (1024, 768, 3072), 0, 7, layer, 5, image)
            add(prefix+'mlp.fc2', (1024, 3072, 768), 0, 7, layer, 6, image)
        add('vlm_with_expert.vlm.model.connector.modality_projection.proj',
            (64, 12288, 960), 0, 7, 12, 7, image)
    add('state_proj', (1, 32, 960), 1, 0)
    for layer in range(16):
        widths = ((960, 960), (960, 320), (960, 320), (960, 960), (960, 2560), (960, 2560), (2560, 960))
        for role, (k, n) in enumerate(widths):
            add(TEXT+f'layers.{layer}.'+ROLES[role], (177, k, n), 0, 5, layer, role)
    for step in range(10):
        begin = len(result)
        add('action_in_proj', (50, 32, 720), 1, 1)
        add('action_time_mlp_in', (50, 1440, 720), 1, 2)
        add('action_time_mlp_out', (50, 720, 720), 1, 3)
        for layer in range(16):
            widths = ((50, 720, 960), (177, 320, 320) if layer % 2 else (50, 720, 320),
                (177, 320, 320) if layer % 2 else (50, 720, 320), (50, 960, 720),
                (50, 720, 2048), (50, 720, 2048), (50, 2048, 720))
            for role, shape in enumerate(widths):
                add(EXPERT+f'layers.{layer}.'+ROLES[role], shape,
                    1 if layer % 2 and role in (1, 2) else 0, 6, layer, role)
        add('action_out_proj', (50, 720, 32), 1, 4)
        for row in result[begin:]: row['step'] = step
    return result


def panels():
    result = []
    for call in calls():
        for start in range(0, call['rows'], 128):
            rows = min(128, call['rows']-start)
            k, n = call['input_width'], call['output_width']
            result.append(dict(index=len(result), call=call['index'], module=call['module'],
                logical_row_start=start, logical_mkn=[rows, k, n],
                physical_mkn=[(rows+31)//32*32, (k+31)//32*32, (n+31)//32*32]))
    return result


def reference_geometry():
    result = {
        'prefix_embeddings': ([1, 177, 960], 'float32'),
        'prefix_pad_mask': ([1, 177], 'bool'),
        'prefix_attention_mask': ([1, 177], 'bool'),
        'prefix_key_cache': ([16, 1, 177, 5, 64], 'uint16'),
        'prefix_value_cache': ([16, 1, 177, 5, 64], 'uint16'),
        'final_full_state': ([1, 50, 32], 'float32'),
        'final_actions': ([1, 50, 6], 'float32'),
    }
    for step in range(10):
        for field in ('input_state', 'velocity', 'output_state'):
            result[f'step_{step:02d}_{field}'] = ([1, 50, 32], 'float32')
        result[f'step_{step:02d}_timestep'] = ([1], 'float32')
    return result


def validate_reference(replay, manifest):
    """Admit this exact architecture's complete ordered reference, not a subset."""
    if (replay.get('status') != 'PASS_ATLAS_ARITHMETIC_FULL_POLICY_REPLAY'
        or replay.get('fresh_vision_executed') is not True
        or replay.get('input_state_rng_bookends_equal') is not True
        or manifest.get('schema') != 'atlas.full_kernel_assets.v1'
        or manifest.get('workload') != 'smol-full'
        or manifest.get('source_bookends_equal') is not True
        or manifest.get('finite_trajectory_verified') is not True):
        raise ValueError('Require complete finite full-policy arithmetic reference')
    expected_calls, expected_panels = calls(), panels()
    actual_calls, actual_panels = replay.get('calls'), manifest.get('kernels')
    if (not isinstance(actual_calls, list) or len(actual_calls) != len(expected_calls)
        or not isinstance(actual_panels, list) or len(actual_panels) != len(expected_panels)
        or manifest.get('kernel_count') != len(expected_panels)):
        raise ValueError('Incomplete full-policy call/panel count')
    for expected, actual in zip(expected_calls, actual_calls):
        rows, k, n = expected['rows'], expected['input_width'], expected['output_width']
        input_shape, output_shape = ([1, k], [1, n]) if expected['site'] == 0 else ([1, rows, k], [1, rows, n])
        if expected['site'] == 7 and expected['role'] == 0:
            input_shape, output_shape = [1, 3, 512, 512], [1, 768, 32, 32]
        dtype = 'torch.bfloat16' if expected['input_dtype'] == 0 else 'torch.float32'
        if (type(actual.get('index')) is not int or actual['index'] != expected['index']
            or actual.get('module') != expected['module']
            or actual['input'].get('shape') != input_shape or actual['output'].get('shape') != output_shape
            or actual['input'].get('dtype') != dtype or actual['output'].get('dtype') != dtype):
            raise ValueError('Full-policy native call order/dtype/geometry differs')
    scales = {}
    for expected, actual in zip(expected_panels, actual_panels):
        if any(type(actual.get(name)) is not type(value) or actual.get(name) != value
               or isinstance(value, list) and any(type(item) is not int for item in actual[name])
               for name, value in expected.items()):
            raise ValueError('Full-policy panel order or zero-padding geometry differs')
        for key in ('a_scale', 'b_scale', 'output_rescale_exponent'):
            if type(actual.get(key)) is not int or not -126 <= actual[key] <= 127:
                raise ValueError('Full-policy finite scale range differs')
        if actual['output_rescale_exponent'] != actual['a_scale'] + actual['b_scale']:
            raise ValueError('Full-policy rescale differs from global input/weight scales')
        if actual['call'] in scales and scales[actual['call']] != (actual['a_scale'], actual['b_scale']):
            raise ValueError('Full-policy operator scale changes between row panels')
        scales[actual['call']] = actual['a_scale'], actual['b_scale']
        m, k, n = actual['physical_mkn']
        loads = (m*k*(n//32)+k*n if k*n <= 262144 else m*k+k*n*(m//32)
                 if m*k <= 262144 else (m//32)*(n//32)*(2*k//32*1024))
        if (actual.get('dram_base') != 0x100000000 or actual.get('output_offset') != m*k+k*n+256
            or actual.get('total_dram_bytes') != m*k+k*n+512+2*m*n
            or actual.get('requested_dma') != {'load_bytes': loads, 'store_bytes': 2*m*n}):
            raise ValueError('Full-policy panel staging or DMA differs')
    return expected_calls
