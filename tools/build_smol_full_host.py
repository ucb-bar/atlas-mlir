"""Build the fixed-observation full Smol policy guest from prepared kernel assets.

The resulting ELF executes real host math and raw Atlas launches. Building it
does not execute or qualify the guest or any model/device numerical result.
"""
import argparse
import copy
import io
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import struct
import subprocess
import sys

sys.dont_write_bytecode = True
import numpy as np
from smol_full_plan import calls, reference_geometry, validate_reference
from atlas_path_guard import checked as safe, lexical_absolute_path, map_metadata_paths
sys.path.insert(0, str(Path(__file__).absolute().parent / 'host_iss'))
from chipyard.arena import add_arguments as arena_arguments, audit, relocate_assembly, window
from chipyard.result_transport import add_arguments as uart_arguments, write_config
from chipyard.firesim_target import package as package_firesim
from placement import placement, arena_extent


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def dtype_name(value):
    return {'torch.bfloat16': 'uint16', 'torch.float32': 'float32'}[value]


def make_globals(kernel_count, parameters, references, panel_rows, call_rows, sizes):
    """Generate shared production declarations for the full and segmented driver."""
    ctype = {'uint16': 'uint16_t', 'float32': 'float', 'bool': 'uint8_t', 'uint8': 'uint8_t', 'int64': 'int64_t'}
    declarations, arrays = [], {}
    for name, item in parameters.items():
        declarations.append(f'extern const unsigned char {item["symbol"]}[];')
    for item in references:
        elements = item['bytes'] // np.dtype(item['dtype']).itemsize
        declarations += [f'{ctype[item["dtype"]]} {item["actual_symbol"]}[{elements}];',
                         f'extern const unsigned char {item["expected_symbol"]}[];']
        arrays[item['name']] = item['actual_symbol']

    def p(name, cast='uint16_t'):
        return f'((const {cast}*){parameters[name]["symbol"]})'

    def ref(name): return arrays[name]
    data = ['#include "smol_policy_raw.h"', '#include "atlas_result_transport.h"', '#include "smol_vision_attention.h"',
            '#include "smol_host_bits.h"', '#include <string.h>', *declarations]
    data += ['_Static_assert(sizeof(smol_policy_panel)==144,"panel ABI");',
             '_Static_assert(sizeof(smol_policy_call)==80,"call ABI");',
             '_Static_assert(sizeof(smol_policy_raw_context)==160,"context ABI");',
             '_Static_assert(sizeof(atlas_host)==112,"host ABI");']
    data += ['extern const unsigned char '+name+'[];' for name in sorted({x for row in panel_rows for x in row['symbols']})]
    data += [f'const smol_policy_panel smol_full_panels[{kernel_count}]={{',
             *[row['c']+',' for row in panel_rows], '};',
             'const smol_policy_call smol_full_calls[1421]={', *[row+',' for row in call_rows], '};']
    data += ['__attribute__((section(".tohost"),aligned(64),used)) volatile uint64_t tohost=0;',
             '__attribute__((section(".tohost"),aligned(64),used)) volatile uint64_t fromhost=0;',
             '__attribute__((section(".handoff"),aligned(64),used)) volatile uint64_t atlas_workload_mailbox[16]={UINT64_C(0x534d4f4c46554c4c)};',
             'atlas_host smol_full_host;', 'static atlas_result_uart smol_result_uart;', 'static atlas_result smol_result;', 'smol_policy_state smol_full_graph_state;',
             'smol_vision_state smol_full_vision_states[2];',
             f'uint32_t smol_full_cycles[{kernel_count}];',
             'uint16_t smol_full_input_copy[3145728];', 'uint8_t smol_full_activation_copy[786432];',
             'uint16_t smol_full_image[786432],smol_full_image_output[2][61440];']
    for name in ('tokens', 'normalized', 'query', 'key', 'value', 'attended', 'projected', 'shuffle'):
        data.append(f'uint16_t smol_v_{name}[786432];')
    data += ['uint16_t smol_v_large[3145728],smol_v_activated[3145728],smol_v_output[61440];']
    frame_names = ('normalized', 'query', 'key', 'value', 'rotary_query', 'rotary_key', 'joined_key',
        'joined_value', 'attention', 'projected', 'residual', 'postnorm', 'gate', 'up', 'activated',
        'down', 'output', 'bf16_probabilities', 'scores', 'float_probabilities', 'workspace',
        'cross_input', 'cross_key', 'cross_value')
    for name in frame_names:
        count, kind = 177*960, 'uint16_t'
        if name in ('key', 'value', 'rotary_key'): count = 177*320
        elif name in ('joined_key', 'joined_value'): count = 227*320
        elif name in ('gate', 'up', 'activated'): count = 177*2560
        elif name in ('scores', 'float_probabilities', 'bf16_probabilities'):
            count, kind = 15*177*227, 'uint16_t' if name == 'bf16_probabilities' else 'float'
        elif name == 'workspace': count, kind = 227, 'float'
        elif name.startswith('cross_'): count, kind = 177*320, 'float'
        data.append(f'{kind} smol_b_{name}[{count}];')
    data += ['float smol_g_state_projection[960],smol_g_action_embedding[36000],smol_g_action_time[72000];',
        'float smol_g_time_hidden[36000],smol_g_time_activated[36000],smol_g_suffix_input[36000],smol_g_velocity[1600];',
        'uint16_t smol_g_prefix_final[169920],smol_g_expert_final[36000];',
        'uint8_t smol_g_prefix_mask[31329],smol_g_self_mask[11350],smol_g_cross_mask[8850];',
        'int64_t smol_g_groups[177],smol_g_positions[177],smol_g_cross_positions[50];']
    data += ['smol_policy_raw_context smol_full_raw={', '.host=&smol_full_host,.panels=smol_full_panels,.calls=smol_full_calls,',
        f'.panel_count={kernel_count},.call_count=1421,.dram=(uint8_t*)(uintptr_t)UINT64_C({sizes["arena"]}),',
        f'.dram_capacity={sizes["dram"]},.activation_copy=smol_full_activation_copy,.activation_capacity=786432,',
        '.input_copy=smol_full_input_copy,.input_capacity=3145728,.max_polls=100000000,.cycle_delta=smol_full_cycles};']
    data += ['const smol_vision_parameters smol_full_vision_parameters={.position='+p('vision.position')+',.layers={']
    for layer in range(12):
        data.append('{'+','.join(p(f'vision.{layer}.norm{n}.{field}') for n in (1, 2) for field in ('weight', 'bias'))+'},')
    data += ['},.post_weight='+p('vision.post.weight')+',.post_bias='+p('vision.post.bias')+',.gelu='+p('gelu')+',.epsilon=1e-6f};',
        'const smol_vision_frame smol_full_vision_frame={'+','.join('smol_v_'+name for name in
        ('tokens', 'normalized', 'query', 'key', 'value', 'attended', 'projected', 'large', 'activated', 'shuffle', 'output'))+'};']
    data += ['const smol_policy_parameters smol_full_policy_parameters={.prefix={']
    for family in ('prefix', 'expert'):
        if family == 'expert': data.append('},.expert={')
        for layer in range(16):
            kind = 0 if family == 'prefix' else 1+layer%2
            data.append('{'+str(kind)+','+p(f'{family}.{layer}.input')+','+p(f'{family}.{layer}.post')
                +',1e-5f,1e-5f,'+p('cosine', 'float')+','+p('sine', 'float')+',-1,8193},')
    data += ['},.prefix_final_weight='+p('prefix.final')+',.expert_final_weight='+p('expert.final')
        +',.language_embedding='+p('language.embedding')+',.language_scale=0x1.efbdecp+4f,.final_epsilon=1e-5f,'
        +'.timestep='+p('timestep', 'float')+',.time_embedding='+p('time.embedding', 'float')+'};']
    data += ['const smol_policy_frame smol_full_policy_frame={.block={'+','.join('smol_b_'+name for name in frame_names)+'},',
        '.prefix_input='+ref('prefix_embeddings')+',.state_projection=smol_g_state_projection,',
        '.prefix_final=smol_g_prefix_final,.expert_final=smol_g_expert_final,',
        '.cache_key='+ref('prefix_key_cache')+',.cache_value='+ref('prefix_value_cache')+',',
        '.prefix_mask=smol_g_prefix_mask,.self_mask=smol_g_self_mask,.cross_mask=smol_g_cross_mask,',
        '.groups=smol_g_groups,.positions=smol_g_positions,.cross_positions=smol_g_cross_positions,',
        '.action_embedding=smol_g_action_embedding,.action_time=smol_g_action_time,.time_hidden=smol_g_time_hidden,',
        '.time_activated=smol_g_time_activated,.suffix_input=smol_g_suffix_input,.velocity=smol_g_velocity};']
    for field in ('input_state', 'output_state', 'velocity', 'timestep'):
        data.append('float *const smol_step_'+field+'[10]={'+','.join(ref(f'step_{step:02d}_{field}') for step in range(10))+'};')
    return '\n'.join(data)+'\n'


def relocation_pairs(values):
    pairs = []
    for value in values:
        if value.count('=') != 1:
            raise ValueError('Relocation requires OLD=NEW')
        old, new = (lexical_absolute_path(part) for part in value.split('=', 1))
        safe(new)
        if old == new or any(old.is_relative_to(a) or a.is_relative_to(old) for a, _ in pairs):
            raise ValueError('Relocation prefixes must be distinct and nonoverlapping')
        pairs.append((old, new))
    return pairs


def relocation_mapper(pairs):
    def mapper(path):
        for old, new in pairs:
            if path.is_relative_to(old):
                return new / path.relative_to(old)
        return path
    return mapper if pairs else None


def progress_launches(manifest):
    """Emit one checked boundary per vision, prefix/expert layer and Euler step."""
    ends = {item['call']: item['index'] + 1 for item in manifest['kernels']}
    return [ends[call['index']] for call in calls()
            if (call['site'] == 7 and call['role'] in (6, 7))
            or (call['site'] in (5, 6) and call['role'] == 6) or call['site'] == 4]


def select_programs(args, captured, bundle, out, products, read):
    """Admit captured bytes before generating any selected or relocated stream."""
    from atlas_program import assembly, code_sources, dma_load_bytes, _TRAFFIC_REUSE_MODULES
    from run_model_tile import configure_model, model_binding
    _, _, assemble, _ = configure_model(args.model_root)
    args.assemble = assemble
    for name in code_sources(traffic_reuse=args.traffic_reuse):
        read(Path(__file__).absolute().parent / name)
    read(Path(__file__).with_name('run_model_tile.py'))
    selected = copy.deepcopy(captured)
    selected['placement'] = args.physical
    selected['program_policy'] = 'qualified-reuse-v1' if args.traffic_reuse else 'captured-baseline-v1'
    assembled = {}
    def encode(source):
        if source not in assembled:
            words = assemble(io.StringIO(source)).assemble()
            assembled[source] = (words, b''.join(struct.pack('<I', word) for word in words))
        return assembled[source]
    for item in selected['kernels']:
        original_source = bundle.read(item['source'])
        original_raw = bundle.read(item['original'])
        _, admitted_raw = encode(original_source.decode())
        if admitted_raw != original_raw:
            raise ValueError('Captured source differs from original binary')
        m, k, n = item['physical_mkn']
        reuse = args.traffic_reuse and (m, k, n) in _TRAFFIC_REUSE_MODULES
        source = (assembly(m, k, n, item['output_offset'], item['output_rescale_exponent'],
                           traffic_reuse=True) if reuse else original_source.decode())
        source = relocate_assembly(source, item['dram_base'], args.arena_base)
        words, raw = encode(source)
        if not words or words[-1] != 0x73 or len(words) > 8192:
            raise ValueError('Selected Smol program exceeds raw adapter capacity or lacks ECALL')
        for field, payload, suffix in (('source', source.encode(), '.S'), ('original', raw, '.bin')):
            name = 'assets/' + sha(payload) + suffix
            if name not in products and name not in bundle.sums:
                path = safe(out/name); path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(payload)
                products[name] = sha(payload)
            item[field] = name
        if reuse:
            item['requested_dma'] = dict(load_bytes=dma_load_bytes(m, k, n, traffic_reuse=True), store_bytes=2*m*n)
        item['dram_base'] = args.arena_base
        item['program_policy'] = 'traffic-reuse-v1' if reuse else 'baseline-v1'
    binding = model_binding(args.model_root)
    for name, digest in binding['imported_source_sha256'].items():
        read(args.model_root/name, digest)
    read(args.model_root/'npu_model/configs/memory_curves/ee290sim_vcs_probe.json', binding['curve_sha256'])
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('assets', 'replay', 'merlin', 'attention', 'runtime', 'bundle-tools',
                 'support', 'compiler', 'linker', 'out'):
        parser.add_argument('--'+name, required=True, type=safe)
    arena_arguments(parser)
    uart_arguments(parser)
    parser.add_argument('--target', choices=('sim', 'firesim'), default='sim')
    parser.add_argument('--traffic-reuse', action='store_true')
    parser.add_argument('--phase-reference', type=safe, required=True)
    parser.add_argument('--rodata-after-arena', action='store_true')
    parser.add_argument('--relocate', action='append', default=[], metavar='OLD=NEW')
    parser.add_argument('--model-root', required=True, type=safe)
    parser.add_argument('--table-python', required=True, type=safe)
    args = parser.parse_args()
    if args.arena_base is None: args.arena_base = 0x100000000
    args.relocations = relocation_pairs(args.relocate)
    args.physical = placement(dict(placement=dict(arena_base=args.arena_base,
        dram_base=args.dram_base, dram_bytes=args.dram_bytes)))
    if not os.environ.get('TMPDIR') or os.environ.get('PYTHONDONTWRITEBYTECODE') != '1':
        raise ValueError('Require explicit bounded TMPDIR and disabled bytecode')
    safe(os.environ['TMPDIR'])
    pins = {}

    def merge(path, digest):
        name = str(safe(path))
        if name in pins and pins[name] != digest: raise ValueError('Source/artifact pin conflict: '+name)
        pins[name] = digest

    def pin(path, expected=None):
        path = safe(path)
        with path.open('rb') as stream: digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if expected is not None and digest != expected: raise ValueError('Changed bound file: '+str(path))
        merge(path, digest)
        return digest

    def read(path, expected=None):
        path = safe(path)
        raw = path.read_bytes()
        digest = sha(raw)
        if expected is not None and digest != expected: raise ValueError('Changed bound bytes: '+str(path))
        merge(path, digest)
        return raw

    read(__file__)
    read(Path(__file__).parent / 'smol_full_plan.py')
    for relative in ('chipyard/arena.py', 'chipyard/result_transport.py', 'chipyard/firesim_target.py',
                     'host_iss/placement.py'):
        read(Path(__file__).absolute().parent / relative)
    # Select the already-reviewed common reader explicitly. Never import a
    # same-name cached module from an unrelated checkout.
    sys.path[:0] = [str(args.bundle_tools), str(args.bundle_tools.parent)]
    from full_bundle import KernelAssets
    from elf_image import Image
    from atlas_host_build import elf_sections
    for name, module in tuple(sys.modules.items()):
        if name in ('full_bundle', 'bundle', 'block_bundle', 'elf_image') or name.startswith('atlas_'):
            path = safe(module.__file__)
            if path.parent not in (args.bundle_tools, args.bundle_tools.parent):
                raise ValueError('Build helper imported outside selected tools: '+name)
            read(path)
    bundle = KernelAssets(args.assets, 'smol-full')
    pin(args.assets/'SHA256SUMS', bundle.checksums_sha256)
    manifest_raw = read(args.assets / 'kernels.json')
    manifest = json.loads(manifest_raw)
    captured_manifest = json.loads(manifest_raw)
    replay_raw = read(args.replay)
    replay = json.loads(replay_raw)
    graph = validate_reference(replay, manifest)
    arena_extent(dict(placement=args.physical), max(item['total_dram_bytes'] for item in manifest['kernels']))
    replay = map_metadata_paths(replay, relocation_mapper(args.relocations))
    if manifest['replay_receipt']['sha256'] != sha(replay_raw): raise ValueError('Staged reference binding differs')
    # Earlier producer bookends remain authenticated historical provenance.
    # This build pins the exact prepared assets and operands it actually reads.
    # Admission, input identity and geometry finish before either output tree
    # exists or a compiler is invoked.
    weights_path = safe(replay['source']['weights']['path'])
    pin(weights_path, replay['source']['weights']['sha256'])
    pin(args.compiler)
    pin(args.linker)
    trajectory = replay['trajectory']
    read(trajectory['path'], trajectory['sha256'])
    with np.load(safe(trajectory['path']), allow_pickle=False) as archive:
        data = {name: archive[name].copy() for name in archive.files}
    geometries = reference_geometry()
    if set(data) != set(geometries): raise ValueError('Incomplete full-policy final reference inventory')
    for name, (shape, dtype) in geometries.items():
        value = data[name]
        if list(value.shape) != shape or value.dtype != np.dtype(dtype): raise ValueError('Reference type/shape differs: '+name)
        if value.dtype == np.uint16:
            if np.any((value & 0x7f80) == 0x7f80): raise ValueError('Nonfinite BF16 reference: '+name)
        elif not np.isfinite(value).all(): raise ValueError('Nonfinite reference: '+name)
    for step in range(10):
        before, velocity, after = [data[f'step_{step:02d}_'+field] for field in ('input_state', 'velocity', 'output_state')]
        if not np.array_equal((before + np.float32(-0.1)*velocity).view(np.uint32), after.view(np.uint32)):
            raise ValueError('Reference own Euler update differs')
        if step and data[f'step_{step-1:02d}_output_state'].tobytes() != before.tobytes():
            raise ValueError('Reference denoising feedback chain differs')
    if (data['final_full_state'].tobytes() != data['step_09_output_state'].tobytes()
        or data['final_actions'].tobytes() != data['final_full_state'][:, :, :6].copy().tobytes()):
        raise ValueError('Reference final action selection differs')
    command = replay['command']
    observation_path = safe(command[command.index('--observation-input')+1])
    observation_digest = next(item['sha256'] for item in replay['source_bookends'] if item['path'] == str(observation_path))
    read(observation_path, observation_digest)
    with np.load(observation_path, allow_pickle=False) as archive:
        observation = {name: archive[name].copy() for name in archive.files}
    with weights_path.open('rb') as stream:
        length = struct.unpack('<Q', stream.read(8))[0]
        if not 0 < length <= 16*1024*1024: raise ValueError('Checkpoint header extent differs')
        header = json.loads(stream.read(length))

    def checkpoint_tensor(name):
        row = header['model.'+name]
        dtype = {'BF16': '<u2', 'F32': '<f4'}[row['dtype']]
        start, end = row['data_offsets']
        elements = int(np.prod(row['shape'], dtype=np.int64))
        if (type(start) is not int or type(end) is not int or not 0 <= start < end
            or end-start != elements*np.dtype(dtype).itemsize
            or 8+length+end > replay['source']['weights']['bytes']): raise ValueError('Checkpoint tensor extent differs')
        with weights_path.open('rb') as stream:
            stream.seek(8+length+start)
            raw = stream.read(end-start)
        if len(raw) != end-start: raise ValueError('Truncated checkpoint tensor')
        return np.frombuffer(raw, dtype).copy().reshape(row['shape'])

    # All learned host operands come directly from the admitted original
    # checkpoint. Only the selected observation's raw embedding rows are kept.
    host_values = {}
    vision = 'vlm_with_expert.vlm.model.vision_model.'
    host_values['vision.position'] = checkpoint_tensor(vision+'embeddings.position_embedding.weight')
    for layer in range(12):
        for n in (1, 2):
            for field in ('weight', 'bias'):
                host_values[f'vision.{layer}.norm{n}.{field}'] = checkpoint_tensor(vision+f'encoder.layers.{layer}.layer_norm{n}.{field}')
    for field in ('weight', 'bias'): host_values['vision.post.'+field] = checkpoint_tensor(vision+'post_layernorm.'+field)
    for family, source in (('prefix', 'vlm_with_expert.vlm.model.text_model.'), ('expert', 'vlm_with_expert.lm_expert.')):
        for layer in range(16):
            for short, field in (('input', 'input_layernorm'), ('post', 'post_attention_layernorm')):
                host_values[f'{family}.{layer}.{short}'] = checkpoint_tensor(source+f'layers.{layer}.{field}.weight')
        host_values[family+'.final'] = checkpoint_tensor(source+'norm.weight')
    language = checkpoint_tensor('vlm_with_expert.vlm.model.text_model.embed_tokens.weight')
    tokens = observation['language_tokens'].reshape(48)
    if tokens.dtype != np.int64 or np.any(tokens < 0) or np.any(tokens >= language.shape[0]): raise ValueError('Language token domain differs')
    host_values['language.embedding'] = language[tokens].copy()
    for name, shape in (('images', (2, 3, 512, 512)), ('state', (1, 32)), ('noise', (1, 50, 32)),
                        ('image_masks', (2,)), ('language_mask', (48,)), ('language_tokens', (48,))):
        host_values['observation.'+name] = observation[name].reshape(shape).copy()
    for name in ('cosine', 'sine'):
        # Reuse the already-qualified source function table without changing
        # source arithmetic or generating a new trigonometric approximation.
        contract_path = safe(args.support.parents[1] / 'host/smol-epoch3-contract.json')
        contract_raw = read(contract_path)
        table_contract = map_metadata_paths(json.loads(contract_raw), relocation_mapper(args.relocations))
        item = table_contract['assets'][name]
        raw = read(item['file']['path'], item['file']['sha256'])
        host_values[name] = np.frombuffer(raw, '<f4').copy().reshape(item['shape'])
    for name, value in host_values.items():
        if value.dtype == np.uint16 and np.any((value & 0x7f80) == 0x7f80): raise ValueError('Nonfinite host BF16 parameter: '+name)
        if value.dtype.kind == 'f' and not np.isfinite(value).all(): raise ValueError('Nonfinite host FP32 parameter: '+name)
    for path in (args.out, args.out.with_name(args.out.name+'-build')):
        if path.exists(): raise ValueError('Refuse to overwrite a full-host artifact tree')
    # The archival native Torch provider runs separately from the newer model assembler.
    work = args.out.with_name(args.out.name+'-build')
    work.mkdir(parents=True, exist_ok=False)
    tables = work/'tables'
    worker = Path(__file__).with_name('smol_full_tables.py')
    pin(args.table_python)
    read(worker)
    command = [str(args.table_python), '-B', str(worker), '--replay', str(args.replay), '--out', str(tables)]
    for value in args.relocate:
        command += ['--relocate', value]
    env = dict(os.environ, PYTHONPATH=str(Path(replay['source']['torch_cpu_library']['path']).parents[2]),
               PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', HF_HOME=str(work/'tables-cache/hf'),
               TORCH_HOME=str(work/'tables-cache/torch'), XDG_CACHE_HOME=str(work/'tables-cache/xdg'))
    result = subprocess.run(command, cwd=work, env=env, capture_output=True, timeout=300)
    (work/'tables.log').write_bytes(result.stdout+result.stderr)
    if result.returncode:
        raise ValueError('Native function-table producer failed: '+result.stderr.decode(errors='replace'))
    table_receipt = json.loads(read(tables/'receipt.json'))
    if (table_receipt['status'] != 'PREPARED_NATIVE_FUNCTION_TABLES_ONLY'
            or table_receipt['source_bookends_equal'] is not True or table_receipt['replay_sha256'] != sha(replay_raw)):
        raise ValueError('Native function-table receipt differs')
    for path, digest in table_receipt['sources_sha256'].items():
        pin(path, digest)
    expected_tables = {'gelu':([65536], 'uint16'), 'timestep':([10], 'float32'),
                       'time.embedding':([10,720], 'float32')}
    if set(table_receipt['products']) != set(expected_tables):
        raise ValueError('Native function-table inventory differs')
    for name, (shape, dtype) in expected_tables.items():
        item = table_receipt['products'][name]
        if item['shape'] != shape or item['dtype'] != dtype or Path(item['file']).name != item['file']:
            raise ValueError('Native function-table geometry or member differs')
        payload = read(tables/item['file'], item['sha256'])
        value = np.frombuffer(payload, dtype).copy().reshape(shape)
        if (dtype == 'uint16' and np.any((value & 0x7f80) == 0x7f80)
                or dtype == 'float32' and not np.isfinite(value).all()):
            raise ValueError('Native function-table values are nonfinite')
        host_values[name] = value
    args.table_worker = dict(command=command, pythonpath=env['PYTHONPATH'], operational_wall_cap_seconds=300,
                             receipt_sha256=pins[str(safe(tables/'receipt.json'))])
    def capture_tensor(row):
        path = Path(row['file'])
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('Invalid capture operand member')
        raw = read(args.replay.parent/path, row['sha256'])
        return np.frombuffer(raw, '<u2' if row['dtype'] == 'torch.bfloat16' else '<f4').copy().reshape(row['shape'])
    for module, row in replay['parameters'].items():
        if 'bias' in row:
            value = checkpoint_tensor(module+'.bias')
            if value.tobytes() != capture_tensor(row['bias']).tobytes(): raise ValueError('Captured bias differs from original checkpoint')
            host_values['bias.'+module] = value
    # Compiler/linker output begins only after all complete-reference gates.
    out, work = args.out, args.out.with_name(args.out.name+'-build')
    out.mkdir(parents=True, exist_ok=False)
    safe(out/'host/data').mkdir(parents=True)
    safe(work/'source').mkdir()
    args.uart = write_config(work/'source', args)
    symbols, products, payload_labels = {}, {}, {}
    (out/'host/captured-kernels.json').write_bytes(manifest_raw)
    products['host/captured-kernels.json'] = sha(manifest_raw)

    def asset(raw, label, relative=None):
        relative = relative or 'host/data/'+sha(raw)+'.bin'
        path = safe(out/relative)
        if relative not in products:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
            products[relative] = sha(raw)
        payload_labels.setdefault(relative, []).append(label)
        symbols[label] = raw
        return relative

    parameters = {}
    for index, (name, value) in enumerate(sorted(host_values.items())):
        value = np.ascontiguousarray(value)
        label = f'smol_full_param_{index:03d}'
        raw = value.tobytes()
        parameters[name] = dict(symbol=label, file=asset(raw,label), sha256=sha(raw), bytes=len(raw),
                                shape=list(value.shape), dtype=str(value.dtype))
    references = []
    for index, (name, value) in enumerate(sorted(data.items())):
        raw = value.tobytes()
        actual, expected = 'smol_full_actual_'+name, 'smol_full_expected_'+name
        references.append(dict(name=name, shape=list(value.shape), dtype=str(value.dtype),
            actual_symbol=actual, expected_symbol=expected, bytes=len(raw), sha256=sha(raw), file=asset(raw,expected)))
    manifest = select_programs(args, manifest, bundle, out, products, read)
    weight_symbols, program_symbols, panel_rows = {}, {}, []
    def payload(name):
        return read(out/name, products[name]) if name in products else bundle.read(name)
    for item in manifest['kernels']:
        names = []
        for field, registry in (('weights', weight_symbols), ('original', program_symbols)):
            name = item[field]
            raw = payload(name)
            if name not in registry:
                label = f'smol_full_{field}_{len(registry):03d}'
                registry[name] = label
                asset(raw,label,name)
            names.append(registry[name])
        weight_symbol, program_symbol = names
        program = np.frombuffer(payload(item['original']), '<u4')
        positions = np.flatnonzero((program&0x7fffff) == 0x25f)
        if positions.size != 1: raise ValueError('Require unique original rescale literal')
        m, k, n = item['physical_mkn']
        rows, lk, ln = item['logical_mkn']
        values = [f'(const uint32_t*){program_symbol}', weight_symbol, str(program.size), str(int(positions[0])),
            *map(str,(m,k,n,item['logical_row_start'],rows,lk,ln,item['call'],item['output_offset'],
                     item['total_dram_bytes'],k*n,item['requested_dma']['load_bytes'],item['requested_dma']['store_bytes'],
                     item['a_scale'],item['b_scale']))]
        panel_rows.append(dict(c='{'+','.join(values)+'}', symbols=[program_symbol,weight_symbol]))
    call_rows = []
    for call in graph:
        indices = [row['index'] for row in manifest['kernels'] if row['call'] == call['index']]
        bias = parameters.get('bias.'+call['module'])
        values = [str(call['rows']),str(call['input_width']),str(call['output_width']),str(indices[0]),str(len(indices)),
            '(const void*)'+bias['symbol'] if bias else '0', *[str(call[name]) for name in
            ('site','layer','role','image','input_dtype','output_dtype')],str(call['output_dtype']),
            str(manifest['kernels'][indices[0]]['a_scale'])]
        call_rows.append('{'+','.join(values)+'}')
    dram_bytes = max(item['total_dram_bytes'] for item in manifest['kernels'])
    source = make_globals(len(panel_rows), parameters, references, panel_rows, call_rows,
                       {'dram':dram_bytes, 'vocabulary':language.shape[0], 'arena':args.arena_base})
    from smol_full_driver_emit import emit
    read(Path(__file__).parent/'smol_full_driver_emit.py')
    read(Path(__file__).parent/'smol_full_phases.py')
    source, args.phase_data = emit(source, parameters, references, language.shape[0],
        args.phase_reference, read, asset)
    safe(work/'main.c').write_text(source)
    # Remaining source-copy, compilation and strict full-host ELF admission are
    # below; source and artifact bookends cover generated inputs as well.
    return finish_build(args, bundle, manifest, replay, out, work, parameters, references,
                        symbols, products, payload_labels, dram_bytes, pins, read, pin)


def full_link(template, dram_bytes, arena_base=0x100000000, *, rodata_after_arena=False, dram_end=0x1080000000):
    """Retain the arena load segment and bound low host memory before it."""
    window(arena_base, dram_bytes)
    if template.count('dram PT_LOAD FLAGS(6);') != 1 or template.count(':dram') != 1:
        raise ValueError('Unexpected admitted linker template')
    bound = f'__stack_top <= {dram_end}' if rodata_after_arena else f'__stack_top < {arena_base}'
    value = template.replace('__stack_top < 0x90000000', bound)
    if rodata_after_arena:
        value = value.replace('  .rodata :', f'  . = MAX(., {arena_base + dram_bytes});\n  .rodata :')
    value = value.replace('.atlas_dram 0x100000000', f'.atlas_dram {arena_base}')
    value = value.replace('__atlas_dram_end <= 0x200000000',
                          f'__atlas_dram_end <= {(arena_base & ~0xffffffff) + (1 << 32)}')
    value, count = re.subn(r'(\.atlas_dram[^{}]*\{.*?\. \+= )\d+;',
                          lambda match: match[1]+str(dram_bytes)+';', value, flags=re.S)
    if count != 1:
        raise ValueError('Unexpected DMA reservation template')
    return value


def copy_header_closure(starts, target, providers, copy, read):
    """Close flat quoted includes across explicitly selected source providers."""
    queue, seen = list(starts), set()
    while queue:
        path = safe(queue.pop())
        if path in seen:
            continue
        seen.add(path)
        for name in re.findall(r'^\s*#\s*include\s+"([^"\n]+)"', read(path).decode(), re.MULTILINE):
            if Path(name).name != name or name in ('.', '..'):
                raise ValueError('Quoted source headers must be flat members')
            destination = safe(target/name)
            if destination.exists():
                queue.append(destination)
                continue
            candidates = []
            for provider in providers:
                candidate = safe(provider/name)
                if candidate.exists():
                    candidates.append((candidate, read(candidate)))
            if not candidates or any(raw != candidates[0][1] for _, raw in candidates):
                raise ValueError('Missing or conflicting quoted source header: '+name)
            queue.append(copy(candidates[0][0], destination))


def finish_build(args, bundle, manifest, replay, out, work, parameters, references,
                 symbols, products, payload_labels, dram_bytes, pins, read, pin):
    from elf_image import Image
    from atlas_host_build import elf_sections
    sys.path.insert(0, str(Path(__file__).absolute().parent/'host_iss'))
    import smol_full_contract as contract_module
    read(contract_module.__file__)
    source_dir = safe(work/'source')
    copied = {}

    def copy(source, target):
        raw = read(source)
        target = safe(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if target.read_bytes() != raw: raise ValueError('Conflicting copied compilation input')
        else: target.write_bytes(raw)
        read(target, sha(raw))
        copied[str(target.relative_to(work))] = sha(raw)
        return target

    merlin_names = ('smol_host_ops', 'smol_host_attention', 'smol_host_softmax', 'smol_host_rope', 'smol_host_controls',
        'host_matrix_fp8pack', 'smol_policy_ops', 'smol_policy_block', 'smol_policy_graph',
        'smol_vision_ops', 'smol_vision_graph')
    compile_sources = [safe(work/'main.c')]
    for name in merlin_names:
        for suffix in ('.c', '.h'):
            target = copy(args.merlin/(name+suffix), source_dir/(name+suffix))
            if suffix == '.c': compile_sources.append(target)
    copy(args.merlin/'smol_host_bits.h', source_dir/'smol_host_bits.h')
    for name in ('smol_policy_raw.c', 'smol_policy_raw.h', 'atlas_result.h'):
        target = copy(Path(__file__).absolute().parents[1]/'runtime'/name, source_dir/name)
        if name.endswith('.c'): compile_sources.append(target)
    for name in ('smol_full_driver.c','smol_full_driver.h','smol_full_phase.h'):
        target = copy(Path(__file__).absolute().parents[1]/'runtime'/name, source_dir/name)
        if name.endswith('.c'): compile_sources.append(target)
    read(source_dir/'atlas_result_transport.h')
    read(source_dir/'result-transport.json')
    for name in ('smol_vision_attention.c', 'smol_vision_attention.h', 'smol_host_bits.h'):
        target = copy(args.attention/name, source_dir/name)
        if name.endswith('.c'): compile_sources.append(target)
    for name in ('atlas_host.c', 'atlas_host.h', 'atlas_host_layout.h', 'atlas_numeric.h', 'atlas_resnet_graph_math.c'):
        target = copy(args.runtime/'runtime'/name, source_dir/name)
        if name == 'atlas_host.c': compile_sources.append(target)
    copy_header_closure([work/'main.c', *[work/name for name in copied
        if name.startswith('source/') and Path(name).suffix in ('.c', '.h')]], source_dir,
                        (args.attention, args.merlin, Path(__file__).absolute().parents[1]/'runtime'), copy, read)
    support_dir = safe(work/'support')
    copy(args.support/'rv64_support.c', support_dir/'rv64_support.c')
    copy(args.support/'expf_bf16_values.inc', support_dir/'expf_bf16_values.inc')
    compile_sources.append(support_dir/'rv64_support.c')
    # Enumerate only the explicitly admitted small freestanding include tree;
    # reject restricted names before probing any child or following a symlink.
    def copy_headers(source, target):
        source = safe(source)
        for entry in sorted(source.iterdir(), key=lambda path: path.name):
            entry = safe(entry)
            if entry.is_dir(): copy_headers(entry, target/entry.name)
            elif entry.is_file(): copy(entry, target/entry.name)
            else: raise ValueError('Unsupported support input kind')
    copy_headers(args.support/'include', support_dir/'include')
    start = copy(args.runtime/'runtime/host_templates/smol_start.S', work/'start.S')
    linker_template = read(args.runtime/'runtime/host_templates/smol_link.ld').decode()
    safe(work/'link.ld').write_text(full_link(linker_template, dram_bytes, args.arena_base,
        rodata_after_arena=args.rodata_after_arena, dram_end=args.dram_base+args.dram_bytes))
    for name in ('main.c', 'link.ld'): read(work/name)
    compile_sources.append(start)
    # Retain each original asset exactly once in the package. Original stage
    # checksums authenticate these bytes; no upstream producer is rerun.
    for name, expected in bundle.sums.items():
        merge_source = args.assets/name
        pin(merge_source, expected)
        if name == 'kernels.json': continue
        raw = bundle.read(name)
        target = safe(out/name)
        if target.exists():
            if target.read_bytes() != raw: raise ValueError('Conflicting retained stage asset')
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
        products[name] = expected
    selected_raw = (json.dumps(manifest, sort_keys=True, indent=2)+'\n').encode()
    (out/'kernels.json').write_bytes(selected_raw)
    products['kernels.json'] = sha(selected_raw)
    replay_raw = read(args.replay)
    (out/'host/reference-replay.json').write_bytes(replay_raw)
    products['host/reference-replay.json'] = sha(replay_raw)
    trajectory_raw = read(replay['trajectory']['path'], replay['trajectory']['sha256'])
    (out/'host/reference-trajectory.npz').write_bytes(trajectory_raw)
    products['host/reference-trajectory.npz'] = sha(trajectory_raw)
    assembly = ['.section .rodata.smol_assets,"a",@progbits']
    for relative, labels in sorted(payload_labels.items()):
        raw = safe(out/relative).read_bytes()
        if sha(raw) != products[relative]: raise ValueError('Changed generated payload')
        read(out/relative)
        assembly.append('.balign 64')
        for label in labels: assembly.extend(('.globl '+label, '.type '+label+',@object', label+':'))
        # Paths are restricted to our content-addressed inventory, with no
        # quotes, newline escapes or assembler substitutions admitted.
        path = str(safe(out/relative))
        if any(char in path for char in ('"', '\n', '\\')): raise ValueError('Unsafe incbin path')
        assembly.append('.incbin "'+path+'"')
        assembly.extend('.size '+label+',.-'+label for label in labels)
    safe(work/'assets.S').write_text('\n'.join(assembly)+'\n')
    read(work/'assets.S')
    compile_sources.append(work/'assets.S')
    flags = ['--target=riscv64-unknown-elf', '-march=rv64gc', '-mabi=lp64d', '-mcmodel=medany',
        '-std=c11', '-O2', '-Wall', '-Wextra', '-Werror', '-ffp-contract=off', '-fno-fast-math',
        '-ffreestanding', '-fno-stack-protector', '-fno-pic', '-fno-vectorize',
        '-fno-slp-vectorize', '-ffunction-sections', '-fdata-sections',
        '-I'+str(support_dir/'include'), '-I'+str(source_dir)]
    commands, objects = [], []
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    env.pop('PYTHONPATH', None)
    def invoke(command):
        commands.append(command)
        result = subprocess.run(command, cwd=work, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        with safe(work/'build.log').open('ab') as stream: stream.write(result.stdout)
        if result.returncode: raise RuntimeError('Full host compilation failed: '+result.stdout.decode(errors='replace'))
    for index, source in enumerate(compile_sources):
        read(source)
        obj = safe(work/f'part-{index:02d}.o')
        invoke([str(args.compiler), *flags, '-c', str(source), '-o', str(obj)])
        read(obj)
        objects.append(str(obj))
    elf_path = safe(out/'host/host.elf')
    invoke([str(args.linker), '-flavor', 'gnu', '--threads=1', '-m', 'elf64lriscv',
        '--gc-sections', '-T', str(work/'link.ld'), '-Map='+str(work/'host.map'),
        '-o', str(elf_path), *objects])
    elf_raw = read(elf_path)
    image = Image(elf_raw)
    entry, sections, _ = elf_sections(elf_raw)
    footprint = audit(out/'host', args.dram_base, args.dram_bytes)
    contract_module.audit_image(image, sections, dram_bytes, args.physical)
    for label, raw in symbols.items(): contract_module.blob(image, label, raw)
    value = dict(schema='atlas.smol_full_contract.v1', cycle_cap=contract_module.CYCLE_CAP,
        kernel_count=len(manifest['kernels']), projection_calls=1421, kernels=manifest['kernels'],
        reference_buffers=references, host_parameters=parameters, dram_bytes=dram_bytes,
        panel_symbol='smol_full_panels', call_symbol='smol_full_calls', context_symbol='smol_full_raw',
        graph_state_symbol='smol_full_graph_state', vision_state_symbol='smol_full_vision_states',
        mailbox_symbol='atlas_workload_mailbox', source_reference_sha256=sha(replay_raw),
        source_reference='host/reference-replay.json', source_weights=replay['source']['weights'],
        captured_kernels='host/captured-kernels.json', captured_kernels_sha256=products['host/captured-kernels.json'],
        placement=args.physical, program_policy=manifest['program_policy'])
    contract_name = 'host/smol-full-contract.json'
    contract_raw = (json.dumps(value, sort_keys=True, indent=2)+'\n').encode()
    (out/contract_name).write_bytes(contract_raw)
    products[contract_name] = sha(contract_raw)
    (out/'contract.json').write_bytes(contract_raw)
    products['contract.json'] = sha(contract_raw)
    for name in ('smol_full_driver.c','smol_full_driver.h','smol_policy_graph.c','smol_policy_graph.h',
                 'smol_vision_graph.c','smol_vision_graph.h'):
        raw = read(source_dir/name)
        (out/'host'/name).write_bytes(raw)
        products['host/'+name] = sha(raw)
    for name, expected in tuple(pins.items()): pin(name, expected)
    for name, expected in products.items(): pin(out/name, expected)
    receipt = dict(schema='atlas.smol_full_host_build.v1', status='BUILT_AUDITED_ONLY',
        actual_returncode=0, source_bookends_verified=True, ELF_executed=False,
        elf=dict(path='host/host.elf', sha256=sha(elf_raw), bytes=len(elf_raw)),
        contract_sha256=sha(contract_raw), source_sha256=dict(pins), compilation_inputs=copied,
        commands=commands, command=[sys.executable,*sys.argv], python_version=sys.version,
        footprint=footprint, compiler_sha256=pins[str(args.compiler)],
        linker_sha256=pins[str(args.linker)], kernel_count=len(manifest['kernels']), projection_calls=1421,
        placement=args.physical, source_relocations=args.relocate, native_table_provider=args.table_worker, uart=args.uart, result_name='smol-full',
        result_progress_launches=progress_launches(manifest))
    receipt.update(segment_mode=True, segment_result_name='smol-segment',
        capture_sha256=args.phase_data['reference_sha256'])
    build_raw = (json.dumps(receipt, sort_keys=True, indent=2)+'\n').encode()
    (out/'host/build.json').write_bytes(build_raw)
    products.update({'host/build.json':sha(build_raw), 'host/host.elf':sha(elf_raw)})
    provenance = dict(schema='atlas.full_bundle.v1', workload='smol-full', kernel_count=len(manifest['kernels']),
        kernels=manifest['kernels'], contract=contract_name, contract_sha256=sha(contract_raw),
        kernel_assets_sha256=products['kernels.json'], placement=args.physical,
        program_policy=manifest['program_policy'])
    raw = (json.dumps(provenance, sort_keys=True, indent=2)+'\n').encode()
    (out/'provenance.json').write_bytes(raw)
    products['provenance.json'] = sha(raw)
    from smol_full_driver_emit import write_pending_chain
    write_pending_chain(out, args, image, value, sha(elf_raw), sha(contract_raw), products)
    (out/'SHA256SUMS').write_text(''.join(digest+'  '+name+'\n' for name,digest in sorted(products.items())))
    # Admit the concrete final ELF through the same model-specific contract the
    # continuous runner will use, before publishing build-only success.
    from full_bundle import FullBundle
    final = FullBundle(out, 'smol-full')
    contract_module.validate_bundle(final, image, assemble=args.assemble)
    final.verify_bookends()
    if args.target == 'firesim':
        package_firesim(out/'host', 'atlas-smol-full-policy', args.dram_base, args.dram_bytes, uart=args.uart)
    for name, expected in tuple(pins.items()):
        pin(name, expected)
    final.verify_bookends()
    print(json.dumps(dict(status='BUILT_AUDITED_ONLY', elf_sha256=sha(elf_raw), footprint=footprint), sort_keys=True))
    return receipt


if __name__ == '__main__':
    main()
