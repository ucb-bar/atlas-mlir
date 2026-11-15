"""Default-budget sizing of counted images, separate from native publication.

An allocated untimed plan can reject an image: timing only adds words with
the same CFG ownership. Passing that lower bound never admits an image. Only
the fully checked physical program supplies the final per-launch bound.
"""
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from math import prod
import os
from pathlib import Path
import tempfile

from merlin.semantic_compiler.buffer_allocation import current_allocation_wall_budget_ms
from merlin.semantic_compiler.codegen import current_codegen_options
from merlin.semantic_compiler.search import SearchLimits
from merlin.semantic_compiler.target_binding import NativeCompilationError, verify_native_publication

from .binding import (AtlasNativeBinding, DRAM_BASE, TILE_BYTES, NATIVE_CHECKS,
                      NATIVE_PIPELINES, RTL_REVISION, _native_pipeline, _run, _tile_io_base, dram_slots)
from .restoration_budget import instruction_bound, fallback_tiles
from .limits import functional_instruction_limit

# Process-local decisions avoid repeating expensive planning for the same
# exact source, tools, profile, and limits, including failed attempts.
# A retained failure wins over late successes from already in-flight probes.
# Concurrent successes may retain either checked result; callers keep their own.
_CACHE = {}


def _json(value):
    return (json.dumps(value, sort_keys=True, indent=2) + '\n').encode()


def _digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


@dataclass(frozen=True)
class Sizing:
    directory: Path | None
    metadata: bytes
    files: tuple[tuple[str, str], ...] = ()

    def record(self):
        if self.directory is not None:
            if self.directory.is_symlink():
                raise ValueError('checked sizing cache directory changed')
            for name, digest in self.files:
                path = self.directory / name
                parents = [self.directory/Path(*Path(name).parts[:n])
                           for n in range(1,len(Path(name).parts))]
                if (any(parent.is_symlink() for parent in parents) or path.is_symlink()
                        or not path.is_file() or _digest(path) != digest):
                    raise ValueError('checked sizing cache changed: ' + name)
        return json.loads(self.metadata)

    def read(self, name):
        self.record()
        if self.directory is None or name not in dict(self.files):
            raise ValueError('sizing cache has no checked member: ' + name)
        payload = (self.directory / name).read_bytes()
        if sha256(payload).hexdigest() != dict(self.files)[name]:
            raise ValueError('checked sizing cache changed while reading: '+name)
        return payload


@dataclass(frozen=True)
class SizingFailure:
    status: str
    reason: str
    directory: Path


def untimed_bound(source, report):
    """Use the existing spill-aware CFG proof, without inventing word costs."""
    if (report.get('schema') != 'atlas.virtual_resource_plan.v1'
            or report.get('planning_succeeded') is not True
            or report.get('machine_artifact_qualified') is not False
            or report.get('planned_helper_control') is not False):
        raise ValueError('resource plan is not a supported unqualified CFG plan')
    blocks = report.get('cfg_blocks')
    if not isinstance(blocks, list) or not blocks:
        raise ValueError('resource plan has no CFG blocks')
    words = []
    for index, block in enumerate(blocks):
        count = block.get('planned_words')
        if (type(block.get('id')) is not int or block['id'] != index or type(count) is not int or count < 0
                or len(words) + count > 32768):
            raise ValueError('resource plan has invalid word ownership')
        start = len(words)
        words.extend({'word_index': start + i,
                      'fields': {'atlas.virtual_cfg_block': index}} for i in range(count))
    # These count-only rows are never an executable image or native evidence.
    bound = instruction_bound(source, report['cfg_contract'], {'instructions': words})
    bound['exceeds_physical_instruction_limit'] = not bound.pop('admitted')
    bound.update(kind='untimed_rejection_lower_bound', machine_artifact_qualified=False)
    return bound


def _identity(plan, virtual, target_source, tools, limits, operand_reuse):
    from .exponent_page_cache import enabled as exponent_page_cache_enabled
    options = current_codegen_options()
    pipeline = _native_pipeline()
    arguments = list(NATIVE_PIPELINES[pipeline])
    if options.program_size_limit_bytes is not None:
        arguments = [arg+'=program-size-limit-bytes='+str(options.program_size_limit_bytes) for arg in arguments]
    capture = os.environ.get('ATLAS_CAPTURE_POSTSPILL') == '1'
    if capture:
        arguments = ['--mlir-disable-threading','--mlir-print-ir-before=lower-atlas-virtual-to-machine',
                     '--mlir-print-ir-module-scope',*arguments]
    return dict(source_request_digest=plan.source.digest(), target_identity=plan.source.target_identity,
        virtual_sha256=sha256(virtual.encode()).hexdigest(), pipeline=pipeline,
        pipeline_arguments=list(NATIVE_PIPELINES[pipeline]), codegen=options.identity_record(),
        effective_pipeline_arguments=arguments,
        selected_io_memory_bytes=dram_slots()*TILE_BYTES,
        allocation_timeout_ms=current_allocation_wall_budget_ms(), search_limits=asdict(limits),
        compiler_proof_limits='defaults', tool_timeout_seconds=30,
        physical_instruction_limit=functional_instruction_limit(),
        profile_sha256=AtlasNativeBinding.profile().digest(),
        target_inventory_sha256=_digest(target_source/'docs/selected-variant-inventory.json'),
        tool_sha256={name: _digest(tools/name) for name in ('atlas-opt','atlas-emit')},
        capture_postspill=capture,
        **({'exponent_page_cache': True} if exponent_page_cache_enabled() else {}),
        **({"operand_reuse_requested": operand_reuse} if operand_reuse != "ordered" else {}))


def _retain_checked(directory, record):
    names = ['source-virtual.mlir','resource.json','resource-virtual.mlir','resource-stderr.log','untimed-bound.json','checked-attempt.json',
             'checked/compilation-pipeline.json','checked/atlas-machine.mlir',
             'checked/physical-program.json','checked/physical-bound.json',
             'checked/mandatory-spilled-virtual.mlir']
    files = tuple((name,_digest(directory/name)) for name in names if (directory/name).is_file())
    record['sizing_cache'] = dict(directory=str(directory), files=dict(files),
                                native_publication=False)
    payload = _json(record)
    (directory/'sizing.json').write_bytes(payload)
    files += (('sizing.json',sha256(payload).hexdigest()),)
    return Sizing(directory,payload,files)


def size_counted(plan, *, target_source=None, limits=None, operand_reuse=None):
    """Return a retained, checked route decision at the current exact settings."""
    from .quantized_operand_reuse import reuse_policy

    policy = reuse_policy() if operand_reuse is None else operand_reuse
    result = _size_counted(plan, target_source=target_source, limits=limits, operand_reuse=policy)
    # Retry only a proved capacity refusal. Compiler errors and failed proof
    # checks must remain visible. Each candidate keeps its own immutable sizing
    # evidence; the caller publishes the actual selected policy and geometry.
    if policy in ('panel-cache', 'batch-panel-cache') and result.record().get('reason') in {
            'encoded_instruction_memory', 'untimed_instruction_bound', 'physical_instruction_bound'}:
        if policy == 'batch-panel-cache':
            return size_counted(plan, target_source=target_source, limits=limits, operand_reuse='panel-cache')
        return _size_counted(plan, target_source=target_source, limits=limits, operand_reuse='paired-cache')
    return result


def _size_counted(plan, *, target_source=None, limits=None, operand_reuse=None):
    from .exponent_page_cache import enabled as exponent_page_cache_enabled
    from .quantized_loop import _execution_geometry, execution_layout, virtual_source
    from .region_composition import compile_virtual_with_receipt

    target_source = Path(target_source or os.environ.get('ATLAS_ROOT', Path(__file__).resolve().parents[2]))
    limits = SearchLimits() if limits is None else limits
    from .quantized_operand_reuse import reuse_policy
    operand_reuse = reuse_policy() if operand_reuse is None else operand_reuse
    geometry, end = _execution_geometry(plan)
    window = dram_slots()*TILE_BYTES
    io_base = _tile_io_base.get()
    base = dict(schema='atlas.quantized_counted_sizing.v1', source_request_digest=plan.source.digest(),
                target_identity=plan.source.target_identity, selected_io_memory_bytes=window,
                counted_frame_bytes=end-io_base, physical_instruction_limit=functional_instruction_limit(),
                native_publication=False)
    if exponent_page_cache_enabled():
        base['exponent_page_cache'] = True
    if io_base != DRAM_BASE:
        base['io_memory_base'] = io_base
    if operand_reuse != 'ordered':
        base['operand_reuse_requested'] = operand_reuse
    if end > io_base+window:
        return Sizing(None,_json(dict(base,route='k32_tiles',reason='selected_dram_window')))
    layout = execution_layout(plan, operand_reuse=operand_reuse)
    virtual = virtual_source(plan,layout)
    tools = Path(os.environ.get('ATLAS_OOT_BIN_DIR',target_source/'build/bin'))
    identity = _identity(plan,virtual,target_source,tools,limits,operand_reuse)
    key = sha256(_json(identity)).hexdigest()
    if key in _CACHE:
        result = _CACHE[key]
        if isinstance(result, SizingFailure):
            raise NativeCompilationError(result.status,
                result.reason+'; retained identical sizing failure: '+str(result.directory))
        if result.record()['identity'] != identity:
            raise ValueError('checked sizing identity changed')
        return result
    directory = Path(tempfile.mkdtemp(prefix='atlas-counted-sizing-'))
    (directory/'source-virtual.mlir').write_text(virtual)
    report_path = directory/'resource.json'
    prefix = (['--spill-atlas-virtual','--verify-atlas-virtual-stream',
               '--lower-atlas-pack-relayout','--spill-atlas-virtual','--verify-atlas-virtual-stream']
              if identity['pipeline'] == 'mandatory' else
              ['--atlas-compile-verify','--atlas-compile-schedule','--atlas-compile-spill'])
    options = 'resource-planning-report='+str(report_path)
    if current_codegen_options().program_size_limit_bytes is not None:
        options += ' program-size-limit-bytes='+str(current_codegen_options().program_size_limit_bytes)
    capture_options = (['--mlir-disable-threading',
        '--mlir-print-ir-before=lower-atlas-virtual-to-machine','--mlir-print-ir-module-scope']
        if identity['capture_postspill'] else [])
    command = [str(tools/'atlas-opt'),*capture_options,*prefix,'--lower-atlas-virtual-to-machine='+options,'-']
    diagnostics = []
    try:
        prepared = _run(command,input_text=virtual,diagnostics=diagnostics)
    except NativeCompilationError as error:
        # The report is written even when planning cannot fit IMEM. Only the
        # explicit word-count inequality licenses this construction refusal.
        if error.status != 'compile_error' or not report_path.is_file():
            raise
        report = json.loads(report_path.read_text())
        words = report.get('untimed_words',{})
        count,limit = words.get('words'),words.get('limit')
        if (report.get('schema') != 'atlas.virtual_resource_plan.v1'
                or report.get('planning_succeeded') is not False
                or report.get('machine_artifact_qualified') is not False
                or type(count) is not int or type(limit) is not int
                or not 0 < limit <= 32768 or count <= limit):
            raise
        # Static IMEM capacity needs no loop/helper interpretation. This is
        # not a claim about the one-million dynamic instruction bound.
        if _identity(plan,virtual,target_source,tools,limits,operand_reuse) != identity:
            raise ValueError('resource planning tools or settings changed')
        (directory/'resource-stderr.log').write_text(''.join(diagnostics))
        result = _retain_checked(directory,dict(base,identity=identity,resource_command=command,
            route='k32_tiles',reason='encoded_instruction_memory',
            planned_words=count,program_word_limit=limit))
        _CACHE.setdefault(key, result)
        return result
    (directory/'resource-stderr.log').write_text(''.join(diagnostics))
    (directory/'resource-virtual.mlir').write_text(prepared)
    report = json.loads(report_path.read_text())
    bound = untimed_bound(virtual,report)
    (directory/'untimed-bound.json').write_bytes(_json(bound))
    record = dict(base, identity=identity, resource_command=command,
                  resource_prepared_virtual_sha256=sha256(prepared.encode()).hexdigest(),
                  untimed_instruction_bound=bound['instruction_bound'])
    if bound['exceeds_physical_instruction_limit']:
        if _identity(plan,virtual,target_source,tools,limits,operand_reuse) != identity:
            raise ValueError('resource planning tools or settings changed')
        result = _retain_checked(directory,dict(record,route='k32_tiles',reason='untimed_instruction_bound'))
        _CACHE.setdefault(key, result)
        return result
    checked = directory/'checked'
    checked.mkdir()
    (directory/'checked-attempt.json').write_bytes(_json(dict(record,status='STARTED',
        argv=[str(tools/'atlas-opt'),*identity['effective_pipeline_arguments'],'-'])))
    try:
        machine = compile_virtual_with_receipt(virtual,tools=tools,pipeline=identity['pipeline'],destination=checked)
    except NativeCompilationError as error:
        (directory/'checked-failure.json').write_bytes(_json(dict(record,status='FAILED',
            error_status=error.status,error_reason=error.reason)))
        _CACHE[key] = SizingFailure(error.status,error.reason,directory)
        raise
    (checked/'atlas-machine.mlir').write_text(machine)
    if report['cfg_contract'] not in machine:
        raise NativeCompilationError('compile_error','checked lowering changed the resource plan CFG identity')
    covered = (identity['pipeline'] == 'mandatory'
               and identity['pipeline_arguments'] == ['--atlas-compile-unscheduled'])
    record['checks_covered_by_immutable_mandatory_result'] = covered
    if covered:
        record['mandatory_check_coverage'] = dict(verifier='verifyAtlasArtifact(llvmBlock=true)',
            contracts=['cfg','dma','mxu','tile','source_memory','buffer'],
            checks=['strict_timing','CFG_DMA','machine_encoding','alignment','program_size','LLVM_targets'],
            metadata='shared original timing metadata stamp; checked lowering only, not native publication')
    record['checked_commands'] = []
    if not covered:
        command = [str(tools/'atlas-opt'),*NATIVE_CHECKS,'--verify-atlas-timing','-']
        _run(command,input_text=machine)
        record['checked_commands'].append(command)
    command = [str(tools/'atlas-emit'),'--program-json','-']
    program = json.loads(_run(command,input_text=machine))
    record['checked_commands'].append(command)
    if program.get('selected_rtl_revision') != RTL_REVISION:
        raise NativeCompilationError('unqualified_target','sizing emitter revision differs')
    word_limit = report.get('untimed_words',{}).get('limit')
    if type(word_limit) is not int or not 0 < word_limit <= 32768:
        raise ValueError('resource plan lacks the selected instruction memory limit')
    if len(program['instructions']) > word_limit:
        if _identity(plan,virtual,target_source,tools,limits,operand_reuse) != identity:
            raise ValueError('sizing tools or settings changed during compilation')
        (checked/'atlas-machine.mlir').write_text(machine)
        (checked/'physical-program.json').write_bytes(_json(program))
        result = _retain_checked(directory,dict(record,route='k32_tiles',reason='encoded_instruction_memory',
            encoded_words=len(program['instructions']),program_word_limit=word_limit))
        _CACHE.setdefault(key, result)
        return result
    final = instruction_bound(virtual,machine,program)
    if final['instruction_bound'] < bound['instruction_bound']:
        raise NativeCompilationError('compile_error','resource plan is not a lower bound of the checked pipeline')
    if _identity(plan,virtual,target_source,tools,limits,operand_reuse) != identity:
        raise ValueError('sizing tools or settings changed during compilation')
    (checked/'atlas-machine.mlir').write_text(machine)
    (checked/'physical-program.json').write_bytes(_json(program))
    (checked/'physical-bound.json').write_bytes(_json(final))
    record.update(physical_instruction_bound=final['instruction_bound'],
                  route='counted' if final['admitted'] else 'k32_tiles',
                  reason='physical_bound_admitted' if final['admitted'] else 'physical_instruction_bound')
    result = _retain_checked(directory,record)
    _CACHE.setdefault(key, result)
    return result


def _evidence_names(reason, record):
    if reason == 'selected_dram_window':
        return ()
    names = ['source-virtual.mlir','resource.json']
    if reason in {'untimed_instruction_bound','physical_instruction_bound'}:
        names += ['resource-virtual.mlir','untimed-bound.json']
    if reason == 'physical_instruction_bound' or (reason == 'encoded_instruction_memory' and 'encoded_words' in record):
        names += ['checked/compilation-pipeline.json','checked/atlas-machine.mlir',
                  'checked/physical-program.json']
    if reason == 'physical_instruction_bound':
        names += ['checked/physical-bound.json']
    if reason not in {'untimed_instruction_bound','physical_instruction_bound','encoded_instruction_memory'}:
        raise ValueError('unsupported quantized construction evidence')
    return tuple(names)


def _check_construction_evidence(plan, compiled, sizing, route, role, manifest):
    """Rederive a refusal from source-bound evidence in a real native leaf."""
    from .quantized_loop import execution_layout, virtual_source

    names = _evidence_names(route['reason'],sizing)
    inventory = route.get('construction_evidence')
    if not isinstance(inventory,dict) or set(inventory) != set(names):
        raise ValueError('quantized construction evidence inventory differs')
    payloads = {}
    for name in names:
        relative = 'quantized-selection-evidence/'+name
        digest = manifest.get('stage_artifacts',{}).get(relative)
        if inventory[name] != dict(path=role+'/'+relative,sha256=digest) or not isinstance(digest,str):
            raise ValueError('quantized construction evidence is not native-bound')
        # The caller has verified all native stage paths and their hashes.
        payloads[name] = (compiled/role/relative).read_bytes()
        if sha256(payloads[name]).hexdigest() != digest:
            raise ValueError('quantized construction evidence changed while reading')
    if not names:
        return
    from .exponent_page_cache import recorded_enabled
    layout = execution_layout(plan, operand_reuse=sizing.get('operand_reuse_requested', 'ordered'),
                              exponent_page_cache=recorded_enabled(sizing))
    virtual = virtual_source(plan,layout)
    identity = sizing.get('identity',{})
    if (payloads['source-virtual.mlir'] != virtual.encode()
            or identity.get('virtual_sha256') != sha256(virtual.encode()).hexdigest()
            or identity.get('source_request_digest') != plan.source.digest()
            or identity.get('target_identity') != plan.source.target_identity
            or identity.get('codegen') != current_codegen_options().identity_record()
            or identity.get('profile_sha256') != AtlasNativeBinding.profile().digest()
            or identity.get('allocation_timeout_ms') != current_allocation_wall_budget_ms()
            or identity.get('search_limits') != asdict(SearchLimits())
            or identity.get('compiler_proof_limits') != 'defaults'
            or identity.get('tool_timeout_seconds') != 30
            or identity.get('physical_instruction_limit') != functional_instruction_limit()
            or any(identity.get('tool_sha256',{}).get(name) != manifest.get(name.replace('-','_')+'_sha256')
                   for name in ('atlas-opt','atlas-emit'))):
        raise ValueError('quantized construction evidence source or settings differ')
    report = json.loads(payloads['resource.json'])
    if (report.get('schema') != 'atlas.virtual_resource_plan.v1'
            or report.get('machine_artifact_qualified') is not False):
        raise ValueError('quantized construction report is not unqualified planning evidence')
    reason = route['reason']
    if reason in {'untimed_instruction_bound','physical_instruction_bound'}:
        untimed = untimed_bound(virtual,report)
        if (json.loads(payloads['untimed-bound.json']) != untimed
                or sizing.get('untimed_instruction_bound') != untimed['instruction_bound']
                or sizing.get('resource_prepared_virtual_sha256') != sha256(payloads['resource-virtual.mlir']).hexdigest()):
            raise ValueError('quantized construction untimed proof differs')
        if reason == 'untimed_instruction_bound':
            if not untimed['exceeds_physical_instruction_limit']:
                raise ValueError('quantized construction untimed proof does not refuse the image')
            return
    if reason == 'encoded_instruction_memory' and 'encoded_words' not in sizing:
        words = report.get('untimed_words',{})
        count,limit = words.get('words'),words.get('limit')
        if (report.get('planning_succeeded') is not False
                or type(count) is not int or type(limit) is not int
                or not 0 < limit <= 32768 or count <= limit
                or sizing.get('planned_words') != count or sizing.get('program_word_limit') != limit):
            raise ValueError('quantized construction instruction-memory proof differs')
        return
    machine = payloads['checked/atlas-machine.mlir'].decode()
    program = json.loads(payloads['checked/physical-program.json'])
    receipt = json.loads(payloads['checked/compilation-pipeline.json'])
    if (receipt.get('input_sha256') != identity['virtual_sha256']
            or receipt.get('output_sha256') != sha256(machine.encode()).hexdigest()
            or receipt.get('tool_sha256') != identity['tool_sha256']
            or receipt.get('wall_limit_s') != 30 or receipt.get('verifier_limits') != 'compiler defaults'):
        raise ValueError('quantized construction checked lowering receipt differs')
    if reason == 'physical_instruction_bound':
        actual = instruction_bound(virtual,machine,program)
        if (json.loads(payloads['checked/physical-bound.json']) != actual or actual['admitted']
                or sizing.get('physical_instruction_bound') != actual['instruction_bound']):
            raise ValueError('quantized construction physical proof differs')
    else:
        count,limit = len(program['instructions']),report.get('untimed_words',{}).get('limit')
        if (type(limit) is not int or not 0 < limit <= 32768 or count <= limit
                or sizing.get('encoded_words') != count or sizing.get('program_word_limit') != limit):
            raise ValueError('quantized construction encoded instruction-memory proof differs')


def publish_route(plan, compiled, sizing, *, target_source=None):
    """Bind the actual selected publication, rather than infer it at export."""
    from .already_quantized import _native_bodies

    record = sizing.record()
    target_source = Path(target_source or os.environ.get('ATLAS_ROOT',Path(__file__).resolve().parents[2]))
    tools = Path(os.environ.get('ATLAS_OOT_BIN_DIR',target_source/'build/bin'))
    tool_hashes = {name:_digest(tools/name) for name in ('atlas-opt','atlas-emit')}
    identity = record.get('identity')
    if identity is not None and (identity['tool_sha256'] != tool_hashes
            or identity['profile_sha256'] != AtlasNativeBinding.profile().digest()
            or identity['codegen'] != current_codegen_options().identity_record()
            or identity['allocation_timeout_ms'] != current_allocation_wall_budget_ms()):
        raise ValueError('sizing context changed before native publication')
    programs = []
    evidence = {}
    if record['route'] == 'counted':
        manifest = json.loads((compiled/'manifest.json').read_text())
        if manifest.get('io_memory_bytes') != dram_slots()*TILE_BYTES:
            raise ValueError('counted publication selected I/O profile differs')
        if any(manifest.get(name.replace('-','_')+'_sha256') != digest for name,digest in tool_hashes.items()):
            raise ValueError('counted publication tools changed')
        verify_native_publication(compiled,manifest,engine='merlin_native',
            request_digest=plan.source.digest(),target_identity=plan.source.target_identity)
        actual = instruction_bound((compiled/'atlas-virtual.mlir').read_text(),
            (compiled/'atlas-machine.mlir').read_text(),
            json.loads((compiled/'physical-program.json').read_text()))
        if actual['instruction_bound'] != record['physical_instruction_bound'] or not actual['admitted']:
            raise ValueError('counted publication physical instruction proof differs')
        programs.append(dict(path='.',binary_sha256=manifest['binary_sha256']))
    else:
        for index,(role,body) in enumerate(_native_bodies(plan)):
            path = compiled/role/'manifest.json'
            manifest = json.loads(path.read_text())
            if manifest.get('io_memory_bytes') != dram_slots()*TILE_BYTES:
                raise ValueError('tiled publication selected I/O profile differs')
            if any(manifest.get(name.replace('-','_')+'_sha256') != digest for name,digest in tool_hashes.items()):
                raise ValueError('tiled publication tools changed')
            verify_native_publication(compiled/role,manifest,engine='merlin_native',
                request_digest=body.digest(),target_identity=plan.source.target_identity)
            if index == 0:
                for name in _evidence_names(record['reason'],record):
                    relative = 'quantized-selection-evidence/'+name
                    payload = sizing.read(name)
                    member = compiled/role/relative
                    member.parent.mkdir(parents=True,exist_ok=True)
                    member.write_bytes(payload)
                    digest = sha256(payload).hexdigest()
                    manifest['stage_artifacts'][relative] = digest
                    evidence[name] = dict(path=role+'/'+relative,sha256=digest)
                if evidence:
                    path.write_bytes(_json(manifest))
                    verify_native_publication(compiled/role,manifest,engine='merlin_native',
                        request_digest=body.digest(),target_identity=plan.source.target_identity)
            programs.append(dict(path=role,manifest_sha256=_digest(path),
                                 binary_sha256=manifest['binary_sha256']))
    payload = _json(publication_record(record))
    route = dict(schema='atlas.quantized_selected_route.v1',route=record['route'],reason=record['reason'],
        source_request_digest=plan.source.digest(),target_identity=plan.source.target_identity,
        selected_io_memory_bytes=dram_slots()*TILE_BYTES,physical_instruction_limit=functional_instruction_limit(),
        sizing_sha256=sha256(payload).hexdigest(),programs=programs,construction_evidence=evidence,
        launches_per_invocation=(1 if record['route']=='counted' else
            prod(plan.batch_shape)*((plan.m+31)//32)*((plan.n+31)//32)*((plan.k+31)//32)),
        restoration_placement='device' if record['route']=='counted' else 'host')
    route_bytes = _json(route)
    (compiled/'quantized-sizing.json').write_bytes(payload)
    (compiled/'quantized-route.json').write_bytes(route_bytes)
    if record['route'] == 'counted':
        manifest['stage_artifacts'].update({'quantized-sizing.json':sha256(payload).hexdigest(),
                                            'quantized-route.json':sha256(route_bytes).hexdigest()})
        (compiled/'manifest.json').write_bytes(_json(manifest))


def publication_record(record):
    """Preserve default sizing bytes; strip cache bookkeeping only on opt-in.

    The operational cache retains and authenticates its exact directory/files.
    Native artifacts carry the same semantic identity, proof values and tool
    arguments, with a relative report output path and no private cache locator.
    """
    from .publication_options import deterministic_receipts_enabled

    result = dict(record)
    if not deterministic_receipts_enabled():
        return result
    cache = result.pop('sizing_cache', None)
    if cache is not None and 'resource_command' in result:
        report = str(Path(cache['directory']) / 'resource.json')
        prefix = '--lower-atlas-virtual-to-machine=resource-planning-report='
        arguments = []
        for argument in result['resource_command']:
            if argument.startswith(prefix):
                path, separator, remaining = argument[len(prefix):].partition(' ')
                if path != report:
                    raise ValueError('sizing resource output differs from private checked cache')
                argument = prefix + 'resource.json' + separator + remaining
            arguments.append(argument)
        result['resource_command'] = arguments
    return result


def selected_route(plan, compiled):
    """Check source/profile and native identities of the published choice."""
    from .already_quantized import _native_bodies
    from .quantized_loop import _execution_geometry

    compiled = Path(compiled)
    if compiled.is_symlink():
        raise ValueError('linked quantized publication root')
    for name in ('quantized-route.json','quantized-sizing.json'):
        path = compiled/name
        if path.is_symlink() or not path.is_file():
            raise ValueError('quantized publication needs authenticated route selection')
    raw = (compiled/'quantized-sizing.json').read_bytes()
    sizing = json.loads(raw)
    route = json.loads((compiled/'quantized-route.json').read_text())
    if (route.get('schema') != 'atlas.quantized_selected_route.v1'
            or sizing.get('schema') != 'atlas.quantized_counted_sizing.v1'
            or route.get('sizing_sha256') != sha256(raw).hexdigest()
            or sizing.get('native_publication') is not False):
        raise ValueError('quantized route selection identity differs')
    for key,expected in dict(source_request_digest=plan.source.digest(),
            target_identity=plan.source.target_identity,selected_io_memory_bytes=dram_slots()*TILE_BYTES,
            physical_instruction_limit=functional_instruction_limit()).items():
        if route.get(key) != expected or sizing.get(key) != expected:
            raise ValueError('quantized route differs from source or selected profile: '+key)
    if (route.get('route') not in {'counted','k32_tiles'} or route['route'] != sizing.get('route')
            or route.get('reason') != sizing.get('reason')):
        raise ValueError('quantized publication has inconsistent route')
    launches = (1 if route['route']=='counted' else
        prod(plan.batch_shape)*((plan.m+31)//32)*((plan.n+31)//32)*((plan.k+31)//32))
    if (type(route.get('launches_per_invocation')) is not int or route['launches_per_invocation'] != launches
            or route.get('restoration_placement') != ('device' if route['route']=='counted' else 'host')):
        raise ValueError('quantized route launch or restoration publication differs')
    _,end = _execution_geometry(plan)
    io_base = _tile_io_base.get()
    if sizing.get('counted_frame_bytes') != end-io_base or sizing.get('io_memory_base', DRAM_BASE) != io_base:
        raise ValueError('quantized publication frame geometry differs')
    if route['route'] == 'counted':
        bound = json.loads((compiled/'restoration-instruction-bound.json').read_text())
        if (route['reason'] != 'physical_bound_admitted' or bound.get('admitted') is not True
                or bound.get('max_instructions') != functional_instruction_limit()
                or type(bound.get('instruction_bound')) is not int
                or not 0 < bound['instruction_bound'] <= functional_instruction_limit()
                or sizing.get('physical_instruction_bound') != bound['instruction_bound']):
            raise ValueError('counted publication lacks its physical bound')
        manifest = json.loads((compiled/'manifest.json').read_text())
        if manifest.get('io_memory_bytes') != dram_slots()*TILE_BYTES:
            raise ValueError('counted publication selected I/O profile differs')
        verify_native_publication(compiled,manifest,engine='merlin_native',
            request_digest=plan.source.digest(),target_identity=plan.source.target_identity)
        actual = instruction_bound((compiled/'atlas-virtual.mlir').read_text(),
            (compiled/'atlas-machine.mlir').read_text(),
            json.loads((compiled/'physical-program.json').read_text()), schema=bound['schema'])
        actual['scalar_fallback_tile_budget'] = fallback_tiles(plan)
        from .quantized_loop import execution_layout, virtual_source
        execution = json.loads((compiled/'execution_plan.json').read_text())
        from .exponent_page_cache import recorded_enabled
        layout = execution_layout(plan, operand_reuse=sizing.get('operand_reuse_requested', 'ordered'),
                                  exponent_page_cache=recorded_enabled(sizing))
        if execution != layout or (compiled/'atlas-virtual.mlir').read_text() != virtual_source(plan, layout):
            raise ValueError('counted publication execution schedule differs')
        actual['restoration_admission'] = layout['restoration_admission']
        if actual != bound or not actual['admitted']:
            raise ValueError('counted publication physical instruction proof differs')
        expected = [dict(path='.',binary_sha256=manifest['binary_sha256'])]
    else:
        reason = route['reason']
        if reason == 'selected_dram_window':
            admitted = end > io_base+dram_slots()*TILE_BYTES
        elif reason in {'untimed_instruction_bound','physical_instruction_bound'}:
            key = 'untimed_instruction_bound' if reason.startswith('untimed') else 'physical_instruction_bound'
            value = sizing.get(key)
            admitted = type(value) is int and value > functional_instruction_limit()
        elif reason == 'encoded_instruction_memory':
            count,limit = sizing.get('encoded_words',sizing.get('planned_words')),sizing.get('program_word_limit')
            admitted = type(count) is int and type(limit) is int and 0 < limit <= 32768 and count > limit
        else:
            admitted = False
        if not admitted or getattr(plan,'epilogue_precision',None) is not None:
            raise ValueError('tiled publication lacks a supported construction refusal')
        expected = []
        for index,(role,body) in enumerate(_native_bodies(plan)):
            path = compiled/role/'manifest.json'
            if path.is_symlink() or not path.is_file():
                raise ValueError('quantized native leaf is absent or linked')
            manifest = json.loads(path.read_text())
            if manifest.get('io_memory_bytes') != dram_slots()*TILE_BYTES:
                raise ValueError('tiled publication selected I/O profile differs')
            verify_native_publication(compiled/role,manifest,engine='merlin_native',
                request_digest=body.digest(),target_identity=plan.source.target_identity)
            if index == 0:
                _check_construction_evidence(plan,compiled,sizing,route,role,manifest)
            expected.append(dict(path=role,manifest_sha256=_digest(path),
                                 binary_sha256=manifest['binary_sha256']))
    if route.get('programs') != expected:
        raise ValueError('quantized route native identities differ')
    return route['route']
