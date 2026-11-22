"""Checked address binding of a counted image without repeating native compilation.

This is a provider primitive, not an application loop or a deployment loader.
Only immutable operand addresses may change; instruction order and register
allocation are retained. Every binding is checked against the original virtual
source and the compiler's current resident allocation contract.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
import re

from merlin.compiler.build.materialization import canonical_request
from merlin.compiler.semantic.target_binding import verify_native_publication
from .already_quantized import prepare_source
from .binding import TARGET_IDENTITY, NATIVE_PIPELINES
from .quantized_loop import execution_layout, virtual_source
from .options import current_codegen_options


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _words(program):
    return b''.join(row['word_u32'].to_bytes(4, 'little') for row in program['instructions'])


def _constant_sites(program, address):
    """Recognize complete LUI[/ADDI] address materializations, never raw bytes."""
    rows = program['instructions']
    result = []
    for i, row in enumerate(rows):
        f = row['fields']
        if row['operation'] != 'atlas.upper' or f.get('kind') != 'lui':
            continue
        high = (f['immediate'] << 12) & 0xffffffff
        low = 0
        pair = None
        if i+1 < len(rows):
            next_row = rows[i+1]
            nf = next_row['fields']
            if (next_row['operation'] == 'atlas.alu_imm' and nf.get('kind') == 'addi'
                    and nf.get('src') == f['dst'] and nf.get('dst') == f['dst']
                    and 'atlas.virtual_cfg_source' in f
                    and nf.get('atlas.virtual_cfg_source') == f['atlas.virtual_cfg_source']):
                low, pair = nf['immediate'], i+1
        if (high+low) & 0xffffffff == address:
            if pair is None and 'atlas.virtual_cfg_source' in f:
                for later in rows[i+2:]:
                    lf = later['fields']
                    if (later['operation'] == 'atlas.alu_imm' and lf.get('kind') == 'addi'
                            and lf.get('src') == f['dst'] and lf.get('dst') == f['dst']
                            and lf.get('atlas.virtual_cfg_source') == f['atlas.virtual_cfg_source']):
                        raise ValueError('nonadjacent resident LUI/ADDI materialization')
            # Verify the exact selected RV32 encodings before editing any bits.
            if row['word_u32'] != ((f['immediate'] & 0xfffff) << 12 | f['dst'] << 7 | 0x37):
                raise ValueError('resident LUI encoding differs from its physical fields')
            if pair is not None:
                nf = rows[pair]['fields']
                if rows[pair]['word_u32'] != ((low & 0xfff) << 20 | nf['src'] << 15 | nf['dst'] << 7 | 0x13):
                    raise ValueError('resident ADDI encoding differs from its physical fields')
            result.append((i, pair))
    if not result:
        raise ValueError('resident address has no complete scalar materialization')
    return tuple(result)


def _checked_constant_sites(program, addresses):
    """Refuse a resident high part unless every occurrence is accounted for.

    A scheduled LUI/ADDI pair can be separated by independent instructions.
    Finding another adjacent pair does not establish that all uses were found.
    Shared high parts belonging to another resident address are admitted only
    when that address's complete materialization is recognized as well.
    """
    sites = {key: _constant_sites(program, address) for key, address in addresses.items()}
    recognized = {upper for pairs in sites.values() for upper, _ in pairs}
    highs = {((address + 0x800) >> 12) & 0xfffff for address in addresses.values()}
    for index, row in enumerate(program['instructions']):
        fields = row['fields']
        if (row['operation'] == 'atlas.upper' and fields.get('kind') == 'lui'
                and (fields['immediate'] & 0xfffff) in highs):
            if 'atlas.virtual_cfg_source' not in fields:
                raise ValueError('resident LUI lacks virtual source provenance')
            if index not in recognized:
                raise ValueError('unaccounted resident LUI materialization')
    return sites


@dataclass(frozen=True)
class BoundResidentImage:
    binary: bytes
    physical_program: dict
    execution_plan: dict
    receipt: dict


@dataclass(frozen=True)
class ResidentImageC:
    header: str
    source: str
    receipt: dict


class ResidentLoopImage:
    """One authenticated instruction template with compiler-checked trip bindings."""
    def __init__(self, source, compiled, *, target_source=None):
        root = Path(compiled)
        manifest = json.loads((root/'manifest.json').read_bytes())
        if manifest.get('schema') != 'atlas.quantized_loop_compilation.v1':
            raise ValueError('resident loop requires one complete counted image')
        verify_native_publication(root, manifest, engine='merlin_native',
                                  request_digest=source.digest(), target_identity=TARGET_IDENTITY)
        target = Path(target_source) if target_source is not None else Path(__file__).resolve().parents[2]
        inventory = target/'docs/selected-variant-inventory.json'
        if sha256(inventory.read_bytes()).hexdigest() != manifest['target_inventory_sha256']:
            raise ValueError('resident template target inventory differs')
        pipeline = json.loads((root/'compilation-pipeline.json').read_bytes())
        if pipeline.get('argv', [])[1:-1] != list(NATIVE_PIPELINES[current_codegen_options().native_pipeline]):
            raise ValueError('resident template compilation pipeline differs from current options')
        if current_codegen_options().native_pipeline != 'mandatory':
            raise ValueError('resident loop supports only the mandatory compilation pipeline')
        self._options = asdict(current_codegen_options())
        self._options.pop('resident_bindings', None)
        self._source = source
        self._program = json.loads((root/'physical-program.json').read_bytes())
        self._binary = (root/'program.bin').read_bytes()
        if _words(self._program) != self._binary:
            raise ValueError('resident template differs from physical instruction words')
        options = current_codegen_options()
        bound = json.loads((root/'restoration-instruction-bound.json').read_bytes())
        if (bound.get('admitted') is not True
                or type(bound.get('instruction_bound')) is not int
                or bound['instruction_bound'] > (options.execution_instruction_limit or 1000000)
                or options.program_size_limit_bytes is not None
                and len(self._binary) > options.program_size_limit_bytes):
            raise ValueError('resident template exceeds current instruction or image budget')
        self._layout = json.loads((root/'execution_plan.json').read_bytes())
        self._plan = prepare_source(source)
        # Refuse nondefault schedule variants until their exact virtual replay is supported.
        if self._layout != execution_layout(self._plan):
            raise ValueError('resident template execution layout differs from current options')
        self._virtual = virtual_source(self._plan, self._layout)
        if (root/'atlas-virtual.mlir').read_text() != self._virtual:
            raise ValueError('resident template virtual source differs')
        # Content hashes alone cannot prove that a stored machine/word stream
        # was emitted from this source. Replay the trusted virtual construction
        # through the current mandatory tools (or their immutable sizing cache).
        # This runs once per template; bind() never selects or lowers a trip.
        from merlin.compiler.semantic.search.exploration import SearchLimits
        from .quantized_loop_sizing import size_counted
        from .restoration_budget import fallback_tiles, instruction_bound

        sizing = size_counted(self._plan, target_source=target, limits=SearchLimits())
        if sizing.record().get('route') != 'counted':
            raise ValueError('resident template current source replay does not admit a counted image')
        machine = sizing.read('checked/atlas-machine.mlir')
        program = json.loads(sizing.read('checked/physical-program.json'))
        if ((root/'atlas-machine.mlir').read_bytes() != machine
                or self._program != program or self._binary != _words(program)):
            raise ValueError('resident template machine or emitted words differ from trusted source replay')
        actual_bound = instruction_bound(self._virtual, machine.decode(), program)
        if actual_bound != json.loads(sizing.read('checked/physical-bound.json')):
            raise ValueError('resident template checked sizing bound differs from source replay')
        actual_bound['scalar_fallback_tile_budget'] = fallback_tiles(self._plan)
        actual_bound['restoration_admission'] = self._layout['restoration_admission']
        if actual_bound != bound:
            raise ValueError('resident template instruction bound differs from trusted source replay')
        self._resident = {r['node_id']: r for r in self._layout.get('resident_bindings', [])}
        if not self._resident:
            raise ValueError('resident loop requires immutable bindings')
        self._sites = _checked_constant_sites(
            self._program, {key: row['address'] for key, row in self._resident.items()})
        self._publication_sha256 = sha256(_encoded(manifest)).hexdigest()
        self._issued_bindings = {}

    def bind(self, source):
        """Bind current compiler allocations; no select, atlas-opt or atlas-emit call."""
        options = asdict(current_codegen_options())
        options.pop('resident_bindings', None)
        if options != self._options:
            raise ValueError('resident loop codegen options differ')
        # Atlas preparation reads numerical/calibration policy from the source
        # fields and codegen options, never source_identity. Extracted regions
        # carry distinct provenance there; bind authenticates each current
        # source digest below while retaining exact semantic field equality.
        if canonical_request(source, source_identity_policy='provenance').record() != canonical_request(
                self._source, source_identity_policy='provenance').record():
            raise ValueError('resident loop source structure differs')
        names = dict(zip((n.id for n in self._source.nodes), (n.id for n in source.nodes), strict=True))
        plan = prepare_source(source)
        layout = execution_layout(plan)
        resident = {r['node_id']: r for r in layout.get('resident_bindings', [])}
        if set(resident) != {names[key] for key in self._resident}:
            raise ValueError('resident loop binding ports differ')
        old, new = deepcopy(self._layout), deepcopy(layout)
        # The exact geometry, layouts and all numerical/schedule policies must agree.
        for key, original in self._resident.items():
            current = resident[names[key]]
            if {k:v for k,v in original.items() if k not in ('node_id', 'address')} != {
                    k:v for k,v in current.items() if k not in ('node_id', 'address')}:
                raise ValueError('resident loop input layout differs')
            for field in ('input_addresses', 'exponent_addresses'):
                old[field] = [0 if a == original['address'] else a for a in old[field]]
                new[field] = [0 if a == current['address'] else a for a in new[field]]
        old.pop('resident_bindings'); new.pop('resident_bindings')
        if old != new:
            raise ValueError('resident loop nonresident geometry differs')
        virtual = virtual_source(plan, layout)
        def normalized(text, bindings):
            for index, row in enumerate(bindings):
                address = row['address']
                signed = address if address < 1 << 31 else address-(1 << 32)
                text = text.replace(f'arith.constant {signed} : i32', f'arith.constant ADDRESS_{index} : i32')
            return text
        if normalized(self._virtual, self._resident.values()) != normalized(
                virtual, (resident[names[key]] for key in self._resident)):
            raise ValueError('resident loop virtual body differs beyond address bindings')
        program = deepcopy(self._program)
        relocations = []
        for key, sites in self._sites.items():
            address = resident[names[key]]['address']
            high = ((address+0x800) >> 12) & 0xfffff
            low = address-(high << 12)
            low = ((low+0x800) & 0xfff)-0x800
            for upper, lower in sites:
                if lower is None and low:
                    raise ValueError('resident address needs an absent ADDI instruction')
                row = program['instructions'][upper]
                row['fields']['immediate'] = high
                row['word_u32'] = high << 12 | row['fields']['dst'] << 7 | 0x37
                changed = [upper]
                if lower is not None:
                    row = program['instructions'][lower]
                    row['fields']['immediate'] = low
                    row['word_u32'] = (low & 0xfff) << 20 | row['fields']['src'] << 15 | row['fields']['dst'] << 7 | 0x13
                    changed.append(lower)
                for i in changed:
                    program['instructions'][i]['word_hex'] = f"{program['instructions'][i]['word_u32']:08x}"
                relocations.append(dict(port=names[key], address=address, word_indices=changed))
        binary = _words(program)
        receipt = dict(schema='atlas.resident_loop_image_binding.v1',
                       template_sha256=sha256(self._binary).hexdigest(),
                       publication_sha256=self._publication_sha256,
                       options_sha256=sha256(_encoded(self._options)).hexdigest(),
                       binary_sha256=sha256(binary).hexdigest(), source_digest=source.digest(),
                       execution_plan_sha256=sha256(_encoded(layout)).hexdigest(),
                       resident_addresses=[resident[names[key]]['address'] for key in self._resident],
                       relocations=relocations)
        receipt['binding_sha256'] = sha256(_encoded(receipt)).hexdigest()
        self._issued_bindings[receipt['binding_sha256']] = _encoded(receipt)
        return BoundResidentImage(binary, program, layout, receipt)

    def binding_addresses(self, bound):
        """Verify an issued binding before publishing its compact address table.

        The receipt authenticates the checked source, allocation and native
        publication. Accepting a self-consistent hash from an arbitrary caller
        would not establish that compiler validation ever ran.
        """
        receipt = bound.receipt
        if (self._issued_bindings.get(receipt.get('binding_sha256')) != _encoded(receipt)
                or sha256(bound.binary).hexdigest() != receipt['binary_sha256']
                or _words(bound.physical_program) != bound.binary
                or sha256(_encoded(bound.execution_plan)).hexdigest() != receipt['execution_plan_sha256']):
            raise ValueError('resident binding proof or emitted image differs')
        return tuple(receipt['resident_addresses'])

    def verify_binding(self, source, bound):
        """Recheck a binding against the current source and allocation contract."""
        expected = self.bind(source)
        if expected != bound:
            raise ValueError('resident binding differs from current checked allocation')
        return self.binding_addresses(bound)

    def emit_c(self, symbol):
        """Emit one shared template/site table and a checked dynamic loader.

        Per-trip plans publish only ``binding_addresses`` and their binding
        receipts. Scratch belongs to the caller and remains valid through the
        launch. Every call forces a load: pointer equality cannot identify the
        contents of this mutable relocation buffer.
        """
        if not isinstance(symbol, str) or re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', symbol) is None:
            raise ValueError('resident C symbol must be an identifier')
        macro = symbol.upper()
        count = len(self._binary) // 4
        rows = list(self._resident.values())
        sites = [(port, upper, lower) for port, key in enumerate(self._resident)
                 for upper, lower in self._sites[key]]
        # All geometry is authenticated by the original publication and every
        # per-trip allocation has already passed the same compiler contract.
        from .binding import validate_io_window
        from .memory_windows import dma_memory_windows
        staging, staging_bytes = validate_io_window()
        options = asdict(current_codegen_options())
        options.pop('resident_bindings', None)
        if options != self._options:
            raise ValueError('resident loop codegen options differ')
        base = self._options['io_memory_capacity_base']
        end = base + self._options['io_memory_capacity_bytes']
        header = f'''#ifndef {macro}_H
#define {macro}_H
#include "atlas_host.h"
#define {macro}_WORD_COUNT {count}u
#define {macro}_ADDRESS_COUNT {len(rows)}u
atlas_host_result {symbol}_bind(atlas_host *, uint32_t *, size_t, const uint32_t *, size_t);
#endif
'''
        words = ','.join(f'0x{int.from_bytes(self._binary[i:i+4], "little"):08x}u'
                         for i in range(0, len(self._binary), 4))
        site_text = ','.join(f'{{{port}u,{upper}u,{lower if lower is not None else "UINT32_MAX"}}}'
                             for port, upper, lower in sites)
        sizes = ','.join(str(row['bytes'])+'u' for row in rows)
        alignments = ','.join(str(row['alignment'])+'u' for row in rows)
        dma_windows = ' || '.join(f'(a>=UINT64_C({start}) && z<=UINT64_C({stop}))'
                                  for start, stop in dma_memory_windows())
        source = f'''#include "{symbol}.h"
/* One immutable native template; per-trip tables contain addresses only. */
static const uint32_t {symbol}_template[{count}]={{{words}}};
static const struct {{ uint32_t port,upper,lower; }} {symbol}_sites[]={{{site_text}}};
static const uint32_t {symbol}_sizes[]={{{sizes}}};
static const uint32_t {symbol}_alignments[]={{{alignments}}};
static int {symbol}_overlap(uintptr_t a,size_t n,uintptr_t b,size_t m) {{
  return a<b ? b-a<n : a-b<m;
}}
atlas_host_result {symbol}_bind(atlas_host *host,uint32_t *scratch,size_t capacity,
                                      const uint32_t *addresses,size_t address_count) {{
  const size_t bytes={count}u*sizeof(uint32_t), table_bytes={len(rows)}u*sizeof(uint32_t);
  if (!host) return ATLAS_HOST_INVALID;
  if (host->active) return ATLAS_HOST_BUSY;
  if (!scratch || !addresses || capacity<{count}u || address_count!={len(rows)}u ||
      (uintptr_t)scratch%_Alignof(uint32_t) || (uintptr_t)addresses%_Alignof(uint32_t) ||
      (uintptr_t)scratch>UINTPTR_MAX-bytes || (uintptr_t)addresses>UINTPTR_MAX-table_bytes ||
      {symbol}_overlap((uintptr_t)scratch,bytes,(uintptr_t)host,sizeof(*host)) ||
      {symbol}_overlap((uintptr_t)scratch,bytes,(uintptr_t)addresses,table_bytes) ||
      {symbol}_overlap((uintptr_t)addresses,table_bytes,(uintptr_t)host,sizeof(*host)) ||
      !host->mmio.read32 || !host->mmio.write32 || host->map.imem_capacity_bytes/4u<{count}u)
    return ATLAS_HOST_INVALID;
  for (size_t p=0;p<{len(rows)}u;++p) {{
    uint64_t a=addresses[p],z=a+{symbol}_sizes[p];
    if (a%{symbol}_alignments[p] || a<UINT64_C({base}) || z>UINT64_C({end}) ||
        !({dma_windows}) ||
        (a<UINT64_C({staging+staging_bytes}) && UINT64_C({staging})<z)) return ATLAS_HOST_INVALID;
    for (size_t q=0;q<p;++q)
      if (a<(uint64_t)addresses[q]+{symbol}_sizes[q] && addresses[q]<z) return ATLAS_HOST_INVALID;
  }}
  for (size_t i=0;i<sizeof({symbol}_sites)/sizeof({symbol}_sites[0]);++i) {{
    uint32_t p={symbol}_sites[i].port,a=addresses[p];
    if ({symbol}_sites[i].lower==UINT32_MAX && (a&0xfffu)) return ATLAS_HOST_INVALID;
  }}
  for (size_t i=0;i<{count}u;++i) scratch[i]={symbol}_template[i];
  for (size_t i=0;i<sizeof({symbol}_sites)/sizeof({symbol}_sites[0]);++i) {{
    uint32_t a=addresses[{symbol}_sites[i].port];
    uint32_t high=(uint32_t)(((uint64_t)a+0x800u)>>12)&0xfffffu;
    uint32_t upper={symbol}_sites[i].upper,lower={symbol}_sites[i].lower;
    scratch[upper]=(scratch[upper]&0xfffu)|(high<<12);
    if (lower!=UINT32_MAX) scratch[lower]=(scratch[lower]&0xfffffu)|((a&0xfffu)<<20);
  }}
  /* Never ensure_loaded: its retained pointer may alias this mutable buffer. */
  return atlas_host_load(host,scratch,{count}u);
}}
'''
        receipt = dict(schema='atlas.resident_loop_image_c.v1',
                       template_sha256=sha256(self._binary).hexdigest(),
                       publication_sha256=self._publication_sha256,
                       options_sha256=sha256(_encoded(self._options)).hexdigest(),
                       word_count=count, address_ports=list(self._resident),
                       relocation_sites=[dict(port=p, upper=u, lower=l) for p, u, l in sites],
                       header_sha256=sha256(header.encode()).hexdigest(),
                       source_sha256=sha256(source.encode()).hexdigest(), forced_reload=True)
        receipt['abi_sha256'] = sha256(_encoded(receipt)).hexdigest()
        return ResidentImageC(header, source, receipt)
