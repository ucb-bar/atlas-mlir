"""Bound kit preflight shared by dry checks, execution, and receipt replay."""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import struct
import sys

# macOS links /tmp, /var, and /etc into /private. Only these system links are read through; every other link refuses.
SYSTEM_LINKS = {'/etc': '/private/etc', '/tmp': '/private/tmp', '/var': '/private/var'} if sys.platform == 'darwin' else {}


def system_path(p):
    """An absolute path with a leading system link replaced by its target."""
    for link, target in SYSTEM_LINKS.items():
        if (p == Path(link) or Path(link) in p.parents) and os.path.realpath(link) == target:
            return Path(target, p.relative_to(link))
    return p


# Installed tools consume typed facts; standalone kits retain the same source
# declaration in their authenticated inventory and need only the standard library.
_declaration = Path(__file__).with_name('soc.json')
if _declaration.is_file():
    _document = json.loads(_declaration.read_text())
    if _document.get('schema') != 'merlin.soc_description.v1':
        raise ValueError('unsupported kit SoC description')
    _windows = {row['id']: (row['base'], row['base'] + row['capacity_bytes'])
                for row in _document['memory_regions']}
else:
    from atlas_mlir_native.soc import SOC as _soc

    _windows = {region.id: (region.base, region.end) for region in _soc.memory_regions}
HOST_WINDOW = _windows['host_scratch']
DATA_WINDOW = _windows['atlas_data']
IMEM_WINDOW = _windows['atlas_imem']
PAGE = 4096


def inside(window, address, size):
    return window[0] <= address and address + size <= window[1]


def safe_path(value):
    p = Path(value)
    if '..' in p.parts:
        raise ValueError('traversing path')
    p = system_path(Path(os.path.abspath(p)))
    cur = Path(p.anchor)
    for x in p.parts[1:]:
        cur /= x
        try:
            if stat.S_ISLNK(cur.lstat().st_mode):
                raise ValueError('symlink path refused')
        except FileNotFoundError:
            break
    return p


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_bytes(path):
    return safe_path(path).read_bytes()


def load(bundle, elf='chip.elf'):
    bundle = safe_path(bundle)
    def member(name):
        if not isinstance(name, str) or Path(name).is_absolute():
            raise ValueError('kit member must be relative')
        return safe_path(bundle / name)
    listed = {}
    for line in read_bytes(bundle/'SHA256SUMS').decode().splitlines():
        digest, name = line.split('  ', 1)
        if not re.fullmatch('[0-9a-f]{64}', digest) or name in listed:
            raise ValueError('malformed or duplicate inventory')
        if sha(read_bytes(member(name))) != digest:
            raise ValueError('kit integrity mismatch: '+name)
        listed[name] = digest
    def bound(name):
        if name not in listed:
            raise ValueError('required member not inventoried: '+name)
        return member(name)
    m = json.loads(read_bytes(bound('manifest.json')))
    r = m.get('runner')
    if r is None:
        raise ValueError('bundle must declare runner metadata with its UART record tag')
    if r.get('schema') != 'atlas.kit.v1' or set(r)-{'schema','record_tag','paths','reference_paths','negative_paths','timeout_seconds','negative_patch'}:
        raise ValueError('unsupported runner schema or metadata')
    if not isinstance(r.get('record_tag'), str) or not re.fullmatch('[A-Za-z0-9_]+', r['record_tag']):
        raise ValueError('runner metadata must declare the UART record tag')
    paths = dict(deployment='application/deployment.json', window='window.bin', config='bench/atlas-soc.cfg', positive='chip.elf', negative='negative/chip.elf')
    if set(r.get('paths', {}))-set(paths):
        raise ValueError('unknown runner path')
    paths.update(r.get('paths', {}))
    if elf == 'chip.elf': elf = paths['positive']
    elif elf == 'negative/chip.elf': elf = paths['negative']
    if elf not in (paths['positive'], paths['negative']):
        raise ValueError('ELF not declared by manifest')
    dep = json.loads(read_bytes(bound(paths['deployment'])))
    if dep != m['deployment'] or 'paging' in dep:
        raise ValueError('deployment differs or requires paging')
    start, size = dep['window']['address'], dep['window']['bytes']
    window = bound(paths['window'])
    if start < 0 or size <= 0 or len(read_bytes(window)) > size:
        raise ValueError('invalid deployment window')
    ranges = dep['hash_ranges']
    if not ranges or ranges != dep['outputs']:
        raise ValueError('hash ranges must cover every published output')
    expected = {(x['name'], x['role']): x for x in m['expected_hashes']}
    if len(expected) != len(m['expected_hashes']) or len(expected) != len(ranges):
        raise ValueError('duplicate or missing expected hashes')
    refs, negs, intervals, keys = [], [], [], set()
    for row in ranges:
        key = (row['id'], row['role'])
        if key in keys: raise ValueError('duplicate output range')
        keys.add(key)
        a, n = row['address'], row['bytes']
        if n <= 0 or a < start or a+n > start+size or any(a < end and begin < a+n for begin,end in intervals):
            raise ValueError('invalid or overlapping output range')
        intervals.append((a,a+n))
        name = r.get('reference_paths', {}).get(str(row['index']), 'reference/output_%03d.bin' % row['index'])
        data = read_bytes(bound(name))
        h = expected[key]
        if len(data)!=n or h['bytes']!=n or sha(data)!=h['sha256']:
            raise ValueError('invalid bound reference: '+name)
        refs.append(dict(row=row,path=member(name),data=data,sha256=sha(data)))
        name = r.get('negative_paths', {}).get(str(row['index']), 'negative/negative-evidence/output_%03d.bin' % row['index'])
        if name in listed:
            data = read_bytes(bound(name))
            if len(data)!=n: raise ValueError('invalid negative evidence size')
            negs.append(dict(row=row,path=member(name),data=data,sha256=sha(data)))
    if not any(any(ref['data']) for ref in refs):
        raise ValueError('all-zero reference evidence')
    settings = dict(baud=m['bench_facts']['receiver_baud'], core_clock_hz=m['bench_facts']['core_clock_hz'],
                    clock_source=m['bench_facts']['source'], timeout=r.get('timeout_seconds',m['platform_assumptions']['timeout_seconds']),
                    record_tag=r['record_tag'])
    if any(not isinstance(v,(int,float)) or not math.isfinite(v) or v<=0 for k,v in settings.items() if k not in ('clock_source','record_tag')):
        raise ValueError('invalid clock, baud, or timeout')
    image = bound(elf)
    raw = read_bytes(image)
    if len(raw)<64 or raw[:6]!=b'\x7fELF\x02\x01' or struct.unpack_from('<Q',raw,24)[0]!=m['bench']['entry_address']:
        raise ValueError('ELF format or entry mismatch')
    selected_name = str(Path(elf).parent/'manifest.json')
    selected = json.loads(read_bytes(bound(selected_name)))
    meta = selected['sizes']['chip']
    if sha(raw)!=meta['sha256']: raise ValueError('selected ELF metadata hash drift')
    if elf==paths['negative'] and (len(negs)!=len(refs) or not any(x['data']!=y['data'] for x,y in zip(refs,negs))):
        raise ValueError('negative must have bound output-changing evidence')
    patch = r.get('negative_patch')
    if patch is not None:
        if not isinstance(patch, dict) or set(patch) != {'address', 'before_byte', 'after_byte'}:
            raise ValueError('invalid negative byte patch')
        address, before, after = (patch[k] for k in ('address', 'before_byte', 'after_byte'))
        if (type(address) is not int or type(before) is not int or type(after) is not int
                or not 0 <= before <= 255 or not 0 <= after <= 255 or before == after):
            raise ValueError('invalid negative byte patch values')
        qualified = json.loads(read_bytes(bound('qualification/check.json')))['byte_flip']
        if any(qualified.get(k) != patch[k] for k in patch):
            raise ValueError('negative patch differs from qualification')
        offset = address - start
        data = read_bytes(window)
        if not 0 <= offset < len(data) or data[offset] != before:
            raise ValueError('negative patch preimage differs from bound window')
        if not any(row['address'] <= address < row['address'] + row['bytes'] for row in dep.get('inputs', [])):
            raise ValueError('negative patch is outside declared inputs')
    result = dict(manifest=m,dep=dep,elf=image,window=window,config=bound(paths['config']),elf_bytes=sum(x['file_bytes'] for x in meta['loads']),references=refs,negative=negs,settings=settings,paths=paths,negative_patch=patch if elf==paths['negative'] else None)
    if __package__:
        from .preflight import physical
    else:
        from preflight import physical
    physical(result)
    return result


def console_errors(manifest, data, tag):
    """Strict complete UART records with every named digest, in emitted order."""
    text = data.decode('utf-8').replace('\r\n','\n')
    text = re.sub(r'WARNING: address 0x[0-9a-f]+ is not aligned to the request size of 64\n','',text)
    lines = text.splitlines()
    wanted = ['%s HASH %s %s %s' % (tag,x['name'],x['role'],x['sha256']) for x in manifest['expected_hashes']]
    errors=[]
    if len(lines)!=len(wanted)+2 or not lines or not re.fullmatch(re.escape(tag)+r' STATUS 0 CYCLES [0-9]+',lines[0]) or lines[-1]!=tag+' COMPLETE' or not text.endswith('\n'):
        errors.append('missing or malformed successful completion')
    actual=lines[1:-1]
    if sorted(actual)!=sorted(wanted): errors.append('UART hashes differ from bound reference')
    return errors
