"""Ordered staged execution shared by simulated and persistent Tcl transports."""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
import sys

if __package__:
    from .kit_manifest import DATA_WINDOW, HOST_WINDOW, PAGE, inside, safe_path, sha
else:
    from kit_manifest import DATA_WINDOW, HOST_WINDOW, PAGE, inside, safe_path, sha


class BoundaryError(ValueError):
    pass


def chunks(spec, size, extent):
    """`count` runs of `bytes` at `offset + i*stride`, totalling `size`, inside `extent`.

    An omitted range is the whole value or file, so its length must equal `size`.
    """
    if spec is None:
        if size != extent:
            raise ValueError('omitted byte range must cover the whole value')
        return [(0, size)]
    if set(spec) != {'offset','bytes','count','stride'} or any(type(spec[k]) != int or spec[k] < 0 for k in spec):
        raise ValueError('invalid byte range')
    offset, n, count, stride = spec['offset'], spec['bytes'], spec['count'], spec['stride']
    if n <= 0 or count <= 0 or n*count != size or (count > 1 and stride < n) or offset+(count-1)*stride+n > extent:
        raise ValueError('invalid byte range')
    return [(offset+i*stride, n) for i in range(count)]


def gather(data, spec, size):
    return b''.join(data[a:a+n] for a, n in chunks(spec, size, len(data)))


def load_plan(root):
    root = safe_path(root)
    members = {}
    for line in (root/'SHA256SUMS').read_text().splitlines():
        digest, name = line.split('  ', 1)
        if not re.fullmatch('[0-9a-f]{64}', digest) or name in members:
            raise ValueError('invalid inventory')
        if Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError('invalid member')
        path = safe_path(root/name)
        if sha(path.read_bytes()) != digest:
            raise ValueError('inventory drift: '+name)
        members[name] = digest
    def bound(name):
        if name not in members:
            raise ValueError('unbound member: '+str(name))
        return safe_path(root/name)
    manifest = json.loads(bound('manifest.json').read_text())
    plan = manifest['composed']
    if set(plan)-{'regions','timeout_seconds','values','negative','receiver_baud'}:
        raise ValueError('unknown composed metadata')
    timeout = plan.get('timeout_seconds', 120)
    if not isinstance(timeout, (int,float)) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('invalid completion deadline')
    if 'receiver_baud' in plan and (type(plan['receiver_baud'])!=int or plan['receiver_baud']<=0):
        raise ValueError('invalid receiver baud')
    available, ids = {}, set()
    references = []
    # Whole tensors the bench host holds; regions read and write byte ranges of them.
    written = {}
    for row in plan.get('values', []):
        vid, size = row['id'], row['bytes']
        if set(row)-{'id','bytes','file','reference'} or not isinstance(vid,str) or vid in written or type(size)!=int or size<=0:
            raise ValueError('invalid composed value')
        written[vid] = bytearray(size)
        if 'file' in row:
            if len(bound(row['file']).read_bytes())!=size: raise ValueError('value file length mismatch')
            written[vid][:] = b'\x01'*size
        if 'reference' in row:
            data=bound(row['reference']).read_bytes()
            if len(data)!=size: raise ValueError('invalid value reference length')
            references.append(data)
    if 'negative' in plan:
        negative=plan['negative']; value=next((row for row in plan.get('values',[]) if row['id']==negative.get('value')),None)
        if set(negative)!={'value','file'} or value is None or 'file' not in value:
            raise ValueError('negative must replace one file-backed value')
        data=bound(negative['file']).read_bytes()
        if len(data)!=value['bytes'] or data==bound(value['file']).read_bytes():
            raise ValueError('negative value must change the packaged bytes')
    if not plan['regions']:
        raise ValueError('empty composed plan')
    for region in plan['regions']:
        rid = region['id']
        if not isinstance(rid,str) or not re.fullmatch('[A-Za-z0-9_-]+',rid) or rid in ids:
            raise ValueError('invalid or duplicate region id')
        ids.add(rid)
        kind = region['kind']
        if kind not in ('device','host','elf','application'):
            raise ValueError('unsupported region kind')
        # A step owns its declared window: page-aligned and inside the physical window.
        window=region.get('window')
        if window is not None and (window['address']%PAGE or window['bytes']%PAGE or window['bytes']<=0
                                   or not inside(DATA_WINDOW,window['address'],window['bytes'])):
            raise ValueError('step window is unaligned or exceeds the physical window')
        if kind in ('elf','application'):
            # A linked step ELF, or a step application without one: the device images it launches.
            for name in [region['deployment'], *(x[k] for x in region['images'] for k in ('program','plan'))]:
                bound(name)
            if window is None or type(region.get('launches'))!=int or region['launches']<0:
                raise ValueError('step needs a window and a launch count')
        if kind=='elf':
            # Its UART hashes every declared range; each output must be one, so no unhashed byte is judged.
            ranges={(row['address'],row['bytes']) for row in json.loads(bound(region['deployment']).read_text())['hash_ranges']}
            if any((row['address'],row['bytes']) not in ranges for row in region['outputs']):
                raise ValueError('step ELF output is not a UART hash range')
        if kind=='application':
            for name in region['sources']: bound(name)
        elif kind=='device':
            bound(region['program'])
        else:
            bound(region['elf'])
        if 'inventory' in region: bound(region['inventory'])
        code=[]
        if 'elf' in region:
            if __package__:
                from .composed_sim import _elf_loads
            else:
                from composed_sim import _elf_loads
            entry, loads = _elf_loads(bound(region['elf']).read_bytes())
            if region['entry']!=entry or not any(a<=entry<a+n and flags&1 for a,data,n,flags in loads):
                raise ValueError('invalid host entry')
            code=[(start,start+length) for start,data,length,flags in loads if flags&1]
            # A step ELF also carries its window image; host code stays in the scratchpad.
            if any(not (inside(HOST_WINDOW,a,n) or kind=='elf' and not flags&1
                        and window['address']<=a and a+n<=window['address']+window['bytes']) for a,data,n,flags in loads):
                raise ValueError('host ELF exceeds scratchpad')
        if kind=='host':
            status=region['status']; a=status['address']; n=status.get('bytes',8)
            if n!=8 or a%n or not inside(HOST_WINDOW,a,n):
                raise ValueError('invalid completion status range')
            if not 0<=status['done_value']<1<<(8*n) or not 0<=status.get('initial_value',0)<1<<(8*n) or status.get('initial_value',0)==status['done_value']:
                raise ValueError('invalid completion sentinel')
            if any(a<end and start<a+n for start,end in code):
                raise ValueError('status overlaps executable code')
        intervals, outputs = [], set()
        if not region['outputs']: raise ValueError('region has no outputs')
        for role in ('inputs','weights','outputs'):
            names = set()
            for row in region.get(role,[]):
                name, address, size = row['id'], row['address'], row['bytes']
                if name in names: raise ValueError('duplicate buffer id')
                names.add(name)
                if type(address)!=int or type(size)!=int or size<=0:
                    raise ValueError('invalid buffer range')
                if not (inside(DATA_WINDOW,address,size) or kind=='host' and inside(HOST_WINDOW,address,size)):
                    raise ValueError('buffer exceeds physical window')
                if any(address<end and start<address+size for start,end in intervals):
                    raise ValueError('overlapping region buffers')
                if any(address<end and start<address+size for start,end in code):
                    raise ValueError('buffer overlaps executable code')
                if kind=='host':
                    status=region['status']; a=status['address']; n=status.get('bytes',8)
                    if address<a+n and a<address+size: raise ValueError('status overlaps buffer')
                if window is not None and not window['address']<=address<address+size<=window['address']+window['bytes']:
                    raise ValueError('buffer exceeds step window')
                intervals.append((address,address+size))
                if role=='outputs':
                    outputs.add(name)
                    if 'reference' in row:
                        data=bound(row['reference']).read_bytes()
                        if len(data)!=size:
                            raise ValueError('invalid boundary reference length')
                        references.append(data)
                    if 'value' in row:
                        target=written.get(row['value']['id'])
                        if target is None: raise ValueError('unknown composed value')
                        for a,n in chunks(row['value'].get('range'),size,len(target)):
                            if any(target[a:a+n]): raise ValueError('value bytes written twice')
                            target[a:a+n]=b'\x01'*n
                else:
                    if ('file' in row)==('source' in row): raise ValueError('buffer needs exactly one source')
                    if 'file' in row:
                        chunks(row.get('range'),size,len(bound(row['file']).read_bytes()))
                    elif 'value' in row['source']:
                        target=written.get(row['source']['value'])
                        if target is None or any(not all(target[a:a+n]) for a,n in chunks(row['source'].get('range'),size,len(target))):
                            raise ValueError('missing producer or size mismatch')
                    else:
                        source=row['source']; key=(source['region'],source['output'])
                        if available.get(key)!=size: raise ValueError('missing producer or size mismatch')
        for row in region['outputs']: available[rid,row['id']]=row['bytes']
    if references and not any(any(data) for data in references):
        raise ValueError('all-zero reference evidence')
    if any(not all(written[row['id']]) for row in plan.get('values', []) if 'reference' in row):
        raise ValueError('referenced value is never completed')
    rows=manifest.get('deployment',{}).get('outputs',[])
    if [row['index'] for row in rows]!=list(range(len(rows))):
        raise ValueError('deployment outputs must be indexed in order')
    for row in rows:
        source=row['source']; producer=[x for x in plan['regions'] if x['id']==source['region']]
        match=[x for x in producer[0]['outputs'] if x['id']==source['output']] if producer else []
        if not match or (match[0]['address'],match[0]['bytes'])!=(row['address'],row['bytes']):
            raise ValueError('deployment output must name one region output range')
    return manifest, members, bound


def atomic_json(path, value):
    temporary=path.with_suffix('.tmp')
    with temporary.open('w') as f:
        json.dump(value,f,sort_keys=True,indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())
    temporary.replace(path)


def _spans(manifest, bound):
    """Byte ranges each plan member writes, keyed by owner."""
    if __package__:
        from .composed_sim import _elf_loads
    else:
        from composed_sim import _elf_loads
    spans=[]
    for region in manifest['composed']['regions']:
        rid=region['id']
        for role in ('inputs','weights','outputs'):
            for row in region.get(role,[]): spans.append((row['address'],row['address']+row['bytes'],(rid,role,row['id'])))
        if 'elf' in region:
            for a,data,n,flags in _elf_loads(bound(region['elf']).read_bytes())[1]: spans.append((a,a+n,(rid,'code','')))
        if 'status' in region:
            status=region['status']; spans.append((status['address'],status['address']+status.get('bytes',8),(rid,'status','')))
        if region['kind']=='application':
            # An application step clears its whole window, so nothing inside it stays resident.
            window=region['window']; spans.append((window['address'],window['address']+window['bytes'],(rid,'window','')))
    return spans


class Session:
    """Resident composed image for repeated calls on one transport.

    Host code and weights that no other plan range or step application window
    overlaps are written once; shared ranges are rewritten before their region
    on every call. Each call
    re-verifies the inventory against the digest pinned at open.
    """

    def __init__(self, root, work, backend='sim', transport_factory=None, **options):
        self.root=safe_path(root)
        self.manifest, self.inventory, self.bound = load_plan(self.root)
        self.pin=sha((self.root/'SHA256SUMS').read_bytes())
        self.backend, self.work, self.options = backend, safe_path(work), options
        self.factory, self.transport = transport_factory, None
        spans=_spans(self.manifest,self.bound)
        # An owner stays resident only if none of its spans (every ELF segment) meets another owner's.
        shared={key for start,end,key in spans if any(s<end and start<e and k!=key for s,e,k in spans)}
        self.resident={key for _,_,key in spans if key[1] in ('weights','code')}-shared

    def _open(self):
        factory=self.factory
        if factory is None:
            if self.backend=='sim':
                if __package__:
                    from .composed_sim import SimTransport
                else:
                    from composed_sim import SimTransport
                factory=SimTransport
            else:
                if __package__:
                    from .composed_tcl import TclTransport
                else:
                    from composed_tcl import TclTransport
                factory=lambda r,w,m:TclTransport(r,w,m,**self.options)
        self.transport=factory(self.root,self.work,self.manifest)
        for region in self.manifest['composed']['regions']:
            self._load(region,resident=True)

    def _load(self, region, resident):
        rid=region['id']
        if 'elf' in region and ((rid,'code','') in self.resident)==resident and hasattr(self.transport,'prepare'):
            self.transport.prepare(region)
        for row in region.get('weights',[]):
            if ((rid,'weights',row['id']) in self.resident)==resident:
                self.transport.write(row['address'],gather(self.bound(row['file']).read_bytes(),row.get('range'),row['bytes']))

    def call(self, logs, inputs=None, mutation=False, resume=False):
        manifest, inventory, bound = load_plan(self.root)
        if sha((self.root/'SHA256SUMS').read_bytes())!=self.pin or manifest!=self.manifest:
            raise ValueError('bundle changed during session')
        logs=safe_path(logs)
        staged = []
        if inputs is not None:
            if __package__:
                from .step_inputs import load_inputs
            else:
                from step_inputs import load_inputs
            staged = load_inputs(manifest.get('deployment', {}), inputs)
        overrides = {}
        for item in staged:
            matches = [(region['id'], row['id']) for region in manifest['composed']['regions']
                       for row in region.get('inputs', []) if 'file' in row
                       and row['address'] == item['row']['address'] and row['bytes'] == item['row']['bytes']]
            if len(matches) != 1:
                raise ValueError('deployment input must match exactly one composed input range')
            overrides[matches[0]] = item['data']
        plan = manifest['composed']
        value_files = {row['id']: row['file'] for row in plan.get('values', []) if 'file' in row}
        if mutation:
            if staged: raise ValueError('mutation qualification cannot override step inputs')
            candidates = [(region, row) for region in plan['regions']
                          for row in region.get('inputs', []) if 'file' in row]
            if 'negative' in plan:
                # The packaged negative replaces one value's initial bytes.
                value_files[plan['negative']['value']] = plan['negative']['file']
            elif not candidates: raise ValueError('mutation needs a file-backed input')
            else:
                region, row = candidates[0]
                data = bytearray(gather(bound(row['file']).read_bytes(), row.get('range'), row['bytes'])); data[0] ^= 1
                overrides[region['id'], row['id']] = bytes(data)
        declared = {(region['id'], row['id']): row['bytes'] for region in plan['regions'] for row in region.get('inputs', [])}
        for (rid, name), data in overrides.items():
            if len(data) != declared[rid, name]: raise ValueError('input bytes differ from its declared range: '+rid+'/'+name)
        identity={'inventory':inventory,'backend':self.backend,'runner':sha(Path(__file__).read_bytes())}
        identity['step_inputs'] = {region+'/'+name: sha(data) for (region,name),data in overrides.items()}
        identity['values'] = value_files
        if self.backend=='sim':
            if __package__:
                from .composed_sim import runtime_identity
            else:
                from composed_sim import runtime_identity
            identity['runtime']=runtime_identity()
        # Bind both execution layers, not just the producer's artifacts.
        for filename in ('composed_sim.py','tcl_rpc.py','composed_tcl.py'):
            path=Path(__file__).with_name(filename)
            if path.exists(): identity[filename]=sha(path.read_bytes())
        if resume:
            if not logs.is_dir(): raise ValueError('resume requires existing journal')
        else: logs.mkdir(parents=True,exist_ok=False)
        lock=safe_path(logs/'active.lock')
        fd=os.open(lock,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        try:
            journal_path=safe_path(logs/'journal.json')
            if resume:
                journal=json.loads(journal_path.read_text())
                if journal['identity']!=identity or journal.get('active') is not None:
                    raise ValueError('stale or incomplete checkpoint')
            else:
                journal={'identity':identity,'completed':[],'active':None}
                atomic_json(journal_path,journal)
            regions=plan['regions']; actual={}
            files={}
            def member(name):
                if name not in files: files[name]=bound(name).read_bytes()
                return files[name]
            values={row['id']:bytearray(member(value_files[row['id']]) if row['id'] in value_files else bytes(row['bytes']))
                    for row in plan.get('values',[])}
            value_references={row['id']:row['reference'] for row in plan.get('values',[]) if 'reference' in row}
            def scatter(row,data):
                target=values[row['value']['id']]
                for i,(a,n) in enumerate(chunks(row['value'].get('range'),len(data),len(target))):
                    target[a:a+n]=data[i*n:(i+1)*n]
            def differs(row,data):
                # A written slice is compared with the same byte ranges of its value's reference.
                reference=value_references.get(row['value']['id'])
                return reference is not None and data!=gather(member(reference),row['value'].get('range'),len(data))
            if len(journal['completed'])>len(regions): raise ValueError('invalid checkpoint count')
            for region,receipt in zip(regions,journal['completed']):
                if receipt['id']!=region['id']: raise ValueError('checkpoint order mismatch')
                if set(receipt['outputs'])!={row['id'] for row in region['outputs']}: raise ValueError('incomplete checkpoint outputs')
                for index,row in enumerate(region['outputs']):
                    meta=receipt['outputs'][row['id']]
                    if meta['file']!=f'{region["id"]}-output-{index}.bin': raise ValueError('invalid checkpoint member')
                    data=safe_path(logs/meta['file']).read_bytes()
                    if len(data)!=row['bytes'] or sha(data)!=meta['after']:
                        raise ValueError('checkpoint byte drift')
                    if not staged and 'reference' in row and data!=bound(row['reference']).read_bytes():
                        raise ValueError('checkpoint reference mismatch')
                    if 'value' in row:
                        if not staged and not mutation and differs(row,data): raise ValueError('checkpoint reference mismatch')
                        scatter(row,data)
                    actual[region['id'],row['id']]=data
            for region in regions[len(journal['completed']):]:
                if self.transport is None: self._open()
                transport=self.transport
                rid=region['id']; journal['active']=rid; atomic_json(journal_path,journal)
                if region['kind']=='application':
                    # Without an ELF window image, clear the step window as that image would.
                    transport.write(region['window']['address'],bytes(region['window']['bytes']))
                self._load(region,resident=False)
                if 'status' in region:
                    status=region['status']
                    transport.write(status['address'],status.get('initial_value',0).to_bytes(status.get('bytes',8),'little'))
                before={}
                for row in region['outputs']:
                    # A write that returns bytes returns what it read back from the device after writing (Tcl).
                    poison=b'\xa5'*row['bytes']; data=transport.write(row['address'],poison)
                    if not isinstance(data,bytes): data=transport.read(row['address'],row['bytes'])
                    if data!=poison: raise ValueError('output poison verification failed')
                    before[row['id']]=sha(data)
                written={}
                for row in region.get('inputs',[]):
                    if 'file' in row: data=gather(member(row['file']),row.get('range'),row['bytes'])
                    elif 'value' in row['source']:
                        data=gather(values[row['source']['value']],row['source'].get('range'),row['bytes'])
                    else:
                        source=row['source']; data=actual[source['region'],source['output']]
                    data = overrides.get((rid, row['id']), data)
                    transport.write(row['address'],data)
                    written['inputs:'+row['id']]=sha(data)
                for row in region.get('weights',[]):
                    written['weights:'+row['id']]=sha(gather(member(row['file']),row.get('range'),row['bytes']))
                transport.launch(region)
                transport.wait(region,manifest['composed'].get('timeout_seconds',120))
                receipt={'id':rid,'inputs':written,'outputs':{}}
                errors, fatal = [], False
                # Dump and keep every output before judging any of them.
                for index,row in enumerate(region['outputs']):
                    data=transport.read(row['address'],row['bytes'])
                    filename=f'{rid}-output-{index}.bin'
                    safe_path(logs/filename).write_bytes(data)
                    receipt['outputs'][row['id']]={'file':filename,'before':before[row['id']],'after':sha(data)}
                    actual[rid,row['id']]=data
                    name=rid+'/'+row['id']
                    if len(data)!=row['bytes']: errors.append('short output read: '+name); fatal=True
                    elif data==b'\xa5'*row['bytes']: errors.append('output remained poisoned: '+name); fatal=True
                    elif not staged and 'reference' in row and data!=bound(row['reference']).read_bytes():
                        errors.append('boundary mismatch: '+name)
                    elif not staged and 'value' in row and differs(row,data):
                        errors.append('boundary mismatch: '+name+' (value '+row['value']['id']+')')
                if errors:
                    journal['rejected']=dict(receipt,errors=errors); atomic_json(journal_path,journal)
                    # Only a region whose every failure is a numerical mismatch is a BoundaryError.
                    raise (ValueError if fatal else BoundaryError)('; '.join(errors))
                for row in region['outputs']:
                    if 'value' in row: scatter(row,actual[rid,row['id']])
                journal['completed'].append(receipt); journal['active']=None
                atomic_json(journal_path,journal)
            for row in manifest.get('deployment',{}).get('outputs',[]):
                source=row['source']
                safe_path(logs/f'output-{row["index"]}.bin').write_bytes(actual[source['region'],source['output']])
            return journal
        finally:
            os.close(fd); lock.unlink()

    def close(self):
        if self.transport is not None: self.transport.close()


def recheck(root, logs):
    """Recheck a finished positive run from its saved outputs alone, without a transport.

    The bundle is authenticated as for execution, and the journal must be this bundle's packaged-input run with every
    region completed in plan order. Each saved output must have its declared length and journal hash. Plan values are
    rebuilt from those outputs, and every boundary and value reference is compared byte for byte. All-zero references
    are listed and not counted as coverage. Returns the counts; any difference raises BoundaryError.
    """
    manifest, inventory, bound = load_plan(root)
    logs=safe_path(logs); journal=json.loads(safe_path(logs/'journal.json').read_text())
    plan=manifest['composed']; regions=plan['regions']; identity=journal.get('identity',{})
    files={row['id']:row['file'] for row in plan.get('values',[]) if 'file' in row}
    if identity.get('inventory')!=inventory or identity.get('step_inputs')!={} or identity.get('values')!=files:
        raise ValueError('journal is not a packaged-input run of this bundle')
    if journal.get('active') is not None or 'rejected' in journal or [r['id'] for r in journal['completed']]!=[r['id'] for r in regions]:
        raise ValueError('journal does not complete every region in plan order')
    values={row['id']:bytearray(row['bytes']) for row in plan.get('values',[])}
    checked, differing, zero, slices = 0, [], [], 0
    def compare(name, data, reference):
        nonlocal checked
        if not any(reference): zero.append(name)
        else: checked+=1
        if data!=reference: differing.append(name)
    for region,receipt in zip(regions,journal['completed']):
        if set(receipt['outputs'])!={row['id'] for row in region['outputs']}: raise ValueError('incomplete saved outputs: '+region['id'])
        for index,row in enumerate(region['outputs']):
            meta=receipt['outputs'][row['id']]; name=region['id']+'/'+row['id']
            if meta['file']!=f'{region["id"]}-output-{index}.bin': raise ValueError('invalid saved output member: '+name)
            data=safe_path(logs/meta['file']).read_bytes()
            if len(data)!=row['bytes'] or sha(data)!=meta['after']: raise ValueError('saved output differs from its journal hash: '+name)
            slices+=1
            if 'reference' in row: compare(name,data,bound(row['reference']).read_bytes())
            if 'value' in row:
                target=values[row['value']['id']]
                for i,(a,n) in enumerate(chunks(row['value'].get('range'),len(data),len(target))): target[a:a+n]=data[i*n:(i+1)*n]
    referenced=[row for row in plan.get('values',[]) if 'reference' in row]
    for row in referenced: compare('value '+row['id'],bytes(values[row['id']]),bound(row['reference']).read_bytes())
    result=dict(regions=len(regions),completed=len(journal['completed']),output_slices=slices,referenced_values=len(referenced),
                compared_nonzero=checked,differing=differing,all_zero_references=zero)
    if differing: raise BoundaryError('recheck mismatch: '+', '.join(differing))
    if not checked: raise ValueError('no nonzero reference to recheck')
    return result


def execute(root, logs, backend='sim', resume=False, transport_factory=None, inputs=None, mutation=False, **options):
    """One call on a fresh session; the journal supports resume."""
    session=Session(root, logs, backend, transport_factory, **options)
    try:
        return session.call(logs, inputs=inputs, mutation=mutation, resume=resume)
    finally:
        # A cleanup error always surfaces; a call failure it replaces is its __context__.
        session.close()
