"""Bind actual native dual-driver controls to preparation-only segment metadata."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).absolute().parents[1]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'tools/host_iss')]
from atlas_path_guard import checked
from full_bundle import FullBundle


def digest(raw): return hashlib.sha256(raw).hexdigest()


def finalize(root, controls):
    bundle=FullBundle(root,'smol-full')
    data=json.loads(bundle.read(bundle.provenance['contract']))
    records=[];compiles=[]
    for mode,directory in zip(('full','phased','corrupt'),controls):
        record=json.loads(checked(directory/'driver-result.json').read_bytes())
        compiled=json.loads(checked(directory/'compile.json').read_bytes())
        if (record.get('status')!='PASS_NATIVE_DUAL_DRIVER_'+mode.upper()
            or compiled['bundle_manifest_sha256']!=bundle.checksums_sha256
            or record.get('device_execution') is not False or record.get('guest_execution') is not False):
            raise ValueError('Exact successful native driver controls required')
        for name,expected in compiled['source_sha256'].items():
            if digest(checked(name).read_bytes())!=expected: raise ValueError('Changed native source '+name)
        if digest(checked(directory/'host.so').read_bytes())!=compiled['library_sha256']:
            raise ValueError('Changed compiled native control')
        records.append(record);compiles.append(compiled)
    full,phased,corrupt=records
    if (full['starts']!=2715 or full['full_canonical_boundaries_exact']!=200
        or full['full_cycle_observations_restored'] is not True or len(full['predecessors'])!=1421 or len(full['actual_buffers'])!=47
        or not all(x['exact'] for x in full['actual_buffers'])
        or phased['starts']!=2715 or phased['boundaries']!=200 or len(phased['predecessors'])!=1421
        or corrupt['starts']!=0 or corrupt['boundaries']!=0):
        raise ValueError('Incomplete native full/phase/negative traversal')
    products=dict(bundle.sums)
    def write(name,raw):
        target=checked(root/name);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
        products[name]=digest(raw)
    def json_write(name,value): write(name,(json.dumps(value,indent=2,sort_keys=True)+'\n').encode())
    snapshots={};pins={}
    for compiled in compiles:
        for path,expected in compiled['source_sha256'].items():
            source=checked(path)
            # Bind every actual production unit, generated driver, and the
            # qualification implementation; large tool executables stay in
            # the complete native compile receipt rather than copied blobs.
            if source.suffix not in ('.c','.h','.py'): continue
            if path in pins and pins[path]!=expected: raise ValueError('Native source pin conflict')
            member='sources/'+expected+'-'+source.name
            raw=source.read_bytes()
            if digest(raw)!=expected: raise ValueError('Changed source snapshot')
            write('segments/'+member,raw);snapshots[path]=member;pins[path]=expected
    for name in ('smol_full_driver.c','smol_full_driver.h','smol_policy_graph.c','smol_policy_graph.h',
                 'smol_vision_graph.c','smol_vision_graph.h'):
        matches={v for k,v in pins.items() if Path(k).name==name}
        if matches!={bundle.sums['host/'+name]}: raise ValueError('RV64/native source disagreement '+name)
    for mode,directory in zip(('full','phased','corrupt'),controls):
        for name in ('driver-result.json','compile.json'):
            write('segments/controls/'+mode+'-'+name,checked(directory/name).read_bytes())
        uart='uart-positive.log' if mode=='full' else 'uart-phases.log'
        write('segments/controls/'+mode+'-uart.log',checked(directory/uart).read_bytes())
    original_checksums=checked(root/'SHA256SUMS').read_bytes()
    original_segments=bundle.read('segments.json')
    original_receipt=bundle.read('segments/receipt.json')
    write('segments/native-input-SHA256SUMS',original_checksums)
    write('segments/native-input-segments.json',original_segments)
    write('segments/native-input-receipt.json',original_receipt)
    chain=json.loads(original_segments)
    chain['contract']=bundle.provenance['contract']
    chain['contract_sha256']=bundle.provenance['contract_sha256']
    semantic={item['name']:item['bytes'] for item in chain['live_regions']}
    receipt=dict(schema='atlas.smol_boundary_preparation.v1',
        status='PREPARED_REFERENCE_BOUNDARIES_NOT_EXECUTED',reference_only=True,guest_execution=False,
        device_execution=False,segments=200,boundaries=201,projection_calls=1421,physical_panels=2715,
        source_reference_sha256=data['source_reference_sha256'],
        native_input_checksums_sha256=digest(original_checksums),
        native_input_segments_sha256=digest(original_segments),
        native_input_receipt_sha256=digest(original_receipt),
        compiled_phase_layout=[1,40,28,16,16,160,144,80,112,2715,1421,200],
        source_bookends_verified=True,all_actual_predecessors_exact=True,all_final_references_exact=True,
        full_and_phases_exact=True,scratch_reset_each_phase=True,arena_reset_each_phase=True,
        canonical_zero_observations=True,live_corruption_rejected=True,
        semantic_regions=semantic,source_sha256=pins,source_snapshots=snapshots,
        observation_policy='Native protocol deltas are synthetic and separate; canonical live cycle slots are zero. '
            'Real segment runner must compare every own-range observation with actual Device completion cycles.',
        controls={mode:'segments/controls/'+mode+'-driver-result.json' for mode in ('full','phased','corrupt')},
        control_uart={mode:'segments/controls/'+mode+'-uart.log' for mode in ('full','phased','corrupt')},
        control_compiles={mode:'segments/controls/'+mode+'-compile.json' for mode in ('full','phased','corrupt')})
    json_write('segments/receipt.json',receipt)
    chain['capture_receipt_sha256']=products['segments/receipt.json']
    json_write('segments.json',chain)
    checked(root/'SHA256SUMS').write_text(''.join(value+'  '+name+'\n' for name,value in sorted(products.items())))
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=checked,required=True)
    for name in ('full','phased','corrupt'):p.add_argument('--'+name,type=checked,required=True)
    a=p.parse_args();finalize(a.bundle,[a.full,a.phased,a.corrupt])
