"""Generate equal-budget hand/egglog shortlists without claiming measured speed."""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
from atlas_egglog_ir import load_program, parse_program
from atlas_egglog_cost import TimingCosts
from atlas_hand_reference import enumerate_hand
from explore_atlas_egglog import search, implementation_pins


def digest(raw):return hashlib.sha256(raw).hexdigest()
def counts(source):
    result=Counter()
    for op in parse_program(source).operations:
        result[op.op+(':'+json.loads(dict(op.attrs)['kind']) if 'kind' in dict(op.attrs) else '')]+=1
    return dict(sorted(result.items()))


def compare(program,timing,*,budget=4,attempts=64):
    if not 1<=budget<=16:raise ValueError('measurement shortlist budget must be1..16 per strategy')
    egg,report=search(program,timing,attempts=attempts,limit=64)
    egg_rows=[dict(id=c.id,source=c.source.decode(),source_sha256=c.source_sha256,cost=asdict(c.cost),score=c.cost.score,
                   families=list(c.families),strategy='egglog',choices=report['details'][c.id]['choices']) for c in egg if c.id.startswith('egg-')]
    controls=[dict(id=c.id,source=c.source.decode(),source_sha256=c.source_sha256,cost=asdict(c.cost),score=c.cost.score,
                   families=list(c.families),strategy='native' if c.id=='native' else 'legacy-uniform',choices=report['details'][c.id]['choices']) for c in egg if not c.id.startswith('egg-')]
    # Shared shortlist deduplicates an extraction identical to a mandatory
    # native/hand control. Keep its independently generated strategy identity
    # for this comparison even when the resulting source bytes are a tie.
    by_digest={row['source_sha256']:row for row in controls+egg_rows}
    for name,detail in report['details'].items():
        if name.startswith('egg-') and detail['source_sha256'] in by_digest and not any(r['id']==name for r in egg_rows):
            egg_rows.append(by_digest[detail['source_sha256']]|dict(id=name,strategy='egglog',choices=detail['choices']))
    start=time.perf_counter();hand=[];rejected=[]
    for name,source,assignment,error in enumerate_hand(program):
        if name=='native':continue
        if error:rejected.append(dict(id=name,assignment=assignment,reason=error));continue
        cost=timing.estimate(parse_program(source))
        hand.append(dict(id=name,source=source,source_sha256=digest(source.encode()),cost=asdict(cost),score=cost.score,
                         families=['directed_exact_rules','guarded_pack'],strategy='hand',choices=assignment))
    hand_seconds=time.perf_counter()-start
    def unique_best(rows):
        seen=set();selected=[]
        for row in sorted(rows,key=lambda r:(r['score'],r['source_sha256'],r['id'])):
            if row['source_sha256'] not in seen:
                selected.append(row);seen.add(row['source_sha256'])
            if len(selected)>=budget:break
        return selected
    selected=controls+unique_best(hand)+unique_best(egg_rows)
    for row in selected:row['operation_counts']=counts(row['source'])
    aliases={};canonical={};measured=[]
    for row in selected:
        identity=row['source_sha256']
        if identity not in canonical:
            canonical[identity]=row['id'];measured.append(row)
        aliases[row['id']]=canonical[identity]
    report.update(hand=dict(pool_size=len(hand),rejected=rejected,distinct_sources=len({r['source_sha256'] for r in hand}),seconds=hand_seconds,
        enumeration='All native/unit0/unit1 choices at each automatically finite PACK, at most6 sites; one directed normal form of identical local exact laws; no global optimum.',
        pool=[{k:v for k,v in r.items() if k!='source'} for r in hand]),
        measured_budget_per_strategy=budget,measurement_scope='Cost-ranked distinct source shortlists, not exhaustive measured hand optimum; all native and legacy uniform controls retained.',
        comparison_selected=[r['id'] for r in selected],measurement_aliases=aliases,
        strategy_selections=[{k:v for k,v in row.items() if k!='source'} for row in selected],
        identical_source_policy='Identical SHA-256 source bytes share one compiled measurement; aliases preserve independent strategy selections and report ties.')
    return measured,report


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('source',type=Path);ap.add_argument('--timing-facts',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--budget',type=int,default=4);ap.add_argument('--attempts',type=int,default=64);args=ap.parse_args()
    program=load_program(args.source);rows,report=compare(program,TimingCosts(args.timing_facts),budget=args.budget,attempts=args.attempts)
    args.output.mkdir(parents=True,exist_ok=True)
    if any(args.output.iterdir()):raise ValueError('output must be empty')
    pins=implementation_pins()
    import atlas_hand_reference
    for path in (Path(__file__).resolve(),Path(atlas_hand_reference.__file__).resolve()):pins[str(path)]=digest(path.read_bytes())
    proof=report['proof']|dict(implementation_pins=pins,qualification='Provider declaration; compiler and exact measured output gates required.',
        candidate_bindings={r['id']:dict(source_sha256=r['source_sha256'],choices=r['choices'],families=r['families']) for r in rows})
    proof_path=args.output/'proof.json';proof_path.write_text(json.dumps(proof,indent=2)+'\n');proof_ref=dict(path=str(proof_path.resolve()),sha256=digest(proof_path.read_bytes()),schema=proof['schema'])
    candidates=[]
    for row in rows:
        path=args.output/(row['id']+'.mlir');path.write_text(row['source'])
        candidates.append(dict(id=row['id'],source=str(path.resolve()),source_sha256=row['source_sha256'],admitted=True,estimated_cost=row['score'],proof=proof_ref,families=row['families'],strategy=row['strategy']))
    report['selected_details']=[{k:v for k,v in r.items() if k!='source'} for r in rows]
    (args.output/'candidates.json').write_text(json.dumps(dict(candidates=candidates),indent=2)+'\n')
    (args.output/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status='GENERATED_NOT_MEASURED',selected=[r['id'] for r in rows],egglog_seconds=report['seconds'],hand_seconds=report['hand']['seconds'])))


if __name__=='__main__':main()
