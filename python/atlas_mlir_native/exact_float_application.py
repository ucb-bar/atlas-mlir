"""Target-owned exact float slicing, with atomic logical result publication."""
from dataclasses import dataclass,replace
from hashlib import sha256
import json
from pathlib import Path
from . import exact_float_device as device
from merlin.semantic_compiler.model import pointwise_identity_maps

def _json(r):return (json.dumps(r,sort_keys=True,indent=2)+"\n").encode()

@dataclass(frozen=True)
class FloatSchedule:
    plan: object
    bodies: tuple
    launches: tuple
    @property
    def source(self):return self.plan.source
    @property
    def chunk_elements(self):return device.chunk_elements(self.plan)
    def record(self):
        return dict(schema="atlas.exact_float_schedule.v1",source_request_digest=self.source.digest(),
                    operation=self.plan.op,chunk_elements=self.chunk_elements,input_node_ids=[n.id for n in self.plan.operands],
                    output_node_ids=list(self.source.outputs),logical_count=self.plan.count,
                    bodies=[dict(index=i,path=f"segments/body_{i:03d}",request_digest=b.digest()) for i,b in enumerate(self.bodies)],
                    launches=list(self.launches),launches_per_invocation=len(self.launches),runtime_arithmetic="none",
                    publication="copy logical output only after every device status succeeds",weight_conversion="none")

def prepare_plan(source):
    p=device.prepare_plan(source);bodies=[];sizes={};launches=[]
    chunk=device.chunk_elements(p)
    for start in range(0,p.count,chunk):
        size=min(chunk,p.count-start)
        if size not in sizes:
            sizes[size]=len(bodies)
            body=replace(source,nodes=tuple(replace(n,type=replace(n.type,shape=(size,)),
                index_maps=() if n.effect=="input" else pointwise_identity_maps(1,len(n.inputs))) for n in source.nodes),
                source_identity=sha256(_json(dict(source_request_digest=source.digest(),count=size))).hexdigest())
            device.prepare_plan(body);bodies.append(body)
        launches.append(dict(body_index=sizes[size],input_offset_bytes=start*p.width,input_bytes=size*p.width,
                             output_offset_bytes=start*p.output_width,output_bytes=size*p.output_width))
    return FloatSchedule(p,tuple(bodies),tuple(launches))

def compile_source(snapshot,region,*,target_source,destination,limits):
    p=prepare_plan(region);destination=Path(destination);destination.mkdir(parents=True,exist_ok=False);(destination/"segments").mkdir()
    records={"source-request.json":region.record(),"float-schedule.json":p.record()}
    for filename,r in records.items():(destination/filename).write_bytes(_json(r))
    segments=[]
    for i,body in enumerate(p.bodies):
        relative=f"segments/body_{i:03d}"
        detail=device.compile_source(snapshot,body,target_source=target_source,destination=destination/relative,limits=limits)
        segments.append(dict(index=i,path=relative,request_digest=body.digest(),binary_sha256=detail["binary_sha256"],
                             manifest_sha256=sha256((destination/relative/"manifest.json").read_bytes()).hexdigest()))
    manifest=dict(schema="atlas.exact_float_application.v1",artifact_kind="program_set",engine="merlin_native",compile_mode="diagnostic",
                  qualification_scope="diagnostic_only",request_digest=region.digest(),target_identity=region.target_identity,
                  segments=segments,launches_per_invocation=len(p.launches),execution_requires="ordered_exact_float_slices_then_atomic_publication",
                  stage_artifacts={filename:sha256((destination/filename).read_bytes()).hexdigest() for filename in records})
    (destination/"float-manifest.json").write_bytes(_json(manifest));_verify_publication(destination,manifest,p)
    return manifest

def _verify_publication(root,manifest,p):
    from merlin.semantic_compiler.target_binding import verify_native_publication
    root=Path(root);expected={"source-request.json","float-schedule.json","float-manifest.json","segments"}
    if root.is_symlink() or not root.is_dir() or {f.name for f in root.iterdir()}!=expected:raise ValueError("exact float publication inventory differs")
    if any((root/n).is_symlink() or not (root/n).is_file() for n in expected-{"segments"}):raise ValueError("exact float metadata absent or linked")
    if (json.loads((root/"float-manifest.json").read_text())!=manifest or manifest.get("schema")!="atlas.exact_float_application.v1" or
        manifest.get("artifact_kind")!="program_set" or manifest.get("engine")!="merlin_native" or manifest.get("request_digest")!=p.source.digest() or
        manifest.get("target_identity")!=p.source.target_identity or manifest.get("launches_per_invocation")!=len(p.launches)):
        raise ValueError("exact float manifest differs from source")
    if json.loads((root/"source-request.json").read_text())!=p.source.record() or json.loads((root/"float-schedule.json").read_text())!=p.record():
        raise ValueError("exact float source or slice schedule differs")
    artifacts=manifest.get("stage_artifacts")
    if not isinstance(artifacts,dict) or set(artifacts)!={"source-request.json","float-schedule.json"}:raise ValueError("exact float metadata map differs")
    for n,digest in artifacts.items():
        if sha256((root/n).read_bytes()).hexdigest()!=digest:raise ValueError("exact float metadata digest differs")
    segments=root/"segments";names=[f"body_{i:03d}" for i in range(len(p.bodies))]
    if segments.is_symlink() or not segments.is_dir() or {f.name for f in segments.iterdir()}!=set(names):raise ValueError("exact float body set differs")
    rows=manifest.get("segments")
    if not isinstance(rows,list) or len(rows)!=len(names):raise ValueError("exact float body count differs")
    for i,(body,n,row) in enumerate(zip(p.bodies,names,rows,strict=True)):
        leaf=segments/n
        if leaf.is_symlink() or not leaf.is_dir() or not isinstance(row,dict) or (leaf/"manifest.json").is_symlink():raise ValueError("exact float body absent or linked")
        detail=json.loads((leaf/"manifest.json").read_text())
        if (row.get("index")!=i or row.get("path")!="segments/"+n or row.get("request_digest")!=body.digest() or
            row.get("manifest_sha256")!=sha256((leaf/"manifest.json").read_bytes()).hexdigest() or row.get("binary_sha256")!=detail.get("binary_sha256")):
            raise ValueError("exact float body identity differs")
        members=detail.get("stage_artifacts")
        if not isinstance(members,dict) or {f.name for f in leaf.iterdir()}!={*members,"manifest.json"}:raise ValueError("exact float leaf inventory differs")
        if any((leaf/n).is_symlink() or not (leaf/n).is_file() for n in members):raise ValueError("exact float leaf file absent or linked")
        verify_native_publication(leaf,detail,engine="merlin_native",request_digest=body.digest(),target_identity=p.source.target_identity)


def prepare_c_package(region,compiled,name,*,target_source,runtime_linkage="separate",input_layout=None):
    from merlin.compile.c_sources import c_source_pair
    from merlin.compile.semantic_application import CompiledCRegion
    from merlin.compile.launch_schedule import make_launch_schedule
    from .c_runtime import RUNTIME_SHARED_FILES
    if runtime_linkage!="separate":raise ValueError("scheduled exact float requires shared runtime linkage")
    p=prepare_plan(region);compiled=Path(compiled);manifest=json.loads((compiled/"float-manifest.json").read_text());_verify_publication(compiled,manifest,p)
    paths=[f"segments/body_{i:03d}" for i in range(len(p.bodies))]
    leaves=[device.prepare_c_package(body,compiled/path,name+f"_body_{i:03d}",target_source=target_source,runtime_linkage=runtime_linkage,input_layout=input_layout)
            for i,(body,path) in enumerate(zip(p.bodies,paths,strict=True))]
    maximum=max(leaf.description["workspace_bytes"] for leaf in leaves)
    result_bytes=p.plan.count*p.plan.output_width;workspace_bytes=maximum+result_bytes;n=len(p.plan.operands);upper=name.upper()
    input_bytes = [p.plan.count * p.plan.width] * n
    mapped_index = None
    if input_layout is not None:
        from math import prod
        physical, view = input_layout
        mapped_index = [operand.id for operand in p.plan.operands].index(view.id)
        if p.plan.operands[mapped_index].type != view.type:
            raise ValueError("exact float input layout differs from its logical operand")
        input_bytes[mapped_index] = prod(physical.type.shape) * p.plan.width
    fields="\n".join(f"  {name}_body_{i:03d}_context body_{i}; {name}_body_{i:03d}_source_context source_{i};" for i in range(len(leaves)))
    bind_calls=[]
    for i in range(len(leaves)):
        inner=name+f"_body_{i:03d}"
        bind_calls.append(f"  if ({inner}_bind(&ctx->body_{i},host,io,workspace,{maximum}u,polls) || {inner}_source_bind(&ctx->source_{i},&ctx->body_{i},0,0)) return {upper}_INVALID;")
    calls=[]
    for call in p.launches:
        i=call["body_index"];inner=name+f"_body_{i:03d}"
        operands=','.join(f"inputs[{j}]" if j == mapped_index else
                         f"(const uint8_t *)inputs[{j}]+{call['input_offset_bytes']}u" for j in range(n))
        if input_layout is not None:
            calls.append(f"  region->source_{i}.input_first={call['input_offset_bytes']//p.plan.width}u;")
        calls.append(f"  {{ const void *body_inputs[{n}]={{{operands}}}; void *body_outputs[1]={{region->workspace+{maximum+call['output_offset_bytes']}u}}; int status={inner}_source_dispatch(&region->source_{i},body_inputs,body_outputs); if(status) return status; }}")
    from .slice_dispatch import render_slice_dispatch
    declarations,implementation=render_slice_dispatch(name,workspace_bytes=workspace_bytes,fields=fields,
        bind_calls=bind_calls,input_bytes=input_bytes,
        output_slices=[(maximum,result_bytes)],launch_calls=calls)
    public={};sources={};files={}
    for i,item in enumerate(leaves):
        for filename,data in item.files.items():
            if filename in RUNTIME_SHARED_FILES:
                if filename in files and files[filename]!=data:raise ValueError("exact float leaf runtimes disagree")
                files[filename]=data
            elif filename.endswith(".h"):public[filename]=data.decode()
            elif filename.endswith(".c"):sources[filename]=data.decode()
            elif filename=="program.bin":files[f"body_{i:03d}.bin"]=data
            else:raise ValueError("unexpected exact float export member")
    sources["float_schedule.c"]=implementation
    files.update({filename:data.encode() for filename,data in c_source_pair(name,public_headers=public,private_headers={},sources=sources,declarations=declarations).items()})
    runs=[]
    for call in p.launches:
        i=call["body_index"]
        if runs and runs[-1]["image"]==i:runs[-1]["count"]+=1
        else:runs.append(dict(image=i,count=1))
    launch_schedule=make_launch_schedule([r["binary_sha256"] for r in manifest["segments"]],[dict(repeat=1,runs=runs)])
    description=dict(kind="exact_float_device",context_type=name+"_context",source_context_type=name+"_source_context",binding_function=name+"_bind",
                     source_binding_function=name+"_source_bind",dispatch_function=name+"_source_dispatch",workspace_bytes=workspace_bytes,source_workspace_bytes=1,
                     input_node_ids=[n.id for n in p.plan.operands],output_node_ids=list(region.outputs),device_launch_schedule=launch_schedule,
                     launches_per_invocation=len(p.launches),float_schedule=p.record(),weight_conversion="none",runtime_arithmetic="none",
                     runtime_linkage="separate",runtime_source="runtime/atlas_host.c",native_manifest_sha256=sha256((compiled/"float-manifest.json").read_bytes()).hexdigest())
    return CompiledCRegion(name+"_source_dispatch",name+".h",tuple(n.id for n in p.plan.operands),tuple(region.outputs),files,RUNTIME_SHARED_FILES,
                           tuple("segments/"+name+"/"+path for path in paths),description)
