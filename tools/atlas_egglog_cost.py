"""Atlas provider costs from an explicit observed model timing facts file.

The score omits variable DMA latency and common scalar address/control code.
It is an additive shortlist heuristic, not a cycle or scheduling prediction.
Physical compilation remains the allocation authority.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from egglog.deconstruct import get_callable_args, get_literal_value
from atlas_egglog_schema import Value
from merlin.semantic_compiler.rewrite_selection import CostNode, estimate_dag


class TimingCosts:
    def __init__(self, path):
        self.path=Path(path);raw=self.path.read_bytes();self.sha256=hashlib.sha256(raw).hexdigest()
        doc=json.loads(raw)
        self.timing={row['name']:row['first_free_age'] for row in doc['op_timing'] if row['resolved']}
        if any(type(v) is not int or v<0 for v in self.timing.values()):raise ValueError('invalid timing fact')
        self.omitted=('variable DMA memory latency','common scalar address/control instructions')

    def cost(self, op, attrs):
        short=op.removeprefix('virtual_');kind=json.loads(attrs['kind']) if 'kind' in attrs else None
        def unit():return int(attrs['unit'].split(':')[0])
        t=self.timing
        if short=='vpu_binary':return t[{'min':'vpu.pairmin','max':'vpu.pairmax'}.get(kind,'vpu.'+kind)]
        if short=='vpu_unary':return t[{'recip':'vpu.rcp','log2':'vpu.log'}.get(kind,'vpu.'+kind)]
        if short=='vpu_reduce':return t['vpu.'+{'row_sum':'rsum','row_max':'rmax','row_min':'rmin','col_sum':'csum','col_max':'cmax','col_min':'cmin'}[kind]]
        if short=='vpu_splat':return t['vpu.vliAll']
        if short=='xlu_transpose':return t['xlu.transpose']
        if short=='pack_fp8':
            return t['vpu.fp8pack']+t['vlsu.vstore']+t['vlsu.vload']+32*8*(t['scalar_lsu.load']+t['scalar_lsu.store'])+t['scalar_lsu.scale_load']
        if short=='scale_constant':return t['scalar_lsu.scale_load']
        if short=='egg_pack_mxu':return t[f'mxu{unit()}.push_acc_bf16']+t[f'mxu{unit()}.pop_acc_fp8']+t['scalar_lsu.scale_load']
        if short=='mxu_matmul':return t[f'mxu{unit()}.push_weight']+t[f'mxu{unit()}.matmul']+t[f'mxu{unit()}.pop_acc_bf16']
        raise ValueError('unresolved mutable operation timing: '+op)

    def marginal(self, graph, node):
        args=get_callable_args(node,Value.apply)
        if args is None:return 0
        op=get_literal_value(args[0]);attrs=dict(json.loads(get_literal_value(args[1])))
        return self.cost(op,attrs)

    def estimate(self, program, *, pressure_weight=66):
        nodes=[];costs={'fixed-control':0};last={};types={}
        # Match the allocator's disjoint banks: mixed FP8 reserves m0..m31
        # for FP8 and m32..m61 for fifteen BF16 pairs. BF16-only programs
        # can use thirty-one pairs. Scalar reservations depend on PACK.
        mixed_fp8=any('!atlas.virtual_fp8' in op.output_types for op in program.operations)
        has_pack=any(op.op=='virtual_pack_fp8' for op in program.operations)
        capacities={'bf16-pairs':15 if mixed_fp8 else 31,'fp8-registers':32,
                    'scalar-values':9 if has_pack else 17,'acc0':2,'acc1':2,'weight0':2,'weight1':2}
        for op in program.operations:
            attrs=dict(op.attrs);pure=op.pure
            # Effects are preserved; count their known unit occupancy where it
            # exists, leaving variable DMA/control costs explicitly out.
            price=0
            if pure:price=self.cost(op.op,attrs)
            elif op.op.startswith('virtual_mxu_'):
                if op.op=='virtual_mxu_load_weight':name='push_weight';u=int(attrs['unit'].split(':')[0])
                elif op.op in ('virtual_mxu_load_acc_fp8','virtual_mxu_load_acc_bf16'):
                    name='push_acc_'+op.op.rsplit('_',1)[1];u=int(attrs['unit'].split(':')[0])
                elif op.op in ('virtual_mxu_reset','virtual_mxu_accumulate'):
                    name='matmul' if op.op.endswith('reset') else 'matmul_acc';u=int(op.output_types[1][-2])
                elif op.op.startswith('virtual_mxu_readout_'):
                    name='pop_acc_'+op.op.rsplit('_',1)[1];u=int(op.input_types[1][-2])
                else:raise ValueError('unresolved MXU operation')
                price=self.timing[f'mxu{u}.{name}']
            elif op.op in ('virtual_input_bf16','virtual_input_fp8','virtual_dma_load_bf16','virtual_dma_load_fp8'):
                price=self.timing['vlsu.vload']
            elif op.op in ('virtual_output_bf16','virtual_dma_store_bf16','virtual_dma_store_fp8'):
                price=self.timing['vlsu.vstore']
            tag='op-'+str(op.line);costs[tag]=price
            deps=tuple(dict.fromkeys(last[n] for n in op.operands if n in last))
            # Each result depends on the consumed inputs, retaining operand
            # buffers through production. Charge the operation on its first
            # result only; state tokens have zero storage demand.
            for index,(name,type_) in enumerate(zip(op.results,op.output_types)):
                if type_=='!atlas.virtual_bf16':resource=(('bf16-pairs',1),)
                elif type_=='!atlas.virtual_fp8':resource=(('fp8-registers',1),)
                elif type_=='i32':resource=(('scalar-values',1),)
                elif type_.startswith('!atlas.virtual_mxu_acc<'):resource=((f'acc{type_[-2]}',1),)
                elif type_.startswith('!atlas.virtual_mxu_weight<'):resource=((f'weight{type_[-2]}',1),)
                else:resource=()
                nodes.append(CostNode(name,tag if index==0 else 'fixed-control',deps,resource));last[name]=name;types[name]=type_
        if not nodes:raise ValueError('empty cost program')
        return estimate_dag(nodes,(nodes[-1].id,),operation_costs=costs,
            pressure_weights={'bf16-pairs':2*pressure_weight,'fp8-registers':pressure_weight,
                              'scalar-values':self.timing['scalar_lsu.load'],'acc0':pressure_weight,'acc1':pressure_weight,'weight0':pressure_weight,'weight1':pressure_weight},
            capacities=capacities,facts_sha256=self.sha256)
