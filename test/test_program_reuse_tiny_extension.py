"""Independent Tiny scalar flow, DMA readiness and operand-order proofs."""
from collections import Counter
import hashlib
from pathlib import Path
import re
import sys
import unittest
sys.path.insert(0, str(Path(__file__).absolute().parents[1] / 'tools'))
import atlas_program_reuse_tiny as resident
import atlas_program_reuse_tiny_breadth as prefetch

NEW_SHAPES = ((128,2048,1536),(32,5632,1024),(32,2048,1536),
              (128,2048,256),(32,2048,256),(32,2048,1280),(128,2048,1280))

def implementation(m,k,n):
    return resident if (m,k,n) in resident.SHAPES else prefetch

def assembly(m,k,n,offset,exponent):
    return implementation(m,k,n).assembly(m,k,n,offset,exponent)

def dma_load_bytes(m,k,n):
    return implementation(m,k,n).dma_load_bytes(m,k,n)

def layout(m,k,n):
    if (m,k,n) in resident.SHAPES:
        return dict(a=[(262144,262144)],b_buffers=[(0,65536)],output=(65536,2048),rows=4)
    return prefetch.allocation(m,k,n)

def prove(source,m,k,n,exponent):
    lines=source.splitlines(); labels={s[:-1]:i for i,s in enumerate(lines) if s.endswith(':')}
    regs,memory,matrix,accum={'x0':0},{},{},{}
    pending={}; loads=[]; stores={}; counts=Counter(); scaled=set(); pc=0; weight=None
    def tile(a):
        assert a%1024==0 and 0<=a<=524288-1024
        assert a//262144==(a+1023)//262144
        return a
    def ready(a,size=1024):
        assert all(a+size<=start or end<=a for start,end in pending.values())
    def addr(s):
        return tile(4*regs[re.fullmatch(r'0\((x\d+)\)',s).group(1)])
    for _ in range(3000000):
        op,*args=lines[pc].replace(',','').split();pc+=1
        if op.endswith(':'):continue
        counts[op]+=1
        if op=='dma.wait':
            assert args[0] in ('ch0','ch1');pending.pop(args[0],None)
        elif op in ('nop','delay'):pass
        elif op=='dma.config':assert regs[args[0]]==1 and args[1] in ('ch0','ch1')
        elif op=='li':regs[args[0]]=int(args[1],0)
        elif op in ('add','addi'):
            regs[args[0]]=regs[args[1]]+(regs[args[2]] if op=='add' else int(args[2]))
            assert 0<=regs[args[0]]<2**32
        elif op in ('beq','jal'):
            assert lines[pc]=='nop'
            if op=='jal' or regs[args[0]]==regs[args[1]]:pc=labels[args[-1]]
        elif op=='dma.load':
            dst,src,size=(regs[a] for a in args[:3]);dst*=4;ch=args[3]
            assert ch not in pending and size in (1024,4096) and src%1024==0
            assert 0<=src and src+size<=m*k+k*n
            assert dst//262144==(dst+size-1)//262144
            ready(dst,size)
            for delta in range(0,size,1024):memory[tile(dst+delta)]=('input',src+delta,False)
            pending[ch]=(dst,dst+size);loads.append((dst,src,size))
        elif op=='vload':
            a=addr(args[1]);ready(a);matrix[args[0]]=memory[a]
        elif op=='vstore':
            a=addr(args[1]);ready(a);memory[a]=matrix[args[0]]
        elif op=='vtrpose.xlu':
            kind,origin,tr=matrix[args[1]];assert kind=='input';matrix[args[0]]=(kind,origin,not tr)
        elif op=='vmatpush.weight.mxu0':
            assert args[0]=='w0';weight=matrix[args[1]]
        elif op in ('vmatmul.mxu0','vmatmul.acc.mxu0'):
            acc=args[0];assert acc in ('acc0','acc1') and args[2]=='w0'
            kind,a,at=matrix[args[1]];wkind,b,bt=weight
            assert kind==wkind=='input' and not at and bt
            row,inner=divmod(a//1024,k//32);col,binner=divmod((b-m*k)//1024,k//32)
            assert 0<=a<m*k<=b<m*k+k*n and inner==binner
            assert 0<=row<m//32 and 0<=col<n//32
            if op=='vmatmul.mxu0':
                assert inner==0 and acc not in accum;accum[acc]=(row,col,1)
            else:
                assert accum[acc]==(row,col,inner);accum[acc]=(row,col,inner+1)
        elif op=='vmatpop.bf16.acc.mxu0':
            row,col,inner=accum.pop(args[1]);assert inner==k//32 and args[0]=='m2'
            matrix['m2'],matrix['m3']=('output',row,col,0),('output',row,col,1)
        elif op=='vli.all':
            assert args==['m4',str((exponent+127)<<7)];matrix['m4']=exponent
        elif op=='vmul.bf16':
            assert args==['m2','m2','m4'] and matrix['m4']==exponent
            kind,row,col,half=matrix['m2'];assert kind=='output' and (row,col) not in scaled
            scaled.add((row,col))
        elif op=='dma.store':
            dst,src,size=(regs[a] for a in args[:3]);src*=4;ch=args[3]
            assert ch not in pending and size==2048;ready(src,size)
            first,second=memory[tile(src)],memory[tile(src+1024)]
            assert first[0]=='output' and first[3]==0 and second==(*first[:3],1)
            _,row,col,_=first
            assert dst==m*k+k*n+256+(row*(n//32)+col)*2048
            if exponent is not None:
                assert (row,col) in scaled;scaled.remove((row,col))
            assert (row,col) not in stores;stores[row,col]=dst;pending[ch]=(src,src+size)
        elif op=='ecall':
            assert not pending and not scaled and not accum and len(stores)==m*n//1024
            repeats=2 if (m,k,n)==(128,5632,1024) else 1
            origins=[src+d for _,src,size in loads for d in range(0,size,1024)]
            expected=list(range(0,m*k,1024))+list(range(m*k,m*k+k*n,1024))*repeats
            assert Counter(origins)==Counter(expected)
            assert counts['vmatmul.mxu0']==m*n//1024
            assert counts['vmatmul.acc.mxu0']==m*n//1024*(k//32-1)
            assert counts['vmatpop.bf16.acc.mxu0']==m*n//1024
            assert counts['vmul.bf16']==(m*n//1024 if exponent is not None else 0)
            return dict(load_bytes=sum(s for _,_,s in loads),store_bytes=len(stores)*2048,counts=dict(counts),loads=loads)
        else:raise AssertionError(op)
    raise AssertionError('nontermination')

OLD_PROGRAMS = [{'module': 'atlas_program_reuse_tiny', 'shape': [128, 2048, 2048], 'exponent': None, 'assembly_sha256': '8dcfb90a3478048a1389ed4161942ae01ab2789a3fabddfcb7b0e8575bfa8585', 'dma_load_bytes': 4456448}, {'module': 'atlas_program_reuse_tiny', 'shape': [128, 2048, 2048], 'exponent': -126, 'assembly_sha256': '26e357c4b2a2b00fb59ff9369d2304af9c35b2658669423c4e1c6dd04f82b900', 'dma_load_bytes': 4456448}, {'module': 'atlas_program_reuse_tiny', 'shape': [128, 2048, 2048], 'exponent': -13, 'assembly_sha256': 'ec1fb64a903a6f803b993e798838a69b87a42ccd90bbeedc69b8570b97e805d3', 'dma_load_bytes': 4456448}, {'module': 'atlas_program_reuse_tiny', 'shape': [128, 2048, 2048], 'exponent': 127, 'assembly_sha256': '704e53af779605be8b48cfbed8ba98d1471a07995115ab702fe412e710ae3102', 'dma_load_bytes': 4456448}, {'module': 'atlas_program_reuse_tiny_breadth', 'shape': [128, 5632, 1024], 'exponent': None, 'assembly_sha256': '7ea5863fc3af0be60477f5721d9b69aef4b8793e384c27eb5b6f823de8742268', 'dma_load_bytes': 12255232}, {'module': 'atlas_program_reuse_tiny_breadth', 'shape': [128, 5632, 1024], 'exponent': -126, 'assembly_sha256': '17301fffaff07b6b541567b61928d0a6536c246a2a9afe423ac7f1da2027b332', 'dma_load_bytes': 12255232}, {'module': 'atlas_program_reuse_tiny_breadth', 'shape': [128, 5632, 1024], 'exponent': -13, 'assembly_sha256': 'e02189db3adb16921c6be99b02cd65cf7283a067eeb00b266268d786b17d0b41', 'dma_load_bytes': 12255232}, {'module': 'atlas_program_reuse_tiny_breadth', 'shape': [128, 5632, 1024], 'exponent': 127, 'assembly_sha256': '13b1d97e0a70157a1f903926757ab08fcc94ffc89ca7cee8fffa160e42451474', 'dma_load_bytes': 12255232}, {'module': 'atlas_program_reuse_tiny_breadth', 'shape': [32, 2048, 2048], 'exponent': None, 'assembly_sha256': '14492c7f3473f1e02a90529eca84084b955e028e6ee8c44839ddbbf55db9b42e', 'dma_load_bytes': 4259840}, {'module': 'atlas_program_reuse_tiny_breadth', 'shape': [32, 2048, 2048], 'exponent': -126, 'assembly_sha256': 'c8eae3e9f4d8342ee23f1a772f3c8d3d9588858b0c59258e6b985a4baa0e0418', 'dma_load_bytes': 4259840}, {'module': 'atlas_program_reuse_tiny_breadth', 'shape': [32, 2048, 2048], 'exponent': -13, 'assembly_sha256': 'afc4efdd002d7c7b70264ead199951bc79ee1a1608d418ef2dd2b970f61348ef', 'dma_load_bytes': 4259840}, {'module': 'atlas_program_reuse_tiny_breadth', 'shape': [32, 2048, 2048], 'exponent': 127, 'assembly_sha256': '7265620e1343524b3d1e4931129f5fdc4ae136c6b25a8910494c116281768c43', 'dma_load_bytes': 4259840}, {'module': 'atlas_program_reuse_tiny_breadth', 'shape': [32, 2048, 1024], 'exponent': None, 'assembly_sha256': '6aa77c94e9491be353fd84807f2368da084f3be9599eb1949b49e11d015df59b', 'dma_load_bytes': 2162688}, {'module': 'atlas_program_reuse_tiny_breadth', 'shape': [32, 2048, 1024], 'exponent': -126, 'assembly_sha256': 'd8170bc0c2546865068937e8e1ac90b59e3f5d5f5a41cbbbf5986949da70cad9', 'dma_load_bytes': 2162688}, {'module': 'atlas_program_reuse_tiny_breadth', 'shape': [32, 2048, 1024], 'exponent': -13, 'assembly_sha256': '2744b31c28fb5326198625d68ad43bb1a730e206e01e10a1740fbb809e1318f6', 'dma_load_bytes': 2162688}, {'module': 'atlas_program_reuse_tiny_breadth', 'shape': [32, 2048, 1024], 'exponent': 127, 'assembly_sha256': 'aee531047a652e578f7581b51112b37efc16933523a106feea52501e5761069b', 'dma_load_bytes': 2162688}]

class TinyExtensionTests(unittest.TestCase):
    def test_all_new_operand_order_dma_readiness_and_outputs(self):
        for m,k,n in NEW_SHAPES:
            with self.subTest(shape=(m,k,n)):
                r=prove(assembly(m,k,n,m*k+k*n+256,-13),m,k,n,-13)
                self.assertEqual(r['load_bytes'],dma_load_bytes(m,k,n))
                self.assertEqual(r['store_bytes'],2*m*n)
    def test_all_old_programs_unchanged(self):
        for record in OLD_PROGRAMS:
            mod=resident if record['module']=='atlas_program_reuse_tiny' else prefetch
            m,k,n=record['shape'];s=mod.assembly(m,k,n,m*k+k*n+256,record['exponent'])
            self.assertEqual(hashlib.sha256(s.encode()).hexdigest(),record['assembly_sha256'])
            self.assertEqual(mod.dma_load_bytes(m,k,n),record['dma_load_bytes'])
    def test_bank_regions(self):
        for shape in NEW_SHAPES:
            p=layout(*shape);used=set()
            for start,size in p['a']+p['b_buffers']+[p['output']]:
                self.assertLessEqual(start+size,524288)
                self.assertEqual(start//262144,(start+size-1)//262144)
                blocks=set(range(start,start+size,1024));self.assertFalse(used&blocks);used|=blocks
    def test_negative_controls(self):
        for m,k,n in ((128,2048,256),(32,5632,1024)):
            s=assembly(m,k,n,m*k+k*n+256,-13)
            for old,new in [('vtrpose.xlu m1, m1','nop'),('vmatmul.acc.mxu0','vmatmul.mxu0'),('vstore m3, 0(x9)','vstore m2, 0(x9)'),('dma.wait ch0','nop'),('vmul.bf16 m2, m2, m4','nop'),('add x23, x23, x22','add x23, x23, x8')]:
                # Singleton has only one row: output row stride does not execute between outputs.
                if old=='add x23, x23, x22' and m==32:continue
                with self.subTest(shape=(m,k,n),mutation=old),self.assertRaises((AssertionError,KeyError)):
                    prove(s.replace(old,new),m,k,n,-13)
        m,k,n=32,5632,1024;s=assembly(m,k,n,m*k+k*n+256,-13)
        with self.assertRaises(AssertionError):prove(s.replace('dma.wait ch1','nop'),m,k,n,-13)
    def test_one_static_vli_patch(self):
        for m,k,n in NEW_SHAPES:
            original=assembly(m,k,n,m*k+k*n+256,0).splitlines();sites=[i for i,s in enumerate(original) if s.startswith('vli.all')]
            self.assertEqual(len(sites),1)
            for exponent in (-126,-13,127):
                expected=original.copy();expected[sites[0]]=f'vli.all m4, {(exponent+127)<<7}'
                self.assertEqual(expected,assembly(m,k,n,m*k+k*n+256,exponent).splitlines())
    def test_strict_shapes(self):
        self.assertEqual(set(resident.SHAPES),{(128,2048,n) for n in (256,1280,1536,2048)})
        self.assertEqual(set(prefetch.SHAPES),{(128,5632,1024),(32,2048,2048),(32,2048,1024),(32,5632,1024),(32,2048,1536),(32,2048,256),(32,2048,1280)})
        for mod in (resident,prefetch):
            for shape in ((128,2048,512),(32,5632,256),(32.,2048,256)):
                with self.assertRaises(ValueError):mod.dma_load_bytes(*shape)
if __name__=='__main__':unittest.main()
