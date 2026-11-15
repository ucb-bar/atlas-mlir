"""Host C interpreter for verified float machine streams; independent ISA model.

Vector add/sub use FP32 then BF16 chop; multiply and row reduction use
BF16 nearest even, matching the unchanged npu_model primitives.
"""
import ctypes
import gzip
import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from atlas_mlir_native.binding import DRAM_BASE, _run
from atlas_mlir_native.limits import DEFAULT_FUNCTIONAL_INSTRUCTION_LIMIT, functional_instruction_limit

@lru_cache(maxsize=32)
def executor(operation,count,shared=False,execution_limit=DEFAULT_FUNCTIONAL_INSTRUCTION_LIMIT):
    from test_exact_float_device import request
    from atlas_mlir_native.exact_float_device import prepare_plan,machine_program
    p=prepare_plan(request(operation,count,shared));machine=machine_program(p)
    rows=[];kinds={}
    fields=('dst','lhs','rhs','src','base','offset','immediate','reg','dram','size','source','offset_bytes')
    for op,raw in re.findall(r'"atlas\.([a-z_]+)"\([^\n]*?\) [<{]+([^\n]*)[}>]+',machine):
        if op=='start':continue
        attrs=dict((key,string or int(integer)) for key,string,integer in re.findall(r'(\w+) = (?:"([^"]+)"|(-?\d+) : i32)',raw))
        name=op+'.'+str(attrs.get('kind',attrs.get('direction',attrs.get('mode',''))))
        if name not in kinds:kinds[name]=len(kinds)
        rows.append('{'+','.join(str(x) for x in [kinds[name],*(attrs.get(f,0) for f in fields)])+'}')
    cases=[]
    for name,kind in kinds.items():
        op,mode=name.split('.',1);c=''
        if op=='alu_reg':
            expressions={'add':'a+b','sub':'a-b','and':'a&b','or':'a|b','xor':'a^b','sll':'a<<(b&31)','srl':'a>>(b&31)','sltu':'a<b','slt':'(int32_t)a<(int32_t)b'}
            c=f'x[in.dst]={expressions[mode]};'
        elif op=='alu_imm':
            expressions={'addi':'a+b','andi':'a&b','ori':'a|b','xori':'a^b','slli':'a<<(b&31)','srli':'a>>(b&31)','sltiu':'a<b','slti':'(int32_t)a<(int32_t)b'}
            c=f'a=x[in.src];b=(uint32_t)in.immediate;x[in.dst]={expressions[mode]};'
        elif op=='upper':c='x[in.dst]=(uint32_t)in.immediate<<12;'
        elif op=='scalar_load':
            width={'lw':4,'lhu':2,'lbu':1}[mode];c=f'address=x[in.base]+in.offset;if(address>{1<<20}-{width})return 91;x[in.dst]=0;memcpy(&x[in.dst],vm+address,{width});'
        elif op=='scalar_store':
            width={'sw':4,'sh':2,'sb':1}[mode];c=f'address=x[in.base]+in.offset;if(address>{1<<20}-{width})return 92;memcpy(vm+address,&x[in.src],{width});'
        elif op=='dma':
            c=f'address=x[in.dram]-DRAM_BASE;if(address>{p.workspace_bytes}u-x[in.size] || x[in.reg]*4u>{1<<20}-x[in.size])return 93;'
            c+=('memcpy(vm+x[in.reg]*4u,dram+address,x[in.size]);' if mode=='load' else 'memcpy(dram+address,vm+x[in.reg]*4u,x[in.size]);')
        elif op=='vload':c='memcpy(tensor[in.dst],vm+4u*(x[in.base]+in.offset),1024);'
        elif op=='vstore':c='memcpy(vm+4u*(x[in.base]+in.offset),tensor[in.src],1024);'
        elif op=='vli':c='for(unsigned i=0;i<1024;i++)((uint16_t*)tensor[in.dst])[i]=(uint16_t)in.immediate;'
        elif op=='vpu_binary':
            expression={'add':'a+b','sub':'a-b','mul':'a*b','max':'fmaxf(a,b)'}[mode]
            c=f'for(unsigned i=0;i<1024;i++){{float a=decode(((uint16_t*)tensor[in.lhs])[i]),b=decode(((uint16_t*)tensor[in.rhs])[i]);temporary[i]={"rne" if mode=="mul" else "chop"}({expression});}}memcpy(tensor[in.dst],temporary,2048);'
            if mode=='max':c='for(unsigned i=0;i<1024;i++){uint16_t a=((uint16_t*)tensor[in.lhs])[i],b=((uint16_t*)tensor[in.rhs])[i];unsigned ak=a&32768?a^65535:a^32768,bk=b&32768?b^65535:b^32768;temporary[i]=ak>bk?a:b;}memcpy(tensor[in.dst],temporary,2048);'
        elif op=='vpu_unary':
            table=0 if mode=='tanh' else 1
            c=f'for(unsigned i=0;i<1024;i++)temporary[i]=unary_tables[{table}][((uint16_t*)tensor[in.src])[i]];memcpy(tensor[in.dst],temporary,2048);'
        elif op=='vpu_reduce':
            assert mode=='row_sum'
            c='for(unsigned r=0;r<32;r++){float v[32];for(unsigned i=0;i<32;i++)v[i]=decode(((uint16_t*)tensor[in.src])[i/16*512+r*16+i%16]);for(unsigned n=32;n>1;n/=2)for(unsigned i=0;i<n/2;i++)v[i]=v[2*i]+v[2*i+1];uint16_t code=rne(v[0]);for(unsigned h=0;h<2;h++)for(unsigned i=0;i<16;i++)temporary[h*512+r*16+i]=code;}memcpy(tensor[in.dst],temporary,2048);'
        elif op=='branch':
            condition={'beq':'a==b','bne':'a!=b','bgeu':'a>=b','bltu':'a<b','bge':'(int32_t)a>=(int32_t)b','blt':'(int32_t)a<(int32_t)b'}[mode]
            c=f'if({condition})next=pc+in.offset_bytes/2;'
        elif op=='jump':c='next=pc+in.offset/2;'
        elif op=='trap':c='pc=sizeof(program)/sizeof(*program);continue;'
        elif op not in ('delay','dma_wait','dma_config','csr'):raise ValueError(name)
        cases.append(f'case {kind}: {c} break;')
    code='''#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
struct instruction {int kind,dst,lhs,rhs,src,base,offset,immediate,reg,dram,size,source,offset_bytes;};
static float decode(uint16_t code){uint32_t bits=(uint32_t)code<<16;float f;memcpy(&f,&bits,4);return f;}
static uint16_t chop(float f){uint32_t bits;memcpy(&bits,&f,4);return bits>>16;}
static uint16_t rne(float f){uint32_t bits;memcpy(&bits,&f,4);if((bits&0x7f800000u)==0x7f800000u)return bits>>16;return (bits+32767u+((bits>>16)&1u))>>16;}
static uint16_t unary_tables[2][65536];
void initialize(const void *tanh,const void *recip){memcpy(unary_tables[0],tanh,131072);memcpy(unary_tables[1],recip,131072);}
static const struct instruction program[]={'''+','.join(rows)+'''};
int execute(const void *left,const void *right,void *output,unsigned *status){
 uint32_t x[32]={0};unsigned char *vm=calloc(1,1u<<20),(*tensor)[1024]=calloc(256,1024),*dram=malloc(WORKSPACE);
 uint16_t temporary[1024];if(!vm||!tensor||!dram)return 98;memset(dram,0xa5,WORKSPACE);
 memset(dram,0,INPUT_COUNT*INPUT_BYTES);
 for(unsigned i=0;i<COUNT;i+=32){unsigned lanes=COUNT-i<32?COUNT-i:32;memcpy(dram+i*WIDTH,(const char*)left+i*WIDTH,lanes*WIDTH);
 if(INPUT_COUNT==2)memcpy(dram+INPUT_BYTES+i*WIDTH,(const char*)right+i*WIDTH,lanes*WIDTH);}
 unsigned pc=0,steps=0;
 while(pc<sizeof(program)/sizeof(*program)){if(++steps>INSTRUCTION_LIMIT)return 94;
 struct instruction in=program[pc];uint32_t a=x[in.lhs],b=x[in.rhs],address=0;int next=(int)pc+1;
 switch(in.kind){'''+''.join(cases)+'''}
 x[0]=0;pc=next;}
 memcpy(status,dram+STATUS,4);
 for(unsigned i=0;i<COUNT;i+=32){unsigned lanes=COUNT-i<32?COUNT-i:32;memcpy((char*)output+i*OUT_WIDTH,dram+OUTPUT+i*OUT_WIDTH,lanes*OUT_WIDTH);}
 free(vm);free(tensor);free(dram);return steps;}
'''
    constants=dict(INSTRUCTION_LIMIT=execution_limit,DRAM_BASE=DRAM_BASE,WORKSPACE=p.workspace_bytes,COUNT=count,INPUT_COUNT=len(p.operands),INPUT_BYTES=p.input_bytes,WIDTH=p.width,OUT_WIDTH=p.output_width,OUTPUT=p.output_address-DRAM_BASE,STATUS=p.status_address-DRAM_BASE)
    code=''.join(f'#define {key} {value}u\n' for key,value in constants.items())+code
    temporary=tempfile.TemporaryDirectory(prefix='float-isa-c-');directory=Path(temporary.name)
    (directory/'interpreter.c').write_text(code)
    subprocess.run(['cc','-std=c11','-O2','-shared','-fPIC',str(directory/'interpreter.c'),'-lm','-o',str(directory/'interpreter.so')],check=True,capture_output=True)
    library=ctypes.CDLL(str(directory/'interpreter.so'))
    if any(name.startswith('vpu_unary') for name in kinds):
        model=Path(os.environ['ATLAS_NPU_MODEL_ROOT'])
        tables=[ctypes.create_string_buffer(gzip.decompress((model/'npu_model/configs/data'/f'{name}.bin.gz').read_bytes())) for name in ('tanh','rcp')]
        library.initialize.argtypes=[ctypes.c_void_p,ctypes.c_void_p];library.initialize(*tables)
    function=library.execute
    function.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.POINTER(ctypes.c_uint)]
    tools=Path(os.environ.get('ATLAS_OOT_BIN_DIR',Path(__file__).resolve().parents[1]/'build/bin'))
    program=json.loads(_run([str(tools/'atlas-emit'),'--program-json','-'],input_text=machine))
    binary=b''.join(row['word_u32'].to_bytes(4,'little') for row in program['instructions'])
    function.program_sha256=hashlib.sha256(binary).hexdigest()
    return p,function,temporary

def execute(source,raw):
    limit=functional_instruction_limit()
    p,fn,_=executor(source.nodes[-1].op,source.nodes[-1].type.shape[0],len(source.nodes)==2,limit)
    inputs=[ctypes.create_string_buffer(a) for a in raw];output=ctypes.create_string_buffer(p.count*p.output_width);status=ctypes.c_uint()
    steps=fn(inputs[0],inputs[-1],output,ctypes.byref(status))
    if steps<0 or steps>limit:raise RuntimeError(steps)
    if steps in (91,92,93,94,98):raise RuntimeError(('ISA interpreter fault',steps))
    return output.raw,status.value,steps


def provenance():
    root=Path(__file__).resolve().parents[1]
    return dict(python_executable=sys.executable,python_version=sys.version,
                compiler_pure_python_site=os.environ.get('ATLAS_COMPILER_PURE_PYTHON_SITE'),git_head=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),
                git_status_porcelain=subprocess.check_output(['git','-C',str(root),'status','--porcelain'],text=True).splitlines())
