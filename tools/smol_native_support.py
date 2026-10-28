"""Check the generated Smol host with exact native MMIO captured replies.

The production graphs, raw adapter and host runtime execute. Native libm and
captured replies replace guest support and device arithmetic respectively.
Inputs are an audited full bundle plus its retained build work directory: main.c,
assets.S and source/ must match host/build.json source and compilation hashes.
Each mode uses a fresh process and an exclusive fixed-address arena mapping.
"""
from __future__ import annotations
import ctypes as ct
import hashlib
import importlib
import inspect
import json
import mmap
from pathlib import Path
import re
import struct
from types import SimpleNamespace
import subprocess
import sys
import time

TOOLS = Path(__file__).absolute().parent
sys.path[:0] = [str(TOOLS),str(TOOLS/'host_iss')]
from atlas_path_guard import checked
from atlas_tooling import sha256
from bundle import EPILOGUE
from full_bundle import FullBundle
from result_lines import verify as verify_result_lines

STEMS = ('smol_host_ops','smol_host_attention','smol_host_softmax','smol_host_rope','smol_host_controls',
         'host_matrix_fp8pack','smol_policy_ops','smol_policy_block','smol_policy_graph',
         'smol_vision_ops','smol_vision_graph','smol_policy_raw','smol_vision_attention','atlas_host','smol_full_driver')
SCOPE = 'native generated host and production raw adapter with captured MMIO replies; no device execution'


def require(value, message):
    if not value: raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def pin(pins, path, expected=None):
    path = checked(path); actual = sha256(path)
    require(expected is None or actual == expected,'changed source '+str(path))
    require(str(path) not in pins or pins[str(path)] == actual,'conflicting source '+str(path))
    pins[str(path)] = actual
    return path


def bookends(pins):
    for name,expected in pins.items(): require(sha256(checked(name)) == expected,'changed bookend '+name)


def inputs(args):
    bundle = FullBundle(args.bundle,'smol-full')
    build = bundle.build
    require(build.get('schema') == 'atlas.smol_full_host_build.v1'
            and build.get('status') == 'BUILT_AUDITED_ONLY' and build.get('actual_returncode') == 0
            and build.get('source_bookends_verified') is True,'complete admitted Smol build required')
    data = json.loads(bundle.read(bundle.provenance['contract']))
    require(bundle.provenance['contract_sha256'] == digest(bundle.read(bundle.provenance['contract']))
            and data['kernels'] == bundle.kernels and bundle.kernel_count == 2715
            and data['projection_calls'] == 1421 and len(data['reference_buffers']) == 47,
            'full Smol contract extent/binding')
    require(len(build.get('result_progress_launches',[])) == 212,'exact progress inventory required')
    return bundle,data


def validate_static(args,bundle):
    directory = checked(args.validator_tools)
    sys.path[:0] = [str(directory),str(directory/'host_iss')]
    for name,relative in (('atlas_program','atlas_program.py'),('run_model_tile','run_model_tile.py'),
                          ('smol_full_plan','smol_full_plan.py'),('smol_full_contract','host_iss/smol_full_contract.py')):
        module = importlib.import_module(name)
        require(checked(module.__file__) == directory/relative,'foreign validator provider '+name)
    module = importlib.import_module('smol_full_contract')
    require(checked(module.__file__) == directory/'host_iss/smol_full_contract.py','foreign full validator')
    assemble = None
    if args.model_root:
        from run_model_tile import configure_model
        _,_,assemble,_ = configure_model(args.model_root)
        from run_model_tile import model_binding
        args.model_binding = model_binding(args.model_root)
    from elf_image import Image
    parameters = inspect.signature(module.validate_bundle).parameters
    args.validator_binding = {str(checked(mod.__file__)):sha256(checked(mod.__file__))
        for name,mod in list(sys.modules.items()) if getattr(mod,'__file__',None)
        and (name.startswith(('atlas_','smol_full','chipyard.'))
             or name in ('run_model_tile','bundle','full_bundle','block_bundle','elf_image','placement'))}
    from atlas_program import code_sources
    for relative in code_sources(traffic_reuse=True):
        pin(args.validator_binding,directory/relative)
    if 'assemble' in parameters:
        module.validate_bundle(bundle,Image(bundle.read('host/host.elf')),assemble=assemble)
    else:
        module.validate_bundle(bundle,Image(bundle.read('host/host.elf')))
    for name,provider in list(sys.modules.items()):
        if name.startswith('atlas_program_'):
            require(checked(provider.__file__) == directory/(name+'.py'),'foreign public generator '+name)
    bookends(args.validator_binding)


def compile_host(args,bundle,data):
    work,out = checked(args.work),checked(args.out)
    require(not out.exists(),'fresh output directory required')
    pins = dict(args.validator_binding)
    bookends(pins)
    build = bundle.build
    expected = build['source_sha256']
    main = pin(pins,work/'main.c',expected.get(str(work/'main.c')))
    require(str(main) in expected,'generated main absent from source bookends')
    assets = pin(pins,work/'assets.S',expected.get(str(work/'assets.S')))
    require(str(assets) in expected,'generated assets absent from source bookends')
    units = [work/'source'/(stem+'.c') for stem in STEMS]
    recorded_units = {checked(command[command.index('-c')+1]) for command in build['commands'] if '-c' in command}
    require(set(units) <= recorded_units and main in recorded_units and assets in recorded_units,
            'native compilation must use the actual guest compilation units')
    for relative,expected_hash in build['compilation_inputs'].items():
        pin(pins,work/relative,expected_hash)
    for path in units:
        require(str(path) in expected,'unbound production compilation unit')
        pin(pins,path,expected[str(path)])
    for name in ('atlas_result_transport.h','result-transport.json'):
        path = work/'source'/name
        require(str(path) in expected,'unbound generated result transport')
        pin(pins,path,expected[str(path)])
    pin(pins,__file__)
    device = pin(pins,TOOLS/'host_iss/smol_native_device.c')
    pin(pins,TOOLS/'host_iss/smol_native_device.h')
    compiler = pin(pins,args.cc)
    for name,module in list(sys.modules.items()):
        path = getattr(module,'__file__',None)
        if path and (name.startswith(('atlas_','smol_full','chipyard.'))
                     or name in ('bundle','full_bundle','block_bundle','elf_image','placement','result_lines')):
            pin(pins,path)
    text = main.read_text()
    fence = '__asm__ volatile("fence rw,rw":::"memory")'
    count = text.count(fence)
    require(count == 2,'expected exactly two generated ISA fences')
    adapted = text.replace(fence,'__sync_synchronize()')
    require('__asm__' not in adapted,'unadmitted generated ISA adaptation')
    out.mkdir(parents=True)
    derived = out/'native-source';derived.mkdir()
    native_main = derived/'main.c';native_main.write_text(adapted)
    uart = derived/'atlas_result_transport.h'
    uart.write_text('''#ifndef ATLAS_RESULT_TRANSPORT_H
#define ATLAS_RESULT_TRANSPORT_H
#include "atlas_result.h"
#include "smol_native_device.h"
typedef struct { uintptr_t base; } atlas_result_uart;
static inline void atlas_result_uart_init(atlas_result_uart *uart) { uart->base=0; }
static inline void atlas_result_uart_putchar(void *context,char value) {
    (void)context; smol_native_putchar(value);
}
#endif
''')
    pin(pins,native_main);pin(pins,uart)
    library = out/'host.so'
    command = [str(compiler),'-std=c11','-O2','-fPIC','-shared','-Wall','-Wextra','-Werror',
               '-ffp-contract=off','-fno-fast-math','-I'+str(derived),'-I'+str(work/'source'),
               '-I'+str(TOOLS/'host_iss'),str(native_main),*[str(p) for p in units],str(assets),str(device),
               '-Wl,--wrap=atlas_host_direct_read32','-Wl,--wrap=atlas_host_direct_write32',
               '-Wl,-z,defs','-Wl,-z,noexecstack','-Wl,-Bsymbolic','-lm','-o',str(library)]
    with (out/'compile.log').open('w') as log:
        result = subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=args.timeout)
    require(result.returncode == 0,'native compilation failed; see compile.log')
    bookends(pins);bundle.verify_bookends()
    if args.model_root:
        from run_model_tile import model_binding
        require(model_binding(args.model_root) == args.model_binding,'changed configured model binding')
    record = dict(schema='atlas.smol_native_compile.v1',scope=SCOPE,source_sha256=pins,
                  command=command,library_sha256=sha256(library),bundle_manifest_sha256=bundle.checksums_sha256,
                  adaptations=dict(generated_fences=count,uart='bounded native capture',
                                   direct_mmio='linker wrapped callbacks',guest_support='excluded; native libc/libm'),
                  source_bookends=True,guest_execution=False,device_execution=False,
                  model_root=str(args.model_root) if args.model_root else None,
                  model_binding=getattr(args,'model_binding',None))
    (out/'compile.json').write_text(json.dumps(record,indent=2)+'\n')
    return record


class Payloads:
    def __init__(self,bundle): self.bundle,self.cache = bundle,{}
    def get(self,name):
        if name not in self.cache:
            require(name in self.bundle.sums,'unlisted reply payload')
            path = self.bundle.path(name)
            require(sha256(path) == self.bundle.sums[name],'changed reply payload')
            with path.open('rb') as stream: mapped = mmap.mmap(stream.fileno(),0,access=mmap.ACCESS_COPY)
            view = (ct.c_ubyte*len(mapped)).from_buffer(mapped)
            self.cache[name] = (mapped,view,ct.addressof(view),len(mapped))
        return self.cache[name][2:]
    def close(self):
        while self.cache:
            _,row = self.cache.popitem();mapped,view,_,_ = row
            del row,view;mapped.close()


class Map(ct.Structure):
    _fields_ = [('imem',ct.c_size_t),('csr',ct.c_size_t),('capacity',ct.c_size_t)]


class Buffer(ct.Structure):
    _fields_ = [('address',ct.c_size_t),('size',ct.c_size_t),('access',ct.c_int)]


class Raw(ct.Structure):
    _fields_ = [(name,ct.c_size_t) for name in ('host','panels','calls','panel_count','call_count',
        'panel_cursor','call_cursor','dram','activation_copy','input_copy','dram_capacity',
        'activation_capacity','input_capacity','max_polls','cycle_delta')]+[
        ('active_buffer',Buffer),('image',ct.c_uint),('failed',ct.c_uint),('result',ct.c_int)]


class Host(ct.Structure):
    _fields_ = [('map',Map),('read',ct.c_size_t),('write',ct.c_size_t),('mmio_context',ct.c_size_t),
        ('coherent',ct.c_int),('sync',ct.c_size_t),('memory_context',ct.c_size_t),
        ('active_buffers',ct.c_size_t),('active_buffer_count',ct.c_size_t),('program_loaded',ct.c_int),
        ('active',ct.c_int),('acknowledged',ct.c_int),('retired',ct.c_uint32),
        ('raw_completion',ct.c_int),('sequence',ct.c_uint32)]


def actual_state(host):
    raw = Raw.in_dll(host,'smol_full_raw');runtime = Host.in_dll(host,'smol_full_host')
    return dict(panels=raw.panel_cursor,calls=raw.call_cursor,failed=raw.failed,result=raw.result,
                sequence=runtime.sequence,active=runtime.active,buffers=runtime.active_buffer_count,
                buffer_pointer=runtime.active_buffers,acknowledged=runtime.acknowledged,image=raw.image,
                graph=list((ct.c_uint32*4).in_dll(host,'smol_full_graph_state')),
                vision=list((ct.c_uint32*8).in_dll(host,'smol_full_vision_states')))


def check_state(state,negative=False):
    require(state['panels'] == (2714 if negative else 2715) and state['calls'] == (1420 if negative else 1421)
            and state['failed'] == int(negative) and state['result'] == (6 if negative else 0),
            'actual raw adapter traversal/state')
    require(state['sequence'] == 2715 and not any(state[key] for key in
            ('active','buffers','buffer_pointer','acknowledged')),'actual runtime ownership released')
    require(state['image'] == 1 and state['vision'] == [1,74,1,0]*2,'both actual vision states')
    require(state['graph'] == ([1,9,1272,6] if negative else [1,10,1273,0]),'actual live policy graph state')


def execute(args,bundle,data):
    out = checked(args.out);record = json.loads((out/'compile.json').read_text())
    require(record['bundle_manifest_sha256'] == bundle.checksums_sha256,'compiled bundle binding')
    bookends(record['source_sha256'])
    if record.get('model_root'):
        sys.path[:0] = [str(checked(args.validator_tools))]
        from run_model_tile import configure_model, model_binding
        configure_model(checked(record['model_root']))
        require(model_binding(checked(record['model_root'])) == record['model_binding'],'changed configured model')
    library = out/'host.so';require(sha256(library) == record['library_sha256'],'compiled library changed')
    host = ct.CDLL(str(library))
    callback_type = ct.CFUNCTYPE(ct.c_int,ct.c_uint32,ct.POINTER(ct.c_uint32),ct.c_size_t,ct.c_uint32,ct.c_void_p)
    host.smol_native_configure.argtypes = [ct.c_size_t,ct.c_size_t,ct.c_size_t,callback_type,ct.c_void_p]
    host.smol_native_map.argtypes = [ct.c_size_t,ct.c_size_t]
    for name in ('smol_native_starts','smol_native_error'):
        getattr(host,name).restype = ct.c_uint32
    host.smol_native_output.restype = ct.c_void_p;host.smol_native_output_bytes.restype = ct.c_size_t
    for name in ('smol_native_imem_base','smol_native_csr_base','smol_native_imem_bytes'):
        getattr(host,name).restype = ct.c_size_t
    memory = ct.CDLL(None);memory.memcmp.argtypes = [ct.c_void_p,ct.c_void_p,ct.c_size_t];memory.memcmp.restype = ct.c_int
    arena = bundle.provenance.get('placement',{}).get('arena_base',0x100000000)
    require(ct.sizeof(Raw) == 160 and ct.sizeof(Host) == 112,'native actual ABI sizes')
    raw_context = Raw.in_dll(host,'smol_full_raw');runtime = Host.in_dll(host,'smol_full_host')
    require(raw_context.dram == arena and raw_context.panel_count == 2715 and raw_context.call_count == 1421
            and raw_context.host == ct.addressof(runtime) and not raw_context.panel_cursor
            and not raw_context.call_cursor,'actual initial raw context')
    require(host.smol_native_map(arena,data['dram_bytes']) == 0,'MAP_FIXED_NOREPLACE arena reservation failed')
    payloads = Payloads(bundle);events,errors = [],[]
    def compare(address,name,size):
        expected,count = payloads.get(name);require(count == size,'captured payload size')
        require(memory.memcmp(address,expected,size) == 0,'actual staged bytes differ: '+name)
    def fill(address,size,value):
        require(ct.string_at(address,size) == bytes([value])*size,'actual initial poison/guard differs')
    @callback_type
    def reply(index,program,words,sequence,context):
        try:
            require(index == len(events) and index < 2715 and sequence == index+1,'ordered native start')
            item = data['kernels'][index];m,k,n = item['physical_mkn']
            require(raw_context.panel_cursor == index and raw_context.call_cursor == item['call']
                    and runtime.active == 1 and runtime.sequence == sequence
                    and runtime.active_buffer_count == 1
                    and runtime.active_buffers == ct.addressof(raw_context.active_buffer)
                    and raw_context.active_buffer.address == arena
                    and raw_context.active_buffer.size == item['total_dram_bytes']
                    and raw_context.active_buffer.access == 2,'actual active raw buffer ownership')
            raw = bundle.read(item['original']);wrapped = raw[:-4]+EPILOGUE
            require(ct.string_at(program,words*4) == wrapped,'actual wrapped IMEM differs')
            compare(arena,item['activation'],m*k);compare(arena+m*k,item['weights'],k*n)
            offset,size = item['output_offset'],2*m*n
            require(offset == m*k+k*n+256 and item['total_dram_bytes'] == offset+size+256,'physical staging extent')
            fill(arena+m*k+k*n,256,0xa5);fill(arena+offset,size,0xcd);fill(arena+offset+size,256,0x5a)
            expected,count = payloads.get(item['expected']);require(count == size,'physical reply extent')
            ct.memmove(arena+offset,expected,size)
            corrupted = args.mode == 'negative' and index == 2714
            if corrupted:
                rows = item['logical_mkn'][0]
                require(rows == 50 and m == 64 and n == 32,'negative requires the final padded action panel')
                word = (rows//32)*(n//32)*1024+(rows%32)*16
                address = arena+offset+2*word
                require(ct.c_uint16.from_address(address).value == 0,'captured padded word must be positive zero')
                ct.c_ubyte.from_address(address).value ^= 1
            events.append(dict(index=index,sequence=sequence,wrapped_program_sha256=digest(wrapped),
                               initial_sha256=item['initial_sha256'],all_staged_bytes_exact=True,
                               captured_output_sha256=bundle.sums[item['expected']],corrupted=corrupted))
            return 0
        except BaseException as error:
            errors.append(dict(index=int(index),error=str(error)));return 1
    require(host.smol_native_configure(host.smol_native_imem_base(),host.smol_native_csr_base(),
                                      host.smol_native_imem_bytes(),reply,None) == 0,'native device configuration')
    start = time.monotonic()
    try:
        host.atlas_workload_main()
        output = ct.string_at(host.smol_native_output(),host.smol_native_output_bytes()).decode('utf-8')
        (out/('uart-'+args.mode+'.log')).write_text(output)
        mailbox = list((ct.c_uint64*16).in_dll(host,'atlas_workload_mailbox'))
        tohost = ct.c_uint64.in_dll(host,'tohost').value
        state = actual_state(host)
        (out/('observed-'+args.mode+'.json')).write_text(json.dumps(dict(actual_state=state,
            mailbox=mailbox,tohost=tohost,starts=host.smol_native_starts(),
            protocol_error=host.smol_native_error(),reply_errors=errors,last_event=events[-1:]
            ),indent=2)+'\n')
        check_state(state,args.mode == 'negative')
        require(not errors and host.smol_native_error() == 0,'native reply/protocol failure '+str(errors[:1]))
        require(host.smol_native_starts() == len(events) == 2715,'all native starts required')
        progress = [int(value) for value in re.findall(r'ATLAS_KERNEL_PROGRESS[^\n]*launches_completed=(\d+)',output)]
        require(len(re.findall(r'ATLAS_KERNEL_BEGIN\b',output)) == 1,'one real BEGIN record required')
        expected_progress = bundle.build['result_progress_launches']
        checked_buffers = []
        if args.mode == 'positive':
            verify_result_lines(output.encode('ascii'),bundle.build['result_name'],bundle.kernel_count,
                                progress_launches=expected_progress)
            require(tohost == 1 and mailbox == [0x534d4f4c46554c4c,2,2715,1421,1,10,1,0,0,2715,1421,2715,0,0,0,0],
                    'exact live completion mailbox/HTIF required')
            require(progress == expected_progress and len(progress) == 212,'exact 212 live progress cursors')
            require(len(re.findall(r'ATLAS_KERNEL_PASS\b',output)) == 1 and 'ATLAS_KERNEL_FAIL' not in output,
                    'one PASS and no FAIL required')
            for item in data['reference_buffers']:
                actual = (ct.c_ubyte*item['bytes']).in_dll(host,item['actual_symbol'])
                compare(ct.addressof(actual),item['file'],item['bytes'])
                checked_buffers.append(dict(name=item['name'],sha256=item['sha256'],exact=True))
        else:
            require(tohost != 1 and mailbox[1] == 3 and 'ATLAS_KERNEL_FAIL' in output
                    and 'ATLAS_KERNEL_PASS' not in output,'corruption must cause real host FAIL without PASS')
            require(progress == expected_progress[:-1],'corruption must stop before final step progress')
            require(events[-1]['corrupted'],'negative did not corrupt actual padded output')
        result = dict(status='PASS_NATIVE_CAPTURED_REPLY_'+args.mode.upper(),scope=SCOPE,mode=args.mode,
                      starts=len(events),completed_projection_calls=mailbox[3] if args.mode == 'positive' else None,
                      actual_buffers=checked_buffers,mailbox=mailbox,tohost=tohost,progress=progress,
                      actual_state=state,
                      ledger=events,source_bookends=True,synthetic_counter_is_timing=False,
                      guest_support='native libc/libm; freestanding RV64 helpers excluded',
                      device_execution=False,guest_execution=False,hardware_execution=False,
                      elapsed_seconds=time.monotonic()-start)
        bookends(record['source_sha256']);bundle.verify_bookends()
        if record.get('model_root'):
            require(model_binding(checked(record['model_root'])) == record['model_binding'],'changed configured model')
        (out/('receipt-'+args.mode+'.json')).write_text(json.dumps(result,indent=2)+'\n')
        return result
    finally:
        payloads.close();host.smol_native_unmap()



def predecessor_wrapper():
    """Observe original projection inputs without replacing production graph bodies."""
    return '''#include "smol_policy_raw.h"
typedef int (*predecessor_callback)(unsigned,unsigned,const void*,size_t,size_t,unsigned,unsigned,unsigned);
static predecessor_callback predecessor_observe;
void smol_predecessor_configure(predecessor_callback callback) { predecessor_observe=callback; }
extern int __real_smol_policy_raw_vision(void*,unsigned,unsigned,const uint16_t*,size_t,size_t,uint16_t*,size_t);
extern int __real_smol_policy_raw_graph(void*,unsigned,unsigned,unsigned,unsigned,const void*,size_t,size_t,
    unsigned,void*,size_t);
int __wrap_smol_policy_raw_vision(void *context,unsigned layer,unsigned role,const uint16_t *input,
    size_t rows,size_t k,uint16_t *output,size_t n) {
    smol_policy_raw_context *raw=context;
    if(!predecessor_observe || predecessor_observe((unsigned)raw->call_cursor,0,input,rows,k,7,layer,role))
        return SMOL_RAW_INPUT;
    return __real_smol_policy_raw_vision(context,layer,role,input,rows,k,output,n);
}
int __wrap_smol_policy_raw_graph(void *context,unsigned site,unsigned layer,unsigned role,unsigned dtype,
    const void *input,size_t rows,size_t k,unsigned output_dtype,void *output,size_t n) {
    smol_policy_raw_context *raw=context;
    if(!predecessor_observe || predecessor_observe((unsigned)raw->call_cursor,dtype,input,rows,k,site,layer,role))
        return SMOL_RAW_INPUT;
    return __real_smol_policy_raw_graph(context,site,layer,role,dtype,input,rows,k,output_dtype,output,n);
}
'''


def native_image(path):
    raw = path.read_bytes()
    header = struct.unpack_from('<16sHHIQQQIHHHHHH',raw)
    if header[0][:7] != b'\x7fELF\x02\x01\x01' or header[1:3] != (3,62) or header[11] != 64:
        raise ValueError('native AMD64 shared library required')
    sections = [struct.unpack_from('<IIQQQQIIQQ',raw,header[6]+i*64) for i in range(header[12])]
    symbols = {}
    for section in sections:
        if section[1] != 2: continue
        if section[9] != 24: raise ValueError('native symbol entry size')
        strings = sections[section[6]]
        table = raw[strings[4]:strings[4]+strings[5]]
        for offset in range(section[4],section[4]+section[5],24):
            no,info,other,index,value,size = struct.unpack_from('<IBBHQQ',raw,offset)
            if no:
                name = table[no:table.index(b'\0',no)].decode()
                symbols[name] = dict(address=value,size=size,section=index)
    return SimpleNamespace(symbols=symbols)

