"""Execute exactly the first ten relocated ResNet launches with the real model.

This diagnostic stops at the eleventh START instruction before forwarding it.
It cannot qualify the remaining launches, final logits, a SoC or hardware.
"""
import argparse
from contextlib import suppress
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import resource
import signal
import struct
import subprocess
import sys
import time

TOOLS=Path(__file__).absolute().parents[1]
sys.path[:0]=[str(TOOLS),str(TOOLS/'host_iss')]
from atlas_path_guard import checked
from atlas_tooling import sha256,sha as digest
from atlas_host_layout import ATLAS_CSR_BASE,ATLAS_CSR_EXEC_CONTROL,HEADER,HEADER_SHA256
from bundle import Bundle,runtime_program
from elf_image import Image,mmio_sites
from gdb_remote import Remote
from model_device import Device
from run_model_tile import configure_model,model_binding
from run import resnet_stop_sites,mmio_access_cap,interrupted

COUNT=10
PHYSICAL=dict(arena_base=0x90000000,dram_base=0x80000000,dram_bytes=0x80000000)


def require(value,message):
    if not value:raise ValueError(message)


def stop_before_start(device,address,value):
    if address != ATLAS_CSR_BASE+ATLAS_CSR_EXEC_CONTROL:return False
    require(value==1,'only START is admitted')
    require(len(device.completed)<=COUNT,'smoke already exceeded first ten launches')
    return len(device.completed)==COUNT


def store_or_stop(device,address,value):
    if stop_before_start(device,address,value):return True
    device.write32(address,value)
    return False


def uart_prefix(raw,name):
    records=[]
    for line in raw.decode('ascii').splitlines():
        if not line.startswith('ATLAS_KERNEL_'):continue
        fields=line.split();kind=fields[0].removeprefix('ATLAS_KERNEL_');values={}
        for field in fields[1:]:
            require('=' in field,'malformed UART field')
            key,value=field.split('=',1);require(key not in values,'duplicate UART field');values[key]=value
        require(values.get('name')==name and values.get('launches_expected')=='54','UART workload identity')
        index=len(records)
        require(kind==('BEGIN' if index==0 else 'PROGRESS'),'smoke requires BEGIN/progress without PASS/FAIL')
        require(values.get('launches_completed')==str(index),'UART first-ten cursor')
        require(index==0 or bool(values.get('boundary')),'UART progress boundary')
        records.append(dict(kind=kind,**values))
    require(len(records)==COUNT+1,'exact BEGIN plus ten progress records required')
    return records


def verify_ledgers(device):
    require(len(device.completed)==COUNT,'exactly ten completed real launches required')
    for index,row in enumerate(device.completed):
        _,words,paths=device.programs[index]
        initial=device.bundle.read(paths['initial']);expected=device.bundle.read(paths['expected'])
        dma=dict(paths['requested_dma']);dma['response_beats']=(dma['load_bytes']+dma['store_bytes'])//32
        require(row['index']==index and row['sequence']==index+1 and row['reset_count']==1
            and row['exact'] is True and row['drained'] is True and type(row['cycles']) is int
            and 0<row['cycles']<=device.cycle_cap,'ordered exact drained real-model ledger')
        require(row['program_sha256']==digest(words) and row['host_input_sha256']==digest(initial)
            and row['actual_output_sha256']==row['expected_output_sha256']==digest(expected)
            and row['observed_dma']==dma,'actual program/input/output/DMA binding')
    state=device.core.arch_state
    require(state.halted and state.halt_reason=='ecall'
        and not any(exu.has_in_flight for exu in device.core.exus)
        and not any(state.check_flag(channel) for channel in range(8)),'tenth real model fully drained')
    require(state.read_csrf(0xC10)==0 and state.read_csrf(0xC11)==11,'pending eleventh START marker/sequence')


def pin_helpers():
    pins={str(checked(HEADER)):HEADER_SHA256}
    for module in tuple(sys.modules.values()):
        name=getattr(module,'__file__',None)
        if name and Path(name).is_absolute() and Path(name).is_relative_to(TOOLS):
            path=checked(name);pins[str(path)]=sha256(path)
    return pins


def bookends(pins):
    require(all(sha256(checked(p))==h for p,h in pins.items()),'changed source/input bookend')


def run(args):
    out=checked(args.out);out.mkdir(parents=True,exist_ok=False)
    remaining=(args.stop_by_utc-datetime.now(timezone.utc)).total_seconds()
    require(remaining>0 and args.wall_seconds>0,'positive wall cap before absolute stop required')
    seconds=min(args.wall_seconds,remaining);deadline=time.monotonic()+seconds;started=time.monotonic()
    receipt=dict(status='FAIL_RESNET_RELOCATED_FIRST10_SMOKE',scope=__doc__,command=[sys.executable,*sys.argv],
        launch_range=[0,COUNT],whole_model_qualified=False,final_logits_checked=False,hardware_execution=False,
        host_elf_executed=False,model_device_execution=False,wall_seconds_requested=args.wall_seconds,
        effective_wall_seconds=seconds,absolute_stop_utc=args.stop_by_utc.isoformat(),
        runner_pid=os.getpid(),cpu_affinity=sorted(os.sched_getaffinity(0)))
    process=remote=device=None;pins={}
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    try:
        pins=pin_helpers();bundle=Bundle(args.bundle_root)
        require(bundle.placement==PHYSICAL,'R3 requires specified physical2GiB DRAM and0x90000000 arena')
        build=bundle.build
        require((build['uart']['kind'],build['uart']['base'])==('ns16550',0x10000000),'explicit QEMU NS16550 ELF required')
        require(build['result_progress_launches']==list(range(1,55)),'full ResNet progress inventory')
        elf,qemu=checked(args.elf),checked(args.qemu)
        require(sha256(elf)==bundle.sums['host/host.elf'],'ELF differs from bundle')
        for p in (elf,qemu):pins[str(p)]=sha256(p)
        image=Image(elf.read_bytes());sites=mmio_sites(image);stops=resnet_stop_sites(image)
        require(bool(stops),'normal completion stops required')
        _,_,assemble,_=configure_model(args.model_root);before=model_binding(args.model_root)
        selector=Device.__new__(Device);selector.bundle=bundle;selector.pins={}
        for index in range(bundle.kernel_count):
            paths=selector.resnet_kernel_paths(index)
            runtime_program(bundle.read(paths['source']).decode(),bundle.read(paths['original']),assemble)
        require(model_binding(args.model_root)==before,'assembler admission model bookends')
        bookends(pins);bundle.verify_bookends()
        command=[str(qemu),'-machine','virt','-m','8G','-smp','1','-accel','tcg,thread=single','-bios','none',
            '-nodefaults','-nographic','-monitor','none','-serial','file:uart.log','-S','-gdb',
            'unix:qemu.sock,server=on,wait=off','-kernel',str(elf)]
        receipt.update(qemu_command=command,inputs_sha256=pins,bundle_sha256sums_sha256=bundle.checksums_sha256,
            elf_sha256=sha256(elf),placement=bundle.placement,uart=build['uart'],assembler_binding=before)
        with (out/'qemu.log').open('wb') as log:
            process=subprocess.Popen(command,cwd=out,stdout=log,stderr=subprocess.STDOUT)
        receipt['qemu_pid']=process.pid
        (out/'runtime.json').write_text(json.dumps(dict(runner_pid=os.getpid(),qemu_pid=process.pid,
            process_group=os.getpgrp(),qemu_command=command),indent=2)+'\n')
        while not (out/'qemu.sock').exists():
            if process.poll() is not None or time.monotonic()>=deadline:raise TimeoutError('QEMU startup/deadline')
            time.sleep(.05)
        remote=Remote(str(out/'qemu.sock'),timeout=seconds,deadline=deadline)
        require(remote.request('?')[:1] in (b'S',b'T'),'all-stop initial guest')
        remote.set_reg(32,image.entry)
        for pc,site in sites.items():
            require(remote.read(pc,site['length'])==image.bytes_at(pc,site['length']),'actual MMIO accessor bytes')
            remote.breakpoint(pc,site['length'])
        for pc,length in stops.items():remote.breakpoint(pc,length)
        original=Path.cwd();os.chdir(out)
        try:device=Device(remote,out,deadline,bundle,args.model_root)
        finally:os.chdir(original)
        require(device.binding==before,'fresh persistent real device model binding')
        receipt.update(model_device_constructed=True,memory_backend=device.memory_backend,kernel_cycle_cap=device.cycle_cap)
        accesses=0;cap=mmio_access_cap(device,'resnet');pending=None
        while time.monotonic()<deadline:
            pc,stop=remote.resume();receipt['host_elf_executed']=True
            require(pc not in stops,'whole-host stop reached before smoke frontier')
            require(pc in sites,'unexpected guest stop')
            access=sites[pc];address=remote.reg(access['base'])+access['offset']
            if access['store']:
                value=remote.reg(access['reg'])&0xffffffff
                if store_or_stop(device,address,value):
                    pending=dict(pc=pc,address=address,value=value,next_global_index=COUNT,next_sequence=11,
                                 stopped_before_start=True);break
            else:
                value=device.read32(address)
                if access['signed'] and value&0x80000000:value|=0xffffffff00000000
                remote.set_reg(access['reg'],value)
            remote.set_reg(32,pc+access['length']);accesses+=1
            require(accesses<cap,'existing MMIO access cap')
        require(pending is not None and remote.reg(32)==pending['pc'],'actual eleventh START must remain unexecuted')
        verify_ledgers(device)
        uart=(out/'uart.log').read_bytes();records=uart_prefix(uart,build['result_name'])
        require(int.from_bytes(remote.read(image.symbols['tohost']['address'],8),'little')==0,'no full host completion')
        bundle.verify_bookends();device.verify_bookends();bookends(pins)
        require(time.monotonic()<deadline,'absolute completion deadline')
        receipt.update(status='PASS_RESNET_RELOCATED_FIRST10_MODEL_SMOKE',kernels=device.completed,model_device_execution=True,
            source_bookends_equal=True,model_binding=device.binding,device_payloads_sha256=device.pins,
            mmio_accesses=accesses,mmio_access_cap=cap,pending_start=pending,uart_records=records,
            uart_sha256=digest(uart),one_persistent_device=True,completed_launches=COUNT,remaining_launches=44)
    except BaseException as error:
        receipt['error']=dict(type=type(error).__name__,message=str(error))
        if device:
            receipt['kernels']=device.completed
            receipt['model_device_execution']=device.core.cycle_count>0
    finally:
        signal.signal(signal.SIGTERM,signal.SIG_IGN);signal.signal(signal.SIGINT,signal.SIG_IGN)
        cleanup=[]
        try:
            if process and process.poll() is None:
                with suppress(ProcessLookupError):process.terminate()
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    with suppress(ProcessLookupError):process.kill()
                    process.wait(timeout=10)
        except Exception as error:cleanup.append(str(error))
        for obj in (remote,device):
            try:
                if obj:obj.close()
            except Exception as error:cleanup.append(str(error))
        receipt['qemu_child_reaped']=process is not None and process.poll() is not None
        if cleanup:receipt.update(status='FAIL_RESNET_SMOKE_CLEANUP',cleanup_errors=cleanup)
        receipt.update(elapsed_seconds=time.monotonic()-started,
            parent_max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            children_max_rss_kib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
        (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:receipt[k] for k in ('status','error') if k in receipt}),flush=True)
    return 0 if receipt['status'].startswith('PASS_') else 1


def utc(value):
    parsed=datetime.fromisoformat(value)
    if parsed.tzinfo is None:raise argparse.ArgumentTypeError('explicit timezone required')
    return parsed.astimezone(timezone.utc)


def arguments(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--first-ten',action='store_true',required=True)
    for name in ('elf','bundle-root','model-root','qemu','out'):parser.add_argument('--'+name,type=checked,required=True)
    parser.add_argument('--wall-seconds',type=int,required=True)
    parser.add_argument('--stop-by-utc',type=utc,required=True)
    return parser.parse_args(argv)


if __name__=='__main__':sys.exit(run(arguments()))
