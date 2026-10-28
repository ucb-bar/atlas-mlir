"""Execute a full Smol RV64 host with authenticated captured projection replies.

No accelerator is constructed or executed. CSR counters count protocol accesses
only and must never be used as model or hardware cycle measurements.
"""

import argparse
from contextlib import suppress
import hashlib
import json
import os
import resource
import re
from pathlib import Path
import signal
import struct
import subprocess
import sys
import time
from types import SimpleNamespace

TOOLS = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(TOOLS))
from atlas_path_guard import checked
from atlas_tooling import sha256, sha as digest
from atlas_host_layout import ATLAS_CSR_BASE, ATLAS_IMEM_BASE, ATLAS_IMEM_BYTES
from atlas_program import code_sources
from bundle import EPILOGUE
from elf_image import Image, mmio_sites
from full_bundle import FullBundle
from gdb_remote import Remote
from qmp_memory import PhysicalMemory
from result_lines import verify as verify_result_lines
from run_model_tile import configure_model, model_binding
import smol_full_contract as contract


class Replies:
    def __init__(self, remote, physical, out, bundle, negative=False):
        self.remote, self.physical, self.out, self.bundle = remote, physical, out, bundle
        self.negative = negative
        self.imem, self.written, self.completed = bytearray(ATLAS_IMEM_BYTES), set(), []
        self.counter, self.marker, self.sequence = 0, 0, 0

    def read32(self, address):
        self.counter = (self.counter+1) & 0xffffffff
        values = {ATLAS_CSR_BASE:self.counter, ATLAS_CSR_BASE+8:5,
                  ATLAS_CSR_BASE+16:self.marker, ATLAS_CSR_BASE+20:self.sequence}
        if address not in values:
            raise ValueError('unadmitted captured-reply CSR read '+hex(address))
        return values[address]

    def write32(self, address, value):
        self.counter = (self.counter+1) & 0xffffffff
        if ATLAS_IMEM_BASE <= address < ATLAS_IMEM_BASE+ATLAS_IMEM_BYTES and address % 4 == 0:
            offset = address-ATLAS_IMEM_BASE
            struct.pack_into('<I', self.imem, offset, value)
            self.written.add(offset//4)
        elif address == ATLAS_CSR_BASE+16:
            if value != 0:
                raise ValueError('host must clear the invocation marker')
            self.marker = value
        elif address == ATLAS_CSR_BASE+20:
            self.sequence = value
        elif address == ATLAS_CSR_BASE+24 and value == 1:
            self.start()
        else:
            raise ValueError('unadmitted captured-reply CSR write '+hex(address))

    def start(self):
        index = len(self.completed)
        if index >= self.bundle.kernel_count or self.marker or self.sequence != index+1:
            raise ValueError('actual host start order/marker differs')
        item = self.bundle.kernels[index]
        program = self.bundle.read(item['original'])[:-4]+EPILOGUE
        if self.written != set(range(len(program)//4)) or self.imem[:len(program)] != program:
            raise ValueError('actual wrapped host IMEM differs at '+str(index))
        initial = self.bundle.initial_dram(index)
        actual = self.physical.read(item['dram_base'],len(initial))
        if actual != initial:
            at = next(i for i,(a,b) in enumerate(zip(actual,initial)) if a != b)
            raise ValueError(f'actual RV64 predecessor/staging differs at launch {index} byte {at}')
        output = self.bundle.read(item['expected'])
        corrupted = self.negative and index == self.bundle.kernel_count-1
        if corrupted:
            rows = item['logical_mkn'][0]
            m,_,n = item['physical_mkn']
            if (rows,m,n) != (50,64,32):
                raise ValueError('negative requires final padded action panel')
            word = (rows//32)*(n//32)*1024+(rows%32)*16
            output = bytearray(output)
            if output[2*word:2*word+2] != bytes(2):
                raise ValueError('negative padded word must start at positive zero')
            output[2*word] ^= 1
            output = bytes(output)
        self.remote.write(item['dram_base']+item['output_offset'],output)
        result = dict(index=index,sequence=self.sequence,wrapped_program_sha256=digest(program),
                      actual_staging_sha256=digest(actual),captured_reply_sha256=self.bundle.sums[item['expected']],
                      injected_reply_sha256=digest(output),corrupted=corrupted,
                      captured_reply_injected=True,device_execution=False)
        (self.out/f'reply-{index:04d}.json').write_text(json.dumps(result,indent=2)+'\n')
        self.completed.append(result)
        self.marker = self.sequence
        self.written.clear()
        (self.out/'progress.json').write_text(json.dumps(dict(status='RUNNING',launches=len(self.completed),
            projection_calls=item['call']+1,protocol_accesses=self.counter,device_execution=False))+'\n')


def interrupted(signum, frame):
    raise InterruptedError('captured-reply host run interrupted: '+str(signum))


def run(args):
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    out = checked(args.out)
    out.mkdir(parents=True,exist_ok=False)
    deadline = time.monotonic()+args.wall_seconds
    started = time.monotonic()
    receipt = dict(status='FAIL_RV64_CAPTURED_REPLIES',scope=__doc__,command=[sys.executable,*sys.argv],
                   guest_execution=False,device_execution=False,hardware_execution=False,
                   synthetic_counter_is_timing=False,mode=args.mode)
    process = remote = physical = replies = None
    pins = {}
    try:
        bundle = FullBundle(args.bundle,'smol-full')
        raw = bundle.read('host/host.elf'); image = Image(raw)
        build = bundle.build
        if (build['uart']['kind'],build['uart']['base']) != ('ns16550',0x10000000):
            raise ValueError('captured-reply QEMU requires the explicit NS16550 virt build')
        for name,expected in code_sources(traffic_reuse=True).items():
            path = checked(TOOLS/name)
            if sha256(path) != expected:
                raise ValueError('generator changed during admission')
            pins[str(path)] = expected
        for module in tuple(sys.modules.values()):
            path = getattr(module,'__file__',None)
            if path and Path(path).is_absolute() and Path(path).is_relative_to(TOOLS):
                path = checked(path);pins[str(path)] = sha256(path)
        qemu = checked(args.qemu)
        pins[str(qemu)] = sha256(qemu)
        _,_,assemble,_ = configure_model(args.model_root)
        binding = model_binding(args.model_root)
        if (binding['revision'] != '9e6313bddecb8ba936206eb347f5cd4e5335282c'
                or binding['tracked_source_diff_exit'] != 0):
            raise ValueError('captured host requires the clean pinned model assembler')
        contract.validate_bundle(bundle,image,assemble=assemble)
        if model_binding(args.model_root) != binding or any(sha256(checked(p)) != h for p,h in pins.items()):
            raise ValueError('captured-reply admission source/model bookends changed')
        sites,stops = mmio_sites(image),contract.stop_sites(image)
        command = [str(qemu),'-machine','virt','-m','8G','-smp','1','-accel','tcg,thread=single',
            '-bios','none','-nodefaults','-nographic','-monitor','none','-serial','file:uart.log','-S',
            '-gdb','unix:qemu.sock,server=on,wait=off','-qmp','unix:qmp.sock,server=on,wait=off','-kernel',
            str(checked(args.bundle/'host/host.elf'))]
        receipt.update(elf_sha256=digest(raw),bundle_sha256sums_sha256=bundle.checksums_sha256,
                       qemu_command=command,inputs_sha256=pins,assembler_model_binding=binding,
                       cpu_affinity=sorted(os.sched_getaffinity(0)))
        with (out/'qemu.log').open('wb') as log:
            process = subprocess.Popen(command,cwd=out,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        while not (out/'qemu.sock').exists() or not (out/'qmp.sock').exists():
            if process.poll() is not None or time.monotonic() >= deadline:
                raise TimeoutError('QEMU startup failed or deadline expired')
            time.sleep(0.05)
        remote = Remote(str(out/'qemu.sock'),deadline=deadline,timeout=args.wall_seconds)
        physical = PhysicalMemory(out/'qmp.sock',out,deadline)
        if remote.request('?')[:1] not in (b'S',b'T'):
            raise ValueError('QEMU did not enter all-stop mode')
        remote.set_reg(32,image.entry)
        for pc,access in sites.items():
            if remote.read(pc,access['length']) != image.bytes_at(pc,access['length']):
                raise ValueError('actual MMIO accessor instruction differs')
            remote.breakpoint(pc,access['length'])
        for pc,length in stops.items(): remote.breakpoint(pc,length)
        replies = Replies(remote,physical,out,bundle,args.mode == 'negative')
        accesses = 0
        while time.monotonic() < deadline:
            pc,_ = remote.resume()
            receipt['guest_execution'] = True
            if pc in stops: break
            if pc not in sites: raise ValueError('unexpected guest debugger stop '+hex(pc))
            access = sites[pc]
            address = remote.reg(access['base'])+access['offset']
            if access['store']:
                replies.write32(address,remote.reg(access['reg']) & 0xffffffff)
            else:
                value = replies.read32(address)
                if access['signed'] and value & 0x80000000: value |= 0xffffffff00000000
                remote.set_reg(access['reg'],value)
            remote.set_reg(32,pc+access['length'])
            accesses += 1
        else: raise TimeoutError('captured-reply absolute wall deadline')
        if len(replies.completed) != 2715:
            raise ValueError('actual guest did not traverse all captured replies')
        uart = (out/'uart.log').read_bytes()
        if args.mode == 'positive':
            actual_guest = SimpleNamespace(read=physical.read,reg=remote.reg)
            completion = contract.verify_host_completion(actual_guest,image,bundle,out)
            (out/'cycles.bin').rename(out/'protocol-counts.bin')
            completion['uart'] = verify_result_lines(uart,build['result_name'],2715,
                progress_launches=build['result_progress_launches'])
            receipt.update(status='PASS_RV64_HOST_CAPTURED_REPLIES',**completion)
        else:
            tohost = int.from_bytes(physical.read(image.symbols['tohost']['address'],8),'little')
            raw_state = physical.read(image.symbols['smol_full_raw']['address'],160)
            graph = physical.read(image.symbols['smol_full_graph_state']['address'],16)
            runtime = physical.read(image.symbols['smol_full_host']['address'],112)
            progress = [int(value) for value in re.findall(rb'ATLAS_KERNEL_PROGRESS[^\n]*launches_completed=(\d+)',uart)]
            if (tohost != 9 or b'ATLAS_KERNEL_FAIL' not in uart or b'ATLAS_KERNEL_PASS' in uart
                    or struct.unpack_from('<QQ',raw_state,40) != (2714,1420)
                    or struct.unpack_from('<Ii',raw_state,148) != (1,6)
                    or struct.unpack('<4I',graph) != (1,9,1272,6)
                    or progress != build['result_progress_launches'][:-1]
                    or struct.unpack_from('<QQ',runtime,72) != (0,0)
                    or struct.unpack_from('<6I',runtime,88)[1:3] != (0,0)
                    or struct.unpack_from('<I',runtime,108)[0] != 2715):
                raise ValueError('actual padded-row corruption did not reach expected host FAIL')
            receipt.update(status='PASS_RV64_HOST_CAPTURED_REPLY_NEGATIVE',tohost=tohost,
                           actual_raw_panel_cursor=2714,actual_raw_call_cursor=1420)
        bundle.verify_bookends()
        if model_binding(args.model_root) != binding or any(sha256(checked(p)) != h for p,h in pins.items()):
            raise ValueError('captured-reply source/model bookends changed')
        receipt.update(source_bookends_equal=True,uart_sha256=digest(uart),launches=len(replies.completed),
                       mmio_accesses=accesses,captured_replies=replies.completed)
    except BaseException as error:
        receipt['error'] = dict(type=type(error).__name__,message=str(error))
        if replies: receipt['launches'] = len(replies.completed)
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        cleanup_errors = []
        for connection in (physical,remote):
            try:
                if connection: connection.close()
            except Exception as error:
                cleanup_errors.append(str(error))
        try:
            if process and process.poll() is None:
                with suppress(ProcessLookupError): os.killpg(process.pid,signal.SIGTERM)
                try: process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    with suppress(ProcessLookupError): os.killpg(process.pid,signal.SIGKILL)
                    process.wait(timeout=10)
        except Exception as error:
            cleanup_errors.append(str(error))
        receipt['qemu_child_reaped'] = process is not None and process.poll() is not None
        if cleanup_errors:
            receipt.update(status='FAIL_RV64_CAPTURED_REPLY_CLEANUP',cleanup_errors=cleanup_errors)
        receipt['elapsed_seconds'] = time.monotonic()-started
        receipt['parent_max_rss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        receipt['children_max_rss_kib'] = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('bundle','model-root','qemu','out'):
        parser.add_argument('--'+name,type=checked,required=True)
    parser.add_argument('--mode',choices=('positive','negative'),default='positive')
    parser.add_argument('--wall-seconds',type=int,required=True)
    args = parser.parse_args()
    if not 0 < args.wall_seconds <= 10800:
        parser.error('captured-reply wall cap must be positive and at most three hours')
    result = run(args)
    print(json.dumps({key:result[key] for key in ('status','elapsed_seconds')}))
    sys.exit(0 if result['status'].startswith('PASS_') else 1)


if __name__ == '__main__': main()
