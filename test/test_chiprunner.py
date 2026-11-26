import hashlib
import json
import os
import signal
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from atlas_mlir.chiprunner.kit_manifest import load, console_errors
from atlas_mlir.chiprunner import run_kit


def synthetic(root):
    """Packaging stub, no compiler, model, or silicon qualification."""
    def write(name, data):
        p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
    refs=[b'\x01\x02\x03\x04',b'\x05\x06\x07\x08']
    ranges=[dict(id='result'+str(i),role='public_output',address=0x200c0000+i*8,bytes=4,index=i) for i in range(2)]
    allocations=[dict(name='output:'+r['id'],role='public_output',address=r['address'],bytes=4) for r in ranges]
    allocations+=[dict(name='scratch',role='private_scratch',address=0x200c0010,bytes=8),dict(name='parameters',role='immutable_parameters',address=0x200c0030,bytes=16)]
    dep=dict(window=dict(address=0x200c0000,bytes=64),outputs=ranges,hash_ranges=ranges,allocations=allocations)
    raw=bytearray(184);raw[:6]=b'\x7fELF\x02\x01';struct.pack_into('<Q',raw,24,0x08000000);struct.pack_into('<H',raw,18,243);struct.pack_into('<Q',raw,32,64);struct.pack_into('<HH',raw,54,56,1);struct.pack_into('<IIQQQQQQ',raw,64,1,7,120,0x08000000,0x08000000,64,64,8)
    sha=lambda b:hashlib.sha256(b).hexdigest()
    m=dict(deployment=dep,bench=dict(entry_address=0x08000000),bench_facts=dict(receiver_baud=115200,core_clock_hz=100000000,source='operator-stated, unmeasured'),platform_assumptions=dict(timeout_seconds=5),expected_hashes=[dict(name=r['id'],role=r['role'],bytes=4,sha256=sha(b)) for r,b in zip(ranges,refs)],sizes=dict(chip=dict(sha256=sha(raw),loads=[dict(address=0x08000000,file_bytes=64,memory_bytes=64)])),runner=dict(schema='atlas.kit.v1',record_tag='KIT',paths=dict(deployment='bound/layout.json',window='images/data.bin',config='board/target.cfg',positive='images/app.elf',negative='mutation/app.elf'),reference_paths={'0':'oracle/a.bin','1':'oracle/b.bin'},negative_paths={'0':'mutation/a.bin','1':'mutation/b.bin'}))
    write('images/app.elf',raw);write('mutation/app.elf',raw)
    write('bound/layout.json',json.dumps(dep).encode());write('images/data.bin',b'\x01'*64);write('board/target.cfg',b'# stub\n')
    for i,b in enumerate(refs):write(['oracle/a.bin','oracle/b.bin'][i],b);write(['mutation/a.bin','mutation/b.bin'][i], b'\x09'+b[1:] if i==0 else b)
    for name in ['manifest.json','images/manifest.json','mutation/manifest.json']:write(name,json.dumps(m).encode())
    inventory(root)
    return m


def inventory(root):
    files=sorted(p for p in root.rglob('*') if p.is_file() and p.name!='SHA256SUMS')
    (root/'SHA256SUMS').write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+str(p.relative_to(root))+'\n' for p in files))


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name);self.m=synthetic(self.root)

    def test_both_selections_and_all_output_bytes(self):
        k=load(self.root);self.assertEqual(len(k['references']),2)
        self.assertEqual(len(load(self.root,'negative/chip.elf')['negative']),2)
        log=('KIT STATUS 0 CYCLES 9\n'+''.join('KIT HASH %s %s %s\n'%(r['name'],r['role'],r['sha256']) for r in self.m['expected_hashes'])+'KIT COMPLETE\n').encode()
        self.assertEqual(console_errors(self.m,log,'KIT'),[])
        self.assertTrue(console_errors(self.m,log,'OTHER'))
        self.assertTrue(console_errors(self.m,log+ b'KIT COMPLETE\n','KIT'))
        self.assertTrue(console_errors(self.m,log.replace(b'STATUS 0',b'STATUS 1'),'KIT'))
        self.assertTrue(console_errors(self.m,log.replace(b'COMPLETE\n',b''),'KIT'))

    def test_record_tag_is_bundle_metadata(self):
        self.assertEqual(load(self.root)['settings']['record_tag'],'KIT')
        for tag in (None,'bad tag'):
            if tag is None: del self.m['runner']['record_tag']
            else: self.m['runner']['record_tag']=tag
            for name in ['manifest.json','images/manifest.json','mutation/manifest.json']:(self.root/name).write_text(json.dumps(self.m))
            inventory(self.root)
            with self.subTest(tag=tag), self.assertRaisesRegex(ValueError,'record tag'):load(self.root)

    def test_fail_closed_drift_and_uncovered_output(self):
        (self.root/'oracle/b.bin').write_bytes(b'bad!')
        with self.assertRaises(ValueError):load(self.root)
        inventory(self.root)
        with self.assertRaises(ValueError):load(self.root)

    def test_cleanup_after_kill_is_bounded(self):
        import subprocess
        a=run_kit.arguments([str(self.root)]);kit=run_kit.prepare(a)
        logs=self.root/'logs';logs.mkdir()
        session=run_kit.Session(a,kit,logs)
        waits=[]
        def wait(timeout=None):
            waits.append(timeout);raise subprocess.TimeoutExpired('openocd',timeout)
        session.process=type('Process',(),dict(terminate=lambda self:None,kill=lambda self:None,wait=lambda self,timeout=None:wait(timeout)))()
        summary=session.close('TRANSPORT_ERROR','stuck')
        self.assertEqual(waits,[5,5])
        self.assertEqual((summary['verdict'],summary['cleanup_error']),('TRANSPORT_ERROR','OpenOCD did not exit after kill'))

    def test_every_cleanup_step_runs_when_the_first_raises(self):
        a=run_kit.arguments([str(self.root)]);kit=run_kit.prepare(a)
        logs=self.root/'logs';logs.mkdir()
        session=run_kit.Session(a,kit,logs)
        calls=[]
        def step(name,fails=False):
            def call(*args,**kwargs):
                calls.append(name)
                if fails: raise OSError(name+' failed')
            return call
        session.rpc=type('RPC',(),dict(broken=False,close=lambda self:step('rpc close',True)()))()
        session.uart=type('UART',(),dict(close=lambda self:step('uart close',True)()))()
        # A stub OpenOCD: a real process that only exits when the session stops it.
        session.process=subprocess.Popen(['sleep','60'],stdin=subprocess.DEVNULL)
        self.addCleanup(lambda: session.process.poll() is None and session.process.kill())
        summary=session.close('TRANSPORT_ERROR','stuck')
        self.assertEqual(calls,['rpc close','uart close'])
        self.assertEqual(session.process.poll(),-signal.SIGTERM)
        self.assertTrue(session.openocd_log.closed)
        self.assertEqual((summary['verdict'],summary['cleanup_error']),('TRANSPORT_ERROR','rpc close failed; uart close failed'))

    def test_fresh_preflight_and_no_transport(self):
        with patch.object(run_kit,'UART',side_effect=AssertionError('transport opened')), patch.object(run_kit.subprocess,'Popen',side_effect=AssertionError('process launched')):
            with patch.object(sys,'argv',['run_kit',str(self.root),'--dry-run']):self.assertEqual(run_kit.main(),0)
            (self.root/'images/data.bin').write_bytes(b'changed')
            with patch.object(sys,'argv',['run_kit',str(self.root),'--dry-run']):self.assertNotEqual(run_kit.main(),0)

    def test_physical_fit_rejects_wrong_machine_and_overflow(self):
        from atlas_mlir.chiprunner.preflight import physical
        k=load(self.root)
        original=(self.root/'images/app.elf').read_bytes()
        for offset,fmt,value in [(18,'<H',0),(64+24,'<Q',0x0800fff0)]:
            raw=bytearray(original);struct.pack_into(fmt,raw,offset,value)
            (self.root/'images/app.elf').write_bytes(raw)
            with self.assertRaises(ValueError):physical(k)
        (self.root/'images/app.elf').write_bytes(original)

    def test_invalid_waits_and_bound_baud(self):
        for flags in [ ['--rpc-timeout','nan'],['--start-timeout','-1'],['--timeout','inf'],['--baud','384615'] ]:
            with patch.object(sys,'argv',['run_kit',str(self.root),'--dry-run']+flags):self.assertNotEqual(run_kit.main(),0)


class BenchPrerequisiteTests(unittest.TestCase):
    """preflight.live with a stub OpenOCD that only prints a banner, and stub process checks; never the real binary."""
    banner='Open On-Chip Debugger 0.12.0-test (2026-01-01-00:00)'

    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        synthetic(self.root/'kit');self.kit=load(self.root/'kit')
        bin=self.root/'bin';bin.mkdir()
        self.openocd=bin/'openocd';self.openocd.write_text("#!/bin/sh\necho '%s' >&2\n"%self.banner);self.openocd.chmod(0o755)
        for name in ('pgrep','fuser','lsof'):self.stub(name,'',1)
        path=patch.dict(os.environ,PATH=str(bin)+':'+os.environ.get('PATH',''));path.start();self.addCleanup(path.stop)
        self.digest=hashlib.sha256(self.openocd.read_bytes()).hexdigest()

    def stub(self,name,output,rc):
        p=self.root/'bin'/name;p.write_text('#!/bin/sh\necho "%s $*" >> "$(dirname "$0")/calls"\nprintf "%s"\nexit %d\n'%(name,output,rc));p.chmod(0o755)

    def live(self,builds,serial=None):
        from atlas_mlir.chiprunner import preflight
        with patch.object(preflight,'openocd_builds',return_value=builds):
            preflight.live(self.kit,str(self.openocd),serial or str(self.root/'tty'))

    def test_shipped_table_declares_the_bench_and_macos_builds(self):
        # Linux: the bench build. macOS: conda-forge's osx-64 build, which laptop setup installs. On an Apple silicon Mac
        # under Rosetta it printed this banner and, in a dry connect (halt, reg pc) at 100 MHz, found TAP 0x00000001 and
        # one hart (misa 0x800000000034112d) and read the PC; no live run had used it when it was declared. Later one K32 and one K64 live pair passed with it.
        from atlas_mlir.chiprunner.preflight import openocd_builds
        self.assertEqual([(row['sha256'],row['banner'],row['package']) for row in openocd_builds()],
                         [('37be76bfaaaa8c9e57ad70a6130eb7420ead18ebc65aede1399c2eafd36fdb5b',
                           'Open On-Chip Debugger 0.12.0-gc66e1e0-dirty (2026-05-01-09:15)','conda-forge linux-64 openocd-0.12.0-h6376dc9_0'),
                          ('e5a381645225c5078e5374a02e29f4675cf35d2b830a272943b799549a0ad913',
                           'Open On-Chip Debugger 0.12.0-snapshot (2026-05-01-09:22)','conda-forge osx-64 openocd-0.12.0-h57f6e69_0')])

    def test_declared_build_with_its_exact_banner_passes(self):
        other=dict(sha256='0'*64,banner='other',package='other')
        self.live([other,dict(sha256=self.digest,banner=self.banner,package='stub')])

    def test_undeclared_digest_and_banner_mismatch_refuse(self):
        for builds,message in [([dict(sha256='0'*64,banner=self.banner,package='other')],'not declared: sha256 '+self.digest),
                               ([dict(sha256=self.digest,banner=self.banner.rsplit(' ',1)[0],package='stub')],'banner differs'),
                               ([dict(sha256=self.digest,banner='Open On-Chip Debugger 0.12.0-gc66e1e0-dirty (2026-05-01-09:15)',package='stub')],'banner differs')]:
            with self.subTest(message=message),self.assertRaisesRegex(ValueError,message):self.live(builds)

    def test_process_checks_per_host_os(self):
        import types
        from atlas_mlir.chiprunner import preflight
        build=[dict(sha256=self.digest,banner=self.banner,package='stub')]
        tty=str(self.root/'tty')
        for platform,expected in [('linux',['pgrep -fa','fuser '+tty]),('darwin',['pgrep -afl','lsof -t -- '+tty])]:
            with self.subTest(platform=platform),patch.object(preflight,'sys',types.SimpleNamespace(platform=platform)):
                (self.root/'bin/calls').unlink(missing_ok=True);self.live(build)
                calls=(self.root/'bin/calls').read_text().splitlines()
                self.assertEqual(calls,[expected[0]+' [r]un_full|[c]hain22|[o]penocd',expected[1]])
                for name,output,message in [(expected[0].split()[0],'4242 openocd -f x\\n','bench process check is not empty: 4242 openocd'),
                                            (expected[1].split()[0],'4242\\n','UART ownership check is not empty')]:
                    self.stub(name,output,0)
                    with self.assertRaisesRegex(ValueError,message):self.live(build)
                    self.stub(name,'',1)

    def test_missing_check_tool_refuses_by_name(self):
        import types
        from atlas_mlir.chiprunner import preflight
        build=[dict(sha256=self.digest,banner=self.banner,package='stub')]
        for platform,tool in [('linux','pgrep'),('linux','fuser'),('darwin','lsof')]:
            with self.subTest(tool=tool),patch.object(preflight,'sys',types.SimpleNamespace(platform=platform)),\
                    patch.dict(os.environ,PATH=str(self.root/'bin')):
                (self.root/'bin'/tool).unlink()
                try:
                    with self.assertRaisesRegex(ValueError,'^%s not found; the bench checks need it'%tool):self.live(build)
                finally:self.stub(tool,'',1)

    def test_macos_checks_both_names_of_the_uart(self):
        import types
        from atlas_mlir.chiprunner import preflight
        build=[dict(sha256=self.digest,banner=self.banner,package='stub')]
        lsof=self.root/'bin/lsof'
        for serial,held in [('/dev/cu.usbserial-B','/dev/tty.usbserial-B'),('/dev/tty.usbserial-B','/dev/cu.usbserial-B')]:
            with self.subTest(serial=serial),patch.object(preflight,'sys',types.SimpleNamespace(platform='darwin')):
                lsof.write_text('#!/bin/sh\necho "lsof $*" >> "$(dirname "$0")/calls"\n[ "$3" = %s ] && echo 4242 && exit 0\nexit 1\n'%held)
                (self.root/'bin/calls').unlink(missing_ok=True)
                with self.assertRaisesRegex(ValueError,'UART ownership check is not empty: '+held):
                    self.live(build,serial)
                self.assertEqual((self.root/'bin/calls').read_text().splitlines()[-2:],['lsof -t -- '+serial,'lsof -t -- '+held])
        with patch.object(preflight,'sys',types.SimpleNamespace(platform='linux')):
            self.assertEqual(preflight.serial_devices('/dev/cu.usbserial-B'),['/dev/cu.usbserial-B'])

    def test_malformed_table_refuses(self):
        from atlas_mlir.chiprunner import preflight
        row=dict(sha256=self.digest,banner=self.banner,package='stub')
        other=dict(row,sha256='0'*64,banner=self.banner.replace('test','other'))
        with patch.object(preflight,'read_bytes',return_value=json.dumps([row,other]).encode()):
            self.assertEqual(preflight.openocd_builds(),[row,other])
        for rows in [[],{},[row,row],[dict(row,extra='x')],[dict(row,banner='')],[dict(row,sha256=self.digest.upper())],
                     [dict(row,sha256=self.digest[:-1])],[dict(row,sha256=self.digest+'0')],[dict(row,banner='Info : clock speed 10000 kHz')],
                     [row,dict(other,banner=self.banner)]]:
            with self.subTest(rows=rows),patch.object(preflight,'read_bytes',return_value=json.dumps(rows).encode()):
                with self.assertRaisesRegex(ValueError,'malformed OpenOCD build table'):preflight.openocd_builds()

class SystemLinkTests(unittest.TestCase):
    """Only the declared system links are read through; on Linux there are none."""
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(os.path.realpath(self.tmp.name))
        (self.root/'private/tmp/kit').mkdir(parents=True);(self.root/'tmp').symlink_to('private/tmp');(self.root/'other').symlink_to('private/tmp')

    def test_only_macos_declares_system_links(self):
        from atlas_mlir.chiprunner import kit_manifest
        expected={'/etc':'/private/etc','/tmp':'/private/tmp','/var':'/private/var'} if sys.platform=='darwin' else {}
        self.assertEqual(kit_manifest.SYSTEM_LINKS,expected)

    def test_declared_system_link_is_read_through_and_others_refuse(self):
        from atlas_mlir.chiprunner import kit_manifest
        from atlas_mlir.deploy import Refusal, safe_path as deploy_safe_path
        links={str(self.root/'tmp'):str(self.root/'private/tmp')}
        for check,refusal in [(kit_manifest.safe_path,ValueError),(deploy_safe_path,Refusal)]:
            with self.subTest(check=check.__module__):
                with self.assertRaises(refusal):check(self.root/'tmp/kit')
                with patch.object(kit_manifest,'SYSTEM_LINKS',links):
                    self.assertEqual(check(self.root/'tmp/kit'),self.root/'private/tmp/kit')
                    self.assertEqual(check(self.root/'tmp'),self.root/'private/tmp')
                    with self.assertRaises(refusal):check(self.root/'other/kit')
                    (self.root/'private/tmp/link').symlink_to('kit')
                    with self.assertRaises(refusal):check(self.root/'tmp/link')
                    (self.root/'private/tmp/link').unlink()
                with patch.object(kit_manifest,'SYSTEM_LINKS',{str(self.root/'other'):str(self.root/'elsewhere')}):
                    with self.assertRaises(refusal):check(self.root/'other/kit')

if __name__=='__main__':unittest.main()
