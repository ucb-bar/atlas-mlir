"""Synthetic receipts exercise replay validation, never silicon acceptance.

All transfer and terminal receipts are explicitly synthetic.
"""
import json
from pathlib import Path
import sys
import tempfile
import unittest

from atlas_mlir.chiprunner import check_rung as checker
from atlas_mlir.chiprunner.kit_manifest import load
from atlas_mlir.chiprunner.preflight import openocd_builds
from test_chiprunner import synthetic



def receipts(root, bundle, negative=False):
    k = load(bundle, 'negative/chip.elf' if negative else 'chip.elf')
    prefix = 'synthetic-replay'
    logs = root / (prefix + '-20261117T000000Z')
    logs.mkdir()
    changed = [x['row']['id'] for x,y in zip(k['references'],k['negative']) if x['data'] != y['data']]
    error = ('UART hash mismatch: ' + changed[0] + '; ' + '; '.join('dump differs from frozen reference: ' + x for x in changed)) if negative else None
    selected = k['negative'] if negative else k['references']
    uart = ('KIT STATUS 0 CYCLES 9\n' + ''.join('KIT HASH %s %s %s\n' % (x['row']['id'],x['row']['role'],x['sha256']) for x in selected) + 'KIT COMPLETE\n').encode()
    (logs/'uart.log').write_bytes(uart)
    (logs/'console-check.txt').write_text('CONSOLE FAIL\n' + ''.join('hash mismatch or missing '+x+'\n' for x in changed) if negative else 'CONSOLE PASS\n')
    selected = k['negative'] if negative else k['references']
    for i,x in enumerate(selected): (logs/('output-%d.bin'%i)).write_bytes(x['data'])
    operations=['reset','load_image','load_image','verify_image','resume','halt'] + ['dump_image']*len(selected)
    counts=[None,len(k['window'].read_bytes()),k['elf_bytes'],k['elf_bytes'],None,None] + [x['row']['bytes'] for x in selected]
    commands=['reset halt','load_image {%s} %#x bin'%(k['window'],k['dep']['window']['address']),'load_image {%s}'%k['elf'],'verify_image {%s}'%k['elf'],'resume %#x'%k['manifest']['bench']['entry_address'],'halt']
    commands += ['dump_image {%s} %#x %d'%(logs/('output-%d.bin'%i),x['row']['address'],x['row']['bytes']) for i,x in enumerate(selected)]
    events=[]
    for op,count,cmd in zip(operations,counts,commands):
        e=dict(operation=op,command=cmd,bytes=count,started_utc='2026-11-17T00:00:00+00:00',success=True,wall_seconds=0.5,bytes_per_second=None if count is None else count/0.5)
        if op=='dump_image': e['observed_bytes']=count
        events.append(e)
    totals={}
    for e in events:
        if e['bytes'] is not None:
            row=totals.setdefault(e['operation'],dict(attempted_bytes=0,completed_bytes=0,wall_seconds=0))
            row['attempted_bytes']+=e['bytes'];row['completed_bytes']+=e['bytes'];row['wall_seconds']+=e['wall_seconds']
    settings=k['settings']
    summary=dict(verdict='FAIL' if negative else 'PASS',started_utc='2026-11-17T00:00:00+00:00',wall_seconds=5,operations=events,transfer_totals=totals,byte_semantics=checker.BYTE_SEMANTICS,error=error,cleanup_error=None,bundle=str(bundle),elf=str(k['elf'].relative_to(bundle)),baud=settings['baud'],timeout_seconds=settings['timeout'],core_clock_hz=settings['core_clock_hz'],clock_source=settings['clock_source'])
    (logs/'summary.json').write_text(json.dumps(summary))
    (logs/'transport.jsonl').write_text(''.join(json.dumps(e,sort_keys=True)+'\n' for e in events))
    (logs/'run.log').write_text(''.join('> '+e['command']+'\n'+json.dumps(e,sort_keys=True)+'\n' for e in events))
    (logs/'openocd.log').write_text(openocd_builds()[0]['banner']+'\n')
    (logs/'openocd-command.json').write_text(json.dumps([checker.OPENOCD_PATH,'-c','gdb_port disabled','-c','telnet_port disabled','-c','tcl_port 29666','-f',str(k['config']),'-c','adapter speed 10000']))
    capture=logs/'terminal.txt'
    capture.write_text('KIT FAIL '+error+'\nrc=1\n' if negative else 'KIT PASS UART and dumped bytes match frozen reference\nrc=0\n')
    return logs,capture


class CheckerTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)

    def test_generic_positive_and_expected_fail_negative(self):
        for negative in (False,True):
            with self.subTest(negative=negative), tempfile.TemporaryDirectory() as tmp:
                bundle=Path(tmp);synthetic(bundle)
                logs,capture=receipts(self.root,bundle,negative)
                r=checker.check(capture,logs,bundle=bundle,negative=negative)
                self.assertEqual(r.verdict,'PASS',r.checks)
                for p in logs.iterdir(): p.unlink()
                logs.rmdir()

    def test_every_byte_completion_hash_and_summary_are_required(self):
        bundle=self.root/'kit';bundle.mkdir();synthetic(bundle)
        logs,capture=receipts(self.root,bundle)
        def result(): return checker.check(capture,logs,bundle=bundle)
        for name,changed in [('output-1.bin',b'xxxx'),('uart.log',(logs/'uart.log').read_bytes().replace(b'KIT COMPLETE\n',b'')),('uart.log',(logs/'uart.log').read_bytes().replace(b'STATUS 0 ',b'STATUS 1 ')),('uart.log',(logs/'uart.log').read_bytes().replace(b'HASH result1 ',b'HASH other ')),('summary.json',b'{}')]:
            p=logs/name;before=p.read_bytes();p.write_bytes(changed)
            with self.subTest(name=name): self.assertEqual(result().verdict,'FAIL')
            p.write_bytes(before)
        self.assertEqual(result().verdict,'PASS')
        (bundle/'oracle/a.bin').write_bytes(b'drift')
        self.assertEqual(result().verdict,'FAIL')

    def test_negative_must_match_predicted_bytes(self):
        bundle=self.root/'kit';bundle.mkdir();synthetic(bundle)
        logs,capture=receipts(self.root,bundle,True)
        (logs/'output-0.bin').write_bytes(b'xxxx')
        r=checker.check(capture,logs,bundle=bundle,negative=True)
        self.assertEqual(r.verdict,'FAIL')

    def test_missing_receipt_is_incomplete_and_inputs_are_preserved(self):
        bundle=self.root/'kit';bundle.mkdir();synthetic(bundle)
        logs,capture=receipts(self.root,bundle)
        before={p.name:(p.read_bytes(),p.stat().st_mtime_ns) for p in logs.iterdir()}
        self.assertEqual(checker.check(capture,logs,bundle=bundle).verdict,'PASS')
        self.assertEqual(before,{p.name:(p.read_bytes(),p.stat().st_mtime_ns) for p in logs.iterdir()})
        (logs/'summary.json').unlink()
        self.assertEqual(checker.check(capture,logs,bundle=bundle).verdict,'INCOMPLETE')
    def test_dry_run_authenticates_bundle_without_evidence(self):
        import contextlib, io
        bundle=self.root/'kit';bundle.mkdir();synthetic(bundle)
        def run(*argv):
            out=io.StringIO()
            with contextlib.redirect_stdout(out): code=checker.main(['kit',*argv,'--bundle',str(bundle),'--dry-run'])
            return code,out.getvalue()
        for flags in ([],['--negative']):
            with self.subTest(flags=flags):
                self.assertEqual(run(*flags),(0,'KIT CHECKER DRY PASS authenticated preflight; no execution evidence checked\n'))
        for flags in (['capture.txt'],['--logs',str(self.root)],['--run-stamp','20261117T000000Z']):
            with self.subTest(flags=flags), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit): run(*flags)
        (bundle/'oracle/a.bin').write_bytes(b'drift')
        code,text=run()
        self.assertEqual(code,1); self.assertTrue(text.startswith('KIT CHECKER DRY FAIL '))

    def test_openocd_banner_must_belong_to_a_declared_build(self):
        bundle=self.root/'kit';bundle.mkdir();synthetic(bundle)
        logs,capture=receipts(self.root,bundle)
        declared=openocd_builds()[0]['banner']
        self.assertEqual(checker.check(capture,logs,bundle=bundle,openocd_banner=declared).verdict,'PASS')
        self.assertEqual(checker.check(capture,logs,bundle=bundle,openocd_banner='Open On-Chip Debugger 0.12.0').verdict,'FAIL')
        for banner in ['Open On-Chip Debugger 0.12.0-gc66e1e0-dirty (2026-05-02-00:00)', declared.rsplit(' ',1)[0], declared+' ']:
            (logs/'openocd.log').write_text(banner+'\n')
            with self.subTest(banner=banner):
                r=checker.check(capture,logs,bundle=bundle)
                self.assertEqual(r.verdict,'FAIL')
                self.assertIn(('FAIL','OpenOCD version banner'),[x[:2] for x in r.checks])

if __name__=='__main__': unittest.main()
