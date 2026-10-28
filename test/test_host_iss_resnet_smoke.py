"""First-ten smoke controls cannot become whole-graph qualification."""
from copy import deepcopy
import io
from contextlib import redirect_stderr
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

TOOLS=Path(__file__).absolute().parents[1]/'tools'
sys.path[:0]=[str(TOOLS),str(TOOLS/'host_iss')]
from atlas_tooling import sha as digest
from resnet_smoke import (COUNT,ATLAS_CSR_BASE,ATLAS_CSR_EXEC_CONTROL,arguments,store_or_stop,uart_prefix,verify_ledgers)


class StartBoundaryTest(unittest.TestCase):
    def test_real_dispatch_is_never_called_for_eleventh_start(self):
        device=SimpleNamespace(completed=[])
        def start(address,value):device.completed.append(object())
        device.write32=Mock(side_effect=start)
        for index in range(11):
            stopped=store_or_stop(device,ATLAS_CSR_BASE+ATLAS_CSR_EXEC_CONTROL,1)
            self.assertEqual(stopped,index==10)
        self.assertEqual(len(device.completed),COUNT);self.assertEqual(device.write32.call_count,COUNT)
        device.write32=Mock()
        self.assertFalse(store_or_stop(device,0x10000000,17));device.write32.assert_called_once()
        with self.assertRaises(ValueError):store_or_stop(device,ATLAS_CSR_BASE+ATLAS_CSR_EXEC_CONTROL,0)
        device.completed.append(object())
        with self.assertRaises(ValueError):store_or_stop(device,ATLAS_CSR_BASE+ATLAS_CSR_EXEC_CONTROL,1)


class PrefixTest(unittest.TestCase):
    def transcript(self):
        lines=['ATLAS_KERNEL_BEGIN name=resnet50-full launches_expected=54 launches_completed=0']
        lines.extend(f'ATLAS_KERNEL_PROGRESS name=resnet50-full launches_expected=54 launches_completed={i} boundary=layer' for i in range(1,11))
        return ('\n'.join(lines)+'\n').encode()

    def test_exact_prefix_and_each_completion_or_identity_error(self):
        raw=self.transcript();self.assertEqual(len(uart_prefix(raw,'resnet50-full')),11)
        for mutation in (raw.replace(b'completed=8 ',b'completed=9 '),raw.replace(b'expected=54',b'expected=10'),
            raw.replace(b'name=resnet50-full',b'name=other'),b'\n'.join(raw.splitlines()[:-1]),
            raw+b'ATLAS_KERNEL_PASS name=resnet50-full launches_expected=54 launches_completed=54\n',
            raw.replace(b'PROGRESS',b'FAIL',1),raw.replace(b'boundary=layer',b'boundary=',1)):
            with self.subTest(mutation=mutation[-100:]),self.assertRaises(ValueError):uart_prefix(mutation,'resnet50-full')

    def test_cli_requires_explicit_partial_selection_and_absolute_stop(self):
        required=['--elf','elf','--bundle-root','bundle','--model-root','model','--qemu','qemu','--out','out','--wall-seconds','4800','--stop-by-utc','2026-10-28T02:45:00+00:00']
        with redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):arguments(required)
        parsed=arguments(['--first-ten',*required]);self.assertTrue(parsed.first_ten)
        with redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):arguments(['--first-ten',*required[:-1],'2026-10-28T02:45:00'])


class LedgerTest(unittest.TestCase):
    def fixture(self):
        program,initial,expected=b'program',b'input',b'output'
        paths=dict(initial='initial',expected='expected',requested_dma=dict(load_bytes=32,store_bytes=32))
        rows=[dict(index=i,sequence=i+1,reset_count=1,exact=True,drained=True,cycles=4,
            program_sha256=digest(program),host_input_sha256=digest(initial),actual_output_sha256=digest(expected),
            expected_output_sha256=digest(expected),observed_dma=dict(load_bytes=32,store_bytes=32,response_beats=2)) for i in range(10)]
        state=SimpleNamespace(halted=True,halt_reason='ecall',check_flag=lambda channel:False,
                              read_csrf=lambda address:0 if address==0xC10 else 11)
        return SimpleNamespace(completed=rows,programs=[(None,program,paths)]*10,cycle_cap=20,
            bundle=SimpleNamespace(read={'initial':initial,'expected':expected}.__getitem__),
            core=SimpleNamespace(arch_state=state,exus=[]))

    def test_actual_hash_dma_drain_and_count_mutations_are_rejected(self):
        device=self.fixture();verify_ledgers(device)
        mutations=[('index',11),('sequence',11),('reset_count',2),('exact',False),('drained',False),('cycles',0),
            ('program_sha256','wrong'),('host_input_sha256','wrong'),('actual_output_sha256','wrong'),
            ('observed_dma',dict(load_bytes=32,store_bytes=32,response_beats=1))]
        for key,value in mutations:
            changed=self.fixture();changed.completed[9][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):verify_ledgers(changed)
        for count in (9,11):
            changed=self.fixture();changed.completed=changed.completed[:count] if count==9 else changed.completed+[deepcopy(changed.completed[-1])]
            with self.subTest(count=count),self.assertRaises(ValueError):verify_ledgers(changed)
        device.core.exus=[SimpleNamespace(has_in_flight=True)]
        with self.assertRaises(ValueError):verify_ledgers(device)
        device=self.fixture();device.core.arch_state.check_flag=lambda channel:channel==7
        with self.assertRaises(ValueError):verify_ledgers(device)


if __name__=='__main__':unittest.main()
