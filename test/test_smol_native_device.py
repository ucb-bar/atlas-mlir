"""Native reply device checks exercise the unchanged host ownership protocol."""
import ctypes as ct
import errno
import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).absolute().parents[1]
spec = importlib.util.spec_from_file_location('smol_native_check',ROOT/'tools/smol_native_support.py')
checker = importlib.util.module_from_spec(spec);spec.loader.exec_module(checker)


class DeviceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix='smol-native-test-')
        root = Path(cls.temporary.name)
        shim = root/'test.c'
        shim.write_text('''#include "atlas_host.h"
#include "smol_native_device.h"
static atlas_host host;
static int rejected;
static int reply(uint32_t index,const uint32_t *words,size_t count,uint32_t sequence,void *context) {
    (void)context;
    static const uint32_t expected[]={0x13,0xc1102ff3,0x13,0xc10f9073,0x13,0x73};
    if(index||sequence!=1||count!=6)return 1;
    for(size_t i=0;i<count;++i)if(words[i]!=expected[i])return 1;
    return rejected;
}
int exercise(int fault) {
    rejected=fault;
    smol_native_configure(ATLAS_IMEM_BASE,ATLAS_CSR_BASE,ATLAS_IMEM_BYTES,reply,0);
    atlas_host_map map={ATLAS_IMEM_BASE,ATLAS_CSR_BASE,ATLAS_IMEM_BYTES};
    atlas_host_mmio io={atlas_host_direct_read32,atlas_host_direct_write32,0};
    atlas_host_memory memory={1,0,0};uint32_t program[]={0x73};
    atlas_host_buffer buffer={0x100000000,64,ATLAS_HOST_DEVICE_READ_WRITE};
    if(atlas_host_init(&host,map,io,memory)||atlas_host_load_raw(&host,program,1)
        ||atlas_host_start(&host,&buffer,1))return -1;
    return atlas_host_wait(&host,2);
}
int active(void){return host.active;}
unsigned sequence(void){return host.invocation_sequence;}
''')
        library = root/'device.so'
        subprocess.run(['/usr/bin/gcc','-std=c11','-O2','-fPIC','-shared','-Wall','-Wextra','-Werror',
                        '-I'+str(ROOT/'runtime'),'-I'+str(ROOT/'tools/host_iss'),str(shim),
                        str(ROOT/'runtime/atlas_host.c'),str(ROOT/'tools/host_iss/smol_native_device.c'),
                        '-Wl,--wrap=atlas_host_direct_read32','-Wl,--wrap=atlas_host_direct_write32',
                        '-Wl,-Bsymbolic','-o',str(library)],check=True)
        cls.library = ct.CDLL(str(library))

    @classmethod
    def tearDownClass(cls): cls.temporary.cleanup()

    def test_fixed_arena_refuses_existing_mapping_without_overwrite(self):
        memory = ct.CDLL(None)
        memory.mmap.argtypes = [ct.c_void_p,ct.c_size_t,ct.c_int,ct.c_int,ct.c_int,ct.c_size_t]
        memory.mmap.restype = ct.c_void_p
        memory.munmap.argtypes = [ct.c_void_p,ct.c_size_t]
        address = memory.mmap(None,4096,3,0x22,-1,0)
        self.assertNotEqual(address,ct.c_void_p(-1).value)
        self.library.smol_native_map.argtypes = [ct.c_size_t,ct.c_size_t]
        try:
            ct.c_ubyte.from_address(address).value = 0x6d
            self.assertEqual(self.library.smol_native_map(address,4096),errno.EEXIST)
            self.assertEqual(ct.c_ubyte.from_address(address).value,0x6d)
        finally:
            self.assertEqual(memory.munmap(address,4096),0)

    def test_wrapped_epilogue_sequence_and_success_release(self):
        self.assertEqual(self.library.exercise(0),0)
        self.assertEqual(self.library.smol_native_starts(),1)
        self.assertEqual(self.library.smol_native_error(),0)
        self.assertEqual((self.library.active(),self.library.sequence()),(0,1))

    def test_rejected_reply_is_fault_and_retains_ownership(self):
        self.assertEqual(self.library.exercise(1),5)
        self.assertEqual(self.library.smol_native_starts(),1)
        self.assertEqual((self.library.active(),self.library.sequence()),(1,1))


class StateTest(unittest.TestCase):
    def test_actual_traversal_and_ownership_mutations_fail_closed(self):
        state = dict(panels=2715,calls=1421,failed=0,result=0,sequence=2715,active=0,buffers=0,
                     buffer_pointer=0,acknowledged=0,image=1,graph=[1,10,1273,0],vision=[1,74,1,0]*2)
        checker.check_state(state)
        for key,value in [('panels',2714),('calls',1420),('active',1),('sequence',2714),
                          ('graph',[1,9,1273,0]),('vision',[1,74,1,0,1,73,1,0])]:
            with self.subTest(key=key),self.assertRaises(ValueError):checker.check_state(dict(state,**{key:value}))


if __name__ == '__main__': unittest.main()
