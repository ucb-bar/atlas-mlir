"""Native tile address options refuse invalid complete DMA spans."""
import unittest
from types import SimpleNamespace
from atlas_mlir_native.binding import _with_tile_io_base, _tile_io_base, DRAM_BASE
from merlin.semantic_compiler.target_binding import NativeCompilationError

class TileIOBaseTest(unittest.TestCase):
    def test_base_validation_and_invocation_isolation(self):
        calls=[]
        @_with_tile_io_base
        def emit(snapshot,request,**kwargs):
            calls.append(_tile_io_base.get())
            return calls[-1]
        request=SimpleNamespace(nodes=[SimpleNamespace(type=SimpleNamespace(shape=(32,32)))])
        args=dict(fixed_inputs={'a':0,'w':1},fixed_outputs=(2,))
        self.assertEqual(emit(None,request,**args),DRAM_BASE)
        self.assertEqual(emit(None,request,io_memory_base=0x0800b000,**args),0x0800b000)
        self.assertEqual(_tile_io_base.get(),DRAM_BASE)
        for base in [True,1.0,'0x0800b000',-2048,0x0800b001,0x07000000,0x0800f000,0xfffff800,1<<32]:
            with self.subTest(base=base), self.assertRaises(NativeCompilationError):
                emit(None,request,io_memory_base=base,**args)
        self.assertEqual(len(calls),2)
        with self.assertRaises(NativeCompilationError):
            emit(None,request,io_memory_base=0x0800b000,fixed_inputs={'a':10},fixed_outputs=(2,))
        with self.assertRaises(NativeCompilationError):
            emit(None,SimpleNamespace(nodes=[SimpleNamespace(type=SimpleNamespace(shape=(64,32)))]),io_memory_base=0x0800b000,**args)
        @_with_tile_io_base
        def fail(*args,**kwargs):raise RuntimeError('test')
        with self.assertRaises(RuntimeError):fail(None,request,io_memory_base=0x0800b000,**args)
        self.assertEqual(_tile_io_base.get(),DRAM_BASE)
