package atlas.scalar

import chisel3._
import chisel3.simulator._
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.file.{Files, Paths}
import java.security.MessageDigest
import org.scalatest.flatspec.AnyFlatSpec
import scala.collection.mutable.ArrayBuffer

// Stage with DmaScalarCoreDiagnosticTest beside selected ScalarCore.
class CompiledProgramScalarCoreDiagnosticTest extends AnyFlatSpec with PeekPokeAPI {
  behavior of "selected ScalarCore on one Merlin-native program"

  it should "issue the checked command stream and halt at ECALL" in {
    val path = Paths.get(sys.env.getOrElse("ATLAS_DMA_DECODE_PROGRAM",
      fail("set ATLAS_DMA_DECODE_PROGRAM to the verified crossed-tail program.bin")))
    val bytes = Files.readAllBytes(path)
    val digest = MessageDigest.getInstance("SHA-256").digest(bytes)
      .map(byte => f"${byte & 0xff}%02x").mkString
    assert(digest == "848c70a4aa373e89a50c3a2984f753e9b470e6a21d7f5c17b34fec45badda95c")
    assert(bytes.length == 38 * 4)
    val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
    val program = (0 until 38).map(_ => buffer.getInt() & 0xffffffffL)

    EphemeralSimulator.simulate(new DmaScalarCoreDiagnosticHarness(program)) { dut =>
      dut.io.start.poke(false.B)
      dut.io.dmaBusy.poke(false.B)
      dut.reset.poke(true.B)
      dut.clock.step(3)
      dut.reset.poke(false.B)
      dut.clock.step()
      dut.io.start.poke(true.B)
      dut.clock.step()
      dut.io.start.poke(false.B)

      val fetchPcs = ArrayBuffer.empty[Int]
      var retired = 0
      val dma = ArrayBuffer.empty[(Int, Int, BigInt, BigInt)]
      val lsu = ArrayBuffer.empty[(Int, Int)]
      val mxu0 = ArrayBuffer.empty[(Int, Int)]
      var reachedEcall = false
      var cycles = 0
      while (!reachedEcall && cycles < 1000) {
        assert(dut.io.illegal.peek().litValue == 0,
          s"illegal instruction after $cycles cycles")
        fetchPcs += dut.io.fetchPc.peek().litValue.toInt
        if (dut.io.retireValid.peek().litValue == 1)
          retired += 1
        if (dut.io.dmaValid.peek().litValue == 1)
          dma += ((dut.io.dmaSize.peek().litValue.toInt,
                   dut.io.dmaChannel.peek().litValue.toInt,
                   dut.io.dmaAddress.peek().litValue,
                   dut.io.dmaBase.peek().litValue))
        if (dut.io.lsuValid.peek().litValue == 1)
          lsu += ((dut.io.lsuOp.peek().litValue.toInt,
                   dut.io.lsuBank.peek().litValue.toInt))
        if (dut.io.mxu0Valid.peek().litValue == 1)
          mxu0 += ((dut.io.mxu0Op.peek().litValue.toInt,
                    dut.io.mxu0Bank.peek().litValue.toInt))
        assert(dut.io.mxu1Valid.peek().litValue == 0)
        assert(dut.io.vpuValid.peek().litValue == 0)
        assert(dut.io.xluValid.peek().litValue == 0)
        reachedEcall = dut.io.ecall.peek().litValue == 1
        dut.clock.step()
        cycles += 1
      }
      assert(reachedEcall, "compiled stream did not reach ECALL within 1000 cycles")
      assert(dut.io.halted.peek().litValue == 1)
      assert(retired == 37, s"expected 37 non-ECALL retires, got $retired")
      assert(fetchPcs.distinct == (0 to 38),
        s"unexpected fetch path: ${fetchPcs.distinct}")
      assert(dma.toSeq == Seq(
        (1024, 0, BigInt("90001800", 16), BigInt(0)),
        (1024, 0, BigInt("90002000", 16), BigInt(0)),
        (1024, 1, BigInt("90011800", 16), BigInt(0)),
        (1024, 1, BigInt("90011c00", 16), BigInt(0))
      ), s"DMA commands: ${dma.toSeq}")
      assert(lsu.toSeq == Seq((1, 0), (1, 1), (2, 32), (2, 33)),
        s"LSU commands: ${lsu.toSeq}")
      assert(mxu0.toSeq == Seq((1, 1), (6, 0), (5, 32)),
        s"MXU0 commands: ${mxu0.toSeq}")
      println(s"Checked crossed-tail scalar stream through ECALL in $cycles cycles")
    }
  }
}
