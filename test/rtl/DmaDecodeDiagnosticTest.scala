package atlas.scalar

import chisel3._
import chisel3.simulator._
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.file.{Files, Paths}
import java.security.MessageDigest
import org.scalatest.flatspec.AnyFlatSpec

// Stage beside the selected RTL's ScalarDecoder; production RTL is unchanged.
class DmaDecodeDiagnosticTest extends AnyFlatSpec with PeekPokeAPI {
  behavior of "selected Atlas DMA instruction decode"

  private def check(dut: ScalarDecoder, word: Long, legal: Boolean, dma: Int): Unit = {
    dut.io.instr.poke(BigInt(word).U)
    dut.clock.step()
    val actualLegal = dut.io.decoded.legal.peek().litValue == 1
    val actualDma = dut.io.decoded.dma_cmd.peek().litValue.toInt
    assert(actualLegal == legal,
      f"word 0x$word%08x legal=$actualLegal expected=$legal")
    assert(actualDma == dma,
      f"word 0x$word%08x DMA command=$actualDma expected=$dma")
  }

  it should "distinguish config, wait, and their reserved fields" in {
    EphemeralSimulator.simulate(new ScalarDecoder) { dut =>
      for (channel <- 0 until 8; baseReg <- 0 until 32) {
        val config = (baseReg.toLong << 15) | (channel.toLong << 12) | 0x7fL
        check(dut, config, legal = true, dma = 3)
        assert(dut.io.decoded.rs1.peek().litValue == baseReg)
        assert(dut.io.decoded.funct3.peek().litValue == channel)
      }
      for (channel <- 0 until 8) {
        val wait = 0x0200007fL | (channel.toLong << 12)
        check(dut, wait, legal = true, dma = 4)
        check(dut, wait | (1L << 15), legal = false, dma = 0)
        check(dut, wait | (1L << 20), legal = false, dma = 0)
      }
      check(dut, 0x0200007fL, legal = true, dma = 4)
      check(dut, 0x0000007fL, legal = true, dma = 3)
    }
  }

  it should "decode every instruction in the compiled crossed-tail program" in {
    val file = Paths.get(sys.env.getOrElse("ATLAS_DMA_DECODE_PROGRAM",
      fail("set ATLAS_DMA_DECODE_PROGRAM to the verified crossed-tail program.bin")))
    val bytes = Files.readAllBytes(file)
    val digest = MessageDigest.getInstance("SHA-256").digest(bytes)
      .map(byte => f"${byte & 0xff}%02x").mkString
    assert(digest == "848c70a4aa373e89a50c3a2984f753e9b470e6a21d7f5c17b34fec45badda95c",
      "compiled program identity differs from the reviewed fixture")
    assert(bytes.length == 38 * 4)
    val words = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
    val dmaByWord = Map(0 -> 3, 5 -> 1, 6 -> 4, 10 -> 1, 11 -> 4,
      25 -> 2, 26 -> 4, 33 -> 2, 34 -> 4)
    EphemeralSimulator.simulate(new ScalarDecoder) { dut =>
      for (index <- 0 until 38) {
        val word = words.getInt() & 0xffffffffL
        check(dut, word, legal = true, dma = dmaByWord.getOrElse(index, 0))
      }
    }
  }
}
