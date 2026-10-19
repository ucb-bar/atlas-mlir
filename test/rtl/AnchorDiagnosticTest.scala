package atlas.ipt

import chisel3._
import chisel3.simulator._
import org.scalatest.flatspec.AnyFlatSpec
import atlas.common.InnerProductTreeParams
import java.nio.file.{Files, Paths}

// Stage this test in a disposable checkout of the selected Atlas RTL source.
class AnchorDiagnosticTest extends AnyFlatSpec with PeekPokeAPI {
  behavior of "selected MXU1 anchor lane"

  private def checkTileFixture(root: java.nio.file.Path, expectedFile: String): Unit = {
    val activations = Files.readAllBytes(root.resolve("activation.bin"))
    val weights = Files.readAllBytes(root.resolve("weight.bin"))
    val expected = Files.readAllBytes(root.resolve(expectedFile))
    assert(activations.length == 1024 && weights.length == 1024 && expected.length == 2048)

    EphemeralSimulator.simulate(new AnchorAccumulationTree(InnerProductTreeParams())) { dut =>
      dut.io.bufReadSel.poke(false.B)
      dut.io.accumulate.poke(false.B)
      dut.io.psum.poke(0.U)
      for (index <- 0 until 32) dut.io.weightBuf1(index).poke(0.U)
      for (row <- 0 until 32) {
        for (index <- 0 until 32)
          dut.io.act(index).poke((activations(row * 32 + index) & 0xff).U)
        for (column <- 0 until 32) {
          for (index <- 0 until 32)
            dut.io.weightBuf0(index).poke((weights(column * 32 + index) & 0xff).U)
          dut.clock.step(3)
          val at = 2 * (row * 32 + column)
          val bits = (expected(at) & 0xff) | ((expected(at + 1) & 0xff) << 8)
          assert(dut.io.out.peek().litValue == bits,
            s"MXU1 reset cell ($row,$column): got ${dut.io.out.peek().litValue}, expected $bits")
        }
      }
    }
  }

  it should "distinguish exact accumulation from truncated anchor alignment" in {
    EphemeralSimulator.simulate(new AnchorAccumulationTree(InnerProductTreeParams())) { dut =>
      def check(weights: Seq[Int], acts: Seq[Int], expected: Int): Unit = {
        require(weights.length == 32 && acts.length == 32)
        for (index <- 0 until 32) {
          dut.io.weightBuf0(index).poke(weights(index).U)
          dut.io.weightBuf1(index).poke(0.U)
          dut.io.act(index).poke(acts(index).U)
        }
        dut.io.bufReadSel.poke(false.B)
        dut.io.accumulate.poke(false.B)
        dut.io.psum.poke(0.U)
        dut.clock.step(3)
        assert(dut.io.out.peek().litValue == expected)
      }

      val one = 0x38
      val tiny = 0x08
      val tieWeights = Seq(one, 0x18, 0x18) ++ Seq.fill(29)(0)
      val tieActs = Seq(one, 0x18, 0x18) ++ Seq.fill(29)(0)
      check(tieWeights, tieActs, 0x3f81)

      val cancelledWeights = Seq(0x77, 0xf7) ++ Seq.fill(30)(tiny)
      val cancelledActs = Seq(0x77, 0x77) ++ Seq.fill(30)(tiny)
      check(cancelledWeights, cancelledActs, 0)
    }
  }

  it should "match the independent exact BF16 result on the captured varied tile" in {
    val root = Paths.get(sys.env.getOrElse("ATLAS_MXU1_TILE_FIXTURE",
      cancel("set ATLAS_MXU1_TILE_FIXTURE to a prepared 32x32 diagnostic folder")))
    checkTileFixture(root, "exact-bf16.bin")
  }

  it should "match the source anchor mirror across a broad finite-normal tile" in {
    val root = Paths.get(sys.env.getOrElse("ATLAS_MXU1_BROAD_FIXTURE",
      cancel("set ATLAS_MXU1_BROAD_FIXTURE to a prepared 32x32 diagnostic folder")))
    checkTileFixture(root, "source-bf16.bin")
  }
}
