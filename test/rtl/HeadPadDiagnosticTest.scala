package atlas.sa

import atlas.common.{MregParams, SystolicArrayParams}
import atlas.mxu.MxuOp
import chisel3._
import chisel3.simulator._
import org.scalatest.flatspec.AnyFlatSpec
import java.nio.file.{Files, Paths}

// Stage this test beside the selected RTL's SystolicArrayUnitHarness.
class HeadPadDiagnosticTest extends AnyFlatSpec with PeekPokeAPI {
  behavior of "selected MXU0 head padding"

  it should "keep source reduction values after a positive-zero prefix" in {
    val p = SystolicArrayParams()
    val helper = new SystolicArrayTest()
    EphemeralSimulator.simulate(new SystolicArrayUnitHarness(p, MregParams())) { dut =>
      helper.idle(dut)
      dut.reset.poke(true.B)
      dut.clock.step(5)
      dut.reset.poke(false.B)
      dut.clock.step()
      helper.idle(dut)

      val zero = 0x00
      val one = 0x38
      val minusOne = 0xB8
      val tiny = 0x18
      val directed = Seq(
        ("all_ones", Seq.fill(31)(one), Seq.fill(31)(one), 0x41f8),
        ("ordered_rounding", Seq(one, tiny, tiny) ++ Seq.fill(28)(zero),
         Seq(one, tiny, tiny) ++ Seq.fill(28)(zero), 0x3f80),
        ("negative_first", Seq(one) ++ Seq.fill(30)(zero),
         Seq(minusOne) ++ Seq.fill(30)(zero), 0xbf80),
        ("negative_zero_products", Seq(zero, one) ++ Seq.fill(29)(zero),
         Seq(minusOne, one) ++ Seq.fill(29)(zero), 0x3f80)
      ).map { case (name, sourceA, sourceW, expected) =>
        (name, Seq(zero) ++ sourceA, Seq(zero) ++ sourceW, expected)
      }
      val generated = sys.env.get("ATLAS_MXU0_HEAD_PAD_FIXTURE").toSeq.flatMap { directory =>
        val bytes = Files.readAllBytes(Paths.get(directory).resolve("head-pad-cases.bin"))
        require(bytes.length % 66 == 0 && bytes.length >= 32 * 66,
          "head-pad fixture needs at least 32 complete cases")
        (0 until bytes.length / 66).map { index =>
          val offset = index * 66
          val activation = bytes.slice(offset, offset + 32).map(_ & 0xff).toSeq
          val weight = bytes.slice(offset + 32, offset + 64).map(_ & 0xff).toSeq
          val expected = (bytes(offset + 64) & 0xff) | ((bytes(offset + 65) & 0xff) << 8)
          (s"generated_$index", activation, weight, expected)
        }
      }
      val cases = directed ++ generated
      for ((name, activation, weight, expected) <- cases) {
        assert(activation.size == 32 && weight.size == 32)
        val aRow = helper.packElems(activation, 8)
        val wRow = helper.packElems(weight, 8)
        for (row <- 0 until p.accSize) {
          helper.trfWriteRow(dut, helper.ACT_BANK, row, aRow)
          helper.trfWriteRow(dut, helper.WGT_BANK, row, wRow)
        }
        helper.sendCmd(dut, MxuOp.PushWeight, mregId = helper.WGT_BANK)
        helper.waitDataIdle(dut)
        helper.sendCmd(dut, MxuOp.Matmul, mregId = helper.ACT_BANK)
        helper.waitComputeIdle(dut)
        helper.sendCmd(dut, MxuOp.PopAccBF16, mregId = helper.OUT_BANK_LO)
        helper.waitDataIdle(dut)
        val result = helper.readBF16Results(dut, p, helper.OUT_BANK_LO, 0)
        assert(result.forall(_ == expected),
          s"$name expected 0x${expected.toHexString}; got ${result.map(_.toHexString)}")
      }
      val continuation = sys.env.get("ATLAS_MXU0_HEAD_PAD_FIXTURE").toSeq.flatMap { directory =>
        val bytes = Files.readAllBytes(
          Paths.get(directory).resolve("head-pad-continuation-cases.bin"))
        require(bytes.length % 194 == 0 && bytes.length >= 16 * 194,
          "head-pad continuation fixture needs at least 16 complete cases")
        (0 until bytes.length / 194).map { index =>
          val offset = index * 194
          val activationTiles = (0 until 3).map { tile =>
            bytes.slice(offset + tile * 64, offset + tile * 64 + 32).map(_ & 0xff).toSeq
          }
          val weightTiles = (0 until 3).map { tile =>
            bytes.slice(offset + tile * 64 + 32, offset + tile * 64 + 64)
              .map(_ & 0xff).toSeq
          }
          val expected = (bytes(offset + 192) & 0xff) | ((bytes(offset + 193) & 0xff) << 8)
          (s"continuation_$index", activationTiles, weightTiles, expected)
        }
      }
      for ((name, activationTiles, weightTiles, expected) <- continuation) {
        for (tile <- 0 until 3) {
          val aRow = helper.packElems(activationTiles(tile), 8)
          val wRow = helper.packElems(weightTiles(tile), 8)
          for (row <- 0 until p.accSize) {
            helper.trfWriteRow(dut, helper.ACT_BANK, row, aRow)
            helper.trfWriteRow(dut, helper.WGT_BANK, row, wRow)
          }
          helper.sendCmd(dut, MxuOp.PushWeight, mregId = helper.WGT_BANK)
          helper.waitDataIdle(dut)
          helper.sendCmd(dut, if (tile == 0) MxuOp.Matmul else MxuOp.MatmulAcc,
            mregId = helper.ACT_BANK)
          helper.waitComputeIdle(dut)
        }
        helper.sendCmd(dut, MxuOp.PopAccBF16, mregId = helper.OUT_BANK_LO)
        helper.waitDataIdle(dut)
        val result = helper.readBF16Results(dut, p, helper.OUT_BANK_LO, 0)
        assert(result.forall(_ == expected),
          s"$name expected 0x${expected.toHexString}; got ${result.map(_.toHexString)}")
      }
      println(s"MXU0 head-pad cases checked: ${cases.size}; K65 continuation cases: ${continuation.size}")
    }
  }
}
