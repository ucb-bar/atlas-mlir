package atlas.sota

import chisel3._
import chisel3.util._
import chisel3.simulator._
import atlas.common._
import atlas.mxu.{MxuCmd, MxuOp}
import atlas.sa.SystolicArrayTop
import org.scalatest.flatspec.AnyFlatSpec
import java.nio.file.{Files, Paths}
import svsim.CommonCompilationSettings
import svsim.verilator.{Backend => VerilatorBackend}

class WorkloadMxuHarness extends Module {
  val p = MregParams()
  val io = IO(new Bundle {
    val cmd = Flipped(Valid(new MxuCmd(p.mregIdBits)))
    val readData = Input(Vec(2, UInt(p.mregRowBits.W)))
    val reads = Vec(2, Valid(new MregReadReq(p)))
    val writes = Vec(2, Valid(new MregWriteReq(p)))
  })
  val engine = Module(new SystolicArrayTop(SystolicArrayParams(), p))
  engine.io.cmd <> io.cmd
  io.reads(0) := engine.io.mregReadReq0
  io.reads(1) := engine.io.mregReadReq1
  io.writes(0) := engine.io.mregWriteReq0
  io.writes(1) := engine.io.mregWriteReq1
  engine.io.mregReadResp0.valid := RegNext(io.reads(0).valid, false.B)
  engine.io.mregReadResp1.valid := RegNext(io.reads(1).valid, false.B)
  engine.io.mregReadResp0.bits := RegNext(io.readData(0))
  engine.io.mregReadResp1.bits := RegNext(io.readData(1))
}

object PersistentWorkloadSimulator extends Simulator[VerilatorBackend] with PeekPokeAPI {
  override val backend = new VerilatorBackend(sys.env("ATLAS_VERILATOR_WRAPPER"))
  override val tag = "workload-mxu0"
  override val workspacePath = sys.env("ATLAS_SIM_WORKSPACE")
  override val commonCompilationSettings = CommonCompilationSettings(
    availableParallelism = CommonCompilationSettings.AvailableParallelism.UpTo(2))
  override val backendSpecificCompilationSettings = VerilatorBackend.CompilationSettings(
    disableFatalExitOnWarnings = true, outputSplit = Some(20000), outputSplitCFuncs = Some(20000))
}

class WorkloadMxuTraceTest extends AnyFlatSpec with PeekPokeAPI {
  it should "check source-bound encoded contractions and hardware refusal controls" in {
    val input = ujson.read(Files.readString(Paths.get(sys.env("ATLAS_TILE_INPUT"))))
    require(input("schema_version").num.toInt == 2)
    val cases = input("cases").arr.toSeq
    require(cases.nonEmpty && cases.size <= 8)
    val p = MregParams()
    val sa = SystolicArrayParams()
    require(p.mregRows == 32 && p.mregRowBits == 256 && sa.rows == 32 && sa.cols == 32)
    val allRuns = ujson.Arr()
    PersistentWorkloadSimulator.simulate(new WorkloadMxuHarness) { module =>
      val dut = module.wrapped
      for ((scenario, caseIndex) <- cases.zipWithIndex) {
        val name = scenario("id").str
        val a = scenario("A").arr.map(_.arr.map(_.num.toInt).toSeq).toSeq
        val b = scenario("B").arr.map(_.arr.map(_.num.toInt).toSeq).toSeq
        val expected = scenario("expected_bf16").arr.map(_.arr.map(_.num.toInt).toSeq).toSeq
        val k = a.head.size
        require(a.size == 32 && a.forall(_.size == k) && b.size == k && b.forall(_.size == 32))
        require(k > 0 && k <= 2048 && k % 32 == 0)
        require(a.flatten.forall(v => v >= 0 && v <= 255) && b.flatten.forall(v => v >= 0 && v <= 255))
        val tileCount = k / 32
        val popCycle = 36 + (tileCount - 1) * 200 + 128
        val sampleCycles = popCycle + 56
        val quietAfter = popCycle + 46
        val runs = Seq(("normal", 0), ("normal", 1)) ++
          scenario("controls").obj.keys.toSeq.sorted.map(kind => (kind, 0))
        for (((kind, repeat), runIndex) <- runs.zipWithIndex) {
          val declared = if (kind == "normal") expected else
            scenario("controls")(kind).arr.map(_.arr.map(_.num.toInt).toSeq).toSeq
          val pattern = caseIndex * 4 + runIndex
          val memory = Array.tabulate(p.numMreg, p.mregRows) { (bank, row) =>
            (0 until p.mregRowBytes).foldLeft(BigInt(0))((value, lane) =>
              value | (BigInt((bank * 7 + row * 3 + lane + 0x51 + pattern * 29) & 255) << (8 * lane)))
          }
          val events = ujson.Arr()
          val commands = ujson.Arr()
          var writesSeen = 0
          val writtenRows = scala.collection.mutable.Set.empty[(Int, Int)]
          var quietAfterMargin = true
          dut.io.cmd.valid.poke(false.B)
          dut.io.cmd.bits.op.poke(MxuOp.PushWeight)
          dut.io.cmd.bits.mregId.poke(0.U)
          dut.io.cmd.bits.accSel.poke(false.B)
          dut.io.cmd.bits.weightSlot.poke(false.B)
          dut.io.cmd.bits.scaleE8M0.poke(127.U)
          dut.io.readData.foreach(_.poke(0.U))
          dut.reset.poke(true.B)
          dut.clock.step(2)
          dut.reset.poke(false.B)
          // Fixed conservative spacing is diagnostic, not scalar-program timing.
          for (cycle <- 1 to sampleCycles) {
            var op: Option[(MxuOp.Type, Int)] = None
            if ((cycle - 1) % 200 == 0 && (cycle - 1) / 200 < tileCount) {
              val tile = (cycle - 1) / 200
              for (row <- 0 until 32) {
                memory(0)(row) = (0 until 32).foldLeft(BigInt(0))((v, inner) =>
                  v | (BigInt(a(row)(tile * 32 + inner)) << (8 * inner)))
                memory(1)(row) = (0 until 32).foldLeft(BigInt(0))((v, inner) => {
                  val value = if (kind == "wrong_transpose") b(tile * 32 + row)(inner) else b(tile * 32 + inner)(row)
                  v | (BigInt(value) << (8 * inner))
                })
              }
              op = Some((MxuOp.PushWeight, 1))
            } else if (cycle >= 36 && (cycle - 36) % 200 == 0 && (cycle - 36) / 200 < tileCount) {
              val tile = (cycle - 36) / 200
              val clear = tile == 0 || (kind == "wrong_final_reset" && tile == tileCount - 1)
              op = Some((if (clear) MxuOp.Matmul else MxuOp.MatmulAcc, 0))
            } else if (cycle == popCycle) op = Some((MxuOp.PopAccBF16, 2))
            dut.io.cmd.valid.poke(op.nonEmpty.B)
            op.foreach { case (command, bank) =>
              dut.io.cmd.bits.op.poke(command)
              dut.io.cmd.bits.mregId.poke(bank.U)
              commands.value += ujson.Obj("cycle" -> cycle, "op" -> command.litValue.toInt, "mreg" -> bank)
            }
            val reads = ujson.Arr()
            val writes = ujson.Arr()
            for (port <- 0 until 2) {
              val read = dut.io.reads(port)
              if (read.valid.peek().litToBoolean) {
                val bank = read.bits.mregId.peek().litValue.toInt
                val row = read.bits.row.peek().litValue.toInt
                val value = memory(bank)(row)
                reads.value += ujson.Obj("port" -> port, "mreg" -> bank, "row" -> row, "data_hex" -> value.toString(16))
                dut.io.readData(port).poke(value.U)
              } else dut.io.readData(port).poke(0.U)
            }
            for (port <- 0 until 2) {
              val write = dut.io.writes(port)
              if (write.valid.peek().litToBoolean) {
                val bank = write.bits.mregId.peek().litValue.toInt
                val row = write.bits.row.peek().litValue.toInt
                val value = write.bits.data.peek().litValue
                require(bank == 2 || bank == 3, "Unexpected MREG write or input/guard corruption")
                require(!writtenRows((bank, row)), "Duplicate output bank/row write")
                writtenRows += ((bank, row))
                memory(bank)(row) = value
                writesSeen += 1
                writes.value += ujson.Obj("port" -> port, "mreg" -> bank, "row" -> row, "data_hex" -> value.toString(16))
              }
            }
            events.value += ujson.Obj("cycle" -> cycle, "reads" -> reads, "writes" -> writes)
            if (cycle > quietAfter && (reads.value.nonEmpty || writes.value.nonEmpty)) quietAfterMargin = false
            dut.clock.step()
          }
          val actual = (0 until 32).map(row => (0 until 32).map(column =>
            ((memory(2 + column / 16)(row) >> (16 * (column % 16))) & 65535).toInt))
          def mismatches(reference: Seq[Seq[Int]]) = (for (row <- 0 until 32; column <- 0 until 32
              if actual(row)(column) != reference(row)(column)) yield
            ujson.Obj("row" -> row, "column" -> column, "actual" -> actual(row)(column), "expected" -> reference(row)(column)))
          val originalMismatches = mismatches(expected)
          val declaredMismatches = mismatches(declared)
          allRuns.value += ujson.Obj("case_id" -> name, "run_kind" -> kind, "repeat" -> repeat,
            "initial_pattern" -> pattern, "k" -> k, "sample_cycles" -> sampleCycles, "reset_cycles" -> 2,
            "commands" -> commands, "events" -> events, "mreg_write_rows" -> writesSeen,
            "actual_bf16" -> ujson.Arr.from(actual.map(ujson.Arr.from(_))),
            "unique_output_rows" -> writtenRows.size, "quiet_after_margin" -> quietAfterMargin,
            "quiet_after_cycle" -> quietAfter, "mismatches" -> ujson.Arr.from(originalMismatches),
            "declared_mismatches" -> ujson.Arr.from(declaredMismatches))
          Files.writeString(Paths.get(sys.env("ATLAS_TILE_TRACE")), ujson.write(allRuns, indent=2) + "\n")
          assert(writesSeen == 64 && writtenRows.toSet == (for (bank <- 2 to 3; row <- 0 until 32) yield (bank, row)).toSet)
          assert(quietAfterMargin, "Late MREG event after conservative completion margin")
          assert(declaredMismatches.isEmpty, s"$name/$kind differs from declared arithmetic")
          assert(originalMismatches.isEmpty == (kind == "normal"), s"$name/$kind did not show expected equality/refusal")
        }
      }
    }.result
  }
}
