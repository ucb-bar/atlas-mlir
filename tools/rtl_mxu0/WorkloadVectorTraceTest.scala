package atlas.sota

import chisel3._
import chisel3.util._
import chisel3.simulator._
import atlas.common._
import atlas.scalar.VpuCmd
import atlas.vector.VectorEngineTop
import org.scalatest.flatspec.AnyFlatSpec
import java.nio.file.{Files, Paths}
import svsim.CommonCompilationSettings
import svsim.verilator.{Backend => VerilatorBackend}

class WorkloadVectorHarness extends Module {
  val p = MregParams()
  val io = IO(new Bundle {
    val cmd = Flipped(Valid(new VpuCmd))
    val readData = Input(Vec(2, UInt(p.mregRowBits.W)))
    val reads = Vec(2, Valid(new MregReadReq(p)))
    val writes = Vec(2, Valid(new MregWriteReq(p)))
    val busy = Output(Bool())
  })
  val engine = Module(new VectorEngineTop(VpuParams(), p))
  engine.io.cmd <> io.cmd
  io.reads(0) := engine.io.mregReadReq0
  io.reads(1) := engine.io.mregReadReq1
  io.writes(0) := engine.io.mregWriteReq0
  io.writes(1) := engine.io.mregWriteReq1
  engine.io.mregReadResp0.valid := RegNext(io.reads(0).valid, false.B)
  engine.io.mregReadResp1.valid := RegNext(io.reads(1).valid, false.B)
  engine.io.mregReadResp0.bits := RegNext(io.readData(0))
  engine.io.mregReadResp1.bits := RegNext(io.readData(1))
  io.busy := engine.io.busy
}

object PersistentVectorSimulator extends Simulator[VerilatorBackend] with PeekPokeAPI {
  override val backend = new VerilatorBackend(sys.env("ATLAS_VERILATOR_WRAPPER"))
  override val tag = "workload-vector"
  override val workspacePath = sys.env("ATLAS_SIM_WORKSPACE")
  override val commonCompilationSettings = CommonCompilationSettings(
    availableParallelism = CommonCompilationSettings.AvailableParallelism.UpTo(2))
  override val backendSpecificCompilationSettings = VerilatorBackend.CompilationSettings(
    disableFatalExitOnWarnings = true, outputSplit = Some(20000), outputSplitCFuncs = Some(20000))
}

class WorkloadVectorTraceTest extends AnyFlatSpec with PeekPokeAPI {
  it should "check every P2 pack tile and the in-place output rescale" in {
    val input = ujson.read(Files.readString(Paths.get(sys.env("ATLAS_TILE_INPUT"))))
    require(input("schema_version").num.toInt == 1)
    val tasks = input("tasks").arr.toSeq
    require(tasks.nonEmpty)
    val p = MregParams()
    require(p.mregRowBits == 256 && p.mregRows == 32)
    val allRuns = ujson.Arr()
    PersistentVectorSimulator.simulate(new WorkloadVectorHarness) { module =>
      val dut = module.wrapped
      for ((task, taskIndex) <- tasks.zipWithIndex; repeat <- 0 until 2) {
        val kind = task("kind").str
        require(kind == "pack" || kind == "rescale")
        val pack = kind == "pack"
        val source = task("source_bf16").arr.map(_.arr.map(_.num.toInt).toSeq).toSeq
        val expected = task(if (pack) "expected_fp8" else "expected_bf16").arr.map(_.arr.map(_.num.toInt).toSeq).toSeq
        require(source.size == 32 && source.forall(_.size == 32))
        require(expected.size == 32 && expected.forall(_.size == 32))
        val memory = Array.tabulate(p.numMreg, p.mregRows) { (bank, row) =>
          (0 until p.mregRowBytes).foldLeft(BigInt(0))((value, lane) =>
            value | (BigInt((bank * 7 + row * 3 + lane + 0x51 + (taskIndex * 2 + repeat) * 29) & 255) << (8 * lane)))
        }
        val sourceBank = if (pack) 0 else 2
        for (row <- 0 until 32; half <- 0 until 2) {
          memory(sourceBank + half)(row) = (0 until 16).foldLeft(BigInt(0))((value, lane) => {
            // Pack consumes a contiguous BF16 stream; MXU pop uses column halves.
            val linear = half * 512 + row * 16 + lane
            val item = if (pack) source(linear / 32)(linear % 32) else source(row)(half * 16 + lane)
            value | (BigInt(item) << (16 * lane))
          })
        }
        val sampleCycles = if (pack) 135 else 240
        val quietAfter = sampleCycles - 10
        val events = ujson.Arr()
        val commands = ujson.Arr()
        val writtenRows = scala.collection.mutable.Set.empty[(Int, Int)]
        var quiet = true
        dut.io.cmd.valid.poke(false.B)
        dut.io.cmd.bits.op.poke(17.U)
        dut.io.cmd.bits.vs1.poke(sourceBank.U)
        dut.io.cmd.bits.vs2.poke((if (pack) 0 else 4).U)
        dut.io.cmd.bits.vd.poke(2.U)
        dut.io.cmd.bits.scaleE8M0.poke((if (pack) task("scale_code").num.toInt else 127).U)
        dut.io.cmd.bits.imm.poke((if (pack) 0 else task("constant_bf16").num.toInt).U)
        dut.io.readData.foreach(_.poke(0.U))
        dut.reset.poke(true.B)
        dut.clock.step(2)
        dut.reset.poke(false.B)
        for (cycle <- 1 to sampleCycles) {
          val issue = cycle == 1 || (!pack && cycle == 100)
          dut.io.cmd.valid.poke(issue.B)
          if (issue) {
            val op = if (pack) 17 else if (cycle == 1) 30 else 3
            val destination = if (!pack && cycle == 1) 4 else 2
            dut.io.cmd.bits.op.poke(op.U)
            dut.io.cmd.bits.vd.poke(destination.U)
            commands.value += ujson.Obj("cycle" -> cycle, "op" -> op, "destination" -> destination,
              "vs1" -> sourceBank, "vs2" -> (if (pack) 0 else 4),
              "scale_code" -> (if (pack) task("scale_code").num.toInt else 127),
              "immediate" -> (if (pack) 0 else task("constant_bf16").num.toInt))
          }
          val reads = ujson.Arr()
          val writes = ujson.Arr()
          for (port <- 0 until 2) {
            val read = dut.io.reads(port)
            if (read.valid.peek().litToBoolean) {
              val bank = read.bits.mregId.peek().litValue.toInt
              val row = read.bits.row.peek().litValue.toInt
              require((if (pack) Set(0, 1) else Set(2, 3, 4, 5))(bank), "Unexpected source bank")
              val data = memory(bank)(row)
              reads.value += ujson.Obj("port" -> port, "mreg" -> bank, "row" -> row, "data_hex" -> data.toString(16))
              dut.io.readData(port).poke(data.U)
            } else dut.io.readData(port).poke(0.U)
          }
          for (port <- 0 until 2) {
            val write = dut.io.writes(port)
            if (write.valid.peek().litToBoolean) {
              val bank = write.bits.mregId.peek().litValue.toInt
              val row = write.bits.row.peek().litValue.toInt
              val data = write.bits.data.peek().litValue
              require((if (pack) Set(2) else Set(2, 3, 4, 5))(bank), "Unexpected output/guard write")
              require(!writtenRows((bank, row)), "Duplicate bank/row write")
              writtenRows += ((bank, row))
              memory(bank)(row) = data
              writes.value += ujson.Obj("port" -> port, "mreg" -> bank, "row" -> row, "data_hex" -> data.toString(16))
            }
          }
          events.value += ujson.Obj("cycle" -> cycle, "reads" -> reads, "writes" -> writes)
          if (cycle > quietAfter && (reads.value.nonEmpty || writes.value.nonEmpty)) quiet = false
          dut.clock.step()
        }
        val bits = if (pack) 8 else 16
        val mask = (BigInt(1) << bits) - 1
        val actual = (0 until 32).map(row => (0 until 32).map(column => {
          val bank = if (pack) 2 else 2 + column / 16
          val lane = if (pack) column else column % 16
          ((memory(bank)(row) >> (bits * lane)) & mask).toInt
        }))
        val mismatches = for (row <- 0 until 32; column <- 0 until 32 if actual(row)(column) != expected(row)(column)) yield
          ujson.Obj("row" -> row, "column" -> column, "actual" -> actual(row)(column), "expected" -> expected(row)(column))
        val expectedBanks = if (pack) Seq(2) else Seq(2, 3, 4, 5)
        val extent = writtenRows.toSet == (for (bank <- expectedBanks; row <- 0 until 32) yield (bank, row)).toSet
        val constantPreserved = pack || (4 to 5).forall(bank => (0 until 32).forall(row =>
          (0 until 16).forall(lane => ((memory(bank)(row) >> (16 * lane)) & 65535).toInt == task("constant_bf16").num.toInt)))
        val idle = !dut.io.busy.peek().litToBoolean
        allRuns.value += ujson.Obj("task_id" -> task("id"), "case_id" -> task("case_id"), "kind" -> kind,
          "repeat" -> repeat, "reset_cycles" -> 2, "initial_pattern" -> (taskIndex * 2 + repeat), "commands" -> commands,
          "events" -> events, "sample_cycles" -> sampleCycles, "quiet_after_cycle" -> quietAfter,
          "quiet_after_margin" -> quiet, "unique_output_rows" -> writtenRows.size, "extent_exact" -> extent,
          "constant_exact" -> constantPreserved, "idle_at_end" -> idle,
          "actual_words" -> ujson.Arr.from(actual.map(ujson.Arr.from(_))), "word_bits" -> bits,
          "mismatches" -> ujson.Arr.from(mismatches))
        val failed = !extent || !constantPreserved || !idle || !quiet || mismatches.nonEmpty
        if (failed || allRuns.value.size % 16 == 0 || allRuns.value.size == tasks.size * 2) {
          Files.writeString(Paths.get(sys.env("ATLAS_TILE_TRACE")), ujson.write(allRuns, indent=2) + "\n")
        }
        assert(extent && constantPreserved && idle && quiet, "VPU extent, guard, constant or completion gate failed")
        assert(mismatches.isEmpty, s"${task("id").str} differs on ${mismatches.size}/1024 words")
      }
    }.result
  }
}
