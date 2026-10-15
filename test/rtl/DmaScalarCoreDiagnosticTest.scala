package atlas.scalar

import atlas.common.VmemParams
import chisel3._
import chisel3.simulator._
import org.scalatest.flatspec.AnyFlatSpec

// Stage beside selected ScalarCore; the ROM models one-cycle IMEM read latency.
class DmaScalarCoreDiagnosticHarness(program: Seq[Long]) extends Module {
  val io = IO(new Bundle {
    val start = Input(Bool())
    val dmaBusy = Input(Bool())
    val dmaBase = Output(UInt(32.W))
    val dmaValid = Output(Bool())
    val dmaAddress = Output(UInt(64.W))
    val dmaSize = Output(UInt(32.W))
    val dmaChannel = Output(UInt(3.W))
    val lsuValid = Output(Bool())
    val lsuOp = Output(UInt(2.W))
    val lsuBank = Output(UInt(6.W))
    val mxu0Valid = Output(Bool())
    val mxu0Op = Output(UInt(4.W))
    val mxu0Bank = Output(UInt(6.W))
    val mxu1Valid = Output(Bool())
    val vpuValid = Output(Bool())
    val xluValid = Output(Bool())
    val retireValid = Output(Bool())
    val fetchPc = Output(UInt(32.W))
    val ecall = Output(Bool())
    val halted = Output(Bool())
    val illegal = Output(Bool())
  })

  val core = Module(new ScalarCore(VmemParams()))
  val rom = VecInit(program.map(word => BigInt(word).U(32.W)))
  val fetchAddr = RegNext(core.io.imemFetch.addr, 0.U)
  core.io.imemFetch.rdata := Mux(
    fetchAddr < program.length.U, rom(fetchAddr), 0x00000073.U(32.W))
  core.io.execRun := io.start
  core.io.execRunWrite := io.start
  for (channel <- 0 until 8)
    core.io.dma_busy(channel) := (if (channel == 0) io.dmaBusy else false.B)
  core.io.vpu_status := 0.U.asTypeOf(new VpuStatus)
  core.io.lsu_scalar_busy := false.B
  core.io.lsu_vload_busy := false.B
  core.io.lsu_vstore_busy := false.B
  core.io.mregReadBusy := 0.U
  core.io.mregWriteBusy := 0.U
  core.io.scalarMemResp.valid := false.B
  core.io.scalarMemResp.bits := 0.U
  core.io.csrPort.rdata := 0.U
  io.dmaBase := core.io.dmaBaseReg
  io.dmaValid := core.io.dmaCmd.valid
  io.dmaAddress := core.io.dmaCmd.bits.addr
  io.dmaSize := core.io.dmaCmd.bits.size
  io.dmaChannel := core.io.dmaCmd.bits.channel
  io.lsuValid := core.io.lsuCmd.valid
  io.lsuOp := core.io.lsuCmd.bits.op
  io.lsuBank := core.io.lsuCmd.bits.mregBank
  io.mxu0Valid := core.io.mxu0Cmd.valid
  io.mxu0Op := core.io.mxu0Cmd.bits.op
  io.mxu0Bank := core.io.mxu0Cmd.bits.mregBank
  io.mxu1Valid := core.io.mxu1Cmd.valid
  io.vpuValid := core.io.vpuCmd.valid
  io.xluValid := core.io.xluCmd.valid
  io.retireValid := core.io.csrPort.inst_retire
  io.fetchPc := core.io.imemFetch.addr
  io.ecall := core.io.csrPort.set_ecall
  io.halted := core.io.halted
  io.illegal := core.io.csrPort.set_illegal
}

class DmaScalarCoreDiagnosticTest extends AnyFlatSpec with PeekPokeAPI {
  behavior of "selected ScalarCore DMA config and wait"

  private def run(program: Seq[Long], expectedBase: Int): Unit = {
    EphemeralSimulator.simulate(new DmaScalarCoreDiagnosticHarness(program)) { dut =>
      dut.io.start.poke(false.B)
      dut.io.dmaBusy.poke(true.B)
      dut.reset.poke(true.B)
      dut.clock.step(3)
      dut.reset.poke(false.B)
      dut.clock.step()
      dut.io.start.poke(true.B)
      dut.clock.step()
      dut.io.start.poke(false.B)
      dut.clock.step(10)
      assert(dut.io.halted.peek().litValue == 0,
        "channel-zero DMA.WAIT must hold the scalar core while busy")
      assert(dut.io.dmaBase.peek().litValue == expectedBase)
      assert(dut.io.illegal.peek().litValue == 0)
      dut.io.dmaBusy.poke(false.B)
      dut.clock.step(8)
      assert(dut.io.halted.peek().litValue == 1,
        "releasing DMA.WAIT must reach ECALL")
      assert(dut.io.illegal.peek().litValue == 0)
    }
  }

  it should "update the DMA base only for the RTL config word" in {
    val addiX1One = 0x00100093L
    val configFromX1 = 0x0000807fL
    val waitChannelZero = 0x0200007fL
    val ecall = 0x00000073L
    run(Seq(addiX1One, configFromX1, waitChannelZero, ecall), expectedBase = 1)
    run(Seq(addiX1One, waitChannelZero, ecall), expectedBase = 0)
  }

  it should "use the configured base when issuing a DMA load" in {
    val addiX1One = 0x00100093L
    val addiX2SixtyFour = 0x04000113L
    val addiX3Four = 0x00400193L
    val dmaLoad = 0x0031027bL
    val ecall = 0x00000073L
    for ((config, expectedAddress) <- Seq(
      (0x0000807fL, BigInt("100000040", 16)),
      (0x0200007fL, BigInt(64))
    )) {
      val program = Seq(addiX1One, config, addiX2SixtyFour,
        addiX3Four, dmaLoad, ecall)
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
        var launches = 0
        for (_ <- 0 until 16) {
          if (dut.io.dmaValid.peek().litValue == 1) {
            launches += 1
            assert(dut.io.dmaAddress.peek().litValue == expectedAddress)
            assert(dut.io.dmaSize.peek().litValue == 4)
            assert(dut.io.dmaChannel.peek().litValue == 0)
          }
          dut.clock.step()
        }
        assert(launches == 1)
        assert(dut.io.halted.peek().litValue == 1)
        assert(dut.io.illegal.peek().litValue == 0)
      }
    }
  }
}
