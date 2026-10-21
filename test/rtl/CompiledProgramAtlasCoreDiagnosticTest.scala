package atlas.tile

import atlas.common.AtlasParams
import atlas.scalar.AtlasMemMap
import chisel3._
import chisel3.simulator._
import freechips.rocketchip.tilelink.{TLBundle, TLBundleParameters}
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.file.{Files, Paths}
import java.security.MessageDigest
import java.util.Properties
import org.scalatest.flatspec.AnyFlatSpec
import scala.collection.mutable

// Stage under src/test/scala/atlas/tile/ in the selected Atlas RTL checkout.
class CompiledProgramAtlasCoreDiagnosticTest extends AnyFlatSpec with PeekPokeAPI {
  behavior of "selected AtlasCore on one Merlin-native MXU0 program"

  private val hostParams = TLBundleParameters(
    addressBits = 64, dataBits = 32, sourceBits = 1, sinkBits = 1,
    sizeBits = 7, echoFields = Nil, requestFields = Nil,
    responseFields = Nil, hasBCE = false)
  private val dataParams = TLBundleParameters(
    addressBits = 64, dataBits = 256, sourceBits = 6, sinkBits = 1,
    sizeBits = 7, echoFields = Nil, requestFields = Nil,
    responseFields = Nil, hasBCE = false)

  private def idleHost(port: TLBundle): Unit = {
    port.a.valid.poke(false.B)
    port.d.ready.poke(true.B)
  }

  private def hostPut(dut: AtlasCore, port: TLBundle, address: Long, data: Long): Unit = {
    port.a.bits.opcode.poke(0.U) // TL PutFullData
    port.a.bits.param.poke(0.U)
    port.a.bits.size.poke(2.U)
    port.a.bits.source.poke(0.U)
    port.a.bits.address.poke(address.U)
    port.a.bits.mask.poke(15.U)
    port.a.bits.data.poke(data.U)
    port.a.bits.corrupt.poke(false.B)
    port.a.valid.poke(true.B)
    var waited = 0
    while (port.a.ready.peek().litValue == 0 && waited < 100) {
      dut.clock.step()
      waited += 1
    }
    assert(waited < 100, s"host TL write not accepted at $address")
    dut.clock.step()
    port.a.valid.poke(false.B)
    waited = 0
    while (port.d.valid.peek().litValue == 0 && waited < 100) {
      dut.clock.step()
      waited += 1
    }
    assert(waited < 100, s"host TL write has no response at $address")
    assert(port.d.bits.opcode.peek().litValue == 0)
    dut.clock.step()
  }

  private def hostGet(dut: AtlasCore, port: TLBundle, address: Long): BigInt = {
    port.a.bits.opcode.poke(4.U) // TL Get
    port.a.bits.param.poke(0.U)
    port.a.bits.size.poke(2.U)
    port.a.bits.source.poke(0.U)
    port.a.bits.address.poke(address.U)
    port.a.bits.mask.poke(15.U)
    port.a.bits.data.poke(0.U)
    port.a.bits.corrupt.poke(false.B)
    port.a.valid.poke(true.B)
    var waited = 0
    while (port.a.ready.peek().litValue == 0 && waited < 100) {
      dut.clock.step()
      waited += 1
    }
    assert(waited < 100, s"host TL read not accepted at $address")
    dut.clock.step()
    port.a.valid.poke(false.B)
    waited = 0
    while (port.d.valid.peek().litValue == 0 && waited < 100) {
      dut.clock.step()
      waited += 1
    }
    assert(waited < 100, s"host TL read has no response at $address")
    assert(port.d.bits.opcode.peek().litValue == 1)
    val data = port.d.bits.data.peek().litValue
    dut.clock.step()
    data
  }

  private case class Response(opcode: Int, size: BigInt, source: BigInt, data: BigInt)
  private case class Pending(response: Response, readyCycle: Int)
  private case class InputCase(name: String,
                               inputCode: (Int, Int) => Int,
                               expectedBF16: Int => Int,
                               mStart: Int = -1, nStart: Int = -1,
                               mxu1: Boolean = false, k64: Boolean = false,
                               programOverride: Seq[Long] = Seq.empty,
                               inputBasesOverride: Seq[Long] = Seq.empty,
                               outputBaseOverride: Long = 0L,
                               inputPayloadsOverride: Seq[Array[Byte]] = Seq.empty,
                               outputCells: Int = 1024,
                               inputReadCountsOverride: Seq[Int] = Seq.empty,
                               residentBuffers: Boolean = false,
                               cycleLimit: Int = 20000)

  it should "restart after ECALL and preserve the memory ABI" in {
    val path = Paths.get(sys.env.getOrElse("ATLAS_DMA_DECODE_PROGRAM",
      fail("set ATLAS_DMA_DECODE_PROGRAM to the verified crossed-tail program.bin")))
    val bytes = Files.readAllBytes(path)
    val digest = MessageDigest.getInstance("SHA-256").digest(bytes)
      .map(byte => f"${byte & 0xff}%02x").mkString
    assert(digest == "848c70a4aa373e89a50c3a2984f753e9b470e6a21d7f5c17b34fec45badda95c")
    assert(bytes.length == 38 * 4)
    val words = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
    val program = (0 until 38).map(_ => words.getInt() & 0xffffffffL)
    val mxu1Path = Paths.get(sys.env.getOrElse("ATLAS_MXU1_PROGRAM",
      fail("set ATLAS_MXU1_PROGRAM to the verified 39-word MXU1 program.bin")))
    val mxu1Bytes = Files.readAllBytes(mxu1Path)
    val mxu1Digest = MessageDigest.getInstance("SHA-256").digest(mxu1Bytes)
      .map(byte => f"${byte & 0xff}%02x").mkString
    assert(mxu1Digest == "9df497b39c02f9e6a153b9480231886408f074bf7829cd8effb712752d43622d")
    assert(mxu1Bytes.length == 39 * 4)
    val mxu1Words = ByteBuffer.wrap(mxu1Bytes).order(ByteOrder.LITTLE_ENDIAN)
    val mxu1Program = (0 until 39).map(_ => mxu1Words.getInt() & 0xffffffffL)
    val k64Path = Paths.get(sys.env.getOrElse("ATLAS_K64_PROGRAM",
      fail("set ATLAS_K64_PROGRAM to the verified 54-word K64 program.bin")))
    val k64Bytes = Files.readAllBytes(k64Path)
    val k64Digest = MessageDigest.getInstance("SHA-256").digest(k64Bytes)
      .map(byte => f"${byte & 0xff}%02x").mkString
    assert(k64Digest == "d365b77beb3bded04fca491eed614dad8105b5706768ec9463905c84ef0f32d8")
    assert(k64Bytes.length == 54 * 4)
    val k64Words = ByteBuffer.wrap(k64Bytes).order(ByteOrder.LITTLE_ENDIAN)
    val k64Program = (0 until 54).map(_ => k64Words.getInt() & 0xffffffffL)

    val simulator = new SingleBackendSimulator[svsim.verilator.Backend] {
      override val workspacePath = sys.env.getOrElse("ATLAS_CORE_TEST_WORKSPACE",
        fail("set ATLAS_CORE_TEST_WORKSPACE to a fresh scratch directory"))
      override val backend = svsim.verilator.Backend.initializeFromProcessEnvironment()
      override val tag = "default"
      override val commonCompilationSettings = svsim.CommonCompilationSettings()
      override val backendSpecificCompilationSettings =
        svsim.verilator.Backend.CompilationSettings()
      override val verbose = true
    }
    simulator.simulate(new AtlasCore(
      AtlasParams(), hostParams, hostParams, dataParams, dataParams)) { module =>
      val dut = module.wrapped
      idleHost(dut.io.imemTL)
      idleHost(dut.io.csrTL)
      idleHost(dut.io.vmemTL)
      dut.io.dmaTL.a.ready.poke(true.B)
      dut.io.dmaTL.d.valid.poke(false.B)
      dut.reset.poke(true.B)
      dut.clock.step(3)
      dut.reset.poke(false.B)
      dut.clock.step()

      val inputBases = Seq(0x90001800L, 0x90002000L)
      val k64InputBases = Seq(0x90001800L, 0x90002000L, 0x90002800L, 0x90003000L)
      val outputBases = Seq(0x90011800L, 0x90011c00L)
      val arithmeticCases = Seq(
        InputCase("zeros", (_, _) => 0, _ => 0),
        InputCase("unit_cell", (_, index) => if (index == 0) 0x38 else 0,
          cell => if (cell == 0) 0x3f80 else 0),
        InputCase("ones", (_, _) => 0x38, _ => 0x4200),
        InputCase("ordered_ties", (_, index) => index match {
          case 0 => 0x38
          case 1 | 2 => 0x18
          case _ => 0
        }, cell => if (cell == 0) 0x3f80 else 0))
      def tailBF16(m: Int, n: Int): Int = {
        val activationScale = if (m % 2 == 0) 1 else 2
        val weightScale = if (n % 3 == 0) 2 else 1
        (activationScale * weightScale) match {
          case 1 => 0x41f8 // 31 exact products of 1
          case 2 => 0x4278 // 31 exact products of 2
          case 4 => 0x42f8 // 31 exact products of 4
        }
      }
      val tailCases = Seq(
        ("tail_a", 0, 0), ("tail_b", 0, 32), ("tail_c", 0, 64),
        ("tail_d", 32, 0), ("tail_e", 32, 32), ("tail_f", 32, 64)
      ).map { case (name, mStart, nStart) =>
        InputCase(name, (source, index) => {
          val row = index / 32
          val column = index % 32
          val start = if (source == 0) mStart else nStart
          val limit = if (source == 0) 33 else 65
          if (column == 0 || start + row >= limit) 0
          else if (source == 0 && (start + row) % 2 != 0) 0x40
          else if (source == 1 && (start + row) % 3 == 0) 0x40
          else 0x38
        }, cell => {
          val row = (cell % 512) / 16
          val column = cell % 16 + (if (cell >= 512) 16 else 0)
          if (mStart + row < 33 && nStart + column < 65)
            tailBF16(mStart + row, nStart + column)
          else 0
        }, mStart, nStart)
      }
      val logicalOutput = Array.fill(33 * 65)(-1)
      val mxu1Cases = Seq(
        InputCase("mxu1_zeros", (_, _) => 0, _ => 0, mxu1 = true),
        InputCase("mxu1_unit_cell", (_, index) => if (index == 0) 0x38 else 0,
          cell => if (cell == 0) 0x3f80 else 0, mxu1 = true),
        InputCase("mxu1_ones", (_, _) => 0x38, _ => 0x4200, mxu1 = true),
        InputCase("mxu1_rows", (source, index) => {
          val row = index / 32
          if (source == 0 && row % 2 != 0) 0x40
          else if (source == 1 && row % 3 == 0) 0x40
          else 0x38
        }, cell => {
          val row = (cell % 512) / 16
          val column = cell % 16 + (if (cell >= 512) 16 else 0)
          val activationScale = if (row % 2 == 0) 1 else 2
          val weightScale = if (column % 3 == 0) 2 else 1
          (activationScale * weightScale) match {
            case 1 => 0x4200 // 32 exact products of 1
            case 2 => 0x4280 // 32 exact products of 2
            case 4 => 0x4300 // 32 exact products of 4
          }
        }, mxu1 = true))
      val k64Cases = Seq(
        InputCase("k64_first", (source, index) =>
          if (source < 2 && index == 0) 0x38 else 0,
          cell => if (cell == 0) 0x3f80 else 0, k64 = true),
        InputCase("k64_second", (source, index) =>
          if (source >= 2 && index == 0) 0x38 else 0,
          cell => if (cell == 0) 0x3f80 else 0, k64 = true),
        InputCase("k64_both", (_, index) => if (index == 0) 0x38 else 0,
          cell => if (cell == 0) 0x4000 else 0, k64 = true),
        InputCase("k64_boundary_ties", (source, index) => {
          if (source < 2 && index == 0) 0x38
          else if (source >= 2 && (index == 0 || index == 1)) 0x18
          else 0
        }, cell => if (cell == 0) 0x3f80 else 0, k64 = true),
        InputCase("k64_ones", (_, _) => 0x38, _ => 0x4280, k64 = true))
      val fixtureDirectories = sys.env.get("ATLAS_EXTRA_FIXTURES").toSeq.flatMap(
        _.split(java.io.File.pathSeparator).filter(_.nonEmpty))
      val extraCases = fixtureDirectories.map { directory =>
        val root = Paths.get(directory)
        val properties = new Properties()
        val stream = Files.newInputStream(root.resolve("manifest.properties"))
        try properties.load(stream) finally stream.close()
        val schema = properties.getProperty("schema")
        assert(schema == "atlas.rtl_tile_fixture.v1" || schema == "atlas.rtl_tile_fixture.v2" ||
          schema == "atlas.rtl_tile_fixture.v3")
        val resident = schema == "atlas.rtl_tile_fixture.v3"
        val limit = if (resident) 128 else 16
        val cycleLimit = if (resident) properties.getProperty("max_cycles").toInt else 20000
        assert(cycleLimit >= 1 && cycleLimit <= 10000000)
        val name = properties.getProperty("name")
        assert(name != null && name.matches("[a-z][a-z0-9_]*"))
        val count = properties.getProperty("input_count").toInt
        assert(count >= 1 && count <= limit)
        val addresses = (0 until count).map { index =>
          java.lang.Long.decode(properties.getProperty(s"input.$index.address")).longValue()
        }
        assert(addresses.distinct.size == count && addresses.forall(_ % 32 == 0))
        val inputs = (0 until count).map { index =>
          val data = Files.readAllBytes(root.resolve(s"input-$index.bin"))
          assert(data.length == 1024 || data.length == 2048)
          data
        }
        val ranges = addresses.zip(inputs).map { case (address, data) =>
          (address, address + data.length)
        }
        assert(ranges.indices.forall { left => ranges.indices.forall { right =>
          left == right || ranges(left)._2 <= ranges(right)._1 ||
            ranges(right)._2 <= ranges(left)._1
        }})
        val outputBase = java.lang.Long.decode(properties.getProperty("output.address")).longValue()
        assert(outputBase > 0 && outputBase % 32 == 0)
        val expected = Files.readAllBytes(root.resolve("expected.bin"))
        val outputCount = Option(properties.getProperty("output_count")).map(_.toInt).getOrElse(1)
        assert(outputCount >= 1 && outputCount <= limit && expected.length == outputCount * 2048)
        assert(ranges.forall { case (start, end) =>
          end <= outputBase || start >= outputBase + expected.length
        })
        val extraBytes = Files.readAllBytes(root.resolve("program.bin"))
        assert(extraBytes.nonEmpty && extraBytes.length % 4 == 0 &&
          extraBytes.length <= AtlasMemMap.IMEM_SIZE)
        val extraDigest = MessageDigest.getInstance("SHA-256").digest(extraBytes)
          .map(byte => f"${byte & 0xff}%02x").mkString
        assert(extraDigest == properties.getProperty("program_sha256"))
        val extraWords = ByteBuffer.wrap(extraBytes).order(ByteOrder.LITTLE_ENDIAN)
        val extraProgram = (0 until extraBytes.length / 4).map(_ =>
          extraWords.getInt() & 0xffffffffL)
        InputCase(name,
          (source, index) => inputs(source)(index) & 0xff,
          cell => (expected(2 * cell) & 0xff) | ((expected(2 * cell + 1) & 0xff) << 8),
          programOverride = extraProgram,
          inputBasesOverride = addresses,
          outputBaseOverride = outputBase,
          inputPayloadsOverride = inputs, outputCells = expected.length / 2,
          residentBuffers = resident,
          cycleLimit = cycleLimit,
          inputReadCountsOverride = (0 until count).map { index =>
            val repeats = if (schema == "atlas.rtl_tile_fixture.v2")
              properties.getProperty(s"input.$index.read_count").toInt else 1
            assert(repeats >= 1 && repeats <= 4096)
            repeats
          })
      }
      val cases = arithmeticCases ++ tailCases ++ mxu1Cases ++ k64Cases ++ extraCases
      assert(cases.map(_.name).distinct.size == cases.size)
      cases.foreach { testCase =>
        // Match the host's repeated load/start sequence without a core reset.
        val caseProgram = if (testCase.programOverride.nonEmpty) testCase.programOverride
          else if (testCase.k64) k64Program
          else if (testCase.mxu1) mxu1Program else program
        caseProgram.zipWithIndex.foreach { case (word, index) =>
          hostPut(dut, dut.io.imemTL, AtlasMemMap.IMEM_BASE + index * 4L, word)
        }

        // A fresh external DRAM image per invocation. Internal core state
        // persists between ECALL and the next host start.
        val dram = mutable.Map.empty[Long, Int].withDefaultValue(0xa5)
        val caseInputBases = if (testCase.inputBasesOverride.nonEmpty)
          testCase.inputBasesOverride else if (testCase.k64) k64InputBases else inputBases
        val caseOutputBases = if (testCase.outputBaseOverride != 0L)
          (0 until testCase.outputCells / 512).map(i => testCase.outputBaseOverride + i * 1024L)
          else outputBases
        val caseInputLengths = if (testCase.inputPayloadsOverride.nonEmpty)
          testCase.inputPayloadsOverride.map(_.length)
          else Seq.fill(caseInputBases.size)(1024)
        def caseInputByte(source: Int, index: Int): Int =
          if (testCase.inputPayloadsOverride.nonEmpty)
            testCase.inputPayloadsOverride(source)(index) & 0xff
          else testCase.inputCode(source, index)
        caseInputBases.zipWithIndex.foreach { case (base, source) =>
          (0 until caseInputLengths(source)).foreach(i =>
            dram(base + i) = caseInputByte(source, i))
        }
        // Adjacent input buffers may occupy the guard addresses. Preserve
        // their actual pre-launch bytes rather than assuming uninitialized DRAM.
        val prefixGuard = (1 to 32).map(i => dram(caseOutputBases.head - i))
        val suffixGuard = (0 until 32).map(i => dram(caseOutputBases.last + 1024 + i))
        val responses = mutable.Queue.empty[Pending]
        val transactions = mutable.ArrayBuffer.empty[(Int, Long)]
        val delayedMemory = testCase.mStart >= 0 || testCase.k64 ||
          testCase.programOverride.nonEmpty

        hostPut(dut, dut.io.csrTL, AtlasMemMap.CSR_BASE + 0x18L, 1)
        var seenRunning = false
        var completed = false
        var cycles = 0
        while (!completed && cycles < testCase.cycleLimit) {
          val requestReady = !delayedMemory || cycles % 7 < 4
          dut.io.dmaTL.a.ready.poke(requestReady.B)
          val responseAvailable = responses.nonEmpty && responses.front.readyCycle <= cycles
          if (responseAvailable) {
            val response = responses.front.response
            dut.io.dmaTL.d.valid.poke(true.B)
            dut.io.dmaTL.d.bits.opcode.poke(response.opcode.U)
            dut.io.dmaTL.d.bits.param.poke(0.U)
            dut.io.dmaTL.d.bits.size.poke(response.size.U)
            dut.io.dmaTL.d.bits.source.poke(response.source.U)
            dut.io.dmaTL.d.bits.sink.poke(0.U)
            dut.io.dmaTL.d.bits.denied.poke(false.B)
            dut.io.dmaTL.d.bits.data.poke(response.data.U)
            dut.io.dmaTL.d.bits.corrupt.poke(false.B)
          } else {
            dut.io.dmaTL.d.valid.poke(false.B)
          }
          val responseFired = responseAvailable && dut.io.dmaTL.d.ready.peek().litValue == 1
          val request = if (requestReady && dut.io.dmaTL.a.valid.peek().litValue == 1) {
            val a = dut.io.dmaTL.a.bits
            val opcode = a.opcode.peek().litValue.toInt
            val address = a.address.peek().litValue.toLong
            val size = a.size.peek().litValue
            val source = a.source.peek().litValue
            val mask = a.mask.peek().litValue
            val payload = a.data.peek().litValue
            assert(size == 5 && address % 32 == 0,
              s"unexpected DMA beat size/address at ${address.toHexString}")
            assert(opcode == 0 || opcode == 4, s"unexpected DMA opcode $opcode")
            val data = if (opcode == 4) {
              (0 until 32).foldLeft(BigInt(0)) { (acc, i) =>
                acc | (BigInt(dram(address + i)) << (8 * i))
              }
            } else {
              assert(mask == (BigInt(1) << 32) - 1)
              (0 until 32).foreach { i =>
                dram(address + i) = ((payload >> (8 * i)) & 0xff).toInt
              }
              BigInt(0)
            }
            Some((Response(if (opcode == 4) 1 else 0, size, source, data), opcode, address))
          } else None
          if (dut.io.halted.peek().litValue == 0) seenRunning = true
          dut.clock.step()
          if (responseFired) responses.dequeue()
          request.foreach { case (response, opcode, address) =>
            val delay = if (delayedMemory) 3 + transactions.size % 4 else 1
            responses.enqueue(Pending(response, cycles + delay))
            transactions += ((opcode, address))
          }
          cycles += 1
          completed = seenRunning && dut.io.halted.peek().litValue == 1 &&
            responses.isEmpty && dut.io.dbg.dmaBusy.toSeq.forall(_.peek().litValue == 0)
        }
        assert(completed, s"${testCase.name}: core did not drain and halt in $cycles cycles")
        val expectedTransactions =
          caseInputBases.zip(caseInputLengths).zipWithIndex.flatMap { case ((base, length), index) =>
            val repeats = if (testCase.inputReadCountsOverride.nonEmpty)
              testCase.inputReadCountsOverride(index) else 1
            (0 until repeats).flatMap(_ => (0 until length / 32).map(i => (4, base + 32L * i)))
          } ++
          caseOutputBases.flatMap(base => (0 until 32).map(i => (0, base + 32L * i)))
        if (testCase.residentBuffers) {
          val outputStart = caseOutputBases.head
          val outputEnd = outputStart + testCase.outputCells * 2L
          assert(transactions.forall { case (opcode, address) =>
            val output = outputStart <= address && address + 32 <= outputEnd
            val input = caseInputBases.zip(caseInputLengths).exists { case (base, length) =>
              base <= address && address + 32 <= base + length
            }
            (opcode == 0 && output) || (opcode == 4 && (input || output))
          }, s"${testCase.name}: DMA access exceeds initialized input or private output storage")
        } else assert(transactions.sorted == expectedTransactions.sorted,
          s"${testCase.name}: unexpected DMA address/operation set")
        (0 until testCase.outputCells).foreach { cell =>
          val addr = caseOutputBases.head + 2L * cell
          val actual = dram(addr) | (dram(addr + 1) << 8)
          val expected = testCase.expectedBF16(cell)
          assert(actual == expected,
            s"${testCase.name}: BF16 cell $cell = ${actual.toHexString}, expected ${expected.toHexString}")
        }
        if (testCase.mStart >= 0) {
          for (row <- 0 until 32; column <- 0 until 32
               if testCase.mStart + row < 33 && testCase.nStart + column < 65) {
            val physical = if (column < 16)
              caseOutputBases.head + row * 32L + column * 2L
            else caseOutputBases.last + row * 32L + (column - 16) * 2L
            val logical = (testCase.mStart + row) * 65 + testCase.nStart + column
            assert(logicalOutput(logical) == -1,
              s"${testCase.name}: logical cell $logical written twice")
            logicalOutput(logical) = dram(physical) | (dram(physical + 1) << 8)
          }
        }
        assert((1 to 32).forall(i => dram(caseOutputBases.head - i) == prefixGuard(i - 1)),
          s"${testCase.name}: output prefix guard changed")
        assert((0 until 32).forall(i => dram(caseOutputBases.last + 1024 + i) == suffixGuard(i)),
          s"${testCase.name}: output suffix guard changed")
        assert(caseInputBases.zipWithIndex.forall { case (base, source) =>
          (0 until caseInputLengths(source)).forall(i =>
            dram(base + i) == caseInputByte(source, i))
        },
          s"${testCase.name}: source input changed")
        // io.halted is combinational at ECALL. Allow its CSR reason register
        // one edge to capture the event before the host reads status.
        dut.io.dmaTL.d.valid.poke(false.B)
        dut.clock.step()
        val status = hostGet(dut, dut.io.csrTL, AtlasMemMap.CSR_BASE + 0x08L)
        assert((status & 7) == 5, s"${testCase.name}: expected ECALL halt, got CSR $status")
        if (testCase.programOverride.nonEmpty) {
          sys.env.get("ATLAS_RTL_OUTPUT_DIR").foreach { directory =>
            val bytes = Array.tabulate[Byte](testCase.outputCells * 2) { index =>
              dram(caseOutputBases.head + index).toByte
            }
            Files.write(Paths.get(directory, s"${testCase.name}.bin"), bytes)
          }
        }
        val outputUnit = if (testCase.residentBuffers) "output words" else "BF16 cells"
        println(s"Selected AtlasCore ${testCase.name}: ${transactions.size} DMA beats, ${testCase.outputCells} $outputUnit, $cycles cycles")
      }
      assert(logicalOutput.indices.forall { index =>
        logicalOutput(index) == tailBF16(index / 65, index % 65)
      },
        "six restarted tail tiles did not cover all 33x65 BF16 results")
      println("Selected AtlasCore tail_grid: 6 launches, 2145 BF16 logical cells")
    }.result
  }
}
