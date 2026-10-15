package atlas.sota

import atlas.common.AtlasParams
import atlas.tile.AtlasCore
import freechips.rocketchip.tilelink.TLBundleParameters

/** Current real AtlasCore and genuine TileLink types; no SoC wrapper or stubs. */
object CoreConfiguration {
  def bus(address: Int, data: Int, source: Int, size: Int) = new TLBundleParameters(
    addressBits = address, dataBits = data, sourceBits = source, sinkBits = 1, sizeBits = size,
    echoFields = Nil, requestFields = Nil, responseFields = Nil, hasBCE = false)

  def core() = new AtlasCore(
    tp = AtlasParams(),
    imemBP = bus(32, 32, 1, 3),
    csrBP = bus(32, 32, 1, 3),
    dmaBP = bus(64, 256, 6, 4),
    vmemBP = bus(32, 256, 1, 4))
}

object EmitOwnedAtlasCore {
  def main(arguments: Array[String]): Unit = {
    require(arguments.isEmpty, "Output is supplied only through ATLAS_CORE_RTL_DIR")
    circt.stage.ChiselStage.emitSystemVerilogFile(
      CoreConfiguration.core(),
      args = Array("--target-dir", sys.env("ATLAS_CORE_RTL_DIR")),
      firtoolOpts = Array("-disable-all-randomization", "-strip-debug-info"))
  }
}
