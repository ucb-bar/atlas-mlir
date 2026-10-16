#include "Atlas/AtlasModelTiming.h"
#include "Atlas/AtlasDialect.h"
#include "Atlas/AtlasStream.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>

using namespace mlir;
using namespace mlir::atlas;
using namespace mlir::atlas::timing;

namespace {
int failures = 0;
void check(bool passed, llvm::StringRef what) {
  if (passed) return;
  ++failures;
  llvm::errs() << "FAIL: " << what << '\n';
}
std::string hash(llvm::StringRef bytes) {
  auto digest = llvm::SHA256::hash(llvm::ArrayRef<uint8_t>(
      reinterpret_cast<const uint8_t *>(bytes.data()), bytes.size()));
  return llvm::toHex(llvm::ArrayRef<uint8_t>(digest), true);
}

llvm::json::Object group(int first, int count, int step = 1) {
  return llvm::json::Object{{"first_age", first}, {"last_age", first + (count - 1) * step}, {"count", count}, {"step", step}};
}
llvm::json::Object block(const char *name, const char *engine, int free, llvm::json::Object events) {
  return llvm::json::Object{{"name", name}, {"engine", engine}, {"resolved", true}, {"first_free_age", free},
          {"next_issue_age", free}, {"events", std::move(events)},
          {"assumptions", llvm::json::Object{{"scratchpad_read_latency", 1}}},
          {"observation", llvm::json::Object{{"accepted", true}, {"drained", true}}}};
}
llvm::json::Object fixture() {
  static const llvm::StringRef configuration =
      "{\"arch_state_config\":{\"dram_base\":2147483648,"
      "\"dram_size\":68719476736,"
      "\"init_seed\":42,"
      "\"mrf_depth\":32,"
      "\"mrf_width\":32,"
      "\"num_csrs\":4096,"
      "\"num_e_registers\":32,"
      "\"num_m_registers\":64,"
      "\"num_wb_registers\":2,"
      "\"num_x_registers\":32,"
      "\"numerics\":\"rtl\","
      "\"randomize_init\":false,"
      "\"vmem_size\":1572864,"
      "\"wb_width\":1024},"
      "\"dma_beat_bytes\":32,"
      "\"dma_max_in_flight\":64,"
      "\"dma_max_transfer_bytes\":4096,"
      "\"dma_memory_backend\":\"curve\","
      "\"dma_memory_cycles_per_beat\":null,"
      "\"dma_memory_latency_cycles\":null,"
      "\"dma_memory_params\":{\"curves\":\"ee290sim_vcs_probe.json\"},"
      "\"dma_num_channels\":8,"
      "\"dma_offchip_command_words\":2,"
      "\"execution_units\":{\"DMA0\":\"DmaExecutionUnit\","
      "\"LSU\":\"LoadStoreUnit\","
      "\"Matrix0\":\"MatrixExecutionUnitSystolic\","
      "\"Matrix1\":\"MatrixExecutionUnitInner\","
      "\"Scalar0\":\"ScalarExecutionUnit\","
      "\"Vector0\":\"VectorExecutionUnit\","
      "\"XLU\":\"CrossLaneExecutionUnit\"},"
      "\"fetch_width\":1,"
      "\"ignore_runtime_errors\":false,"
      "\"isa\":\"npu_model.isa.IsaSpec\","
      "\"mxu0_matmul_latency_cycles\":32,"
      "\"mxu1_matmul_latency_cycles\":32,"
      "\"name\":\"SimpleNPU\","
      "\"offchip_link_core_cycles_per_beat\":2,"
      "\"offchip_link_width_bits\":32,"
      "\"vmem_bank_bytes\":262144,"
      "\"vmem_bus_core_cycles_per_beat\":1,"
      "\"vmem_bus_width_bits\":512,"
      "\"vmem_bytes_per_cycle\":64,"
      "\"vpu_non_pipelineable_op_latency_cycles\":16,"
      "\"vpu_simple_op_latency_cycles\":4,"
      "\"xlu_transform_latency_cycles\":4}";
  auto decodedConfiguration = llvm::cantFail(llvm::json::parse(configuration));
  llvm::json::Array operations;
  operations.push_back(block("vpu.mov", "vpu", 73, llvm::json::Object{{"read0", group(1, 64)}, {"write0", group(9, 64)}}));
  operations.push_back(block("vpu.add", "vpu", 73,
                             llvm::json::Object{{"read0", group(1, 64)}, {"read1", group(1, 64)},
                                                {"write0", group(9, 64)}}));
  operations.push_back(block("vpu.fp8pack", "vpu", 66,
                             llvm::json::Object{{"read0", group(0, 64)}, {"write0", group(3, 32, 2)}}));
  operations.push_back(block("vlsu.vload", "vlsu", 40,
                             llvm::json::Object{{"vmem_read", group(1, 32)}, {"mreg_write", group(3, 32)}}));
  operations.push_back(block("mxu0.push_weight", "mxu0", 40,
                             llvm::json::Object{{"mreg_read1", group(1, 32)}, {"weight_write", group(3, 32)},
                              {"data_busy", group(0, 36)}}));
  operations.push_back(block("mxu0.pop_acc_bf16", "mxu0", 40,
                             llvm::json::Object{{"acc_store", group(1, 32)}, {"mreg_write0", group(3, 32)},
                              {"mreg_write1", group(3, 32)}, {"data_busy", group(0, 36)}}));
  return llvm::json::Object{{"schema", "atlas.model_timing.v1"}, {"age_convention", "issue_edge_zero"},
          {"model", llvm::json::Object{{"revision", std::string(40, '1')}, {"source_sha256", std::string(64, '2')},
                                       {"configuration_sha256", hash(configuration)},
                                       {"configuration", std::move(decodedConfiguration)},
                                       {"configuration_json", configuration},
                                       {"observer_sha256", std::string(64, '4')},
                                       {"exporter_sha256", std::string(64, '5')},
                                       {"input_sha256", std::string(64, '6')},
                                       {"numerical_data_sha256", std::string(64, '7')}}},
          {"geometry", llvm::json::Object{{"rows", 32}, {"columns", 32}, {"mreg_count", 64}}},
          {"observation", llvm::json::Object{{"accepted", true}, {"drained", true}}},
          {"op_timing", std::move(operations)}};
}
std::string encode(llvm::json::Object object) {
  std::string bytes;
  llvm::raw_string_ostream stream(bytes);
  stream << llvm::json::Value(std::move(object));
  return bytes;
}
void rejected(llvm::json::Object object, llvm::StringRef why) {
  std::string bytes = encode(std::move(object));
  auto parsed = parseModelTimingFacts(bytes, hash(bytes));
  check(!parsed, why);
  if (!parsed) llvm::consumeError(parsed.takeError());
}
} // namespace

int main() {
  std::string bytes = encode(fixture());
  auto loaded = parseModelTimingFacts(bytes, hash(bytes), true);
  check(bool(loaded), "synthetic observed facts load");
  if (!loaded) {
    llvm::errs() << llvm::toString(loaded.takeError()) << '\n';
    return 1;
  }
  auto facts = std::make_shared<ModelTimingFacts>(std::move(*loaded));
  auto provider = modelTimingProvider(facts);
  Instr move{findOp("vmov"), 4, 0, 0};
  Footprint first = provider.footprint(move, unknownRegs());
  check(first.error.empty() && first.doneAge == 72, "latency comes from synthetic observations");
  check(!provider.footprint({findOp("vrelu.bf16"), 4, 0, 0}, unknownRegs()).error.empty(),
        "missing operation coverage has no builtin fallback");
  check(!provider.footprint({findOp("vmov"), 3, 0, 0}, unknownRegs()).error.empty(), "odd BF16 pair rejects");
  check(provider.footprint({findOp("vmov"), 32, 0, 0}, unknownRegs()).error.empty(),
        "read and write may use different logical registers in the same physical bank");
  check(!provider.footprint({findOp("vadd.bf16"), 4, 0, 32}, unknownRegs()).error.empty(),
        "two read operands cannot share a physical bank");
  Instr pack{findOp("vpack.bf16.fp8"), 1, 0, 32};
  Footprint packed = provider.footprint(pack, unknownRegs());
  auto packReservations = provider.createReservations();
  auto packConflict = packReservations.value->conflict(pack, packed, 0);
  check(packed.error.empty() && packed.doneAge == 65 && packConflict.error.empty() && packConflict.value.empty(),
        "PACK m1 from m32/m33 uses separate read/write rows in bank 1");
  Instr collidingPack{findOp("vpack.bf16.fp8"), 33, 0, 32};
  Footprint colliding = provider.footprint(collidingPack, unknownRegs());
  check(colliding.error.empty() && !packReservations.value->conflict(collidingPack, colliding, 0).error.empty(),
        "PACK same logical source/destination row collision still rejects");
  Footprint duplicateWrite = packed;
  duplicateWrite.accesses.push_back({Res::MReg, true, 33 * 32, 1, 3, 0});
  check(!packReservations.value->conflict(pack, duplicateWrite, 0).error.empty(),
        "two writes cannot use the same physical bank on one edge");
  check(!provider.footprint({findOp("fence")}, unknownRegs()).error.empty(), "unmapped operation rejects");
  Footprint unknown = provider.footprint({findOp("vload"), 6, 2, 0}, unknownRegs());
  check(unknown.error.empty() && unknown.doneAge == 39 && unknown.holds.size() == size_t(kVmemBanks + 1),
        "unknown VLS base reserves every bank with observed ages");
  check(std::any_of(unknown.accesses.begin(), unknown.accesses.end(), [](const Access &a) {
          return a.res == Res::Vmem && a.anywhere;
        }), "unknown VLS address conservatively aliases all VMEM");
  auto reservations = provider.createReservations();
  check(reservations.error.empty() && reservations.value &&
            reservations.value->reserve(move, first, 0).empty(), "observed reservations accept an isolated operation");
  Instr next{findOp("vmov"), 12, 8, 0};
  Footprint second = provider.footprint(next, unknownRegs());
  check(!reservations.value->conflict(next, second, 72).value.empty() &&
            reservations.value->conflict(next, second, 73).value.empty(), "isolated engine window has exact boundary");
  auto dependency = provider.dependence(move, first, {findOp("vmov"), 8, 4, 0},
                                        provider.footprint({findOp("vmov"), 8, 4, 0}, unknownRegs()));
  check(dependency.error.empty() && dependency.value.distance == 73, "observed MREG release drives dependence");
  DialectRegistry registry;
  registry.insert<AtlasDialect>();
  MLIRContext context(registry);
  auto module = parseSourceString<ModuleOp>("module {}", &context);
  auto cost = machineTimingOf(provider, "vload", *module);
  check(succeeded(cost) && cost->latency == 40 && cost->interval == 40,
        "virtual representative unknown-base query uses selected facts");
  for (const char *name : {"vmatpush.weight.mxu0", "vmatpop.bf16.acc.mxu0"}) {
    cost = machineTimingOf(provider, name, *module);
    check(succeeded(cost) && cost->latency == 40 && cost->interval == 40,
          "model MXU cost probes use legal operands");
  }
  check(!provider.footprint({findOp("vmatpush.weight.mxu0"), 1, 2, 1}, unknownRegs()).error.empty(),
        "legal cost probes do not loosen real MXU encoding checks");
  auto failing = provider;
  failing.costProbes = [](const OpInfo &) {
    return TimingRuleResult<std::pair<Instr, Instr>>{{}, "missing synthetic cost probe"};
  };
  check(failed(machineTimingOf(failing, "vmov", *module)), "missing cost probe coverage fails");
  failing.costProbes = [](const OpInfo &) {
    Instr wrong{findOp("addi")};
    return TimingRuleResult<std::pair<Instr, Instr>>{{wrong, wrong}, {}};
  };
  check(failed(machineTimingOf(failing, "vmov", *module)), "cost probes cannot substitute another operation");
  check(!lookupTimingProvider("npu-model", *module).error.empty(), "registry rejects absent model selection");
  check(lookupTimingProvider("builtin").error.empty(), "model registration preserves builtin");
  auto badHash = parseModelTimingFacts(bytes, std::string(64, '0'));
  check(!badHash, "facts content replacement fails the pin");
  if (!badHash) llvm::consumeError(badHash.takeError());
  for (const char *field : {"exporter_sha256", "input_sha256", "numerical_data_sha256"}) {
    auto incomplete = fixture();
    incomplete.getObject("model")->erase(field);
    rejected(std::move(incomplete), "incomplete provenance rejects");
  }
  auto object = fixture();
  object["schema"] = "merlin.op_timing.v1";
  rejected(std::move(object), "RTL schema rejects");
  object = fixture();
  (*object.getObject("model"))["configuration_sha256"] = std::string(64, '0');
  rejected(std::move(object), "configuration digest cannot be forged");
  object = fixture();
  (*object.getObject("model")->getObject("configuration"))["ignore_runtime_errors"] = true;
  rejected(std::move(object), "runtime error suppression is outside the configuration profile");
  object = fixture();
  object["hw_ir"] = llvm::json::Object{};
  rejected(std::move(object), "RTL provenance rejects");
  object = fixture();
  object["pairs"] = llvm::json::Array{};
  rejected(std::move(object), "pair admission rejects");
  object = fixture();
  object.getObject("geometry")->operator[]("rows") = 16;
  rejected(std::move(object), "unsupported geometry rejects");
  object = fixture();
  (*object.getArray("op_timing"))[0].getAsObject()->operator[]("next_issue_age") = 1;
  rejected(std::move(object), "unconfirmed overlap rejects");
  object = fixture();
  (*object.getArray("op_timing"))[0].getAsObject()->getObject("events")->getObject("read0")->operator[]("count") = 1;
  rejected(std::move(object), "inconsistent event shape rejects");
  object = fixture();
  (*object.getArray("op_timing"))[0].getAsObject()->operator[]("name") = "unknown.operation";
  rejected(std::move(object), "unknown fact operation rejects");
  llvm::outs() << "model timing synthetic checks, " << failures << " failures\n";
  return failures != 0;
}
