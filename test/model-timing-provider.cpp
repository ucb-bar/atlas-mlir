#include "Atlas/AtlasModelTiming.h"
#include "Atlas/AtlasModelPairs.h"
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

int testScalarOverlap(llvm::json::Object fixture);

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
  operations.push_back(block("mxu0.matmul", "mxu0", 95,
                             llvm::json::Object{{"mreg_read0", group(0, 32)}, {"acc_write", group(32, 32)},
                                                {"compute_busy", group(0, 95)}}));
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

llvm::json::Object overlapSource(llvm::StringRef pointer) {
  return llvm::json::Object{{"json_pointer", pointer.str()}, {"sha256", std::string(64, '8')},
                            {"proof_sha256", std::string(64, '9')}};
}
llvm::json::Object overlapFixture(const ModelTimingFacts &facts) {
  llvm::json::Object pair{{"first", "vpu.mov"}, {"second", "vpu.mov"},
                          {"relation", llvm::json::Object{{"mreg_banks", "disjoint"}}},
                          {"swept", llvm::json::Array{1, 73}}, {"min_gap", 1}, {"first_completion_age", 72},
                          {"negative", llvm::json::Object{{"gap", 0}, {"result", "inexpressible"}}},
                          {"source_record", overlapSource("/pairs/0")},
                          {"verilator", llvm::json::Object{{"confirmed", true}}}};
  llvm::json::Array sequences;
  for (auto [gap, length] : {std::pair<int, int>{10, 3}, {32, 3}, {40, 3}, {72, 3}, {32, 4}}) {
    llvm::json::Array matrix, operations, gaps;
    for (int i = 0; i < length; ++i) {
      operations.push_back("vpu.mov");
      if (i) gaps.push_back(gap);
      for (int j = i + 1; j < length; ++j)
        matrix.push_back(llvm::json::Object{{"first_index", i}, {"second_index", j},
                                           {"relation", llvm::json::Object{{"mreg_banks", "disjoint"}}}});
    }
    sequences.push_back(llvm::json::Object{
        {"operations", std::move(operations)},
        {"gaps", std::move(gaps)}, {"relation", llvm::json::Object{{"mreg_banks", "disjoint"}}},
        {"all_pair_relations", std::move(matrix)}, {"relation_scope", "adjacent"}, {"result", "accepted"},
        {"in_flight_count", std::min(length, (72 + gap - 1) / gap)}, {"witness_only", true},
        {"no_stable_suffix_claim", true},
        {"source_record", overlapSource("/chains/" + std::to_string(gap * 10 + length))},
        {"verilator", llvm::json::Object{{"confirmed", true}}}});
  }
  llvm::json::Array pairs{std::move(pair)};
  pairs.push_back(llvm::json::Object{
      {"first", "mxu0.push_weight"}, {"second", "mxu0.matmul"},
      {"relation", llvm::json::Object{{"mreg_banks", "disjoint"}, {"weight_slot", "other"}}},
      {"swept", llvm::json::Array{1, 40}}, {"min_gap", 1}, {"first_live_window_end", 40},
      {"negative", llvm::json::Object{{"gap", 0}, {"result", "inexpressible"}}},
      {"source_record", overlapSource("/pairs/1")}, {"verilator", llvm::json::Object{{"confirmed", true}}}});
  pairs.push_back(llvm::json::Object{
      {"first", "mxu0.matmul"}, {"second", "mxu0.pop_acc_bf16"},
      {"relation", llvm::json::Object{{"mreg_banks", "disjoint"}, {"accumulator", "same"}}},
      {"swept", llvm::json::Array{1, 95}}, {"min_gap", 64}, {"first_live_window_end", 95},
      {"negative", llvm::json::Object{{"gap", 63}, {"result", "conflict"},
                                      {"error", llvm::json::Object{{"message", "synthetic MXU busy"}}}}},
      {"source_record", overlapSource("/pairs/2")}, {"verilator", llvm::json::Object{{"confirmed", true}}}});
  auto perOp = fixture();
  return llvm::json::Object{
      {"schema", "atlas.model_pairs.v1"}, {"age_convention", "issue_edge_zero"},
      {"projection", llvm::json::Object{{"version", 1}, {"policy", "confirmed-exact-base-relations-v1"},
                                       {"master_sha256", std::string(64, 'a')},
                                       {"master_path", "synthetic-master.json"},
                                       {"validation_sha256", std::string(64, 'b')},
                                       {"validation_path", "synthetic-validation.json"},
                                       {"per_op_sha256", facts.factsSha256()},
                                       {"projector_sha256", std::string(64, 'c')},
                                       {"counts", llvm::json::Object{}}, {"excluded_reasons", llvm::json::Object{}}}},
      {"model", std::move(*perOp.get("model"))}, {"pairs", std::move(pairs)},
      {"chains", std::move(sequences)}};
}
void testOverlap(const std::shared_ptr<const ModelTimingFacts> &facts) {
  std::string bytes = encode(overlapFixture(*facts));
  auto parsed = parseModelOverlapFacts(bytes, hash(bytes), *facts);
  check(bool(parsed), "synthetic compact overlap facts load");
  if (!parsed) { llvm::errs() << llvm::toString(parsed.takeError()) << '\n'; return; }
  auto overlap = std::make_shared<ModelOverlapFacts>(std::move(*parsed));
  auto replaced = parseModelOverlapFacts(bytes, std::string(64, '0'), *facts);
  check(!replaced, "compact content replacement fails the selected hash pin");
  if (!replaced) llvm::consumeError(replaced.takeError());
  auto pairs = modelTimingProvider(facts, overlap), chains = modelTimingProvider(facts, overlap, true);
  Instr a{findOp("vmov"), 4, 0, 0}, b{findOp("vmov"), 12, 8, 0}, c{findOp("vmov"), 20, 16, 0};
  auto fa = facts->resolve(a, unknownRegs()), fb = facts->resolve(b, unknownRegs()), fc = facts->resolve(c, unknownRegs());
  auto state = pairs.createReservations();
  check(state.value->reserve(a, fa, 0).empty() && state.value->conflict(b, fb, 1).value.empty(),
        "exact confirmed pair admits two live operations");
  check(state.value->reserve(b, fb, 10).empty() && !state.value->conflict(c, fc, 20).value.empty(),
        "pairs alone cannot admit a third live operation");
  state = chains.createReservations();
  check(state.value->reserve(a, fa, 0).empty() && state.value->reserve(c, fc, 20).empty(),
        "out-of-order placement begins with an exact pair");
  check(!state.value->conflict(b, fb, 11).value.empty() && state.value->conflict(b, fb, 10).value.empty(),
        "pure insertion rechecks the complete future chain vector");
  check(state.value->reserve(b, fb, 10).empty(), "exact chain permits chronological insertion");
  Instr fourth{findOp("vmov"), 28, 24, 0};
  check(!state.value->conflict(fourth, facts->resolve(fourth, unknownRegs()), 30).value.empty(),
        "a triple witness does not create universal capacity");
  for (int gap : {40, 72}) {
    state = pairs.createReservations();
    check(state.value->reserve(a, fa, 0).empty() && state.value->reserve(b, fb, gap).empty() &&
              state.value->conflict(c, fc, 2 * gap).value.empty(),
          "a retired historical command does not require a chain when only two remain live");
  }
  state = pairs.createReservations();
  Instr weight{findOp("vmatpush.weight.mxu0"), 0, 2, 0};
  Instr compute{findOp("vmatmul.mxu0"), 0, 4, 1};
  Instr pop{findOp("vmatpop.bf16.acc.mxu0"), 8, 0, 0};
  auto fw = facts->resolve(weight, unknownRegs()), fm = facts->resolve(compute, unknownRegs());
  check(state.value->reserve(weight, fw, 0).empty() && state.value->reserve(compute, fm, 39).empty() &&
            state.value->conflict(pop, facts->resolve(pop, unknownRegs()), 103).value.empty(),
        "retired weight push plus live compute and pop needs only the exact compute/pop pair");
  Instr overwrite{findOp("vmatpush.weight.mxu0"), 1, 12, 0};
  check(pairs.dependence(compute, fm, overwrite, facts->resolve(overwrite, unknownRegs())).value.distance == 92,
        "pair facts preserve the conservative sparse weight lifetime");
  state = chains.createReservations();
  check(state.value->reserve(a, fa, 0).empty() && state.value->reserve(b, fb, 32).empty() &&
            state.value->reserve(c, fc, 64).empty() &&
            state.value->conflict(fourth, facts->resolve(fourth, unknownRegs()), 96).value.empty(),
        "four-command history with post-progress count three matches its exact witness");
  state = chains.createReservations();
  check(state.value->reserve(a, fa, 0).empty() && state.value->onWait({findOp("dma.wait.ch0")}, 1).empty() &&
            !state.value->conflict(b, fb, 10).value.empty() && state.value->conflict(b, fb, 73).value.empty(),
        "runtime wait invalidates prior waivers while preserving drained fallback");
  auto reject = [&](llvm::json::Object object, llvm::StringRef why) {
    std::string modified = encode(std::move(object));
    auto value = parseModelOverlapFacts(modified, hash(modified), *facts);
    check(!value, why);
    if (!value) llvm::consumeError(value.takeError());
  };
  auto load = [&](llvm::json::Object object) -> std::shared_ptr<ModelOverlapFacts> {
    std::string modified = encode(std::move(object));
    auto value = parseModelOverlapFacts(modified, hash(modified), *facts);
    check(bool(value), "supported pair window and bounds load");
    if (!value) { llvm::consumeError(value.takeError()); return {}; }
    return std::make_shared<ModelOverlapFacts>(std::move(*value));
  };
  for (bool nested : {false, true}) {
    auto alternative = overlapFixture(*facts);
    auto *pair = (*alternative.getArray("pairs"))[0].getAsObject();
    pair->erase("first_completion_age");
    if (nested)
      (*pair)["live_window"] = llvm::json::Object{{"first_completion_age", 72}, {"high_includes_idle_boundary", true}};
    else (*pair)["first_live_window_end"] = 73;
    check(bool(load(std::move(alternative))), "all three live-window representations normalize identically");
  }
  auto bounded = overlapFixture(*facts);
  auto *pair = (*bounded.getArray("pairs"))[0].getAsObject();
  (*pair)["min_gap"] = 5;
  (*pair)["swept"] = llvm::json::Array{1, 100};
  (*pair)["negative"] = llvm::json::Object{{"gap", 4}, {"result", "conflict"},
                                          {"error", llvm::json::Object{{"message", "synthetic engine busy"}}}};
  auto domain = load(std::move(bounded));
  if (domain) {
    std::vector<ModelOverlapCommand> commands{
        {"vpu.mov", Engine::Vpu, {0, 1, 4, 5}, {}, {}, 0, 73},
        {"vpu.mov", Engine::Vpu, {8, 9, 12, 13}, {}, {}, 4, 73}};
    check(!domain->admits(commands, false), "one-short pair gap remains blocked");
    for (int gap : {5, 73, 100}) {
      commands[1].issue = gap;
      check(domain->admits(commands, false), "inclusive pair minimum and extended high endpoint admit");
    }
    commands[1].issue = 101;
    check(!domain->admits(commands, false), "pair domain is not extrapolated beyond its high endpoint");
  }
  auto aliased = overlapFixture(*facts);
  (*aliased.getArray("pairs"))[0].getAsObject()->operator[]("relation") =
      llvm::json::Object{{"mreg_banks", "0~0,1~1"}};
  auto aliasFacts = load(std::move(aliased));
  if (aliasFacts) {
    auto aliasProvider = modelTimingProvider(facts, aliasFacts);
    state = aliasProvider.createReservations();
    Instr alias{findOp("vmov"), 12, 32, 0};
    check(state.value->reserve(a, fa, 0).empty() &&
              !state.value->conflict(alias, facts->resolve(alias, unknownRegs()), 1).value.empty(),
          "an exact confirmed physical-alias pair still cannot bypass MREG ports");
  }
  auto dependent = overlapFixture(*facts);
  (*dependent.getArray("pairs"))[0].getAsObject()->operator[]("relation") =
      llvm::json::Object{{"mreg_banks", "2=0,3=1"}};
  auto dataFacts = load(std::move(dependent));
  if (dataFacts) {
    auto dataProvider = modelTimingProvider(facts, dataFacts);
    Instr use{findOp("vmov"), 12, 4, 0};
    check(dataProvider.dependence(a, fa, use, facts->resolve(use, unknownRegs())).value.distance == 73,
          "confirmed pair admission does not weaken observed data-release dependencies");
  }
  auto missingPair = overlapFixture(*facts);
  auto *matrix = (*missingPair.getArray("chains"))[0].getAsObject()->getArray("all_pair_relations");
  (*matrix)[1].getAsObject()->operator[]("relation") = llvm::json::Object{{"mreg_banks", "0=0,1=1"}};
  auto incomplete = load(std::move(missingPair));
  if (incomplete) {
    std::vector<ModelOverlapCommand> commands{
        {"vpu.mov", Engine::Vpu, {0, 1, 4, 5}, {}, {}, 0, 73},
        {"vpu.mov", Engine::Vpu, {8, 9, 12, 13}, {}, {}, 10, 73},
        {"vpu.mov", Engine::Vpu, {0, 1, 20, 21}, {}, {}, 20, 73}};
    check(!incomplete->admits(commands, true), "chain requires a covered exact nonadjacent pair too");
  }
  auto object = overlapFixture(*facts);
  (*object.getArray("pairs"))[0].getAsObject()->getObject("relation")->operator[]("vmem_bank") = "same";
  reject(std::move(object), "unsupported relation excludes the complete selected fact");
  object = overlapFixture(*facts);
  (*object.getArray("pairs"))[0].getAsObject()->getObject("verilator")->operator[]("confirmed") = "true";
  reject(std::move(object), "string confirmation cannot admit overlap");
  object = overlapFixture(*facts);
  object.getObject("projection")->operator[]("per_op_sha256") = std::string(64, '0');
  reject(std::move(object), "overlap facts must bind exact selected operation facts");
  object = overlapFixture(*facts);
  (*object.getArray("pairs"))[0].getAsObject()->operator[]("first_live_window_end") = 73;
  reject(std::move(object), "ambiguous pair live windows reject");
  object = overlapFixture(*facts);
  (*object.getArray("chains"))[0].getAsObject()->getArray("all_pair_relations")->pop_back();
  reject(std::move(object), "nonadjacent chain relations cannot be omitted");
  object = overlapFixture(*facts);
  (*object.getArray("chains"))[0].getAsObject()->operator[]("in_flight_count") = 2;
  reject(std::move(object), "chain count must equal exact post-progress peak");
  object = overlapFixture(*facts);
  (*object.getArray("chains"))[0].getAsObject()->operator[]("in_flight_scope") = "universal capacity 3";
  reject(std::move(object), "unknown semantic scope cannot broaden an exact witness");
  object = overlapFixture(*facts);
  object.getObject("projection")->operator[]("policy") = "unconfirmed-allowed";
  reject(std::move(object), "unreviewed projection policy rejects");
  object = overlapFixture(*facts);
  (*object.getArray("chains"))[0].getAsObject()->operator[]("issue_ages") = "0,10,20";
  reject(std::move(object), "malformed optional semantic ages reject");
  object = overlapFixture(*facts);
  (*object.getArray("pairs"))[0].getAsObject()->getObject("verilator")->operator[]("scope") = "qualified protocol";
  reject(std::move(object), "unsupported qualified proof protocol rejects");
  object = overlapFixture(*facts);
  (*object.getArray("pairs"))[0].getAsObject()->operator[]("previous_swept") = "1,73";
  reject(std::move(object), "malformed retained pair sweep rejects");
}
void testStreamedWeights(const std::shared_ptr<ModelTimingFacts> &facts) {
  Instr push{findOp("vmatpush.weight.mxu0"), 0, 0, 0};
  Instr compute{findOp("vmatmul.mxu0"), 0, 2, 0};
  Footprint first = facts->resolve(push, unknownRegs()), second = facts->resolve(compute, unknownRegs());
  auto conservative = modelTimingProvider(facts);
  int original = conservative.dependence(push, first, compute, second).value.distance;
  check(first.error.empty() && second.error.empty() && original == 35,
        "unqualified weight dependency retains the whole isolated write window");
  auto object = overlapFixture(*facts);
  object["chains"] = llvm::json::Array{};
  object["pairs"] = llvm::json::Array{llvm::json::Object{
      {"first", "mxu0.push_weight"}, {"second", "mxu0.matmul"},
      {"relation", llvm::json::Object{{"mreg_banks", "disjoint"}, {"weight_slot", "same"}}},
      {"swept", llvm::json::Array{1, 40}}, {"min_gap", 1}, {"first_live_window_end", 40},
      {"negative", llvm::json::Object{{"gap", 0}, {"result", "inexpressible"}}},
      {"source_record", overlapSource("/pairs/8")}, {"verilator", llvm::json::Object{{"confirmed", true}}}}};
  std::string bytes = encode(std::move(object));
  auto loaded = parseModelOverlapFacts(bytes, hash(bytes), *facts);
  check(bool(loaded), "exact streamed-weight witness loads");
  if (!loaded) { llvm::consumeError(loaded.takeError()); return; }
  auto overlap = std::make_shared<ModelOverlapFacts>(std::move(*loaded));
  auto qualified = modelTimingProvider(facts, overlap);
  check(qualified.dependence(push, first, compute, second).value.distance == 1,
        "exact confirmed weight push-to-matmul witness replaces the synthetic RAW bound");
  auto reservations = qualified.createReservations();
  check(reservations.value->reserve(push, first, 0).empty() &&
            reservations.value->conflict(compute, second, 1).value.empty(),
        "qualified dependency agrees with exact pair reservation admission");
  compute.rs1 = 0;
  check(qualified.dependence(push, first, compute, facts->resolve(compute, unknownRegs())).value.distance == original,
        "uncovered source alias keeps the conservative weight dependency");
  compute.rs1 = 2;
  check(qualified.dependence(compute, second, push, first).value.distance ==
            conservative.dependence(compute, second, push, first).value.distance,
        "streamed push-to-compute evidence does not weaken compute-to-push weight lifetime");
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
  testOverlap(facts);
  testStreamedWeights(facts);
  failures += testScalarOverlap(fixture());
  auto provider = modelTimingProvider(facts);
  RegValues dmaRegs = unknownRegs();
  dmaRegs[1] = 0x20000 / 4; dmaRegs[2] = 0; dmaRegs[3] = 1024;
  Footprint dma = provider.footprint({findOp("dma.load.ch0"), 1, 2, 3}, dmaRegs);
  check(dma.error.empty() && dma.dmaCycles == 516 && dma.doneAge == 0,
        "DMA duration guides scheduling while completion still requires a wait");
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
