#include "Atlas/AtlasModelTiming.h"
#include "Atlas/AtlasModelVocabulary.h"
#include "Atlas/AtlasModelPairs.h"
#include "Atlas/AtlasDialect.h"
#include "Atlas/AtlasStream.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <tuple>

using namespace mlir;
using namespace mlir::atlas;
using namespace mlir::atlas::timing;
using Object = llvm::json::Object;
using Array = llvm::json::Array;

int testScalarOverlap(Object fixture);

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

Object group(int first, int count, int step = 1) {
  return Object{{"first_age", first}, {"last_age", first + (count - 1) * step}, {"count", count}, {"step", step}};
}
Object block(const char *name, const char *engine, int free, Object events) {
  return Object{{"name", name}, {"engine", engine}, {"resolved", true}, {"first_free_age", free},
          {"next_issue_age", free}, {"events", std::move(events)},
          {"assumptions", Object{{"scratchpad_read_latency", 1}}},
          {"observation", Object{{"accepted", true}, {"drained", true}}}};
}
Object dmaFixture() {
  Object sources, shapes;
  for (const auto &[key, value] : model_schema::dmaIdentity()) {
    if (key == "curve_sha256") continue;
    if (llvm::StringRef(key).starts_with("shape:")) shapes[llvm::StringRef(key).drop_front(6)] = value;
    else sources[key] = value;
  }
  // Synthetic points intentionally differ from the pinned production values.
  Array curves;
  for (auto [pct, latency, peak] : {std::tuple<int, double, double>{0, 3., .5}, {100, 5., .4}})
    curves.push_back(Object{{"read_percentage", pct}, {"unloaded_latency_cycles", latency},
        {"peak_beats_per_cycle", peak}, {"cycles_per_beat", 1. / peak},
        {"points", Array{Array{peak / 2, latency}, Array{peak, latency * 2}}}});
  return Object{{"schema", "atlas.dma_timing.v1"}, {"backend", "curve"}, {"resolved", true}, {"estimate_only", true},
      {"curve_sha256", model_schema::dmaIdentity().at("curve_sha256")},
      {"source_sha256", std::move(sources)}, {"source_shapes", std::move(shapes)},
      {"beat_bytes", 32}, {"max_in_flight", 64}, {"num_channels", 8}, {"max_transfer_bytes", 4096},
      {"queue_depth", 64}, {"window_beats", 32}, {"frequency_ghz", .5}, {"channels", 1},
      {"measured_channels", 1}, {"access_bytes", 32}, {"latency_mode", "unloaded"}, {"converge", 1.},
      {"knee", .9}, {"enforce_peak", true}, {"min_latency", 1},
      {"startup_cycles", Object{{"load", 1}, {"store", 3}}}, {"issue_interval_cycles", 1},
      {"rounding_epsilon_cycles", 1e-9}, {"reset_last_ready_cycles", 0.}, {"curves", std::move(curves)},
      {"semantics", Object{{"history", "last_window_beats_issued_including_current_beat"},
          {"curve_selection", "nearest_read_percentage_tie_lower"},
          {"acceptance", "pending_count_below_queue_depth_including_presented_response"},
          {"ready", "max_issue_plus_unloaded_latency_and_previous_fractional_ready_plus_curve_spacing"},
          {"response", "oldest_ready_then_issue_order_held_until_pop"},
          {"history_lifetime", "reset_only_not_command_boundary"}, {"address_dependence", "none"},
          {"engine", "registered_slots_and_in_flight;D_before_A;retire_after_last_ack"}}},
      {"limits", Array{"Cold isolated proposal; completion waits remain required."}},
      {"validation", Object{{"passed", true}, {"report_sha256", std::string(64, 'a')},
                            {"backend_streams", 2}, {"isolated_dma_runs", 2}, {"mixed_dma_runs", 2}}}};
}
Object fixture() {
  static const llvm::StringRef configuration =
      "{\"arch_state_config\":{\"dram_base\":2147483648,\"dram_size\":68719476736,\"init_seed\":42,"
      "\"mrf_depth\":32,\"mrf_width\":32,\"num_csrs\":4096,\"num_e_registers\":32,\"num_m_registers\":64,"
      "\"num_wb_registers\":2,\"num_x_registers\":32,\"numerics\":\"rtl\",\"randomize_init\":false,"
      "\"vmem_size\":1572864,\"wb_width\":1024},\"dma_beat_bytes\":32,\"dma_max_in_flight\":64,"
      "\"dma_max_transfer_bytes\":4096,\"dma_memory_backend\":\"curve\",\"dma_memory_cycles_per_beat\":null,"
      "\"dma_memory_latency_cycles\":null,\"dma_memory_params\":{\"curves\":\"ee290sim_vcs_probe.json\"},"
      "\"dma_num_channels\":8,\"dma_offchip_command_words\":2,\"execution_units\":{\"DMA0\":\"DmaExecutionUnit\","
      "\"LSU\":\"LoadStoreUnit\",\"Matrix0\":\"MatrixExecutionUnitSystolic\","
      "\"Matrix1\":\"MatrixExecutionUnitInner\",\"Scalar0\":\"ScalarExecutionUnit\","
      "\"Vector0\":\"VectorExecutionUnit\",\"XLU\":\"CrossLaneExecutionUnit\"},\"fetch_width\":1,"
      "\"ignore_runtime_errors\":false,\"isa\":\"npu_model.isa.IsaSpec\",\"mxu0_matmul_latency_cycles\":32,"
      "\"mxu1_matmul_latency_cycles\":32,\"name\":\"SimpleNPU\",\"offchip_link_core_cycles_per_beat\":2,"
      "\"offchip_link_width_bits\":32,\"vmem_bank_bytes\":262144,\"vmem_bus_core_cycles_per_beat\":1,"
      "\"vmem_bus_width_bits\":512,\"vmem_bytes_per_cycle\":64,\"vpu_non_pipelineable_op_latency_cycles\":16,"
      "\"vpu_simple_op_latency_cycles\":4,\"xlu_transform_latency_cycles\":4}";
  auto decodedConfiguration = llvm::cantFail(llvm::json::parse(configuration));
  Array operations;
  operations.push_back(block("vpu.mov", "vpu", 73, Object{{"read0", group(1, 64)}, {"write0", group(9, 64)}}));
  operations.push_back(block("vpu.add", "vpu", 73,
                             Object{{"read0", group(1, 64)}, {"read1", group(1, 64)},
                                                {"write0", group(9, 64)}}));
  operations.push_back(block("vpu.fp8pack", "vpu", 66,
                             Object{{"read0", group(0, 64)}, {"write0", group(3, 32, 2)}}));
  operations.push_back(block("vlsu.vload", "vlsu", 40,
                             Object{{"vmem_read", group(1, 32)}, {"mreg_write", group(3, 32)}}));
  operations.push_back(block("vlsu.vstore", "vlsu", 40,
                             Object{{"mreg_read", group(1, 32)}, {"vmem_write", group(5, 32)}}));
  operations.push_back(block("mxu0.push_weight", "mxu0", 40,
                             Object{{"mreg_read1", group(1, 32)}, {"weight_write", group(3, 32)},
                              {"data_busy", group(0, 36)}}));
  operations.push_back(block("mxu0.pop_acc_bf16", "mxu0", 40,
                             Object{{"acc_store", group(1, 32)}, {"mreg_write0", group(3, 32)},
                              {"mreg_write1", group(3, 32)}, {"data_busy", group(0, 36)}}));
  operations.push_back(block("mxu0.matmul", "mxu0", 95,
                             Object{{"mreg_read0", group(0, 32)}, {"acc_write", group(32, 32)},
                                                {"compute_busy", group(0, 95)}}));
  return Object{{"schema", "atlas.model_timing.v1"}, {"age_convention", "issue_edge_zero"},
          {"model", Object{{"revision", std::string(40, '1')}, {"source_sha256", std::string(64, '2')},
                                       {"configuration_sha256", hash(configuration)},
                                       {"configuration", std::move(decodedConfiguration)},
                                       {"configuration_json", configuration},
                                       {"observer_sha256", std::string(64, '4')},
                                       {"exporter_sha256", std::string(64, '5')},
                                       {"input_sha256", std::string(64, '6')},
                                       {"numerical_data_sha256", std::string(64, '7')}}},
          {"geometry", Object{{"rows", 32}, {"columns", 32}, {"mreg_count", 64}}},
          {"observation", Object{{"accepted", true}, {"drained", true}}},
          {"op_timing", std::move(operations)}, {"dma_timing", dmaFixture()}};
}
std::string encode(Object object) {
  std::string bytes;
  llvm::raw_string_ostream stream(bytes);
  stream << llvm::json::Value(std::move(object));
  return bytes;
}
void rejected(Object object, llvm::StringRef why) {
  std::string bytes = encode(std::move(object));
  auto parsed = parseModelTimingFacts(bytes, hash(bytes));
  check(!parsed, why);
  if (!parsed) {
    std::string error = llvm::toString(parsed.takeError());
    check(llvm::StringRef(error).ends_with("See docs/model-timing.md#updating-for-npu-model-changes."),
          "facts refusal ends with model update guidance");
    check(modelTimingDiagnostic(error) == error, "nested model refusals retain one guidance line");
  }
}

struct InvalidFact {
  const char *why;
  std::string path;
  llvm::json::Value value = nullptr;
  bool remove = false;
};

// Paths describe fixture mutations only; every row gets a fresh copy and pin.
template <typename Reject>
void invalidFacts(Object original, std::initializer_list<InvalidFact> cases, Reject reject) {
  for (const auto &test : cases) {
    Object object(original);
    llvm::StringRef path(test.path);
    Object *parent = &object;
    while (path.contains('/')) {
      auto [key, rest] = path.split('/');
      auto *value = parent->get(key);
      if (auto *array = value->getAsArray()) {
        auto [index, tail] = rest.split('/');
        unsigned position = 0;
        bool malformed = index.getAsInteger(10, position);
        assert(!malformed && position < array->size());
        parent = (*array)[position].getAsObject();
        path = tail;
      } else {
        parent = value->getAsObject();
        path = rest;
      }
      assert(parent && !path.empty());
    }
    if (test.remove) parent->erase(path);
    else (*parent)[path] = test.value;
    reject(std::move(object), test.why);
  }
}
void rejectedFacts(Object original, std::initializer_list<InvalidFact> cases) {
  invalidFacts(std::move(original), cases, [](Object object, const char *why) {
    rejected(std::move(object), why);
  });
}

Object overlapSource(llvm::StringRef pointer) {
  return Object{{"json_pointer", pointer.str()}, {"sha256", std::string(64, '8')},
                            {"proof_sha256", std::string(64, '9')}};
}
Object overlapFixture(const ModelTimingFacts &facts) {
  Object pair{{"first", "vpu.mov"}, {"second", "vpu.mov"},
                          {"relation", Object{{"mreg_banks", "disjoint"}}},
                          {"swept", Array{1, 73}}, {"min_gap", 1}, {"first_completion_age", 72},
                          {"negative", Object{{"gap", 0}, {"result", "inexpressible"}}},
                          {"source_record", overlapSource("/pairs/0")},
                          {"verilator", Object{{"confirmed", true}}}};
  Array sequences;
  for (auto [gap, length] : {std::pair<int, int>{10, 3}, {32, 3}, {40, 3}, {72, 3}, {32, 4}}) {
    Array matrix, operations, gaps;
    for (int i = 0; i < length; ++i) {
      operations.push_back("vpu.mov");
      if (i) gaps.push_back(gap);
      for (int j = i + 1; j < length; ++j)
        matrix.push_back(Object{{"first_index", i}, {"second_index", j},
                                           {"relation", Object{{"mreg_banks", "disjoint"}}}});
    }
    sequences.push_back(Object{
        {"operations", std::move(operations)},
        {"gaps", std::move(gaps)}, {"relation", Object{{"mreg_banks", "disjoint"}}},
        {"all_pair_relations", std::move(matrix)}, {"relation_scope", "adjacent"}, {"result", "accepted"},
        {"in_flight_count", std::min(length, (72 + gap - 1) / gap)}, {"witness_only", true},
        {"no_stable_suffix_claim", true},
        {"source_record", overlapSource("/chains/" + std::to_string(gap * 10 + length))},
        {"verilator", Object{{"confirmed", true}}}});
  }
  Array pairs{std::move(pair)};
  pairs.push_back(Object{
      {"first", "mxu0.push_weight"}, {"second", "mxu0.matmul"},
      {"relation", Object{{"mreg_banks", "disjoint"}, {"weight_slot", "other"}}},
      {"swept", Array{1, 40}}, {"min_gap", 1}, {"first_live_window_end", 40},
      {"negative", Object{{"gap", 0}, {"result", "inexpressible"}}},
      {"source_record", overlapSource("/pairs/1")}, {"verilator", Object{{"confirmed", true}}}});
  pairs.push_back(Object{
      {"first", "mxu0.matmul"}, {"second", "mxu0.pop_acc_bf16"},
      {"relation", Object{{"mreg_banks", "disjoint"}, {"accumulator", "same"}}},
      {"swept", Array{1, 95}}, {"min_gap", 64}, {"first_live_window_end", 95},
      {"negative", Object{{"gap", 63}, {"result", "conflict"},
                                      {"error", Object{{"message", "synthetic MXU busy"}}}}},
      {"source_record", overlapSource("/pairs/2")}, {"verilator", Object{{"confirmed", true}}}});
  auto perOp = fixture();
  return Object{
      {"schema", "atlas.model_pairs.v1"}, {"age_convention", "issue_edge_zero"},
      {"projection", Object{{"version", 1}, {"policy", "confirmed-exact-base-relations-v1"},
                                       {"master_sha256", std::string(64, 'a')},
                                       {"master_path", "synthetic-master.json"},
                                       {"validation_sha256", std::string(64, 'b')},
                                       {"validation_path", "synthetic-validation.json"},
                                       {"per_op_sha256", facts.factsSha256()},
                                       {"projector_sha256", std::string(64, 'c')},
                                       {"counts", Object{}}, {"excluded_reasons", Object{}}}},
      {"model", std::move(*perOp.get("model"))}, {"pairs", std::move(pairs)},
      {"chains", std::move(sequences)}};
}
void testOverlap(const std::shared_ptr<const ModelTimingFacts> &facts) {
  std::string bytes = encode(overlapFixture(*facts));
  auto parsed = parseModelOverlapFacts(bytes, hash(bytes), *facts);
  check(bool(parsed), "synthetic compact overlap facts load");
  if (!parsed) { llvm::errs() << llvm::toString(parsed.takeError()) << '\n'; return; }
  llvm::SmallString<128> cachePath;
  int descriptor;
  auto created = llvm::sys::fs::createTemporaryFile("atlas-overlap-cache", "json", descriptor, cachePath);
  check(!created, "create overlap cache fixture");
  if (!created) {
    {
      llvm::raw_fd_ostream output(descriptor, true);
      output << bytes;
    }
    for (int attempt = 0; attempt < 2; ++attempt) {
      auto loaded = loadModelOverlapFacts(cachePath, hash(bytes), *facts);
      check(bool(loaded) && loaded->metadata() == parsed->metadata(), "repeated overlap load preserves proof metadata");
      if (!loaded) llvm::consumeError(loaded.takeError());
    }
    std::error_code error;
    {
      llvm::raw_fd_ostream output(cachePath, error);
      check(!error, "replace overlap cache fixture");
      output << bytes << ' ';
    }
    auto corrupted = loadModelOverlapFacts(cachePath, hash(bytes), *facts);
    check(!corrupted, "cached overlap facts still reject changed file contents");
    if (!corrupted) llvm::consumeError(corrupted.takeError());
    auto refreshed = loadModelOverlapFacts(cachePath, hash(bytes + " "), *facts);
    check(bool(refreshed), "new content pin reloads overlap proof");
    if (!refreshed) llvm::consumeError(refreshed.takeError());
    llvm::sys::fs::remove(cachePath);
  }
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
  auto reject = [&](Object object, llvm::StringRef why) {
    std::string modified = encode(std::move(object));
    auto value = parseModelOverlapFacts(modified, hash(modified), *facts);
    check(!value, why);
    if (!value) llvm::consumeError(value.takeError());
  };
  auto load = [&](Object object) -> std::shared_ptr<ModelOverlapFacts> {
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
      (*pair)["live_window"] = Object{{"first_completion_age", 72}, {"high_includes_idle_boundary", true}};
    else (*pair)["first_live_window_end"] = 73;
    check(bool(load(std::move(alternative))), "all three live-window representations normalize identically");
  }
  auto bounded = overlapFixture(*facts);
  auto *pair = (*bounded.getArray("pairs"))[0].getAsObject();
  (*pair)["min_gap"] = 5;
  (*pair)["swept"] = Array{1, 100};
  (*pair)["negative"] = Object{{"gap", 4}, {"result", "conflict"},
                                          {"error", Object{{"message", "synthetic engine busy"}}}};
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
  (*aliased.getArray("pairs"))[0].getAsObject()->operator[]("relation") = Object{{"mreg_banks", "0~0,1~1"}};
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
      Object{{"mreg_banks", "2=0,3=1"}};
  auto dataFacts = load(std::move(dependent));
  if (dataFacts) {
    auto dataProvider = modelTimingProvider(facts, dataFacts);
    Instr use{findOp("vmov"), 12, 4, 0};
    check(dataProvider.dependence(a, fa, use, facts->resolve(use, unknownRegs())).value.distance == 73,
          "confirmed pair admission does not weaken observed data-release dependencies");
  }
  auto missingPair = overlapFixture(*facts);
  auto *matrix = (*missingPair.getArray("chains"))[0].getAsObject()->getArray("all_pair_relations");
  (*matrix)[1].getAsObject()->operator[]("relation") = Object{{"mreg_banks", "0=0,1=1"}};
  auto incomplete = load(std::move(missingPair));
  if (incomplete) {
    std::vector<ModelOverlapCommand> commands{
        {"vpu.mov", Engine::Vpu, {0, 1, 4, 5}, {}, {}, 0, 73},
        {"vpu.mov", Engine::Vpu, {8, 9, 12, 13}, {}, {}, 10, 73},
        {"vpu.mov", Engine::Vpu, {0, 1, 20, 21}, {}, {}, 20, 73}};
    check(!incomplete->admits(commands, true), "chain requires a covered exact nonadjacent pair too");
  }
  invalidFacts(overlapFixture(*facts), {
      {"unsupported relation excludes the complete selected fact", "pairs/0/relation/vmem_bank", "same"},
      {"string confirmation cannot admit overlap", "pairs/0/verilator/confirmed", "true"},
      {"overlap facts must bind exact selected operation facts", "projection/per_op_sha256", std::string(64, '0')},
      {"ambiguous pair live windows reject", "pairs/0/first_live_window_end", 73},
      {"chain count must equal exact post-progress peak", "chains/0/in_flight_count", 2},
      {"unknown semantic scope cannot broaden an exact witness", "chains/0/in_flight_scope", "universal capacity 3"},
      {"unreviewed projection policy rejects", "projection/policy", "unconfirmed-allowed"},
      {"malformed optional semantic ages reject", "chains/0/issue_ages", "0,10,20"},
      {"unsupported qualified proof protocol rejects", "pairs/0/verilator/scope", "qualified protocol"},
      {"malformed retained pair sweep rejects", "pairs/0/previous_swept", "1,73"}}, reject);
  auto omitted = overlapFixture(*facts);
  (*omitted.getArray("chains"))[0].getAsObject()->getArray("all_pair_relations")->pop_back();
  reject(std::move(omitted), "nonadjacent chain relations cannot be omitted");
}

void testComposed() {
  auto source = fixture();
  (*source.getObject("model"))["revision"] = "9e6313bddecb8ba936206eb347f5cd4e5335282c";
  (*source.getObject("model"))["source_sha256"] = "a9a919c3d118af94e2783c8969e05ad19f8d2523ec093922c4f9957641b54426";
  source.getArray("op_timing")->push_back(block("mxu0.push_acc_fp8", "mxu0", 33,
      Object{{"mreg_read1", group(0, 32)}, {"acc_load", group(1, 32)}, {"data_busy", group(0, 32)}}));
  std::string sourceBytes = encode(std::move(source));
  auto parsed = parseModelTimingFacts(sourceBytes, hash(sourceBytes));
  check(bool(parsed), "composition model fixture loads");
  if (!parsed) { llvm::consumeError(parsed.takeError()); return; }
  auto facts = std::make_shared<ModelTimingFacts>(std::move(*parsed));
  auto original = overlapFixture(*facts);
  for (const char *key : {"revision", "source_sha256"})
    (*original.getObject("model"))[key] = facts->modelProvenance().at(key);
  original["chains"] = Array{};
  std::string base = encode(std::move(original));
  auto loaded = parseModelOverlapFacts(base, hash(base), *facts);
  check(bool(loaded), "composition base loads");
  if (!loaded) { llvm::consumeError(loaded.takeError()); return; }
  auto side = [&] {
    return Object{{"schema", "atlas.model_composed.v1"}, {"rule", "row-footprints-and-structural-capacity-v1"},
        {"age_convention", "issue_edge_zero"},
        {"provenance", Object{{"per_op_sha256", facts->factsSha256()}, {"base_pairs_sha256", hash(base)},
            {"master_sha256", std::string(64, 'a')}, {"deriver_sha256", std::string(64, 'b')},
            {"pair_replay_sha256", std::string(64, 'c')}, {"chain_replay_sha256", std::string(64, 'd')},
            {"holdouts_sha256", std::string(64, 'e')}, {"revision", facts->modelProvenance().at("revision")},
            {"configuration_sha256", facts->modelProvenance().at("configuration_sha256")}}},
        {"engines", Object{{"vpu", Object{{"operations", Array{"vpu.mov"}}, {"chains", 1},
            {"confirmed_chains", 1}, {"accepted_chains", 1}, {"negative_points", 1}, {"rejected_negatives", 1},
            {"holdout_chains", 1}, {"holdout_pairs", 0}, {"holdout_failures", 0}}}}},
        {"pair_classes", Array{Object{{"engine", "vpu"}, {"relation", Object{{"mreg_banks", "disjoint"}}},
            {"pairs", 1}, {"confirmed", 1}, {"equal", 1}, {"negative_points", 1}, {"negative_violations", 0}}}},
        {"pairs", Array{}}, {"chains", Array{}}};
  };
  std::vector<ModelOverlapCommand> commands{
      {"vpu.mov", Engine::Vpu, {0, 1, 4, 5}, {}, {}, 0, 73},
      {"vpu.mov", Engine::Vpu, {8, 9, 12, 13}, {}, {}, 40, 73},
      {"vpu.mov", Engine::Vpu, {16, 17, 20, 21}, {}, {}, 73, 73}};
  check(!loaded->admits(commands, true), "unmeasured vector refuses without side evidence");
  std::string bytes = encode(side());
  auto composed = *loaded;
  auto error = composed.addComposedFacts(bytes, hash(bytes), *facts);
  check(!error, "validated composition side loads");
  if (error) { llvm::consumeError(std::move(error)); return; }
  check(composed.admits(commands, true), "validated pair composition admits a new exact vector");
  check(!composed.admits(commands, false), "composition still requires explicit chains selection");
  commands[1].issue = 1; commands[2].issue = 71;
  check(!composed.admits(commands, true), "three live VPU slots refuse despite individually admitted pairs");
  commands[2].issue = 72;
  check(composed.admits(commands, true), "final write edge frees slot while retaining pair coverage");
  commands[2].registers = {32, 33, 36, 37};
  check(!composed.admits(commands, true), "unwitnessed relation class remains refused");
  commands[2].registers = {16, 17, 20, 21}; commands[2].block = "vpu.add";
  check(!composed.admits(commands, true), "operation outside selected composition domain remains refused");
  auto reject = [&](Object object, const char *message) {
    auto candidate = *loaded;
    std::string contents = encode(std::move(object));
    auto error = candidate.addComposedFacts(contents, hash(contents), *facts);
    check(bool(error), message);
    if (error) llvm::consumeError(std::move(error));
  };
  for (const char *key : {"equal", "confirmed"})
    invalidFacts(side(), {{"class requires full equality and a confirmed witness",
                           std::string("pair_classes/0/") + key, 0}}, reject);
  for (const char *key : {"accepted_chains", "rejected_negatives", "holdout_chains"})
    invalidFacts(side(), {{"chain class requires complete replay and holdouts",
                           std::string("engines/vpu/") + key, 0}}, reject);
  invalidFacts(side(), {
      {"any negative violation rejects class", "pair_classes/0/negative_violations", 1},
      {"failed holdout rejects class", "engines/vpu/holdout_failures", 1},
      {"side cannot select different footprints", "provenance/per_op_sha256", std::string(64, '0')},
      {"side cannot omit master provenance", "provenance/master_sha256", nullptr, true},
      {"unknown composition policy fails closed", "unknown_policy", true},
      {"confirmed chain count cannot exceed replay count", "engines/vpu/confirmed_chains", 2},
      {"negative holdout count rejects", "engines/vpu/holdout_pairs", -1},
      {"confirmed pair count cannot exceed replay count", "pair_classes/0/confirmed", 2}}, reject);
  auto wrongHash = loaded->addComposedFacts(bytes, std::string(64, '0'), *facts);
  check(bool(wrongHash), "side replacement fails the selected hash");
  if (wrongHash) llvm::consumeError(std::move(wrongHash));
  auto object = side();
  Object mxuAudit(*object.getObject("engines")->getObject("vpu"));
  mxuAudit["operations"] = Array{"mxu0.push_weight", "mxu0.push_acc_fp8", "mxu0.matmul"};
  (*object.getObject("engines"))["mxu0"] = std::move(mxuAudit);
  auto addPair = [&](const char *first, const char *second, Object relation) {
    object.getArray("pair_classes")->push_back(Object{{"engine", "mxu0"}, {"relation", Object(relation)},
        {"pairs", 1}, {"confirmed", 1}, {"equal", 1}, {"negative_points", 1}, {"negative_violations", 0}});
    object.getArray("pairs")->push_back(Object{{"first", first}, {"second", second},
        {"relation", std::move(relation)}, {"minimum", 1}, {"high", facts->block(first)->firstFreeAge}});
  };
  addPair("mxu0.push_weight", "mxu0.push_acc_fp8", Object{{"mreg_banks", "disjoint"}});
  addPair("mxu0.push_weight", "mxu0.matmul", Object{{"mreg_banks", "disjoint"}, {"weight_slot", "other"}});
  addPair("mxu0.push_acc_fp8", "mxu0.matmul", Object{{"mreg_banks", "disjoint"}, {"accumulator", "other"}});
  object.getArray("pairs")->push_back(Object{{"first", "mxu0.push_acc_fp8"}, {"second", "mxu0.push_weight"},
      {"relation", Object{{"mreg_banks", "disjoint"}}}, {"minimum", 1}, {"high", 33}});
  std::string mxuBytes = encode(std::move(object));
  auto mxu = *loaded;
  auto mxuError = mxu.addComposedFacts(mxuBytes, hash(mxuBytes), *facts);
  check(!mxuError, "MXU composition fixture loads");
  if (mxuError) { llvm::consumeError(std::move(mxuError)); return; }
  commands = {{"mxu0.push_weight", Engine::Mxu0, {0}, {}, 0, 0, 40},
              {"mxu0.push_acc_fp8", Engine::Mxu0, {4}, 0, {}, 1, 33},
              {"mxu0.matmul", Engine::Mxu0, {8}, 1, 1, 2, 95}};
  check(!mxu.admits(commands, true), "pair-safe triple refuses when P1 preference moves accumulator push onto P0");
  commands[2].issue = 33;
  check(mxu.admits(commands, true), "same triple admits once allocated P0 finishes");
  auto provider = modelTimingProvider(facts, std::make_shared<ModelOverlapFacts>(mxu), true);
  Instr accumulator{findOp("vmatpush.acc.fp8.mxu0"), 0, 4, 0};
  Instr weight{findOp("vmatpush.weight.mxu0"), 0, 0, 0};
  Instr compute{findOp("vmatmul.mxu0"), 1, 8, 1};
  auto state = provider.createReservations();
  check(state.value->reserve(accumulator, facts->resolve(accumulator, unknownRegs()), 0).empty() &&
        state.value->reserve(weight, facts->resolve(weight, unknownRegs()), 2).empty(),
        "retired-prefix regression begins with an admitted accumulator/weight pair");
  check(!state.value->conflict(compute, facts->resolve(compute, unknownRegs()), 33).value.empty(),
        "retired prefix still determines the live weight push port");
  check(state.value->conflict(compute, facts->resolve(compute, unknownRegs()), 34).value.empty(),
        "retired-prefix chain admits after its P0 request window ends");
}
void testStreamedWeights(const std::shared_ptr<ModelTimingFacts> &facts) {
  Instr push{findOp("vmatpush.weight.mxu0"), 0, 0, 0};
  Instr compute{findOp("vmatmul.mxu0"), 0, 2, 0};
  Footprint first = facts->resolve(push, unknownRegs()), second = facts->resolve(compute, unknownRegs());
  auto conservative = modelTimingProvider(facts);
  int original = conservative.dependence(push, first, compute, second).value.distance;
  check(first.error.empty() && second.error.empty() && original == 35,
        "unqualified weight dependency retains the whole isolated write window");
  auto qualify = [&](const char *a, const char *b, int minimum, int boundary) {
    auto object = overlapFixture(*facts);
    object["chains"] = Array{};
    object["pairs"] = Array{Object{
        {"first", a}, {"second", b},
        {"relation", Object{{"mreg_banks", "disjoint"}, {"weight_slot", "same"}}},
        {"swept", Array{1, boundary}}, {"min_gap", minimum}, {"first_live_window_end", boundary},
        {"negative", minimum == 1 ? Object{{"gap", 0}, {"result", "inexpressible"}} :
            Object{{"gap", minimum - 1}, {"result", "conflict"},
                   {"error", Object{{"message", "synthetic weight slot busy"}}}}},
        {"source_record", overlapSource("/pairs/8")}, {"verilator", Object{{"confirmed", true}}}}};
    std::string bytes = encode(std::move(object));
    auto loaded = parseModelOverlapFacts(bytes, hash(bytes), *facts);
    check(bool(loaded), "exact directional weight witness loads");
    if (!loaded) { llvm::consumeError(loaded.takeError()); return std::shared_ptr<ModelOverlapFacts>(); }
    return std::make_shared<ModelOverlapFacts>(std::move(*loaded));
  };
  auto overlap = qualify("mxu0.push_weight", "mxu0.matmul", 1, 40);
  if (!overlap) return;
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
  overlap = qualify("mxu0.matmul", "mxu0.push_weight", 63, 95);
  if (!overlap) return;
  auto reuse = modelTimingProvider(facts, overlap);
  check(reuse.dependence(compute, second, push, first).value.distance == 63,
        "exact confirmed matmul-to-push witness replaces the synthetic weight WAR bound");
  check(reuse.dependence(push, first, compute, second).value.distance == original,
        "compute-to-push evidence does not weaken push-to-compute weight lifetime");
  push.rd = 1;
  check(reuse.dependence(compute, second, push, facts->resolve(push, unknownRegs())).value.distance ==
            conservative.dependence(compute, second, push, facts->resolve(push, unknownRegs())).value.distance,
        "other weight slot keeps its own dependency");
}

void testVloadWriteDuringRead(const std::shared_ptr<ModelTimingFacts> &facts) {
  auto provider = modelTimingProvider(facts);
  Instr reader{findOp("vmov"), 4, 0, 0}, load{findOp("vload"), 0, 2, 0};
  auto first = facts->resolve(reader, unknownRegs());
  auto second = facts->resolve(load, unknownRegs());
  check(second.writeDuringRead && provider.dependence(reader, first, load, second).value.distance == 1,
        "VLOAD writes may follow the first source's observed read sweep");
  auto reservations = provider.createReservations();
  check(reservations.value->reserve(reader, first, 0).empty() &&
            reservations.value->conflict(load, second, 1).value.empty(),
        "VLOAD write-during-read preserves independent physical MREG ports");
  load.rd = 1;
  second = facts->resolve(load, unknownRegs());
  check(provider.dependence(reader, first, load, second).value.distance == 31,
        "VLOAD to the second BF16 source waits for that register's row sweep");
  check(!reservations.value->conflict(load, second, 30).value.empty() &&
            reservations.value->conflict(load, second, 31).value.empty(),
        "one-short VLOAD WAR spacing still collides on a physical MREG row");
  load.rd = 4;
  second = facts->resolve(load, unknownRegs());
  check(provider.dependence(reader, first, load, second).value.distance == first.writeRelease + 1,
        "VLOAD exemption never bypasses a pending logical-register writer");
  Instr writer{findOp("vmov"), 0, 8, 0};
  check(provider.dependence(reader, first, writer, facts->resolve(writer, unknownRegs())).value.distance ==
            first.readRelease + 1,
        "non-VLOAD writers retain whole-register WAR release");
}
} // namespace

void testExplicitReleases() {
  auto release = [](int last, Array banks = {}) {
    Object value{{"first_age", 0}, {"release_age", last}, {"available_age", last + 1}};
    if (!banks.empty()) value["banks"] = std::move(banks);
    return value;
  };
  auto sample = fixture();
  sample["op_timing"] = Array{(*sample.getArray("op_timing"))[0]};
  sample["reservation_releases"] = Object{{"vpu.mov", Object{
      {"unit", release(72)}, {"mreg_read", release(64, Array{0, 1})},
      {"mreg_write", release(72, Array{4, 5})}}}};
  auto bytes = encode(Object(sample));
  auto parsed = parseModelTimingFacts(bytes, hash(bytes));
  check(bool(parsed), "explicit reservation releases load");
  if (!parsed) { llvm::consumeError(parsed.takeError()); return; }
  auto provider = modelTimingProvider(std::make_shared<ModelTimingFacts>(std::move(*parsed)));
  Instr first{findOp("vmov"), 4, 0, 0}, second{findOp("vmov"), 0, 8, 0};
  Footprint a = provider.footprint(first, unknownRegs()), b = provider.footprint(second, unknownRegs());
  check(a.error.empty() && a.readRelease == 64 && a.writeRelease == 72 && a.doneAge == 72,
        "explicit MREG release replaces the legacy last-read-plus-latency rule");
  auto dependency = provider.dependence(first, a, second, b);
  check(dependency.error.empty() && dependency.value.distance == 65,
        "a conflicting write rejects the release edge and admits only the next age");
  for (const char *field : {"release_age", "available_age"}) {
    auto bad = Object(sample);
    (*bad.getObject("reservation_releases")->getObject("vpu.mov")->getObject("unit"))[field] = 71;
    rejected(std::move(bad), "unit reservation must match isolated drain and next-cycle availability");
  }
  auto bad = Object(sample);
  bad["reservation_releases"] = Object{};
  rejected(std::move(bad), "partial reservation coverage rejects");
  bad = Object(sample);
  (*bad.getObject("reservation_releases")->getObject("vpu.mov")->getObject("mreg_read"))["banks"] = Array{0, 0};
  rejected(std::move(bad), "duplicate reservation banks reject");
  bad = Object(sample);
  (*bad.getObject("reservation_releases")->getObject("vpu.mov")->getObject("mreg_read"))["banks"] = Array{0, 2};
  rejected(std::move(bad), "reservation banks must match canonical operand probes, not only their count");
  bad = Object(sample);
  (*bad.getObject("reservation_releases")->getObject("vpu.mov"))["mreg_read"] = release(63, Array{0, 1});
  bytes = encode(std::move(bad));
  parsed = parseModelTimingFacts(bytes, hash(bytes));
  check(bool(parsed), "early release is structurally bounded");
  if (parsed)
    check(!parsed->resolve(first, unknownRegs()).error.empty(), "reservation cannot end before the final MREG access");
  else llvm::consumeError(parsed.takeError());
  auto config = fixture();
  auto *model = config.getObject("model");
  (*model->getObject("configuration"))["vpu_simple_op_latency_cycles"] = 17;
  auto canonical = canonicalJson(llvm::json::Value(Object(*model->getObject("configuration"))));
  (*model)["configuration_json"] = canonical;
  (*model)["configuration_sha256"] = hash(canonical);
  bytes = encode(Object(config));
  auto changed = parseModelTimingFacts(bytes, hash(bytes));
  check(bool(changed), "configuration identity is data; bounded latency-profile changes need no compiler rebuild");
  if (!changed) llvm::consumeError(changed.takeError());
  auto rejectConfiguration = [&](const char *key, llvm::json::Value value, const char *why) {
    auto changed = Object(config);
    auto *model = changed.getObject("model");
    (*model->getObject("configuration"))[key] = std::move(value);
    auto canonical = canonicalJson(llvm::json::Value(Object(*model->getObject("configuration"))));
    (*model)["configuration_json"] = canonical;
    (*model)["configuration_sha256"] = hash(canonical);
    rejected(std::move(changed), why);
  };
  rejectConfiguration("dma_memory_backend", "fifo", "isolated facts do not qualify a changed DMA estimate profile");
  rejectConfiguration("ignore_runtime_errors", true, "even a correctly hashed configuration cannot suppress model runtime errors");
}

void testDMAEstimates() {
  auto parse = [](Object object) {
    std::string bytes = encode(std::move(object));
    return parseModelTimingFacts(bytes, hash(bytes), true);
  };
  auto baseline = parse(fixture());
  check(bool(baseline), "synthetic source-pinned DMA proposal facts load");
  if (!baseline) { llvm::consumeError(baseline.takeError()); return; }
  check(baseline->estimateDMACycles(32, false).value == 6 && baseline->estimateDMACycles(1024, false).value == 84 &&
        baseline->estimateDMACycles(32, true).value == 6 && baseline->estimateDMACycles(1024, true).value == 68,
        "cold estimates replay synthetic endpoint startup, latency and fractional pacing");
  check(!baseline->estimateDMACycles(0, false).error.empty() && !baseline->estimateDMACycles(33, true).error.empty() &&
        !baseline->estimateDMACycles(8192, false).error.empty(), "DMA proposals reject invalid transfer geometry");
  Object changed = fixture();
  changed.getObject("dma_timing")->getObject("startup_cycles")->operator[]("load") = 8;
  auto startup = parse(std::move(changed));
  check(bool(startup) && startup->estimateDMACycles(1024, false).value == 91,
        "DMA proposal consumes derived launch startup");
  if (!startup) llvm::consumeError(startup.takeError());
  changed = fixture();
  Object &read = *(*changed.getObject("dma_timing")->getArray("curves"))[1].getAsObject();
  read["peak_beats_per_cycle"] = .2;
  read["cycles_per_beat"] = 5.;
  (*(*read.getArray("points"))[0].getAsArray())[0] = .1;
  (*(*read.getArray("points"))[1].getAsArray())[0] = .2;
  auto slower = parse(std::move(changed));
  check(bool(slower) && slower->estimateDMACycles(1024, false).value == 161,
        "DMA proposal responds to derived curve pacing, not builtin calibration");
  if (!slower) llvm::consumeError(slower.takeError());
  changed = fixture();
  Object &longRead = *(*changed.getObject("dma_timing")->getArray("curves"))[1].getAsObject();
  longRead["unloaded_latency_cycles"] = 1000.;
  (*(*longRead.getArray("points"))[0].getAsArray())[1] = 1000.;
  (*(*longRead.getArray("points"))[1].getAsArray())[1] = 2000.;
  auto bounded = parse(std::move(changed));
  check(bool(bounded) && bounded->estimateDMACycles(4096, false).value == 2160,
        "isolated replay observes registered in-flight capacity before same-cycle response");
  if (!bounded) llvm::consumeError(bounded.takeError());
  changed = fixture();
  (*changed.getObject("dma_timing"))["reset_last_ready_cycles"] = 1000.;
  auto reset = parse(std::move(changed));
  check(bool(reset) && reset->estimateDMACycles(1024, false).value == 1080,
        "isolated replay consumes source-derived fractional ready reset");
  if (!reset) llvm::consumeError(reset.takeError());
  changed = fixture();
  changed.erase("dma_timing");
  auto legacy = parse(std::move(changed));
  check(bool(legacy), "old per-operation facts can still parse without DMA proposal coverage");
  if (legacy) {
    auto facts = std::make_shared<ModelTimingFacts>(std::move(*legacy));
    auto provider = modelTimingProvider(facts);
    check(facts->resolve({findOp("vmov"), 4, 0, 0}, unknownRegs()).error.empty(), "legacy non-DMA facts remain usable");
    check(!provider.dmaEstimate(1024, false).error.empty(), "missing DMA proposal coverage has no generic fallback");
    check(!facts->resolve({findOp("dma.wait.ch0"), 0, 0, 0}, unknownRegs()).error.empty(),
          "DMA resolution fails closed without dma_timing even for waits");
  } else llvm::consumeError(legacy.takeError());
  auto builtin = builtinTimingProvider();
  check(!builtin.dmaEstimate && dmaTransferCycles(1024, false) == 781 && dmaTransferCycles(1024, true) == 724,
        "builtin retains its exact existing DMA proposal behavior");
  check(baseline->estimateDMACycles(1024, false).value != dmaTransferCycles(1024, false),
        "model proposals use their own facts independently of builtin");
  for (const char *field : {"resolved", "estimate_only"})
    rejectedFacts(fixture(), {{"unresolved or non-proposal DMA facts reject",
                               std::string("dma_timing/") + field, false}});
  for (const char *field : {"beat_bytes", "max_transfer_bytes", "max_in_flight", "num_channels", "queue_depth"})
    rejectedFacts(fixture(), {{"DMA geometry must match explicit model configuration",
                               std::string("dma_timing/") + field, 1}});
  rejectedFacts(fixture(), {
      {"unreviewed DMA raw source rejects", "dma_timing/source_sha256/dma.py", std::string(64, '0')},
      {"unreviewed DMA source shape rejects", "dma_timing/source_shapes/memory_backend.py", std::string(64, '0')},
      {"unreviewed DMA curve rejects", "dma_timing/curve_sha256", std::string(64, '0')}});
  rejectedFacts(fixture(), {{"missing validation rejects", "dma_timing/validation", nullptr, true}});
  rejectedFacts(fixture(), {
      {"failed validation rejects", "dma_timing/validation/passed", false},
      {"malformed validation report pin rejects", "dma_timing/validation/report_sha256", "stale"},
      {"missing successful isolated probes reject", "dma_timing/validation/isolated_dma_runs", 0}});
  rejectedFacts(fixture(), {{"missing successful mixed probes reject", "dma_timing/validation/mixed_dma_runs", nullptr, true}});
  rejectedFacts(fixture(), {
      {"non-integer backend channel scaling rejects", "dma_timing/channels", .5},
      {"non-integer minimum latency rejects", "dma_timing/min_latency", 1.5},
      {"unknown DMA fact fields reject", "dma_timing/completion_proof", true}});
  rejectedFacts(fixture(), {{"missing DMA cost field rejects", "dma_timing/issue_interval_cycles", nullptr, true}});
  rejectedFacts(fixture(), {
      {"unsupported controller mode rejects", "dma_timing/latency_mode", "loaded"},
      {"unreviewed history semantics reject", "dma_timing/semantics/history_lifetime", "command_boundary"},
      {"inconsistent reciprocal pacing rejects", "dma_timing/curves/1/cycles_per_beat", 100.},
      {"point and unloaded latency disagreement rejects", "dma_timing/curves/1/unloaded_latency_cycles", 100.},
      {"out of range numeric estimate rejects", "dma_timing/rounding_epsilon_cycles", 2.}});
}

int main(int argc, char **argv) {
  if (argc == 2 && llvm::StringRef(argv[1]) == "--fixture") {
    llvm::outs() << encode(fixture()) << '\n';
    return 0;
  }
  if (argc != 1) return 2;
  testDMAEstimates();
  testExplicitReleases();
  std::string bytes = encode(fixture());
  auto loaded = parseModelTimingFacts(bytes, hash(bytes), true);
  check(bool(loaded), "synthetic observed facts load");
  if (!loaded) {
    llvm::errs() << llvm::toString(loaded.takeError()) << '\n';
    return 1;
  }
  auto facts = std::make_shared<ModelTimingFacts>(std::move(*loaded));
  testOverlap(facts);
  testComposed();
  testStreamedWeights(facts);
  testVloadWriteDuringRead(facts);
  failures += testScalarOverlap(fixture());
  auto provider = modelTimingProvider(facts);
  RegValues dmaRegs = unknownRegs();
  dmaRegs[1] = 0x20000 / 4; dmaRegs[2] = 0; dmaRegs[3] = 1024;
  Footprint dma = provider.footprint({findOp("dma.load.ch0"), 1, 2, 3}, dmaRegs);
  check(dma.error.empty() && dma.dmaCycles == 84 && dma.doneAge == 0,
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
  Instr store{findOp("vstore"), 5, 2, 0}, overwrite{findOp("vmov"), 4, 8, 0};
  auto storeWar = provider.dependence(store, provider.footprint(store, unknownRegs()), overwrite,
                                      provider.footprint(overwrite, unknownRegs()));
  check(storeWar.error.empty() && storeWar.value.distance == 37, "VSTORE source stays reserved through its last VMEM write");
  DialectRegistry registry;
  registry.insert<AtlasDialect>();
  MLIRContext context(registry);
  auto module = parseSourceString<ModuleOp>("module {}", &context);
  auto dmaScope = [&](unsigned nextChannel, bool conflictingStore) {
    std::string text = "module {\n  %s0 = \"atlas.start\"() : () -> !atlas.state\n";
    unsigned state = 0;
    auto append = [&](llvm::StringRef name, const std::string &properties) {
      unsigned previous = state++;
      text += "  %s" + std::to_string(state) + " = \"atlas." + name.str() + "\"(%s" +
              std::to_string(previous) + ") <{" + properties + "}> : (!atlas.state) -> !atlas.state\n";
    };
    append("dma_config", "channel = 0 : i32, base_reg = 0 : i32");
    append("alu_imm", "kind = \"addi\", dst = 1 : i32, src = 0 : i32, immediate = 0 : i32");
    append("alu_imm", "kind = \"addi\", dst = 3 : i32, src = 0 : i32, immediate = 32 : i32");
    append("upper", "kind = \"lui\", dst = 2 : i32, immediate = 524288 : i32");
    for (unsigned channel = 0; channel < 8; ++channel) {
      append("dma", "channel = " + std::to_string(channel) +
             " : i32, direction = \"load\", dram = 2 : i32, reg = 1 : i32, size = 3 : i32");
      append("alu_imm", "kind = \"addi\", dst = 2 : i32, src = 2 : i32, immediate = 32 : i32");
    }
    append("dma_wait", "channel = 7 : i32");
    if (conflictingStore) {
      append("dma_wait", "channel = 0 : i32");
      append("upper", "kind = \"lui\", dst = 2 : i32, immediate = 524288 : i32");
      append("alu_imm", "kind = \"addi\", dst = 2 : i32, src = 2 : i32, immediate = 32 : i32");
    }
    append("dma", "channel = " + std::to_string(nextChannel) + " : i32, direction = \"" +
           (conflictingStore ? "store" : "load") + "\", dram = 2 : i32, reg = 1 : i32, size = 3 : i32");
    text += "}\n";
    auto scopeModule = parseSourceString<ModuleOp>(text, &context);
    check(bool(scopeModule), "DMA scope fixture parses");
    if (!scopeModule) return std::string("fixture parse failed");
    auto stream = readAtlasStream(*scopeModule, AtlasStreamReadMode::Verification);
    check(succeeded(stream), "DMA scope fixture decodes");
    if (failed(stream)) return std::string("fixture decode failed");
    return provider.validateScope(*stream);
  };
  auto ringRefusal = dmaScope(7, false);
  check(ringRefusal.find("command slot is still pending") != std::string::npos,
        "a free channel does not free its RTL command ring slot");
  check(ringRefusal.find("launch 8 reuses slot 0 of unwaited launch 0") != std::string::npos,
        "a command ring refusal names the reused slot and its pending launch");
  check(dmaScope(0, false).find("free DMA channel") != std::string::npos,
        "out-of-order DMA wait does not release another pending channel");
  check(dmaScope(7, true).find("pending DMA DRAM spans overlap") != std::string::npos,
        "out-of-order DMA wait does not complete another transfer's DRAM access");

  auto cost = machineTimingOf(provider, "vload", *module);
  check(succeeded(cost) && cost->latency == 40 && cost->interval == 40,
        "virtual representative unknown-base query uses selected facts");
  for (const char *name : {"vmatpush.weight.mxu0", "vmatpop.bf16.acc.mxu0"}) {
    cost = machineTimingOf(provider, name, *module);
    check(succeeded(cost) && cost->latency == 40 && cost->interval == 40, "model MXU cost probes use legal operands");
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
  auto absentSelection = lookupTimingProvider("model", *module);
  check(!absentSelection.error.empty(), "registry rejects absent model selection");
  check(llvm::StringRef(absentSelection.error).ends_with(
      "See docs/model-timing.md#updating-for-npu-model-changes."),
      "missing selection refusal ends with model update guidance");
  check(lookupTimingProvider("builtin").error.empty(), "model registration preserves builtin");
  auto badHash = parseModelTimingFacts(bytes, std::string(64, '0'));
  check(!badHash, "facts content replacement fails the pin");
  if (!badHash) llvm::consumeError(badHash.takeError());
  for (const char *field : {"exporter_sha256", "input_sha256", "numerical_data_sha256"}) {
    auto incomplete = fixture();
    incomplete.getObject("model")->erase(field);
    rejected(std::move(incomplete), "incomplete provenance rejects");
  }
  rejectedFacts(fixture(), {
      {"RTL schema rejects", "schema", "merlin.op_timing.v1"},
      {"configuration digest cannot be forged", "model/configuration_sha256", std::string(64, '0')},
      {"runtime error suppression is outside the configuration profile", "model/configuration/ignore_runtime_errors", true},
      {"RTL provenance rejects", "hw_ir", Object{}},
      {"pair admission rejects", "pairs", Array{}},
      {"unsupported geometry rejects", "geometry/rows", 16},
      {"unconfirmed overlap rejects", "op_timing/0/next_issue_age", 1},
      {"inconsistent event shape rejects", "op_timing/0/events/read0/count", 1},
      {"unknown fact operation rejects", "op_timing/0/name", "unknown.operation"}});
  llvm::outs() << "model timing synthetic checks, " << failures << " failures\n";
  return failures != 0;
}
