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
#include <climits>
#include <tuple>

using namespace mlir;
using namespace mlir::atlas;
using namespace mlir::atlas::timing;
using Object = llvm::json::Object;
using Array = llvm::json::Array;

int testScalarOverlap(Object fixture);
int testEngineAdmission();

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
Object block(const std::string &name, const std::string &engine, int free, Object events) {
  return Object{{"name", name}, {"engine", engine}, {"resolved", true}, {"first_free_age", free},
          {"next_issue_age", free}, {"events", std::move(events)},
          {"assumptions", Object{{"scratchpad_read_latency", 1}}},
          {"observation", Object{{"accepted", true}, {"drained", true}}}};
}
Object dmaSourceFixture() {
  Object sources, shapes;
  for (const auto &[key, value] : model_schema::dmaIdentity())
    if (key != "curve_sha256" && !llvm::StringRef(key).starts_with("shape:")) sources[key] = value;
  for (const auto &[name, shape] : model_schema::dmaSourceShapes()) {
    shapes[name] = shape;
    if (!sources.get(name)) sources[name] = std::string(64, 'a');
  }
  Array phases;
  for (const auto &phase : model_schema::dmaEdgeOrder()) phases.push_back(phase);
  return Object{{"schema", "atlas.dma_source.v1"}, {"source_sha256", std::move(sources)},
      {"source_shapes", std::move(shapes)}, {"core_tick_shape", std::string(64, 'a')},
      {"core_s1_shape", std::string(64, 'b')}, {"edge_order", std::move(phases)},
      {"issue", Object{{"ordinary_gap", 1}, {"first_engine_age", 1}, {"store_read_to_request_cycles", 2}}}};
}
Object proposalFixture(double maximum = 5., int quantum = 6) {
  auto policy = llvm::json::parse(search_policy::kDMAProposalPolicyJson);
  assert(policy && policy->getAsObject());
  return Object{{"schema", "atlas.dma_proposal.v1"}, {"kind", "serialized_all_mix_upper_bound"},
      {"response_to_retire_cycles", 0}, {"retire_to_wait_issue_cycles", 1}, {"per_beat_bound_cycles", quantum},
      {"preconditions", Object{{"engine_backend_drained_at_launch", true}, {"vmem_grant_denials", 0},
                               {"history", "arbitrary_retained_issue_history"},
                               {"maximum_cycle", INT_MAX}, {"maximum_arithmetic_operand", INT_MAX}}},
      {"policy", std::move(*policy)},
      {"recurrence", Object{{"retire", "launch+startup[direction]+beats*per_beat_bound_cycles"},
                            {"wait", "retire+retire_to_wait_issue_cycles"}}},
      {"source_evidence", Object{{"core_tick_shape", std::string(64, 'a')}, {"core_s1_shape", std::string(64, 'b')},
          {"expressions", Object{{"fixture", "synthetic proof premises"}}},
          {"edge_order", Array{"Core.S1", "DMA.D", "DMA.store_read", "DMA.A", "DMA.retire_clear_flag", "DMA.enqueue_launch"}},
          {"service_maximum_cycles", maximum},
          {"service_quantum", "ceil(max(all_unloaded_latency,min_latency,all_spacing))+issue_interval"}}},
      {"proof", Array{"Synthetic contract fixture; production proof comes from checked model source."}},
      {"limitations", Array{"Drained engine/backend and no VMEM grant denial."}}};
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
      {"proposal_contract", proposalFixture()},
      {"source_contract", dmaSourceFixture()},
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
// Snapshot of derive_arch(configuration()) on the pinned model, used only by
// synthetic provider tests. Production facts are always derived by the driver.
Object architectureFixture() {
  return Object{
      {"bf16_element_bytes", 2},
      {"fp8_element_bytes", 1},
      {"fp8_registers_per_tile", 1},
      {"fp8_tile_bytes", 1024},
      {"bf16_pair_stride", 1},
      {"bf16_registers_per_tile", 2},
      {"bf16_tile_bytes", 2048},
      {"csr_count", 4096},
      {"dma_beat_bytes", 32},
      {"dma_channels", 8},
      {"dma_max_in_flight", 64},
      {"dma_max_transfer_bytes", 4096},
      {"dma_offchip_command_words", 2},
      {"dram_base", 2147483648LL},
      {"dram_bytes", 68719476736LL},
      {"ereg_count", 32},
      {"fetch_width", 1},
      {"mreg_bank_modulus", 32},
      {"mreg_bytes", 1024},
      {"mreg_count", 64},
      {"mreg_logical_limit", 64},
      {"mreg_physical_bank_count", 32},
      {"mreg_physical_row_stride", 32},
      {"mreg_registers_per_bank", 2},
      {"mreg_row_bytes", 32},
      {"mreg_rows", 32},
      {"offchip_link_core_cycles_per_beat", 2},
      {"offchip_link_width_bits", 32},
      {"scalar_word_bytes", 4},
      {"schema", "atlas.architecture.v1"},
      {"vmem_bank_bytes", 262144},
      {"vmem_bank_count", 6},
      {"vmem_bus_core_cycles_per_beat", 1},
      {"vmem_bus_width_bits", 512},
      {"vmem_bytes", 1572864},
      {"vmem_bytes_per_cycle", 64},
      {"weight_buffer_bytes", 1024},
      {"weight_buffer_count", 2},
      {"xreg_count", 32}};
}
Object isaFixture() {
  Object metadata;
  auto add = [&](const std::string &name, const char *engine) {
    metadata[name] = Object{{"compiler_operands", findOp(name)->operands}, {"engine", engine}};
  };
  for (const auto &[name, block] : model_schema::vpuBlocks()) add(name, "vpu");
  for (int m = 0; m < 2; ++m)
    for (const char *name : {"vmatpush.weight", "vmatpush.acc.fp8", "vmatpush.acc.bf16", "vmatpop.fp8.acc",
                             "vmatpop.bf16.acc", "vmatmul", "vmatmul.acc"})
      add(std::string(name) + ".mxu" + std::to_string(m), m == 0 ? "mxu0" : "mxu1");
  for (const char *name : {"vload", "vstore"}) add(name, "vlsu");
  for (const char *name : {"lw", "sw", "seld"}) add(name, "scalar_lsu");
  add("vtrpose.xlu", "xlu");
  auto roles = [&](const char *name, const char *destinationType, const char *destinationResource,
                   const char *destinationField, bool destinationWrite, bool sourceMatrix = false) {
    Array operands;
    operands.push_back(Object{{"type", destinationType}, {"resource", destinationResource},
        {"compiler_operand", destinationField}, {"read", !destinationWrite}, {"write", destinationWrite}});
    operands.push_back(Object{{"type", sourceMatrix ? "MatrixReg" : "ScalarReg"},
        {"resource", sourceMatrix ? "MReg" : "XReg"}, {"compiler_operand", "rs1"}, {"read", true}, {"write", false}});
    (*metadata.getObject(name))["operands"] = std::move(operands);
  };
  roles("lw", "ScalarReg", "XReg", "rd", true);
  roles("sw", "ScalarReg", "XReg", "rs2", false);
  roles("seld", "ExponentReg", "EReg", "rd", true);
  roles("vload", "MatrixReg", "MReg", "rd", true);
  roles("vstore", "MatrixReg", "MReg", "rd", false);
  roles("vtrpose.xlu", "MatrixReg", "MReg", "rd", true, true);
  return metadata;
}
Object weightFixture() {
  Object engines;
  for (int unit = 0; unit < 2; ++unit) {
    int step = unit == 0 ? 1 : 0;
    Array rows;
    for (int row = 0; row < 32; ++row)
      rows.push_back(Object{{"first_age", 1 + row * step}, {"last_age", 32 + (row + 31) * step}});
    engines["mxu" + std::to_string(unit)] = Object{
        {"use_window", Object{{"first_age", 1}, {"last_age", 32}, {"row_step", step}, {"column_step", step}}},
        {"use_rows", std::move(rows)}, {"reads_before_writes", true}, {"resource_row_axis", "output_column"}};
  }
  return Object{{"schema", "atlas.mxu_weight_timing.v1"}, {"engines", std::move(engines)}};
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
  return Object{{"schema", "atlas.model_timing.v1"}, {"test_only", true}, {"age_convention", "issue_edge_zero"},
          {"model", Object{{"revision", std::string(40, '1')}, {"source_sha256", std::string(64, '2')},
                                       {"configuration_sha256", hash(configuration)},
                                       {"configuration", std::move(decodedConfiguration)},
                                       {"configuration_json", configuration},
                                       {"observer_sha256", std::string(64, '4')},
                                       {"exporter_sha256", std::string(64, '5')},
                                       {"input_sha256", std::string(64, '6')},
                                       {"numerical_data_sha256", std::string(64, '7')}}},
          {"geometry", Object{{"rows", 32}, {"columns", 32}, {"mreg_count", 64}}},
          {"architecture", architectureFixture()}, {"isa_metadata", isaFixture()},
          {"mxu_weight_timing", weightFixture()},
          {"operand_vocabulary", llvm::cantFail(llvm::json::parse(model_schema::vocabularyJson()))},
          {"observation", Object{{"accepted", true}, {"drained", true}}},
          {"op_timing", std::move(operations)}, {"dma_timing", dmaFixture()}};
}
std::string encode(Object object) {
  std::string bytes;
  llvm::raw_string_ostream stream(bytes);
  stream << llvm::json::Value(std::move(object));
  return bytes;
}
// Internal parser fixtures exercise the production validation code. They are
// deliberately constructed test data and are never exported by --fixture.
void sealContractFixture(Object &object) {
  Object semanticModel;
  for (const char *key : {"revision", "revision_kind", "source_sha256", "numerical_data_sha256",
                          "configuration", "configuration_sha256"})
    semanticModel[key] = *object.getObject("model")->get(key);
  Object semantic{{"model", std::move(semanticModel)}};
  for (const char *key : {"geometry", "age_convention", "op_timing", "reservation_releases", "vocabulary_sha256",
                          "architecture", "operand_vocabulary", "isa_metadata"})
    if (const auto *value = object.get(key)) semantic[key] = *value;
  for (const char *key : {"frontend_timing", "scratchpad_contract", "overlap_rules_sha256", "mxu_weight_timing", "replay_timing", "proposal_costs"})
    if (const auto *value = object.get(key)) semantic[key] = *value;
  for (const char *key : {"dma_timing", "engine_admission", "dma_admission"})
    if (auto *document = object.getObject(key)) {
      Object deterministic(*document);
      deterministic.erase("validation");
      deterministic.erase("producer_sources");
      semantic[key] = std::move(deterministic);
    }
  auto bytes = canonicalJson(llvm::json::Value(std::move(semantic)));
  auto payloadHash = hash(bytes);
  object["derivation_contract"] = Object{{"schema", "atlas.derived_facts.v1"}, {"payload_json", bytes},
      {"payload_sha256", payloadHash},
      {"derivation", Object{{"schema", "atlas.derivation.v1"}, {"kind", "source_derived"}, {"payload_sha256", payloadHash}}},
      {"validation", Object{{"schema", "atlas.validation.v1"}, {"passed", true}, {"payload_sha256", payloadHash}}}};
  object["validation"] = Object{{"passed", true}, {"operation_count", int64_t(object.getArray("op_timing")->size())},
      {"observed_runs", 1}, {"unobserved_runs", 1}, {"gate_report_sha256", std::string(64, 'a')},
      {"payload_sha256", payloadHash}};
  if (object.get("mxu_weight_timing"))
    (*object.getObject("validation"))["mxu_weight_validation"] = Object{{"passed", true}};
  if (auto *dma = object.getObject("dma_timing"))
    (*dma->getObject("validation"))["payload_sha256"] = payloadHash;
  for (const char *key : {"engine_admission", "dma_admission"})
    if (auto *document = object.getObject(key)) (*document->getObject("validation"))["payload_sha256"] = payloadHash;
}

Object replayFixture() {
  auto parsed = llvm::json::parse(search_policy::kReplayProjectionJson);
  assert(parsed && parsed->getAsObject());
  Object result(*parsed->getAsObject());
  result["schema"] = "atlas.projected_core_replay.v1";
  result["source"] = Object{{"schema", "atlas.core_edges.v1"}, {"clock_step", 1},
      {"dma_wait_observes", "previous_edge_flags"}, {"implicit_exu_stalls", false},
      {"source_contract", "atlas.model_frontend.v1"},
      {"phases", Array{"s1_previous_state", "execution_units", "fetch"}},
      {"checked_methods", Array{"Core.tick", "Core._tick_s1"}}};
  (*result.getObject("projection_policy"))["template_sha256"] = search_policy::kReplayProjectionSHA256;
  for (const char *stage : {"start", "place"})
    for (auto &assignment : *result.getArray(stage))
      for (auto &token : *(*assignment.getAsArray())[1].getAsArray())
        if (token.getAsString() == "$clock_step") token = 1;
  return result;
}

Object costsFixture() {
  Object probes;
  for (const char *name : {"vmov", "addi"})
    probes[name] = Array{Object{{"rd", 0}, {"rs1", 2}, {"rs2", 0}, {"imm", 0}},
                         Object{{"rd", 6}, {"rs1", 8}, {"rs2", 0}, {"imm", 0}}};
  return Object{{"schema", "atlas.proposal_costs.v1"},
      {"placement_policy", search_policy::kProposalOperandPlacement}, {"probes", std::move(probes)},
      {"timing_class", Object{{"kind", "guarded_operand_independent_isolated_ages"},
          {"vpu", "mnemonic_latency_and_register_independent_row_predicates"},
          {"mxu", "guarded_age_geometry_expressions"}, {"memory", "guarded_age_geometry_predicates"},
          {"interval_scope", "selected_representative_pair_only"}}}};
}

Object admissionFixture(const Object &object) {
  const auto &model = *object.getObject("model");
  Object identity, sources, final;
  for (const char *key : {"source_sha256", "configuration_sha256", "numerical_data_sha256"}) identity[key] = *model.get(key);
  for (const char *key : {"core.py", "xlu.py", "lsu.py", "vpu.py", "mxu.py", "bank_conflict.py"}) sources[key] = std::string(64, 'a');
  for (const auto &value : *object.getArray("op_timing")) {
    const auto &block = *value.getAsObject();
    if (block.getString("engine") == "vpu") final[*block.getString("name")] = *block.getInteger("first_free_age") - 1;
  }
  return Object{{"schema", "atlas.engine_admission.v1"}, {"age_convention", "issue_edge_zero"},
      {"contract", "source-engine-initiation-v1"}, {"model", std::move(identity)},
      {"source_contract", Object{{"source_sha256", std::move(sources)},
          {"core_init_shape", "5abd67591485003d3f345e7b4f15564b3b3d8a4ff8449ed3efd5a392cf03adf9"},
          {"execution_units", *model.getObject("configuration")->get("execution_units")}}},
      {"vpu", Object{{"capacity", 2}, {"live", "age_lt_final_write"}, {"final_write_age", std::move(final)},
          {"logic_groups", Array{}}, {"double", Array{}}}},
      {"xlu", Object{{"independent_from_vpu", true}, {"initiation", "prior_first_free"}, {"initiation_age", 66}}},
      {"vlsu", Object{{"independent_load_store_paths", true}, {"initiation", "same_path_prior_first_free"},
          {"initiation_ages", Object{{"vlsu.vload", 35}, {"vlsu.vstore", 35}}}}},
      {"scalar_lsu", Object{{"load_response_gap", 3}, {"store_busy", "none"}, {"independent_vector_paths", true}}},
      {"policy", Object{{"mreg_ports", "vpu_same_logical_register_row_read_coalescing"}, {"unknown_vmem", "aliases_all_banks"},
          {"scalar_lsu", "existing_alias_dependency_and_writeback_checks"}, {"mxu", "existing_source_overlap_rules"},
          {"waits", "invalidate_live_admission"}}},
      {"validation", Object{{"passed", true}, {"report_sha256", std::string(64, 'a')},
          {"positive_runs", 1}, {"negative_runs", 1}, {"numerical_checks", 1}}}};
}

Object dmaAdmissionFixture(const Object &object) {
  Object geometry;
  const auto &shared = *object.getObject("dma_timing")->getObject("source_contract");
  const auto &config = *object.getObject("model")->getObject("configuration");
  for (auto [key, field] : std::initializer_list<std::pair<const char *, const char *>>{
      {"num_channels", "dma_num_channels"}, {"beat_bytes", "dma_beat_bytes"}, {"max_in_flight", "dma_max_in_flight"},
      {"max_transfer_bytes", "dma_max_transfer_bytes"}, {"vmem_bank_bytes", "vmem_bank_bytes"}})
    geometry[key] = *config.get(field);
  return Object{{"schema", "atlas.dma_admission.v1"}, {"age_convention", "issue_edge_zero"},
      {"contract", "source-asynchronous-dma-admission-v1"}, {"source_sha256", *shared.get("source_sha256")},
      {"source_shapes", *shared.get("source_shapes")}, {"configuration", std::move(geometry)},
      {"source_contract_sha256", hash(canonicalJson(llvm::json::Value(Object(shared))))},
      {"issue", *shared.get("issue")},
      {"semantics", Object{{"operands", "captured_at_issue"}, {"launch", "enqueue_after_registered_progress"},
          {"requests", "oldest_undispatched_slot"}, {"vmem_grants", "lsu_priority_then_dma_load_write_then_dma_store_read"},
          {"completion", "all_beats_dispatched_and_acknowledged"}, {"wait", "busy_flag_checked_before_engine_progress"},
          {"channel_reuse", "matching_wait_required"}, {"aliases", "existing_completion_conflict_policy"},
          {"curve_history", "persists_across_wait_and_engine_drain"}, {"fixed_dma_windows", "none"},
          {"command_ring", "separate_unchanged_rtl_contract"}}},
      {"producer_sources", Object{{"atlas_derive_dma_admission.py", std::string(64, 'a')},
                                    {"atlas_validate_dma_admission.py", std::string(64, 'b')}}},
      {"validation", Object{{"passed", true}, {"report_sha256", std::string(64, 'a')},
                               {"positive_runs", 94}, {"negative_runs", 6}, {"numerical_checks", 94}}}};
}


Object contractFixture() {
  Object object = fixture();
  object.erase("test_only");
  (*object.getObject("model"))["revision_kind"] = "locked_source";
  object["vocabulary_sha256"] = model_schema::vocabularySha256();
  object["op_timing"] = Array{(*object.getArray("op_timing"))[0]};
  auto release = [](int age, Array banks = {}) {
    Object value{{"first_age", 0}, {"release_age", age}, {"available_age", age + 1}};
    if (!banks.empty()) value["banks"] = std::move(banks);
    return value;
  };
  object["reservation_releases"] = Object{{"vpu.mov", Object{
      {"unit", release(72)}, {"mreg_read", release(64, Array{0, 1})},
      {"mreg_write", release(72, Array{4, 5})}}}};
  object["frontend_timing"] = Object{{"schema", "atlas.model_frontend.v1"}, {"ordinary_issue_gap", 1},
      {"delay", Object{{"base_gap", 1}, {"immediate_mask", 4095}, {"decrement_per_tick", 1}}},
      {"scalar", Object{{"read_age", 0}, {"write_age", 0}, {"completion_delay", 1}}},
      {"control", Object{{"delay_slots", 1}, {"operand_read_age", 0}, {"jal_link_pc_offset", 1}}},
      {"instructions", Object{
          {"addi", Object{{"kind", "scalar"},
              {"reads", Array{Object{{"resource", "xreg"}, {"operand", "rs1"}}}},
              {"writes", Array{Object{{"resource", "xreg"}, {"operand", "rd"}}}}}},
          {"delay", Object{{"kind", "delay"}, {"reads", Array{}}, {"writes", Array{}}}}}}};
  object["scratchpad_contract"] = Object{{"schema", "atlas.scratchpad_contract.v1"},
      {"status", "explicit-target-assumption"}, {"assumptions", Object{{"scratchpad_read_latency", 1}}},
      {"response_latency_derived", false}, {"dma_store_read_to_request_cycles", 2},
      {"limitation", "Synthetic parser fixture; response latency remains an explicit target assumption."},
      {"checked_source_methods", Array{"synthetic:read"}}};
  object["replay_timing"] = replayFixture();
  object["proposal_costs"] = costsFixture();
  object["engine_admission"] = admissionFixture(object);
  object["dma_admission"] = dmaAdmissionFixture(object);
  sealContractFixture(object);
  return object;
}

void rejected(Object object, llvm::StringRef why) {
  std::string bytes = encode(std::move(object));
  auto parsed = parseModelTimingFactsForTesting(bytes, hash(bytes));
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

void testDerivedContract() {
  auto parse = [](Object object) {
    auto bytes = encode(std::move(object));
    return parseModelTimingFacts(bytes, hash(bytes), true);
  };
  auto strict = parse(contractFixture());
  check(bool(strict), "production parser accepts a complete payload-bound derived contract fixture");
  if (!strict) llvm::errs() << llvm::toString(strict.takeError()) << '\n';
  if (strict) {
    auto provider = modelTimingProvider(std::make_shared<ModelTimingFacts>(*strict));
    check(provider.proposalGeometry.registerBytes == 1024 && provider.proposalGeometry.bf16Registers == 2 &&
          provider.proposalGeometry.fp8Registers() == 1 && provider.proposalGeometry.packRowWords() == 8 &&
          provider.proposalGeometry.rows == 32, "model proposal geometry retains source-derived architecture");
    check(provider.frontendSpacing.successorGap() == 2, "branch successor spacing uses derived slot count and issue gap");
    auto probes = provider.costProbes(*findOp("vmov"));
    check(probes.error.empty() && probes.value.second.rd == 6 && probes.value.second.rs1 == 8,
          "model costs use sealed generated operand probes");
    auto missing = provider.costProbes(*findOp("vadd.bf16"));
    check(!missing.error.empty(), "missing production cost probes refuse instead of using fixture defaults");
  }
  auto raw = parse(fixture());
  check(!raw, "production parser refuses test-only raw measured fixtures");
  if (!raw) llvm::consumeError(raw.takeError());
  auto unmarked = fixture();
  unmarked.erase("test_only");
  raw = parse(std::move(unmarked));
  check(!raw, "production parser refuses unmarked raw observations without a derivation contract");
  if (!raw) llvm::consumeError(raw.takeError());
  auto rejects = [&](Object object, const char *why, const char *diagnostic) {
    auto result = parse(std::move(object));
    check(!result, why);
    if (!result) check(llvm::toString(result.takeError()).find(diagnostic) != std::string::npos, why);
  };
  invalidFacts(contractFixture(), {
      {"versioned contract required", "derivation_contract/schema", "atlas.derived_facts.v0"},
      {"measured derivation refuses production", "derivation_contract/derivation/kind", "measured"},
      {"failed operation validation refuses production", "validation/passed", false},
      {"missing validation-only run coverage refuses production", "validation/unobserved_runs", 0},
      {"vocabulary is mandatory", "vocabulary_sha256", nullptr, true},
      {"stale semantic input binding rejects", "dma_timing/startup_cycles/load", 8},
      {"stale DMA validation payload rejects", "dma_timing/validation/payload_sha256", std::string(64, 'b')},
      {"self-reported DMA report cannot replace payload binding", "dma_timing/validation/payload_sha256", nullptr, true},
      {"test-only contract refuses production", "test_only", true}},
      [&](Object object, const char *why) {
        auto result = parse(std::move(object));
        check(!result, why);
        if (!result) llvm::consumeError(result.takeError());
      });
  invalidFacts(contractFixture(), {
      {"DMA source reference must bind its authoritative record", "dma_admission/source_contract_sha256", std::string(64, 'b')},
      {"DMA source copies cannot disagree", "dma_admission/source_sha256/core.py", std::string(64, 'b')},
      {"DMA staging copies cannot disagree", "dma_admission/issue/store_read_to_request_cycles", 1},
      {"DMA proposal Core phase copies cannot disagree", "dma_timing/proposal_contract/source_evidence/core_tick_shape", std::string(64, 'b')},
      {"DMA shared source cannot change supported phase semantics", "dma_timing/source_contract/source_shapes/core.py", std::string(64, 'b')},
      {"DMA frontend issue copy cannot disagree", "dma_timing/source_contract/issue/ordinary_gap", 2},
      {"VPU metadata capacity is bounded", "engine_admission/vpu/capacity", 4}},
      [&](Object object, const char *why) {
        sealContractFixture(object);
        auto result = parse(std::move(object));
        check(!result, why);
        if (!result) llvm::consumeError(result.takeError());
      });
  for (int capacity : {1, 3}) {
    auto metadata = contractFixture();
    (*metadata.getObject("engine_admission")->getObject("vpu"))["capacity"] = capacity;
    sealContractFixture(metadata);
    auto result = parse(std::move(metadata));
    check(bool(result), "VPU metadata accepts supported capacities before selected overlap cross-check");
    if (!result) llvm::consumeError(result.takeError());
  }
  auto receiptOnly = contractFixture();
  (*receiptOnly.getObject("dma_timing"))["producer_sources"] = Object{{"receipt", "not a semantic rule"}};
  // Parser schema remains exact; this field is not a supported DMA extension,
  // even though all validated-record digest exclusions use the same helper.
  rejects(std::move(receiptOnly), "unknown DMA producer field does not change semantic binding", "resolved proposal-only curve schema");
  for (const char *key : {"architecture", "operand_vocabulary", "isa_metadata", "mxu_weight_timing"}) {
    auto stale = contractFixture();
    (*stale.getObject(key))["stale_metadata"] = true;
    rejects(std::move(stale), "stale metadata is rejected by the semantic payload binding",
            "derived semantic payload differs");
  }
  for (const char *field : {"replay_timing", "proposal_costs"}) {
    auto missing = contractFixture();
    missing.erase(field);
    sealContractFixture(missing);
    rejects(std::move(missing), "production proposal facts refuse missing derived extensions", "missing source-derived");
  }
  for (const char *field : {"replay_timing_sha256", "proposal_costs_sha256"}) {
    auto spoofed = contractFixture();
    (*spoofed.getObject("model"))[field] = std::string(64, 'b');
    sealContractFixture(spoofed);
    rejects(std::move(spoofed), "model metadata cannot override computed proposal identity", "reserved derived provenance");
  }
  auto alteredProof = contractFixture();
  (*alteredProof.getObject("proposal_costs")->getObject("timing_class"))["interval_scope"] = "all_operands";
  sealContractFixture(alteredProof);
  rejects(std::move(alteredProof), "representative intervals cannot claim universal operand coverage", "timing-class proof");
  auto missingRelease = contractFixture();
  missingRelease.getObject("reservation_releases")->getObject("vpu.mov")->erase("mreg_read");
  sealContractFixture(missingRelease);
  rejects(std::move(missingRelease), "missing resource coverage rejects even with rebound payload", "resource coverage");
  auto missingReleases = contractFixture();
  missingReleases.erase("reservation_releases");
  sealContractFixture(missingReleases);
  rejects(std::move(missingReleases), "missing releases reject even with rebound payload", "reservation_releases");
  auto changed = contractFixture();
  (*changed.getObject("dma_timing")->getObject("startup_cycles"))["load"] = 8;
  sealContractFixture(changed);
  rejects(std::move(changed), "rebound startup cannot disagree with its source contract", "launch/staging/issue");
  llvm::Expected<ModelTimingFacts> rebound = parse(contractFixture());
  if (!rebound) llvm::consumeError(rebound.takeError());
  changed = contractFixture();
  (*changed.getObject("frontend_timing")->getObject("scalar"))["write_age"] = 3;
  sealContractFixture(changed);
  rebound = parse(std::move(changed));
  check(bool(rebound), "supported source-derived frontend operands load");
  if (rebound) {
    auto provider = modelTimingProvider(std::make_shared<ModelTimingFacts>(std::move(*rebound)));
    auto scalar = provider.footprint({findOp("addi"), 1, 2, 0}, unknownRegs());
    check(provider.issueGap({findOp("addi")}).value == 1 &&
          provider.issueGap({findOp("delay"), 0, 0, 0, 511}).value == 512,
          "ordinary issue gap and DELAY mask consume supported derived frontend inputs");
    check(scalar.error.empty() && scalar.doneAge == 3 && scalar.accesses.back().age == 3,
          "synchronous register write age comes from derived frontend inputs");
  } else llvm::consumeError(rebound.takeError());
  for (const char *field : {"ordinary_issue_gap", "base_gap", "immediate_mask"}) {
    changed = contractFixture();
    Object *record = changed.getObject("frontend_timing");
    if (llvm::StringRef(field) != "ordinary_issue_gap") record = record->getObject("delay");
    (*record)[field] = llvm::StringRef(field) == "immediate_mask" ? 255 : 2;
    sealContractFixture(changed);
    rejects(std::move(changed), "source frontend change outside compiler encoding capability refuses precisely",
            "compiler scheduling/encoding capability");
  }
  changed = contractFixture();
  (*changed.getObject("scratchpad_contract")->getObject("assumptions"))["scratchpad_read_latency"] = 2;
  sealContractFixture(changed);
  rejects(std::move(changed), "changed target response assumption rejects with a precise capability diagnostic", "scratchpad target assumption");
  changed = contractFixture();
  changed.erase("mxu_weight_timing");
  sealContractFixture(changed);
  rejects(std::move(changed), "production cannot fall back to a hand-chosen weight lifetime", "weight-use windows");
  changed = contractFixture();
  auto *engine = changed.getObject("mxu_weight_timing")->getObject("engines")->getObject("mxu0");
  (*(*engine->getArray("use_rows"))[31].getAsObject())["last_age"] = 93;
  rejects(changed, "changed weight row is rejected before accepting stale validation", "derived semantic payload differs");
  sealContractFixture(changed);
  rejects(std::move(changed), "row projection must agree with source-derived PE windows", "row projection");
  changed = contractFixture();
  engine = changed.getObject("mxu_weight_timing")->getObject("engines")->getObject("mxu0");
  (*engine)["reads_before_writes"] = false;
  sealContractFixture(changed);
  rejects(std::move(changed), "unsupported weight edge order refuses", "edge order");
  changed = contractFixture();
  changed.getObject("validation")->erase("mxu_weight_validation");
  rejects(std::move(changed), "missing weight validation refuses", "MXU weight validation");
  changed = contractFixture();
  (*changed.getObject("validation")->getObject("mxu_weight_validation"))["passed"] = false;
  rejects(std::move(changed), "failed weight validation refuses", "MXU weight validation");
}

Object derivedFixture(const ModelTimingFacts &facts) {
  Object contract{{"capacity", Object{{"sequencer_cycles", 32}, {"mxu0_compute", 3},
                                      {"mxu1_compute", 2}, {"vpu_slots", 2}}}};
  auto perOp = fixture();
  return Object{{"schema", "atlas.model_overlap_derived.v1"}, {"age_convention", "issue_edge_zero"},
      {"per_op_sha256", facts.factsSha256()}, {"model", std::move(*perOp.get("model"))},
      {"derivation", Object{{"contract", "source-resource-occupancy-v1"}, {"tool_sha256", std::string(64, 'a')},
          {"source_contract_sha256", hash(canonicalJson(llvm::json::Value(Object(contract))))},
          {"source_contract", std::move(contract)}}},
      {"validation", Object{{"passed", true}, {"report_sha256", std::string(64, 'b')},
          {"sample_policy", "source-mechanism-boundaries-v2"}, {"positive_runs", 2}, {"negative_runs", 3},
          {"negative_rejections", 1}, {"numerical_checks", 2}, {"engines", Object{{"vpu", Object{
              {"positive_runs", 2}, {"negative_runs", 3}, {"negative_rejections", 1}, {"numerical_checks", 2},
              {"chain_positive_runs", 1}, {"chain_negative_runs", 1}}}}}}},
      {"pairs", Array{Object{{"first", "vpu.mov"}, {"second", "vpu.mov"},
          {"relation", Object{{"mreg_banks", "disjoint"}}}, {"minimum", 1}, {"high", 73}}}},
      {"composition", Object{{"engines", Object{{"vpu", Array{"vpu.mov"}}}},
          {"classes", Array{Object{{"engine", "vpu"}, {"relation", Object{{"mreg_banks", "disjoint"}}}}}}}}};
}

void testDerivedOverlap(const std::shared_ptr<ModelTimingFacts> &facts) {
  auto original = derivedFixture(*facts);
  std::string bytes = encode(original);
  auto loaded = parseModelOverlapFacts(bytes, hash(bytes), *facts);
  check(bool(loaded), "source-derived evidence with bounded native gate loads");
  if (!loaded) { llvm::errs() << llvm::toString(loaded.takeError()) << '\n'; return; }
  check(loaded->metadata().count("derived_sha256") && !loaded->metadata().count("composed_sha256"),
        "source-derived evidence preserves truthful distinct provenance");
  std::vector<ModelOverlapCommand> commands{
      {"vpu.mov", Engine::Vpu, {0, 1, 4, 5}, {}, {}, 0, 73},
      {"vpu.mov", Engine::Vpu, {8, 9, 12, 13}, {}, {}, 40, 73},
      {"vpu.mov", Engine::Vpu, {16, 17, 20, 21}, {}, {}, 73, 73}};
  check(loaded->admits({commands[0], commands[1]}, false), "source pair uses exact relation and bounded gap");
  auto second = commands[1]; second.issue = 0;
  check(!loaded->admits({commands[0], second}, false), "source pair refuses zero gap");
  second.issue = 74;
  check(!loaded->admits({commands[0], second}, false), "source pair cannot exceed selected live domain");
  second = commands[1]; second.registers[0] = 32;
  check(!loaded->admits({commands[0], second}, false), "source disjoint pair cannot waive a bank alias");
  check(loaded->admits(commands, true), "source composition uses supported complete pair domain");
  check(!loaded->admits(commands, false), "source composition retains explicit chains option");
  commands[1].issue = 1; commands[2].issue = 71;
  check(!loaded->admits(commands, true), "source composition preserves VPU structural tracker capacity");
  for (int slots : {1, 3}) {
    // Private synthetic artifacts isolate consumption of the selected source
    // parameter; production rules also bind the slot-dependent pair minima.
    auto changed = original;
    auto *derivation = changed.getObject("derivation");
    auto *contract = derivation->getObject("source_contract");
    (*contract->getObject("capacity"))["vpu_slots"] = slots;
    (*derivation)["source_contract_sha256"] = hash(canonicalJson(llvm::json::Value(Object(*contract))));
    auto capacityBytes = encode(std::move(changed));
    auto selected = parseModelOverlapFacts(capacityBytes, hash(capacityBytes), *facts);
    check(bool(selected), "supported source VPU slot parameter loads");
    if (selected)
      check(selected->admits(commands, true) == (slots == 3),
            "source VPU slot parameter drives live composition admission");
    else llvm::consumeError(selected.takeError());
  }
  commands[2].issue = 72;
  check(loaded->admits(commands, true), "source composition preserves completed-edge slot release");
  commands[2].registers[0] = 32;
  check(!loaded->admits(commands, true), "source composition checks every exact live pair");
  auto provider = modelTimingProvider(facts, std::make_shared<ModelOverlapFacts>(*loaded), true);
  Instr a{findOp("vmov"), 4, 0, 0}, b{findOp("vmov"), 12, 8, 0};
  auto state = provider.createReservations();
  auto fa = facts->resolve(a, unknownRegs()), fb = facts->resolve(b, unknownRegs());
  check(state.value->reserve(a, fa, 0).empty() && state.value->conflict(b, fb, 10).value.empty(),
        "source proof enables the existing reservation waiver");
  check(state.value->onWait({findOp("dma.wait.ch0")}, 1).empty() &&
        !state.value->conflict(b, fb, 10).value.empty(), "source proof retains runtime wait invalidation");
  Instr duplicateRead{findOp("vadd.bf16"), 4, 0, 0};
  auto duplicateFootprint = facts->resolve(duplicateRead, unknownRegs());
  check(duplicateFootprint.error.empty(), "VPU identical source reads are legal");
  state = provider.createReservations();
  check(state.value->conflict(duplicateRead, duplicateFootprint, 0).error.empty(),
        "VPU identical source rows use the source read cache");
  Instr bankAliasRead{findOp("vadd.bf16"), 4, 0, 32};
  check(!facts->resolve(bankAliasRead, unknownRegs()).error.empty(),
        "VPU distinct logical sources sharing a physical bank remain illegal");

  auto crossOperations = fixture();
  crossOperations.getArray("op_timing")->push_back(block("xlu.transpose", "xlu", 66,
      Object{{"read", group(1, 32)}, {"write", group(34, 32)}}));
  auto crossBytes = encode(std::move(crossOperations));
  auto crossFacts = parseModelTimingFactsForTesting(crossBytes, hash(crossBytes));
  check(bool(crossFacts), "cross-unit operation fixture loads");
  if (crossFacts) {
    auto cross = derivedFixture(*crossFacts);
    auto *validation = cross.getObject("validation");
    (*validation->getObject("engines"))["vpu_xlu"] = Object{
        {"positive_runs", 2}, {"negative_runs", 2}, {"negative_rejections", 1},
        {"numerical_checks", 2}, {"chain_positive_runs", 0}, {"chain_negative_runs", 0}};
    (*validation)["positive_runs"] = 4;
    (*validation)["negative_runs"] = 5;
    (*validation)["negative_rejections"] = 2;
    (*validation)["numerical_checks"] = 4;
    cross.getArray("pairs")->push_back(Object{{"first", "vpu.mov"}, {"second", "xlu.transpose"},
        {"relation", Object{{"mreg_banks", "disjoint"}}}, {"minimum", 1}, {"high", 73}});
    cross.getArray("pairs")->push_back(Object{{"first", "xlu.transpose"}, {"second", "vpu.mov"},
        {"relation", Object{{"mreg_banks", "disjoint"}}}, {"minimum", 1}, {"high", 66}});
    auto bytes = encode(cross);
    auto overlap = parseModelOverlapFacts(bytes, hash(bytes), *crossFacts);
    check(bool(overlap), "validated VPU/XLU pairs load");
    if (overlap) {
      auto provider = modelTimingProvider(std::make_shared<ModelTimingFacts>(*crossFacts),
                                         std::make_shared<ModelOverlapFacts>(*overlap), true);
      Instr transpose{findOp("vtrpose.xlu"), 24, 22, 0};
      auto ft = crossFacts->resolve(transpose, unknownRegs());
      for (bool reverse : {false, true}) {
        auto reservations = provider.createReservations();
        auto first = reverse ? transpose : a, second = reverse ? a : transpose;
        auto left = reverse ? ft : fa, right = reverse ? fa : ft;
        check(reservations.value->reserve(first, left, 0).empty() &&
              reservations.value->conflict(second, right, 1).value.empty(),
              "disjoint cross-unit pair waives only the isolated engine window");
      }
      ModelOverlapCommand xlu{"xlu.transpose", Engine::Xlu, {22, 24}, {}, {}, 1, 66};
      check(!overlap->admits({commands[0], xlu, commands[1]}, true),
            "mixed-unit chains retain conservative admission");
    } else llvm::consumeError(overlap.takeError());
    validation->getObject("engines")->erase("vpu_xlu");
    (*validation)["positive_runs"] = 2;
    (*validation)["negative_runs"] = 3;
    (*validation)["negative_rejections"] = 1;
    (*validation)["numerical_checks"] = 2;
    bytes = encode(cross);
    auto unvalidated = parseModelOverlapFacts(bytes, hash(bytes), *crossFacts);
    check(!unvalidated, "same-unit validation cannot qualify a cross-unit pair");
    if (!unvalidated) llvm::consumeError(unvalidated.takeError());
  } else llvm::consumeError(crossFacts.takeError());
  auto reject = [&](Object object, const char *why) {
    std::string contents = encode(std::move(object));
    auto result = parseModelOverlapFacts(contents, hash(contents), *facts);
    check(!result, why);
    if (!result) llvm::consumeError(result.takeError());
  };
  invalidFacts(original, {
      {"source artifact hash cannot be bypassed", "schema", "unknown"},
      {"source artifact must bind exact operation facts", "per_op_sha256", std::string(64, '0')},
      {"source model identity cannot change", "model/source_sha256", std::string(64, '0')},
      {"source numerical identity cannot change", "model/numerical_data_sha256", std::string(64, '0')},
      {"source explicit configuration cannot change", "model/configuration/fetch_width", 2},
      {"unknown source claim rejects", "verilator", Object{{"confirmed", true}}},
      {"source contract version must be reviewed", "derivation/contract", "unreviewed"},
      {"source contract contents must match their digest", "derivation/source_contract_sha256", std::string(64, '0')},
      {"source tool digest is required", "derivation/tool_sha256", "bad"},
      {"unvalidated candidate cannot be selected", "validation/passed", false},
      {"source validation report hash must be well formed", "validation/report_sha256", "bad"},
      {"source sample policy must be recognized", "validation/sample_policy", "unreviewed"},
      {"source validation numerical checks are mandatory", "validation/engines/vpu/numerical_checks", 0},
      {"source native rejections are mandatory", "validation/engines/vpu/negative_rejections", 0},
      {"source root totals must aggregate the engine probes", "validation/positive_runs", 3},
      {"source composition needs native chain positives", "validation/engines/vpu/chain_positive_runs", 0},
      {"source composition needs native chain rejections", "validation/engines/vpu/chain_negative_runs", 0},
      {"source chain rejects cannot exceed native rejects", "validation/engines/vpu/chain_negative_runs", 2},
      {"source pair lower bound must be positive", "pairs/0/minimum", 0},
      {"source pair lower bound cannot exceed upper bound", "pairs/0/minimum", 74},
      {"source pair high must equal live boundary", "pairs/0/high", 74},
      {"source pair relation must be complete", "pairs/0/relation/mreg_banks", nullptr, true},
      {"source pair unknown relation rejects", "pairs/0/relation/unknown", "other"},
      {"source pair unknown fields reject", "pairs/0/unknown", true},
      {"source composition cannot expand operation domain", "composition/engines/vpu", Array{"vpu.add"}},
      {"source composition cannot expand relation classes", "composition/classes/0/relation/mreg_banks", "bank_alias"},
      {"source scalar engine cannot claim general composition", "composition/engines/scalar_lsu", Array{"scalar_lsu.load"}}
  }, reject);
  for (const char *key : {"sequencer_cycles", "mxu0_compute", "mxu1_compute", "vpu_slots"}) {
    auto changed = original;
    auto *derivation = changed.getObject("derivation");
    auto *contract = derivation->getObject("source_contract");
    (*contract->getObject("capacity"))[key] = 99;
    (*derivation)["source_contract_sha256"] = hash(canonicalJson(llvm::json::Value(Object(*contract))));
    reject(std::move(changed), "rehashed structural capacity outside supported source bounds rejects");
  }
  auto duplicate = original;
  duplicate.getArray("pairs")->push_back((*duplicate.getArray("pairs"))[0]);
  reject(std::move(duplicate), "duplicate exact source pair rejects");
  duplicate = original;
  auto *classes = duplicate.getObject("composition")->getArray("classes");
  classes->push_back((*classes)[0]);
  reject(std::move(duplicate), "duplicate source composition class rejects");
  auto wrongHash = parseModelOverlapFacts(bytes, std::string(64, '0'), *facts);
  check(!wrongHash, "source artifact must match selected file digest");
  if (!wrongHash) llvm::consumeError(wrongHash.takeError());

  auto scalarSource = fixture();
  scalarSource.getArray("op_timing")->push_back(block("scalar_lsu.load", "scalar_lsu", 4,
      Object{{"vmem_read", group(0, 1)}, {"load_write", group(3, 1)}}));
  scalarSource.getArray("op_timing")->push_back(block("scalar_lsu.store", "scalar_lsu", 2,
      Object{{"vmem_write", group(0, 1)}}));
  std::string scalarBytes = encode(std::move(scalarSource));
  auto scalarFacts = parseModelTimingFactsForTesting(scalarBytes, hash(scalarBytes));
  check(bool(scalarFacts), "source scalar operation fixture loads");
  if (!scalarFacts) { llvm::consumeError(scalarFacts.takeError()); return; }
  auto scalar = derivedFixture(*scalarFacts);
  auto *scalarValidation = scalar.getObject("validation");
  Object scalarRuns(*scalarValidation->getObject("engines")->getObject("vpu"));
  scalarRuns.erase("chain_positive_runs"); scalarRuns.erase("chain_negative_runs");
  (*scalarValidation)["engines"] = Object{{"scalar_lsu", std::move(scalarRuns)}};
  scalar["composition"] = Object{{"engines", Object{}}, {"classes", Array{}}};
  scalar["pairs"] = Array{Object{{"first", "scalar_lsu.load"}, {"second", "scalar_lsu.store"},
      {"minimum", 1}, {"high", 4}, {"relation", Object{{"mreg_banks", "disjoint"},
          {"first_mnemonic", "lw"}, {"second_mnemonic", "sw"}, {"scalar_registers", "independent"},
          {"scalar_address_alignment", "word"}, {"vmem_bank", "same"}, {"vmem_window", "disjoint"}}}}};
  scalarBytes = encode(scalar);
  auto scalarOverlap = parseModelOverlapFacts(scalarBytes, hash(scalarBytes), *scalarFacts);
  check(bool(scalarOverlap), "source scalar pairs load with complete immutable-context predicates");
  if (!scalarOverlap) { llvm::consumeError(scalarOverlap.takeError()); return; }
  check(!scalarOverlap->admits({{"scalar_lsu.load", Engine::Scalar, {}, {}, {}, 0, 4},
                                {"scalar_lsu.store", Engine::Scalar, {}, {}, {}, 1, 2}}, false),
        "source scalar evidence cannot admit absent replay contexts");
  (*scalar.getArray("pairs"))[0].getAsObject()->getObject("relation")->erase("scalar_registers");
  scalarBytes = encode(std::move(scalar));
  auto incompleteScalar = parseModelOverlapFacts(scalarBytes, hash(scalarBytes), *scalarFacts);
  check(!incompleteScalar, "source scalar relation must retain all context predicates");
  if (!incompleteScalar) llvm::consumeError(incompleteScalar.takeError());

  auto mxuSource = fixture();
  mxuSource.getArray("op_timing")->push_back(block("mxu0.push_acc_fp8", "mxu0", 33,
      Object{{"mreg_read1", group(0, 32)}, {"acc_load", group(1, 32)}, {"data_busy", group(0, 32)}}));
  std::string mxuFactsBytes = encode(std::move(mxuSource));
  auto mxuParsed = parseModelTimingFactsForTesting(mxuFactsBytes, hash(mxuFactsBytes));
  if (!mxuParsed) { llvm::consumeError(mxuParsed.takeError()); check(false, "derived MXU fixture loads"); return; }
  auto mxuFacts = std::make_shared<ModelTimingFacts>(std::move(*mxuParsed));
  auto object = derivedFixture(*mxuFacts);
  auto *validation = object.getObject("validation");
  Object runs(*validation->getObject("engines")->getObject("vpu"));
  (*validation)["engines"] = Object{{"mxu0", std::move(runs)}};
  object["pairs"] = Array{};
  auto addPair = [&](const char *first, const char *second, Object rel) {
    object.getArray("pairs")->push_back(Object{{"first", first}, {"second", second},
        {"relation", std::move(rel)}, {"minimum", 1}, {"high", mxuFacts->block(first)->firstFreeAge}});
  };
  addPair("mxu0.push_acc_fp8", "mxu0.push_weight", Object{{"mreg_banks", "disjoint"}});
  addPair("mxu0.push_acc_fp8", "mxu0.matmul", Object{{"mreg_banks", "disjoint"}, {"accumulator", "other"}});
  addPair("mxu0.push_weight", "mxu0.matmul", Object{{"mreg_banks", "disjoint"}, {"weight_slot", "other"}});
  object["composition"] = Object{{"engines", Object{{"mxu0", Array{
      "mxu0.push_acc_fp8", "mxu0.push_weight", "mxu0.matmul"}}}}, {"classes", Array{
          Object{{"engine", "mxu0"}, {"relation", Object{{"mreg_banks", "disjoint"}}}},
          Object{{"engine", "mxu0"}, {"relation", Object{{"mreg_banks", "disjoint"}, {"accumulator", "other"}}}},
          Object{{"engine", "mxu0"}, {"relation", Object{{"mreg_banks", "disjoint"}, {"weight_slot", "other"}}}}}}};
  bytes = encode(object);
  auto mxu = parseModelOverlapFacts(bytes, hash(bytes), *mxuFacts);
  check(bool(mxu), "source-derived MXU composition fixture loads");
  if (!mxu) { llvm::errs() << llvm::toString(mxu.takeError()) << '\n'; return; }
  std::vector<ModelOverlapCommand> computes{
      {"mxu0.matmul", Engine::Mxu0, {0}, 0, 0, 0, 95},
      {"mxu0.matmul", Engine::Mxu0, {4}, 1, 0, 32, 95},
      {"mxu0.matmul", Engine::Mxu0, {8}, 0, 0, 64, 95}};
  check(!mxu->rejectsPortHistory(computes), "three source compute trackers preserve the default legal capacity");
  for (int depth : {1, 2, 4}) {
    auto changed = object;
    auto *derivation = changed.getObject("derivation");
    auto *contract = derivation->getObject("source_contract");
    (*contract->getObject("capacity"))["mxu0_compute"] = depth;
    (*derivation)["source_contract_sha256"] = hash(canonicalJson(llvm::json::Value(Object(*contract))));
    auto capacityBytes = encode(std::move(changed));
    auto selected = parseModelOverlapFacts(capacityBytes, hash(capacityBytes), *mxuFacts);
    check(bool(selected), "supported source compute tracker parameter loads");
    if (selected)
      check(selected->rejectsPortHistory(computes) == (depth < 3),
            "selected source compute tracker depth drives retained issue history");
    else llvm::consumeError(selected.takeError());
  }
  auto mxuProvider = modelTimingProvider(mxuFacts, std::make_shared<ModelOverlapFacts>(*mxu), true);
  Instr accumulator{findOp("vmatpush.acc.fp8.mxu0"), 0, 4, 0};
  Instr weight{findOp("vmatpush.weight.mxu0"), 0, 0, 0};
  Instr compute{findOp("vmatmul.mxu0"), 1, 8, 1};
  state = mxuProvider.createReservations();
  check(state.value->reserve(accumulator, mxuFacts->resolve(accumulator, unknownRegs()), 0).empty() &&
        state.value->reserve(weight, mxuFacts->resolve(weight, unknownRegs()), 2).empty(),
        "source retired-prefix regression starts with an admitted pair");
  check(!state.value->conflict(compute, mxuFacts->resolve(compute, unknownRegs()), 33).value.empty(),
        "source-derived provenance activates retired-command MXU port-history check");
  check(state.value->conflict(compute, mxuFacts->resolve(compute, unknownRegs()), 34).value.empty(),
        "source-derived retired-prefix check releases at the existing port boundary");
}

void testOverlapSafety(const std::shared_ptr<ModelTimingFacts> &facts) {
  std::string bytes = encode(derivedFixture(*facts));
  auto parsed = parseModelOverlapFacts(bytes, hash(bytes), *facts);
  check(bool(parsed), "source overlap safety fixture loads");
  if (!parsed) { llvm::consumeError(parsed.takeError()); return; }
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
  auto overlap = std::make_shared<ModelOverlapFacts>(*parsed);
  auto pairs = modelTimingProvider(facts, overlap);
  auto chains = modelTimingProvider(facts, overlap, true);
  Instr a{findOp("vmov"), 4, 0, 0}, b{findOp("vmov"), 12, 8, 0}, c{findOp("vmov"), 20, 16, 0};
  auto fa = facts->resolve(a, unknownRegs()), fb = facts->resolve(b, unknownRegs()), fc = facts->resolve(c, unknownRegs());
  auto state = pairs.createReservations();
  check(state.value->reserve(a, fa, 0).empty() && state.value->reserve(b, fb, 40).empty() &&
        state.value->conflict(c, fc, 80).value.empty(), "retired historical command needs only the remaining live pair");
  state = chains.createReservations();
  check(state.value->reserve(a, fa, 0).empty() && state.value->reserve(c, fc, 80).empty() &&
        state.value->conflict(b, fb, 40).value.empty(), "insertion rechecks every live pair and capacity");
  state = chains.createReservations();
  check(state.value->reserve(a, fa, 0).empty() && state.value->reserve(c, fc, 40).empty() &&
        !state.value->conflict(b, fb, 10).value.empty(), "insertion cannot exceed tracker capacity");
  auto load = [&](Object object) -> std::shared_ptr<ModelOverlapFacts> {
    object["composition"] = Object{{"engines", Object{}}, {"classes", Array{}}};
    std::string modified = encode(std::move(object));
    auto value = parseModelOverlapFacts(modified, hash(modified), *facts);
    check(bool(value), "source safety mutation loads");
    if (!value) { llvm::consumeError(value.takeError()); return {}; }
    return std::make_shared<ModelOverlapFacts>(std::move(*value));
  };
  auto aliased = derivedFixture(*facts);
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
  auto dependent = derivedFixture(*facts);
  (*dependent.getArray("pairs"))[0].getAsObject()->operator[]("relation") =
      Object{{"mreg_banks", "2=0,3=1"}};
  auto dataFacts = load(std::move(dependent));
  if (dataFacts) {
    auto dataProvider = modelTimingProvider(facts, dataFacts);
    Instr use{findOp("vmov"), 12, 4, 0};
    check(dataProvider.dependence(a, fa, use, facts->resolve(use, unknownRegs())).value.distance == 73,
          "confirmed pair admission does not weaken observed data-release dependencies");
  }
  auto catalog = derivedFixture(*facts); catalog["schema"] = "atlas.model_pairs.v1";
  auto rejected = encode(std::move(catalog));
  auto unsupported = parseModelOverlapFacts(rejected, hash(rejected), *facts);
  check(!unsupported, "historical catalog schema cannot be selected");
  if (!unsupported) llvm::consumeError(unsupported.takeError());
}

void testStreamedWeights(const std::shared_ptr<ModelTimingFacts> &facts) {
  auto provider = modelTimingProvider(facts);
  Instr push{findOp("vmatpush.weight.mxu0"), 0, 0, 0};
  Instr compute{findOp("vmatmul.mxu0"), 0, 2, 0};
  auto fp = facts->resolve(push, unknownRegs()), fc = facts->resolve(compute, unknownRegs());
  check(fp.error.empty() && fc.error.empty() &&
            provider.dependence(push, fp, compute, fc).value.distance == 3 &&
            provider.dependence(compute, fc, push, fp).value.distance == 61,
        "derived windows order streamed weights without overlap evidence, including shifted write ages");
  auto state = provider.createReservations();
  check(state.value->reserve(push, fp, 0).empty() && !state.value->conflict(compute, fc, 3).value.empty(),
        "weight data dependencies do not waive isolated engine occupancy or overlap admission");
  auto rules = derivedFixture(*facts);
  auto *validation = rules.getObject("validation");
  Object runs(*validation->getObject("engines")->getObject("vpu"));
  (*validation)["engines"] = Object{{"mxu0", std::move(runs)}};
  rules["composition"] = Object{{"engines", Object{}}, {"classes", Array{}}};
  // Deliberately admit earlier than the shifted synthetic writes permit. A
  // source overlap rule cannot replace either derived resource dependency.
  rules["pairs"] = Array{Object{
      {"first", "mxu0.push_weight"}, {"second", "mxu0.matmul"},
      {"relation", Object{{"mreg_banks", "disjoint"}, {"weight_slot", "same"}}},
      {"minimum", 1}, {"high", facts->block("mxu0.push_weight")->firstFreeAge}}};
  std::string ruleBytes = encode(std::move(rules));
  auto overlap = parseModelOverlapFacts(ruleBytes, hash(ruleBytes), *facts);
  check(bool(overlap), "permissive synthetic source-derived overlap rule loads");
  if (overlap) {
    auto qualified = modelTimingProvider(facts, std::make_shared<ModelOverlapFacts>(std::move(*overlap)));
    check(qualified.dependence(push, fp, compute, fc).value.distance == 3 &&
              qualified.dependence(compute, fc, push, fp).value.distance == 61,
          "source-derived overlap rules cannot substitute for derived weight dependencies");
    state = qualified.createReservations();
    check(state.value->reserve(push, fp, 0).empty() && state.value->conflict(compute, fc, 3).value.empty(),
          "source-derived overlap independently admits a resource-safe derived weight boundary");
    compute.rs1 = 0;
    check(qualified.dependence(push, fp, compute, facts->resolve(compute, unknownRegs())).value.distance == 3 &&
              !state.value->conflict(compute, facts->resolve(compute, unknownRegs()), 3).value.empty(),
          "precise weight dependency cannot waive an uncovered MREG alias relation");
  } else llvm::consumeError(overlap.takeError());

  // Use model-shaped events to check both engines and slots independently of
  // the deliberately shifted synthetic operation fixture above.
  auto object = fixture();
  Array operations;
  for (int unit = 0; unit < 2; ++unit) {
    std::string engine = "mxu" + std::to_string(unit);
    int begin = unit == 0 ? 63 : 3;
    operations.push_back(block(engine + ".push_weight", engine, 33,
        Object{{"mreg_read1", group(0, 32)}, {"weight_write", group(1, 32)}, {"data_busy", group(0, 33)}}));
    for (const char *suffix : {"matmul", "matmul_acc"}) {
      Object events{{"mreg_read0", group(0, 32)}, {"acc_write", group(begin, 32)},
                    {"compute_busy", group(0, begin + 32)}};
      if (llvm::StringRef(suffix) == "matmul_acc") events["acc_read"] = group(0, 32);
      operations.push_back(block(engine + "." + suffix, engine, begin + 32, std::move(events)));
    }
  }
  object["op_timing"] = std::move(operations);
  std::string bytes = encode(std::move(object));
  auto parsed = parseModelTimingFactsForTesting(bytes, hash(bytes));
  check(bool(parsed), "model-shaped weight fixture loads");
  if (!parsed) { llvm::errs() << llvm::toString(parsed.takeError()) << '\n'; return; }
  auto model = std::make_shared<ModelTimingFacts>(std::move(*parsed));
  provider = modelTimingProvider(model);
  for (int unit = 0; unit < 2; ++unit)
    for (int slot = 0; slot < 2; ++slot)
      for (const char *mode : {"vmatmul", "vmatmul.acc"}) {
        std::string engine = ".mxu" + std::to_string(unit);
        push = {findOp("vmatpush.weight" + engine), slot, 0, 0};
        compute = {findOp(std::string(mode) + engine), 0, 2, slot};
        fp = model->resolve(push, unknownRegs());
        fc = model->resolve(compute, unknownRegs());
        check(fp.error.empty() && fc.error.empty(), "both matmul modes resolve derived weight windows");
        check(provider.dependence(push, fp, compute, fc).value.distance == (unit == 0 ? 1 : 32),
              "source-derived weight RAW boundary");
        check(provider.dependence(compute, fc, push, fp).value.distance == (unit == 0 ? 63 : 32),
              "source-derived weight WAR boundary");
        for (int row : {0, 31}) {
          std::vector<int> ages;
          int element = (unit * 2 + slot) * 32 + row;
          for (const Access &access : fc.accesses)
            if (access.res == Res::Weight && access.first <= element && element < access.first + access.count)
              ages.push_back(access.age + (element - access.first) * access.step);
          check(ages == std::vector<int>({unit == 0 ? 1 + row : 1, unit == 0 ? 63 + row : 32}),
                "first and last row uses retain the engine and slot mapping");
        }
        push.rd = 1 - slot;
        check(provider.dependence(compute, fc, push, model->resolve(push, unknownRegs())).value.distance == 0,
              "different weight slots have no data dependency");
      }
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
  auto parsed = parseModelTimingFactsForTesting(bytes, hash(bytes));
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
  parsed = parseModelTimingFactsForTesting(bytes, hash(bytes));
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
  auto changed = parseModelTimingFactsForTesting(bytes, hash(bytes));
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
  rejectConfiguration("vmem_bank_bytes", 131072, "rehashed VMEM configuration must agree with derived architecture");
  rejectConfiguration("dma_num_channels", 4, "rehashed DMA configuration must agree with derived architecture");
  rejectConfiguration("dma_memory_backend", "fifo", "isolated facts do not qualify a changed DMA estimate profile");
  rejectConfiguration("ignore_runtime_errors", true, "even a correctly hashed configuration cannot suppress model runtime errors");
}

void testScalarCosts() {
  auto sample = fixture(); // Marked raw observations stay inside the test API.
  sample["op_timing"] = Array{
      block("scalar_lsu.load", "scalar_lsu", 4, Object{{"vmem_read", group(0, 1)}, {"load_write", group(3, 1)}}),
      block("scalar_lsu.store", "scalar_lsu", 2, Object{{"vmem_write", group(0, 1)}}),
      block("scalar_lsu.scale_load", "scalar_lsu", 4, Object{{"vmem_read", group(0, 1)}, {"scale_write", group(3, 1)}})};
  auto unit = [](int free) {
    return Object{{"first_age", 0}, {"release_age", free - 1}, {"available_age", free}};
  };
  auto port = []() {
    return Object{{"first_age", 0}, {"release_age", 0}, {"available_age", 1}, {"banks", Array{0}}};
  };
  sample["reservation_releases"] = Object{
      {"scalar_lsu.load", Object{{"unit", unit(4)}, {"vmem_read", port()}}},
      {"scalar_lsu.store", Object{{"unit", unit(2)}, {"vmem_write", port()}}},
      {"scalar_lsu.scale_load", Object{{"unit", unit(4)}, {"vmem_read", port()}}}};
  auto parse = [](Object object) -> std::optional<TimingProvider> {
    auto bytes = encode(std::move(object));
    auto facts = parseModelTimingFactsForTesting(bytes, hash(bytes));
    check(bool(facts), "synthetic scalar cost fixture loads through the explicit test API");
    if (!facts) { llvm::consumeError(facts.takeError()); return std::nullopt; }
    return modelTimingProvider(std::make_shared<ModelTimingFacts>(std::move(*facts)));
  };
  MLIRContext context;
  auto module = ModuleOp::create(UnknownLoc::get(&context));
  auto costIs = [&](const TimingProvider &provider, const char *mnemonic, int expected, const char *why) {
    auto cost = machineTimingOf(provider, mnemonic, module);
    check(succeeded(cost) && cost->latency == expected && cost->interval == expected, why);
  };
  if (auto provider = parse(Object(sample))) {
    costIs(*provider, "lw", 4, "scalar load cost consumes the selected release envelope");
    costIs(*provider, "sw", 2, "scalar store cost consumes the selected release envelope");
    costIs(*provider, "seld", 4, "scale load cost consumes the selected release envelope");
  }
  for (auto [name, mnemonic, free] : {std::tuple<const char *, const char *, int>{"scalar_lsu.load", "lw", 11},
                                      {"scalar_lsu.store", "sw", 7}, {"scalar_lsu.scale_load", "seld", 9}}) {
    Object changed(sample);
    for (auto &value : *changed.getArray("op_timing")) {
      Object &entry = *value.getAsObject();
      if (entry.getString("name") == name) entry["first_free_age"] = entry["next_issue_age"] = free;
    }
    (*changed.getObject("reservation_releases")->getObject(name))["unit"] = unit(free);
    if (auto provider = parse(std::move(changed)))
      costIs(*provider, mnemonic, free, "longer scalar unit releases change costs with unchanged event ages");
    for (bool missingBlock : {false, true}) {
      Object missing(sample);
      if (missingBlock) {
        Array retained;
        for (const auto &value : *missing.getArray("op_timing"))
          if (value.getAsObject()->getString("name") != name) retained.push_back(value);
        missing["op_timing"] = std::move(retained);
        missing.getObject("reservation_releases")->erase(name);
      } else {
        missing.getObject("reservation_releases")->getObject(name)->erase(
            llvm::StringRef(mnemonic) == "sw" ? "vmem_write" : "vmem_read");
      }
      if (auto provider = parse(std::move(missing))) {
        auto instruction = provider->costProbes(*findOp(mnemonic)).value.first;
        auto footprint = provider->footprint(instruction, unknownRegs());
        const std::string expected = std::string(name) + (missingBlock ? ": facts block is missing" :
            std::string(": missing reservation release ") + (llvm::StringRef(mnemonic) == "sw" ? "vmem_write" : "vmem_read"));
        check(footprint.error.find(expected) != std::string::npos,
              "missing scalar timing or release coverage refuses precisely in the test API");
        check(failed(machineTimingOf(*provider, mnemonic, module)),
              "missing scalar timing or release coverage has no cost fallback");
      }
    }
  }
  module.erase();
}

void testDMAEstimates() {
  auto parse = [](Object object) {
    std::string bytes = encode(std::move(object));
    return parseModelTimingFactsForTesting(bytes, hash(bytes), true);
  };
  auto baseline = parse(fixture());
  check(bool(baseline), "synthetic source-pinned DMA proposal facts load");
  if (!baseline) { llvm::consumeError(baseline.takeError()); return; }
  check(baseline->estimateDMACycles(32, false).value == 7 && baseline->estimateDMACycles(1024, false).value == 193 &&
        baseline->estimateDMACycles(32, true).value == 9 && baseline->estimateDMACycles(1024, true).value == 195,
        "all-mixture bounds consume startup and source-derived service quantum");
  check(!baseline->estimateDMACycles(0, false).error.empty() && !baseline->estimateDMACycles(33, true).error.empty() &&
        !baseline->estimateDMACycles(8192, false).error.empty(), "DMA proposals reject invalid transfer geometry");
  Object changed = fixture();
  changed.getObject("dma_timing")->getObject("startup_cycles")->operator[]("load") = 8;
  auto startup = parse(std::move(changed));
  check(!startup, "DMA proposal rejects startup that disagrees with its shared source contract");
  if (!startup) llvm::consumeError(startup.takeError());
  changed = fixture();
  Object &read = *(*changed.getObject("dma_timing")->getArray("curves"))[1].getAsObject();
  read["peak_beats_per_cycle"] = .1;
  read["cycles_per_beat"] = 10.;
  (*(*read.getArray("points"))[0].getAsArray())[0] = .05;
  (*(*read.getArray("points"))[1].getAsArray())[0] = .1;
  (*changed.getObject("dma_timing"))["proposal_contract"] = proposalFixture(10., 11);
  auto slower = parse(std::move(changed));
  check(bool(slower) && slower->estimateDMACycles(1024, false).value == 353,
        "DMA proposal responds to derived curve pacing, not builtin calibration");
  if (!slower) llvm::consumeError(slower.takeError());
  changed = fixture();
  Object &longRead = *(*changed.getObject("dma_timing")->getArray("curves"))[1].getAsObject();
  longRead["unloaded_latency_cycles"] = 1000.;
  (*(*longRead.getArray("points"))[0].getAsArray())[1] = 1000.;
  (*(*longRead.getArray("points"))[1].getAsArray())[1] = 2000.;
  (*changed.getObject("dma_timing"))["proposal_contract"] = proposalFixture(1000., 1001);
  auto bounded = parse(std::move(changed));
  check(bool(bounded) && bounded->estimateDMACycles(4096, false).value == 128129,
        "service envelope includes registered-capacity progress after responses");
  if (!bounded) llvm::consumeError(bounded.takeError());
  changed = fixture();
  (*changed.getObject("dma_timing"))["reset_last_ready_cycles"] = 1000.;
  auto reset = parse(std::move(changed));
  check(!reset, "bound refuses unsupported nonzero cold reset pacing debt");
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
      {"malformed DMA curve hash rejects", "dma_timing/curve_sha256", "invalid"}});
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
      {"out of range numeric estimate rejects", "dma_timing/rounding_epsilon_cycles", 2.},
      {"service quantum must agree with all mixtures", "dma_timing/proposal_contract/per_beat_bound_cycles", 5},
      {"unrecognized wait visibility rejects", "dma_timing/proposal_contract/retire_to_wait_issue_cycles", 2},
      {"missing proposal contract rejects", "dma_timing/proposal_contract", nullptr, true}});
  auto isolated = baseline->proposeDMA(193, 10, std::nullopt);
  auto queued = baseline->proposeDMA(193, 11, isolated.value.queueEnd);
  check(isolated.error.empty() && isolated.value.queueEnd == 203 && isolated.value.waitIssue == 204 &&
        queued.error.empty() && queued.value.queueEnd == 397 && queued.value.waitIssue == 398,
        "one shared proposal interpreter applies derived visibility and explicit serialized policy");
  check(!baseline->proposeDMA(193, INT_MAX, std::nullopt).error.empty(), "proposal arithmetic refuses overflow");
  auto old = builtin.dmaProposal(781, 10, std::nullopt);
  check(old.error.empty() && old.value.queueEnd == 790 && old.value.waitIssue == 792,
        "builtin retains old queue coordinate and wait-priority offsets");
}

int main(int argc, char **argv) {
  if (argc == 2 && llvm::StringRef(argv[1]) == "--fixture") {
    llvm::outs() << encode(fixture()) << '\n';
    return 0;
  }
  if (argc != 1) return 2;
  failures += testEngineAdmission();
  testDerivedContract();
  testScalarCosts();
  testDMAEstimates();
  testExplicitReleases();
  std::string bytes = encode(fixture());
  auto loaded = parseModelTimingFactsForTesting(bytes, hash(bytes), true);
  check(bool(loaded), "synthetic observed facts load");
  if (!loaded) {
    llvm::errs() << llvm::toString(loaded.takeError()) << '\n';
    return 1;
  }
  auto facts = std::make_shared<ModelTimingFacts>(std::move(*loaded));
  testOverlapSafety(facts);
  testDerivedOverlap(facts);
  testStreamedWeights(facts);
  testVloadWriteDuringRead(facts);
  failures += testScalarOverlap(fixture());
  auto provider = modelTimingProvider(facts);
  RegValues dmaRegs = unknownRegs();
  dmaRegs[1] = 0x20000 / 4; dmaRegs[2] = 0; dmaRegs[3] = 1024;
  Footprint dma = provider.footprint({findOp("dma.load.ch0"), 1, 2, 3}, dmaRegs);
  check(dma.error.empty() && dma.dmaCycles == 193 && dma.doneAge == 0,
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
  auto badHash = parseModelTimingFactsForTesting(bytes, std::string(64, '0'));
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
      {"architecture schema checked", "architecture/schema", "unrecognized"},
      {"architecture integer type checked", "architecture/vmem_bytes", "1572864"},
      {"architecture metadata required", "architecture", nullptr, true},
      {"derived vocabulary required", "operand_vocabulary", nullptr, true},
      {"derived vocabulary direction checked", "operand_vocabulary/vpu_operand_roles/VpuElementwise/0/write", true},
      {"ISA metadata required", "isa_metadata", nullptr, true},
      {"ISA operand type checked", "isa_metadata/vmov/compiler_operands", "md x1"},
      {"ISA engine checked", "isa_metadata/vmov/engine", "mxu0"},
      {"ISA memory directions checked", "isa_metadata/vstore/operands/0/write", true},
      {"ISA memory resource types checked", "isa_metadata/lw/operands/0/type", "MatrixReg"},
      {"ISA memory operand coverage checked", "isa_metadata/vload/operands", Array{}},
      {"ISA coverage checked", "isa_metadata/vmov", nullptr, true},
      {"unconfirmed overlap rejects", "op_timing/0/next_issue_age", 1},
      {"inconsistent event shape rejects", "op_timing/0/events/read0/count", 1},
      {"unknown fact operation rejects", "op_timing/0/name", "unknown.operation"}});
  for (const auto &[key, value] : architectureFixture()) {
    if (auto number = value.getAsInteger()) {
      Object changed = fixture();
      (*changed.getObject("architecture"))[key] = *number + 1;
      rejected(std::move(changed), "every derived architecture capability is checked");
      Object missing = fixture();
      missing.getObject("architecture")->erase(key);
      rejected(std::move(missing), "every derived architecture capability is required");
    }
  }
  llvm::outs() << "model timing synthetic checks, " << failures << " failures\n";
  return failures != 0;
}
