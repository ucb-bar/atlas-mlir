#include "Atlas/AtlasModelPairs.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SHA256.h"
#include <algorithm>
#include <climits>
#include <set>

using namespace mlir::atlas::timing;

namespace {
using Object = llvm::json::Object;
using Array = llvm::json::Array;
using Relation = std::map<std::string, std::string>;

llvm::Error invalid(const std::string &why) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), "model overlap facts: " + why);
}
bool hashSyntax(llvm::StringRef value) {
  return value.size() == 64 && std::all_of(value.begin(), value.end(), [](char c) {
    return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
  });
}
bool keys(const Object &object, std::initializer_list<llvm::StringRef> allowed) {
  for (const auto &[key, value] : object)
    if (std::find(allowed.begin(), allowed.end(), llvm::StringRef(key)) == allowed.end()) return false;
  return true;
}
bool relation(const Object *object, Relation &out) {
  if (!object || !keys(*object, {"mreg_banks", "accumulator", "weight_slot"})) return false;
  auto banks = object->getString("mreg_banks");
  if (!banks) return false;
  if (*banks != "disjoint") {
    llvm::SmallVector<llvm::StringRef> terms;
    banks->split(terms, ',');
    std::pair<int, int> previous{-1, -1};
    for (llvm::StringRef term : terms) {
      size_t separator = term.find_first_of("=~");
      int first, second;
      if (separator == llvm::StringRef::npos || term.take_front(separator).getAsInteger(10, first) ||
          term.drop_front(separator + 1).getAsInteger(10, second) || first < 0 || second < 0 ||
          std::pair<int, int>{first, second} <= previous) return false;
      previous = {first, second};
    }
  }
  for (const auto &[key, value] : *object) {
    auto text = value.getAsString();
    if (!text || (key != "mreg_banks" && *text != "same" && *text != "other")) return false;
    out[key.str()] = text->str();
  }
  return true;
}
Relation actualRelation(const ModelOverlapCommand &first, const ModelOverlapCommand &second) {
  Relation result;
  std::string banks;
  for (size_t i = 0; i < first.registers.size(); ++i)
    for (size_t j = 0; j < second.registers.size(); ++j)
      if (first.registers[i] % 32 == second.registers[j] % 32)
        banks += (banks.empty() ? "" : ",") + std::to_string(i) +
            (first.registers[i] == second.registers[j] ? "=" : "~") + std::to_string(j);
  result["mreg_banks"] = banks.empty() ? "disjoint" : banks;
  if (first.engine == second.engine) {
    if (first.accumulator && second.accumulator)
      result["accumulator"] = first.accumulator == second.accumulator ? "same" : "other";
    if (first.weightSlot && second.weightSlot)
      result["weight_slot"] = first.weightSlot == second.weightSlot ? "same" : "other";
  }
  return result;
}
int peak(const std::vector<ModelOverlapCommand> &commands) {
  int result = 0;
  for (const ModelOverlapCommand &edge : commands) {
    int live = 0;
    for (const ModelOverlapCommand &command : commands)
      live += command.issue <= edge.issue &&
          int64_t(edge.issue) < int64_t(command.issue) + command.firstFreeAge - 1;
    result = std::max(result, live);
  }
  return result;
}
bool sameGroup(llvm::StringRef first, llvm::StringRef second) {
  if (first.starts_with("scalar") || second.starts_with("scalar")) return false;
  auto vector = [](llvm::StringRef engine) { return engine == "vpu" || engine == "xlu"; };
  return first == second || (vector(first) && vector(second));
}
bool constraints(const Object &record) {
  for (const char *name : {"relation_semantics", "capacity_scope", "producer_consumer_requirements",
                           "domain_extension_scope"})
    if (record.get(name)) return false;
  static const std::map<std::string, std::set<std::string>> text = {
      {"resource_semantics", {"Accumulator and weight slot local to named MXU. Disjoint MREG footprint."}},
      {"consumer_requirement", {
          "Cross-engine namespace matching required. No implicit general capacity or source alias coverage."}},
      {"matching_requirement", {
          "Match the exact gaps and every all_pair_relations entry; top relation describes only the first adjacent pair."}},
      {"bound_reason", {"last gap reaches observed drain of both fixed-prefix commands plus one edge"}},
      {"sweep_axis", {"final adjacent gap; all earlier gaps fixed to fixed_prefix_gaps",
                       "final adjacent gap; three earlier gaps fixed"}},
      {"in_flight_scope", {
          "Persistent completed-tick exact sequence witness; five-path prefix capacity only, no universal capacity claim.",
          "actual persistent command count after each completed tick; witnessed count for this sequence, "
          "not a universal engine capacity",
          "persistent after completed tick at issue ages0,32,64,65,66 for this exact mixed operation/operand sequence; "
          "no universal capacity claim",
          "persistent after completed tick for this exact ordered alias chain, not a universal capacity limit",
          "persistent after completed tick for this exact three-command issue vector",
          "persistent commands after completed tick for this exact nonuniform issue vector",
          "persistent commands after completed tick for this exact sequence; no universal capacity claim"}}};
  for (const auto &[name, allowed] : text)
    if (record.get(name)) {
      auto value = record.getString(name);
      if (!value || !allowed.count(value->str())) return false;
    }
  if (record.get("producer_relation_summary")) {
    const Object *summary = record.getObject("producer_relation_summary");
    if (!summary || !keys(*summary, {"mreg_banks", "accumulator", "weight_slot"})) return false;
    for (const auto &[key, entry] : *summary) {
      auto value = entry.getAsString();
      if (!value || (key == "mreg_banks" ? *value != "disjoint" && *value != "mixed" && *value != "0~0" :
                                          *value != "same" && *value != "other" && *value != "mixed")) return false;
    }
  }
  if (record.get("operand_order")) {
    const Object *operands = record.getObject("operand_order");
    if (!operands || operands->size() != 2 || !keys(*operands, {"first", "second"})) return false;
    for (const auto &[key, entry] : *operands) {
      const Array *registers = entry.getAsArray();
      if (!registers) return false;
      for (const auto &reg : *registers)
        if (!reg.getAsInteger() || *reg.getAsInteger() < 0 || *reg.getAsInteger() > 63) return false;
    }
  }
  return !record.get("evidence_summary") || record.getObject("evidence_summary");
}
bool source(const Object &record, llvm::StringRef kind) {
  const Object *descriptor = record.getObject("source_record"), *proof = record.getObject("verilator");
  if (!descriptor || !keys(*descriptor, {"json_pointer", "sha256", "proof_sha256"}) || !proof ||
      proof->getBoolean("confirmed") != true) return false;
  auto pointer = descriptor->getString("json_pointer");
  if (!pointer || !pointer->consume_front(("/" + kind + "/").str())) return false;
  uint64_t index;
  if (pointer->empty() || pointer->getAsInteger(10, index)) return false;
  for (const char *key : {"sha256", "proof_sha256"}) {
    auto hash = descriptor->getString(key);
    if (!hash || !hashSyntax(*hash)) return false;
  }
  if (!keys(*proof, {"confirmed", "rtl_sha256", "source", "scope", "rtl_gap_points", "full_suffix_rtl_swept",
                    "negative_basis", "measured_gap", "negative_gap", "negative_kind", "negative_scope",
                    "classification", "run", "negative_run", "result", "first", "second", "operations", "gaps",
                    "negative", "source_mregs", "source_mregs_disjoint", "source_mregs_equality", "accumulator_slots",
                    "accumulator_slots_equality", "weight_slots", "weight_slots_equality", "same_weight_slot",
                    "maximum_live_commands", "maximum_rtl_tracker_occupancy", "all_contributions_preserved",
                    "current_model", "fragment", "fragment_sha256", "full_memory_checked_bytes", "attempted_dispatch_gap"}))
    return false;
  if (proof->get("full_suffix_rtl_swept") && proof->getBoolean("full_suffix_rtl_swept") != false) return false;
  for (const char *name : {"scope", "classification"})
    if (proof->get(name)) {
      auto value = proof->getString(name);
      if (!value || (llvm::StringRef(name) == "scope" ?
          *value != "Exact issued operation sequence and complete operand relations; sampled RTL boundaries only." :
          *value != "PASS" && *value != "PASS_STREAMING_WEIGHT_CHECK")) return false;
    }
  if (proof->get("rtl_sha256")) {
    auto hash = proof->getString("rtl_sha256");
    if (!hash || !hashSyntax(*hash)) return false;
  }
  if (proof->get("source")) {
    const Object *original = proof->getObject("source");
    if (!original || !original->getString("path") || !original->getString("sha256") ||
        !hashSyntax(*original->getString("sha256"))) return false;
  }
  return true;
}
bool proofGaps(const Object &record, const std::vector<int> &expected) {
  const Object *proof = record.getObject("verilator");
  if (proof->get("rtl_gap_points")) {
    const Array *gaps = proof->getArray("rtl_gap_points");
    if (!gaps || gaps->size() != expected.size()) return false;
    for (size_t i = 0; i < expected.size(); ++i)
      if ((*gaps)[i].getAsInteger() != expected[i]) return false;
  }
  if (expected.size() == 1) {
    if (proof->get("measured_gap") && proof->getInteger("measured_gap") != expected[0]) return false;
    if (proof->get("negative_gap") && proof->getInteger("negative_gap") != expected[0] - 1) return false;
  }
  return true;
}
} // namespace

bool ModelOverlapFacts::admits(const std::vector<ModelOverlapCommand> &commands, bool chains) const {
  auto admitsPair = [&](const ModelOverlapCommand &first, const ModelOverlapCommand &second) {
    int gap = second.issue - first.issue;
    Relation matched = actualRelation(first, second);
    for (const Pair &pair : pairs)
      if (pair.first == first.block && pair.second == second.block && pair.relation == matched &&
          pair.minimum <= gap && gap <= pair.high) return true;
    return false;
  };
  if (commands.size() == 2) return admitsPair(commands[0], commands[1]);
  if (!chains || commands.size() < 3) return false;
  for (size_t i = 0; i < commands.size(); ++i)
    for (size_t j = i + 1; j < commands.size(); ++j)
      if (commands[j].issue - commands[i].issue < commands[i].firstFreeAge &&
          !admitsPair(commands[i], commands[j])) return false;
  for (const Chain &chain : sequences) {
    if (chain.operations.size() != commands.size() || chain.inFlight != peak(commands)) continue;
    bool matched = true;
    for (size_t i = 0; i < commands.size(); ++i) {
      matched &= chain.operations[i] == commands[i].block;
      if (i) matched &= chain.gaps[i - 1] == commands[i].issue - commands[i - 1].issue;
      for (size_t j = 0; j < i; ++j)
        matched &= chain.relations.at({int(j), int(i)}) == actualRelation(commands[j], commands[i]);
    }
    if (matched) return true;
  }
  return false;
}

llvm::Expected<ModelOverlapFacts> mlir::atlas::timing::parseModelOverlapFacts(
    llvm::StringRef bytes, llvm::StringRef expectedSha256, const ModelTimingFacts &operations) {
  auto digest = llvm::SHA256::hash(llvm::ArrayRef<uint8_t>(
      reinterpret_cast<const uint8_t *>(bytes.data()), bytes.size()));
  if (!hashSyntax(expectedSha256) || llvm::toHex(llvm::ArrayRef<uint8_t>(digest), true) != expectedSha256)
    return invalid("selected compact facts hash mismatch");
  auto decoded = llvm::json::parse(bytes);
  if (!decoded) return invalid(llvm::toString(decoded.takeError()));
  const Object *root = decoded->getAsObject();
  if (!root || root->getString("schema") != "atlas.model_pairs.v1" ||
      root->getString("age_convention") != "issue_edge_zero") return invalid("unsupported schema or age convention");
  if (!keys(*root, {"schema", "age_convention", "model", "projection", "pairs", "chains", "provenance",
                    "consumer_rules", "limitations", "per_operation_reference"})) return invalid("unknown root constraint");
  const Object *projection = root->getObject("projection"), *model = root->getObject("model");
  if (!projection || projection->getInteger("version") != 1 ||
      projection->getString("policy") != "confirmed-exact-base-relations-v1" || !model)
    return invalid("requires a version 1 confirmed-only projection");
  if (!keys(*projection, {"version", "policy", "master_sha256", "master_path", "validation_sha256", "validation_path",
                          "per_op_sha256", "projector_sha256", "counts", "excluded_reasons"}) ||
      !projection->getObject("counts") || !projection->getObject("excluded_reasons"))
    return invalid("unsupported projection descriptor");
  ModelOverlapFacts result;
  for (const char *name : {"master_sha256", "validation_sha256", "per_op_sha256", "projector_sha256"}) {
    auto hash = projection->getString(name);
    if (!hash || !hashSyntax(*hash)) return invalid(std::string("missing projection.") + name);
    result.provenance[std::string("pairs_") + name] = hash->str();
  }
  for (const char *name : {"master_path", "validation_path"}) {
    auto path = projection->getString(name);
    if (!path || path->empty()) return invalid(std::string("missing projection.") + name);
    result.provenance[std::string("pairs_") + name] = path->str();
  }
  if (projection->getString("per_op_sha256") != operations.factsSha256())
    return invalid("projection does not bind the selected per-operation facts");
  for (const char *name : {"revision", "configuration_sha256"}) {
    auto value = model->getString(name);
    auto expected = operations.modelProvenance().find(name);
    if (!value || expected == operations.modelProvenance().end() || *value != expected->second)
      return invalid(std::string("incompatible model.") + name);
    result.provenance[std::string("pairs_") + name] = value->str();
  }
  const Object *configuration = model->getObject("configuration");
  auto expectedConfiguration = llvm::json::parse(operations.modelProvenance().at("configuration_json"));
  if (!expectedConfiguration) return invalid(llvm::toString(expectedConfiguration.takeError()));
  if (!configuration || llvm::json::Value(Object(*configuration)) != *expectedConfiguration)
    return invalid("incompatible explicit model configuration");
  for (const char *name : {"source_sha256", "exporter_sha256"}) {
    auto hash = model->getString(name);
    if (!hash || !hashSyntax(*hash)) return invalid(std::string("missing model.") + name);
    result.provenance[std::string("pairs_") + name] = hash->str();
  }
  result.provenance["pairs_sha256"] = expectedSha256.str();
  result.provenance["pairs_projection_version"] = "1";
  result.provenance["pairs_policy"] = "confirmed-exact-base-relations-v1";
  const Array *pairs = root->getArray("pairs"), *chains = root->getArray("chains");
  if (!pairs || !chains || (pairs->empty() && chains->empty())) return invalid("empty or missing projected coverage");
  std::set<std::string> sourcePointers;
  auto proof = [&](const Object &record, llvm::StringRef kind) {
    return source(record, kind) &&
        sourcePointers.insert(record.getObject("source_record")->getString("json_pointer")->str()).second;
  };
  for (const auto &value : *pairs) {
    const Object *record = value.getAsObject();
    if (!record || !proof(*record, "pairs")) return invalid("invalid or duplicate confirmed pair proof");
    if (!constraints(*record)) return invalid("unsupported pair semantic constraint");
    if (!keys(*record, {"source_record", "verilator", "evidence_summary", "first", "second", "relation", "swept",
                        "min_gap", "negative", "live_window", "first_completion_age", "first_live_window_end",
                        "max_in_flight", "domain_extension_scope", "previous_swept", "consumer_requirement",
                        "producer_consumer_requirements", "relation_semantics", "resource_semantics", "operand_order"}))
      return invalid("unknown pair constraint");
    auto first = record->getString("first"), second = record->getString("second");
    const auto *a = first ? operations.block(*first) : nullptr;
    const auto *b = second ? operations.block(*second) : nullptr;
    if (!a || !b || !sameGroup(a->engine, b->engine)) return invalid("unsupported pair operation or group");
    ModelOverlapFacts::Pair pair{first->str(), second->str(), {}, 0, 0};
    if (!relation(record->getObject("relation"), pair.relation)) return invalid("unsupported complete pair relation");
    const Array *swept = record->getArray("swept");
    auto minimum = record->getInteger("min_gap");
    if (!swept || swept->size() != 2 || (*swept)[0].getAsInteger() != 1 || !minimum ||
        !(*swept)[1].getAsInteger() || *minimum < 1 || *minimum > *(*swept)[1].getAsInteger() ||
        *(*swept)[1].getAsInteger() > INT_MAX) return invalid("invalid bounded pair domain");
    pair.minimum = int(*minimum); pair.high = int(*(*swept)[1].getAsInteger());
    if (!proofGaps(*record, {pair.minimum})) return invalid("pair proof does not bind its minimum gap");
    if (record->get("max_in_flight") &&
        (!record->getInteger("max_in_flight") || *record->getInteger("max_in_flight") < 1 ||
         *record->getInteger("max_in_flight") > 2)) return invalid("invalid pair post-progress count");
    int forms = bool(record->get("live_window")) + bool(record->get("first_completion_age")) +
        bool(record->get("first_live_window_end"));
    std::optional<int64_t> boundary;
    if (const Object *window = record->getObject("live_window")) {
      if (!keys(*window, {"first_completion_age", "high_includes_idle_boundary"}) ||
          window->getBoolean("high_includes_idle_boundary") != true) return invalid("invalid nested live window");
      boundary = window->getInteger("first_completion_age");
      if (boundary && *boundary >= 0 && *boundary < INT_MAX) ++*boundary;
      else return invalid("invalid completion age");
    } else if (record->get("first_completion_age")) {
      boundary = record->getInteger("first_completion_age");
      if (boundary && *boundary >= 0 && *boundary < INT_MAX) ++*boundary;
      else return invalid("invalid completion age");
    } else boundary = record->getInteger("first_live_window_end");
    if (forms != 1 || !boundary || *boundary != a->firstFreeAge || pair.high < *boundary)
      return invalid("pair live window does not match the selected observations");
    if (record->get("previous_swept")) {
      const Array *previous = record->getArray("previous_swept");
      if (!previous || previous->size() != 2 || (*previous)[0].getAsInteger() != 1 ||
          !(*previous)[1].getAsInteger() || *(*previous)[1].getAsInteger() < *boundary ||
          *(*previous)[1].getAsInteger() > pair.high) return invalid("invalid previous pair sweep domain");
    }
    const Object *negative = record->getObject("negative");
    if (!negative || negative->getInteger("gap") != pair.minimum - 1 ||
        negative->getString("result") != (pair.minimum == 1 ? "inexpressible" : "conflict"))
      return invalid("missing exact one-short pair control");
    if (pair.minimum > 1) {
      const Object *error = negative->getObject("error");
      if (!error || !error->getString("message") || error->getString("message")->empty())
        return invalid("one-short pair control needs a concrete resource conflict");
    }
    result.pairs.push_back(std::move(pair));
  }
  // Chain records are exact vectors; sweep metadata never broadens admission.
  for (const auto &value : *chains) {
    const Object *record = value.getAsObject();
    if (!record || !proof(*record, "chains") || record->getString("result") != "accepted" ||
        record->getString("relation_scope") != "adjacent") return invalid("invalid confirmed chain proof");
    if (!constraints(*record)) return invalid("unsupported chain semantic constraint");
    if (!keys(*record, {"source_record", "verilator", "evidence_summary", "operations", "gaps", "relation",
                        "all_pair_relations", "relation_scope", "result", "in_flight_count", "witness_only",
                        "no_stable_suffix_claim", "fixed_prefix_gaps", "swept_last_gap", "minimum_last_gap",
                        "swept", "min_gap", "accepted_gaps", "accepted_last_gaps", "minimum_accepted_gap",
                        "maximum_witness_gap", "first_live_window_end", "prefix_isolated_live_ends",
                        "previous_isolated_live_window_end", "prefix_observed_live_window_end", "bounded_final_gap_end",
                        "prefix_issue_ages", "issue_ages", "initial_gaps", "searched_final_gap", "swept_gap_index",
                        "swept_gap_range", "sweep_axis", "admitted_before_edge_progress_peak", "peak_admission_in_flight",
                        "in_flight_scope", "capacity_scope", "bound_reason", "matching_requirement", "consumer_requirement",
                        "producer_consumer_requirements", "relation_semantics", "producer_relation_summary", "negative"}))
      return invalid("unknown chain constraint");
    for (const char *flag : {"witness_only", "no_stable_suffix_claim"})
      if (record->get(flag) && record->getBoolean(flag) != true) return invalid("unsupported chain witness claim");
    if (record->getBoolean("witness_only") == true && record->getBoolean("no_stable_suffix_claim") != true)
      return invalid("witness chain cannot claim a stable suffix");
    if (record->get("negative")) {
      const Object *negative = record->getObject("negative");
      if (!negative || (negative->getString("result") != "conflict" &&
                        negative->getString("result") != "inexpressible"))
        return invalid("invalid retained chain control");
    }
    const Array *names = record->getArray("operations"), *gaps = record->getArray("gaps");
    const Array *matrix = record->getArray("all_pair_relations");
    auto count = record->getInteger("in_flight_count");
    if (!names || names->size() < 3 || !gaps || gaps->size() + 1 != names->size() || !matrix || !count ||
        *count < 1 || uint64_t(*count) > names->size() ||
        matrix->size() != names->size() * (names->size() - 1) / 2) return invalid("incomplete exact chain");
    ModelOverlapFacts::Chain chain;
    std::vector<ModelOverlapCommand> commands;
    int64_t issue = 0;
    std::string group;
    for (size_t i = 0; i < names->size(); ++i) {
      auto name = (*names)[i].getAsString();
      const auto *block = name ? operations.block(*name) : nullptr;
      if (!block || (i && !sameGroup(group, block->engine)) || llvm::StringRef(block->engine).starts_with("scalar"))
        return invalid("unsupported chain operation or group");
      group = block->engine;
      if (i) {
        auto gap = (*gaps)[i - 1].getAsInteger();
        if (!gap || *gap < 1 || *gap > INT_MAX || issue + *gap > INT_MAX)
          return invalid("invalid exact chain gap");
        chain.gaps.push_back(int(*gap)); issue += *gap;
      }
      chain.operations.push_back(name->str());
      commands.push_back({name->str(), Engine::Scalar, {}, {}, {}, int(issue), block->firstFreeAge});
    }
    for (const char *name : {"accepted_gaps", "accepted_last_gaps", "prefix_isolated_live_ends", "initial_gaps",
                             "prefix_issue_ages", "issue_ages"}) {
      if (!record->get(name)) continue;
      const Array *values = record->getArray(name);
      if (!values || values->empty()) return invalid("invalid retained chain age vector");
      for (const auto &value : *values)
        if (!value.getAsInteger() || *value.getAsInteger() < 0 || *value.getAsInteger() > INT_MAX)
          return invalid("invalid retained chain age");
      if (llvm::StringRef(name) == "issue_ages" || llvm::StringRef(name) == "prefix_issue_ages") {
        size_t length = names->size() - (llvm::StringRef(name) == "prefix_issue_ages" ? 1 : 0);
        if (values->size() != length) return invalid("retained issue vector has the wrong length");
        for (size_t i = 1; i < length; ++i)
          if (*(*values)[i].getAsInteger() - *(*values)[i - 1].getAsInteger() != chain.gaps[i - 1])
            return invalid("retained issue ages differ from exact chain gaps");
      }
    }
    for (const char *name : {"minimum_last_gap", "min_gap", "minimum_accepted_gap", "maximum_witness_gap",
                             "first_live_window_end", "previous_isolated_live_window_end",
                             "prefix_observed_live_window_end", "bounded_final_gap_end",
                             "admitted_before_edge_progress_peak", "peak_admission_in_flight", "swept_gap_index"})
      if (record->get(name)) {
        auto value = record->getInteger(name);
        if (!value || *value < 0 || *value > INT_MAX) return invalid("invalid retained chain bound");
        if (llvm::StringRef(name) == "swept_gap_index" && uint64_t(*value) >= chain.gaps.size())
          return invalid("chain sweep index exceeds exact gap vector");
      }
    if (record->get("fixed_prefix_gaps")) {
      const Array *prefix = record->getArray("fixed_prefix_gaps");
      if (!prefix || prefix->size() + 1 != chain.gaps.size()) return invalid("invalid fixed chain prefix");
      for (size_t i = 0; i < prefix->size(); ++i)
        if ((*prefix)[i].getAsInteger() != chain.gaps[i]) return invalid("chain prefix differs from exact vector");
    }
    for (const char *name : {"swept", "swept_last_gap", "swept_gap_range", "searched_final_gap"}) {
      if (!record->get(name)) continue;
      const Array *axis = record->getArray(name);
      if (!axis || axis->size() != 2 || !(*axis)[0].getAsInteger() || !(*axis)[1].getAsInteger() ||
          *(*axis)[0].getAsInteger() < 1 || *(*axis)[0].getAsInteger() > *(*axis)[1].getAsInteger() ||
          *(*axis)[1].getAsInteger() > INT_MAX) return invalid("invalid retained chain axis");
      if (llvm::StringRef(name) == "swept" || llvm::StringRef(name) == "swept_last_gap") {
        auto minimum = record->getInteger(llvm::StringRef(name) == "swept" ? "min_gap" : "minimum_last_gap");
        if (!minimum || *minimum < *(*axis)[0].getAsInteger() || *minimum > chain.gaps.back() ||
            chain.gaps.back() > *(*axis)[1].getAsInteger()) return invalid("chain vector lies outside its axis");
        if (llvm::StringRef(name) == "swept" &&
            !std::all_of(chain.gaps.begin(), chain.gaps.end(), [&](int gap) { return gap == chain.gaps.front(); }))
          return invalid("uniform chain axis does not match exact vector");
      }
    }
    for (const auto &entry : *matrix) {
      const Object *pair = entry.getAsObject();
      auto first = pair ? pair->getInteger("first_index") : std::nullopt;
      auto second = pair ? pair->getInteger("second_index") : std::nullopt;
      Relation matched;
      if (!pair || !keys(*pair, {"first_index", "second_index", "relation"}) || !first || !second || *first < 0 ||
          *first >= *second || uint64_t(*second) >= names->size() ||
          !relation(pair->getObject("relation"), matched) ||
          !chain.relations.emplace(std::pair<int, int>{int(*first), int(*second)}, std::move(matched)).second)
        return invalid("invalid complete chain relation matrix");
    }
    Relation summary;
    if (!relation(record->getObject("relation"), summary) || summary != chain.relations.at({0, 1}) ||
        peak(commands) != *count || !proofGaps(*record, chain.gaps))
      return invalid("inconsistent chain summary, proof gaps or post-progress membership");
    chain.inFlight = int(*count);
    result.sequences.push_back(std::move(chain));
  }
  return result;
}

llvm::Expected<ModelOverlapFacts> mlir::atlas::timing::loadModelOverlapFacts(
    llvm::StringRef path, llvm::StringRef expectedSha256, const ModelTimingFacts &operations) {
  auto buffer = llvm::MemoryBuffer::getFile(path);
  if (!buffer) return invalid("cannot read selected compact facts: " + path.str());
  return parseModelOverlapFacts((*buffer)->getBuffer(), expectedSha256, operations);
}
