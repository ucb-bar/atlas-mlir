#include "Atlas/AtlasTarget.h"
#include "Atlas/AtlasModelPairs.h"
#include "Atlas/AtlasModelVocabulary.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SHA256.h"
#include <algorithm>
#include <climits>
#include <cstdlib>
#include <iterator>
#include <set>

using namespace mlir::atlas::timing;
using mlir::atlas::kAtlasVmemBytes;
using mlir::atlas::kAtlasRegisterBytes;

namespace {
using Object = llvm::json::Object;
using Array = llvm::json::Array;
using Relation = std::map<std::string, std::string>;

llvm::Error invalid(const std::string &why) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), "model overlap facts: " + why);
}
bool keys(const Object &object, std::initializer_list<llvm::StringRef> allowed) {
  for (const auto &[key, value] : object)
    if (std::find(allowed.begin(), allowed.end(), llvm::StringRef(key)) == allowed.end()) return false;
  return true;
}
bool relation(const Object *object, Relation &out, bool qualified) {
  if (!object) return false;
  const auto &values = model_schema::relationValues();
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
    if (!text) return false;
    if (key != "mreg_banks") {
      auto allowed = values.find(key.str());
      if ((!qualified && key != "accumulator" && key != "weight_slot") ||
          allowed == values.end() || !allowed->second.count(text->str())) return false;
    }
    out[key.str()] = text->str();
  }
  auto get = [&](const char *key) { auto it = out.find(key); return it == out.end() ? std::string() : it->second; };
  std::string mode = get("scalar_registers"), file = get("scalar_destination_file"), others = get("scalar_other_registers");
  if (mode.empty() && (!file.empty() || !others.empty())) return false;
  if ((mode == "independent" || mode == "shared_base_read_only") && !file.empty()) return false;
  if (!mode.empty() && mode != "independent") {
    if (others != "independent") return false;
    if (mode != "shared_base_read_only" && (file.empty() || (mode != "same_load_destination" && file != "xrf"))) return false;
  }
  bool explicitOffsets = get("scalar_address_alignment") == "explicit_byte_offsets";
  for (const char *side : {"first", "second"}) {
    std::string mnemonic = get((std::string(side) + "_mnemonic").c_str());
    bool offset = out.count(std::string(side) + "_scalar_byte_offset");
    if (offset && !explicitOffsets) return false;
    if (explicitOffsets && mnemonic != "vload" && mnemonic != "vstore" && !offset) return false;
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
Relation relationClass(Relation relation) {
  auto &banks = relation["mreg_banks"];
  if (banks != "disjoint") {
    bool equal = banks.find('=') != std::string::npos, shared = banks.find('~') != std::string::npos;
    banks = equal && shared ? "mixed_alias" : equal ? "register_alias" : "bank_alias";
  }
  return relation;
}
// Match the complete producer relation. Optional qualifiers are predicates;
// absent qualifiers never acquire invented constraints or resource capacity.
bool matchesRelation(const Relation &expected, const ModelOverlapCommand &first, const ModelOverlapCommand &second) {
  Relation base, extra;
  for (const auto &[key, value] : expected)
    (key == "mreg_banks" || key == "accumulator" || key == "weight_slot" ? base : extra)[key] = value;
  if (base != actualRelation(first, second)) return false;
  if (extra.empty()) return !llvm::StringRef(first.block).starts_with("scalar") &&
      !llvm::StringRef(second.block).starts_with("scalar");
  if (llvm::StringRef(first.block).starts_with("scalar") || llvm::StringRef(second.block).starts_with("scalar"))
    for (const char *key : {"first_mnemonic", "second_mnemonic", "scalar_address_alignment", "scalar_registers", "vmem_bank"})
      if (!extra.count(key)) return false;
  if (!first.scalarInstruction || !second.scalarInstruction || !first.context || !second.context ||
      !first.context->matches(*first.scalarInstruction) || !second.context->matches(*second.scalarInstruction) ||
      !first.context->sameRegion(*second.context)) return false;
  const Instr &a = *first.scalarInstruction, &b = *second.scalarInstruction;
  if (!a.op || !b.op) return false;
  auto memory = [](const Instr &in) {
    return in.op->opClass == OpClass::ScalarLoad || in.op->opClass == OpClass::ScalarStore ||
        in.op->opClass == OpClass::ScaleLoad || in.op->opClass == OpClass::VLoad || in.op->opClass == OpClass::VStore;
  };
  if (!memory(a) || !memory(b)) return false;
  auto blockMatches = [](const ModelOverlapCommand &command, const Instr &in) {
    if (command.block == "scalar_lsu.load") return in.op->opClass == OpClass::ScalarLoad;
    if (command.block == "scalar_lsu.store") return in.op->opClass == OpClass::ScalarStore;
    if (command.block == "scalar_lsu.scale_load") return in.op->opClass == OpClass::ScaleLoad;
    if (command.block == "vlsu.vload") return in.op->opClass == OpClass::VLoad;
    if (command.block == "vlsu.vstore") return in.op->opClass == OpClass::VStore;
    return false;
  };
  if (!blockMatches(first, a) || !blockMatches(second, b)) return false;
  using Resource = std::pair<std::string, int>;
  struct Roles {
    Resource base;
    std::optional<Resource> destination, data;
    std::set<Resource> resources;
    bool vector = false;
  };
  auto roles = [](const Instr &in) {
    Roles r{{"xrf", in.rs1}, {}, {}, {}, in.op->opClass == OpClass::VLoad || in.op->opClass == OpClass::VStore};
    r.resources.insert(r.base);
    if (in.op->opClass == OpClass::ScalarLoad || in.op->opClass == OpClass::ScaleLoad) {
      r.destination = Resource{in.op->opClass == OpClass::ScaleLoad ? "erf" : "xrf", in.rd};
      r.resources.insert(*r.destination);
    }
    if (in.op->opClass == OpClass::ScalarStore) {
      r.data = Resource{"xrf", in.rs2}; r.resources.insert(*r.data);
    }
    return r;
  };
  Roles ra = roles(a), rb = roles(b);
  auto get = [&](const char *key) { auto it = extra.find(key); return it == extra.end() ? std::string() : it->second; };
  if ((!get("first_mnemonic").empty() && get("first_mnemonic") != a.op->name) ||
      (!get("second_mnemonic").empty() && get("second_mnemonic") != b.op->name)) return false;
  std::set<Resource> common;
  std::set_intersection(ra.resources.begin(), ra.resources.end(), rb.resources.begin(), rb.resources.end(),
                        std::inserter(common, common.begin()));
  std::string mode = get("scalar_registers"), file = get("scalar_destination_file");
  auto only = [&](const Resource &resource) { return common == std::set<Resource>{resource}; };
  auto writable = [](const std::optional<Resource> &destination) {
    return destination && (destination->first != "xrf" || destination->second != 0);
  };
  if (mode == "independent") {
    if (!common.empty()) return false;
  } else if (mode == "same_load_destination") {
    if (!writable(ra.destination) || ra.destination != rb.destination || ra.destination->first != file ||
        !only(*ra.destination) || ra.base == *ra.destination || rb.base == *ra.destination) return false;
  } else if (mode == "store_data_equals_load_destination") {
    if (!writable(rb.destination) || rb.destination->first != "xrf" || ra.data != rb.destination ||
        !only(*rb.destination) || ra.base == *rb.destination || rb.base == *rb.destination) return false;
  } else if (mode == "first_address_base_equals_second_load_destination") {
    if (!writable(rb.destination) || rb.destination->first != "xrf" || ra.base != *rb.destination ||
        rb.base == *rb.destination || ra.data == rb.destination || ra.destination == rb.destination ||
        !only(*rb.destination)) return false;
  } else if (mode == "shared_base_read_only") {
    if (ra.base != rb.base || !only(ra.base) || ra.destination == ra.base || rb.destination == ra.base ||
        ra.data == ra.base || rb.data == ra.base) return false;
  } else if (!mode.empty()) return false;
  const auto &aa = first.context->scalarAddresses(), &ba = second.context->scalarAddresses();
  for (size_t ordinal = 0; ordinal < aa.size(); ++ordinal) {
    uint64_t x = aa[ordinal], y = ba[ordinal];
    uint64_t xs = ra.vector ? x : x & ~uint64_t(3), ys = rb.vector ? y : y & ~uint64_t(3);
    uint64_t xw = ra.vector ? kAtlasRegisterBytes : 4, yw = rb.vector ? kAtlasRegisterBytes : 4;
    if (x >= kAtlasVmemBytes || y >= kAtlasVmemBytes || xs + xw > kAtlasVmemBytes || ys + yw > kAtlasVmemBytes ||
        xs / kVmemBankBytes != (xs + xw - 1) / kVmemBankBytes ||
        ys / kVmemBankBytes != (ys + yw - 1) / kVmemBankBytes ||
        (ra.vector && x % kAtlasRegisterBytes) || (rb.vector && y % kAtlasRegisterBytes)) return false;
    std::string bank = xs / kVmemBankBytes == ys / kVmemBankBytes ? "same" : "other";
    std::string window = xs == ys && xw == yw ? "same" : xs < ys + yw && ys < xs + xw ? "overlap" : "disjoint";
    if ((!get("vmem_bank").empty() && get("vmem_bank") != bank) ||
        (!get("vmem_window").empty() && get("vmem_window") != window) ||
        (!get("vmem_word_window").empty() && get("vmem_word_window") != window)) return false;
    if (mode == "shared_base_read_only" && x != y) return false;
    if (get("scalar_address_alignment") == "word" && ((!ra.vector && x % 4) || (!rb.vector && y % 4))) return false;
    if (!get("first_scalar_byte_offset").empty() &&
        (ra.vector || get("first_scalar_byte_offset") != std::to_string(x % 4))) return false;
    if (!get("second_scalar_byte_offset").empty() &&
        (rb.vector || get("second_scalar_byte_offset") != std::to_string(y % 4))) return false;
  }
  return true;
}

int peak(const std::vector<ModelOverlapCommand> &commands) {
  int result = 0;
  for (const ModelOverlapCommand &edge : commands) {
    int live = 0;
    for (const ModelOverlapCommand &command : commands)
      live += command.issue <= edge.issue && int64_t(edge.issue) < int64_t(command.issue) + command.firstFreeAge - 1;
    result = std::max(result, live);
  }
  return result;
}
bool sameGroup(llvm::StringRef first, llvm::StringRef second) {
  if (first.starts_with("scalar") || second.starts_with("scalar"))
    return first == "scalar_lsu" && second == "scalar_lsu";
  auto vector = [](llvm::StringRef engine) { return engine == "vpu" || engine == "xlu"; };
  return first == second || (vector(first) && vector(second));
}

bool composedCapacity(const std::vector<ModelOverlapCommand> &commands) {
  std::vector<std::set<int>> ports(commands.size());
  for (size_t j = 0; j < commands.size(); ++j) {
    const auto &next = commands[j];
    llvm::StringRef name(next.block);
    if (name.starts_with("vpu.")) {
      int live = 0;
      for (size_t i = 0; i < j; ++i)
        live += next.issue - commands[i].issue < commands[i].firstFreeAge - 1;
      if (live >= 2) return false;
      continue;
    }
    std::set<int> busy;
    int computes = 0;
    for (size_t i = 0; i < j; ++i) {
      int gap = next.issue - commands[i].issue;
      if (gap < 32) busy.insert(ports[i].begin(), ports[i].end());
      computes += llvm::StringRef(commands[i].block).contains(".matmul") && gap <= commands[i].firstFreeAge - 1;
    }
    if (name.contains(".matmul")) {
      if (computes >= (name.starts_with("mxu0.") ? 3 : 2)) return false;
      ports[j] = {0};
    } else if (name.ends_with("push_acc_bf16")) ports[j] = {0, 1};
    else if (name.ends_with("pop_acc_bf16")) ports[j] = {2, 3};
    else if (name.ends_with("pop_acc_fp8")) ports[j] = {2};
    else ports[j] = {busy.count(1) ? 0 : 1};
    for (int port : ports[j]) if (busy.count(port)) return false;
  }
  return true;
}

bool qualifiedConstraint(llvm::StringRef key, const llvm::json::Value &value) {
  const auto &fingerprints = model_schema::qualifiedConstraintHashes();
  auto allowed = fingerprints.find(key.str());
  if (allowed == fingerprints.end()) return false;
  std::string bytes = canonicalJson(value);
  auto digest = llvm::SHA256::hash(llvm::ArrayRef<uint8_t>(
      reinterpret_cast<const uint8_t *>(bytes.data()), bytes.size()));
  return allowed->second.count(llvm::toHex(llvm::ArrayRef<uint8_t>(digest), true));
}

bool constraints(const Object &record, bool qualified) {
  for (const char *name : {"relation_semantics", "capacity_scope", "producer_consumer_requirements",
                           "domain_extension_scope"})
    if (record.get(name) && (!qualified || !qualifiedConstraint(name, *record.get(name)))) return false;
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
      if ((!value || !allowed.count(value->str())) &&
          (!qualified || !qualifiedConstraint(name, *record.get(name)))) return false;
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
                    "current_model", "fragment", "fragment_sha256", "full_memory_checked_bytes", "attempted_dispatch_gap",
                    "relation", "result_sha256", "same_accumulator"}))
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
  if (proof->get("relation") && (!record.get("relation") || *proof->get("relation") != *record.get("relation"))) return false;
  if (proof->get("result_sha256") &&
      (!proof->getString("result_sha256") || !hashSyntax(*proof->getString("result_sha256")))) return false;
  if (proof->get("same_accumulator")) {
    const Object *rel = record.getObject("relation");
    if (!rel || !proof->getBoolean("same_accumulator") ||
        *proof->getBoolean("same_accumulator") != (rel->getString("accumulator") == "same")) return false;
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

bool ModelOverlapFacts::rejectsPortHistory(const std::vector<ModelOverlapCommand> &commands) const {
  if (!provenance.count("composed_sha256") || commands.empty()) return false;
  auto engine = llvm::StringRef(commands.front().block).split('.').first;
  if (engine != "mxu0" && engine != "mxu1") return false;
  for (const auto &command : commands) {
    auto [unit, kind] = llvm::StringRef(command.block).split('.');
    if (unit != engine || (kind != "matmul" && kind != "matmul_acc" && kind != "push_weight" &&
        kind != "push_acc_fp8" && kind != "push_acc_bf16" && kind != "pop_acc_fp8" && kind != "pop_acc_bf16"))
      return true;
  }
  return !composedCapacity(commands);
}

bool ModelOverlapFacts::admits(const std::vector<ModelOverlapCommand> &commands, bool chains) const {
  auto reportQuery = [&](llvm::StringRef reason, bool admitted = false) {
    if (!std::getenv("ATLAS_MODEL_LOG_UNMATCHED")) return admitted;
    Array operations, gaps, operands, matrix;
    for (size_t i = 0; i < commands.size(); ++i) {
      const auto &command = commands[i];
      operations.push_back(command.block);
      if (i) gaps.push_back(command.issue - commands[i - 1].issue);
      Array registers;
      for (int reg : command.registers) registers.push_back(reg);
      Object operand{{"registers", std::move(registers)}};
      if (command.accumulator) operand["accumulator"] = *command.accumulator;
      if (command.weightSlot) operand["weight_slot"] = *command.weightSlot;
      operands.push_back(std::move(operand));
      for (size_t j = 0; j < i; ++j) {
        Object relation;
        for (const auto &[key, value] : actualRelation(commands[j], command)) relation[key] = value;
        matrix.push_back(Object{{"first_index", int64_t(j)}, {"second_index", int64_t(i)},
                               {"relation", std::move(relation)}});
      }
    }
    std::string query = canonicalJson(Object{{"reason", reason.str()}, {"operations", std::move(operations)},
        {"gaps", std::move(gaps)}, {"operands", std::move(operands)}, {"all_pair_relations", std::move(matrix)},
        {"in_flight_count", peak(commands)}});
    std::string message = (admitted ? "ATLAS_MODEL_COMPOSED " : "ATLAS_MODEL_UNMATCHED ") + query;
    if (loggedQueries.insert(message).second) llvm::errs() << message << "\n";
    return admitted;
  };
  auto admitsPair = [&](const ModelOverlapCommand &first, const ModelOverlapCommand &second) {
    int gap = second.issue - first.issue;
    for (const Pair &pair : pairs)
      if (pair.first == first.block && pair.second == second.block && pair.minimum <= gap && gap <= pair.high &&
          matchesRelation(pair.relation, first, second)) return true;
    return false;
  };
  if (commands.size() == 2) return admitsPair(commands[0], commands[1]) || reportQuery("pair");
  if (!chains || commands.size() < 3) return false;
  for (size_t i = 0; i < commands.size(); ++i)
    for (size_t j = i + 1; j < commands.size(); ++j)
      if (commands[j].issue - commands[i].issue < commands[i].firstFreeAge &&
          !admitsPair(commands[i], commands[j])) return reportQuery("chain_live_pair");
  std::string engine = llvm::StringRef(commands.front().block).split('.').first.str();
  auto scope = compositionClasses.find(engine);
  if (scope != compositionClasses.end()) {
    bool covered = composedCapacity(commands);
    for (size_t i = 0; i < commands.size(); ++i) {
      covered &= llvm::StringRef(commands[i].block).split('.').first == engine;
      covered &= composedOperations.count(commands[i].block);
      for (size_t j = 0; j < i; ++j)
        if (commands[i].issue - commands[j].issue < commands[j].firstFreeAge)
          covered &= scope->second.count(relationClass(actualRelation(commands[j], commands[i])));
    }
    if (covered) return reportQuery("chain", true);
  }
  for (const Chain &chain : sequences) {
    if (chain.operations.size() != commands.size() || chain.inFlight != peak(commands)) continue;
    bool matched = true;
    for (size_t i = 0; i < commands.size(); ++i) {
      matched &= chain.operations[i] == commands[i].block;
      if (i) matched &= chain.gaps[i - 1] == commands[i].issue - commands[i - 1].issue;
      for (size_t j = 0; j < i; ++j)
        matched &= matchesRelation(chain.relations.at({int(j), int(i)}), commands[j], commands[i]);
    }
    if (matched) return true;
  }
  return reportQuery("chain");
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
  bool qualified = projection && projection->getInteger("version") == 2 &&
      projection->getString("policy") == "confirmed-exact-scalar-relations-v2";
  bool base = projection && projection->getInteger("version") == 1 &&
      projection->getString("policy") == "confirmed-exact-base-relations-v1";
  if ((!qualified && !base) || !model) return invalid("requires a recognized confirmed-only projection");
  if (!keys(*projection, {"version", "policy", "master_sha256", "master_path", "validation_sha256", "validation_path",
                          "per_op_sha256", "projector_sha256", "counts", "excluded_reasons"}) ||
      !projection->getObject("counts") || !projection->getObject("excluded_reasons"))
    return invalid("unsupported projection descriptor");
  ModelOverlapFacts result;
  for (const char *name : {"master_sha256", "validation_sha256", "per_op_sha256", "projector_sha256",
                           "master_path", "validation_path"}) {
    auto value = projection->getString(name);
    bool hash = llvm::StringRef(name).ends_with("_sha256");
    if (!value || (hash ? !hashSyntax(*value) : value->empty()))
      return invalid(std::string("missing projection.") + name);
    result.provenance[std::string("pairs_") + name] = value->str();
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
  result.provenance["pairs_projection_version"] = qualified ? "2" : "1";
  result.provenance["pairs_policy"] = projection->getString("policy")->str();
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
    if (!constraints(*record, qualified)) return invalid("unsupported pair semantic constraint");
    if (!keys(*record, {"source_record", "verilator", "evidence_summary", "first", "second", "relation", "swept",
                        "min_gap", "negative", "live_window", "first_completion_age", "first_live_window_end",
                        "max_in_flight", "domain_extension_scope", "previous_swept", "consumer_requirement",
                        "producer_consumer_requirements", "relation_semantics", "resource_semantics", "operand_order"}))
      return invalid("unknown pair constraint");
    auto first = record->getString("first"), second = record->getString("second");
    const auto *a = first ? operations.block(*first) : nullptr;
    const auto *b = second ? operations.block(*second) : nullptr;
    if (!a || !b || (!sameGroup(a->engine, b->engine) || (!qualified && llvm::StringRef(a->engine).starts_with("scalar"))))
      return invalid("unsupported pair operation or group");
    ModelOverlapFacts::Pair pair{first->str(), second->str(), {}, 0, 0};
    if (!relation(record->getObject("relation"), pair.relation, qualified)) return invalid("unsupported complete pair relation");
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
    const Object *window = record->getObject("live_window");
    if (window || record->get("first_completion_age")) {
      if (window && (!keys(*window, {"first_completion_age", "high_includes_idle_boundary"}) ||
          window->getBoolean("high_includes_idle_boundary") != true)) return invalid("invalid nested live window");
      boundary = (window ? window : record)->getInteger("first_completion_age");
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
    if (!constraints(*record, qualified)) return invalid("unsupported chain semantic constraint");
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
      if (!block || (i && !sameGroup(group, block->engine)) ||
          (!qualified && llvm::StringRef(block->engine).starts_with("scalar")))
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
          !relation(pair->getObject("relation"), matched, qualified) ||
          !chain.relations.emplace(std::pair<int, int>{int(*first), int(*second)}, std::move(matched)).second)
        return invalid("invalid complete chain relation matrix");
    }
    Relation summary;
    if (!relation(record->getObject("relation"), summary, qualified) || summary != chain.relations.at({0, 1}) ||
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

llvm::Error ModelOverlapFacts::loadComposedFacts(
    llvm::StringRef path, llvm::StringRef expectedSha256, const ModelTimingFacts &operations) {
  auto buffer = llvm::MemoryBuffer::getFile(path);
  if (!buffer) return invalid("cannot read selected composed facts: " + path.str());
  return addComposedFacts((*buffer)->getBuffer(), expectedSha256, operations);
}

llvm::Error ModelOverlapFacts::addComposedFacts(
    llvm::StringRef bytes, llvm::StringRef expectedSha256, const ModelTimingFacts &operations) {
  const auto &model = operations.modelProvenance();
  if (model.at("revision") != "9e6313bddecb8ba936206eb347f5cd4e5335282c" ||
      model.at("source_sha256") != "a9a919c3d118af94e2783c8969e05ad19f8d2523ec093922c4f9957641b54426" ||
      model.at("configuration_sha256") != "c41ab668a2a6bf6a9351cb0df4e544703e14818f050f248d83a8e250d260e33a")
    return invalid("composition capacity rules require the audited model source and configuration");
  auto digest = llvm::SHA256::hash(llvm::ArrayRef<uint8_t>(
      reinterpret_cast<const uint8_t *>(bytes.data()), bytes.size()));
  if (!hashSyntax(expectedSha256) || llvm::toHex(llvm::ArrayRef<uint8_t>(digest), true) != expectedSha256)
    return invalid("selected composed facts hash mismatch");
  auto decoded = llvm::json::parse(bytes);
  if (!decoded) return invalid(llvm::toString(decoded.takeError()));
  const Object *root = decoded->getAsObject();
  if (!root || !keys(*root, {"schema", "rule", "age_convention", "provenance", "pair_classes", "engines",
                            "pairs", "chains"}) ||
      root->getString("schema") != "atlas.model_composed.v1" ||
      root->getString("rule") != "row-footprints-and-structural-capacity-v1" ||
      root->getString("age_convention") != "issue_edge_zero")
    return invalid("unsupported composed facts schema or rule");
  const Object *pins = root->getObject("provenance"), *engines = root->getObject("engines");
  const Array *classes = root->getArray("pair_classes"), *derivedPairs = root->getArray("pairs");
  const Array *derivedChains = root->getArray("chains");
  if (!pins || !engines || !classes || !derivedPairs || !derivedChains ||
      !keys(*pins, {"per_op_sha256", "base_pairs_sha256", "master_sha256", "deriver_sha256",
                    "pair_replay_sha256", "chain_replay_sha256", "holdouts_sha256", "revision",
                    "configuration_sha256"}) ||
      pins->getString("per_op_sha256") != operations.factsSha256() ||
      pins->getString("base_pairs_sha256") != provenance["pairs_sha256"])
    return invalid("composed facts do not bind selected observations and pairs");
  ModelOverlapFacts result = *this;
  for (const auto &[key, value] : *pins) {
    auto text = value.getAsString();
    if (!text || (key != "revision" && !hashSyntax(*text))) return invalid("invalid composed provenance");
    if (key == "revision" || key == "configuration_sha256") {
      auto expected = operations.modelProvenance().find(key.str());
      if (expected == operations.modelProvenance().end() || *text != expected->second)
        return invalid("incompatible composed model identity");
    }
    result.provenance["composed_" + key.str()] = text->str();
  }
  if (pins->size() != 9) return invalid("incomplete composed provenance");
  auto positive = [](const Object &object, llvm::StringRef key) {
    auto value = object.getInteger(key);
    return value && *value > 0 && *value <= INT_MAX;
  };
  std::set<std::string> supported;
  for (const auto &[name, value] : *engines) {
    const Object *audit = value.getAsObject();
    if ((name != "mxu0" && name != "mxu1" && name != "vpu") || !audit ||
        !keys(*audit, {"operations", "chains", "confirmed_chains", "accepted_chains", "negative_points", "rejected_negatives",
                       "holdout_chains", "holdout_pairs", "holdout_failures"}) ||
        !positive(*audit, "chains") || !positive(*audit, "confirmed_chains") ||
        audit->getInteger("confirmed_chains") > audit->getInteger("chains") ||
        audit->getInteger("accepted_chains") != audit->getInteger("chains") ||
        !positive(*audit, "negative_points") ||
        audit->getInteger("rejected_negatives") != audit->getInteger("negative_points") ||
        audit->getInteger("holdout_failures") != 0 ||
        !audit->getInteger("holdout_chains") || *audit->getInteger("holdout_chains") < 0 ||
        !audit->getInteger("holdout_pairs") || *audit->getInteger("holdout_pairs") < 0 ||
        (!positive(*audit, "holdout_chains") && !positive(*audit, "holdout_pairs")))
      return invalid("composed engine lacks complete agreement and holdouts");
    const Array *names = audit->getArray("operations");
    if (!names || names->empty()) return invalid("composed engine lacks an operation domain");
    for (const auto &entry : *names) {
      auto operation = entry.getAsString();
      const auto *block = operation ? operations.block(*operation) : nullptr;
      if (!block || block->engine != name || !result.composedOperations.insert(operation->str()).second)
        return invalid("invalid composed operation domain");
    }
    supported.insert(name.str());
  }
  std::set<std::pair<std::string, Relation>> admittedClasses;
  for (const auto &entry : *classes) {
    const Object *audit = entry.getAsObject();
    auto engine = audit ? audit->getString("engine") : std::nullopt;
    Relation rel;
    const Object *category = audit ? audit->getObject("relation") : nullptr;
    if (category && keys(*category, {"mreg_banks", "accumulator", "weight_slot"}))
      for (const auto &[key, value] : *category)
        if (auto text = value.getAsString()) rel[key.str()] = text->str();
    bool validClass = category && rel.size() == category->size() &&
        (rel["mreg_banks"] == "disjoint" || rel["mreg_banks"] == "register_alias" ||
         rel["mreg_banks"] == "bank_alias" || rel["mreg_banks"] == "mixed_alias");
    for (const auto &[key, value] : rel)
      if (key != "mreg_banks" && value != "same" && value != "other") validClass = false;
    if (!audit || !keys(*audit, {"engine", "relation", "pairs", "confirmed", "equal", "negative_points",
                                 "negative_violations"}) ||
        !engine || !supported.count(engine->str()) || !validClass ||
        !positive(*audit, "pairs") || !positive(*audit, "confirmed") ||
        audit->getInteger("confirmed") > audit->getInteger("pairs") ||
        audit->getInteger("equal") != audit->getInteger("pairs") ||
        !positive(*audit, "negative_points") || audit->getInteger("negative_violations") != 0 ||
        !admittedClasses.emplace(engine->str(), std::move(rel)).second)
      return invalid("composed pair class lacks complete agreement and witnessed negatives");
  }
  auto operationEngine = [&](llvm::StringRef name) {
    const auto *block = operations.block(name);
    return block && supported.count(block->engine) && result.composedOperations.count(name.str())
        ? block->engine : std::string();
  };
  for (const auto &entry : *derivedPairs) {
    const Object *record = entry.getAsObject();
    auto first = record ? record->getString("first") : std::nullopt;
    auto second = record ? record->getString("second") : std::nullopt;
    auto minimum = record ? record->getInteger("minimum") : std::nullopt;
    auto high = record ? record->getInteger("high") : std::nullopt;
    Relation rel;
    if (!record || !keys(*record, {"first", "second", "relation", "minimum", "high"}) || !first || !second ||
        !minimum || !high || *minimum < 1 || *minimum > *high || *high > INT_MAX ||
        operationEngine(*first).empty() || operationEngine(*first) != operationEngine(*second) ||
        !relation(record->getObject("relation"), rel, false) ||
        !admittedClasses.count({operationEngine(*first), relationClass(rel)}) ||
        *high != operations.block(*first)->firstFreeAge)
      return invalid("derived pair is outside validated class or bounded live domain");
    result.pairs.push_back({first->str(), second->str(), std::move(rel), int(*minimum), int(*high)});
  }
  for (const auto &entry : *derivedChains) {
    const Object *record = entry.getAsObject();
    const Array *names = record ? record->getArray("operations") : nullptr;
    const Array *gaps = record ? record->getArray("gaps") : nullptr;
    const Array *matrix = record ? record->getArray("all_pair_relations") : nullptr;
    auto count = record ? record->getInteger("in_flight_count") : std::nullopt;
    if (!record || !keys(*record, {"operations", "gaps", "all_pair_relations", "in_flight_count"}) ||
        !names || names->size() < 3 || names->size() > 64 || !gaps || gaps->size() + 1 != names->size() ||
        !matrix || matrix->size() != names->size() * (names->size() - 1) / 2 || !count)
      return invalid("incomplete derived chain");
    Chain chain;
    std::vector<ModelOverlapCommand> commands;
    std::string engine;
    int64_t issue = 0;
    for (size_t i = 0; i < names->size(); ++i) {
      auto name = (*names)[i].getAsString();
      if (!name || operationEngine(*name).empty() || (i && operationEngine(*name) != engine))
        return invalid("derived chain crosses validated engine class");
      engine = operationEngine(*name);
      if (i) {
        auto gap = (*gaps)[i - 1].getAsInteger();
        if (!gap || *gap < 1 || *gap > INT_MAX || issue + *gap > INT_MAX)
          return invalid("invalid derived chain gaps");
        chain.gaps.push_back(int(*gap)); issue += *gap;
      }
      chain.operations.push_back(name->str());
      commands.push_back({name->str(), Engine::Scalar, {}, {}, {}, int(issue), operations.block(*name)->firstFreeAge});
    }
    if (!positive(*engines->getObject(engine), "holdout_chains"))
      return invalid("derived chain engine lacks chain holdouts");
    for (const auto &value : *matrix) {
      const Object *pair = value.getAsObject();
      auto i = pair ? pair->getInteger("first_index") : std::nullopt;
      auto j = pair ? pair->getInteger("second_index") : std::nullopt;
      Relation rel;
      if (!pair || !keys(*pair, {"first_index", "second_index", "relation"}) || !i || !j || *i < 0 ||
          *i >= *j || uint64_t(*j) >= names->size() || !relation(pair->getObject("relation"), rel, false) ||
          !chain.relations.emplace(std::make_pair(*i, *j), rel).second)
        return invalid("derived chain relation is outside validated class");
      int gap = commands[*j].issue - commands[*i].issue;
      if (gap < commands[*i].firstFreeAge && !admittedClasses.count({engine, relationClass(rel)}))
        return invalid("derived live pair relation is outside validated class");
      if (gap < commands[*i].firstFreeAge &&
          !std::any_of(result.pairs.begin(), result.pairs.end(), [&](const Pair &witness) {
            return witness.first == commands[*i].block && witness.second == commands[*j].block &&
                witness.relation == rel && witness.minimum <= gap && gap <= witness.high;
          })) return invalid("derived chain lacks an admitted live pair");
    }
    if (peak(commands) != *count || !composedCapacity(commands))
      return invalid("derived chain violates structural capacity or post-progress count");
    chain.inFlight = int(*count);
    result.sequences.push_back(std::move(chain));
  }
  for (const auto &[engine, rel] : admittedClasses)
    if (positive(*engines->getObject(engine), "holdout_chains")) result.compositionClasses[engine].insert(rel);
  result.provenance["composed_sha256"] = expectedSha256.str();
  *this = std::move(result);
  return llvm::Error::success();
}
