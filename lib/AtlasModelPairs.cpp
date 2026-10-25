#include "Atlas/AtlasTarget.h"
#include "Atlas/AtlasModelPairs.h"
#include "Atlas/AtlasModelVocabulary.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include <algorithm>
#include <climits>
#include <iterator>
#include <set>
#include <tuple>

using namespace mlir::atlas::timing;

namespace {
using Object = llvm::json::Object;
using Array = llvm::json::Array;
using Relation = std::map<std::string, std::string>;

llvm::Error invalid(const std::string &why) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), modelTimingDiagnostic("model overlap facts: " + why));
}
bool keys(const Object &object, std::initializer_list<llvm::StringRef> allowed) {
  for (const auto &[key, value] : object)
    if (std::find(allowed.begin(), allowed.end(), llvm::StringRef(key)) == allowed.end()) return false;
  return true;
}
bool relation(const Object *object, Relation &out) {
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
      if (key != "accumulator" && key != "weight_slot") return false;
      auto allowed = values.find(key.str());
      if (allowed == values.end() || !allowed->second.count(text->str())) return false;
    }
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
  return expected == actualRelation(first, second);
}

bool composedCapacity(const std::vector<ModelOverlapCommand> &commands,
                      int sequencerCycles, int mxu0Compute, int mxu1Compute, int vpuSlots) {
  std::vector<std::set<int>> ports(commands.size());
  for (size_t j = 0; j < commands.size(); ++j) {
    const auto &next = commands[j];
    llvm::StringRef name(next.block);
    if (name.starts_with("vpu.")) {
      int live = 0;
      for (size_t i = 0; i < j; ++i)
        live += next.issue - commands[i].issue < commands[i].firstFreeAge - 1;
      if (live >= vpuSlots) return false;
      continue;
    }
    std::set<int> busy;
    int computes = 0;
    for (size_t i = 0; i < j; ++i) {
      int gap = next.issue - commands[i].issue;
      if (gap < sequencerCycles) busy.insert(ports[i].begin(), ports[i].end());
      computes += llvm::StringRef(commands[i].block).contains(".matmul") && gap <= commands[i].firstFreeAge - 1;
    }
    if (name.contains(".matmul")) {
      if (computes >= (name.starts_with("mxu0.") ? mxu0Compute : mxu1Compute)) return false;
      ports[j] = {0};
    } else if (name.ends_with("push_acc_bf16")) ports[j] = {0, 1};
    else if (name.ends_with("pop_acc_bf16")) ports[j] = {2, 3};
    else if (name.ends_with("pop_acc_fp8")) ports[j] = {2};
    else ports[j] = {busy.count(1) ? 0 : 1};
    for (int port : ports[j]) if (busy.count(port)) return false;
  }
  return true;
}

} // namespace

bool ModelOverlapFacts::rejectsPortHistory(const std::vector<ModelOverlapCommand> &commands) const {
  if (!structuralCapacity || commands.empty()) return false;
  auto engine = llvm::StringRef(commands.front().block).split('.').first;
  if (engine != "mxu0" && engine != "mxu1") return false;
  for (const auto &command : commands) {
    auto [unit, kind] = llvm::StringRef(command.block).split('.');
    if (unit != engine || (kind != "matmul" && kind != "matmul_acc" && kind != "push_weight" &&
        kind != "push_acc_fp8" && kind != "push_acc_bf16" && kind != "pop_acc_fp8" && kind != "pop_acc_bf16"))
      return true;
  }
  return !composedCapacity(commands, sequencerCycles, mxu0Compute, mxu1Compute, vpuSlots);
}

bool ModelOverlapFacts::admits(const std::vector<ModelOverlapCommand> &commands, bool chains) const {
  auto admitsPair = [&](const ModelOverlapCommand &first, const ModelOverlapCommand &second) -> const Pair * {
    int gap = second.issue - first.issue;
    auto domain = pairIndices.find({first.block, second.block});
    if (domain == pairIndices.end()) return nullptr;
    for (size_t index : domain->second) {
      const Pair &pair = pairs[index];
      if (pair.minimum <= gap && gap <= pair.high &&
          matchesRelation(pair.relation, first, second)) return &pair;
    }
    return nullptr;
  };
  if (commands.size() == 2) return admitsPair(commands[0], commands[1]) != nullptr;
  if (!chains || commands.size() < 3) return false;
  for (size_t i = 0; i < commands.size(); ++i)
    for (size_t j = i + 1; j < commands.size(); ++j)
      if (commands[j].issue - commands[i].issue < commands[i].firstFreeAge &&
          !admitsPair(commands[i], commands[j])) return false;
  std::string engine = llvm::StringRef(commands.front().block).split('.').first.str();
  auto scope = compositionClasses.find(engine);
  if (scope != compositionClasses.end()) {
    bool covered = composedCapacity(commands, sequencerCycles, mxu0Compute, mxu1Compute, vpuSlots);
    for (size_t i = 0; i < commands.size(); ++i) {
      covered &= llvm::StringRef(commands[i].block).split('.').first == engine;
      covered &= composedOperations.count(commands[i].block);
      for (size_t j = 0; j < i; ++j)
        if (commands[i].issue - commands[j].issue < commands[j].firstFreeAge)
          covered &= scope->second.count(relationClass(actualRelation(commands[j], commands[i])));
    }
    if (covered) return true;
  }
  return false;
}

llvm::Expected<ModelOverlapFacts> ModelOverlapFacts::parseDerivedFacts(
    const Object &root, llvm::StringRef expectedSha256, const ModelTimingFacts &operations) {
  if (!keys(root, {"schema", "age_convention", "per_op_sha256", "model", "derivation", "validation",
                   "pairs", "composition"}) || root.getString("age_convention") != "issue_edge_zero" ||
      root.getString("per_op_sha256") != operations.factsSha256())
    return invalid("source-derived facts do not bind selected operation facts and ages");
  const Object *model = root.getObject("model"), *derivation = root.getObject("derivation");
  const Object *validation = root.getObject("validation"), *composition = root.getObject("composition");
  const auto &identity = operations.modelProvenance();
  if (!model || !derivation || !validation || !composition)
    return invalid("source-derived facts require model, derivation, validation and composition");
  for (const char *key : {"revision", "source_sha256", "numerical_data_sha256", "configuration_sha256"}) {
    auto found = identity.find(key);
    if (found == identity.end() || model->getString(key) != found->second)
      return invalid(std::string("incompatible source-derived model.") + key);
  }
  auto expectedConfiguration = llvm::json::parse(identity.at("configuration_json"));
  if (!expectedConfiguration) return invalid(llvm::toString(expectedConfiguration.takeError()));
  const Object *configuration = model->getObject("configuration");
  if (!configuration || llvm::json::Value(Object(*configuration)) != *expectedConfiguration)
    return invalid("incompatible source-derived explicit configuration");
  for (const auto &[key, value] : *model) {
    if (key == "configuration") continue;
    auto text = value.getAsString();
    auto found = identity.find(key.str());
    if (!text || found == identity.end() || *text != found->second)
      return invalid("source-derived model differs from selected operation provenance");
  }
  const Object *contract = derivation->getObject("source_contract");
  const Object *capacity = contract ? contract->getObject("capacity") : nullptr;
  auto boundedCapacity = [&](llvm::StringRef key, int maximum) {
    auto value = capacity ? capacity->getInteger(key) : std::nullopt;
    return value && *value >= 1 && *value <= maximum;
  };
  auto toolHash = derivation->getString("tool_sha256"), contractHash = derivation->getString("source_contract_sha256");
  if (!keys(*derivation, {"contract", "tool_sha256", "source_contract_sha256", "source_contract"}) ||
      derivation->getString("contract") != "source-resource-occupancy-v1" ||
      !toolHash || !hashSyntax(*toolHash) || !contractHash || !hashSyntax(*contractHash) || !contract ||
      modelSha256(canonicalJson(llvm::json::Value(Object(*contract)))) != *contractHash ||
      !capacity || !keys(*capacity, {"sequencer_cycles", "mxu0_compute", "mxu1_compute", "vpu_slots"}) ||
      capacity->getInteger("sequencer_cycles") != 32 || !boundedCapacity("mxu0_compute", 64) ||
      !boundedCapacity("mxu1_compute", 64) || !boundedCapacity("vpu_slots", 3))
    return invalid("source contract is missing, changed or incompatible with structural capacity");
  if (const auto *admission = operations.engineAdmission()) {
    if (capacity->getInteger("vpu_slots") != admission->vpuCapacity)
      return invalid("engine admission capacity differs from authoritative overlap contract");
    const Object *sources = contract->getObject("source_sha256"), *vpu = contract->getObject("vpu");
    const Object *scalar = contract->getObject("scalar_lsu");
    if (!sources || !vpu || !scalar || scalar->getInteger("load_response_gap") != admission->scalarLoadGap)
      return invalid("engine admission differs from authoritative overlap source predicates");
    for (const char *name : {"mxu.py", "vpu.py", "bank_conflict.py", "lsu.py", "xlu.py"}) {
      auto expected = admission->sourceHashes.find(name);
      if (expected == admission->sourceHashes.end() || sources->getString(name) != expected->second)
        return invalid("engine admission source differs from authoritative overlap contract");
    }
    auto names = [](const Array *array) -> std::optional<std::set<std::string>> {
      if (!array) return std::nullopt;
      std::set<std::string> result;
      for (const auto &value : *array) {
        auto name = value.getAsString();
        if (!name || !result.insert("vpu." + name->str()).second) return std::nullopt;
      }
      return result;
    };
    auto doubles = names(vpu->getArray("double"));
    if (!doubles || *doubles != admission->vpuDouble)
      return invalid("engine admission double-input predicates differ from authoritative overlap contract");
    const Array *groups = vpu->getArray("logic_groups");
    std::vector<std::set<std::string>> actual;
    if (!groups) return invalid("missing authoritative VPU logic groups");
    for (const auto &group : *groups) {
      auto members = names(group.getAsArray());
      if (!members) return invalid("invalid authoritative VPU logic group");
      actual.push_back(std::move(*members));
    }
    if (actual != admission->vpuLogicGroups)
      return invalid("engine admission logic groups differ from authoritative overlap contract");
  }
  Object semanticContract(*contract);
  semanticContract.erase("producer_sources");
  Object semantic{{"schema", *root.get("schema")}, {"age_convention", *root.get("age_convention")},
                  {"contract", *derivation->get("contract")},
                  {"source_contract", std::move(semanticContract)},
                  {"pairs", root.get("pairs") ? *root.get("pairs") : llvm::json::Value(nullptr)},
                  {"composition", llvm::json::Value(Object(*composition))}};
  if (operations.overlapRulesSha256().empty() ? !operations.allowsUnboundTestOverlap() :
      modelSha256(canonicalJson(llvm::json::Value(std::move(semantic)))) != operations.overlapRulesSha256())
    return invalid("overlap rules differ from the selected derived semantic payload");
  auto reportHash = validation->getString("report_sha256");
  // This is a bounded native gate for the source contract, not an observation
  // claim for every generated pair or a claim about RTL execution.
  const Object *engineRuns = validation->getObject("engines");
  auto positive = [](const Object &object, llvm::StringRef key) {
    auto value = object.getInteger(key);
    return value && *value > 0 && *value <= INT_MAX;
  };
  if (!keys(*validation, {"passed", "report_sha256", "sample_policy", "positive_runs", "negative_runs", "negative_rejections",
                          "numerical_checks", "engines"}) || validation->getBoolean("passed") != true ||
      validation->getString("sample_policy") != "source-mechanism-boundaries-v2" ||
      !reportHash || !hashSyntax(*reportHash) || !engineRuns || engineRuns->empty() ||
      !positive(*validation, "positive_runs") || !positive(*validation, "negative_runs") ||
      !positive(*validation, "negative_rejections") ||
      !positive(*validation, "numerical_checks"))
    return invalid("source-derived facts require successful bounded native numerical validation");
  std::map<std::string, int64_t> totals;
  std::set<std::string> validated;
  for (const auto &[engine, value] : *engineRuns) {
    const Object *runs = value.getAsObject();
    if ((engine != "mxu0" && engine != "mxu1" && engine != "vpu") ||
        !runs || !keys(*runs, {"positive_runs", "negative_runs", "negative_rejections", "numerical_checks",
                               "chain_positive_runs", "chain_negative_runs"}))
      return invalid("unsupported source validation engine or counter");
    for (const char *key : {"positive_runs", "negative_runs", "negative_rejections", "numerical_checks"}) {
      if (!positive(*runs, key)) return invalid("source validation engine lacks positive, negative or numerical probes");
      totals[key] += *runs->getInteger(key);
    }
    for (auto [chainKey, allKey] : {std::pair<const char *, const char *>{"chain_positive_runs", "positive_runs"},
                                    {"chain_negative_runs", "negative_rejections"}})
      if (runs->get(chainKey) && (!runs->getInteger(chainKey) || *runs->getInteger(chainKey) < 0 ||
                                 runs->getInteger(chainKey) > runs->getInteger(allKey)))
        return invalid("source chain validation counters exceed native engine probes");
    if (runs->getInteger("negative_rejections") > runs->getInteger("negative_runs"))
      return invalid("source native rejections exceed attempted controls");
    validated.insert(engine.str());
  }
  for (const auto &[key, total] : totals)
    if (*validation->getInteger(key) != total)
      return invalid("source validation totals do not cover engine probes");
  ModelOverlapFacts result;
  result.sequencerCycles = *capacity->getInteger("sequencer_cycles");
  result.mxu0Compute = *capacity->getInteger("mxu0_compute");
  result.mxu1Compute = *capacity->getInteger("mxu1_compute");
  result.vpuSlots = *capacity->getInteger("vpu_slots");
  const Array *pairs = root.getArray("pairs");
  if (!pairs || pairs->empty()) return invalid("source-derived facts require nonempty pair coverage");
  std::set<std::tuple<std::string, std::string, Relation>> seen;
  for (const auto &value : *pairs) {
    const Object *record = value.getAsObject();
    auto first = record ? record->getString("first") : std::nullopt;
    auto second = record ? record->getString("second") : std::nullopt;
    auto minimum = record ? record->getInteger("minimum") : std::nullopt;
    auto high = record ? record->getInteger("high") : std::nullopt;
    const auto *a = first ? operations.block(*first) : nullptr, *b = second ? operations.block(*second) : nullptr;
    Relation rel;
    if (!record || !keys(*record, {"first", "second", "relation", "minimum", "high"}) ||
        !a || !b || a->engine == "scalar_lsu" || b->engine == "scalar_lsu" ||
        a->engine != b->engine ||
        !validated.count(a->engine) ||
        !minimum || !high || *minimum < 1 || *minimum > *high || *high != a->firstFreeAge ||
        !relation(record->getObject("relation"), rel) ||
        !seen.emplace(first->str(), second->str(), rel).second)
      return invalid("invalid, duplicate, unvalidated or unbounded source-derived pair");
    for (const auto &[key, value] : rel)
      if (key != "mreg_banks" && key != "accumulator" && key != "weight_slot")
        return invalid("source-derived vector pair carries unsupported semantic qualifiers");
    result.pairIndices[{first->str(), second->str()}].push_back(result.pairs.size());
    result.pairs.push_back({first->str(), second->str(), std::move(rel), int(*minimum), int(*high)});
  }
  const Object *engines = composition->getObject("engines");
  const Array *classes = composition->getArray("classes");
  if (!keys(*composition, {"engines", "classes"}) || !engines || !classes)
    return invalid("invalid source-derived composition descriptor");
  std::set<std::string> composedEngines;
  for (const auto &[engine, value] : *engines) {
    const Array *names = value.getAsArray();
    if ((engine != "mxu0" && engine != "mxu1" && engine != "vpu") ||
        !validated.count(engine.str()) || !names || names->empty() ||
        !positive(*engineRuns->getObject(engine), "chain_positive_runs") ||
        !positive(*engineRuns->getObject(engine), "chain_negative_runs"))
      return invalid("source composition lacks a supported validated operation domain");
    composedEngines.insert(engine.str());
    for (const auto &entry : *names) {
      auto name = entry.getAsString();
      const auto *block = name ? operations.block(*name) : nullptr;
      if (!block || block->engine != engine || !result.composedOperations.insert(name->str()).second)
        return invalid("invalid or duplicate source composition operation");
    }
  }
  for (const auto &entry : *classes) {
    const Object *category = entry.getAsObject();
    auto engine = category ? category->getString("engine") : std::nullopt;
    const Object *descriptor = category ? category->getObject("relation") : nullptr;
    Relation rel;
    if (!category || !keys(*category, {"engine", "relation"}) || !engine ||
        !composedEngines.count(engine->str()) || !descriptor ||
        !keys(*descriptor, {"mreg_banks", "accumulator", "weight_slot"}))
      return invalid("unsupported source composition class");
    for (const auto &[key, value] : *descriptor) {
      auto text = value.getAsString();
      if (!text || (key == "mreg_banks" ?
          *text != "disjoint" && *text != "register_alias" && *text != "bank_alias" && *text != "mixed_alias" :
          *text != "same" && *text != "other")) return invalid("invalid source composition relation class");
      rel[key.str()] = text->str();
    }
    if (!rel.count("mreg_banks") || !std::any_of(result.pairs.begin(), result.pairs.end(), [&](const Pair &pair) {
          return operations.block(pair.first)->engine == *engine && result.composedOperations.count(pair.first) &&
              result.composedOperations.count(pair.second) && relationClass(pair.relation) == rel;
        }) || !result.compositionClasses[engine->str()].insert(rel).second)
      return invalid("duplicate source composition class or class outside exact pair domain");
  }
  for (const auto &engine : composedEngines)
    if (result.compositionClasses[engine].empty()) return invalid("source composition engine lacks validated classes");
  result.structuralCapacity = true;
  result.provenance = {{"pairs_sha256", expectedSha256.str()}, {"derived_sha256", expectedSha256.str()},
      {"derived_policy", "source-resource-occupancy-v1"}, {"derived_per_op_sha256", operations.factsSha256()},
      {"derived_tool_sha256", toolHash->str()}, {"derived_source_contract_sha256", contractHash->str()},
      {"derived_validation_sha256", reportHash->str()}, {"derived_sample_policy", "source-mechanism-boundaries-v2"}};
  for (const char *key : {"revision", "source_sha256", "numerical_data_sha256", "configuration_sha256"})
    result.provenance[std::string("derived_") + key] = identity.at(key);
  return result;
}

llvm::Expected<ModelOverlapFacts> mlir::atlas::timing::parseModelOverlapFacts(
    llvm::StringRef bytes, llvm::StringRef expectedSha256, const ModelTimingFacts &operations) {
  if (!hashSyntax(expectedSha256) || modelSha256(bytes) != expectedSha256)
    return invalid("selected source-derived facts hash mismatch");
  auto decoded = llvm::json::parse(bytes);
  if (!decoded) return invalid(llvm::toString(decoded.takeError()));
  const Object *root = decoded->getAsObject();
  if (!root || root->getString("schema") != "atlas.model_overlap_derived.v1")
    return invalid("only source-derived overlap facts are supported");
  return ModelOverlapFacts::parseDerivedFacts(*root, expectedSha256, operations);
}

llvm::Expected<ModelOverlapFacts> mlir::atlas::timing::loadModelOverlapFacts(
    llvm::StringRef path, llvm::StringRef expectedSha256, const ModelTimingFacts &operations) {
  auto buffer = llvm::MemoryBuffer::getFile(path);
  if (!buffer) return invalid("cannot read selected source-derived facts: " + path.str());
  llvm::StringRef bytes = (*buffer)->getBuffer();
  if (!hashSyntax(expectedSha256) || modelSha256(bytes) != expectedSha256)
    return invalid("selected source-derived facts hash mismatch");
  // Recheck file contents on every load; only parsing and proof validation are cached.
  static thread_local std::string cachedHash, cachedOperations;
  static thread_local std::optional<ModelOverlapFacts> cached;
  if (cached && cachedHash == expectedSha256 && cachedOperations == operations.factsSha256())
    return *cached;
  auto parsed = parseModelOverlapFacts(bytes, expectedSha256, operations);
  if (!parsed)
    return parsed.takeError();
  cachedHash = expectedSha256.str();
  cachedOperations = operations.factsSha256();
  cached = *parsed;
  return std::move(*parsed);
}
