#include "Atlas/AtlasModelTiming.h"
#include "Atlas/AtlasModelVocabulary.h"
#include "Atlas/AtlasModelPairs.h"
#include "Atlas/AtlasGeneratedArtifact.h"
#include "Atlas/AtlasStream.h"
#include "Atlas/AtlasTarget.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SHA256.h"
#include <algorithm>
#include <climits>
#include <set>

using namespace mlir;
using namespace mlir::atlas;
using namespace mlir::atlas::timing;

namespace {
using Object = llvm::json::Object;
using Array = llvm::json::Array;

constexpr int kLinesPerBank = kVmemBankBytes / kLineBytes;

llvm::Error factsError(const std::string &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), "model timing facts: " + message);
}

bool supportedConfiguration(const Object &cfg) {
  const Object *arch = cfg.getObject("arch_state_config"), *units = cfg.getObject("execution_units");
  const Object *memory = cfg.getObject("dma_memory_params");
  const auto *memoryLatency = cfg.get("dma_memory_latency_cycles"), *memoryBeat = cfg.get("dma_memory_cycles_per_beat");
  if (!arch || !units || cfg.getBoolean("ignore_runtime_errors") != false ||
      arch->getString("numerics") != "rtl" || arch->getBoolean("randomize_init") != false)
    return false;
  // The DMA estimate is independently qualified for this profile, not derived from isolated engine facts.
  if (cfg.getString("dma_memory_backend") != "curve" || !memory || memory->size() != 1 ||
      memory->getString("curves") != "ee290sim_vcs_probe.json" || !memoryLatency || !memoryBeat ||
      memoryLatency->kind() != llvm::json::Value::Null || memoryBeat->kind() != llvm::json::Value::Null)
    return false;
  for (auto [key, value] : std::initializer_list<std::pair<const char *, int64_t>>{
           {"mrf_depth", 32}, {"mrf_width", 32}, {"num_m_registers", 64}, {"num_wb_registers", 2},
           {"wb_width", 1024}, {"num_x_registers", 32}, {"num_e_registers", 32}, {"num_csrs", 4096},
           {"vmem_size", kAtlasVmemBytes}, {"dram_base", 0x80000000LL}, {"dram_size", 0x1000000000LL}})
    if (arch->getInteger(key) != value) return false;
  for (auto [key, value] : std::initializer_list<std::pair<const char *, int64_t>>{
           {"fetch_width", 1}, {"dma_num_channels", 8}, {"dma_beat_bytes", 32}, {"dma_max_transfer_bytes", 4096},
           {"dma_max_in_flight", 64}, {"vmem_bank_bytes", kVmemBankBytes}, {"offchip_link_width_bits", 32},
           {"offchip_link_core_cycles_per_beat", 2}, {"dma_offchip_command_words", 2},
           {"vmem_bus_width_bits", 512}, {"vmem_bus_core_cycles_per_beat", 1}, {"vmem_bytes_per_cycle", 64}})
    if (cfg.getInteger(key) != value) return false;
  for (auto [key, value] : std::initializer_list<std::pair<const char *, const char *>>{
           {"Scalar0", "ScalarExecutionUnit"}, {"Matrix0", "MatrixExecutionUnitSystolic"},
           {"Matrix1", "MatrixExecutionUnitInner"}, {"Vector0", "VectorExecutionUnit"},
           {"XLU", "CrossLaneExecutionUnit"}, {"DMA0", "DmaExecutionUnit"}, {"LSU", "LoadStoreUnit"}})
    if (units->getString(key) != value) return false;
  return cfg.getString("isa") == "npu_model.isa.IsaSpec";
}

using model_schema::RegisterEvent;
using model_schema::vpuBlocks;
using model_schema::vpuEvents;
using model_schema::mxuBlocks;

struct Fill {
  Footprint &f;
  std::string name, error;
  const ModelTimingBlock *block = nullptr;
  int latency = 0;
  std::set<std::string> used;
  std::set<std::string> usedReleases;

  Fill(Footprint &footprint, const std::string &blockName, const std::map<std::string, ModelTimingBlock> &blocks)
      : f(footprint), name(blockName) {
    auto it = blocks.find(name);
    if (it == blocks.end()) fail("facts block is missing");
    else { block = &it->second; latency = block->readLatency; }
  }
  void fail(const std::string &why) { if (error.empty()) error = name + ": " + why; }
  const ModelTimingGroup *group(const char *key) {
    used.insert(key);
    if (!block) return nullptr;
    auto it = block->events.find(key);
    if (it == block->events.end() || it->second.count <= 0) {
      fail(std::string("event group ") + key + " is missing"); return nullptr;
    }
    return &it->second;
  }
  // Element i of pass p is touched at first_age + (p*elements+i)*step.
  void access(const char *key, Res res, bool write, int first, int elements, int passes = 1) {
    const ModelTimingGroup *g = group(key);
    if (!g) return;
    int step = g->step;
    if (g->count != elements * passes || step <= 0 || g->lastAge != g->firstAge + (g->count - 1) * step) {
      fail(std::string("event group ") + key + " does not sweep " +
           std::to_string(passes) + "x" + std::to_string(elements) + " elements evenly");
      return;
    }
    for (int p = 0; p < passes; ++p)
      f.accesses.push_back({res, write, first, elements, g->firstAge + p * elements * step, step});
  }
  int first(const char *key) { const ModelTimingGroup *g = group(key); return g ? g->firstAge : 0; }
  int last(const char *key) { const ModelTimingGroup *g = group(key); return g ? g->lastAge : 0; }
  bool explicitReleases() const { return block && !block->releases.empty(); }
  const ModelReservationRelease *reservation(const char *key) {
    if (!explicitReleases()) return nullptr;
    usedReleases.insert(key);
    auto found = block->releases.find(key);
    if (found == block->releases.end()) {
      fail(std::string("missing reservation release ") + key);
      return nullptr;
    }
    return &found->second;
  }
  int releaseEnd(const char *key, int legacy) {
    const auto *release = reservation(key);
    return release ? release->releaseAge : legacy;
  }
  int mreg(const char *key, int reg, bool write, int registers = 1, int passes = 1, bool mergeRelease = true) {
    access(key, Res::MReg, write, reg * 32, registers * 32, passes);
    auto &list = write ? f.mregWrites : f.mregReads;
    for (int i = 0; i < registers; ++i) list.push_back(reg + i);
    int end = releaseEnd(write ? "mreg_write" : "mreg_read", last(key) + (write ? 0 : latency));
    if (end < last(key)) fail("MREG reservation releases before its last access");
    int &release = write ? f.writeRelease : f.readRelease;
    release = mergeRelease ? std::max(release, end) : end;
    return end;
  }
  int busyEnd() { return releaseEnd("unit", block ? block->firstFreeAge - 1 : 0); }
  int nextIssue() { return block ? block->nextIssueAge : 0; }
  void bank(std::optional<int> line, const char *key) {
    const auto *release = reservation(key);
    int from = release ? release->firstAge : first(key), to = release ? release->releaseAge : last(key);
    if (from > first(key) || to < last(key)) fail("VMEM port window does not cover its accesses");
    if (line) f.holds.push_back({Unit::VmemBank, *line / kLinesPerBank, from, to});
    else
      for (int bank = 0; bank < kVmemBanks; ++bank)
        f.holds.push_back({Unit::VmemBank, bank, from, to});
  }
  void hold(std::initializer_list<Unit> units, int from, int to) {
    if (from <= to)
      for (Unit unit : units) f.holds.push_back({unit, 0, from, to});
  }
  void vpu(int busy) { hold({Unit::Xlu}, 0, busy); }
  Footprint done(int busy) {
    if (block)
      for (const auto &[key, event] : block->events)
        if (event.count > 0 && !used.count(key)) fail("observed event group " + key + " is outside the mapped domain");
    if (block)
      for (const auto &[key, release] : block->releases) {
        if (!usedReleases.count(key)) fail("reservation release " + key + " is outside the mapped domain");
        if (key == "mreg_read" || key == "mreg_write") {
          const auto &regs = key == "mreg_read" ? f.mregReads : f.mregWrites;
          if (std::set<int>(regs.begin(), regs.end()).size() != release.banks.size())
            fail("MREG reservation bank count differs from operand roles");
        }
      }
    if (!error.empty()) {
      f.error = kModelTimingProviderId.str() + ": " + error;
      return f;
    }
    f.doneAge = std::max({busy, f.readRelease, f.writeRelease});
    for (const Access &a : f.accesses) if (!a.atCompletion) f.doneAge = std::max(f.doneAge, a.lastAge());
    for (const Hold &h : f.holds) f.doneAge = std::max(f.doneAge, h.to);
    return f;
  }
};

std::string mxuBlockName(const OpInfo &op) {
  auto it = mxuBlocks().find(op.opClass);
  if (it == mxuBlocks().end() || op.mxu < 0 || op.mxu > 1) return {};
  return "mxu" + std::to_string(op.mxu) + "." + it->second;
}

Footprint mxuFootprint(const Instr &in, const std::map<std::string, ModelTimingBlock> &blocks) {
  Footprint f;
  const OpInfo &op = *in.op;
  std::string blockName = mxuBlockName(op);
  if (blockName.empty()) {
    f.error = kModelTimingProviderId.str() + ": unsupported MXU operation";
    return f;
  }
  const int m = op.mxu;
  Fill fill(f, blockName, blocks);
  int busy = fill.busyEnd();
  auto track = [&](int end) { busy = std::max(busy, end); };
  auto mregRows = [&](const char *key, int reg, bool write) {
    if (reg < 0 || reg > 63) fill.fail("register operand out of range");
    track(fill.mreg(key, reg, write));
  };
  auto slotRows = [&](const char *key, Res res, int slot, bool write) {
    if (slot < 0 || slot > 1) fill.fail("accumulator and weight slots are 0 or 1");
    fill.access(key, res, write, (m * 2 + slot) * 32, 32);
    if (write) f.writeRelease = std::max(f.writeRelease, fill.last(key));
    track(fill.last(key));
  };
  if (in.imm != 0) fill.fail("unsupported MXU encoding");
  const bool weight = op.opClass == OpClass::WeightPush;
  const bool paired = op.opClass == OpClass::AccPushBf16 || op.opClass == OpClass::PopBf16;
  const bool push = weight || op.opClass == OpClass::AccPushFp8 || op.opClass == OpClass::AccPushBf16;
  const bool pop = op.opClass == OpClass::PopFp8 || op.opClass == OpClass::PopBf16;
  if (push && in.rs2 != 0) fill.fail("unsupported MXU encoding");
  if (paired && ((push ? in.rs1 : in.rd) & 1)) fill.fail("BF16 operands must name even register pairs");
  if (op.opClass == OpClass::PopBf16 && in.rs1 != 0) fill.fail("unsupported MXU encoding");
  for (const auto &event : model_schema::mxuEvents().at(op.opClass)) {
    int operand = in.*event.operand + event.offset;
    if (event.resource == Res::MReg) mregRows(event.key, operand, event.write);
    else if (event.resource == Res::EReg) {
      if (operand < 0 || operand > 31) fill.fail("scale register out of range");
      f.accesses.push_back({Res::EReg, false, operand, 1, 0, 1});
    } else slotRows(event.key, event.resource, operand, event.write);
  }
  track(fill.last(push || pop ? "data_busy" : "compute_busy"));
  if (!push && !pop) {
    if (in.rs2 < 0 || in.rs2 > 1) fill.fail("accumulator and weight slots are 0 or 1");
    for (int age : {0, busy})
      f.accesses.push_back({Res::Weight, false, (m * 2 + in.rs2) * 32, 32, age, 0});
  }
  f.holds.push_back({Unit::MxuCompute, m, 0, busy});
  return fill.done(busy);
}

} // namespace

bool mlir::atlas::timing::hashSyntax(llvm::StringRef value) {
  return value.size() == 64 && std::all_of(value.begin(), value.end(), [](char c) {
    return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
  });
}

std::string mlir::atlas::timing::canonicalJson(const llvm::json::Value &value) {
  std::string bytes;
  llvm::raw_string_ostream out(bytes);
  if (const Object *object = value.getAsObject()) {
    std::vector<std::string> keys;
    for (const auto &[key, entry] : *object) keys.push_back(key.str());
    std::sort(keys.begin(), keys.end());
    out << '{';
    for (size_t i = 0; i < keys.size(); ++i) {
      if (i) out << ',';
      out << llvm::json::Value(keys[i]) << ':' << canonicalJson(*object->get(keys[i]));
    }
    out << '}';
  } else if (const Array *array = value.getAsArray()) {
    out << '[';
    for (size_t i = 0; i < array->size(); ++i) {
      if (i) out << ',';
      out << canonicalJson((*array)[i]);
    }
    out << ']';
  } else {
    out << value;
  }
  return bytes;
}

llvm::Expected<ModelTimingFacts> mlir::atlas::timing::parseModelTimingFacts(
    llvm::StringRef bytes, llvm::StringRef expectedSha256, bool dmaWait) {
  if (!hashSyntax(expectedSha256)) return factsError("facts SHA-256 is required");
  auto digest = llvm::SHA256::hash(llvm::ArrayRef<uint8_t>(
      reinterpret_cast<const uint8_t *>(bytes.data()), bytes.size()));
  if (llvm::toHex(llvm::ArrayRef<uint8_t>(digest), true) != expectedSha256)
    return factsError("selected facts hash mismatch");
  auto parsed = llvm::json::parse(bytes);
  if (!parsed) return factsError(llvm::toString(parsed.takeError()));
  const Object *root = parsed->getAsObject();
  if (!root || root->getString("schema") != "atlas.model_timing.v1")
    return factsError("expected schema atlas.model_timing.v1");
  if (root->get("vocabulary_sha256") && root->getString("vocabulary_sha256") != model_schema::vocabularySha256())
    return factsError("facts vocabulary differs from the compiled operand and relation schema");
  for (const char *key : {"hw_ir", "assertions", "op_pairs", "pairs", "chains", "capacity"})
    if (root->get(key)) return factsError("per-operation model facts cannot carry RTL or overlap claims");
  const Object *model = root->getObject("model");
  if (!model) return factsError("missing model provenance");
  ModelTimingFacts result;
  for (const char *key : {"revision", "source_sha256", "configuration_sha256", "observer_sha256",
                          "exporter_sha256", "input_sha256", "numerical_data_sha256"}) {
    auto value = model->getString(key);
    bool valid = value && (llvm::StringRef(key) == "revision" ?
        value->size() == 40 && std::all_of(value->begin(), value->end(), [](char c) {
          return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
        }) : hashSyntax(*value));
    if (!valid) return factsError(std::string("missing or malformed model.") + key);
  }
  const Object *configuration = model->getObject("configuration");
  auto configurationJson = model->getString("configuration_json");
  if (!configuration || !configurationJson || !supportedConfiguration(*configuration))
    return factsError("unsupported or missing explicit model configuration profile");
  auto decodedConfiguration = llvm::json::parse(*configurationJson);
  if (!decodedConfiguration) return factsError(llvm::toString(decodedConfiguration.takeError()));
  if (*decodedConfiguration != llvm::json::Value(Object(*configuration)) ||
      *configurationJson != canonicalJson(llvm::json::Value(Object(*configuration))))
    return factsError("configuration_json is not the canonical explicit configuration");
  auto configurationHash = llvm::SHA256::hash(llvm::ArrayRef<uint8_t>(
      reinterpret_cast<const uint8_t *>(configurationJson->data()), configurationJson->size()));
  if (llvm::toHex(llvm::ArrayRef<uint8_t>(configurationHash), true) != *model->getString("configuration_sha256"))
    return factsError("configuration content hash mismatch");
  for (const auto &[key, value] : *model) {
    if (key == "configuration") continue;
    auto text = value.getAsString();
    if (!text || (llvm::StringRef(key).ends_with("sha256") && !hashSyntax(*text)))
      return factsError("invalid model provenance field " + key.str());
    result.provenance[key.str()] = text->str();
  }
  const Object *geometry = root->getObject("geometry");
  if (!geometry || geometry->getInteger("rows") != 32 || geometry->getInteger("columns") != 32 ||
      geometry->getInteger("mreg_count") != 64)
    return factsError("only 32x32 geometry with 64 MREGs is supported");
  auto accepted = [](const Object *object) {
    return object && object->getBoolean("accepted") == true && object->getBoolean("drained") == true;
  };
  if (root->getString("age_convention") != "issue_edge_zero" || !accepted(root->getObject("observation")))
    return factsError("requires accepted, drained observations with issue_edge_zero ages");
  const Array *timing = root->getArray("op_timing");
  if (!timing || timing->empty()) return factsError("missing op_timing coverage");
  const Object *releases = root->getObject("reservation_releases");
  if (root->get("reservation_releases") && (!releases || releases->size() != timing->size()))
    return factsError("reservation release coverage differs from op_timing");
  for (const auto &value : *timing) {
    const Object *entry = value.getAsObject();
    auto name = entry ? entry->getString("name") : std::nullopt;
    auto engine = entry ? entry->getString("engine") : std::nullopt;
    if (!name || !engine || entry->get("assertions") || entry->getBoolean("resolved") != true ||
        !accepted(entry->getObject("observation")))
      return factsError("invalid or unresolved model timing block");
    bool known = false;
    for (const auto &[mnemonic, block] : vpuBlocks()) known |= *name == block && *engine == "vpu";
    for (int unit = 0; unit < 2; ++unit)
      for (const auto &[kind, block] : mxuBlocks())
        known |= *name == "mxu" + std::to_string(unit) + "." + block && *engine == "mxu" + std::to_string(unit);
    for (const char *block : {"vlsu.vload", "vlsu.vstore", "xlu.transpose", "scalar_lsu.load",
                             "scalar_lsu.store", "scalar_lsu.scale_load"})
      known |= *name == block && *engine == llvm::StringRef(block).split('.').first;
    if (!known) return factsError("unknown operation or engine " + name->str());
    auto free = entry->getInteger("first_free_age"), next = entry->getInteger("next_issue_age");
    if (!free || *free < 1 || *free > 4096 || next != free)
      return factsError("requires bounded serialized next_issue_age=first_free_age");
    const Object *assumptions = entry->getObject("assumptions");
    if (!assumptions || assumptions->getInteger("scratchpad_read_latency") != 1)
      return factsError("only scratchpad_read_latency=1 is supported");
    ModelTimingBlock block{engine->str(), {}, int(*free), int(*next), 1};
    const Object *events = entry->getObject("events");
    if (!events || events->empty()) return factsError("model timing block needs observed events");
    for (const auto &[key, value] : *events) {
      const Object *group = value.getAsObject();
      if (!group) return factsError("invalid event group " + key.str());
      auto first = group->getInteger("first_age"), last = group->getInteger("last_age");
      auto count = group->getInteger("count"), step = group->getInteger("step");
      if (!first || !last || !count || !step || *first < 0 || *last < 0 || *count < 1 || *step < 1 ||
          *first > 4096 || *last > 4096 || *count > 4096 || *step > 4096 ||
          *first + (*count - 1) * *step != *last || *last >= *free)
        return factsError("inconsistent observed event sweep or first-free bound: " + key.str());
      block.events[key.str()] = {int(*first), int(*last), int(*count), int(*step)};
    }
    if (releases) {
      const Object *resources = releases->getObject(*name);
      if (!resources || !resources->getObject("unit")) return factsError("missing unit reservation release");
      for (const auto &[key, value] : *resources) {
        bool unit = key == "unit", mreg = key == "mreg_read" || key == "mreg_write";
        if (!unit && !mreg && key != "vmem_read" && key != "vmem_write")
          return factsError("unknown reservation resource " + key.str());
        const Object *release = value.getAsObject();
        if (!release || release->size() != size_t(unit ? 3 : 4)) return factsError("invalid reservation release");
        auto first = release->getInteger("first_age"), last = release->getInteger("release_age");
        auto available = release->getInteger("available_age");
        if (!first || !last || !available || *first < 0 || *last < *first || *last >= *free ||
            *available != *last + 1 || ((unit || mreg) && *first != 0) || (unit && *available != *free))
          return factsError("reservation release is outside its isolated lifetime");
        ModelReservationRelease resource{int(*first), int(*last), int(*available), {}};
        if (!unit) {
          const Array *banks = release->getArray("banks");
          if (!banks || banks->empty()) return factsError("reservation needs explicit probe banks");
          int previous = -1;
          for (const auto &bank : *banks) {
            auto index = bank.getAsInteger();
            if (!index || *index <= previous || *index >= (mreg ? 64 : kVmemBanks))
              return factsError("reservation banks must be sorted, unique and in range");
            resource.banks.push_back(previous = int(*index));
          }
          // Evidence uses disjoint canonical probes; actual instruction operands are remapped by Fill.
          int base = key == "mreg_write" ? 4 : 0;
          for (size_t i = 0; i < resource.banks.size(); ++i)
            if (resource.banks[i] != base + int(i) || (!mreg && i != 0))
              return factsError("reservation banks differ from the canonical disjoint probe");
        }
        block.releases[key.str()] = std::move(resource);
      }
    }
    if (!result.blocks.emplace(name->str(), std::move(block)).second)
      return factsError("duplicate operation " + name->str());
  }
  result.sha256 = expectedSha256.str();
  result.dma = dmaWait;
  return result;
}

llvm::Expected<ModelTimingFacts> mlir::atlas::timing::loadModelTimingFacts(
    llvm::StringRef path, llvm::StringRef expectedSha256, bool dmaWait) {
  auto buffer = llvm::MemoryBuffer::getFile(path);
  if (!buffer) return factsError("cannot read " + path.str());
  return parseModelTimingFacts((*buffer)->getBuffer(), expectedSha256, dmaWait);
}

Footprint ModelTimingFacts::resolve(const Instr &in, const RegValues &regs) const {
  Footprint f;
  auto reject = [&](const char *why) {
    Footprint rejected;
    rejected.error = kModelTimingProviderId.str() + ": " + why;
    return rejected;
  };
  if (!in.op) return reject("invalid instruction");
  const OpInfo &op = *in.op;
  const std::string &name = op.name;
  auto xreg = [](int r) { return r >= 0 && r <= 31; };
  auto mreg = [](int r) { return r >= 0 && r <= 63; };
  auto x = [&](int reg, bool write) {
    if (reg) f.accesses.push_back({Res::XReg, write, reg, 1, 0, 1});
  };
  if (in.release && op.opClass != OpClass::Csr) return reject("unsupported completion annotation");

  if (op.opClass == OpClass::Alu && name != "auipc") {
    if (!xreg(in.rd) || !xreg(in.rs1) || !xreg(in.rs2)) return reject("unsupported scalar operands");
    if (hasOperand(op, "x1")) x(in.rs1, false);
    if (hasOperand(op, "x2")) x(in.rs2, false);
    x(in.rd, true);
    if (in.rd) f.holds.push_back({Unit::ScalarWriteback, 0, 0, 0});
    return f;
  }
  if (op.opClass == OpClass::ScaleImm) {
    if (!xreg(in.rd)) return reject("unsupported scale register");
    f.accesses.push_back({Res::EReg, true, in.rd, 1, 0, 1});
    f.holds.push_back({Unit::ScalarWriteback, 0, 0, 0});
    return f;
  }
  if (name == "delay" || name == "ecall") return f;
  // The verifier drains every engine before a successor starts, so redirect
  // latency need only cover the delay slot; operands are read at issue.
  if (op.opClass == OpClass::Branch) {
    if (!xreg(in.rs1) || !xreg(in.rs2)) return reject("unsupported branch operands");
    x(in.rs1, false);
    x(in.rs2, false);
    return f;
  }
  if (name == "jal") {
    if (in.rd != 0 || in.rs1 != 0) return reject("JAL must not link");
    return f;
  }
  if (name == "csrrw" && in.rd == 0 && in.imm == 0xC10 && xreg(in.rs1)) {
    x(in.rs1, false);
    return f;
  }

  if (op.engine == Engine::Dma) {
    if (!dma) return reject("DMA requires the dma=wait selection");
    if (op.channel < 0 || op.channel >= int(kAtlasDMAChannels)) return reject("unsupported DMA channel");
    if (op.opClass == OpClass::DmaWait) return f; // Pairing is checked by the stream verifier.
    if (op.opClass == OpClass::DmaConfig) {
      if (!xreg(in.rs1) || !regs[in.rs1] || *regs[in.rs1] > 31)
        return reject("DMA configuration requires a known 5-bit upper DRAM base");
      x(in.rs1, false);
      f.accesses.push_back({Res::DmaBase, true, 0, 1, 0, 1});
      return f;
    }
    if (op.opClass != OpClass::DmaLoad && op.opClass != OpClass::DmaStore) return reject("unsupported DMA operation");
    if (!xreg(in.rd) || !xreg(in.rs1) || !xreg(in.rs2) || !regs[in.rd] || !regs[in.rs1] || !regs[in.rs2])
      return reject("DMA requires known scalar VMEM, DRAM and size operands");
    const bool load = op.opClass == OpClass::DmaLoad;
    const uint64_t words = *regs[load ? in.rd : in.rs1], offset = *regs[load ? in.rs1 : in.rd], bytes = *regs[in.rs2];
    if (bytes < 32 || bytes > 4096 || bytes % 32 || words % 8 || words + bytes / 4 > uint64_t(kAtlasVmemBytes / 4) ||
        words / (kVmemBankBytes / 4) != (words + bytes / 4 - 1) / (kVmemBankBytes / 4) ||
        offset % 32 || offset + bytes > (uint64_t(1) << 32))
      return reject("DMA transfer is unaligned, out of range, oversized, or crosses a bank/address boundary");
    x(in.rd, false); x(in.rs1, false); x(in.rs2, false);
    f.dmaCycles = dmaTransferCycles(bytes, !load);
    f.accesses.push_back({Res::DmaBase, false, 0, 1, 0, 1});
    f.accesses.push_back({Res::Vmem, load, int(words / 8), int(bytes / 32), 0, 0, false, true});
    return f;
  }

  if (name == "lw" || name == "seld" || name == "sw") {
    const bool store = name == "sw", scale = name == "seld";
    if (!xreg(in.rs1) || !xreg(in.rs2) || !xreg(in.rd)) return reject("unsupported scalar memory operands");
    if ((store && in.rd != 0) || (!store && in.rs2 != 0)) return reject("unsupported scalar memory encoding");
    std::optional<uint32_t> address;
    if (regs[in.rs1]) address = (*regs[in.rs1] + uint32_t(signExtend(in.imm, 12))) & 0x1FFFFF;
    if (address && (*address % 4 || *address + 4 > uint32_t(kAtlasVmemBytes)))
      return reject("scalar word access is unaligned or outside VMEM");
    Fill fill(f, store ? "scalar_lsu.store" : scale ? "scalar_lsu.scale_load" : "scalar_lsu.load", blocks);
    std::optional<int> line = address ? std::optional<int>(*address / kLineBytes) : std::nullopt;
    x(in.rs1, false);
    if (store) {
      x(in.rs2, false);
      fill.access("vmem_write", Res::Vmem, true, line.value_or(0), 1);
      fill.bank(line, "vmem_write");
    } else {
      const char *write = scale ? "scale_write" : "load_write";
      fill.access("vmem_read", Res::Vmem, false, line.value_or(0), 1);
      if (scale || in.rd) fill.access(write, scale ? Res::EReg : Res::XReg, true, in.rd, 1);
      else fill.group(write);
      fill.bank(line, "vmem_read");
      f.holds.push_back({Unit::ScalarWriteback, 0, fill.first(write), fill.first(write)});
    }
    int busy = fill.busyEnd();
    f.holds.push_back({Unit::ScalarLoad, 0, 0, busy});
    for (Access &access : f.accesses)
      if (access.res == Res::Vmem) access.anywhere = !line;
    return fill.done(busy);
  }

  if (op.opClass == OpClass::VLoad || op.opClass == OpClass::VStore) {
    const bool load = op.opClass == OpClass::VLoad;
    if (!mreg(in.rd) || !xreg(in.rs1) || in.rs2 != 0) return reject("unsupported VLS operands");
    if (in.imm < -2048 || in.imm > 4095) return reject("VLS immediate is not a 12-bit encoding");
    std::optional<int> line;
    if (regs[in.rs1]) {
      uint32_t words = *regs[in.rs1] + uint32_t(signExtend(in.imm, 12) * 32);
      line = (words >> 3) & 0xffff;
    }
    if (line && (*line % 32 || *line + 32 > kAtlasVmemBytes / kLineBytes ||
                 *line / kLinesPerBank != (*line + 31) / kLinesPerBank))
      return reject("VLS effective tile is unaligned, out of range, or crosses a bank");
    Fill fill(f, load ? "vlsu.vload" : "vlsu.vstore", blocks);
    const char *vmem = load ? "vmem_read" : "vmem_write";
    const char *rows = load ? "mreg_write" : "mreg_read";
    x(in.rs1, false);
    fill.access(vmem, Res::Vmem, !load, line.value_or(0), 32);
    fill.mreg(rows, in.rd, load, 1, 1, /*mergeRelease=*/false);
    // lsu.py permits VLOAD writes during read reservations; observed rows still order WARs.
    f.writeDuringRead = load;
    int busy = fill.busyEnd();
    // Bank holds order VLS paths; MREG ports/releases order their overlap with VPU and XLU.
    fill.hold({load ? Unit::VloadPath : Unit::VstorePath}, 0, busy);
    fill.bank(line, vmem);
    if (!load) {
      f.writeRelease = fill.releaseEnd(vmem, fill.last(vmem));
      // Facts without reservation_releases release the VSTORE read with its VMEM write.
      if (!fill.explicitReleases()) f.readRelease = f.writeRelease;
    }
    // Unknown bases cover all VMEM/banks; ages still come only from the selected facts.
    for (Access &access : f.accesses)
      if (access.res == Res::Vmem) access.anywhere = !line;
    return fill.done(busy);
  }

  if (op.opClass == OpClass::Transpose) {
    if (!mreg(in.rd) || !mreg(in.rs1) || in.rs2 != 0 || in.imm != 0)
      return reject("unsupported XLU operands or encoding");
    Fill fill(f, "xlu.transpose", blocks);
    fill.mreg("read", in.rs1, false, 1, 1, /*mergeRelease=*/false);
    fill.mreg("write", in.rd, true, 1, 1, /*mergeRelease=*/false);
    int busy = fill.busyEnd();
    fill.hold({Unit::Xlu}, 0, busy);
    return fill.done(busy);
  }

  if (op.engine == Engine::Vpu) {
    auto found = vpuBlocks().find(name);
    const char *blockName = found == vpuBlocks().end() ? nullptr : found->second;
    if (!blockName) return reject("unsupported VPU operation");
    const bool immediate = op.opClass == OpClass::VpuLoadImmPair || op.opClass == OpClass::VpuLoadImmSingle;
    if (!immediate && in.imm != 0) return reject("unsupported VPU encoding");
    if ((immediate && (in.rs1 != 0 || in.rs2 != 0)) ||
        ((op.opClass == OpClass::VpuRowReduce || op.opClass == OpClass::VpuColReduce ||
          (op.opClass == OpClass::VpuElementwise && !op.twoInput)) && in.rs2 != 0))
      return reject("unsupported VPU encoding");
    Fill fill(f, blockName, blocks);
    f.writeRelease = fill.busyEnd();
    auto events = vpuEvents().find(op.opClass);
    if (events == vpuEvents().end()) return reject("unsupported VPU operation");
    if (op.opClass == OpClass::VpuRowReduce && ((in.rs1 & 1) || (in.rd & 1)))
      fill.fail("BF16 operands must name even register pairs");
    for (const RegisterEvent &event : events->second) {
      if (event.twoInput && !op.twoInput) continue;
      int reg = in.*event.operand + event.offset;
      if (!event.key) {
        if (!xreg(reg)) fill.fail("scale register out of range");
        f.accesses.push_back({Res::EReg, false, reg, 1, 0, 1});
        continue;
      }
      if (event.registers == 2) {
        if (!mreg(reg) || (reg & 1)) fill.fail("BF16 operands must name even register pairs");
      } else if (!mreg(reg)) fill.fail("register operand out of range");
      fill.mreg(event.key, reg, event.write, event.registers, event.passes);
    }
    for (const std::vector<int> *group : {&f.mregReads, &f.mregWrites}) {
      std::map<int, int> banks;
      for (int reg : *group)
        if (auto [at, inserted] = banks.emplace(reg & 31, reg); !inserted)
          fill.fail("VPU register operands must occupy distinct physical banks (register mod 32): m" +
                    std::to_string(at->second) + " and m" + std::to_string(reg) + " share bank " +
                    std::to_string(reg & 31));
    }
    int busy = fill.busyEnd();
    fill.vpu(busy);
    return fill.done(busy);
  }

  if (op.mxu >= 0) return mxuFootprint(in, blocks);

  return reject("operation is outside the selected bounded scope");
}

namespace {
bool overlaps(const Access &a, const Access &b) {
  return a.res == b.res && (a.anywhere || b.anywhere || (a.first < b.first + b.count && b.first < a.first + a.count));
}

std::optional<ModelOverlapCommand> overlapCommand(const Instr &in, const Footprint &f, int cycle,
                                                  const ModelTimingFacts &facts);

Dependence observedDependence(const Instr &a, const Footprint &fa, const Instr &b, const Footprint &fb,
                              const ModelTimingFacts *facts = nullptr, const ModelOverlapFacts *overlap = nullptr) {
  Dependence best;
  auto consider = [&](int distance, EdgeKind kind, const std::string &reason) {
    if (distance > best.distance) best = {distance, kind, reason};
  };
  if (isBarrier(a) || isBarrier(b)) consider(1, EdgeKind::Order, "architectural barrier");
  if (b.release) consider(fa.doneAge + 1, EdgeKind::Order, "completion waits for observed work");
  if (a.op->engine == Engine::Dma && b.op->engine == Engine::Dma) {
    bool aw = a.op->opClass == OpClass::DmaWait, bw = b.op->opClass == OpClass::DmaWait;
    if ((!aw && !bw) || a.op->channel == b.op->channel || (aw && !bw))
      consider(1, EdgeKind::Order, "DMA launch/wait order");
  }
  // Exact confirmed push-to-matmul (RAW) or matmul-to-push (WAR) evidence
  // replaces the synthetic whole-window weight lifetime.
  auto isMatMul = [](const Instr &in) {
    return in.op->opClass == OpClass::MatMul || in.op->opClass == OpClass::MatMulAcc;
  };
  bool weightRAW = a.op->opClass == OpClass::WeightPush && isMatMul(b);
  bool weightWAR = isMatMul(a) && b.op->opClass == OpClass::WeightPush;
  std::optional<int> streamedWeightGap;
  if (facts && overlap && (weightRAW || weightWAR) && a.op->mxu == b.op->mxu) {
    auto first = overlapCommand(a, fa, 0, *facts), second = overlapCommand(b, fb, 0, *facts);
    if (first && second)
      for (int gap = 1; gap <= first->firstFreeAge; ++gap) {
        second->issue = gap;
        if (overlap->admits({*first, *second}, false)) {
          streamedWeightGap = gap;
          break;
        }
      }
  }
  for (const Access &x : fa.accesses) {
    if (x.atCompletion) continue;
    for (const Access &y : fb.accesses) {
      if (!overlaps(x, y) || (!x.write && !y.write)) continue;
      if (streamedWeightGap && x.res == Res::Weight && x.write != y.write && x.write == weightRAW) {
        consider(*streamedWeightGap, weightRAW ? EdgeKind::RAW : EdgeKind::WAR, "confirmed streamed weight dependency");
        continue;
      }
      int distance;
      if (x.anywhere || y.anywhere) {
        distance = x.lastAge() - (y.atCompletion ? 0 : y.age) + 1;
      } else {
        int lo = std::max(x.first, y.first), hi = std::min(x.first + x.count, y.first + y.count) - 1;
        auto delta = [&](int element) {
          return x.age + (element - x.first) * x.step - (y.atCompletion ? 0 : y.age + (element - y.first) * y.step);
        };
        distance = std::max(delta(lo), delta(hi)) + 1;
      }
      EdgeKind kind = x.write && y.write ? EdgeKind::WAW : x.write ? EdgeKind::RAW : EdgeKind::WAR;
      consider(std::max(1, distance), kind, std::string(edgeKindName(kind)) + " on observed resource");
    }
  }
  auto contains = [](const std::vector<int> &regs, int reg) {
    return std::find(regs.begin(), regs.end(), reg) != regs.end();
  };
  for (int reg : fa.mregWrites) {
    if (contains(fb.mregReads, reg)) consider(fa.writeRelease + 1, EdgeKind::RAW, "observed MREG write release");
    if (contains(fb.mregWrites, reg)) consider(fa.writeRelease + 1, EdgeKind::WAW, "observed MREG write release");
  }
  for (int reg : fa.mregReads)
    if (contains(fb.mregWrites, reg) && !fb.writeDuringRead)
      consider(fa.readRelease + 1, EdgeKind::WAR, "observed MREG read release");
  return best;
}

bool sameEngine(const OpInfo &a, const OpInfo &b) {
  auto vector = [](Engine engine) { return engine == Engine::Vpu || engine == Engine::Xlu; };
  return a.engine == b.engine || (vector(a.engine) && vector(b.engine));
}

std::optional<ModelOverlapCommand> overlapCommand(const Instr &in, const Footprint &f, int cycle,
                                                  const ModelTimingFacts &facts) {
  std::string name = mxuBlockName(*in.op);
  if (auto found = vpuBlocks().find(in.op->name); found != vpuBlocks().end()) name = found->second;
  if (in.op->opClass == OpClass::Transpose) name = "xlu.transpose";
  if (in.op->opClass == OpClass::VLoad) name = "vlsu.vload";
  if (in.op->opClass == OpClass::VStore) name = "vlsu.vstore";
  if (in.op->name == "lw") name = "scalar_lsu.load";
  if (in.op->name == "sw") name = "scalar_lsu.store";
  if (in.op->name == "seld") name = "scalar_lsu.scale_load";
  const ModelTimingBlock *block = facts.block(name);
  if (!block) return std::nullopt;
  ModelOverlapCommand command{name, in.op->engine, f.mregReads, {}, {}, cycle, block->firstFreeAge};
  command.registers.insert(command.registers.end(), f.mregWrites.begin(), f.mregWrites.end());
  if (in.op->mxu >= 0) {
    switch (in.op->opClass) {
    case OpClass::WeightPush: command.weightSlot = in.rd; break;
    case OpClass::AccPushFp8:
    case OpClass::AccPushBf16: command.accumulator = in.rd; break;
    case OpClass::PopFp8:
    case OpClass::PopBf16: command.accumulator = in.rs2; break;
    default: command.accumulator = in.rd; command.weightSlot = in.rs2; break;
    }
  }
  return command;
}

class ObservedReservations final : public TimingReservations {
public:
  ObservedReservations(std::shared_ptr<const ModelTimingFacts> facts,
                       std::shared_ptr<const ModelOverlapFacts> overlap, bool chains)
      : facts(std::move(facts)), overlap(std::move(overlap)), chains(chains) {}
  TimingRuleResult<std::string> conflict(const Instr &in, const Footprint &f, int cycle) const override {
    return conflictWithContext(in, f, cycle, {});
  }
  TimingRuleResult<std::string> conflictWithContext(
      const Instr &in, const Footprint &f, int cycle, const TimingContext &context) const override {
    if (cycle < 0 || cycle > INT_MAX - f.doneAge - 1)
      return {{}, "model reservation exceeds cycle domain"};
    std::vector<Port> ports = requests(f, cycle);
    for (size_t i = 0; i < ports.size(); ++i)
      for (size_t j = 0; j < i; ++j)
        if (portConflict(ports[i], ports[j])) return {{}, "instruction has conflicting MREG port requests"};
    bool waiver = overlap && admitsInsertion(in, f, cycle, context);
    for (const Issued &prior : issued) {
      if (prior.cycle + prior.f.doneAge < cycle || cycle + f.doneAge < prior.cycle) continue;
      if (sameEngine(*in.op, *prior.in.op) && cycle <= prior.cycle + prior.f.doneAge &&
          prior.cycle <= cycle + f.doneAge && !waiver)
        return {"model isolated engine window remains live", {}};
      for (const Hold &a : prior.f.holds)
        for (const Hold &b : f.holds)
          if (a.unit == b.unit && a.index == b.index && prior.cycle + a.from <= cycle + b.to &&
              cycle + b.from <= prior.cycle + a.to)
            if (!(waiver && sameEngine(*in.op, *prior.in.op) &&
                  (a.unit == Unit::MxuCompute || a.unit == Unit::Xlu ||
                   (a.unit == Unit::ScalarLoad && scalarContext(in, context) &&
                    scalarContext(prior.in, prior.context)))))
              return {std::string(unitName(a.unit)) + " occupied", {}};
      for (const Port &a : prior.ports)
        for (const Port &b : ports)
          if (portConflict(a, b)) return {"observed MREG port occupied", {}};
    }
    return {};
  }
  std::string reserve(const Instr &in, const Footprint &f, int cycle) override {
    return reserveWithContext(in, f, cycle, {});
  }
  std::string reserveWithContext(const Instr &in, const Footprint &f, int cycle,
                                 const TimingContext &context) override {
    auto checked = conflictWithContext(in, f, cycle, context);
    if (!checked.error.empty()) return checked.error;
    if (!checked.value.empty()) return checked.value;
    issued.push_back({in, f, cycle, requests(f, cycle), true, context});
    return {};
  }
  std::string onWait(const Instr &, int cycle) override {
    // The wait may stall indefinitely, so future events extend back to its issue.
    for (Issued &prior : issued) {
      prior.eligible = false;
      for (Hold &hold : prior.f.holds)
        if (prior.cycle + hold.from > cycle) hold.from = cycle - prior.cycle;
      for (Port &port : prior.ports)
        if (port.cycle > cycle) port.from = std::min(port.from, cycle);
    }
    return {};
  }

private:
  struct Port { int bank, reg, row, from, cycle; bool write; };
  struct Issued {
    Instr in; Footprint f; int cycle; std::vector<Port> ports; bool eligible;
    TimingContext context;
  };
  static bool scalarContext(const Instr &in, const TimingContext &context) {
    return context && context->matches(in) && (in.op->name == "lw" || in.op->name == "sw");
  }
  bool admitsInsertion(const Instr &in, const Footprint &f, int cycle, const TimingContext &context) const {
    auto candidate = overlapCommand(in, f, cycle, *facts);
    if (!candidate) return false;
    if (scalarContext(in, context)) {
      candidate->scalarInstruction = in;
      candidate->context = context;
    }
    struct Entry { ModelOverlapCommand command; int end; bool eligible, candidate; };
    std::vector<Entry> entries{{*candidate, cycle + f.doneAge, true, true}};
    for (const Issued &prior : issued) {
      if (!sameEngine(*in.op, *prior.in.op)) continue;
      auto command = overlapCommand(prior.in, prior.f, prior.cycle, *facts);
      if (!command) {
        if (prior.cycle <= cycle + f.doneAge && cycle <= prior.cycle + prior.f.doneAge) return false;
        continue;
      }
      if (scalarContext(prior.in, prior.context)) {
        command->scalarInstruction = prior.in;
        command->context = prior.context;
      }
      entries.push_back({*command, prior.cycle + prior.f.doneAge, prior.eligible, false});
    }
    std::sort(entries.begin(), entries.end(), [](const Entry &a, const Entry &b) {
      return a.command.issue < b.command.issue;
    });
    // Retain completed commands until the entire overlap epoch drains.
    std::vector<Entry> affected;
    int end = -1;
    bool containsCandidate = false;
    for (const Entry &entry : entries) {
      if (entry.command.issue > end) {
        if (containsCandidate) break;
        affected.clear();
      }
      affected.push_back(entry);
      containsCandidate |= entry.candidate;
      end = std::max(end, entry.end);
    }
    std::vector<ModelOverlapCommand> epoch;
    std::vector<const Entry *> history;
    bool eligible = true;
    // Recheck every prefix, including commands placed after the candidate.
    for (const Entry &entry : affected) {
      if (!epoch.empty() && epoch.back().issue == entry.command.issue) return false;
      std::vector<const Entry *> live;
      for (const Entry *prior : history)
        if (prior->end >= entry.command.issue) live.push_back(prior);
      epoch.push_back(entry.command);
      eligible &= entry.eligible;
      if (live.size() == 1 && (!entry.eligible || !live[0]->eligible ||
                              !overlap->admits({live[0]->command, entry.command}, false))) return false;
      // A retired MXU command can determine the port held by the remaining live push.
      if (live.size() == 1 && epoch.size() >= 3 && overlap->rejectsPortHistory(epoch)) return false;
      if (live.size() >= 2 && (!eligible || !overlap->admits(epoch, chains))) return false;
      history.push_back(&entry);
    }
    return true;
  }
  static std::vector<Port> requests(const Footprint &f, int cycle) {
    std::vector<Port> ports;
    for (const Access &a : f.accesses)
      if (a.res == Res::MReg)
        for (int i = 0; i < a.count; ++i) {
          int element = a.first + i, at = cycle + a.age + i * a.step;
          ports.push_back({element / 32 % 32, element / 32, element % 32, at, at, a.write});
        }
    return ports;
  }
  static bool portConflict(const Port &a, const Port &b) {
    return a.bank == b.bank && a.from <= b.cycle && b.from <= a.cycle &&
        (a.write == b.write || (a.reg == b.reg && a.row == b.row));
  }
  std::vector<Issued> issued;
  std::shared_ptr<const ModelTimingFacts> facts;
  std::shared_ptr<const ModelOverlapFacts> overlap;
  bool chains;
};

std::string modelScope(const AtlasStream &stream) {
  if (stream.ops.size() > 32768) return "model program exceeds instruction memory";
  struct Transfer { uint64_t first, end; bool write, waited; };
  std::vector<Transfer> transfers;
  std::array<int, kAtlasDMAChannels> pending;
  pending.fill(-1);
  auto upperEntries = atlasDMAUpperWordEntries(stream);
  for (size_t block = 0; block < stream.starts.size(); ++block) {
    RegValues regs = stream.entry[block];
    auto upper = upperEntries[block];
    for (size_t i = stream.starts[block]; i < stream.blockEnd(block); ++i) {
      const Instr &in = stream.instrs[i];
      const OpInfo &op = *in.op;
      if (op.opClass == OpClass::DmaConfig) upper = regs[in.rs1];
      if (op.opClass == OpClass::DmaLoad || op.opClass == OpClass::DmaStore) {
        if (op.channel < 0 || op.channel >= int(kAtlasDMAChannels) || pending[op.channel] >= 0)
          return "model requires a free DMA channel";
        if (!upper || !regs[in.rd] || !regs[in.rs1] || !regs[in.rs2])
          return "model DMA requires known captured operands and upper base";
        if (size_t n = transfers.size(); n >= kAtlasDMACommandSlots && !transfers[n - kAtlasDMACommandSlots].waited)
          return "model DMA command slot is still pending: launch " + std::to_string(n) + " reuses slot " +
                 std::to_string(n % kAtlasDMACommandSlots) + " of unwaited launch " + std::to_string(n - kAtlasDMACommandSlots);
        bool load = op.opClass == OpClass::DmaLoad;
        uint64_t first = (uint64_t(*upper) << 32) | *regs[load ? in.rs1 : in.rd];
        uint64_t end = first + *regs[in.rs2];
        if (first < uint64_t(0x80000000) || end > uint64_t(0x1080000000))
          return "model DMA DRAM span is outside the selected configuration";
        for (const Transfer &prior : transfers)
          if (!prior.waited && first < prior.end && prior.first < end && (!load || prior.write))
            return "model pending DMA DRAM spans overlap";
        pending[op.channel] = int(transfers.size());
        transfers.push_back({first, end, !load, false});
      } else if (op.opClass == OpClass::DmaWait) {
        if (op.channel < 0 || op.channel >= int(kAtlasDMAChannels) || pending[op.channel] < 0)
          return "model DMA wait has no pending transfer";
        transfers[pending[op.channel]].waited = true;
        pending[op.channel] = -1;
      }
      applyScalar(in, regs);
    }
  }
  return {};
}
} // namespace

TimingProvider mlir::atlas::timing::modelTimingProvider(std::shared_ptr<const ModelTimingFacts> facts,
                                                      std::shared_ptr<const ModelOverlapFacts> overlap, bool chains) {
  TimingProvider provider;
  provider.id = kModelTimingProviderId.str();
  provider.costProbes = [](const OpInfo &op) {
    Instr first{&op, 0, 2, 4}, second{&op, 6, 8, 10};
    if (op.engine == Engine::Scalar) {
      if (!hasOperand(op, "x1")) first.rs1 = second.rs1 = 0;
      if (!hasOperand(op, "x2")) first.rs2 = second.rs2 = 0;
    }
    switch (op.opClass) {
    case OpClass::WeightPush:
    case OpClass::AccPushFp8:
    case OpClass::AccPushBf16:
      first = {&op, 0, 0, 0};
      second = {&op, 1, 2, 0};
      break;
    case OpClass::PopFp8:
    case OpClass::PopBf16:
      first = {&op, 0, 0, 0};
      second = {&op, 2, 0, 1};
      break;
    case OpClass::MatMul:
    case OpClass::MatMulAcc:
      first = {&op, 0, 0, 0};
      second = {&op, 1, 2, 1};
      break;
    case OpClass::VLoad:
    case OpClass::VStore:
    case OpClass::ScalarLoad:
    case OpClass::ScaleLoad:
    case OpClass::Transpose:
    case OpClass::VpuRowReduce:
    case OpClass::VpuColReduce:
      first.rs2 = second.rs2 = 0;
      break;
    case OpClass::ScalarStore:
      first.rd = second.rd = 0;
      break;
    case OpClass::VpuElementwise:
      if (!op.twoInput) first.rs2 = second.rs2 = 0;
      break;
    case OpClass::VpuLoadImmPair:
    case OpClass::VpuLoadImmSingle:
      first.rs1 = first.rs2 = second.rs1 = second.rs2 = 0;
      break;
    default:
      break;
    }
    return TimingRuleResult<std::pair<Instr, Instr>>{{first, second}, {}};
  };
  provider.validateScope = modelScope;
  provider.footprint = [facts](const Instr &in, const RegValues &regs) { return facts->resolve(in, regs); };
  provider.dependence = [facts, overlap](const Instr &a, const Footprint &fa, const Instr &b, const Footprint &fb) {
    return TimingRuleResult<Dependence>{observedDependence(a, fa, b, fb, facts.get(), overlap.get()), {}};
  };
  provider.contextualDependence = [base = provider.dependence](
      const Instr &a, const Footprint &fa, const TimingContext &ca,
      const Instr &b, const Footprint &fb, const TimingContext &cb) {
    bool disjoint = ca && cb && ca->matches(a) && cb->matches(b) && ca->sameRegion(*cb);
    if (disjoint) {
      const auto &aa = ca->scalarAddresses(), &bb = cb->scalarAddresses();
      for (size_t row = 0; row < aa.size(); ++row)
        disjoint &= uint64_t(aa[row]) + 4 <= bb[row] || uint64_t(bb[row]) + 4 <= aa[row];
    }
    if (!disjoint) return base(a, fa, b, fb);
    // Exact byte disjointness removes only the line-granularity VMEM alias.
    // All decoded register, release, ordering and port obligations survive.
    Footprint precise = fa;
    precise.accesses.erase(std::remove_if(precise.accesses.begin(), precise.accesses.end(),
        [](const Access &access) { return access.res == Res::Vmem; }), precise.accesses.end());
    return base(a, precise, b, fb);
  };
  provider.dmaConflict = [](const Footprint &dma, const Footprint &other) {
    DMACompletionConflict result;
    bool vls = std::any_of(other.holds.begin(), other.holds.end(), [](const Hold &h) {
      return h.unit == Unit::VloadPath || h.unit == Unit::VstorePath;
    });
    for (const Access &a : dma.accesses)
      if (a.atCompletion)
        for (const Access &b : other.accesses)
          if (overlaps(a, b) && (a.write || b.write || b.atCompletion || vls)) {
            result = {true, a.write && b.write ? EdgeKind::WAW : a.write ? EdgeKind::RAW :
                            b.write ? EdgeKind::WAR : EdgeKind::Order};
            return TimingRuleResult<DMACompletionConflict>{result, {}};
          }
    return TimingRuleResult<DMACompletionConflict>{result, {}};
  };
  provider.waitReusedDMASlots = true;
  provider.issueGap = [](const Instr &in) { return TimingRuleResult<int>{naturalGap(in), {}}; };
  provider.createReservations = [facts, overlap, chains] {
    return TimingRuleResult<std::unique_ptr<TimingReservations>>{
        std::make_unique<ObservedReservations>(facts, overlap, chains), {}};
  };
  return provider;
}

TimingRuleResult<TimingProvider> mlir::atlas::timing::modelTimingProviderFactory(ModuleOp module) {
  if (!module) return {{}, "model requires selected atlas.model_evidence"};
  auto selection = module->getAttrOfType<DictionaryAttr>(kAtlasModelEvidence);
  if (!selection) return {{}, "model requires selected atlas.model_evidence"};
  auto path = selection.getAs<StringAttr>("op_timing"), hash = selection.getAs<StringAttr>("op_timing_sha256");
  auto id = selection.getAs<StringAttr>("provider"), dma = selection.getAs<StringAttr>("dma");
  if (!path || !hash || !id || id.getValue() != kModelTimingProviderId ||
      (selection.get("dma") && (!dma || dma.getValue() != "wait")))
    return {{}, "incomplete or unsupported model evidence selection"};
  auto facts = loadModelTimingFacts(path.getValue(), hash.getValue(), bool(dma));
  if (!facts) return {{}, llvm::toString(facts.takeError())};
  for (const auto &[key, value] : facts->modelProvenance()) {
    auto retained = selection.getAs<StringAttr>(key);
    if (!retained || retained.getValue() != value) return {{}, "model evidence provenance mismatch: " + key};
  }
  std::shared_ptr<const ModelOverlapFacts> overlap;
  auto pairs = selection.getAs<StringAttr>("pairs"), pairsHash = selection.getAs<StringAttr>("pairs_sha256");
  auto composed = selection.getAs<StringAttr>("composed"), composedHash = selection.getAs<StringAttr>("composed_sha256");
  auto chains = selection.getAs<StringAttr>("chains");
  if ((selection.get("pairs") && !pairs) || (selection.get("pairs_sha256") && !pairsHash) ||
      (selection.get("chains") && (!chains || (chains.getValue() != "true" && chains.getValue() != "false"))) ||
      (selection.get("composed") && !composed) || (selection.get("composed_sha256") && !composedHash) ||
      bool(composed) != bool(composedHash) || (composed && !pairs) ||
      bool(pairs) != bool(pairsHash) || (chains && !pairs)) return {{}, "incomplete model overlap selection"};
  if (pairs) {
    auto loaded = loadModelOverlapFacts(pairs.getValue(), pairsHash.getValue(), *facts);
    if (!loaded) return {{}, llvm::toString(loaded.takeError())};
    if (composed)
      if (auto error = loaded->loadComposedFacts(composed.getValue(), composedHash.getValue(), *facts))
        return {{}, llvm::toString(std::move(error))};
    for (const auto &[key, value] : loaded->metadata()) {
      auto retained = selection.getAs<StringAttr>(key);
      if (!retained || retained.getValue() != value) return {{}, "model overlap provenance mismatch: " + key};
    }
    overlap = std::make_shared<ModelOverlapFacts>(std::move(*loaded));
  }
  return {modelTimingProvider(std::make_shared<ModelTimingFacts>(std::move(*facts)), overlap,
                              chains && chains.getValue() == "true"), {}};
}

namespace {
struct SelectAtlasModelEvidencePass : PassWrapper<SelectAtlasModelEvidencePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SelectAtlasModelEvidencePass)
  SelectAtlasModelEvidencePass() = default;
  SelectAtlasModelEvidencePass(const SelectAtlasModelEvidencePass &other) : PassWrapper(other) {}
  Option<std::string> opTiming{*this, "op-timing", llvm::cl::desc("Observed atlas.model_timing.v1 facts")};
  Option<std::string> opTimingHash{*this, "op-timing-sha256", llvm::cl::desc("Expected facts SHA-256")};
  Option<std::string> dmaPolicy{*this, "dma", llvm::cl::desc("Optional DMA completion policy: wait")};
  Option<std::string> pairsPath{*this, "pairs", llvm::cl::desc("Optional confirmed compact atlas.model_pairs.v1 facts")};
  Option<std::string> pairsHash{*this, "pairs-sha256", llvm::cl::desc("Expected compact overlap facts SHA-256")};
  Option<std::string> composedPath{*this, "composed", llvm::cl::desc("Optional validated derived overlap side projection")};
  Option<std::string> composedHash{*this, "composed-sha256", llvm::cl::desc("Expected derived side projection SHA-256")};
  Option<bool> chains{*this, "chains", llvm::cl::desc("Enable exact confirmed chain witnesses"), llvm::cl::init(false)};
  StringRef getArgument() const final { return "select-atlas-model-evidence"; }
  StringRef getDescription() const final { return "Pin model observations; timing passes select provider=model"; }
  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto fail = [&](const std::string &why) { module.emitError(why); signalPassFailure(); };
    if (auto state = module->getAttrOfType<StringAttr>(kAtlasTimingState); state && state.getValue() == "timed")
      return fail("select model evidence before timing; already timed input must be retimed");
    if (module->hasAttr(kAtlasTimingProvider)) return fail("model selection cannot replace a retained timing provider");
    if (!dmaPolicy.empty() && dmaPolicy != "wait") return fail("only dma=wait is supported");
    auto facts = loadModelTimingFacts(opTiming, opTimingHash, dmaPolicy == "wait");
    if (!facts) return fail(llvm::toString(facts.takeError()));
    if (pairsPath.empty() != pairsHash.empty() || (chains && pairsPath.empty()))
      return fail("pairs and pairs-sha256 are required together; chains requires pairs");
    if (composedPath.empty() != composedHash.empty() || (!composedPath.empty() && pairsPath.empty()))
      return fail("composed and composed-sha256 are required together and require pairs");
    std::shared_ptr<const ModelOverlapFacts> overlap;
    if (!pairsPath.empty()) {
      auto loaded = loadModelOverlapFacts(pairsPath, pairsHash, *facts);
      if (!loaded) return fail(llvm::toString(loaded.takeError()));
      if (!composedPath.empty())
        if (auto error = loaded->loadComposedFacts(composedPath, composedHash, *facts))
          return fail(llvm::toString(std::move(error)));
      overlap = std::make_shared<ModelOverlapFacts>(std::move(*loaded));
    }
    Builder builder(module.getContext());
    NamedAttrList selection;
    selection.set("provider", builder.getStringAttr(kModelTimingProviderId));
    selection.set("op_timing", builder.getStringAttr(opTiming));
    selection.set("op_timing_sha256", builder.getStringAttr(opTimingHash));
    if (facts->dmaWait()) selection.set("dma", builder.getStringAttr("wait"));
    for (const auto &[key, value] : facts->modelProvenance()) selection.set(key, builder.getStringAttr(value));
    if (overlap) {
      selection.set("pairs", builder.getStringAttr(pairsPath));
      if (!composedPath.empty()) selection.set("composed", builder.getStringAttr(composedPath));
      selection.set("chains", builder.getStringAttr(chains ? "true" : "false"));
      for (const auto &[key, value] : overlap->metadata()) selection.set(key, builder.getStringAttr(value));
    }
    module->setAttr(kAtlasModelEvidence, selection.getDictionary(module.getContext()));
  }
};
} // namespace

void mlir::atlas::registerSelectAtlasModelEvidencePass() { PassRegistration<SelectAtlasModelEvidencePass>(); }
