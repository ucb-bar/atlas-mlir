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
#include "llvm/Support/Path.h"
#include "llvm/Support/SHA256.h"
#include <algorithm>
#include <climits>
#include <cmath>
#include <deque>
#include <set>
#include <tuple>

using namespace mlir;
using namespace mlir::atlas;
using namespace mlir::atlas::timing;

std::string mlir::atlas::timing::modelTimingDiagnostic(llvm::StringRef message) {
  constexpr llvm::StringLiteral guidance =
      "See docs/model-timing.md#updating-for-npu-model-changes.";
  std::string result = message.str();
  for (size_t at; (at = result.find(guidance.str())) != std::string::npos;)
    result.erase(at, guidance.size());
  return llvm::StringRef(result).rtrim().str() + "\n" + guidance.str();
}

namespace {
using Object = llvm::json::Object;
using Array = llvm::json::Array;

constexpr int kLinesPerBank = kVmemBankBytes / kLineBytes;

llvm::Error factsError(const std::string &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), modelTimingDiagnostic("model timing facts: " + message));
}

// This contract describes lowering capabilities, not a qualified timing profile.
// Timing values belong to the selected facts; legacy link/bus knobs are dormant
// in the curve backend and therefore impose no compiler capability requirement.
std::string unsupportedCapability(const Object &cfg) {
  const Object *arch = cfg.getObject("arch_state_config"), *units = cfg.getObject("execution_units");
  if (!arch) return "arch_state_config";
  if (!units) return "execution_units";
  if (cfg.getBoolean("ignore_runtime_errors") != false) return "ignore_runtime_errors (requires false)";
  if (arch->getString("numerics") != "rtl") return "arch_state_config.numerics (requires rtl)";
  if (arch->getBoolean("randomize_init") != false) return "arch_state_config.randomize_init (requires false)";
  if (cfg.getString("dma_memory_backend") != "curve") return "dma_memory_backend (requires curve)";
  const Object *memory = cfg.getObject("dma_memory_params");
  if (!memory || !memory->getString("curves") || memory->getString("curves")->empty())
    return "dma_memory_params.curves (requires a selected curve file)";
  for (auto [key, value] : std::initializer_list<std::pair<const char *, int64_t>>{
           {"mrf_depth", 32}, {"mrf_width", 32}, {"num_m_registers", 64}, {"num_wb_registers", 2},
           {"wb_width", 1024}, {"num_x_registers", 32}, {"num_e_registers", 32}, {"num_csrs", 4096},
           {"vmem_size", kAtlasVmemBytes}, {"dram_base", 0x80000000LL}, {"dram_size", 0x1000000000LL}})
    if (arch->getInteger(key) != value)
      return std::string("arch_state_config.") + key + " (requires " + std::to_string(value) + ")";
  for (auto [key, value] : std::initializer_list<std::pair<const char *, int64_t>>{
           {"fetch_width", 1}, {"dma_num_channels", 8}, {"dma_beat_bytes", 32},
           {"dma_max_transfer_bytes", 4096}, {"vmem_bank_bytes", kVmemBankBytes}})
    if (cfg.getInteger(key) != value)
      return std::string(key) + " (requires " + std::to_string(value) + ")";
  for (auto [key, value] : std::initializer_list<std::pair<const char *, const char *>>{
           {"Scalar0", "ScalarExecutionUnit"}, {"Matrix0", "MatrixExecutionUnitSystolic"},
           {"Matrix1", "MatrixExecutionUnitInner"}, {"Vector0", "VectorExecutionUnit"},
           {"XLU", "CrossLaneExecutionUnit"}, {"DMA0", "DmaExecutionUnit"}, {"LSU", "LoadStoreUnit"}})
    if (units->getString(key) != value)
      return std::string("execution_units.") + key + " (requires " + value + ")";
  if (cfg.getString("isa") != "npu_model.isa.IsaSpec") return "isa (requires npu_model.isa.IsaSpec)";
  return {};
}

// Fixed-layout compiler capabilities, including the layout used by overlap
// consumers. Extraction is independent: selected facts must establish all these
// values before any provider or overlap rule may use this compiled layout.
llvm::Error checkArchitecture(const Object &root, const Object &cfg) {
  const Object *facts = root.getObject("architecture");
  if (!facts || facts->getString("schema") != "atlas.architecture.v1")
    return factsError("requires source-derived architecture facts (atlas.architecture.v1)");
  struct Capability { const char *key; int64_t value; const char *config; bool arch; };
  for (const auto &cap : std::initializer_list<Capability>{
      {"mreg_count", kAtlasTensorRegisters, "num_m_registers", true},
      {"mreg_rows", kAtlasMregRows, "mrf_depth", true},
      {"mreg_row_bytes", kAtlasMregRowBytes, "mrf_width", true},
      {"mreg_bytes", kAtlasRegisterBytes, nullptr, false},
      {"mreg_physical_bank_count", kAtlasMregBanks, nullptr, false},
      {"mreg_bank_modulus", kAtlasMregBanks, nullptr, false},
      {"mreg_registers_per_bank", kAtlasTensorRegisters / kAtlasMregBanks, nullptr, false},
      {"mreg_physical_row_stride", kAtlasMregRows, nullptr, false},
      {"mreg_logical_limit", kAtlasTensorRegisters, nullptr, false},
      {"bf16_registers_per_tile", kAtlasBF16Registers, nullptr, false},
      {"bf16_pair_stride", 1, nullptr, false},
      {"bf16_tile_bytes", kAtlasBF16TileBytes, nullptr, false},
      {"bf16_element_bytes", 2, nullptr, false},
      {"fp8_element_bytes", 1, nullptr, false},
      {"fp8_registers_per_tile", 1, nullptr, false},
      {"fp8_tile_bytes", kAtlasRegisterBytes, nullptr, false},
      {"vmem_bytes", kAtlasVmemBytes, "vmem_size", true},
      {"vmem_bank_bytes", kVmemBankBytes, "vmem_bank_bytes", false},
      {"vmem_bank_count", kVmemBanks, nullptr, false},
      {"weight_buffer_count", 2, "num_wb_registers", true},
      {"weight_buffer_bytes", kAtlasRegisterBytes, "wb_width", true},
      {"xreg_count", 32, "num_x_registers", true},
      {"ereg_count", 32, "num_e_registers", true},
      {"csr_count", 4096, "num_csrs", true},
      {"scalar_word_bytes", 4, nullptr, false},
      {"dma_channels", kAtlasDMAChannels, "dma_num_channels", false},
      {"dma_beat_bytes", 32, "dma_beat_bytes", false},
      {"dma_max_transfer_bytes", 4096, "dma_max_transfer_bytes", false},
      {"fetch_width", 1, "fetch_width", false},
      {"dram_base", 0x80000000LL, "dram_base", true},
      {"dram_bytes", 0x1000000000LL, "dram_size", true},
      }) {
    auto value = facts->getInteger(cap.key);
    if (!value || *value != cap.value)
      return factsError(std::string("unsupported architecture capability ") + cap.key +
                        ": compiler requires " + std::to_string(cap.value));
    const Object *config = cap.arch ? cfg.getObject("arch_state_config") : &cfg;
    if (cap.config && (!config || config->getInteger(cap.config) != value))
      return factsError(std::string("derived architecture differs from configuration: ") + cap.key);
  }
  // Selected timing capacity and dormant legacy geometry are derived and bound
  // to configuration, but impose no fixed-layout compiler requirement.
  for (const char *key : {"dma_max_in_flight", "offchip_link_width_bits",
                         "offchip_link_core_cycles_per_beat", "dma_offchip_command_words",
                         "vmem_bus_width_bits", "vmem_bus_core_cycles_per_beat", "vmem_bytes_per_cycle"}) {
    auto value = facts->getInteger(key);
    if (!value || cfg.getInteger(key) != value)
      return factsError(std::string("derived architecture differs from configuration: ") + key);
  }
  return llvm::Error::success();
}

// Memory instruction roles are checked against the existing compiler footprint,
// so this compatibility boundary does not introduce a second direction table.
llvm::Error checkMemoryOperandRoles(const OpInfo &op, const Object &entry) {
  const Array *operands = entry.getArray("operands");
  if (!operands) return factsError("missing ISA resource operands for " + op.name);
  Instr probe{&op, 4, 8, 12};
  Footprint expected = footprintOf(probe, unknownRegs());
  std::set<std::pair<Res, int>> covered;
  for (const auto &value : *operands) {
    const Object *operand = value.getAsObject();
    if (!operand) return factsError("invalid ISA resource operand for " + op.name);
    const auto *resource = operand->get("resource");
    if (resource && resource->kind() == llvm::json::Value::Null) continue; // Immediate.
    auto type = operand->getString("type"), field = operand->getString("compiler_operand");
    auto name = operand->getString("resource");
    Res res;
    if (name == "MReg" && type == "MatrixReg") res = Res::MReg;
    else if (name == "XReg" && type == "ScalarReg") res = Res::XReg;
    else if (name == "EReg" && type == "ExponentReg") res = Res::EReg;
    else return factsError("unsupported ISA memory operand resource for " + op.name);
    int reg = field == "rd" ? probe.rd : field == "rs1" ? probe.rs1 : field == "rs2" ? probe.rs2 : -1;
    if (reg < 0 || !covered.emplace(res, reg).second)
      return factsError("invalid or duplicate ISA memory operand role for " + op.name);
    bool read = false, write = false;
    int index = reg * (res == Res::MReg ? kAtlasMregRows : 1);
    for (const Access &access : expected.accesses)
      if (access.res == res && index >= access.first && index < access.first + access.count)
        (access.write ? write : read) = true;
    if ((!read && !write) || operand->getBoolean("read") != read || operand->getBoolean("write") != write)
      return factsError("ISA memory operand direction differs from compiler footprint: " + op.name);
  }
  for (const Access &access : expected.accesses) {
    if (access.res != Res::MReg && access.res != Res::XReg && access.res != Res::EReg) continue;
    int scale = access.res == Res::MReg ? kAtlasMregRows : 1;
    for (int index = access.first; index < access.first + access.count; index += scale)
      if (!covered.count({access.res, index / scale}))
        return factsError("incomplete ISA memory operand roles for " + op.name);
  }
  return llvm::Error::success();
}

// The model supplies operand types and assembly roles. Tokens are the small
// compiler syntax adapter; comparing them also checks the builtin ISA table.
llvm::Error checkIsaMetadata(const Object &root) {
  const Object *metadata = root.getObject("isa_metadata");
  if (!metadata) return factsError("requires source-derived ISA operand metadata");
  std::set<std::string> vectors;
  std::set<std::pair<int, OpClass>> matrices;
  std::set<std::string> memory;
  for (const auto &[name, value] : *metadata) {
    const OpInfo *op = findOp(name.str());
    const Object *entry = value.getAsObject();
    if (!op || !entry || entry->getString("compiler_operands") != op->operands)
      return factsError("ISA operands differ from compiler-op adapter: " + name.str());
    std::string engine;
    if (op->engine == Engine::Vpu && model_schema::vpuBlocks().count(name.str())) {
      engine = "vpu";
      vectors.insert(name.str());
    } else if ((op->engine == Engine::Mxu0 || op->engine == Engine::Mxu1) &&
               model_schema::mxuBlocks().count(op->opClass)) {
      engine = "mxu" + std::to_string(op->mxu);
      matrices.emplace(op->mxu, op->opClass);
    } else if (name == "vload" || name == "vstore" || name == "vtrpose.xlu" ||
               name == "lw" || name == "sw" || name == "seld") {
      engine = name == "vtrpose.xlu" ? "xlu" : name == "vload" || name == "vstore" ? "vlsu" : "scalar_lsu";
      memory.insert(name.str());
      if (auto error = checkMemoryOperandRoles(*op, *entry)) return error;
    } else return factsError("ISA metadata is outside the supported compiler-op adapter: " + name.str());
    if (entry->getString("engine") != engine)
      return factsError("ISA engine differs from compiler-op adapter: " + name.str());
  }
  if (vectors.size() != model_schema::vpuBlocks().size() ||
      matrices.size() != 2 * model_schema::mxuBlocks().size() || memory.size() != 6)
    return factsError("source-derived ISA metadata has incomplete compiler-op coverage");
  return llvm::Error::success();
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
    access(key, Res::MReg, write, reg * kAtlasMregRows, registers * kAtlasMregRows, passes);
    auto &list = write ? f.mregWrites : f.mregReads;
    for (int i = 0; i < registers; ++i) list.push_back(reg + i);
    int end = releaseEnd(write ? "mreg_write" : "mreg_read", last(key) + (write ? 0 : latency));
    if (end < last(key)) fail("MREG reservation releases before its last access");
    int &release = write ? f.writeRelease : f.readRelease;
    release = mergeRelease ? std::max(release, end) : end;
    return end;
  }
  int busyEnd() { return releaseEnd("unit", block ? block->firstFreeAge - 1 : 0); }
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

Footprint mxuFootprint(const Instr &in, const std::map<std::string, ModelTimingBlock> &blocks,
                       const std::map<std::string, std::vector<ModelTimingGroup>> &weightUses) {
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
    if (reg < 0 || reg >= kAtlasTensorRegisters) fill.fail("register operand out of range");
    track(fill.mreg(key, reg, write));
  };
  auto slotRows = [&](const char *key, Res res, int slot, bool write) {
    if (slot < 0 || slot > 1) fill.fail("accumulator and weight slots are 0 or 1");
    fill.access(key, res, write, (m * 2 + slot) * kAtlasMregRows, kAtlasMregRows);
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
    auto uses = weightUses.find("mxu" + std::to_string(m));
    if (uses == weightUses.end()) fill.fail("missing source-derived weight-use windows");
    else {
      // Pushes replace whole weight rows. Their dependencies need the first and
      // last use across all PEs in each row, rather than the compute envelope.
      const auto &rows = uses->second;
      if (rows.back().lastAge > busy) fill.fail("weight use exceeds compute lifetime");
      // The parser checks affine row projection. Two sweeps preserve every
      // row endpoint without quadratic comparisons between scalar PE groups.
      for (int age : {rows.front().firstAge, rows.front().lastAge})
        f.accesses.push_back({Res::Weight, false, (m * 2 + in.rs2) * kAtlasMregRows, kAtlasMregRows,
                              age, rows[1].firstAge - rows[0].firstAge});
    }
  }
  f.holds.push_back({Unit::MxuCompute, m, 0, busy});
  return fill.done(busy);
}

} // namespace

std::string mlir::atlas::timing::modelSha256(llvm::StringRef bytes) {
  auto digest = llvm::SHA256::hash(llvm::ArrayRef<uint8_t>(
      reinterpret_cast<const uint8_t *>(bytes.data()), bytes.size()));
  return llvm::toHex(llvm::ArrayRef<uint8_t>(digest), true);
}

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

namespace {
bool exactKeys(const Object &object, std::initializer_list<const char *> keys) {
  if (object.size() != keys.size()) return false;
  for (const char *key : keys)
    if (!object.get(key)) return false;
  return true;
}

llvm::Expected<std::map<std::string, std::vector<ModelTimingGroup>>> parseWeightUses(const Object &document) {
  const Object *engines = document.getObject("engines");
  if (!exactKeys(document, {"schema", "engines"}) ||
      document.getString("schema") != "atlas.mxu_weight_timing.v1" || !engines ||
      !exactKeys(*engines, {"mxu0", "mxu1"}))
    return factsError("requires source-derived MXU weight-use windows (atlas.mxu_weight_timing.v1)");
  std::map<std::string, std::vector<ModelTimingGroup>> result;
  for (const char *engine : {"mxu0", "mxu1"}) {
    const Object *entry = engines->getObject(engine);
    if (!entry || !exactKeys(*entry, {"use_window", "use_rows", "reads_before_writes", "resource_row_axis"}) ||
        entry->getBoolean("reads_before_writes") != true || entry->getString("resource_row_axis") != "output_column")
      return factsError("unsupported MXU weight resource mapping or edge order");
    const Object *window = entry->getObject("use_window");
    const Array *rows = entry->getArray("use_rows");
    if (!window || !exactKeys(*window, {"first_age", "last_age", "row_step", "column_step"}) ||
        !rows || rows->size() != kAtlasMregRows)
      return factsError("incomplete MXU weight-use geometry");
    auto first = window->getInteger("first_age"), last = window->getInteger("last_age");
    auto rowStep = window->getInteger("row_step"), columnStep = window->getInteger("column_step");
    if (!first || !last || !rowStep || !columnStep || *first < 0 || *last < *first ||
        *rowStep < 0 || *columnStep < 0 || *last >= 4095 || *rowStep >= 4095 || *columnStep >= 4095 ||
        *last + (kAtlasMregRows - 1) * (*rowStep + *columnStep) >= 4095)
      return factsError("unsupported MXU weight-use bounds");
    for (int row = 0; row < kAtlasMregRows; ++row) {
      const Object *use = (*rows)[row].getAsObject();
      int begin = *first + row * *rowStep;
      int end = *last + row * *rowStep + (kAtlasMregRows - 1) * *columnStep;
      if (!use || !exactKeys(*use, {"first_age", "last_age"}) ||
          use->getInteger("first_age") != begin || use->getInteger("last_age") != end)
        return factsError("MXU weight row projection differs from PE use windows");
      result[engine].push_back({begin, end, end - begin + 1, 1});
    }
  }
  return result;
}

// The selected file carries an explicit serialization of its deterministic
// semantic inputs. Hash those exact bytes (Python and LLVM render floats
// differently), then compare both decoded objects using one canonicalizer.
llvm::Expected<std::string> validateDerivedContract(const Object &root) {
  if (root.getBoolean("test_only") == true)
    return factsError("test-only observations cannot select a production provider");
  const Object *contract = root.getObject("derivation_contract");
  if (!contract || !exactKeys(*contract, {"schema", "payload_json", "payload_sha256", "derivation", "validation"}) ||
      contract->getString("schema") != "atlas.derived_facts.v1")
    return factsError("requires versioned source-derived facts contract atlas.derived_facts.v1; raw observations are validation/test-only");
  auto payloadJson = contract->getString("payload_json"), payloadHash = contract->getString("payload_sha256");
  if (!payloadJson || !payloadHash || !hashSyntax(*payloadHash) || modelSha256(*payloadJson) != *payloadHash)
    return factsError("derived semantic payload hash mismatch");
  const Object *derivation = contract->getObject("derivation"), *validation = contract->getObject("validation");
  if (!derivation || !exactKeys(*derivation, {"schema", "kind", "payload_sha256"}) ||
      derivation->getString("schema") != "atlas.derivation.v1" ||
      derivation->getString("kind") != "source_derived" || derivation->getString("payload_sha256") != payloadHash)
    return factsError("requires a source-derived derivation record bound to the semantic payload");
  if (!validation || !exactKeys(*validation, {"schema", "passed", "payload_sha256"}) ||
      validation->getString("schema") != "atlas.validation.v1" || validation->getBoolean("passed") != true ||
      validation->getString("payload_sha256") != payloadHash)
    return factsError("requires a successful validation record bound to the semantic payload");
  const Object *model = root.getObject("model");
  if (!model) return factsError("missing model provenance in derived semantic payload");
  Object semanticModel;
  for (const char *key : {"revision", "revision_kind", "source_sha256", "numerical_data_sha256",
                          "configuration", "configuration_sha256"}) {
    const auto *value = model->get(key);
    if (!value) return factsError(std::string("missing derived semantic model.") + key);
    semanticModel[key] = *value;
  }
  auto revisionKind = model->getString("revision_kind");
  if (!revisionKind || (*revisionKind != "locked_source" && *revisionKind != "source_sha256_prefix"))
    return factsError("unsupported model revision identity kind");
  Object semantic{{"model", std::move(semanticModel)}};
  for (const char *key : {"geometry", "age_convention", "op_timing", "reservation_releases", "vocabulary_sha256",
                          "architecture", "operand_vocabulary", "isa_metadata"}) {
    const auto *value = root.get(key);
    if (!value) return factsError(std::string("missing derived semantic ") + key);
    semantic[key] = *value;
  }
  for (const char *key : {"frontend_timing", "scratchpad_contract", "overlap_rules_sha256", "mxu_weight_timing", "replay_timing", "proposal_costs"})
    if (const auto *value = root.get(key)) semantic[key] = *value;
  if (const auto *value = root.get("dma_timing")) {
    const Object *dma = value->getAsObject();
    if (!dma) return factsError("dma_timing must be an object");
    Object deterministic(*dma);
    deterministic.erase("validation");
    semantic["dma_timing"] = std::move(deterministic);
  }
  for (const char *key : {"engine_admission", "dma_admission"})
    if (const auto *value = root.get(key)) {
      const Object *document = value->getAsObject();
      if (!document) return factsError(std::string(key) + " must be an object");
      Object deterministic(*document);
      deterministic.erase("validation");
      deterministic.erase("producer_sources");
      semantic[key] = std::move(deterministic);
    }
  auto decoded = llvm::json::parse(*payloadJson);
  if (!decoded) return factsError(llvm::toString(decoded.takeError()));
  if (canonicalJson(*decoded) != canonicalJson(llvm::json::Value(std::move(semantic))))
    return factsError("derived semantic payload differs from selected timing inputs");
  const Object *gate = root.getObject("validation");
  const Array *operations = root.getArray("op_timing");
  auto reportHash = gate ? gate->getString("gate_report_sha256") : std::nullopt;
  if (!gate || gate->getBoolean("passed") != true || gate->getString("payload_sha256") != payloadHash ||
      !operations || gate->getInteger("operation_count") != int64_t(operations->size()) ||
      !gate->getInteger("observed_runs") || *gate->getInteger("observed_runs") < 1 ||
      !gate->getInteger("unobserved_runs") || *gate->getInteger("unobserved_runs") < 1 || !reportHash || !hashSyntax(*reportHash))
    return factsError("requires successful operation validation bound to the deterministic payload");
  if (root.get("mxu_weight_timing")) {
    const Object *weights = gate->getObject("mxu_weight_validation");
    if (!weights || weights->getBoolean("passed") != true)
      return factsError("requires successful MXU weight validation bound to the deterministic payload");
  }
  return payloadHash->str();
}

bool completeReleaseCoverage(const ModelTimingBlock &block) {
  std::set<std::string> expected{"unit"};
  for (const auto &[event, group] : block.events) {
    llvm::StringRef key(event);
    if (key == "vmem_read" || key == "vmem_write") expected.insert(event);
    if (key.starts_with("mreg_read") ||
        ((block.engine == "vpu" || block.engine == "xlu") && key.starts_with("read")))
      expected.insert("mreg_read");
    if (key.starts_with("mreg_write") ||
        ((block.engine == "vpu" || block.engine == "xlu") && key.starts_with("write")))
      expected.insert("mreg_write");
  }
  if (expected.size() != block.releases.size()) return false;
  for (const auto &key : expected)
    if (!block.releases.count(key)) return false;
  return true;
}

llvm::Expected<ModelFrontendTiming> parseFrontend(const Object &f) {
  auto error = [](const char *why) -> llvm::Expected<ModelFrontendTiming> { return factsError(why); };
  const Object *delay = f.getObject("delay"), *scalar = f.getObject("scalar"), *control = f.getObject("control");
  if (!exactKeys(f, {"schema", "ordinary_issue_gap", "delay", "scalar", "control", "instructions"}) ||
      f.getString("schema") != "atlas.model_frontend.v1" || !delay || !scalar || !control ||
      !exactKeys(*delay, {"base_gap", "immediate_mask", "decrement_per_tick"}) ||
      !exactKeys(*scalar, {"read_age", "write_age", "completion_delay"}) ||
      !exactKeys(*control, {"delay_slots", "operand_read_age", "jal_link_pc_offset"}))
    return error("requires complete source-derived frontend timing schema");
  auto bounded = [](const Object &object, const char *key, int minimum = 0) {
    auto value = object.getInteger(key);
    return value && *value >= minimum && *value < 65536;
  };
  if (!bounded(f, "ordinary_issue_gap", 1) || !bounded(*delay, "base_gap", 1) ||
      !bounded(*delay, "immediate_mask") || delay->getInteger("decrement_per_tick") != 1 ||
      !bounded(*scalar, "read_age") || !bounded(*scalar, "write_age") ||
      !bounded(*scalar, "completion_delay", 1) || !bounded(*control, "operand_read_age") ||
      control->getInteger("delay_slots") != 1 || control->getInteger("jal_link_pc_offset") != 1)
    return error("unsupported frontend issue, scalar timing or control-flow encoding capability");
  // These are compiler capabilities, separately checked after source extraction:
  // delay insertion encodes 12-bit chunks and the CFG scheduler has one slot.
  if (f.getInteger("ordinary_issue_gap") != 1 || delay->getInteger("base_gap") != 1 ||
      delay->getInteger("immediate_mask") != 4095)
    return error("unsupported compiler scheduling/encoding capability: ordinary_issue_gap=1, delay base_gap=1 and immediate_mask=4095 are required");
  // completion_delay schedules deferred execution-log callbacks. The reviewed
  // scalar source commits register effects during exec at read_age/write_age.
  ModelFrontendTiming result{int(*f.getInteger("ordinary_issue_gap")), int(*delay->getInteger("base_gap")),
      int(*delay->getInteger("immediate_mask")), int(*scalar->getInteger("read_age")),
      int(*scalar->getInteger("write_age")), int(*control->getInteger("operand_read_age")),
      int(*control->getInteger("delay_slots")), {}};
  const Object *instructions = f.getObject("instructions");
  if (!instructions || instructions->empty()) return error("missing frontend instruction access coverage");
  for (const auto &[name, value] : *instructions) {
    const Object *entry = value.getAsObject();
    auto kind = entry ? entry->getString("kind") : std::nullopt;
    if (!entry || !exactKeys(*entry, {"kind", "reads", "writes"}) || !kind ||
        (*kind != "scalar" && *kind != "branch" && *kind != "jump" && *kind != "halt" &&
         *kind != "delay" && *kind != "control"))
      return error("unsupported frontend instruction access record");
    std::vector<ModelFrontendAccess> accesses;
    std::set<std::tuple<bool, std::string, std::string>> unique;
    for (bool write : {false, true}) {
      const Array *records = entry->getArray(write ? "writes" : "reads");
      if (!records) return error("missing frontend instruction read/write roles");
      for (const auto &record : *records) {
        const Object *access = record.getAsObject();
        auto resource = access ? access->getString("resource") : std::nullopt;
        auto operand = access ? access->getString("operand") : std::nullopt;
        if (!access || !exactKeys(*access, {"resource", "operand"}) || !resource || !operand ||
            (*resource != "xreg" && *resource != "ereg") ||
            (*operand != "rd" && *operand != "rs1" && *operand != "rs2") ||
            !unique.emplace(write, resource->str(), operand->str()).second)
          return error("unsupported frontend instruction register mapping");
        accesses.push_back({operand->str(), write, *resource == "ereg"});
      }
    }
    result.instructions[name.str()] = std::move(accesses);
  }
  return result;
}

llvm::Expected<int> parseScratchpadContract(const Object &root) {
  const Object *contract = root.getObject("scratchpad_contract");
  const Object *assumptions = contract ? contract->getObject("assumptions") : nullptr;
  auto pipeline = contract ? contract->getInteger("dma_store_read_to_request_cycles") : std::nullopt;
  auto limitation = contract ? contract->getString("limitation") : std::nullopt;
  const Array *checked = contract ? contract->getArray("checked_source_methods") : nullptr;
  if (!contract || !exactKeys(*contract, {"schema", "status", "assumptions", "response_latency_derived",
          "dma_store_read_to_request_cycles", "limitation", "checked_source_methods"}) ||
      contract->getString("schema") != "atlas.scratchpad_contract.v1" ||
      contract->getString("status") != "explicit-target-assumption" ||
      contract->getBoolean("response_latency_derived") != false || !assumptions ||
      !exactKeys(*assumptions, {"scratchpad_read_latency"}) || assumptions->getInteger("scratchpad_read_latency") != 1 ||
      !pipeline || *pipeline < 1 || *pipeline >= 4096 || !limitation || limitation->empty() || !checked || checked->empty())
    return factsError("requires explicit supported scratchpad target assumption; model response latency is not derived");
  for (const auto &method : *checked)
    if (!method.getAsString() || method.getAsString()->empty())
      return factsError("invalid scratchpad source-method coverage");
  return int(*assumptions->getInteger("scratchpad_read_latency"));
}

llvm::Expected<ModelDMATiming> parseDMATiming(const Object &d, const Object &cfg, llvm::StringRef payloadHash) {
  auto error = [](const std::string &why) -> llvm::Expected<ModelDMATiming> {
    return factsError("dma_timing: " + why);
  };
  if (!exactKeys(d, {"schema", "backend", "resolved", "estimate_only", "curve_sha256", "source_sha256",
      "source_shapes", "beat_bytes", "max_in_flight", "num_channels", "max_transfer_bytes", "queue_depth",
      "window_beats", "frequency_ghz", "channels", "measured_channels", "access_bytes", "latency_mode",
      "converge", "knee", "enforce_peak", "min_latency", "startup_cycles", "issue_interval_cycles",
      "rounding_epsilon_cycles", "curves", "reset_last_ready_cycles", "semantics", "limits", "validation"}) ||
      d.getString("schema") != "atlas.dma_timing.v1" || d.getString("backend") != "curve" ||
      d.getBoolean("resolved") != true || d.getBoolean("estimate_only") != true)
    return error("requires the complete resolved proposal-only curve schema");
  const auto &identity = model_schema::dmaIdentity();
  const Object *sources = d.getObject("source_sha256"), *shapes = d.getObject("source_shapes");
  if (identity.size() != 5 || !sources || !shapes || !exactKeys(*sources, {"memory_backend.py", "dma.py"}) ||
      !exactKeys(*shapes, {"memory_backend.py", "dma.py"}))
    return error("requires compiled DMA source, shape and curve identities");
  for (const auto &[key, pin] : identity) {
    auto actual = key == "curve_sha256" ? d.getString(key) :
        llvm::StringRef(key).starts_with("shape:") ? shapes->getString(llvm::StringRef(key).drop_front(6)) :
        sources->getString(key);
    if (!actual || !hashSyntax(*actual) || (key != "curve_sha256" && *actual != pin))
      return error("unreviewed source, shape or curve: " + key);
  }
  const Object *validation = d.getObject("validation");
  auto reportHash = validation ? validation->getString("report_sha256") : std::nullopt;
  if (!validation || !(payloadHash.empty() ?
          exactKeys(*validation, {"passed", "report_sha256", "backend_streams", "isolated_dma_runs", "mixed_dma_runs"}) :
          exactKeys(*validation, {"passed", "report_sha256", "backend_streams", "isolated_dma_runs", "mixed_dma_runs", "payload_sha256"})) ||
      validation->getBoolean("passed") != true || !reportHash || !hashSyntax(*reportHash) ||
      !validation->getInteger("backend_streams") || *validation->getInteger("backend_streams") < 1 ||
      !validation->getInteger("isolated_dma_runs") || *validation->getInteger("isolated_dma_runs") < 1 ||
      !validation->getInteger("mixed_dma_runs") || *validation->getInteger("mixed_dma_runs") < 1)
    return error("requires successful backend/isolated/mixed validation and its report SHA-256");
  if (!payloadHash.empty() && validation->getString("payload_sha256") != payloadHash)
    return error("validation does not bind the selected deterministic payload");
  const Object *semantics = d.getObject("semantics");
  if (!semantics || semantics->size() != 8) return error("missing reviewed backend/engine semantics");
  for (auto [key, value] : std::initializer_list<std::pair<const char *, const char *>>{
      {"history", "last_window_beats_issued_including_current_beat"},
      {"curve_selection", "nearest_read_percentage_tie_lower"},
      {"acceptance", "pending_count_below_queue_depth_including_presented_response"},
      {"ready", "max_issue_plus_unloaded_latency_and_previous_fractional_ready_plus_curve_spacing"},
      {"response", "oldest_ready_then_issue_order_held_until_pop"},
      {"history_lifetime", "reset_only_not_command_boundary"}, {"address_dependence", "none"},
      {"engine", "registered_slots_and_in_flight;D_before_A;retire_after_last_ack"}})
    if (semantics->getString(key) != value) return error("unsupported semantics: " + std::string(key));
  auto integer = [&](const char *key) {
    auto value = d.getInteger(key);
    return value && *value > 0 && *value <= INT_MAX ? int(*value) : 0;
  };
  auto number = [&](const char *key) {
    auto value = d.getNumber(key);
    return value && std::isfinite(*value) ? *value : -1.;
  };
  int beat = integer("beat_bytes"), maxBytes = integer("max_transfer_bytes");
  int inFlight = integer("max_in_flight"), queue = integer("queue_depth"), interval = integer("issue_interval_cycles");
  if (!beat || !maxBytes || !inFlight || !queue || !interval || !integer("window_beats") ||
      cfg.getInteger("dma_beat_bytes") != beat || cfg.getInteger("dma_max_transfer_bytes") != maxBytes ||
      cfg.getInteger("dma_max_in_flight") != inFlight ||
      cfg.getInteger("dma_num_channels") != integer("num_channels") ||
      maxBytes % beat || !integer("access_bytes") || !integer("measured_channels") ||
      number("frequency_ghz") <= 0 || number("frequency_ghz") > INT_MAX ||
      !integer("channels"))
    return error("invalid numeric geometry or mismatch with selected configuration");
  const Object &params = *cfg.getObject("dma_memory_params");
  const std::set<llvm::StringRef> backendKeys = {
      "curves", "window_beats", "converge", "latency_mode", "knee", "frequency_ghz",
      "beat_bytes", "channels", "enforce_peak", "queue_depth", "min_latency"};
  for (const auto &[key, value] : params) {
    bool nullable = key == "frequency_ghz" || key == "channels" || key == "latency_mode";
    if (!backendKeys.count(key) || (value.kind() == llvm::json::Value::Null && !nullable))
      return error("unsupported capability: dma_memory_params." + key.str());
  }
  for (const char *key : {"window_beats", "beat_bytes", "channels", "queue_depth", "min_latency"})
    if (params.get(key) && params.get(key)->kind() != llvm::json::Value::Null &&
        !params.getInteger(key))
      return error(std::string("unsupported capability: dma_memory_params.") + key);
  auto selectedQueue = params.getInteger("queue_depth");
  if (queue != selectedQueue.value_or(inFlight))
    return error("queue_depth mismatch with selected configuration");
  // Explicit backend settings must agree with the derived effective values.
  // Omitted settings are resolved by the model converter, not compiler defaults.
  for (const char *key : {"frequency_ghz", "window_beats", "channels", "beat_bytes",
                          "min_latency", "converge", "knee"}) {
    if (params.get(key) && params.get(key)->kind() != llvm::json::Value::Null &&
        params.getNumber(key) != d.getNumber(key))
      return error(std::string(key) + " mismatch with selected configuration");
  }
  for (const char *key : {"latency_mode", "enforce_peak"})
    if (params.get(key) && params.get(key)->kind() != llvm::json::Value::Null &&
        *params.get(key) != *d.get(key))
      return error(std::string(key) + " mismatch with selected configuration");
  double minimum = integer("min_latency"), epsilon = number("rounding_epsilon_cycles"), reset = number("reset_last_ready_cycles");
  if (d.getString("latency_mode") != "unloaded" || number("converge") != 1. ||
      d.getBoolean("enforce_peak") != true || minimum < 1 || minimum > INT_MAX ||
      number("knee") <= 0 || number("knee") > 1 || epsilon < 0 || epsilon >= 1 || reset < 0 || reset > INT_MAX)
    return error("unsupported controller mode, rounding or reset state");
  const Object *startup = d.getObject("startup_cycles");
  auto load = startup ? startup->getInteger("load") : std::nullopt;
  auto store = startup ? startup->getInteger("store") : std::nullopt;
  if (!startup || !exactKeys(*startup, {"load", "store"}) || !load || !store ||
      *load < 1 || *store < 1 || *load > INT_MAX || *store > INT_MAX)
    return error("requires bounded source-derived launch startup");
  ModelDMATiming result{beat, maxBytes, inFlight, queue, interval, int(*load), int(*store),
                        0., 0., 0., 0., epsilon, reset};
  const Array *curves = d.getArray("curves");
  if (!curves || curves->empty() || curves->size() > 101) return error("missing bounded curve coverage");
  int previous = -1;
  bool reads = false, writes = false;
  for (const auto &value : *curves) {
    const Object *curve = value.getAsObject();
    auto pct = curve ? curve->getInteger("read_percentage") : std::nullopt;
    if (!curve || !exactKeys(*curve, {"read_percentage", "unloaded_latency_cycles", "peak_beats_per_cycle", "cycles_per_beat", "points"}) ||
        !pct || *pct < 0 || *pct > 100 || *pct <= previous)
      return error("curve percentages must be sorted, unique and in range");
    previous = int(*pct);
    const Array *points = curve->getArray("points");
    auto latency = curve->getNumber("unloaded_latency_cycles"), peak = curve->getNumber("peak_beats_per_cycle");
    auto spacing = curve->getNumber("cycles_per_beat");
    if (!points || points->empty() || points->size() > 4096 || !latency || !peak || !spacing ||
        !std::isfinite(*latency) || !std::isfinite(*peak) || !std::isfinite(*spacing) ||
        *latency < minimum || *latency > INT_MAX || *peak <= 0 || *spacing <= 0 || *spacing > INT_MAX)
      return error("invalid finite curve coordinates");
    double previousBW = 0, firstLatency = 0;
    for (const auto &point : *points) {
      const Array *pair = point.getAsArray();
      auto bw = pair && pair->size() == 2 ? (*pair)[0].getAsNumber() : std::nullopt;
      auto lat = pair && pair->size() == 2 ? (*pair)[1].getAsNumber() : std::nullopt;
      if (!bw || !lat || !std::isfinite(*bw) || !std::isfinite(*lat) || *bw <= previousBW || *lat <= 0 || *lat > INT_MAX)
        return error("curve points must have finite positive coordinates and increasing bandwidth");
      if (previousBW == 0) firstLatency = *lat;
      previousBW = *bw;
    }
    // Both values are computed directly by the reviewed converter from these points.
    if (*latency != std::max(minimum, firstLatency) || *peak != previousBW || *spacing != 1. / *peak)
      return error("unloaded latency or peak pacing disagrees with curve points");
    if (*pct == 0) { result.writeLatency = *latency; result.writeSpacing = *spacing; writes = true; }
    if (*pct == 100) { result.readLatency = *latency; result.readSpacing = *spacing; reads = true; }
  }
  const Array *limits = d.getArray("limits");
  if (!reads || !writes || !limits || limits->empty()) return error("requires pure read/store endpoints and explicit scope limits");
  for (const auto &limit : *limits)
    if (!limit.getAsString() || limit.getAsString()->empty()) return error("invalid proposal scope limit");
  // Reject facts whose largest supported transfer cannot be represented by consumer arithmetic.
  for (bool store : {false, true}) {
    auto checked = result.estimate(result.maxBytes, store);
    if (!checked.error.empty()) return error(checked.error);
  }
  return result;
}
// No scheduling state is added: this qualifies the existing asynchronous policy.
llvm::Error parseDMAAdmission(const Object &root, const Object &cfg, llvm::StringRef payloadHash) {
  auto error = [](const std::string &why) { return factsError("DMA admission: " + why); };
  const Object *d = root.getObject("dma_admission");
  if (!d || !exactKeys(*d, {"schema", "age_convention", "contract", "source_sha256", "source_shapes",
                             "configuration", "issue", "semantics", "producer_sources", "validation"}) ||
      d->getString("schema") != "atlas.dma_admission.v1" ||
      d->getString("age_convention") != "issue_edge_zero" ||
      d->getString("contract") != "source-asynchronous-dma-admission-v1")
    return error("requires source-derived asynchronous DMA issue contract");
  const Object *sources = d->getObject("source_sha256"), *shapes = d->getObject("source_shapes");
  if (!sources || !shapes ||
      !exactKeys(*sources, {"core.py", "dma.py", "lsu.py", "bank_conflict.py", "memory_backend.py"}) ||
      !exactKeys(*shapes, {"core.py", "dma.py", "lsu.py", "bank_conflict.py", "memory_backend.py"}))
    return error("missing complete source identity");
  for (const auto &[name, value] : *sources)
    if (!value.getAsString() || !hashSyntax(*value.getAsString())) return error("malformed source hash");
  if (shapes->getString("core.py") != "faaabbe3276439f901d643b8c32e1746d6c2642fd18630db1283095fce1201d5" ||
      shapes->getString("lsu.py") != "2e1943c7ead558cbd3e83a4ed9fe5fffdfbf0e1d9027cee22e6d824b9eb16722" ||
      shapes->getString("bank_conflict.py") != "e93968432332701d9195dff520cdebfe2dd9db0ab86e0ef8e18492050c104467")
    return error("unreviewed Core/LSU/grant source shapes");
  for (const auto &[key, value] : model_schema::dmaIdentity()) {
    llvm::StringRef name(key);
    if (name == "curve_sha256") continue;
    auto actual = name.starts_with("shape:") ? shapes->getString(name.drop_front(6)) : sources->getString(name);
    if (!actual || *actual != value) return error("DMA/backend source differs from reviewed identity");
  }
  const Object *engine = root.getObject("engine_admission");
  const Object *contract = engine ? engine->getObject("source_contract") : nullptr;
  const Object *engineSources = contract ? contract->getObject("source_sha256") : nullptr;
  if (!engineSources) return error("missing source-qualified engine routing");
  for (const char *name : {"core.py", "lsu.py", "bank_conflict.py"})
    if (sources->getString(name) != engineSources->getString(name)) return error("engine source identities differ");
  const Object *configuration = d->getObject("configuration"), *issue = d->getObject("issue");
  if (!configuration || !exactKeys(*configuration, {"num_channels", "beat_bytes", "max_in_flight", "max_transfer_bytes", "vmem_bank_bytes"}) ||
      !issue || !exactKeys(*issue, {"ordinary_gap", "first_engine_age", "store_read_to_request_cycles"}) ||
      issue->getInteger("ordinary_gap") != 1 || issue->getInteger("first_engine_age") != 1 ||
      issue->getInteger("store_read_to_request_cycles") != 2)
    return error("unsupported launch/staging policy");
  for (auto [key, config] : std::initializer_list<std::pair<const char *, const char *>>{
      {"num_channels", "dma_num_channels"}, {"beat_bytes", "dma_beat_bytes"},
      {"max_in_flight", "dma_max_in_flight"}, {"max_transfer_bytes", "dma_max_transfer_bytes"},
      {"vmem_bank_bytes", "vmem_bank_bytes"}})
    if (!configuration->getInteger(key) || *configuration->getInteger(key) < 1 ||
        configuration->getInteger(key) != cfg.getInteger(config)) return error("configuration mismatch");
  const Object *semantics = d->getObject("semantics");
  if (!semantics || !exactKeys(*semantics, {"operands", "launch", "requests", "vmem_grants", "completion", "wait",
                                            "channel_reuse", "aliases", "curve_history", "fixed_dma_windows", "command_ring"}))
    return error("missing dynamic completion policy");
  for (auto [key, value] : std::initializer_list<std::pair<const char *, const char *>>{
      {"operands", "captured_at_issue"}, {"launch", "enqueue_after_registered_progress"},
      {"requests", "oldest_undispatched_slot"},
      {"vmem_grants", "lsu_priority_then_dma_load_write_then_dma_store_read"},
      {"completion", "all_beats_dispatched_and_acknowledged"},
      {"wait", "busy_flag_checked_before_engine_progress"}, {"channel_reuse", "matching_wait_required"},
      {"aliases", "existing_completion_conflict_policy"}, {"curve_history", "persists_across_wait_and_engine_drain"},
      {"fixed_dma_windows", "none"}, {"command_ring", "separate_unchanged_rtl_contract"}})
    if (semantics->getString(key) != value) return error("unsupported dynamic completion policy");
  const Object *producers = d->getObject("producer_sources");
  if (!producers || !exactKeys(*producers, {"atlas_derive_dma_admission.py", "atlas_validate_dma_admission.py"}))
    return error("missing producer traceability");
  for (const auto &[key, value] : *producers)
    if (!value.getAsString() || !hashSyntax(*value.getAsString())) return error("malformed producer hash");
  const Object *gate = d->getObject("validation");
  auto report = gate ? gate->getString("report_sha256") : std::nullopt;
  if (!gate || gate->getBoolean("passed") != true || !report || !hashSyntax(*report) ||
      (!payloadHash.empty() && gate->getString("payload_sha256") != payloadHash))
    return error("requires successful native validation bound to derived payload");
  for (const char *key : {"positive_runs", "negative_runs", "numerical_checks"})
    if (!gate->getInteger(key) || *gate->getInteger(key) < 1) return error("missing native mechanism coverage");
  return llvm::Error::success();
}


llvm::Expected<ModelEngineAdmission> parseEngineAdmission(
    const Object &document, const Object &model,
    const std::map<std::string, ModelTimingBlock> &blocks, llvm::StringRef payloadHash) {
  auto error = [](const std::string &why) { return factsError("engine admission: " + why); };
  if (!exactKeys(document, {"schema", "age_convention", "contract", "model", "source_contract",
                             "vpu", "xlu", "vlsu", "scalar_lsu", "policy", "validation"}) ||
      document.getString("schema") != "atlas.engine_admission.v1" ||
      document.getString("age_convention") != "issue_edge_zero" ||
      document.getString("contract") != "source-engine-initiation-v1")
    return error("unsupported schema or source contract");
  const Object *identity = document.getObject("model"), *source = document.getObject("source_contract");
  if (!identity || !exactKeys(*identity, {"source_sha256", "configuration_sha256", "numerical_data_sha256"}))
    return error("missing model identity");
  for (const auto &[key, value] : *identity)
    if (!model.get(key) || value != *model.get(key)) return error("model identity mismatch");
  const Object *units = source ? source->getObject("execution_units") : nullptr;
  const Object *config = model.getObject("configuration");
  const Object *hashes = source ? source->getObject("source_sha256") : nullptr;
  if (!source || !exactKeys(*source, {"source_sha256", "core_init_shape", "execution_units"}) ||
      source->getString("core_init_shape") != "5abd67591485003d3f345e7b4f15564b3b3d8a4ff8449ed3efd5a392cf03adf9" ||
      !units || !config || !config->getObject("execution_units") ||
      *units != *config->getObject("execution_units") || !hashes ||
      !exactKeys(*hashes, {"mxu.py", "vpu.py", "bank_conflict.py", "lsu.py", "core.py", "xlu.py"}))
    return error("missing reviewed source/engine routing identity");
  for (const auto &[key, value] : *hashes)
    if (!value.getAsString() || !hashSyntax(*value.getAsString())) return error("malformed source hash");
  const Object *xlu = document.getObject("xlu"), *vlsu = document.getObject("vlsu"), *policy = document.getObject("policy");
  const Object *scalar = document.getObject("scalar_lsu");
  const Object *paths = vlsu ? vlsu->getObject("initiation_ages") : nullptr;
  auto xblock = blocks.find("xlu.transpose"), load = blocks.find("vlsu.vload"), store = blocks.find("vlsu.vstore");
  if (!xlu || !exactKeys(*xlu, {"independent_from_vpu", "initiation", "initiation_age"}) ||
      xlu->getBoolean("independent_from_vpu") != true || xlu->getString("initiation") != "prior_first_free" ||
      !xlu->getInteger("initiation_age") || *xlu->getInteger("initiation_age") < 1 ||
      *xlu->getInteger("initiation_age") > 4096 ||
      (xblock != blocks.end() && xlu->getInteger("initiation_age") != xblock->second.firstFreeAge) ||
      !vlsu || !exactKeys(*vlsu, {"independent_load_store_paths", "initiation", "initiation_ages"}) ||
      vlsu->getBoolean("independent_load_store_paths") != true ||
      vlsu->getString("initiation") != "same_path_prior_first_free" ||
      !paths || !exactKeys(*paths, {"vlsu.vload", "vlsu.vstore"}) ||
      !paths->getInteger("vlsu.vload") || !paths->getInteger("vlsu.vstore") ||
      *paths->getInteger("vlsu.vload") < 1 || *paths->getInteger("vlsu.vload") > 4096 ||
      *paths->getInteger("vlsu.vstore") < 1 || *paths->getInteger("vlsu.vstore") > 4096 ||
      (load != blocks.end() && paths->getInteger("vlsu.vload") != load->second.firstFreeAge) ||
      (store != blocks.end() && paths->getInteger("vlsu.vstore") != store->second.firstFreeAge) ||
      !scalar || !exactKeys(*scalar, {"load_response_gap", "store_busy", "independent_vector_paths"}) ||
      !scalar->getInteger("load_response_gap") || *scalar->getInteger("load_response_gap") < 1 ||
      *scalar->getInteger("load_response_gap") > 4096 || scalar->getString("store_busy") != "none" ||
      scalar->getBoolean("independent_vector_paths") != true ||
      !policy || !exactKeys(*policy, {"mreg_ports", "unknown_vmem", "scalar_lsu", "mxu", "waits"}) ||
      policy->getString("mreg_ports") != "vpu_same_logical_register_row_read_coalescing" ||
      policy->getString("unknown_vmem") != "aliases_all_banks" ||
      policy->getString("scalar_lsu") != "existing_alias_dependency_and_writeback_checks" ||
      policy->getString("mxu") != "existing_source_overlap_rules" ||
      policy->getString("waits") != "invalidate_live_admission")
    return error("unsupported engine/resource policy");
  const Object *gate = document.getObject("validation");
  auto report = gate ? gate->getString("report_sha256") : std::nullopt;
  if (!gate || gate->getBoolean("passed") != true || !report || !hashSyntax(*report) ||
      (!payloadHash.empty() && gate->getString("payload_sha256") != payloadHash))
    return error("requires successful native validation bound to derived payload");
  for (const char *key : {"positive_runs", "negative_runs", "numerical_checks"})
    if (!gate->getInteger(key) || *gate->getInteger(key) < 1) return error("missing native mechanism coverage");
  const Object *vpu = document.getObject("vpu");
  const Object *final = vpu ? vpu->getObject("final_write_age") : nullptr;
  const Array *doubles = vpu ? vpu->getArray("double") : nullptr;
  const Array *groups = vpu ? vpu->getArray("logic_groups") : nullptr;
  if (!vpu || !exactKeys(*vpu, {"capacity", "live", "final_write_age", "logic_groups", "double"}) ||
      vpu->getInteger("capacity") != 2 || vpu->getString("live") != "age_lt_final_write" ||
      !final || final->empty() || !doubles || !groups)
    return error("unsupported VPU live-set predicate");
  ModelEngineAdmission result;
  result.vpuCapacity = int(*vpu->getInteger("capacity"));
  result.scalarLoadGap = int(*scalar->getInteger("load_response_gap"));
  for (const auto &[name, event] : {std::pair<const char *, const char *>{"scalar_lsu.load", "load_write"},
                                   {"scalar_lsu.scale_load", "scale_write"}}) {
    auto block = blocks.find(name);
    if (block != blocks.end()) {
      auto response = block->second.events.find(event);
      if (response == block->second.events.end() || response->second.firstAge != result.scalarLoadGap)
        return error("scalar load issue gap differs from source-derived response event");
    }
  }
  for (const auto &[name, block] : blocks)
    if (block.engine == "vpu") {
      auto age = final->getInteger(name);
      int lastWrite = -1;
      for (const auto &[event, sweep] : block.events)
        if (llvm::StringRef(event).starts_with("write")) lastWrite = std::max(lastWrite, sweep.lastAge);
      if (!age || *age != lastWrite || *age != block.firstFreeAge - 1)
        return error("VPU final write differs from selected operation events");
      result.vpuFinalWrite[name] = int(*age);
    }
  if (result.vpuFinalWrite.size() != final->size()) return error("VPU operation coverage mismatch");
  auto names = [&](const Array &array, std::set<std::string> &out) {
    for (const auto &value : array) {
      auto name = value.getAsString();
      if (!name || !result.vpuFinalWrite.count(name->str()) || !out.insert(name->str()).second) return false;
    }
    return true;
  };
  if (!names(*doubles, result.vpuDouble)) return error("invalid VPU double-input domain");
  for (const auto &[mnemonic, name] : vpuBlocks())
    if (result.vpuFinalWrite.count(name)) {
      const auto *op = findOp(mnemonic);
      bool expected = op && (op->twoInput || op->opClass == OpClass::VpuRowReduce);
      if (bool(result.vpuDouble.count(name)) != expected) return error("VPU double-input domain differs from operand vocabulary");
    }
  for (const auto &value : *groups) {
    const Array *group = value.getAsArray();
    std::set<std::string> members;
    if (!group || group->empty() || !names(*group, members)) return error("invalid VPU shared-logic group");
    result.vpuLogicGroups.push_back(std::move(members));
  }
  return result;
}
} // namespace

TimingRuleResult<int> ModelDMATiming::estimate(int64_t bytes, bool store) const {
  auto error = [](const char *why) -> TimingRuleResult<int> { return {0, std::string("model DMA estimate: ") + why}; };
  if (bytes < beatBytes || bytes > maxBytes || bytes % beatBytes)
    return error("transfer is outside the derived aligned geometry");
  // Cold single-command replay: pure history selects an endpoint on every beat.
  // Completion is the last acknowledgement age relative to the command launch.
  // Capacity uses registered in-flight state; backend acceptance observes D-before-A.
  std::deque<int64_t> pending;
  int64_t issue = store ? storeStartup : loadStartup, last = 0;
  double ready = resetReady, latency = store ? writeLatency : readLatency;
  double spacing = store ? writeSpacing : readSpacing;
  for (int64_t beat = 0; beat < bytes / beatBytes; ++beat) {
    while (!pending.empty() && pending.front() < issue) pending.pop_front();
    if (pending.size() >= size_t(maxInFlight)) issue = std::max(issue, pending.front() + issueInterval);
    if (pending.size() >= size_t(queueDepth)) issue = std::max(issue, pending.front());
    while (!pending.empty() && pending.front() <= issue) pending.pop_front();
    ready = std::max(double(issue) + latency, ready + spacing);
    double rounded = std::ceil(ready - epsilon);
    if (!std::isfinite(rounded) || rounded < 0 || rounded > INT_MAX)
      return error("isolated completion exceeds the nonnegative cycle domain");
    // One response per engine edge, including when several ready values round together.
    last = std::max(int64_t(rounded), last + issueInterval);
    if (last > INT_MAX) return error("isolated completion exceeds the nonnegative cycle domain");
    pending.push_back(last);
    issue += issueInterval;
  }
  return {int(last), {}};
}

TimingRuleResult<int> ModelTimingFacts::estimateDMACycles(int64_t bytes, bool store) const {
  if (!dmaTiming) return {0, "model: DMA proposal requires validated dma_timing facts"};
  return dmaTiming->estimate(bytes, store);
}

llvm::Expected<ModelTimingFacts> ModelTimingFacts::parseImpl(
    llvm::StringRef bytes, llvm::StringRef expectedSha256, bool dmaWait, bool testObservations) {
  if (!hashSyntax(expectedSha256)) return factsError("facts SHA-256 is required");
  if (modelSha256(bytes) != expectedSha256)
    return factsError("selected facts hash mismatch");
  auto parsed = llvm::json::parse(bytes);
  if (!parsed) return factsError(llvm::toString(parsed.takeError()));
  const Object *root = parsed->getAsObject();
  if (!root || root->getString("schema") != "atlas.model_timing.v1")
    return factsError("expected schema atlas.model_timing.v1");
  if (!testObservations && root->getBoolean("test_only") == true)
    return factsError("test-only observations cannot select a production provider");
  if (testObservations && !root->get("derivation_contract") && root->getBoolean("test_only") != true)
    return factsError("raw observation fixtures require the explicit test_only marker");
  for (const char *key : {"hw_ir", "assertions", "op_pairs", "pairs", "chains", "capacity"})
    if (root->get(key)) return factsError("per-operation model facts cannot carry RTL or overlap claims");
  std::string payloadHash;
  if (!testObservations) {
    auto contract = validateDerivedContract(*root);
    if (!contract) return contract.takeError();
    payloadHash = std::move(*contract);
  }
  if ((!testObservations || root->get("vocabulary_sha256")) &&
      root->getString("vocabulary_sha256") != model_schema::vocabularySha256())
    return factsError("facts vocabulary differs from the compiled operand and relation schema");
  const Object *model = root->getObject("model");
  if (!model) return factsError("missing model provenance");
  ModelTimingFacts result;
  result.testOverlap = testObservations;
  if (const auto *rules = root->get("overlap_rules_sha256")) {
    auto hash = rules->getAsString();
    if (!hash || !hashSyntax(*hash)) return factsError("overlap_rules_sha256 must be a SHA-256 digest");
    result.overlapRules = hash->str();
  }
  if (!testObservations || root->get("replay_timing")) {
    const Object *document = root->getObject("replay_timing");
    if (!document) return factsError("missing source-derived replay timing contract");
    auto program = ModelReplayProgram::parse(*document);
    if (!program) return factsError(llvm::toString(program.takeError()));
    result.replay = std::move(*program);
  }
  if (!testObservations || root->get("proposal_costs")) {
    const Object *document = root->getObject("proposal_costs");
    if (!document) return factsError("missing source-derived representative cost probes");
    const Object *proof = document->getObject("timing_class");
    const Object *probes = document ? document->getObject("probes") : nullptr;
    if (!document || !exactKeys(*document, {"schema", "timing_class", "probes", "placement_policy"}) ||
        document->getString("schema") != "atlas.proposal_costs.v1" ||
        document->getString("placement_policy") != search_policy::kProposalOperandPlacement || !proof || !probes ||
        !exactKeys(*proof, {"kind", "vpu", "mxu", "memory", "interval_scope"}) ||
        proof->getString("kind") != "guarded_operand_independent_isolated_ages" ||
        proof->getString("vpu") != "mnemonic_latency_and_register_independent_row_predicates" ||
        proof->getString("mxu") != "guarded_age_geometry_expressions" ||
        proof->getString("memory") != "guarded_age_geometry_predicates" ||
        proof->getString("interval_scope") != "selected_representative_pair_only")
      return factsError("requires source-derived representative cost probes and timing-class proof");
    for (const auto &[name, value] : *probes) {
      const Array *pair = value.getAsArray();
      if (!pair || pair->size() != 2) return factsError("representative cost needs exactly two probes");
      const OpInfo *op = findOp(name.str());
      std::pair<Instr, Instr> decoded;
      for (size_t index = 0; index < pair->size(); ++index) {
        const Object *row = (*pair)[index].getAsObject();
        if (!row || !exactKeys(*row, {"rd", "rs1", "rs2", "imm"}))
          return factsError("invalid representative cost operand record");
        Instr &in = index ? decoded.second : decoded.first;
        in.op = op;
        for (auto [field, destination] : {std::pair<const char *, int *>{"rd", &in.rd},
                                         {"rs1", &in.rs1}, {"rs2", &in.rs2}}) {
          auto number = row->getInteger(field);
          if (!number || *number < 0 || *number > 4095) return factsError("cost probe operand exceeds supported domain");
          *destination = int(*number);
        }
        auto immediate = row->getInteger("imm");
        if (!immediate || *immediate != 0) return factsError("representative cost requires zero immediate");
        in.imm = *immediate;
      }
      if (op) result.probes.emplace(name.str(), decoded);
    }
  }
  int readLatency = 1;
  if (!testObservations || root->get("mxu_weight_timing")) {
    const Object *document = root->getObject("mxu_weight_timing");
    if (!document) return factsError("missing source-derived MXU weight-use windows");
    auto parsedWeights = parseWeightUses(*document);
    if (!parsedWeights) return parsedWeights.takeError();
    result.weightUses = std::move(*parsedWeights);
  }
  if (!testObservations || root->get("frontend_timing")) {
    const Object *document = root->getObject("frontend_timing");
    if (!document) return factsError("missing source-derived frontend timing contract");
    auto parsedFrontend = parseFrontend(*document);
    if (!parsedFrontend) return parsedFrontend.takeError();
    result.frontend = std::move(*parsedFrontend);
  }
  if (!testObservations || root->get("scratchpad_contract")) {
    auto parsedScratchpad = parseScratchpadContract(*root);
    if (!parsedScratchpad) return parsedScratchpad.takeError();
    readLatency = *parsedScratchpad;
  }
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
  if (!configuration || !configurationJson)
    return factsError("missing explicit model configuration profile");
  if (auto capability = unsupportedCapability(*configuration); !capability.empty())
    return factsError("unsupported capability: " + capability);
  if (auto error = checkArchitecture(*root, *configuration)) return std::move(error);
  const Object &architecture = *root->getObject("architecture");
  result.dmaChannels = int(*architecture.getInteger("dma_channels"));
  result.geometry = {int(*architecture.getInteger("mreg_bytes")),
      int(*architecture.getInteger("mreg_row_bytes")), int(*architecture.getInteger("mreg_rows")),
      int(*architecture.getInteger("scalar_word_bytes")), int(*architecture.getInteger("bf16_registers_per_tile")),
      int(*architecture.getInteger("bf16_element_bytes")), int(*architecture.getInteger("fp8_registers_per_tile")),
      int(*architecture.getInteger("fp8_element_bytes"))};
  const Object *vocabulary = root->getObject("operand_vocabulary");
  if (!vocabulary || modelSha256(canonicalJson(llvm::json::Value(Object(*vocabulary)))) !=
                         model_schema::vocabularySha256())
    return factsError("source-derived operand vocabulary differs from the compiled compiler-op adapter");
  if (auto error = checkIsaMetadata(*root)) return std::move(error);

  auto decodedConfiguration = llvm::json::parse(*configurationJson);
  if (!decodedConfiguration) return factsError(llvm::toString(decodedConfiguration.takeError()));
  // Hash the producer's exact serialization after checking its decoded values.
  // LLVM and Python spell valid floats differently (1 vs 1.0, exponents).
  if (*decodedConfiguration != llvm::json::Value(Object(*configuration)))
    return factsError("configuration_json differs from the explicit configuration");
  if (modelSha256(*configurationJson) != *model->getString("configuration_sha256"))
    return factsError("configuration content hash mismatch");
  for (const auto &[key, value] : *model) {
    if (key == "replay_timing_sha256" || key == "proposal_costs_sha256")
      return factsError("reserved derived provenance field " + key.str());
    if (key == "configuration") continue;
    auto text = value.getAsString();
    if (!text || (llvm::StringRef(key).ends_with("sha256") && !hashSyntax(*text)))
      return factsError("invalid model provenance field " + key.str());
    result.provenance[key.str()] = text->str();
  }
  // Derive extension identities after model metadata so arbitrary provenance
  // keys cannot override the selected semantic program/cost binding.
  for (const char *key : {"replay_timing", "proposal_costs"})
    if (const Object *document = root->getObject(key))
      result.provenance[std::string(key) + "_sha256"] = modelSha256(canonicalJson(llvm::json::Value(Object(*document))));
  const Object *geometry = root->getObject("geometry");
  if (!geometry || geometry->getInteger("rows") != kAtlasMregRows || geometry->getInteger("columns") != kAtlasMregRowBytes ||
      geometry->getInteger("mreg_count") != kAtlasTensorRegisters)
    return factsError("only 32x32 geometry with 64 MREGs is supported");
  auto accepted = [](const Object *object) {
    return object && object->getBoolean("accepted") == true && object->getBoolean("drained") == true;
  };
  if (root->getString("age_convention") != "issue_edge_zero" || !accepted(root->getObject("observation")))
    return factsError("requires accepted, drained observations with issue_edge_zero ages");
  const Array *timing = root->getArray("op_timing");
  if (!timing || timing->empty()) return factsError("missing op_timing coverage");
  const Object *releases = root->getObject("reservation_releases");
  if ((!testObservations || root->get("reservation_releases")) &&
      (!releases || releases->size() != timing->size()))
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
    if (!assumptions || assumptions->getInteger("scratchpad_read_latency") != readLatency)
      return factsError("operation scratchpad latency differs from its explicit target assumption");
    ModelTimingBlock block{engine->str(), {}, int(*free), readLatency};
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
        ModelReservationRelease resource{int(*first), int(*last), {}};
        if (!unit) {
          const Array *banks = release->getArray("banks");
          if (!banks || banks->empty()) return factsError("reservation needs explicit probe banks");
          int previous = -1;
          for (const auto &bank : *banks) {
            auto index = bank.getAsInteger();
            if (!index || *index <= previous || *index >= (mreg ? kAtlasTensorRegisters : kVmemBanks))
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
    if (!testObservations && !completeReleaseCoverage(block))
      return factsError("reservation resource coverage differs from operation events: " + name->str());
    if (!result.blocks.emplace(name->str(), std::move(block)).second)
      return factsError("duplicate operation " + name->str());
  }
  if (!testObservations || root->get("engine_admission")) {
    const Object *document = root->getObject("engine_admission");
    if (!document) return factsError("missing source-derived engine admission");
    auto admission = parseEngineAdmission(*document, *model, result.blocks, payloadHash);
    if (!admission) return admission.takeError();
    result.admission = std::move(*admission);
  }
  if (!testObservations || root->get("dma_admission"))
    if (auto error = parseDMAAdmission(*root, *configuration, payloadHash)) return error;
  if (const auto *dmaFacts = root->get("dma_timing")) {
    const Object *document = dmaFacts->getAsObject();
    if (!document) return factsError("dma_timing must be an object");
    auto parsedDMA = parseDMATiming(*document, *configuration, payloadHash);
    if (!parsedDMA) return parsedDMA.takeError();
    result.dmaTiming = std::move(*parsedDMA);
    result.provenance["dma_curve_sha256"] = document->getString("curve_sha256")->str();
  }
  result.sha256 = expectedSha256.str();
  result.dma = dmaWait;
  return result;
}

llvm::Expected<ModelTimingFacts> mlir::atlas::timing::parseModelTimingFacts(
    llvm::StringRef bytes, llvm::StringRef expectedSha256, bool dmaWait) {
  return ModelTimingFacts::parseImpl(bytes, expectedSha256, dmaWait, false);
}

llvm::Expected<ModelTimingFacts> mlir::atlas::timing::parseModelTimingFactsForTesting(
    llvm::StringRef bytes, llvm::StringRef expectedSha256, bool dmaWait) {
  return ModelTimingFacts::parseImpl(bytes, expectedSha256, dmaWait, true);
}

llvm::Expected<ModelTimingFacts> mlir::atlas::timing::loadModelTimingFacts(
    llvm::StringRef path, llvm::StringRef expectedSha256, bool dmaWait) {
  auto buffer = llvm::MemoryBuffer::getFile(path);
  if (!buffer) return factsError("cannot read " + path.str());
  return parseModelTimingFacts((*buffer)->getBuffer(), expectedSha256, dmaWait);
}

TimingRuleResult<int> ModelTimingFacts::issueGap(const Instr &in) const {
  if (!in.op) return {0, modelTimingDiagnostic("invalid instruction for frontend issue gap")};
  if (!frontend) return {naturalGap(in), {}}; // Explicit raw-observation test API only.
  return {in.op->opClass == OpClass::Delay ? frontend->delayBase + int(in.imm & frontend->delayMask) :
                                            frontend->ordinaryGap, {}};
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
  auto synchronous = [&]() {
    if (!frontend) return f; // Explicit raw-observation test API only.
    auto found = frontend->instructions.find(name);
    if (found == frontend->instructions.end()) return reject("missing frontend instruction access coverage");
    f.accesses.clear();
    f.holds.clear();
    for (const auto &access : found->second) {
      int reg = access.operand == "rd" ? in.rd : access.operand == "rs1" ? in.rs1 : in.rs2;
      if (!xreg(reg)) return reject("unsupported frontend register operand");
      if (!access.scale && reg == 0) continue;
      bool control = op.opClass == OpClass::Branch || op.opClass == OpClass::Jump || op.opClass == OpClass::Csr;
      int age = access.write ? frontend->writeAge : control ? frontend->controlReadAge : frontend->readAge;
      f.accesses.push_back({access.scale ? Res::EReg : Res::XReg, access.write, reg, 1, age, 1});
      f.doneAge = std::max(f.doneAge, age);
      if (access.write) f.holds.push_back({Unit::ScalarWriteback, 0, age, age});
    }
    return f;
  };
  if (in.release && op.opClass != OpClass::Csr) return reject("unsupported completion annotation");

  if (op.opClass == OpClass::Alu && name != "auipc") {
    if (!xreg(in.rd) || !xreg(in.rs1) || !xreg(in.rs2)) return reject("unsupported scalar operands");
    if (hasOperand(op, "x1")) x(in.rs1, false);
    if (hasOperand(op, "x2")) x(in.rs2, false);
    x(in.rd, true);
    if (in.rd) f.holds.push_back({Unit::ScalarWriteback, 0, 0, 0});
    return synchronous();
  }
  if (op.opClass == OpClass::ScaleImm) {
    if (!xreg(in.rd)) return reject("unsupported scale register");
    f.accesses.push_back({Res::EReg, true, in.rd, 1, 0, 1});
    f.holds.push_back({Unit::ScalarWriteback, 0, 0, 0});
    return synchronous();
  }
  if (name == "delay" || name == "ecall") return synchronous();
  // The verifier drains every engine before a successor starts, so redirect
  // latency need only cover the delay slot; operands are read at issue.
  if (op.opClass == OpClass::Branch) {
    if (!xreg(in.rs1) || !xreg(in.rs2)) return reject("unsupported branch operands");
    x(in.rs1, false);
    x(in.rs2, false);
    return synchronous();
  }
  if (name == "jal") {
    if (in.rd != 0 || in.rs1 != 0) return reject("JAL must not link");
    return synchronous();
  }
  if (name == "csrrw" && in.rd == 0 && in.imm == 0xC10 && xreg(in.rs1)) {
    x(in.rs1, false);
    return synchronous();
  }

  if (op.engine == Engine::Dma) {
    if (!dma) return reject("DMA requires the model completion-wait policy");
    if (!dmaTiming) return reject("DMA requires validated dma_timing facts");
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
    auto estimate = estimateDMACycles(bytes, !load);
    if (!estimate.error.empty()) return reject(estimate.error.c_str());
    f.dmaCycles = estimate.value;
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
    if (!admission) f.holds.push_back({Unit::ScalarLoad, 0, 0, busy});
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
        if (auto [at, inserted] = banks.emplace(reg & 31, reg);
            !inserted && !(group == &f.mregReads && at->second == reg))
          fill.fail("VPU register operands must occupy distinct physical banks (register mod 32): m" +
                    std::to_string(at->second) + " and m" + std::to_string(reg) + " share bank " +
                    std::to_string(reg & 31));
    }
    int busy = fill.busyEnd();
    if (!admission) fill.hold({Unit::Xlu}, 0, busy);
    return fill.done(busy);
  }

  if (op.mxu >= 0) return mxuFootprint(in, blocks, weightUses);

  return reject("operation is outside the selected bounded scope");
}

namespace {
bool overlaps(const Access &a, const Access &b) {
  return a.res == b.res && (a.anywhere || b.anywhere || (a.first < b.first + b.count && b.first < a.first + a.count));
}

std::optional<ModelOverlapCommand> overlapCommand(const Instr &in, const Footprint &f, int cycle,
                                                  const ModelTimingFacts &facts);

Dependence observedDependence(const Instr &a, const Footprint &fa, const Instr &b, const Footprint &fb) {
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
  for (const Access &x : fa.accesses) {
    if (x.atCompletion) continue;
    for (const Access &y : fb.accesses) {
      if (!overlaps(x, y) || (!x.write && !y.write)) continue;
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

bool sameEngine(const OpInfo &a, const OpInfo &b, const ModelTimingFacts *facts = nullptr) {
  if (facts && facts->engineAdmission()) {
    // LSU's scalar response and separate vector paths have their own predicates.
    if (a.engine == Engine::Lsu && b.engine == Engine::Lsu) return false;
    return a.engine == b.engine;
  }
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
    const auto *admission = facts->engineAdmission();
    std::vector<Port> ports = requests(f, cycle, in.op->engine == Engine::Vpu);
    for (size_t i = 0; i < ports.size(); ++i)
      for (size_t j = 0; j < i; ++j)
        if (portConflict(ports[i], ports[j])) return {{}, "instruction has conflicting MREG port requests"};
    if (admission) {
      std::string structural = admissionConflict(in, cycle);
      if (!structural.empty()) return {structural, {}};
    }
    bool directVpu = admission && in.op->engine == Engine::Vpu;
    bool waiver = !directVpu && overlap && admitsInsertion(in, f, cycle, context);
    for (const Issued &prior : issued) {
      if (prior.cycle + prior.f.doneAge < cycle || cycle + f.doneAge < prior.cycle) continue;
      if (admission) {
        const Footprint &older = prior.cycle < cycle ? prior.f : f;
        const Footprint &newer = prior.cycle < cycle ? f : prior.f;
        int gap = std::abs(int64_t(cycle) - prior.cycle);
        auto intersects = [](const std::vector<int> &a, const std::vector<int> &b) {
          for (int reg : a) if (std::find(b.begin(), b.end(), reg) != b.end()) return true;
          return false;
        };
        if ((gap <= older.writeRelease && (intersects(older.mregWrites, newer.mregReads) ||
                                          intersects(older.mregWrites, newer.mregWrites))) ||
            (gap <= older.readRelease && !newer.writeDuringRead && intersects(older.mregReads, newer.mregWrites)))
          return {"source-derived logical MREG reservation remains live", {}};
      }
      if (sameEngine(*in.op, *prior.in.op, facts.get()) && !directVpu && cycle <= prior.cycle + prior.f.doneAge &&
          prior.cycle <= cycle + f.doneAge && !waiver)
        return {"model isolated engine window remains live", {}};
      for (const Hold &a : prior.f.holds)
        for (const Hold &b : f.holds)
          if (a.unit == b.unit && a.index == b.index && prior.cycle + a.from <= cycle + b.to &&
              cycle + b.from <= prior.cycle + a.to)
            if (!(waiver && sameEngine(*in.op, *prior.in.op, facts.get()) &&
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
    issued.push_back({in, f, cycle, requests(f, cycle, in.op->engine == Engine::Vpu), true, context});
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
  struct Port { int bank, reg, row, from, cycle; bool write, vpu; };
  struct Issued {
    Instr in; Footprint f; int cycle; std::vector<Port> ports; bool eligible;
    TimingContext context;
  };
  static bool scalarContext(const Instr &in, const TimingContext &context) {
    return context && context->matches(in) && (in.op->name == "lw" || in.op->name == "sw");
  }
  std::string admissionConflict(const Instr &in, int cycle) const {
    const auto &admission = *facts->engineAdmission();
    for (const Issued &prior : issued)
      if (prior.cycle == cycle) return "source-derived frontend has one issue slot per edge";
    auto scalarLoad = [](const Instr &command) {
      return command.op->name == "lw" || command.op->name == "seld";
    };
    if (scalarLoad(in))
      for (const Issued &prior : issued)
        if (scalarLoad(prior.in) && std::abs(int64_t(cycle) - prior.cycle) < admission.scalarLoadGap)
          return "source-derived scalar load response remains pending";
    if (in.op->engine != Engine::Vpu) return {};
    struct Command { std::string name; int cycle; bool eligible, candidate; };
    std::vector<Command> commands;
    auto name = [](const Instr &command) { return std::string(vpuBlocks().at(command.op->name)); };
    commands.push_back({name(in), cycle, true, true});
    for (const Issued &prior : issued)
      if (prior.in.op->engine == Engine::Vpu) commands.push_back({name(prior.in), prior.cycle, prior.eligible, false});
    std::sort(commands.begin(), commands.end(), [](const Command &a, const Command &b) { return a.cycle < b.cycle; });
    // A wait invalidates only the live epoch containing this insertion.
    // Retired epochs cannot prevent independently drained work from restarting.
    std::vector<Command> affected;
    int end = -1;
    bool contains = false;
    for (const auto &command : commands) {
      if (command.cycle > end) {
        if (contains) break;
        affected.clear();
      }
      affected.push_back(command);
      contains |= command.candidate;
      end = std::max(end, command.cycle + facts->block(command.name)->firstFreeAge - 1);
    }
    commands = std::move(affected);
    for (size_t j = 0; j < commands.size(); ++j) {
      const auto &current = commands[j];
      size_t live = 0;
      for (size_t i = 0; i < j; ++i) {
        const auto &prior = commands[i];
        if (current.cycle - prior.cycle >= admission.vpuFinalWrite.at(prior.name)) continue;
        if (!current.eligible || !prior.eligible) return "wait invalidates source-derived VPU live admission";
        if (++live >= size_t(admission.vpuCapacity)) return "source-derived VPU issue slots occupied";
        bool shared = current.name == prior.name;
        for (const auto &group : admission.vpuLogicGroups)
          shared |= group.count(current.name) && group.count(prior.name);
        if (shared || admission.vpuDouble.count(current.name) || admission.vpuDouble.count(prior.name))
          return "source-derived VPU issue logic occupied";
      }
    }
    return {};
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
      if (!sameEngine(*in.op, *prior.in.op, facts.get())) continue;
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
                              !overlap->admits({live[0]->command, entry.command}, false, "reservation"))) return false;
      // A retired MXU command can determine the port held by the remaining live push.
      if (live.size() == 1 && epoch.size() >= 3 && overlap->rejectsPortHistory(epoch)) return false;
      if (live.size() >= 2 && (!eligible || !overlap->admits(epoch, chains, "reservation"))) return false;
      history.push_back(&entry);
    }
    return true;
  }
  static std::vector<Port> requests(const Footprint &f, int cycle, bool vpu) {
    std::vector<Port> ports;
    for (const Access &a : f.accesses)
      if (a.res == Res::MReg)
        for (int i = 0; i < a.count; ++i) {
          int element = a.first + i, at = cycle + a.age + i * a.step;
          ports.push_back({element / 32 % 32, element / 32, element % 32, at, at, a.write, vpu});
        }
    return ports;
  }
  static bool portConflict(const Port &a, const Port &b) {
    // The guarded VPU _read cache coalesces identical logical register/row
    // requests on one edge, including duplicate sources in one instruction.
    // XLU has no such cache; wait-expanded uncertain requests remain strict.
    if (!a.write && !b.write && a.vpu && b.vpu && a.reg == b.reg && a.row == b.row &&
        a.from == a.cycle && b.from == b.cycle && a.cycle == b.cycle)
      return false;
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

TimingRuleResult<std::pair<Instr, Instr>> ModelTimingFacts::costProbes(const OpInfo &op) const {
  auto found = probes.find(op.name);
  if (found != probes.end()) return {found->second, {}};
  if (!testOverlap) return {{}, modelTimingDiagnostic("missing generated representative cost operands for " + std::string(op.name))};
  // Explicit raw-observation tests retain their historical fixture operands.
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
}

TimingProvider mlir::atlas::timing::modelTimingProvider(std::shared_ptr<const ModelTimingFacts> facts,
                                                      std::shared_ptr<const ModelOverlapFacts> overlap, bool chains) {
  TimingProvider provider;
  provider.id = kModelTimingProviderId.str();
  provider.proposalFactsSha256 = facts->factsSha256();
  provider.proposalGeometry = facts->proposalGeometry();
  provider.proposalDMACapacity = facts->dmaCapacity();
  provider.frontendSpacing = facts->frontendSpacing();
  if (facts->replayProgram()) provider.proposalReplay = std::make_shared<const ModelReplayProgram>(*facts->replayProgram());
  provider.dmaEstimate = [facts](int64_t bytes, bool store) {
    auto result = facts->estimateDMACycles(bytes, store);
    if (!result.error.empty()) result.error = modelTimingDiagnostic(result.error);
    return result;
  };
  provider.costProbes = [facts](const OpInfo &op) { return facts->costProbes(op); };
  provider.validateScope = [](const AtlasStream &stream) {
    auto error = modelScope(stream);
    return error.empty() ? error : modelTimingDiagnostic(error);
  };
  provider.footprint = [facts](const Instr &in, const RegValues &regs) {
    auto result = facts->resolve(in, regs);
    if (!result.error.empty()) result.error = modelTimingDiagnostic(result.error);
    return result;
  };
  provider.dependence = [](const Instr &a, const Footprint &fa, const Instr &b, const Footprint &fb) {
    return TimingRuleResult<Dependence>{observedDependence(a, fa, b, fb), {}};
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
  provider.fillPendingDelaySlots = true;
  provider.waitReusedDMASlots = true;
  provider.issueGap = [facts](const Instr &in) { return facts->issueGap(in); };
  provider.createReservations = [facts, overlap, chains] {
    return TimingRuleResult<std::unique_ptr<TimingReservations>>{
        std::make_unique<ObservedReservations>(facts, overlap, chains), {}};
  };
  return provider;
}

static TimingRuleResult<TimingProvider> modelTimingProviderFactoryImpl(ModuleOp module) {
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
    // Legacy selected modules could only carry the locked curve. Preserve that
    // default success path; a different curve always needs its explicit digest.
    if (!retained && key == "dma_curve_sha256" && value == model_schema::dmaIdentity().at("curve_sha256"))
      continue;
    if (!retained || retained.getValue() != value) return {{}, "model evidence provenance mismatch: " + key};
  }
  std::shared_ptr<const ModelOverlapFacts> overlap;
  auto pairs = selection.getAs<StringAttr>("pairs"), pairsHash = selection.getAs<StringAttr>("pairs_sha256");
  auto chains = selection.getAs<StringAttr>("chains");
  auto testOnly = selection.getAs<StringAttr>("test_only");
  if ((selection.get("pairs") && !pairs) || (selection.get("pairs_sha256") && !pairsHash) ||
      (selection.get("chains") && (!chains || chains.getValue() != "true")) ||
      selection.get("composed") || selection.get("composed_sha256") ||
      bool(pairs) != bool(pairsHash) || bool(chains) != bool(pairs) ||
      (!pairs && (!testOnly || testOnly.getValue() != "true")))
    return {{}, "incomplete source-derived model overlap selection"};
  if (pairs) {
    auto loaded = loadModelOverlapFacts(pairs.getValue(), pairsHash.getValue(), *facts);
    if (!loaded) return {{}, llvm::toString(loaded.takeError())};
    for (const auto &[key, value] : loaded->metadata()) {
      auto retained = selection.getAs<StringAttr>(key);
      if (!retained || retained.getValue() != value) return {{}, "model overlap provenance mismatch: " + key};
    }
    overlap = std::make_shared<ModelOverlapFacts>(std::move(*loaded));
  }
  return {modelTimingProvider(std::make_shared<ModelTimingFacts>(std::move(*facts)), overlap,
                              chains && chains.getValue() == "true"), {}};
}

TimingRuleResult<TimingProvider> mlir::atlas::timing::modelTimingProviderFactory(ModuleOp module) {
  auto result = modelTimingProviderFactoryImpl(module);
  if (!result.error.empty()) result.error = modelTimingDiagnostic(result.error);
  return result;
}

namespace {
struct SelectAtlasModelEvidencePass : PassWrapper<SelectAtlasModelEvidencePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SelectAtlasModelEvidencePass)
  SelectAtlasModelEvidencePass() = default;
  SelectAtlasModelEvidencePass(const SelectAtlasModelEvidencePass &other) : PassWrapper(other) {}
  Option<std::string> factsDirectory{*this, "facts", llvm::cl::desc("Qualified model facts bundle directory")};
  // Narrow synthetic-fixture hook: production model selection always consumes a bundle.
  Option<std::string> testOpTiming{*this, "test-op-timing", llvm::cl::desc("Test-only isolated timing fixture")};
  Option<std::string> testOpTimingHash{*this, "test-op-timing-sha256", llvm::cl::desc("Test-only fixture SHA-256")};
  StringRef getArgument() const final { return "select-atlas-model-evidence"; }
  StringRef getDescription() const final { return "Pin model observations; timing passes select provider=model"; }
  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto fail = [&](const std::string &why) { module.emitError(modelTimingDiagnostic(why)); signalPassFailure(); };
    if (auto state = module->getAttrOfType<StringAttr>(kAtlasTimingState); state && state.getValue() == "timed")
      return fail("select model evidence before timing; already timed input must be retimed");
    if (module->hasAttr(kAtlasTimingProvider)) return fail("model selection cannot replace a retained timing provider");
    bool testOnly = !testOpTiming.empty() || !testOpTimingHash.empty();
    if (factsDirectory.empty() == !testOnly || (testOnly &&
        (testOpTiming.empty() || testOpTimingHash.empty())))
      return fail("facts requires a qualified bundle directory");
    std::string opTiming = testOpTiming, opTimingHash = testOpTimingHash;
    std::string overlapPath, overlapHash;
    if (!testOnly) {
      auto file = [&](llvm::StringRef name) {
        llvm::SmallString<256> path(factsDirectory.getValue());
        llvm::sys::path::append(path, name);
        return path.str().str();
      };
      auto buffer = llvm::MemoryBuffer::getFile(file("facts.lock.json"));
      if (!buffer) return fail("cannot read selected model facts lock");
      auto decoded = llvm::json::parse((*buffer)->getBuffer());
      if (!decoded) return fail(llvm::toString(decoded.takeError()));
      const auto *lock = decoded->getAsObject();
      if (!lock || lock->getString("schema") != "atlas.model_facts.lock.v1")
        return fail("unsupported model facts lock");
      auto opHash = lock->getString("op_timing_sha256"), pairHash = lock->getString("overlap_sha256");
      if (!opHash || !pairHash || !hashSyntax(*opHash) || !hashSyntax(*pairHash))
        return fail("model facts lock requires operation and qualified overlap hashes");
      opTiming = file("model-timing.json"); opTimingHash = opHash->str();
      overlapPath = file("model-overlap.json"); overlapHash = pairHash->str();
    }
    auto facts = loadModelTimingFacts(opTiming, opTimingHash, true);
    if (!facts) return fail(llvm::toString(facts.takeError()));
    std::shared_ptr<const ModelOverlapFacts> overlap;
    if (!testOnly) {
      auto loaded = loadModelOverlapFacts(overlapPath, overlapHash, *facts);
      if (!loaded) return fail(llvm::toString(loaded.takeError()));
      overlap = std::make_shared<ModelOverlapFacts>(std::move(*loaded));
    }
    Builder builder(module.getContext());
    NamedAttrList selection;
    selection.set("provider", builder.getStringAttr(kModelTimingProviderId));
    selection.set("op_timing", builder.getStringAttr(opTiming));
    selection.set("op_timing_sha256", builder.getStringAttr(opTimingHash));
    selection.set("dma", builder.getStringAttr("wait"));
    if (testOnly) selection.set("test_only", builder.getStringAttr("true"));
    else selection.set("facts", builder.getStringAttr(factsDirectory));
    for (const auto &[key, value] : facts->modelProvenance()) selection.set(key, builder.getStringAttr(value));
    if (overlap) {
      selection.set("pairs", builder.getStringAttr(overlapPath));
      selection.set("chains", builder.getStringAttr("true"));
      for (const auto &[key, value] : overlap->metadata()) selection.set(key, builder.getStringAttr(value));
    }
    module->setAttr(kAtlasModelEvidence, selection.getDictionary(module.getContext()));
  }
};
} // namespace

void mlir::atlas::registerSelectAtlasModelEvidencePass() { PassRegistration<SelectAtlasModelEvidencePass>(); }
