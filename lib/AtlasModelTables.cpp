#include "Atlas/AtlasModelTables.h"
#include "Atlas/AtlasGeneratedArtifact.h"
#include "Atlas/AtlasStream.h"
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
#include <set>

using namespace mlir;
using namespace mlir::atlas;
using namespace mlir::atlas::timing;

namespace {
using Object = llvm::json::Object;
using Array = llvm::json::Array;

struct IntegerBounds {
  int minimum, maximum;
  const char *name;
};
constexpr int kMaximumTimingValue = 4096;
constexpr IntegerBounds kTimingValue{0, kMaximumTimingValue, "timing value"};
constexpr IntegerBounds kPositiveTimingValue{1, kMaximumTimingValue, "positive timing value"};
constexpr IntegerBounds kPositiveInteger{1, INT_MAX, "positive integer"};
constexpr IntegerBounds kTransferBytes{1, kAtlasVmemBytes, "DMA transfer bytes"};
constexpr IntegerBounds kVpuSlots{1, 3, "VPU issue slots"};
constexpr IntegerBounds kMxuCapacity{1, 64, "MXU compute capacity"};

struct Reader {
  std::string error;
  void fail(const std::string &message) { if (error.empty()) error = message; }
  const Object *object(const Object *parent, llvm::StringRef key) {
    auto *value = parent ? parent->getObject(key) : nullptr;
    if (!value) fail("missing object " + key.str());
    return value;
  }
  int integer(const Object *parent, llvm::StringRef key, IntegerBounds bounds) {
    auto value = parent ? parent->getInteger(key) : std::nullopt;
    if (!value || *value < bounds.minimum || *value > bounds.maximum) {
      fail("invalid integer " + key.str() + ": expected " + bounds.name + " in [" +
           std::to_string(bounds.minimum) + ", " + std::to_string(bounds.maximum) + "]");
      return 0;
    }
    return int(*value);
  }
  std::vector<int> ports(const Object *parent, llvm::StringRef key) {
    const auto *array = parent ? parent->getArray(key) : nullptr;
    std::vector<int> result;
    if (!array || array->empty()) fail("missing port list " + key.str());
    else for (const auto &entry : *array) {
      auto value = entry.getAsInteger();
      if (!value || *value < 0 || *value > 3 ||
          std::find(result.begin(), result.end(), *value) != result.end())
        fail("invalid port list " + key.str());
      else result.push_back(int(*value));
    }
    return result;
  }
  std::string mnemonic(llvm::StringRef name) {
    const OpInfo *op = findOp(name.str());
    if (!op || op->engine == Engine::Scalar || op->engine == Engine::Dma) {
      fail("unsupported operation " + name.str());
      return {};
    }
    return name.str();
  }
  TimingEvent event(const Object *value) {
    TimingEvent result{integer(value, "first_age", kTimingValue),
                       integer(value, "last_age", kTimingValue),
                       integer(value, "count", kPositiveTimingValue),
                       integer(value, "step", kPositiveTimingValue)};
    if (result.lastAge != result.firstAge + (result.count - 1) * result.step)
      fail("event ages do not form an affine stream");
    return result;
  }
};

void readOperations(Reader &r, const Object &root, TimingTables &tables) {
  const Array *blocks = root.getArray("op_timing");
  const Object *releases = r.object(&root, "reservation_releases");
  if (!blocks) { r.fail("missing operation timing"); return; }
  for (const auto &value : *blocks) {
    const Object *block = value.getAsObject();
    auto name = block ? block->getString("name") : std::nullopt;
    if (!name || !block) {
      r.fail("missing operation name"); continue;
    }
    std::string mnemonic = r.mnemonic(*name);
    OperationTiming timing;
    timing.firstFreeAge = r.integer(block, "first_free_age", kPositiveTimingValue);
    const Object *events = r.object(block, "events");
    if (events) for (const auto &[key, entry] : *events) {
      const Object *group = entry.getAsObject();
      // Empty optional streams are emitted for inactive operand roles.
      if (group && group->getInteger("count") == 0) continue;
      timing.events.emplace(key.str(), r.event(group));
    }
    const Object *reservations = r.object(releases, *name);
    if (reservations) for (const auto &[key, entry] : *reservations) {
      const Object *record = entry.getAsObject();
      int first = r.integer(record, "first_age", kTimingValue);
      int release = r.integer(record, "release_age", kTimingValue);
      if (release < first || release >= timing.firstFreeAge)
        r.fail("reservation does not fit operation lifetime: " + name->str());
      timing.releases.emplace(key.str(), TimingWindow{first, release});
    }
    if (!tables.operations.emplace(mnemonic, std::move(timing)).second)
      r.fail("duplicate operation timing " + mnemonic);
  }
}

void readFrontend(Reader &r, const Object &root, TimingTables &tables,
                  ModelTimingTables &facts) {
  const Object *frontend = r.object(&root, "frontend_timing");
  const Object *delay = r.object(frontend, "delay");
  const Object *scalar = r.object(frontend, "scalar");
  const Object *control = r.object(frontend, "control");
  tables.ordinaryIssueGap = r.integer(frontend, "ordinary_issue_gap", kPositiveTimingValue);
  tables.delayBaseGap = r.integer(delay, "base_gap", kPositiveTimingValue);
  tables.delayImmediateMask = r.integer(delay, "immediate_mask", kTimingValue);
  tables.scalarReadAge = r.integer(scalar, "read_age", kTimingValue);
  tables.scalarWriteAge = r.integer(scalar, "write_age", kTimingValue);
  tables.controlReadAge = r.integer(control, "operand_read_age", kTimingValue);
  int slots = r.integer(control, "delay_slots", kPositiveTimingValue);
  if (tables.ordinaryIssueGap != 1 || tables.delayBaseGap != 1 ||
      tables.delayImmediateMask != 4095 || slots != 1 ||
      r.integer(delay, "decrement_per_tick", kPositiveTimingValue) != 1)
    r.fail("unsupported frontend issue or delay encoding");
  facts.frontend = {tables.ordinaryIssueGap, slots};
  const Object *instructions = r.object(frontend, "instructions");
  if (instructions && instructions->empty()) r.fail("missing frontend instruction coverage");
  if (instructions) for (const auto &[name, value] : *instructions) {
    const OpInfo *op = findOp(name.str());
    if (!op || op->engine != Engine::Scalar) {
      r.fail("unsupported frontend operation " + name.str());
      continue;
    }
    auto roles = [&](llvm::StringRef key) {
      std::set<std::string> result;
      const auto *entry = value.getAsObject();
      const Array *accesses = entry ? entry->getArray(key) : nullptr;
      if (!accesses) r.fail("missing frontend " + key.str() + " for " + name.str());
      else for (const auto &access : *accesses) {
        const auto *role = access.getAsObject();
        auto resource = role ? role->getString("resource") : std::nullopt;
        auto operand = role ? role->getString("operand") : std::nullopt;
        if (!resource || !operand || !result.insert(resource->str() + "." + operand->str()).second)
          r.fail("invalid frontend role for " + name.str());
      }
      return result;
    };
    std::set<std::string> reads, writes;
    bool immediateCsr = op->opClass == OpClass::Csr && op->name.back() == 'i';
    if (hasOperand(*op, "x1") && !immediateCsr) reads.insert("xreg.rs1");
    if (hasOperand(*op, "x2")) reads.insert("xreg.rs2");
    if (hasOperand(*op, "xd")) writes.insert("xreg.rd");
    if (hasOperand(*op, "ed")) writes.insert("ereg.rd");
    if (roles("reads") != reads || roles("writes") != writes)
      r.fail("frontend register roles differ for " + name.str());
    facts.frontendInstructions.push_back(name.str());
  }
}

void readGeometry(Reader &r, const Object &root, TimingTables &tables,
                  ModelTimingTables &facts) {
  const Object *arch = r.object(&root, "architecture");
  for (auto [key, expected] : std::initializer_list<std::pair<const char *, int>>{
      {"mreg_count", kAtlasTensorRegisters}, {"mreg_rows", kAtlasMregRows},
      {"mreg_row_bytes", kAtlasMregRowBytes}, {"mreg_physical_bank_count", kAtlasMregBanks},
      {"vmem_bytes", kAtlasVmemBytes}, {"vmem_bank_bytes", kVmemBankBytes},
      {"vmem_bank_count", kVmemBanks}, {"dma_channels", kAtlasDMAChannels},
      {"scalar_word_bytes", 4}, {"bf16_registers_per_tile", kAtlasBF16Registers},
      {"bf16_element_bytes", 2}, {"fp8_registers_per_tile", 1}, {"fp8_element_bytes", 1}})
    if (r.integer(arch, key, kPositiveInteger) != expected)
      r.fail(std::string("unsupported architecture ") + key);
  if (!arch || arch->getInteger("dram_base") != 0x80000000LL ||
      arch->getInteger("dram_bytes") != 0x1000000000LL)
    r.fail("unsupported DRAM geometry");
  if (r.integer(arch, "mreg_bytes", kPositiveInteger) != kAtlasRegisterBytes)
    r.fail("unsupported architecture mreg_bytes");
  tables.bf16RegistersPerTile = r.integer(arch, "bf16_registers_per_tile", kPositiveInteger);
  facts.instructionMemoryWords = r.integer(arch, "instruction_memory_words", kPositiveInteger);
  if (facts.instructionMemoryWords != 32768)
    r.fail("unsupported architecture instruction_memory_words");
}

void readDMA(Reader &r, const Object &root, ModelDMATiming &dma) {
  const Object *timing = r.object(&root, "dma_timing");
  const Object *startup = r.object(timing, "startup_cycles");
  dma.beatBytes = r.integer(timing, "beat_bytes", kPositiveInteger);
  dma.maxBytes = r.integer(timing, "max_transfer_bytes", kTransferBytes);
  if (dma.beatBytes != kLineBytes) r.fail("unsupported DMA beat geometry");
  dma.loadStartup = r.integer(startup, "load", kPositiveTimingValue);
  dma.storeStartup = r.integer(startup, "store", kPositiveTimingValue);
  dma.retireToWait = r.integer(timing, "retire_to_wait_issue_cycles", kPositiveTimingValue);
  double maximum = r.integer(timing, "min_latency", kPositiveInteger);
  const Array *curves = timing ? timing->getArray("curves") : nullptr;
  if (!curves || curves->empty()) r.fail("missing DMA curves");
  else for (const auto &value : *curves) {
    const Object *curve = value.getAsObject();
    for (const char *key : {"unloaded_latency_cycles", "cycles_per_beat"}) {
      auto number = curve ? curve->getNumber(key) : std::nullopt;
      if (!number || !std::isfinite(*number) || *number <= 0 || *number > INT_MAX - kMaximumTimingValue)
        r.fail(std::string("invalid DMA curve ") + key);
      else maximum = std::max(maximum, *number);
    }
  }
  // Compiler policy: bound every beat by the slowest curve plus issue spacing.
  int interval = r.integer(timing, "issue_interval_cycles", kPositiveTimingValue);
  if (maximum > INT_MAX - interval) r.fail("DMA per-beat bound exceeds cycle domain");
  else dma.perBeatBound = int(std::ceil(maximum)) + interval;
  if (dma.maxBytes % std::max(dma.beatBytes, 1))
    r.fail("DMA maximum transfer size is not a whole number of beats");
  for (bool store : {false, true}) {
    auto estimate = dma.estimate(dma.maxBytes, store);
    if (!estimate.error.empty()) r.fail(estimate.error);
  }
}

void readVpu(Reader &r, const Object *resources, TimingTables &tables) {
  const Object *capacity = r.object(resources, "capacity");
  const Object *vpu = r.object(resources, "vpu");
  tables.vpuIssueSlots = r.integer(capacity, "vpu_slots", kVpuSlots);
  auto names = [&](const Array *array) {
    std::vector<std::string> result;
    if (!array) r.fail("missing VPU operation set");
    else for (const auto &entry : *array) {
      auto name = entry.getAsString();
      if (!name) r.fail("invalid VPU operation set");
      else {
        const OpInfo *op = findOp(name->str());
        if (!op || op->engine != Engine::Vpu) r.fail("invalid VPU operation set");
        else result.push_back(name->str());
      }
    }
    return result;
  };
  tables.vpuDouble = names(vpu ? vpu->getArray("double") : nullptr);
  const Array *groups = vpu ? vpu->getArray("logic_groups") : nullptr;
  if (!groups) r.fail("missing VPU logic groups");
  else for (const auto &group : *groups) tables.vpuLogicGroups.push_back(names(group.getAsArray()));
  const Object *scalar = r.object(resources, "scalar_lsu");
  tables.scalarLoadGap = r.integer(scalar, "load_response_gap", kPositiveTimingValue);
}

void readWeights(Reader &r, const Object &root, const Object *resources,
                 TimingTables &tables) {
  const Object *capacity = r.object(resources, "capacity");
  const Object *mxu = r.object(resources, "mxu");
  const Object *ports = r.object(mxu, "ports");
  const Object *weights = r.object(r.object(&root, "mxu_weight_timing"), "engines");
  int sequencer = r.integer(capacity, "sequencer_cycles", kPositiveTimingValue);
  for (int i = 0; i < 2; ++i) {
    auto &rules = tables.mxu[i];
    std::string engine = "mxu" + std::to_string(i);
    rules.computeCapacity = r.integer(capacity, engine + "_compute", kMxuCapacity);
    rules.sourcePorts = rules.resultPorts = {0, sequencer - 1};
    rules.weightOverwriteAfterCompute =
        i == 0 ? r.integer(mxu, "mxu0_weight_drain_last", kTimingValue) + 1 : sequencer;
    rules.weightUseAfterPush = sequencer;
    rules.accumulatorUseAfterPush = r.integer(mxu, "acc_push_last", kTimingValue) + 1;
    rules.accumulatorForwardGap = r.integer(mxu, "accumulator_forward_gap", kPositiveTimingValue);
    rules.computePorts = r.ports(ports, "compute");
    rules.pushBf16Ports = r.ports(ports, "push_bf16");
    rules.popFp8Ports = r.ports(ports, "pop_fp8");
    rules.popBf16Ports = r.ports(ports, "pop_bf16");
    rules.singlePushPriority = r.ports(mxu, "single_push_priority");
    const Object *entry = r.object(weights, engine);
    const Array *rows = entry ? entry->getArray("use_rows") : nullptr;
    if (!rows || rows->size() != kAtlasMregRows) { r.fail("incomplete MXU weight rows"); continue; }
    std::vector<std::pair<int, int>> ages;
    for (const auto &value : *rows) {
      const Object *row = value.getAsObject();
      ages.emplace_back(r.integer(row, "first_age", kTimingValue),
                        r.integer(row, "last_age", kTimingValue));
    }
    int step = ages[1].first - ages[0].first;
    if (step < 0) r.fail("negative MXU weight row step");
    for (int row = 0; row < kAtlasMregRows; ++row)
      if (ages[row].first != ages[0].first + row * step ||
          ages[row].second != ages[0].second + row * step || ages[row].second < ages[row].first)
        r.fail("non-affine MXU weight rows");
    rules.weightFirst = {ages.front().first, ages.back().first, kAtlasMregRows, step};
    rules.weightLast = {ages.front().second, ages.back().second, kAtlasMregRows, step};
  }
}
void validateOperations(Reader &reader, const TimingTables &tables) {
  for (const auto &[name, timing] : tables.operations) {
    const OpInfo *op = findOp(name);
    if (!op) { reader.fail("unknown compiler operation " + name); continue; }
    Instr probe{op, 0, 2, 4};
    if (op->mxu >= 0) probe = {op, 0, 2, 0};
    Footprint footprint = footprintOf(probe, unknownRegs(), tables);
    if (!footprint.error.empty()) reader.fail(footprint.error);
    for (const auto &[key, event] : timing.events)
      if (event.lastAge >= timing.firstFreeAge)
        reader.fail(name + ": event exceeds isolated lifetime");
    if (!timing.releases.count("unit")) reader.fail(name + ": missing unit release");
    if (!footprint.mregReads.empty() && !timing.releases.count("mreg_read"))
      reader.fail(name + ": missing MREG read release");
    if (!footprint.mregWrites.empty() && !timing.releases.count("mreg_write"))
      reader.fail(name + ": missing MREG write release");
  }
}
} // namespace

std::string mlir::atlas::timing::modelTimingDiagnostic(llvm::StringRef message) {
  return message.str() + "\nSee docs/model-timing.md#updating-for-npu-model-changes.";
}

std::string mlir::atlas::timing::modelSha256(llvm::StringRef bytes) {
  auto digest = llvm::SHA256::hash(llvm::ArrayRef<uint8_t>(
      reinterpret_cast<const uint8_t *>(bytes.data()), bytes.size()));
  return llvm::toHex(llvm::ArrayRef<uint8_t>(digest), true);
}

TimingRuleResult<int> ModelDMATiming::estimate(int64_t bytes, bool store) const {
  // Compiler policy for ordering; dma.wait establishes actual completion.
  if (beatBytes <= 0 || bytes < beatBytes || bytes > maxBytes || bytes % beatBytes)
    return {0, "model DMA transfer is outside aligned geometry"};
  int64_t cycles = int64_t(store ? storeStartup : loadStartup) + bytes / beatBytes * perBeatBound;
  if (cycles > INT_MAX) return {0, "model DMA estimate exceeds cycle domain"};
  return {int(cycles), {}};
}

TimingRuleResult<DMAProposal> ModelDMATiming::propose(
    int latency, int issue, std::optional<int> previousEnd) const {
  int64_t launch = previousEnd ? std::max(int64_t(issue), int64_t(*previousEnd) + serializationGap) : issue;
  int64_t end = launch + latency, wait = end + retireToWait;
  if (latency < 0 || issue < 0 || (previousEnd && *previousEnd < 0) || wait > INT_MAX)
    return {{}, "model DMA proposal exceeds cycle domain"};
  return {{int(end), int(wait)}, {}};
}

llvm::Expected<ModelTimingTables> mlir::atlas::timing::parseModelTimingTables(llvm::StringRef bytes) {
  auto decoded = llvm::json::parse(bytes);
  if (!decoded) return decoded.takeError();
  const Object *root = decoded->getAsObject();
  if (!root)
    return llvm::createStringError(llvm::inconvertibleErrorCode(), "unsupported model timing facts");
  Reader r;
  ModelTimingTables result;
  auto tables = std::make_shared<TimingTables>();
  readOperations(r, *root, *tables);
  readGeometry(r, *root, *tables, result);
  readFrontend(r, *root, *tables, result);
  readDMA(r, *root, result.dma);
  const Object *resources = r.object(root, "resources");
  readVpu(r, resources, *tables);
  if (r.error.empty()) readWeights(r, *root, resources, *tables);
  if (r.error.empty()) validateOperations(r, *tables);
  if (!r.error.empty()) return llvm::createStringError(llvm::inconvertibleErrorCode(), r.error);
  tables->dmaEstimate = [dma = result.dma](long long bytes, bool store) {
    return dma.estimate(bytes, store);
  };
  result.timing = std::move(tables);
  return result;
}

llvm::Expected<ModelTimingTables> mlir::atlas::timing::loadModelTimingTables(
    llvm::StringRef path, llvm::StringRef expectedHash) {
  auto buffer = llvm::MemoryBuffer::getFile(path);
  if (!buffer)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "cannot read model timing facts: " + path.str());
  if (modelSha256((*buffer)->getBuffer()) != expectedHash)
    return llvm::createStringError(llvm::inconvertibleErrorCode(), "selected model timing facts hash mismatch");
  return parseModelTimingTables((*buffer)->getBuffer());
}

namespace {
std::string memoryOperands(const Instr &in, const RegValues &regs,
                           const ModelTimingTables &facts) {
  const auto &op = *in.op;
  auto xreg = [](int reg) { return reg >= 0 && reg < 32; };
  if (!xreg(in.rs1) || !xreg(in.rs2)) return "scalar operand out of range";
  if (op.engine == Engine::Dma) {
    if (!xreg(in.rd)) return "scalar operand out of range";
    if (op.opClass == OpClass::DmaWait) return {};
    if (op.opClass == OpClass::DmaConfig)
      return regs[in.rs1] && *regs[in.rs1] <= 31 ? "" : "DMA configuration requires known upper base";
    if (!regs[in.rd] || !regs[in.rs1] || !regs[in.rs2]) return "DMA requires known captured operands";
    bool load = op.opClass == OpClass::DmaLoad;
    uint64_t words = *regs[load ? in.rd : in.rs1], offset = *regs[load ? in.rs1 : in.rd];
    uint64_t bytes = *regs[in.rs2];
    if (!facts.dma.estimate(bytes, !load).error.empty() || words % (facts.dma.beatBytes / 4) ||
        words + bytes / 4 > kAtlasVmemBytes / 4 || offset % facts.dma.beatBytes ||
        offset + bytes > (uint64_t(1) << 32) ||
        words / (kVmemBankBytes / 4) != (words + bytes / 4 - 1) / (kVmemBankBytes / 4))
      return "DMA transfer is unaligned or outside supported geometry";
    return {};
  }
  bool vector = op.opClass == OpClass::VLoad || op.opClass == OpClass::VStore;
  if (vector) {
    if (in.rs2) return "unsupported vector memory encoding";
    return {};
  }
  if (!xreg(in.rd)) return "scalar operand out of range";
  bool store = op.opClass == OpClass::ScalarStore;
  if ((store && in.rd) || (!store && in.rs2)) return "unsupported scalar memory encoding";
  if (regs[in.rs1]) {
    uint32_t address = (*regs[in.rs1] + uint32_t(signExtend(in.imm, 12))) & 0x1fffff;
    if (address % 4 || uint64_t(address) + 4 > kAtlasVmemBytes)
      return "scalar memory word is unaligned or outside VMEM";
  }
  return {};
}

std::string vectorOperands(const Instr &in, const Footprint &f) {
  const auto &op = *in.op;
  bool push = op.opClass == OpClass::WeightPush || op.opClass == OpClass::AccPushFp8 ||
              op.opClass == OpClass::AccPushBf16;
  if (op.mxu >= 0) {
    if (in.imm || (push && in.rs2) || (op.opClass == OpClass::PopBf16 && in.rs1))
      return "unsupported MXU encoding";
    return {};
  }
  if (op.opClass == OpClass::Transpose)
    return in.imm || in.rs2 ? "unsupported XLU encoding" : "";
  bool immediate = op.opClass == OpClass::VpuLoadImmPair || op.opClass == OpClass::VpuLoadImmSingle;
  if ((!immediate && in.imm) || (immediate && (in.rs1 || in.rs2)) ||
      ((op.opClass == OpClass::VpuRowReduce || op.opClass == OpClass::VpuColReduce ||
        (op.opClass == OpClass::VpuElementwise && !op.twoInput)) && in.rs2))
    return "unsupported VPU encoding";
  for (const auto *registers : {&f.mregReads, &f.mregWrites}) {
    std::map<int, int> banks;
    for (int reg : *registers) {
      auto [where, inserted] = banks.emplace(reg % kAtlasMregBanks, reg);
      if (!inserted && !(registers == &f.mregReads && where->second == reg))
        return "VPU operands must occupy distinct physical banks";
    }
  }
  return {};
}

Footprint modelFootprint(const Instr &in, const RegValues &regs,
                         const ModelTimingTables &facts) {
  Footprint rejected;
  auto fail = [&](const std::string &message) { rejected.error = message; return rejected; };
  if (!in.op) return fail("invalid instruction");
  const auto &op = *in.op;
  if (op.engine == Engine::Scalar) {
    if (std::find(facts.frontendInstructions.begin(), facts.frontendInstructions.end(), op.name) ==
        facts.frontendInstructions.end()) return fail("unsupported frontend operation " + op.name);
    if (in.rd < 0 || in.rd > 31 || in.rs1 < 0 || in.rs1 > 31 || in.rs2 < 0 || in.rs2 > 31)
      return fail("scalar register out of range");
    if (op.name == "auipc" || op.name == "ebreak" || op.opClass == OpClass::Fence)
      return fail("operation is outside model scope");
    if (op.opClass == OpClass::Csr && (op.name != "csrrw" || in.rd || in.imm != 0xc10))
      return fail("unsupported CSR operation");
    if (op.opClass == OpClass::Jump && (op.name != "jal" || in.rd || in.rs1))
      return fail("JAL must not link");
  } else if (op.engine == Engine::Dma || op.engine == Engine::Lsu) {
    if (op.engine == Engine::Lsu && !facts.timing->operations.count(op.name))
      return fail("unsupported scalar memory operation");
    auto error = memoryOperands(in, regs, facts);
    if (!error.empty()) return fail(error);
  }
  Footprint f = footprintOf(in, regs, *facts.timing);
  if (!f.error.empty()) return f;
  if (op.engine == Engine::Vpu || op.engine == Engine::Xlu || op.mxu >= 0) {
    auto error = vectorOperands(in, f);
    if (!error.empty()) return fail(error);
  }
  return f;
}

std::string modelScope(const AtlasStream &stream, int instructionMemoryWords) {
  if (stream.ops.size() > size_t(instructionMemoryWords))
    return "model program exceeds instruction memory";
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
                 std::to_string(n % kAtlasDMACommandSlots) + " of unwaited launch " +
                 std::to_string(n - kAtlasDMACommandSlots);
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

TimingProvider mlir::atlas::timing::modelTimingProvider(std::shared_ptr<const ModelTimingTables> facts) {
  TimingProvider p;
  p.id = kModelTimingProviderId.str();
  p.model = facts->timing;
  p.frontendSpacing = facts->frontend;
  p.fillPendingDelaySlots = true;
  p.waitReusedDMASlots = true;
  p.validateScope = [facts](const AtlasStream &stream) {
    auto error = modelScope(stream, facts->instructionMemoryWords);
    return error.empty() ? error : modelTimingDiagnostic(error);
  };
  p.footprint = [facts](const Instr &in, const RegValues &regs) {
    auto footprint = modelFootprint(in, regs, *facts);
    if (!footprint.error.empty())
      footprint.error = modelTimingDiagnostic(footprint.error);
    return footprint;
  };
  p.dependence = [tables = facts->timing](const Instr &a, const Footprint &fa, const Instr &b, const Footprint &fb) {
    return TimingRuleResult<Dependence>{dependence(a, fa, b, fb, *tables), {}};
  };
  p.dmaConflict = [tables = facts->timing](const Footprint &dma, const Footprint &other) {
    DMACompletionConflict result;
    result.conflict = conflictsAtCompletion(dma, other, result.kind, tables.get());
    return TimingRuleResult<DMACompletionConflict>{result, {}};
  };
  p.issueGap = [tables = facts->timing](const Instr &in) {
    if (!in.op) return TimingRuleResult<int>{0, modelTimingDiagnostic("invalid instruction")};
    return TimingRuleResult<int>{naturalGap(in, *tables), {}};
  };
  p.createReservations = [tables = facts->timing] {
    return TimingRuleResult<std::unique_ptr<TimingReservations>>{
        createTimingReservations(tables), {}};
  };
  p.dmaEstimate = [facts](int64_t bytes, bool store) {
    auto estimate = facts->dma.estimate(bytes, store);
    if (!estimate.error.empty())
      estimate.error = modelTimingDiagnostic(estimate.error);
    return estimate;
  };
  p.dmaProposal = [facts](int latency, int issue, std::optional<int> previous) {
    auto proposal = facts->dma.propose(latency, issue, previous);
    if (!proposal.error.empty())
      proposal.error = modelTimingDiagnostic(proposal.error);
    return proposal;
  };
  return p;
}

TimingRuleResult<TimingProvider> mlir::atlas::timing::modelTimingProviderFactory(ModuleOp module) {
  auto fail = [](const std::string &error) {
    return TimingRuleResult<TimingProvider>{{}, modelTimingDiagnostic(error)};
  };
  auto selection = module ? module->getAttrOfType<DictionaryAttr>(kAtlasModelEvidence) : DictionaryAttr{};
  if (!selection) return fail("model requires selected atlas.model_evidence");
  auto path = selection.getAs<StringAttr>("op_timing"), hash = selection.getAs<StringAttr>("op_timing_sha256");
  auto provider = selection.getAs<StringAttr>("provider");
  if (!path || !hash || !provider || provider.getValue() != kModelTimingProviderId)
    return fail("incomplete model evidence selection");
  auto facts = loadModelTimingTables(path.getValue(), hash.getValue());
  if (!facts) return fail(llvm::toString(facts.takeError()));
  auto result = modelTimingProvider(std::make_shared<ModelTimingTables>(std::move(*facts)));
  if (auto error = fillMachineTiming(result); !error.empty())
    return {{}, error};
  return {std::move(result), {}};
}

namespace {
struct SelectAtlasModelEvidencePass : PassWrapper<SelectAtlasModelEvidencePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SelectAtlasModelEvidencePass)
  SelectAtlasModelEvidencePass() = default;
  SelectAtlasModelEvidencePass(const SelectAtlasModelEvidencePass &other) : PassWrapper(other) {}
  Option<std::string> factsDirectory{*this, "facts", llvm::cl::desc("Model facts directory")};
  StringRef getArgument() const final { return "select-atlas-model-evidence"; }
  StringRef getDescription() const final { return "Select model timing facts"; }
  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto fail = [&](const std::string &why) { module.emitError(modelTimingDiagnostic(why)); signalPassFailure(); };
    if (auto state = module->getAttrOfType<StringAttr>(kAtlasTimingState); state && state.getValue() == "timed")
      return fail("select model facts before timing");
    if (module->hasAttr(kAtlasTimingProvider)) return fail("model selection cannot replace a retained timing provider");
    if (factsDirectory.empty()) return fail("facts requires a model facts directory");
    auto file = [&](llvm::StringRef name) {
      llvm::SmallString<256> path(factsDirectory.getValue());
      llvm::sys::path::append(path, name);
      return path.str().str();
    };
    auto bytes = llvm::MemoryBuffer::getFile(file("model-timing.json"));
    if (!bytes) return fail("cannot read selected model timing facts");
    auto hash = modelSha256((*bytes)->getBuffer());
    auto loaded = parseModelTimingTables((*bytes)->getBuffer());
    if (!loaded) return fail(llvm::toString(loaded.takeError()));
    Builder builder(module.getContext());
    NamedAttrList selection;
    selection.set("provider", builder.getStringAttr(kModelTimingProviderId));
    selection.set("op_timing", builder.getStringAttr(file("model-timing.json")));
    selection.set("op_timing_sha256", builder.getStringAttr(hash));
    auto timing = llvm::json::parse((*bytes)->getBuffer());
    if (!timing) return fail(llvm::toString(timing.takeError()));
    auto curve = timing->getAsObject()->getObject("dma_timing")->getString("curve_sha256");
    if (curve) selection.set("dma_curve_sha256", builder.getStringAttr(*curve));
    module->setAttr(kAtlasModelEvidence, selection.getDictionary(module.getContext()));
  }
};
} // namespace

void mlir::atlas::registerSelectAtlasModelEvidencePass() { PassRegistration<SelectAtlasModelEvidencePass>(); }
