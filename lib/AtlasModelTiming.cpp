#include "Atlas/AtlasModelTiming.h"
#include "Atlas/AtlasGeneratedArtifact.h"
#include "Atlas/AtlasStream.h"
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

bool hashSyntax(llvm::StringRef value) {
  return value.size() == 64 && std::all_of(value.begin(), value.end(), [](char c) {
    return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
  });
}

// Instruction operands map observed event groups to architectural resources.
const std::map<std::string, const char *> &vpuBlocks() {
  static const std::map<std::string, const char *> table = {
      {"vadd.bf16", "vpu.add"}, {"vsub.bf16", "vpu.sub"}, {"vmul.bf16", "vpu.mul"},
      {"vmaximum.bf16", "vpu.pairmax"}, {"vminimum.bf16", "vpu.pairmin"},
      {"vmov", "vpu.mov"}, {"vrecip.bf16", "vpu.rcp"}, {"vexp.bf16", "vpu.exp"},
      {"vexp2.bf16", "vpu.exp2"}, {"vrelu.bf16", "vpu.relu"}, {"vsin.bf16", "vpu.sin"},
      {"vcos.bf16", "vpu.cos"}, {"vtanh.bf16", "vpu.tanh"}, {"vlog2.bf16", "vpu.log"},
      {"vsqrt.bf16", "vpu.sqrt"}, {"vsquare.bf16", "vpu.square"}, {"vcube.bf16", "vpu.cube"},
      {"vredsum.row.bf16", "vpu.rsum"}, {"vredmax.row.bf16", "vpu.rmax"},
      {"vredmin.row.bf16", "vpu.rmin"}, {"vredsum.bf16", "vpu.csum"},
      {"vredmax.bf16", "vpu.cmax"}, {"vredmin.bf16", "vpu.cmin"},
      {"vpack.bf16.fp8", "vpu.fp8pack"}, {"vunpack.fp8.bf16", "vpu.fp8unpack"},
      {"vli.all", "vpu.vliAll"}, {"vli.row", "vpu.vliRow"}, {"vli.col", "vpu.vliCol"}, {"vli.one", "vpu.vliOne"}};
  return table;
}

std::string canonicalJson(const llvm::json::Value &value) {
  std::string bytes;
  llvm::raw_string_ostream out(bytes);
  if (const Object *object = value.getAsObject()) {
    std::vector<std::string> keys;
    for (const auto &[key, entry] : *object) keys.push_back(key.str());
    std::sort(keys.begin(), keys.end());
    out << '{';
    bool first = true;
    for (const std::string &key : keys) {
      if (!first) out << ',';
      first = false;
      out << llvm::json::Value(key) << ':' << canonicalJson(*object->get(key));
    }
    out << '}';
  } else if (const Array *array = value.getAsArray()) {
    out << '[';
    bool first = true;
    for (const auto &entry : *array) {
      if (!first) out << ',';
      first = false;
      out << canonicalJson(entry);
    }
    out << ']';
  } else {
    out << value;
  }
  return bytes;
}

bool supportedConfiguration(const Object &configuration) {
  static const auto expected = llvm::cantFail(llvm::json::parse(R"json({
  "arch_state_config": {
    "dram_base": 2147483648,
    "dram_size": 68719476736,
    "init_seed": 42,
    "mrf_depth": 32,
    "mrf_width": 32,
    "num_csrs": 4096,
    "num_e_registers": 32,
    "num_m_registers": 64,
    "num_wb_registers": 2,
    "num_x_registers": 32,
    "numerics": "rtl",
    "randomize_init": false,
    "vmem_size": 1572864,
    "wb_width": 1024
  },
  "dma_beat_bytes": 32,
  "dma_max_in_flight": 64,
  "dma_max_transfer_bytes": 4096,
  "dma_memory_backend": "curve",
  "dma_memory_cycles_per_beat": null,
  "dma_memory_latency_cycles": null,
  "dma_memory_params": {
    "curves": "ee290sim_vcs_probe.json"
  },
  "dma_num_channels": 8,
  "dma_offchip_command_words": 2,
  "execution_units": {
    "DMA0": "DmaExecutionUnit",
    "LSU": "LoadStoreUnit",
    "Matrix0": "MatrixExecutionUnitSystolic",
    "Matrix1": "MatrixExecutionUnitInner",
    "Scalar0": "ScalarExecutionUnit",
    "Vector0": "VectorExecutionUnit",
    "XLU": "CrossLaneExecutionUnit"
  },
  "fetch_width": 1,
  "ignore_runtime_errors": false,
  "isa": "npu_model.isa.IsaSpec",
  "mxu0_matmul_latency_cycles": 32,
  "mxu1_matmul_latency_cycles": 32,
  "name": "SimpleNPU",
  "offchip_link_core_cycles_per_beat": 2,
  "offchip_link_width_bits": 32,
  "vmem_bank_bytes": 262144,
  "vmem_bus_core_cycles_per_beat": 1,
  "vmem_bus_width_bits": 512,
  "vmem_bytes_per_cycle": 64,
  "vpu_non_pipelineable_op_latency_cycles": 16,
  "vpu_simple_op_latency_cycles": 4,
  "xlu_transform_latency_cycles": 4
})json"));
  return llvm::json::Value(Object(configuration)) == expected;
}

struct RegisterEvent {
  const char *key;
  int Instr::*operand;
  bool write;
  int registers = 1, passes = 1, offset = 0;
  bool twoInput = false;
};

// A null key captures a scale register at issue, without a facts event.
const std::map<OpClass, std::vector<RegisterEvent>> &vpuEvents() {
  static const std::map<OpClass, std::vector<RegisterEvent>> table = {
      {OpClass::VpuElementwise, {{"read0", &Instr::rs1, false, 2},
          {"read1", &Instr::rs2, false, 2, 1, 0, true}, {"write0", &Instr::rd, true, 2}}},
      {OpClass::VpuPack, {{"read0", &Instr::rs2, false, 2},
          {nullptr, &Instr::rs1, false}, {"write0", &Instr::rd, true}}},
      {OpClass::VpuUnpack, {{"read0", &Instr::rs2, false},
          {nullptr, &Instr::rs1, false}, {"write0", &Instr::rd, true, 2}}},
      {OpClass::VpuRowReduce, {{"read0", &Instr::rs1, false},
          {"read1", &Instr::rs1, false, 1, 1, 1}, {"write0", &Instr::rd, true}, {"write1", &Instr::rd, true, 1, 1, 1}}},
      {OpClass::VpuColReduce, {{"read0", &Instr::rs1, false, 2, 2}, {"write0", &Instr::rd, true, 2}}},
      {OpClass::VpuLoadImmPair, {{"write0", &Instr::rd, true, 2}}},
      {OpClass::VpuLoadImmSingle, {{"write0", &Instr::rd, true}}}};
  return table;
}

struct Fill {
  Footprint &f;
  std::string name, error, provider;
  const ModelTimingBlock *block = nullptr;
  int latency = 0;
  std::set<std::string> used;

  Fill(Footprint &footprint, const std::string &blockName, const std::map<std::string, ModelTimingBlock> &blocks,
       std::string policy = kNpuModelObservedTimingProviderId.str())
      : f(footprint), name(blockName), provider(std::move(policy)) {
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
  int mreg(const char *key, int reg, bool write, int registers = 1, int passes = 1, bool mergeRelease = true) {
    access(key, Res::MReg, write, reg * 32, registers * 32, passes);
    auto &list = write ? f.mregWrites : f.mregReads;
    for (int i = 0; i < registers; ++i) list.push_back(reg + i);
    int end = write ? last(key) : last(key) + latency; // Retain the source through its read response.
    int &release = write ? f.writeRelease : f.readRelease;
    release = mergeRelease ? std::max(release, end) : end;
    return end;
  }
  int busyEnd() { return block ? block->firstFreeAge - 1 : 0; }
  int nextIssue() { return block ? block->nextIssueAge : 0; }
  void bank(std::optional<int> line, const char *key) {
    if (line) f.holds.push_back({Unit::VmemBank, *line / kLinesPerBank, first(key), last(key)});
    else
      for (int bank = 0; bank < kVmemBanks; ++bank)
        f.holds.push_back({Unit::VmemBank, bank, first(key), last(key)});
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
    if (!error.empty()) {
      f.error = provider + ": " + error;
      return f;
    }
    f.doneAge = std::max({busy, f.readRelease, f.writeRelease});
    for (const Access &a : f.accesses) if (!a.atCompletion) f.doneAge = std::max(f.doneAge, a.lastAge());
    for (const Hold &h : f.holds) f.doneAge = std::max(f.doneAge, h.to);
    return f;
  }
};

const std::map<OpClass, const char *> &mxuBlocks() {
  static const std::map<OpClass, const char *> table = {
      {OpClass::WeightPush, "push_weight"}, {OpClass::AccPushFp8, "push_acc_fp8"},
      {OpClass::AccPushBf16, "push_acc_bf16"}, {OpClass::PopFp8, "pop_acc_fp8"},
      {OpClass::PopBf16, "pop_acc_bf16"}, {OpClass::MatMul, "matmul"}, {OpClass::MatMulAcc, "matmul_acc"}};
  return table;
}

std::string mxuBlockName(const OpInfo &op) {
  auto it = mxuBlocks().find(op.opClass);
  if (it == mxuBlocks().end() || op.mxu < 0 || op.mxu > 1) return {};
  return "mxu" + std::to_string(op.mxu) + "." + it->second;
}

Footprint mxuFootprint(const Instr &in, const std::map<std::string, ModelTimingBlock> &blocks,
                       const std::string &provider) {
  Footprint f;
  const OpInfo &op = *in.op;
  std::string blockName = mxuBlockName(op);
  if (blockName.empty()) {
    f.error = provider + ": unsupported MXU operation";
    return f;
  }
  const int m = op.mxu;
  Fill fill(f, blockName, blocks, provider);
  std::set<std::string> modeled;
  int busy = fill.busyEnd();
  auto track = [&](const char *key, int end) { modeled.insert(key); busy = std::max(busy, end); };
  auto mregRows = [&](const char *key, int reg, bool write) {
    if (reg < 0 || reg > 63) fill.fail("register operand out of range");
    track(key, fill.mreg(key, reg, write));
  };
  auto slotRows = [&](const char *key, Res res, int slot, bool write) {
    if (slot < 0 || slot > 1) fill.fail("accumulator and weight slots are 0 or 1");
    fill.access(key, res, write, (m * 2 + slot) * 32, 32);
    if (write) f.writeRelease = std::max(f.writeRelease, fill.last(key));
    track(key, fill.last(key));
  };
  if (in.imm != 0) fill.fail("unsupported MXU encoding");
  const bool weight = op.opClass == OpClass::WeightPush;
  const bool paired = op.opClass == OpClass::AccPushBf16 || op.opClass == OpClass::PopBf16;
  if (weight || op.opClass == OpClass::AccPushFp8 || op.opClass == OpClass::AccPushBf16) {
    if (in.rs2 != 0) fill.fail("unsupported MXU encoding");
    if (paired) {
      if (in.rs1 & 1) fill.fail("BF16 operands must name even register pairs");
      mregRows("mreg_read0", in.rs1, false);
    }
    mregRows("mreg_read1", in.rs1 + (paired ? 1 : 0), false);
    slotRows(weight ? "weight_write" : "acc_load", weight ? Res::Weight : Res::Acc, in.rd, true);
    track("data_busy", fill.last("data_busy"));
  } else if (op.opClass == OpClass::PopFp8 || op.opClass == OpClass::PopBf16) {
    slotRows("acc_store", Res::Acc, in.rs2, false);
    if (paired) {
      if (in.rs1 != 0) fill.fail("unsupported MXU encoding");
      if (in.rd & 1) fill.fail("BF16 operands must name even register pairs");
    } else {
      if (in.rs1 < 0 || in.rs1 > 31) fill.fail("scale register out of range");
      f.accesses.push_back({Res::EReg, false, in.rs1, 1, 0, 1});
    }
    mregRows("mreg_write0", in.rd, true);
    if (paired) mregRows("mreg_write1", in.rd + 1, true);
    track("data_busy", fill.last("data_busy"));
  } else { // matmul, overwrite or accumulate
    mregRows("mreg_read0", in.rs1, false);
    if (op.opClass == OpClass::MatMulAcc) slotRows("acc_read", Res::Acc, in.rd, false);
    slotRows("acc_write", Res::Acc, in.rd, true);
    track("compute_busy", fill.last("compute_busy"));
    if (in.rs2 < 0 || in.rs2 > 1) fill.fail("accumulator and weight slots are 0 or 1");
    for (int age : {0, busy})
      f.accesses.push_back({Res::Weight, false, (m * 2 + in.rs2) * 32, 32, age, 0});
  }
  if (fill.block)
    for (const auto &[key, group] : fill.block->events)
      if (group.count > 0 && !modeled.count(key))
        fill.fail("event group " + key + " is not modeled for this operation");
  f.holds.push_back({Unit::MxuCompute, m, 0, busy});
  return fill.done(busy);
}

} // namespace

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
    rejected.error = kNpuModelObservedTimingProviderId.str() + ": " + why;
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
    if (op.channel < 0 || op.channel >= 8) return reject("unsupported DMA channel");
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
    if (bytes < 32 || bytes > 4096 || bytes % 32 || words % 8 || words + bytes / 4 > uint64_t(kVmemBytes / 4) ||
        words / (kVmemBankBytes / 4) != (words + bytes / 4 - 1) / (kVmemBankBytes / 4) ||
        offset % 32 || offset + bytes > (uint64_t(1) << 32))
      return reject("DMA transfer is unaligned, out of range, oversized, or crosses a bank/address boundary");
    x(in.rd, false); x(in.rs1, false); x(in.rs2, false);
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
    if (address && (*address % 4 || *address + 4 > uint32_t(kVmemBytes)))
      return reject("scalar word access is unaligned or outside VMEM");
    Fill fill(f, store ? "scalar_lsu.store" : scale ? "scalar_lsu.scale_load" : "scalar_lsu.load", blocks, kNpuModelObservedTimingProviderId.str());
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
    if (line && (*line % 32 || *line + 32 > kVmemBytes / kLineBytes ||
                 *line / kLinesPerBank != (*line + 31) / kLinesPerBank))
      return reject("VLS effective tile is unaligned, out of range, or crosses a bank");
    Fill fill(f, load ? "vlsu.vload" : "vlsu.vstore", blocks, kNpuModelObservedTimingProviderId.str());
    const char *vmem = load ? "vmem_read" : "vmem_write";
    const char *rows = load ? "mreg_write" : "mreg_read";
    x(in.rs1, false);
    fill.access(vmem, Res::Vmem, !load, line.value_or(0), 32);
    fill.mreg(rows, in.rd, load, 1, 1, /*mergeRelease=*/false);
    int busy = fill.busyEnd();
    // Bank holds order VLS paths; MREG ports/releases order their overlap with VPU and XLU.
    fill.hold({load ? Unit::VloadPath : Unit::VstorePath}, 0, busy);
    fill.bank(line, vmem);
    if (!load) f.writeRelease = fill.last(vmem);
    // Unknown bases cover all VMEM/banks; ages still come only from the selected facts.
    for (Access &access : f.accesses)
      if (access.res == Res::Vmem) access.anywhere = !line;
    return fill.done(busy);
  }

  if (op.opClass == OpClass::Transpose) {
    if (!mreg(in.rd) || !mreg(in.rs1) || in.rs2 != 0 || in.imm != 0)
      return reject("unsupported XLU operands or encoding");
    Fill fill(f, "xlu.transpose", blocks, kNpuModelObservedTimingProviderId.str());
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
    Fill fill(f, blockName, blocks, kNpuModelObservedTimingProviderId.str());
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

  if (op.mxu >= 0) return mxuFootprint(in, blocks, kNpuModelObservedTimingProviderId.str());

  return reject("operation is outside the selected bounded scope");
}

namespace {
bool overlaps(const Access &a, const Access &b) {
  return a.res == b.res && (a.anywhere || b.anywhere ||
      (a.first < b.first + b.count && b.first < a.first + a.count));
}

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
          return x.age + (element - x.first) * x.step -
              (y.atCompletion ? 0 : y.age + (element - y.first) * y.step);
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
    if (contains(fb.mregWrites, reg)) consider(fa.readRelease + 1, EdgeKind::WAR, "observed MREG read release");
  return best;
}

bool sameEngine(const OpInfo &a, const OpInfo &b) {
  auto vector = [](Engine engine) { return engine == Engine::Vpu || engine == Engine::Xlu; };
  return a.engine == b.engine || (vector(a.engine) && vector(b.engine));
}

class ObservedReservations final : public TimingReservations {
public:
  TimingRuleResult<std::string> conflict(const Instr &in, const Footprint &f, int cycle) const override {
    if (cycle < 0 || cycle > INT_MAX - f.doneAge - 1)
      return {{}, "npu-model reservation exceeds cycle domain"};
    std::vector<Port> ports = requests(f, cycle);
    for (size_t i = 0; i < ports.size(); ++i)
      for (size_t j = 0; j < i; ++j)
        if (portConflict(ports[i], ports[j])) return {{}, "instruction has conflicting MREG port requests"};
    for (const Issued &prior : issued) {
      if (prior.cycle + prior.f.doneAge < cycle || cycle + f.doneAge < prior.cycle) continue;
      if (sameEngine(*in.op, *prior.in.op) && cycle <= prior.cycle + prior.f.doneAge &&
          prior.cycle <= cycle + f.doneAge)
        return {"npu-model isolated engine window remains live", {}};
      for (const Hold &a : prior.f.holds)
        for (const Hold &b : f.holds)
          if (a.unit == b.unit && a.index == b.index && prior.cycle + a.from <= cycle + b.to &&
              cycle + b.from <= prior.cycle + a.to)
            return {std::string(unitName(a.unit)) + " occupied", {}};
      for (const Port &a : prior.ports)
        for (const Port &b : ports)
          if (portConflict(a, b)) return {"observed MREG port occupied", {}};
    }
    return {};
  }
  std::string reserve(const Instr &in, const Footprint &f, int cycle) override {
    auto checked = conflict(in, f, cycle);
    if (!checked.error.empty()) return checked.error;
    if (!checked.value.empty()) return checked.value;
    issued.push_back({in, f, cycle, requests(f, cycle)});
    return {};
  }
  std::string onWait(const Instr &, int cycle) override {
    // The wait may stall indefinitely, so future events extend back to its issue.
    for (Issued &prior : issued) {
      for (Hold &hold : prior.f.holds)
        if (prior.cycle + hold.from > cycle) hold.from = cycle - prior.cycle;
      for (Port &port : prior.ports)
        if (port.cycle > cycle) port.from = std::min(port.from, cycle);
    }
    return {};
  }

private:
  struct Port { int bank, reg, row, from, cycle; bool write; };
  struct Issued { Instr in; Footprint f; int cycle; std::vector<Port> ports; };
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
};

std::string modelScope(const AtlasStream &stream) {
  if (stream.ops.size() > 32768) return "npu-model program exceeds instruction memory";
  struct Transfer { uint64_t first, end; bool write, waited; };
  std::vector<Transfer> transfers;
  std::array<int, 8> pending;
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
        if (op.channel < 0 || op.channel >= 8 || pending[op.channel] >= 0)
          return "npu-model requires a free DMA channel";
        if (!upper || !regs[in.rd] || !regs[in.rs1] || !regs[in.rs2])
          return "npu-model DMA requires known captured operands and upper base";
        if (transfers.size() >= 8 && !transfers[transfers.size() - 8].waited)
          return "npu-model DMA command slot is still pending";
        bool load = op.opClass == OpClass::DmaLoad;
        uint64_t first = (uint64_t(*upper) << 32) | *regs[load ? in.rs1 : in.rd];
        uint64_t end = first + *regs[in.rs2];
        if (first < uint64_t(0x80000000) || end > uint64_t(0x1080000000))
          return "npu-model DMA DRAM span is outside the selected configuration";
        for (const Transfer &prior : transfers)
          if (!prior.waited && first < prior.end && prior.first < end && (!load || prior.write))
            return "npu-model pending DMA DRAM spans overlap";
        pending[op.channel] = int(transfers.size());
        transfers.push_back({first, end, !load, false});
      } else if (op.opClass == OpClass::DmaWait) {
        if (op.channel < 0 || op.channel >= 8 || pending[op.channel] < 0)
          return "npu-model DMA wait has no pending transfer";
        transfers[pending[op.channel]].waited = true;
        pending[op.channel] = -1;
      }
      applyScalar(in, regs);
    }
  }
  return {};
}
} // namespace

TimingProvider mlir::atlas::timing::modelTimingProvider(std::shared_ptr<const ModelTimingFacts> facts) {
  TimingProvider provider;
  provider.id = kNpuModelObservedTimingProviderId.str();
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
  provider.dependence = [](const Instr &a, const Footprint &fa, const Instr &b, const Footprint &fb) {
    return TimingRuleResult<Dependence>{observedDependence(a, fa, b, fb), {}};
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
  // DELAY encoding and issue-edge scalar/control/DMA captures are architectural rules.
  provider.issueGap = [](const Instr &in) { return TimingRuleResult<int>{naturalGap(in), {}}; };
  provider.createReservations = [] {
    return TimingRuleResult<std::unique_ptr<TimingReservations>>{std::make_unique<ObservedReservations>(), {}};
  };
  return provider;
}

TimingRuleResult<TimingProvider> mlir::atlas::timing::modelTimingProviderFactory(ModuleOp module) {
  if (!module) return {{}, "npu-model requires selected atlas.model_evidence"};
  if (module->hasAttr("atlas.rtl_evidence")) return {{}, "model and RTL evidence cannot be selected together"};
  auto selection = module->getAttrOfType<DictionaryAttr>(kAtlasModelEvidence);
  if (!selection) return {{}, "npu-model requires selected atlas.model_evidence"};
  auto path = selection.getAs<StringAttr>("op_timing"), hash = selection.getAs<StringAttr>("op_timing_sha256");
  auto id = selection.getAs<StringAttr>("provider"), dma = selection.getAs<StringAttr>("dma");
  if (!path || !hash || !id || id.getValue() != kNpuModelObservedTimingProviderId ||
      (selection.get("dma") && (!dma || dma.getValue() != "wait")))
    return {{}, "incomplete or unsupported model evidence selection"};
  auto facts = loadModelTimingFacts(path.getValue(), hash.getValue(), bool(dma));
  if (!facts) return {{}, llvm::toString(facts.takeError())};
  for (const auto &[key, value] : facts->modelProvenance()) {
    auto retained = selection.getAs<StringAttr>(key);
    if (!retained || retained.getValue() != value) return {{}, "model evidence provenance mismatch: " + key};
  }
  return {modelTimingProvider(std::make_shared<ModelTimingFacts>(std::move(*facts))), {}};
}

namespace {
struct SelectAtlasModelEvidencePass : PassWrapper<SelectAtlasModelEvidencePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SelectAtlasModelEvidencePass)
  SelectAtlasModelEvidencePass() = default;
  SelectAtlasModelEvidencePass(const SelectAtlasModelEvidencePass &other) : PassWrapper(other) {}
  Option<std::string> opTiming{*this, "op-timing", llvm::cl::desc("Observed atlas.model_timing.v1 facts")};
  Option<std::string> opTimingHash{*this, "op-timing-sha256", llvm::cl::desc("Expected facts SHA-256")};
  Option<std::string> dmaPolicy{*this, "dma", llvm::cl::desc("Optional DMA completion policy: wait")};
  StringRef getArgument() const final { return "select-atlas-model-evidence"; }
  StringRef getDescription() const final { return "Pin model observations; timing passes select provider=npu-model"; }
  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto fail = [&](const std::string &why) { module.emitError(why); signalPassFailure(); };
    if (module->hasAttr("atlas.rtl_evidence")) return fail("model and RTL evidence cannot be selected together");
    if (auto state = module->getAttrOfType<StringAttr>(kAtlasTimingState); state && state.getValue() == "timed")
      return fail("select model evidence before timing; already timed input must be retimed");
    if (module->hasAttr(kAtlasTimingProvider)) return fail("model selection cannot replace a retained timing provider");
    if (!dmaPolicy.empty() && dmaPolicy != "wait") return fail("only dma=wait is supported");
    auto facts = loadModelTimingFacts(opTiming, opTimingHash, dmaPolicy == "wait");
    if (!facts) return fail(llvm::toString(facts.takeError()));
    Builder builder(module.getContext());
    NamedAttrList selection;
    selection.set("provider", builder.getStringAttr(kNpuModelObservedTimingProviderId));
    selection.set("op_timing", builder.getStringAttr(opTiming));
    selection.set("op_timing_sha256", builder.getStringAttr(opTimingHash));
    if (facts->dmaWait()) selection.set("dma", builder.getStringAttr("wait"));
    for (const auto &[key, value] : facts->modelProvenance()) selection.set(key, builder.getStringAttr(value));
    module->setAttr(kAtlasModelEvidence, selection.getDictionary(module.getContext()));
  }
};
} // namespace

void mlir::atlas::registerSelectAtlasModelEvidencePass() { PassRegistration<SelectAtlasModelEvidencePass>(); }
