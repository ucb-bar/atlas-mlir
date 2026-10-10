#include "Atlas/AtlasRTLEvidence.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SHA256.h"
#include <algorithm>
#include <set>

using namespace mlir::atlas::timing;
namespace {
using Object = llvm::json::Object;
using Array = llvm::json::Array;

constexpr int kLinesPerBank = kVmemBankBytes / kLineBytes;

llvm::Error failure(const std::string &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "RTL timing facts: " + message);
}

bool hashSyntax(llvm::StringRef value) {
  return value.size() == 64 && std::all_of(value.begin(), value.end(), [](char c) {
    return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
  });
}

std::optional<int> integer(const Object *object, llvm::StringRef key) {
  auto value = object ? object->getInteger(key) : std::nullopt;
  if (!value) return std::nullopt;
  return static_cast<int>(*value);
}

// Compiler mnemonic -> facts block, following test/rtl-timing-facts-map.yaml.
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
      {"vli.all", "vpu.vliAll"}, {"vli.row", "vpu.vliRow"},
      {"vli.col", "vpu.vliCol"}, {"vli.one", "vpu.vliOne"}};
  return table;
}

// Fills a compiler-structured footprint with one facts block's numbers.
struct Fill {
  Footprint &f;
  std::string name, error;
  const OpTimingBlock *block = nullptr;
  int latency = 0;

  Fill(Footprint &footprint, const std::string &blockName,
       const std::map<std::string, OpTimingBlock> &blocks)
      : f(footprint), name(blockName) {
    auto it = blocks.find(name);
    if (it == blocks.end()) fail("facts block is missing");
    else if (!it->second.resolved) fail("facts block is unresolved");
    else if (!it->second.readLatency) fail("facts block lacks scratchpad_read_latency");
    else { block = &it->second; latency = *block->readLatency; }
  }
  void fail(const std::string &why) { if (error.empty()) error = name + ": " + why; }
  const OpTimingGroup *group(const char *key) {
    if (!block) return nullptr;
    auto it = block->events.find(key);
    if (it == block->events.end() || it->second.count <= 0) {
      fail(std::string("event group ") + key + " is missing"); return nullptr;
    }
    return &it->second;
  }
  // `passes` sweeps over `elements` consecutive elements from `first`, timed by
  // the group: element i of pass p is touched at first_age + (p*elements+i)*step.
  void access(const char *key, Res res, bool write, int first, int elements,
              int passes = 1) {
    const OpTimingGroup *g = group(key);
    if (!g) return;
    int step = g->step.value_or(1);
    if (g->count != elements * passes || step <= 0 ||
        g->lastAge != g->firstAge + (g->count - 1) * step) {
      fail(std::string("event group ") + key + " does not sweep " +
           std::to_string(passes) + "x" + std::to_string(elements) + " elements evenly");
      return;
    }
    for (int p = 0; p < passes; ++p)
      f.accesses.push_back({res, write, first, elements,
                            g->firstAge + p * elements * step, step});
  }
  int first(const char *key) { const OpTimingGroup *g = group(key); return g ? g->firstAge : 0; }
  int last(const char *key) { const OpTimingGroup *g = group(key); return g ? g->lastAge : 0; }
  // A source is retained through its synchronous read response.
  int readEnd(const char *key) { return last(key) + latency; }
  // Inclusive hold end: the RTL's first free age is one past it.
  int busyEnd() {
    if (block && !block->firstFreeAge) fail("facts block lacks first_free_age");
    return block && block->firstFreeAge ? *block->firstFreeAge - 1 : 0;
  }
  int nextIssue() { return block && block->nextIssueAge ? *block->nextIssueAge : 0; }
  void bank(int line, const char *key) {
    f.holds.push_back({Unit::VmemBank, line / kLinesPerBank, first(key), last(key)});
  }
  void engine(Unit unit, int index, int busy) {
    f.holds.push_back({unit, index, 0, busy});
    f.holds.push_back({Unit::VloadPath, 0, 0, busy});
    f.holds.push_back({Unit::VstorePath, 0, 0, busy});
    f.serializeWithDMA = true;
  }
  Footprint done(int busy) {
    if (!error.empty()) {
      f.error = std::string(RTLEvidence::resolverID()) + ": " + error;
      return f;
    }
    f.doneAge = std::max({busy, f.readRelease, f.writeRelease});
    for (const Access &a : f.accesses)
      if (!a.atCompletion) f.doneAge = std::max(f.doneAge, a.lastAge());
    for (const Hold &h : f.holds) f.doneAge = std::max(f.doneAge, h.to);
    return f;
  }
};
} // namespace

llvm::Expected<RTLEvidence> mlir::atlas::timing::loadRTLTimingFacts(
    llvm::StringRef path, llvm::StringRef expectedSha256, bool dmaWait) {
  if (!hashSyntax(expectedSha256)) return failure("facts SHA-256 is required");
  auto buffer = llvm::MemoryBuffer::getFile(path);
  if (!buffer) return failure("cannot read " + path.str());
  llvm::StringRef bytes = (*buffer)->getBuffer();
  auto hash = llvm::SHA256::hash(llvm::ArrayRef<uint8_t>(
      reinterpret_cast<const uint8_t *>(bytes.data()), bytes.size()));
  if (llvm::toHex(llvm::ArrayRef<uint8_t>(hash), true) != expectedSha256)
    return failure("selected facts hash mismatch");
  auto parsed = llvm::json::parse(bytes);
  if (!parsed) return failure(llvm::toString(parsed.takeError()));
  const Object *root = parsed->getAsObject();
  if (!root || root->getString("schema") != "merlin.op_timing.v1")
    return failure("expected schema merlin.op_timing.v1");
  const Object *hardware = root->getObject("hw_ir");
  auto hardwareSha = hardware ? hardware->getString("sha256") : std::nullopt;
  if (!hardwareSha || !hashSyntax(*hardwareSha)) return failure("missing hw_ir.sha256");
  const Array *timing = root->getArray("op_timing");
  if (!timing) return failure("missing op_timing");
  RTLEvidence result;
  for (const auto &value : *timing) {
    const Object *entry = value.getAsObject();
    auto name = entry ? entry->getString("name") : std::nullopt;
    if (!name) return failure("op_timing block without a name");
    OpTimingBlock block;
    block.readLatency = integer(entry->getObject("assumptions"), "scratchpad_read_latency");
    block.firstFreeAge = integer(entry, "first_free_age");
    block.nextIssueAge = integer(entry, "next_issue_age");
    if (const Object *events = entry->getObject("events")) {
      block.resolved = true;
      for (const auto &[key, group] : *events) {
        const Object *g = group.getAsObject();
        auto count = integer(g, "count");
        if (!count) return failure(name->str() + "." + key.str() + " has no count");
        OpTimingGroup out;
        out.count = *count;
        out.step = integer(g, "step");
        if (*count > 0) {
          auto first = integer(g, "first_age"), last = integer(g, "last_age");
          if (!first || !last) return failure(name->str() + "." + key.str() + " lacks ages");
          out.firstAge = *first;
          out.lastAge = *last;
        }
        block.events[key.str()] = out;
      }
    } else if (const llvm::json::Value *events = entry->get("events");
               !events || events->kind() != llvm::json::Value::Null) {
      return failure(name->str() + ": events must be an object or null");
    }
    if (!result.blocks.emplace(name->str(), std::move(block)).second)
      return failure("duplicate op_timing block " + name->str());
  }
  result.sha256 = expectedSha256.str();
  result.hardwareIR = hardwareSha->str();
  result.dma = dmaWait;
  return result;
}

TargetTiming RTLEvidence::targetTiming() const {
  return {[facts = *this](const Instr &in, const RegValues &regs) {
    return facts.resolve(in, regs);
  }};
}

llvm::json::Object RTLEvidence::applicability() const {
  auto firstFree = [&](const char *name) -> llvm::json::Value {
    auto it = blocks.find(name);
    if (it == blocks.end() || !it->second.firstFreeAge) return nullptr;
    return *it->second.firstFreeAge;
  };
  auto vload = blocks.find("vlsu.vload");
  int latency = vload != blocks.end() ? vload->second.readLatency.value_or(0) : 0;
  Array operations{"addi", "lui", "lw", "sw", "seld", "vload", "vstore", "vtrpose.xlu",
                   "delay", "csrrw", "ecall", "beq", "bne", "blt", "bge", "bltu",
                   "bgeu", "jal"};
  for (const auto &[mnemonic, block] : vpuBlocks()) operations.push_back(mnemonic);
  if (dma)
    for (const char *name : {"dma.load.ch0..7", "dma.store.ch0..7",
                             "dma.config.ch0..7", "dma.wait.ch0..7"})
      operations.push_back(name);
  Object result{
      {"scope", dma ? "drained_basic_blocks_serialized_engines_and_wait_governed_dma" :
                      "drained_basic_blocks_serialized_engines_with_scalar_setup_and_completion"},
      {"control_flow", Object{
          {"block_entry_state", "idle_engines_no_pending_dma"},
          {"block_exit", "all_prior_work_complete_by_earliest_successor_issue"},
          {"earliest_successor_issue", "branch_issue_plus_2"},
          {"delay_slots", 1}, {"delay_slot_operations", Array{"addi", "lui"}},
          {"delay_immediately_before_redirect_supported", false},
          {"scalar_values_at_joins", "equal_on_all_paths_or_unknown"},
          {"linking_and_register_targets_supported", false}}},
      {"timing_source", "merlin.op_timing.v1 facts computed from the RTL; footprint structure from the compiler"},
      {"supported_operations", std::move(operations)},
      {"operand_domain", Object{
          {"scalar_registers", Object{{"first", 0}, {"count", 32}}},
          {"mreg_registers", Object{{"first", 0}, {"count", 64}}},
          {"vmem_banks", kVmemBanks}, {"vmem_bank_bytes", kVmemBankBytes},
          {"vmem_line_bytes", kLineBytes}, {"tile_rows", 32},
          {"vmem_start_line_alignment", 32},
          {"vmem_last_start_line", kVmemBytes / kLineBytes - 32},
          {"base_unit", "32_bit_word"},
          {"effective_line", "(((base + 32 * sext12(offset)) mod 2^32) >> 3) & 0xffff"},
          {"known_base_required", true}, {"single_bank_required", true},
          {"scalar_word_access", "known 4-byte aligned address inside VMEM"},
          {"mlir_offset_min", -2048}, {"mlir_offset_max", 2047},
          {"csr_constraint", "csrrw x0,0xC10,rs"},
          {"vpu_operands", "even pair bases; all register operands in distinct physical banks (register mod 32)"},
          {"release_annotation_supported", false}}},
      {"environment_assumptions", Array{
          "reset_deasserted", "accelerator_quiescent_at_entry",
          "other_engines_and_competing_memory_requesters_quiescent",
          "sram_read_response_" + std::to_string(latency) + "_cycles_after_request",
          "instructions_immutable_during_execution",
          "host_does_not_access_vmem_or_rewrite_dbg0_during_execution",
          "caller_supplies_required_initial_operand_data",
          "engine_timing_is_data_independent"}},
      {"admission", Object{
          {"selected_engines_serialized", true}, {"vls_paths_serialized", true},
          {"minimum_vls_issue_gap", firstFree("vlsu.vload")},
          {"xlu_first_free_age", firstFree("xlu.transpose")},
          {"marker_and_terminal_require_prior_writes_complete", true},
          {"delay_immediately_before_terminal_supported", false},
          {"maximum_program_words", static_cast<int64_t>(maximumProgramWords())}}},
      {"unsupported", Array{dma ? "multiple_pending_dma_transfers" : "dma_and_dynamic_completion",
          "mxu_operations", "engine_work_or_dma_live_across_block_boundaries",
          "addresses_that_vary_across_loop_iterations",
          "concurrent_engine_or_host_memory_traffic",
          "alternative_memory_implementations_without_review"}},
      {"domain_qualification", "conditional; computed_timing_is_not_execution_qualification"}};
  if (dma) {
    result["dma_domain"] = Object{
        {"global_pending_limit", 1}, {"channels", 8},
        {"vmem_pointer_unit", "32_bit_word"}, {"vmem_word_alignment", 8},
        {"size_unit", "byte"}, {"size_min", 32}, {"size_max", 4096},
        {"size_alignment", 32}, {"single_vmem_bank_required", true},
        {"dram_offset_alignment", 32}, {"dram_base_max", 31},
        {"dram_offset_wrap_supported", false},
        {"dram_address", "(configured_base << 32) | captured_offset"},
        {"scalar_and_base_reads", "captured_at_launch"},
        {"memory_lifetime", "launch_through_matching_wait"},
        {"completion_latency", nullptr},
        {"vmem_competition_while_pending", false},
        {"publication_requires_matching_wait", true}};
    result.getArray("environment_assumptions")->push_back(
        "fresh_host_START_resets_global_dma_base_to_zero; DMA_queues_already_quiescent");
    result.getArray("environment_assumptions")->push_back(
        "external_memory_responds_correctly_and_eventually; no_fixed_latency_bound");
  }
  return result;
}

Footprint RTLEvidence::resolve(const Instr &in, const RegValues &regs) const {
  Footprint f;
  auto reject = [&](const char *why) {
    f.error = std::string(resolverID()) + ": " + why; return f;
  };
  if (!in.op) return reject("invalid instruction");
  const OpInfo &op = *in.op;
  const std::string &name = op.name;
  auto xreg = [](int r) { return r >= 0 && r <= 31; };
  auto mreg = [](int r) { return r >= 0 && r <= 63; };
  auto x = [&](int reg, bool write) {
    if (reg) f.accesses.push_back({Res::XReg, write, reg, 1, 0, 1});
  };
  if (in.release && op.opClass != OpClass::Csr)
    return reject("unsupported completion annotation");

  if (name == "addi" || name == "lui") {
    if (!xreg(in.rd) || !xreg(in.rs1)) return reject("unsupported scalar operands");
    if (name == "addi") x(in.rs1, false);
    x(in.rd, true);
    if (in.rd) f.holds.push_back({Unit::ScalarWriteback, 0, 0, 0});
    return f;
  }
  if (name == "delay" || name == "ecall") return f;
  // Control flow reads its operands at issue like ADDI. The stream verifier
  // drains every engine before a block's successors can start, so the redirect
  // latency only has to be at least the delay slot.
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
    if (op.opClass == OpClass::DmaWait)
      return f; // pairing is checked by the stream verifier
    if (op.opClass == OpClass::DmaConfig) {
      if (!xreg(in.rs1) || !regs[in.rs1] || *regs[in.rs1] > 31)
        return reject("DMA configuration requires a known 5-bit upper DRAM base");
      x(in.rs1, false);
      f.accesses.push_back({Res::DmaBase, true, 0, 1, 0, 1});
      return f;
    }
    if (op.opClass != OpClass::DmaLoad && op.opClass != OpClass::DmaStore)
      return reject("unsupported DMA operation");
    if (!xreg(in.rd) || !xreg(in.rs1) || !xreg(in.rs2) ||
        !regs[in.rd] || !regs[in.rs1] || !regs[in.rs2])
      return reject("DMA requires known scalar VMEM, DRAM and size operands");
    const bool load = op.opClass == OpClass::DmaLoad;
    const uint64_t words = *regs[load ? in.rd : in.rs1],
                   offset = *regs[load ? in.rs1 : in.rd], bytes = *regs[in.rs2];
    if (bytes < 32 || bytes > 4096 || bytes % 32 || words % 8 ||
        words + bytes / 4 > uint64_t(kVmemBytes / 4) ||
        words / (kVmemBankBytes / 4) != (words + bytes / 4 - 1) / (kVmemBankBytes / 4) ||
        offset % 32 || offset + bytes > (uint64_t(1) << 32))
      return reject("DMA transfer is unaligned, out of range, oversized, or crosses a bank/address boundary");
    x(in.rd, false); x(in.rs1, false); x(in.rs2, false);
    f.accesses.push_back({Res::DmaBase, false, 0, 1, 0, 1});
    f.accesses.push_back({Res::Vmem, load, int(words / 8), int(bytes / 32), 0, 0, false, true});
    // The external address space is retained whole; this is not an alias proof.
    f.accesses.push_back({Res::Dram, !load, 0, 1, 0, 0, true, true});
    f.dmaAsync = true;
    f.exclusiveVmemUntilWait = true;
    return f;
  }

  if (name == "lw" || name == "seld" || name == "sw") {
    const bool store = name == "sw", scale = name == "seld";
    if (!xreg(in.rs1) || !xreg(in.rs2) || (!scale && !xreg(in.rd)))
      return reject("unsupported scalar memory operands");
    if (!regs[in.rs1]) return reject("unknown scalar base cannot establish bounded address domain");
    const uint32_t address = (*regs[in.rs1] + uint32_t(signExtend(in.imm, 12))) & 0x1FFFFF;
    if (address % 4 || address + 4 > uint32_t(kVmemBytes))
      return reject("scalar word access is unaligned or outside VMEM");
    Fill fill(f, store ? "scalar_lsu.store" : scale ? "scalar_lsu.scale_load" : "scalar_lsu.load", blocks);
    const int line = address / kLineBytes;
    x(in.rs1, false);
    if (store) {
      x(in.rs2, false);
      fill.access("vmem_write", Res::Vmem, true, line, 1);
      fill.bank(line, "vmem_write");
    } else {
      const char *write = scale ? "scale_write" : "load_write";
      fill.access("vmem_read", Res::Vmem, false, line, 1);
      if (scale || in.rd) fill.access(write, scale ? Res::EReg : Res::XReg, true, in.rd, 1);
      fill.bank(line, "vmem_read");
      f.holds.push_back({Unit::ScalarWriteback, 0, fill.first(write), fill.first(write)});
    }
    int busy = fill.busyEnd();
    f.holds.push_back({Unit::ScalarLoad, 0, 0, busy});
    return fill.done(busy);
  }

  if (op.opClass == OpClass::VLoad || op.opClass == OpClass::VStore) {
    const bool load = op.opClass == OpClass::VLoad;
    if (!mreg(in.rd) || !xreg(in.rs1)) return reject("unsupported VLS operands");
    if (in.imm < -2048 || in.imm > 4095) return reject("VLS immediate is not a 12-bit encoding");
    if (!regs[in.rs1]) return reject("unknown VLS base cannot establish bounded address domain");
    const uint32_t effectiveWords = *regs[in.rs1] + uint32_t(signExtend(in.imm, 12) * 32);
    const uint32_t line = (effectiveWords >> 3) & 0xffff;
    if (line % 32 || line + 32 > uint32_t(kVmemBytes / kLineBytes) ||
        line / kLinesPerBank != (line + 31) / kLinesPerBank)
      return reject("VLS effective tile is unaligned, out of range, or crosses a bank");
    Fill fill(f, load ? "vlsu.vload" : "vlsu.vstore", blocks);
    const char *vmem = load ? "vmem_read" : "vmem_write";
    const char *rows = load ? "mreg_write" : "mreg_read";
    x(in.rs1, false);
    fill.access(vmem, Res::Vmem, !load, int(line), 32);
    fill.access(rows, Res::MReg, load, in.rd * 32, 32);
    int busy = fill.busyEnd();
    f.holds.push_back({Unit::VloadPath, 0, 0, busy});
    f.holds.push_back({Unit::VstorePath, 0, 0, busy});
    fill.bank(int(line), vmem);
    if (load) {
      f.mregWrites = {in.rd};
      f.writeRelease = fill.last(rows);
    } else {
      f.mregReads = {in.rd};
      f.readRelease = fill.readEnd(rows);
      f.writeRelease = fill.last(vmem);
    }
    return fill.done(busy);
  }

  if (op.opClass == OpClass::Transpose) {
    if (!mreg(in.rd) || !mreg(in.rs1) || in.rs2 != 0 || in.imm != 0)
      return reject("unsupported XLU operands or encoding");
    Fill fill(f, "xlu.transpose", blocks);
    fill.access("read", Res::MReg, false, in.rs1 * 32, 32);
    fill.access("write", Res::MReg, true, in.rd * 32, 32);
    f.mregReads = {in.rs1};
    f.mregWrites = {in.rd};
    f.readRelease = fill.readEnd("read");
    f.writeRelease = fill.last("write");
    int busy = fill.busyEnd();
    fill.engine(Unit::Xlu, 0, busy);
    return fill.done(busy);
  }

  if (op.engine == Engine::Vpu) {
    auto block = vpuBlocks().find(name);
    if (block == vpuBlocks().end()) return reject("unsupported VPU operation");
    const bool immediate = op.opClass == OpClass::VpuLoadImmPair ||
                           op.opClass == OpClass::VpuLoadImmSingle;
    if (!immediate && in.imm != 0) return reject("unsupported VPU encoding");
    Fill fill(f, block->second, blocks);
    std::vector<const char *> reads, writes;
    auto note = [&](bool write, const char *key, std::initializer_list<int> used) {
      auto &list = write ? f.mregWrites : f.mregReads;
      list.insert(list.end(), used);
      (write ? writes : reads).push_back(key);
    };
    auto pair = [&](int reg, bool write, const char *key, int passes = 1) {
      if (!mreg(reg) || (reg & 1)) fill.fail("BF16 operands must name even register pairs");
      fill.access(key, Res::MReg, write, reg * 32, 64, passes);
      note(write, key, {reg, reg + 1});
    };
    auto single = [&](int reg, bool write, const char *key) {
      if (!mreg(reg)) fill.fail("register operand out of range");
      fill.access(key, Res::MReg, write, reg * 32, 32);
      note(write, key, {reg});
    };
    switch (op.opClass) {
    case OpClass::VpuElementwise:
      pair(in.rs1, false, "read0");
      if (op.twoInput) pair(in.rs2, false, "read1");
      pair(in.rd, true, "write0");
      break;
    case OpClass::VpuPack:
      pair(in.rs2, false, "read0");
      f.accesses.push_back({Res::EReg, false, in.rs1, 1, 0, 1});
      single(in.rd, true, "write0");
      break;
    case OpClass::VpuUnpack:
      single(in.rs2, false, "read0");
      f.accesses.push_back({Res::EReg, false, in.rs1, 1, 0, 1});
      pair(in.rd, true, "write0");
      break;
    case OpClass::VpuRowReduce:
      if ((in.rs1 & 1) || (in.rd & 1)) fill.fail("BF16 operands must name even register pairs");
      single(in.rs1, false, "read0");
      single(in.rs1 + 1, false, "read1");
      single(in.rd, true, "write0");
      single(in.rd + 1, true, "write1");
      break;
    case OpClass::VpuColReduce:
      pair(in.rs1, false, "read0", /*passes=*/2);
      pair(in.rd, true, "write0");
      break;
    case OpClass::VpuLoadImmPair:
      pair(in.rd, true, "write0");
      break;
    case OpClass::VpuLoadImmSingle:
      single(in.rd, true, "write0");
      break;
    default:
      return reject("unsupported VPU operation");
    }
    std::set<int> banks;
    for (const std::vector<int> *group : {&f.mregReads, &f.mregWrites})
      for (int reg : *group)
        if (!banks.insert(reg & 31).second)
          fill.fail("VPU register operands must occupy distinct physical banks");
    for (const char *key : reads) f.readRelease = std::max(f.readRelease, fill.readEnd(key));
    f.writeRelease = fill.nextIssue();
    for (const char *key : writes) f.writeRelease = std::max(f.writeRelease, fill.last(key));
    int busy = fill.busyEnd();
    fill.engine(Unit::Vpu, 0, busy);
    return fill.done(busy);
  }

  return reject("operation is outside the selected bounded scope");
}
