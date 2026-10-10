#include "Atlas/AtlasRTLExport.h"
#include "Atlas/AtlasRTLVerification.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/SHA256.h"

using namespace mlir;
using namespace mlir::atlas;
using namespace mlir::atlas::timing;

namespace {
using Object = llvm::json::Object;
using Array = llvm::json::Array;

const char *resourceName(Res resource) {
  switch (resource) {
  case Res::XReg: return "xreg";
  case Res::EReg: return "ereg";
  case Res::MReg: return "mreg";
  case Res::Acc: return "acc";
  case Res::Weight: return "weight";
  case Res::Vmem: return "vmem";
  case Res::DmaBase: return "dma_base";
  case Res::Dram: return "dram";
  }
  llvm_unreachable("unhandled timing resource");
}

Array integers(const std::vector<int> &values) {
  Array result;
  for (int value : values) result.push_back(value);
  return result;
}

Object footprintJSON(const Footprint &footprint, bool dynamic) {
  Array accesses, holds;
  for (const Access &access : footprint.accesses) {
    Object fact{{"resource", resourceName(access.res)},
        {"write", access.write}, {"first", access.first}, {"count", access.count},
        {"age", access.age}, {"step", access.step}, {"anywhere", access.anywhere},
        {"at_completion", access.atCompletion}};
    if (dynamic && access.atCompletion) {
      fact["age"] = nullptr;
      fact["step"] = nullptr;
      fact["lifetime"] = "launch_through_matching_wait";
    }
    accesses.push_back(std::move(fact));
  }
  for (const Hold &hold : footprint.holds)
    holds.push_back(Object{{"unit", unitName(hold.unit)}, {"index", hold.index},
        {"from", hold.from}, {"to", hold.to}, {"alt", hold.alt}});
  Object result{{"accesses", std::move(accesses)}, {"holds", std::move(holds)},
      {"mreg_reads", integers(footprint.mregReads)},
      {"mreg_writes", integers(footprint.mregWrites)},
      {"read_release", footprint.readRelease},
      {"write_release", footprint.writeRelease},
      {"write_during_read", footprint.writeDuringRead},
      {"vpu_live", footprint.vpuLive}, {"done_age", footprint.doneAge},
      {"dma_cycles", footprint.dmaCycles}};
  if (dynamic) {
    result["dma_async"] = footprint.dmaAsync;
    result["exclusive_vmem_until_wait"] = footprint.exclusiveVmemUntilWait;
    result["serialize_with_dma"] = footprint.serializeWithDMA;
    if (footprint.dmaAsync) {
      result["done_age"] = nullptr;
      result["read_release"] = nullptr;
      result["write_release"] = nullptr;
      result["dma_cycles"] = nullptr;
      result["completion"] = "matching_dma_wait";
    }
  }
  return result;
}
} // namespace

FailureOr<llvm::json::Object>
mlir::atlas::exportAtlasRTLTiming(ModuleOp module) {
  ResolvedRTLProgram program;
  if (failed(verifyAtlasRTLTiming(module, &program)))
    return failure();
  Array words, instructions;
  std::vector<uint8_t> encoded;
  for (uint32_t word : program.words) {
    words.push_back(static_cast<int64_t>(word));
    for (unsigned shift = 0; shift < 32; shift += 8)
      encoded.push_back(static_cast<uint8_t>(word >> shift));
  }
  auto hash = llvm::SHA256::hash(encoded);
  const auto &evidence = *program.evidence;
  const bool dynamic = evidence.dmaWait();
  for (size_t i = 0; i < program.instructions.size(); ++i) {
    const auto &entry = program.instructions[i];
    const Instr &in = entry.instruction;
    Object entryJSON{
        {"word_index", static_cast<int64_t>(i)},
        {"word_u32", static_cast<int64_t>(program.words[i])},
        {"mnemonic", in.op->name},
        {"operands", Object{{"rd", in.rd}, {"rs1", in.rs1}, {"rs2", in.rs2},
            {"immediate", static_cast<int64_t>(in.imm)}, {"release", in.release}}},
        {"logical_issue_cycle", entry.cycle},
        {"event_kind", in.op->opClass == OpClass::Halt ?
            "terminal_acceptance" : "instruction_issue"},
        {"footprint", footprintJSON(entry.footprint, dynamic)}};
    if (dynamic) {
      entryJSON.erase("logical_issue_cycle");
      entryJSON["minimum_issue_cycle"] = entry.cycle;
      entryJSON["issue_epoch"] = entry.epoch;
      entryJSON["epoch_offset"] = entry.epochOffset;
      if (in.op->opClass == OpClass::DmaWait) {
        entryJSON["event_kind"] = "matching_wait_acceptance";
        entryJSON["epoch_origin"] = "matching_wait_acceptance";
        entryJSON["channel"] = in.op->channel;
      }
    }
    instructions.push_back(std::move(entryJSON));
  }
  Object result{
      {"schema", dynamic ? "atlas.resolved_rtl_timing.v1" : "atlas.resolved_rtl_timing.v0"},
      {"target_config", "EE290SimConfig"},
      {"qualification", "conditional"},
      {"scheduling_qualified", false},
      {"resolver", Object{{"id", RTLEvidence::resolverID()},
                          {"version", 1},
                          {"dma_policy", dynamic ? "wait" : "none"}}},
      {"evidence", Object{{"op_timing_schema", "merlin.op_timing.v1"},
          {"op_timing_sha256", evidence.factsSha256()},
          {"hardware_ir_sha256", evidence.hardwareIRSha256()}}},
      {"program", Object{{"word_count", static_cast<int64_t>(program.words.size())},
          {"words", std::move(words)},
          {"words_sha256", llvm::toHex(llvm::ArrayRef<uint8_t>(hash), true)},
          {"hash_encoding", "little_endian_u32"}}},
      {"conventions", Object{
          {"cycle_origin", "first_instruction_logical_issue"},
          {"ages", "relative_to_instruction_issue; holds_have_inclusive_endpoints"},
          {"access_elements", "xreg/ereg:register; mreg:register*32+row; vmem:32_byte_line"},
          {"timeline", "conditional_model_timeline_not_measured_elapsed_cycles"},
          {"terminal", "acceptance_not_retirement; ECALL_suppresses_scalar_fire"}}},
      {"applicability", evidence.applicability()},
      {"instructions", std::move(instructions)}};
  if (dynamic) {
    auto *conventions = result.getObject("conventions");
    (*conventions)["cycle_origin"] = "epoch0:first_instruction; later_epochs:matching_wait_acceptance";
    (*conventions)["timeline"] = "minimum_issue_cycle_is_lower_bound; compare_offsets_only_within_epoch";
    (*conventions)["dynamic_completion"] = "memory_lifetime_ends_at_matching_wait; null_age_is_unknown_not_zero";
    (*conventions)["access_elements"] = "xreg/ereg:register; mreg:register*32+row; vmem:32_byte_line; dram:conservative_anywhere";
  }
  return result;
}
