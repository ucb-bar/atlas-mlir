#include "Atlas/AtlasRTLExport.h"
#include "Atlas/AtlasEncoding.h"
#include "Atlas/AtlasGeneratedArtifact.h"
#include "Atlas/AtlasRTLSelection.h"
#include "Atlas/AtlasStream.h"
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

template <typename Range> Array integers(const Range &values) {
  Array result;
  for (auto value : values) result.push_back(static_cast<int64_t>(value));
  return result;
}

Object footprintJSON(const Instr &in, const Footprint &footprint, bool dynamic) {
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
    bool launch = in.op->opClass == OpClass::DmaLoad || in.op->opClass == OpClass::DmaStore;
    result["dma_async"] = launch;
    if (launch) {
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
  auto selected = getSelectedRTLEvidence(module);
  if (failed(selected))
    return failure();
  if (!*selected)
    return module.emitError("RTL timing export requires selected evidence");
  if (auto state = module->getAttrOfType<StringAttr>(kAtlasTimingState);
      state && state.getValue() != "timed")
    return module.emitError("RTL timing export requires a timed stream; schedule or insert delays first");
  // The executable boundary accepts the stream; the trace then comes from the
  // same verifier under the selected provider.
  SmallVector<uint32_t> words;
  if (failed(verifyAtlasExecutable(module, /*llvmBlock=*/false, words)))
    return failure();
  FailureOr<TimingProvider> provider = selectAtlasTimingProvider(module);
  if (failed(provider))
    return failure();
  if (provider->id != kAtlasRTLTimingProviderId)
    return module.emitError("RTL timing export requires the ") << kAtlasRTLTimingProviderId << " provider, not " << provider->id;
  FailureOr<AtlasStream> stream = readAtlasStream(module, AtlasStreamReadMode::Verification);
  AtlasTimedTrace trace;
  if (failed(stream) || failed(verifyAtlasTimedStream(*stream, *provider, &trace)))
    return failure();

  Array instructions, blocks;
  std::vector<uint8_t> encoded;
  for (uint32_t word : words) {
    for (unsigned shift = 0; shift < 32; shift += 8)
      encoded.push_back(static_cast<uint8_t>(word >> shift));
  }
  auto hash = llvm::SHA256::hash(encoded);
  const RTLEvidence &evidence = **selected;
  const bool dynamic = evidence.dmaWait();
  for (size_t b = 0; b < stream->starts.size(); ++b) {
    // Each matching wait starts a new epoch whose offsets are comparable only
    // within it; every block starts idle at its own cycle origin.
    int epoch = 0, epochOrigin = 0;
    for (size_t i = stream->starts[b]; i < stream->blockEnd(b); ++i) {
      const Instr &in = stream->instrs[i];
      int cycle = trace.cycles[i];
      if (in.op->opClass == OpClass::DmaWait) {
        ++epoch;
        epochOrigin = cycle;
      }
      Object entry{
          {"word_index", static_cast<int64_t>(i)},
          {"word_u32", static_cast<int64_t>(words[i])},
          {"mnemonic", in.op->name},
          {"operands", Object{{"rd", in.rd}, {"rs1", in.rs1}, {"rs2", in.rs2},
              {"immediate", static_cast<int64_t>(in.imm)}, {"release", in.release}}},
          {"block", static_cast<int64_t>(b)},
          {"logical_issue_cycle", cycle},
          {"event_kind", in.op->opClass == OpClass::Halt ?
              "terminal_acceptance" : "instruction_issue"},
          {"footprint", trace.contexts[b].empty() ? llvm::json::Value(footprintJSON(in, trace.footprints[i], dynamic)) : nullptr}};
      if (dynamic) {
        entry.erase("logical_issue_cycle");
        entry["minimum_issue_cycle"] = cycle;
        entry["issue_epoch"] = epoch;
        entry["epoch_offset"] = cycle - epochOrigin;
        if (in.op->opClass == OpClass::DmaWait) {
          entry["event_kind"] = "matching_wait_acceptance";
          entry["epoch_origin"] = "matching_wait_acceptance";
          entry["channel"] = in.op->channel;
        }
      }
      instructions.push_back(std::move(entry));
    }
    Object blockJSON{{"index", static_cast<int64_t>(b)},
        {"first_word", static_cast<int64_t>(stream->starts[b])},
        {"word_count", static_cast<int64_t>(stream->blockEnd(b) - stream->starts[b])},
        {"exit", stream->endsInHalt(b) ? "halt" :
                 stream->endsInBranch(b) ? "branch_with_delay_slot" : "fall_through"},
        {"successors", integers(stream->succs[b])},
        {"successor_issue_offset", nullptr},
        {"entry_state", "idle_engines_no_pending_dma"}};
    if (trace.successorIssue[b] >= 0)
      blockJSON["successor_issue_offset"] = trace.successorIssue[b];
    // A block whose footprints differ between entry contexts lists each group:
    // its entry states (known registers) and its instructions' footprints.
    if (!trace.contexts[b].empty()) {
      Array contexts;
      for (const AtlasContextGroup &context : trace.contexts[b]) {
        Array entries, footprints;
        for (const RegValues &regs : context.entries) {
          Object known;
          for (int r = 1; r < 32; ++r)
            if (regs[r])
              known["x" + std::to_string(r)] = static_cast<int64_t>(*regs[r]);
          entries.push_back(std::move(known));
        }
        for (size_t k = 0; k < context.footprints.size(); ++k)
          footprints.push_back(footprintJSON(stream->instrs[stream->starts[b] + k], context.footprints[k], dynamic));
        contexts.push_back(Object{{"entries", std::move(entries)}, {"footprints", std::move(footprints)}});
      }
      blockJSON["entry_contexts"] = std::move(contexts);
    }
    blocks.push_back(std::move(blockJSON));
  }
  Object result{
      {"schema", dynamic ? "atlas.resolved_rtl_timing.v1" : "atlas.resolved_rtl_timing.v0"},
      {"target_config", "EE290SimConfig"},
      {"qualification", "conditional"},
      {"scheduling_qualified", false},
      {"provider", Object{{"id", provider->id},
                          {"version", 1},
                          {"dma_policy", dynamic ? "wait" : "none"}}},
      {"evidence", Object{{"op_timing_schema", "merlin.op_timing.v1"},
          {"op_timing_sha256", evidence.factsSha256()},
          {"hardware_ir_sha256", evidence.hardwareIRSha256()}}},
      {"program", Object{{"word_count", static_cast<int64_t>(words.size())},
          {"words", integers(words)},
          {"words_sha256", llvm::toHex(llvm::ArrayRef<uint8_t>(hash), true)},
          {"hash_encoding", "little_endian_u32"}}},
      {"conventions", Object{
          {"cycle_origin", "per_block_instance:first_instruction_logical_issue"},
          {"ages", "relative_to_instruction_issue; holds_have_inclusive_endpoints"},
          {"access_elements", "xreg/ereg:register; mreg:register*32+row; vmem:32_byte_line"},
          {"timeline", "conditional_model_timeline_not_measured_elapsed_cycles"},
          {"terminal", "acceptance_not_retirement; ECALL_suppresses_scalar_fire"},
          {"entry_contexts", "a block listing entry_contexts has null instruction footprints; an instance takes the footprints of each group with an entry whose registers it holds"}}},
      {"applicability", evidence.applicability()},
      {"blocks", std::move(blocks)},
      {"instructions", std::move(instructions)}};
  if (const RTLPairFacts *pairs = evidence.pairFacts()) {
    auto *selectedFacts = result.getObject("evidence");
    (*selectedFacts)["op_pairs_schema"] = "merlin.op_pairs.v1";
    (*selectedFacts)["op_pairs_sha256"] = pairs->sha256;
  }
  if (dynamic) {
    auto *conventions = result.getObject("conventions");
    (*conventions)["cycle_origin"] = "per_block_instance; epoch0:first_instruction_of_block; later_epochs:matching_wait_acceptance";
    (*conventions)["timeline"] = "minimum_issue_cycle_is_lower_bound; compare_offsets_only_within_epoch";
    (*conventions)["dynamic_completion"] = "memory_lifetime_ends_at_matching_wait; null_age_is_unknown_not_zero";
    (*conventions)["access_elements"] = "xreg/ereg:register; mreg:register*32+row; vmem:32_byte_line; dram:conservative_anywhere";
  }
  return result;
}
