#include "Atlas/AtlasDialect.h"
#include "Atlas/AtlasEncoding.h"
#include "Atlas/AtlasOps.h"
#include "Atlas/AtlasSelectedTarget.h"
#include "Atlas/AtlasStream.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::atlas;

int main(int argc, char **argv) {
  bool mapJson = false, programJson = false, allowUntimed = false;
  const char *input = nullptr;
  bool invalid = false;
  for (int i = 1; i < argc; ++i) {
    StringRef arg(argv[i]);
    if (arg == "--map-json") mapJson = true;
    else if (arg == "--program-json") programJson = true;
    else if (arg == "--allow-untimed") allowUntimed = true;
    else if (arg.starts_with("--") || input) invalid = true;
    else input = argv[i];
  }
  if (invalid || !input || (mapJson && programJson)) {
    llvm::errs() << "usage: atlas-emit [--allow-untimed] [--map-json|--program-json] <flat-mlir-module>\n";
    return 2;
  }
  DialectRegistry registry;
  registry.insert<AtlasDialect>();
  MLIRContext context(registry);
  auto module = parseSourceFile<ModuleOp>(input, &context);
  if (!module || failed(verify(*module))) return 1;

  llvm::SmallVector<uint32_t> words;
  if (failed(verifyAtlasTimingState(*module, !allowUntimed))) return 1;
  if (failed(verifyAtlasArtifact(*module, mapJson || programJson, words))) return 1;
  if (programJson) {
    // This is the physical-stream input boundary for an external functional
    // model. The encoded words remain authoritative; these typed fields and
    // control edges are derived from the same verified machine operations.
    llvm::json::Array instructions;
    size_t index = 0;
    Operation *previous = nullptr;
    for (Operation &op : module->getBody()->getOperations()) {
      if (isa<StartOp>(op)) continue;
      llvm::json::Object fields;
      for (NamedAttribute named : op.getAttrs()) {
        Attribute value = named.getValue();
        std::string key = named.getName().str();
        if (auto boolean = dyn_cast<BoolAttr>(value))
          fields[key] = boolean.getValue();
        else if (auto integer = dyn_cast<IntegerAttr>(value))
          fields[key] = integer.getValue().getSExtValue();
        else if (auto string = dyn_cast<StringAttr>(value))
          fields[key] = string.getValue().str();
        else {
          op.emitError("cannot export unsupported physical attribute '") << key << "'";
          return 1;
        }
      }
      llvm::json::Object control{{"kind", "sequential"}};
      if (auto branch = dyn_cast<BranchOp>(op)) {
        int64_t offset = branch.getOffsetBytesAttr().getValue().getSExtValue();
        control["kind"] = "conditional_direct";
        control["target_word_index"] = static_cast<int64_t>(index) + offset / 2;
        control["fallthrough_word_index"] = static_cast<int64_t>(index) + 2;
        control["delay_slot_word_index"] = static_cast<int64_t>(index) + 1;
      } else if (auto jump = dyn_cast<JumpOp>(op)) {
        control["delay_slot_word_index"] = static_cast<int64_t>(index) + 1;
        if (jump.getKind() == "jal") {
          int64_t offset = jump.getOffsetAttr().getValue().getSExtValue();
          control["kind"] = "jump_direct";
          control["target_word_index"] = static_cast<int64_t>(index) + offset / 2;
        } else {
          control["kind"] = "jump_register";
          control["base_register"] = static_cast<int64_t>(jump.getBase());
          control["offset_words"] = jump.getOffsetAttr().getValue().getSExtValue();
        }
      } else if (isa<TrapOp>(op)) {
        control["kind"] = "trap";
      }
      std::string encoded;
      llvm::raw_string_ostream encodedStream(encoded);
      encodedStream << llvm::format_hex_no_prefix(words[index], 8);
      encodedStream.flush();
      llvm::json::Object entry{
          {"word_index", static_cast<int64_t>(index)},
          {"word_hex", encoded},
          {"word_u32", static_cast<int64_t>(words[index])},
          {"operation", op.getName().getStringRef().str()},
          {"fields", std::move(fields)},
          {"control", std::move(control)},
      };
      if (previous && isa<BranchOp, JumpOp>(previous))
        entry["delay_slot_for_word_index"] = static_cast<int64_t>(index - 1);
      if (auto delay = dyn_cast<DelayOp>(op))
        entry["explicit_delay_cycles"] = static_cast<int64_t>(delay.getCycles());
      instructions.push_back(std::move(entry));
      previous = &op;
      ++index;
    }
    llvm::json::Object root{
        {"schema", "atlas.physical_program.v1"},
        {"selected_rtl_revision", kSelectedRTLRevision},
        {"pc_unit", "instruction_word"},
        {"word_endianness", "little"},
        {"entry_word_index", 0},
        {"word_count", static_cast<int64_t>(words.size())},
        {"timing_scope", "explicit_delay_only; other availability unqualified"},
        {"instructions", std::move(instructions)},
    };
    if (auto state = (*module)->getAttrOfType<StringAttr>("atlas.timing_state"))
      root["timing_state"] = state.getValue().str();
    if (auto provider = (*module)->getAttrOfType<StringAttr>("atlas.timing_provider"))
      root["timing_provider"] = provider.getValue().str();
    llvm::outs() << llvm::formatv("{0:2}\n", llvm::json::Value(std::move(root)));
    return 0;
  }
  if (mapJson) {
    llvm::json::Array operations;
    size_t index = 0;
    Operation *previous = nullptr;
    for (Operation &op : module->getBody()->getOperations()) {
      if (isa<StartOp>(op)) continue;
      llvm::json::Object attributes;
      for (NamedAttribute named : op.getAttrs()) {
        std::string rendered;
        llvm::raw_string_ostream stream(rendered);
        named.getValue().print(stream);
        stream.flush();
        attributes[named.getName().str()] = rendered;
      }
      std::string encoded;
      llvm::raw_string_ostream encodedStream(encoded);
      encodedStream << llvm::format_hex_no_prefix(words[index], 8);
      encodedStream.flush();
      llvm::json::Object entry{
          {"word_index", static_cast<int64_t>(index)},
          {"byte_offset", static_cast<int64_t>(index * 4)},
          {"word_hex", encoded},
          {"operation", op.getName().getStringRef().str()},
          {"attributes", std::move(attributes)},
          {"effect_scope", "conservative_physical_state_read_write"},
          {"availability", "unknown"},
      };
      if (auto delay = dyn_cast<DelayOp>(op)) {
        entry["explicit_stall_cycles"] = static_cast<int64_t>(delay.getCycles());
      }
      if (previous && isa<BranchOp, JumpOp>(previous)) {
        entry["delay_slot_for_word_index"] = static_cast<int64_t>(index - 1);
      }
      if (auto branch = dyn_cast<BranchOp>(op)) {
        int64_t offset = branch.getOffsetBytesAttr().getValue().getSExtValue();
        entry["direct_target_word_index"] = static_cast<int64_t>(index) + offset / 2;
      } else if (auto jump = dyn_cast<JumpOp>(op); jump && jump.getKind() == "jal") {
        int64_t offset = jump.getOffsetAttr().getValue().getSExtValue();
        entry["direct_target_word_index"] = static_cast<int64_t>(index) + offset / 2;
      }
      operations.push_back(std::move(entry));
      previous = &op;
      ++index;
    }
    llvm::json::Object root{{"schema", "atlas.machine_word_map.v1"},
                            {"word_count", static_cast<int64_t>(words.size())},
                            {"operations", std::move(operations)}};
    llvm::outs() << llvm::formatv("{0:2}\n", llvm::json::Value(std::move(root)));
    return 0;
  }
  for (uint32_t word : words)
    llvm::outs() << llvm::format_hex_no_prefix(word, 8) << '\n';
  return 0;
}
