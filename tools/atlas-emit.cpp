#include "Atlas/AtlasDialect.h"
#include "Atlas/AtlasEncoding.h"
#include "Atlas/AtlasOps.h"
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
  bool mapJson = argc == 3 && llvm::StringRef(argv[1]) == "--map-json";
  if ((argc != 2 && !mapJson) || (argc == 2 && llvm::StringRef(argv[1]) == "--map-json")) {
    llvm::errs() << "usage: atlas-emit [--map-json] <flat-mlir-module>\n";
    return 2;
  }
  DialectRegistry registry;
  registry.insert<AtlasDialect>();
  MLIRContext context(registry);
  auto module = parseSourceFile<ModuleOp>(argv[mapJson ? 2 : 1], &context);
  if (!module || failed(verify(*module))) return 1;

  llvm::SmallVector<uint32_t> words;
  if (failed(collectAtlasWords(*module, words, mapJson))) return 1;
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
