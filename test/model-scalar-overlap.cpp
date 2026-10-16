#include "Atlas/AtlasModelPairs.h"
#include "Atlas/AtlasDialect.h"
#include "Atlas/AtlasOps.h"
#include "Atlas/AtlasStream.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::atlas;
using namespace mlir::atlas::timing;
namespace {
using Object = llvm::json::Object;
using Array = llvm::json::Array;
int failures = 0;
void check(bool good, const char *message) {
  if (!good) { ++failures; llvm::errs() << "FAIL: " << message << '\n'; }
}
std::string encode(Object value) {
  std::string bytes;
  llvm::raw_string_ostream stream(bytes);
  stream << llvm::json::Value(std::move(value));
  return bytes;
}
std::string hash(llvm::StringRef bytes) {
  auto digest = llvm::SHA256::hash(llvm::ArrayRef<uint8_t>(reinterpret_cast<const uint8_t *>(bytes.data()), bytes.size()));
  return llvm::toHex(llvm::ArrayRef<uint8_t>(digest), true);
}
Object group(int age) { return Object{{"first_age", age}, {"last_age", age}, {"count", 1}, {"step", 1}}; }
Object scalarBlock(const char *name, bool store) {
  return Object{{"name", name}, {"engine", "scalar_lsu"}, {"resolved", true},
      {"first_free_age", store ? 2 : 4}, {"next_issue_age", store ? 2 : 4},
      {"events", store ? Object{{"vmem_write", group(0)}} : Object{{"vmem_read", group(0)}, {"load_write", group(3)}}},
      {"assumptions", Object{{"scratchpad_read_latency", 1}}},
      {"observation", Object{{"accepted", true}, {"drained", true}}}};
}
Object relation(const char *first = "lw", const char *second = "sw") {
  return Object{{"mreg_banks", "disjoint"}, {"first_mnemonic", first}, {"second_mnemonic", second},
      {"scalar_registers", "independent"}, {"scalar_address_alignment", "word"},
      {"vmem_bank", "same"}, {"vmem_window", "disjoint"}};
}
Object source(const std::string &pointer) {
  return Object{{"json_pointer", pointer}, {"sha256", std::string(64, 'a')}, {"proof_sha256", std::string(64, 'b')}};
}
Object pair(const char *first, const char *second, Object rel, int minimum, int boundary, int index) {
  return Object{{"first", first}, {"second", second}, {"relation", std::move(rel)},
      {"swept", Array{1, boundary}}, {"min_gap", minimum}, {"first_live_window_end", boundary},
      {"negative", minimum == 1 ? Object{{"gap", 0}, {"result", "inexpressible"}} :
          Object{{"gap", minimum - 1}, {"result", "conflict"}, {"error", Object{{"message", "synthetic load path busy"}}}}},
      {"source_record", source("/pairs/" + std::to_string(index))}, {"verilator", Object{{"confirmed", true}}}};
}
Object projection(const ModelTimingFacts &facts, const Object &model, Array pairs, Array chains = {}) {
  return Object{{"schema", "atlas.model_pairs.v1"}, {"age_convention", "issue_edge_zero"},
      {"model", Object(model)}, {"pairs", std::move(pairs)}, {"chains", std::move(chains)},
      {"projection", Object{{"version", 2}, {"policy", "confirmed-exact-scalar-relations-v2"},
          {"master_sha256", std::string(64, 'c')}, {"master_path", "synthetic-master.json"},
          {"validation_sha256", std::string(64, 'd')}, {"validation_path", "synthetic-validation.json"},
          {"per_op_sha256", facts.factsSha256()}, {"projector_sha256", std::string(64, 'e')},
          {"counts", Object{}}, {"excluded_reasons", Object{}}}}};
}
std::optional<ModelOverlapFacts> parse(Object object, const ModelTimingFacts &facts) {
  std::string bytes = encode(std::move(object));
  auto parsed = parseModelOverlapFacts(bytes, hash(bytes), facts);
  if (!parsed) { llvm::errs() << llvm::toString(parsed.takeError()) << '\n'; return {}; }
  return std::move(*parsed);
}
// Exercise the public replay proof factory, with a real BranchOp controlling
// exactly 32 iterations. Other Operation pointers are unused by this replay.
std::vector<ModelOverlapCommand> replay(MLIRContext &context, std::vector<Instr> memory,
    RegValues entry, std::vector<Instr> prefix = {}, std::vector<Instr> suffix = {}) {
  Builder builder(&context);
  OperationState state(builder.getUnknownLoc(), "atlas.branch");
  state.addAttribute("kind", builder.getStringAttr("blt"));
  state.addAttribute("lhs", builder.getI32IntegerAttr(13));
  state.addAttribute("rhs", builder.getI32IntegerAttr(14));
  state.addAttribute("offset_bytes", builder.getI32IntegerAttr(-8));
  Operation *branch = Operation::create(state);
  AtlasStream stream;
  stream.instrs = prefix;
  stream.instrs.insert(stream.instrs.end(), memory.begin(), memory.end());
  stream.instrs.insert(stream.instrs.end(), suffix.begin(), suffix.end());
  stream.instrs.push_back({findOp("addi"), 13, 13, 0, 1});
  size_t branchIndex = stream.instrs.size();
  stream.instrs.push_back({findOp("blt"), 0, 13, 14, -8});
  stream.instrs.push_back({findOp("addi"), 0, 0, 0, 0});
  stream.ops.resize(stream.instrs.size(), nullptr);
  stream.ops[branchIndex] = branch;
  entry[13] = 0; entry[14] = 32;
  auto contexts = proveAtlasPackLoopTimingContexts(stream, {0, branchIndex, 0}, entry);
  std::vector<ModelOverlapCommand> result;
  for (size_t i = 0; i < memory.size(); ++i) {
    const Instr &in = memory[i];
    const char *block = in.op->opClass == OpClass::ScalarStore ? "scalar_lsu.store" : "scalar_lsu.load";
    result.push_back({block, Engine::Lsu, {}, {}, {}, int(i), in.op->opClass == OpClass::ScalarStore ? 2 : 4,
                      in, contexts[prefix.size() + i]});
  }
  branch->destroy();
  return result;
}
} // namespace

int testScalarOverlap(Object perOp) {
  failures = 0;
  perOp.getArray("op_timing")->push_back(scalarBlock("scalar_lsu.load", false));
  perOp.getArray("op_timing")->push_back(scalarBlock("scalar_lsu.store", true));
  Object model(*perOp.getObject("model"));
  std::string bytes = encode(std::move(perOp));
  auto facts = parseModelTimingFacts(bytes, hash(bytes));
  check(bool(facts), "scalar relation fixture loads");
  if (!facts) { llvm::errs() << llvm::toString(facts.takeError()) << '\n'; return failures; }
  DialectRegistry registry; registry.insert<AtlasDialect>();
  MLIRContext context(registry);
  context.loadDialect<AtlasDialect>();
  RegValues entry = unknownRegs(); entry[10] = 0x40000; entry[11] = 0x40400; entry[12] = 0x40800;
  Instr load{findOp("lw"), 16, 10, 0, 0}, store{findOp("sw"), 0, 11, 17, 0};
  auto commands = replay(context, {load, store}, entry);
  check(commands[0].context && commands[1].context, "scalar pair has complete replay contexts");
  auto overlap = parse(projection(*facts, model, Array{pair("scalar_lsu.load", "scalar_lsu.store", relation(), 1, 4, 0)}), *facts);
  if (!overlap) return failures + 1;
  check(overlap->admits(commands, false), "independent LW to SW admits gap one");
  commands[1].issue = 4;
  check(overlap->admits(commands, false), "bounded pair includes its high endpoint");
  commands[1].issue = 5;
  check(!overlap->admits(commands, false), "bounded pair cannot extrapolate beyond its high endpoint");
  commands[1].issue = 0;
  check(!overlap->admits(commands, false), "single-issue pair excludes gap zero");
  commands[1].issue = 1;
  auto changed = commands;
  changed[0].context.reset();
  check(!overlap->admits(changed, false), "unknown address context cannot admit a scalar fact");
  changed = commands; changed[0].scalarInstruction->imm = 4;
  check(!overlap->admits(changed, false), "changed instruction invalidates replay binding");
  auto otherRegion = replay(context, {load, store}, entry);
  changed = commands; changed[1].context = otherRegion[1].context;
  check(!overlap->admits(changed, false), "two replay snapshots cannot be mixed");
  changed = commands; changed[0].block = "scalar_lsu.store";
  check(!overlap->admits(changed, false), "semantic operation identity cannot be relabeled");
  Instr alias = store; alias.rs2 = 16;
  check(!overlap->admits(replay(context, {load, alias}, entry), false), "load-result to store-data RAW cannot use independent fact");
  alias = store; alias.rs1 = 10;
  check(!overlap->admits(replay(context, {load, alias}, entry), false), "shared address base cannot use independent fact");
  auto otherBank = entry; otherBank[11] = 0x80000;
  check(!overlap->admits(replay(context, {load, store}, otherBank), false), "a different physical VMEM bank fails exact relation");
  auto sameWord = entry; sameWord[11] = 0x40000;
  check(!overlap->admits(replay(context, {load, store}, sameWord), false), "different base registers do not prove disjoint addresses");
  auto unaligned = entry; unaligned[10] = 0x40001;
  check(!overlap->admits(replay(context, {load, store}, unaligned), false), "unaligned effective byte address cannot use word fact");
  auto lateBank = entry; lateBank[10] = 0x7ff00; lateBank[11] = 0x70000;
  auto crossed = replay(context, {load, store}, lateBank, {}, {{findOp("addi"), 10, 10, 0, 16}});
  check(crossed[0].context && !overlap->admits(crossed, false), "later iteration crossing a bank invalidates universal same-bank relation");

  // Every published directional role mode gets a positive witness and an
  // independent-register mutation. Prefix restoration makes address-base WAR
  // decidable again on the next iteration without relying on load contents.
  for (const char *mode : {"same_load_destination", "store_data_equals_load_destination",
                          "first_address_base_equals_second_load_destination", "shared_base_read_only"}) {
    bool storeFirst = std::string(mode) == "store_data_equals_load_destination";
    Instr a = storeFirst ? Instr{findOp("sw"), 0, 10, 16, 0} : load;
    Instr b{findOp("lw"), 17, 11, 0, 0};
    Object rel = relation(storeFirst ? "sw" : "lw", "lw");
    rel["scalar_registers"] = mode; rel["scalar_other_registers"] = "independent";
    if (std::string(mode) == "same_load_destination") b.rd = 16;
    if (storeFirst) b.rd = 16;
    if (std::string(mode) == "first_address_base_equals_second_load_destination") b.rd = 10;
    if (std::string(mode) == "shared_base_read_only") { b.rs1 = 10; rel["vmem_window"] = "same"; }
    else rel["scalar_destination_file"] = "xrf";
    auto qualified = parse(projection(*facts, model, Array{pair(storeFirst ? "scalar_lsu.store" : "scalar_lsu.load",
        "scalar_lsu.load", std::move(rel), 1, storeFirst ? 2 : 4, 0)}), *facts);
    auto witness = replay(context, {a, b}, entry, {{findOp("lui"), 10, 0, 0, 64}});
    check(qualified && qualified->admits(witness, false), "published directional scalar register mode matches its complete roles");
    if (std::string(mode) == "shared_base_read_only") b.rs1 = 11;
    else b.rd = 0;
    check(qualified && !qualified->admits(replay(context, {a, b}, entry, {{findOp("lui"), 10, 0, 0, 64}}), false),
          "x0 or changed destination cannot satisfy directional register witness");
  }

  auto loads = replay(context, {load, Instr{findOp("lw"), 17, 11, 0, 0}}, entry);
  auto loadPair = parse(projection(*facts, model, Array{pair("scalar_lsu.load", "scalar_lsu.load", relation("lw", "lw"), 3, 4, 0)}), *facts);
  loads[1].issue = 2;
  check(loadPair && !loadPair->admits(loads, false), "load pair rejects one-short gap two");
  loads[1].issue = 3;
  check(loadPair && loadPair->admits(loads, false), "load pair admits exact confirmed gap three");
  auto triple = replay(context, {load, store, Instr{findOp("sw"), 0, 12, 18, 0}}, entry);
  Array matrix;
  for (int i = 0; i < 3; ++i)
    for (int j = i + 1; j < 3; ++j)
      matrix.push_back(Object{{"first_index", i}, {"second_index", j}, {"relation", relation(i ? "sw" : "lw", "sw")}});
  Object sequence{{"operations", Array{"scalar_lsu.load", "scalar_lsu.store", "scalar_lsu.store"}},
      {"gaps", Array{1, 1}}, {"relation", relation()}, {"all_pair_relations", std::move(matrix)},
      {"relation_scope", "adjacent"}, {"result", "accepted"}, {"in_flight_count", 2},
      {"witness_only", true}, {"no_stable_suffix_claim", true}, {"source_record", source("/chains/0")},
      {"verilator", Object{{"confirmed", true}}}};
  auto chainFacts = [&]() {
    return projection(*facts, model,
        Array{pair("scalar_lsu.load", "scalar_lsu.store", relation(), 1, 4, 0),
              pair("scalar_lsu.store", "scalar_lsu.store", relation("sw", "sw"), 1, 2, 1)},
        Array{Object(sequence)});
  };
  auto chain = parse(chainFacts(), *facts);
  check(chain && !chain->admits(triple, false), "pair facts alone never imply a three-command chain");
  check(chain && chain->admits(triple, true), "whole scalar chain matches exact gap vector and complete matrix");
  auto altered = triple; altered[2].issue = 3;
  check(chain && !chain->admits(altered, true), "a chain witness grants no independent gap suffix");
  auto badMatrix = chainFacts();
  auto &nonadjacent = *(*badMatrix.getArray("chains"))[0].getAsObject()->getArray("all_pair_relations");
  (*nonadjacent[1].getAsObject()->getObject("relation"))["vmem_bank"] = "other";
  auto wrongRelation = parse(std::move(badMatrix), *facts);
  check(wrongRelation && !wrongRelation->admits(triple, true), "a nonadjacent chain relation cannot be skipped");
  auto missingPair = chainFacts(); missingPair.getArray("pairs")->pop_back();
  auto uncovered = parse(std::move(missingPair), *facts);
  check(uncovered && !uncovered->admits(triple, true), "the complete chain still needs every live pair fact");
  auto fourth = replay(context, {load, store, Instr{findOp("sw"), 0, 12, 18, 0}, Instr{findOp("sw"), 0, 12, 19, 4}}, entry);
  check(chain && !chain->admits(fourth, true), "a witnessed triple never grants general scalar capacity");
  llvm::outs() << "scalar overlap relation checks, " << failures << " failures\n";
  return failures;
}
