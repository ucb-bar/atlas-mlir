#include "Atlas/AtlasModelPairs.h"
#include "Atlas/AtlasContractAttr.h"
#include "Atlas/AtlasPackLoop.h"
#include "Atlas/AtlasDialect.h"
#include "Atlas/AtlasOps.h"
#include "Atlas/AtlasGeneratedArtifact.h"
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
Object pair(const char *first, const char *second, Object rel, int minimum, int boundary, int index) {
  return Object{{"first", first}, {"second", second}, {"relation", std::move(rel)},
                {"minimum", minimum}, {"high", boundary}};
}
Object projection(const ModelTimingFacts &facts, const Object &model, Array pairs) {
  Object contract{{"capacity", Object{{"sequencer_cycles", 32}, {"mxu0_compute", 3},
                                      {"mxu1_compute", 2}, {"vpu_slots", 2}}}};
  return Object{{"schema", "atlas.model_overlap_derived.v1"}, {"age_convention", "issue_edge_zero"},
      {"per_op_sha256", facts.factsSha256()}, {"model", Object(model)}, {"pairs", std::move(pairs)},
      {"derivation", Object{{"contract", "source-resource-occupancy-v1"}, {"tool_sha256", std::string(64, 'a')},
          {"source_contract_sha256", hash(canonicalJson(llvm::json::Value(Object(contract))))},
          {"source_contract", std::move(contract)}}},
      {"validation", Object{{"passed", true}, {"report_sha256", std::string(64, 'b')},
          {"sample_policy", "source-mechanism-boundaries-v1"}, {"positive_runs", 2}, {"negative_runs", 3},
          {"negative_rejections", 1}, {"numerical_checks", 2}, {"engines", Object{{"scalar_lsu", Object{
              {"positive_runs", 2}, {"negative_runs", 3}, {"negative_rejections", 1}, {"numerical_checks", 2}}}}}}},
      {"composition", Object{{"engines", Object{}}, {"classes", Array{}}}}};
}
std::optional<ModelOverlapFacts> parse(Object object, const ModelTimingFacts &facts) {
  std::string bytes = encode(std::move(object));
  auto parsed = parseModelOverlapFacts(bytes, hash(bytes), facts);
  if (!parsed) { llvm::errs() << llvm::toString(parsed.takeError()) << '\n'; return {}; }
  return std::move(*parsed);
}
// Exercise the public replay proof factory with typed machine operations
// matching the decoded instructions and a branch controlling 32 iterations.
std::vector<ModelOverlapCommand> replay(MLIRContext &context, std::vector<Instr> memory,
    RegValues entry, std::vector<Instr> prefix = {}, std::vector<Instr> suffix = {}) {
  OpBuilder builder(&context);
  auto loc = builder.getUnknownLoc();
  auto stateType = StateType::get(&context);
  OwningOpRef<ModuleOp> module = ModuleOp::create(loc);
  builder.setInsertionPointToEnd(module->getBody());
  Value state = StartOp::create(builder, loc, stateType).getResult();
  AtlasStream stream;
  stream.instrs = prefix;
  stream.instrs.insert(stream.instrs.end(), memory.begin(), memory.end());
  stream.instrs.insert(stream.instrs.end(), suffix.begin(), suffix.end());
  stream.instrs.push_back({findOp("addi"), 13, 13, 0, 1});
  size_t branchIndex = stream.instrs.size();
  stream.instrs.push_back({findOp("blt"), 0, 13, 14, -8});
  stream.instrs.push_back({findOp("addi"), 0, 0, 0, 0});
  for (size_t pc = 0; pc < stream.instrs.size(); ++pc) {
    const Instr &in = stream.instrs[pc];
    Operation *op;
    if (in.op->opClass == OpClass::ScalarLoad)
      op = ScalarLoadOp::create(builder, loc, stateType, state, in.op->name, in.rd, in.rs1, in.imm);
    else if (in.op->opClass == OpClass::ScalarStore)
      op = ScalarStoreOp::create(builder, loc, stateType, state, in.op->name, in.rs2, in.rs1, in.imm);
    else if (in.op->name == "lui")
      op = UpperOp::create(builder, loc, stateType, state, in.op->name, in.rd, in.imm);
    else if (in.op->opClass == OpClass::Branch)
      op = BranchOp::create(builder, loc, stateType, state, in.op->name, in.rs1, in.rs2, in.imm);
    else
      op = ALUImmOp::create(builder, loc, stateType, state, in.op->name, in.rd, in.rs1, in.imm);
    stream.ops.push_back(op);
    state = op->getResult(0);
    stream.ops[pc]->setAttr(kAtlasTagCFGSource, builder.getI32IntegerAttr(0));
    stream.ops[pc]->setAttr(kAtlasTagCFGHelper, builder.getStringAttr("pack"));
  }
  entry[13] = 0; entry[14] = 32;
  auto contexts = proveAtlasPackLoopTimingContexts(stream, {0, branchIndex}, entry);
  std::vector<ModelOverlapCommand> result;
  for (size_t i = 0; i < memory.size(); ++i) {
    const Instr &in = memory[i];
    const char *block = in.op->opClass == OpClass::ScalarStore ? "scalar_lsu.store" : "scalar_lsu.load";
    result.push_back({block, Engine::Lsu, {}, {}, {}, int(i), in.op->opClass == OpClass::ScalarStore ? 2 : 4,
                      in, contexts[prefix.size() + i]});
  }
  return result;
}
} // namespace

int testScalarOverlap(Object perOp) {
  failures = 0;
  perOp.getArray("op_timing")->push_back(scalarBlock("scalar_lsu.load", false));
  perOp.getArray("op_timing")->push_back(scalarBlock("scalar_lsu.store", true));
  Object model(*perOp.getObject("model"));
  std::string bytes = encode(std::move(perOp));
  auto facts = parseModelTimingFactsForTesting(bytes, hash(bytes));
  check(bool(facts), "scalar relation fixture loads");
  if (!facts) { llvm::errs() << llvm::toString(facts.takeError()) << '\n'; return failures; }
  DialectRegistry registry; registry.insert<AtlasDialect>();
  MLIRContext context(registry);
  context.loadDialect<AtlasDialect>();
  RegValues entry = unknownRegs(); entry[10] = 0x40000; entry[11] = 0x40400; entry[12] = 0x40800;
  Instr load{findOp("lw"), 16, 10, 0, 0}, store{findOp("sw"), 0, 11, 17, 0};
  auto commands = replay(context, {load, store}, entry);
  check(commands[0].context && commands[1].context, "scalar pair has complete replay contexts");
  auto overlap = parse(projection(*facts, model,
      Array{pair("scalar_lsu.load", "scalar_lsu.store", relation(), 1, 4, 0)}), *facts);
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
  check(!overlap->admits(replay(context, {load, alias}, entry), false),
        "load-result to store-data RAW cannot use independent fact");
  alias = store; alias.rs1 = 10;
  check(!overlap->admits(replay(context, {load, alias}, entry), false), "shared address base cannot use independent fact");
  auto otherBank = entry; otherBank[11] = 0x80000;
  check(!overlap->admits(replay(context, {load, store}, otherBank), false),
        "a different physical VMEM bank fails exact relation");
  auto sameWord = entry; sameWord[11] = 0x40000;
  check(!overlap->admits(replay(context, {load, store}, sameWord), false),
        "different base registers do not prove disjoint addresses");
  auto unaligned = entry; unaligned[10] = 0x40001;
  check(!overlap->admits(replay(context, {load, store}, unaligned), false),
        "unaligned effective byte address cannot use word fact");
  auto lateBank = entry; lateBank[10] = 0x7ff00; lateBank[11] = 0x70000;
  auto crossed = replay(context, {load, store}, lateBank, {}, {{findOp("addi"), 10, 10, 0, 16}});
  check(crossed[0].context && !overlap->admits(crossed, false),
        "later iteration crossing a bank invalidates universal same-bank relation");

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
    check(qualified && qualified->admits(witness, false),
          "published directional scalar register mode matches its complete roles");
    if (std::string(mode) == "shared_base_read_only") b.rs1 = 11;
    else b.rd = 0;
    check(qualified && !qualified->admits(replay(context, {a, b}, entry, {{findOp("lui"), 10, 0, 0, 64}}), false),
          "x0 or changed destination cannot satisfy directional register witness");
  }

  auto loads = replay(context, {load, Instr{findOp("lw"), 17, 11, 0, 0}}, entry);
  auto loadPair = parse(projection(*facts, model,
      Array{pair("scalar_lsu.load", "scalar_lsu.load", relation("lw", "lw"), 3, 4, 0)}), *facts);
  loads[1].issue = 2;
  check(loadPair && !loadPair->admits(loads, false), "load pair rejects one-short gap two");
  loads[1].issue = 3;
  check(loadPair && loadPair->admits(loads, false), "load pair admits exact confirmed gap three");
  if (loadPair) {
    auto provider = modelTimingProvider(std::make_shared<ModelTimingFacts>(*facts),
                                        std::make_shared<ModelOverlapFacts>(*loadPair));
    auto module = ModuleOp::create(UnknownLoc::get(&context));
    auto plain = machineTimingOf(provider, "lw", module),
         proven = machineTimingOf(provider, "lw", module, atlasPackRelayoutCopyContexts);
    check(succeeded(plain) && succeeded(proven) && plain->interval == 4 && proven->interval == 3,
          "virtual PACK copy loads take the confirmed gap only under the loop proof");
    module.erase();
  }
  auto triple = replay(context, {load, store, Instr{findOp("sw"), 0, 12, 18, 0}}, entry);
  check(!overlap->admits(triple, true), "source scalar pairs never grant general scalar capacity");
  llvm::outs() << "scalar overlap relation checks, " << failures << " failures\n";
  return failures;
}
