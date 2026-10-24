#ifndef ATLAS_MODEL_REPLAY_H
#define ATLAS_MODEL_REPLAY_H

#include "Atlas/AtlasSearchPolicy.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include <algorithm>
#include <array>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace mlir::atlas::timing {

// A small interpreter, shared arithmetic comes from the facts bytecode.
// Engine/dependency/queue serialization in that bytecode is projection policy.
struct ModelReplayProgram {
  enum Register { Frontend, UnitFree, DependencyEnd, TransferEnd, CapacityEnd, Queued, Drains,
                  FrontendCost, Latency, Busy, Waiting, Makespan, Cycle, Work, Finished, RegisterCount };
  enum Opcode { Literal, Read, Add, Sub, Max, Select };
  struct Token { Opcode opcode; int value; };
  static const std::map<std::string, int> &registers() {
    static const std::map<std::string, int> values = {{"frontend", Frontend}, {"unit_free", UnitFree},
      {"dependency_end", DependencyEnd}, {"transfer_end", TransferEnd}, {"capacity_end", CapacityEnd},
      {"queued", Queued}, {"drains", Drains}, {"frontend_cost", FrontendCost}, {"latency", Latency},
      {"busy", Busy}, {"waiting", Waiting}, {"makespan", Makespan}, {"cycle", Cycle}, {"work", Work}, {"finished", Finished}};
    return values;
  }
  struct Assignment {
    int target;
    std::vector<Token> expression;
  };
  llvm::json::Object document;
  std::vector<Assignment> start, place;
  using State = std::array<int, RegisterCount>;

  static llvm::Expected<ModelReplayProgram> parse(const llvm::json::Object &object) {
    auto error = [](const char *why) -> llvm::Error {
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                    "unsupported projected-Core replay: %s", why);
    };
    if (object.size() != 5 || object.getString("schema") != "atlas.projected_core_replay.v1" ||
        !object.getObject("source") || !object.getObject("projection_policy"))
      return error("schema/source/policy");
    const auto &source = *object.getObject("source");
    if (source.size() != 7 || source.getString("schema") != "atlas.core_edges.v1" ||
        source.getInteger("clock_step") != 1 || source.getString("dma_wait_observes") != "previous_edge_flags" ||
        source.getBoolean("implicit_exu_stalls") != false ||
        source.getString("source_contract") != "atlas.model_frontend.v1" ||
        !source.getArray("phases") || *source.getArray("phases") !=
          llvm::json::Array{"s1_previous_state", "execution_units", "fetch"} ||
        !source.getArray("checked_methods") || *source.getArray("checked_methods") !=
          llvm::json::Array{"Core.tick", "Core._tick_s1"})
      return error("source edge contract");
    auto templateValue = llvm::json::parse(search_policy::kReplayProjectionJson);
    if (!templateValue) return templateValue.takeError();
    auto *policy = templateValue->getAsObject();
    if (!policy) return error("compiled policy template");
    auto *projection = policy->getObject("projection_policy");
    if (!projection) return error("compiled projection policy object");
    (*projection)["template_sha256"] = search_policy::kReplayProjectionSHA256;
    for (const char *stage : {"start", "place"}) {
      auto *assignments = policy->getArray(stage);
      if (!assignments || assignments->empty() || assignments->size() > 16)
        return error("compiled projection stage");
      for (auto &item : *assignments) {
        auto *assignment = item.getAsArray();
        if (!assignment || assignment->size() != 2 || !(*assignment)[0].getAsString())
          return error("compiled projection assignment");
        auto *expression = (*assignment)[1].getAsArray();
        if (!expression || expression->empty() || expression->size() > 64)
          return error("compiled projection expression");
        for (auto &token : *expression)
          if (token.getAsString() == "$clock_step") token = *source.getInteger("clock_step");
      }
    }
    for (const char *field : {"projection_policy", "start", "place"})
      if (!object.get(field) || *object.get(field) != *policy->get(field))
        return error("transition differs from declared projection policy");
    ModelReplayProgram result;
    result.document = object;
    std::set<std::string> available = {"frontend", "unit_free", "dependency_end", "transfer_end", "capacity_end",
        "queued", "drains", "frontend_cost", "latency", "busy", "waiting", "makespan"};
    const std::set<std::string> targets = {"cycle", "work", "finished", "unit_free", "waiting", "frontend", "makespan"};
    for (const char *stage : {"start", "place"}) {
      const auto *assignments = object.getArray(stage);
      if (!assignments || assignments->empty() || assignments->size() > 16)
        return error("transition stage");
      auto &output = std::string(stage) == "start" ? result.start : result.place;
      for (const auto &item : *assignments) {
        const auto *assignment = item.getAsArray();
        if (!assignment || assignment->size() != 2 || !(*assignment)[0].getAsString() ||
            !targets.count((*assignment)[0].getAsString()->str()))
          return error("assignment target");
        const auto *expression = (*assignment)[1].getAsArray();
        if (!expression || expression->empty() || expression->size() > 64)
          return error("expression");
        Assignment parsed{registers().at((*assignment)[0].getAsString()->str()), {}};
        int depth = 0;
        for (const auto &token : *expression) {
          if (auto number = token.getAsInteger()) {
            if (*number < 0 || *number >= 65536) return error("literal");
            parsed.expression.push_back({Literal, int(*number)});
            ++depth;
          } else if (auto name = token.getAsString()) {
            int arity = *name == "select" ? 3 :
                (*name == "add" || *name == "sub" || *name == "max") ? 2 : 0;
            if (arity) {
              if (depth < arity) return error("stack underflow");
              depth -= arity - 1;
            } else if (available.count(name->str())) {
              ++depth;
            } else return error("unknown expression token");
            parsed.expression.push_back({*name == "add" ? Add : *name == "sub" ? Sub : *name == "max" ? Max :
                *name == "select" ? Select : Read, arity ? 0 : registers().at(name->str())});
          } else return error("expression token type");
        }
        if (depth != 1) return error("expression stack");
        available.insert((*assignment)[0].getAsString()->str());
        output.push_back(std::move(parsed));
      }
    }
    for (const char *required : {"cycle", "work", "finished"})
      if (!available.count(required)) return error("missing output");
    return result;
  }

  static void evaluate(const std::vector<Assignment> &assignments, State &state) {
    for (const auto &assignment : assignments) {
      std::array<int, 64> stack;
      unsigned size = 0;
      auto pop = [&]() { return stack[--size]; };
      for (const auto &token : assignment.expression) {
        switch (token.opcode) {
        case Literal: stack[size++] = token.value; break;
        case Read: stack[size++] = state[token.value]; break;
        case Select: {
          int no = pop(), yes = pop(), condition = pop();
          stack[size++] = condition ? yes : no;
          break;
        }
        default: {
          int right = pop(), left = pop();
          stack[size++] = token.opcode == Add ? left + right : token.opcode == Sub ? left - right : std::max(left, right);
          break;
        }
        }
      }
      state[assignment.target] = stack[0];
    }
  }
};
} // namespace mlir::atlas::timing
#endif
