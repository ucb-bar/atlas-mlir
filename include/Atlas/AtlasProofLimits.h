#ifndef ATLAS_PROOF_LIMITS_H
#define ATLAS_PROOF_LIMITS_H

#include "Atlas/AtlasProofWork.h"
#include <algorithm>
#include <set>
#include <vector>

namespace mlir::atlas {
// Work allowance shared by bounded scalar proof traversals. Exhaustion refuses.
inline constexpr unsigned kAtlasScalarProofWorkLimit = 1000000;

// Memoizing an initialization state must not turn a nonterminating scalar
// cycle into a successful proof. Joins may revisit a state; an edge back to an
// ancestor may not. Charge reachability work to the same default allowance.
class AtlasProofStateGraph {
public:
  enum class EdgeResult { Added, Cycle, Exhausted };
  unsigned addNode() {
    edges.emplace_back();
    return edges.size() - 1;
  }
  EdgeResult addEdge(unsigned from, unsigned to, unsigned &remaining) {
    if (std::find(edges[from].begin(), edges[from].end(), to) != edges[from].end())
      return EdgeResult::Added;
    std::vector<unsigned> pending{to};
    std::set<unsigned> seen;
    while (!pending.empty()) {
      unsigned node = pending.back(); pending.pop_back();
      if (!seen.insert(node).second) continue;
      if (!remaining) return EdgeResult::Exhausted;
      --remaining;
      atlasProofTransfer(AtlasProofKind::InitializationReachability);
      if (node == from) return EdgeResult::Cycle;
      pending.insert(pending.end(), edges[node].begin(), edges[node].end());
    }
    edges[from].push_back(to);
    return EdgeResult::Added;
  }
private:
  std::vector<std::vector<unsigned>> edges;
};
}

#endif
