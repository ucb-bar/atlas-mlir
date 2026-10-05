#ifndef ATLAS_TYPES
#define ATLAS_TYPES

include "mlir/IR/AttrTypeBase.td"
include "Atlas/AtlasDialect.td"

def Atlas_StateType : TypeDef<Atlas_Dialect, "State"> {
  let mnemonic = "state";
  let summary = "Ordered Atlas physical machine state";
  let assemblyFormat = "";
}

#endif
