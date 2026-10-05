#ifndef ATLAS_TYPES
#define ATLAS_TYPES

include "mlir/IR/AttrTypeBase.td"
include "Atlas/AtlasDialect.td"

def Atlas_StateType : TypeDef<Atlas_Dialect, "State"> {
  let mnemonic = "state";
  let summary = "Ordered Atlas physical machine state";
  let assemblyFormat = "";
}

def Atlas_VirtualStateType : TypeDef<Atlas_Dialect, "VirtualState"> {
  let mnemonic = "virtual_state";
  let summary = "Ordered pre-allocation Atlas boundary state";
  let assemblyFormat = "";
}

def Atlas_VirtualBF16Type : TypeDef<Atlas_Dialect, "VirtualBF16"> {
  let mnemonic = "virtual_bf16";
  let summary = "Unallocated 32 by 32 BF16 VPU tile value";
  let assemblyFormat = "";
}

def Atlas_VirtualFP8Type : TypeDef<Atlas_Dialect, "VirtualFP8"> {
  let mnemonic = "virtual_fp8";
  let summary = "Unallocated 32 by 32 E4M3 FP8 MXU operand tile";
  let assemblyFormat = "";
}

#endif
