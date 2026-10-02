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

def Atlas_VirtualScaleType : TypeDef<Atlas_Dialect, "VirtualScale"> {
  let mnemonic = "virtual_scale";
  let summary = "Immutable E8M0 scale code without a selected physical register";
  let assemblyFormat = "";
}

def Atlas_VirtualDMALoadFP8Type : TypeDef<Atlas_Dialect, "VirtualDMALoadFP8"> {
  let mnemonic = "virtual_dma_load_fp8";
  let summary = "Pending FP8 load with unallocated staging storage";
  let assemblyFormat = "";
}

def Atlas_VirtualDMALoadBF16Type : TypeDef<Atlas_Dialect, "VirtualDMALoadBF16"> {
  let mnemonic = "virtual_dma_load_bf16";
  let summary = "Pending BF16 load with unallocated staging storage";
  let assemblyFormat = "";
}

def Atlas_VirtualDMAStoreType : TypeDef<Atlas_Dialect, "VirtualDMAStore"> {
  let mnemonic = "virtual_dma_store";
  let summary = "Pending tile store with unallocated staging storage";
  let assemblyFormat = "";
}

def Atlas_VirtualMXUWeightType : TypeDef<Atlas_Dialect, "VirtualMXUWeight"> {
  let mnemonic = "virtual_mxu_weight";
  let summary = "A resident FP8 weight on one selected MXU, before slot allocation";
  let parameters = (ins "unsigned":$unit);
  let assemblyFormat = "`<` $unit `>`";
  let genVerifyDecl = 1;
}

def Atlas_VirtualMXUAccType : TypeDef<Atlas_Dialect, "VirtualMXUAcc"> {
  let mnemonic = "virtual_mxu_acc";
  let summary = "One version of a selected MXU accumulator, before slot allocation";
  let parameters = (ins "unsigned":$unit);
  let assemblyFormat = "`<` $unit `>`";
  let genVerifyDecl = 1;
}

#endif
