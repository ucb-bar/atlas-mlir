#ifndef ATLAS_OPS
#define ATLAS_OPS

include "Atlas/AtlasTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class Atlas_MachineOp<string mnemonic> :
    Atlas_Op<mnemonic, [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]>;

// A state token makes operations non-reorderable, including asynchronous launches.
def Atlas_StartOp : Atlas_Op<"start"> {
  let summary = "Start one physical instruction stream";
  let results = (outs Atlas_StateType:$next);
}

def Atlas_VLoadOp : Atlas_MachineOp<"vload"> {
  let summary = "Load VMEM into a physical tensor register";
  let arguments = (ins Atlas_StateType:$state, I32Attr:$dst, I32Attr:$base,
                   I32Attr:$offset, StrAttr:$format);
  let results = (outs Atlas_StateType:$next);
  let hasVerifier = 1;
}

def Atlas_VStoreOp : Atlas_MachineOp<"vstore"> {
  let summary = "Store a physical tensor register into VMEM";
  let arguments = (ins Atlas_StateType:$state, I32Attr:$src, I32Attr:$base,
                   I32Attr:$offset, StrAttr:$format);
  let results = (outs Atlas_StateType:$next);
  let hasVerifier = 1;
}

def Atlas_DMAOp : Atlas_MachineOp<"dma"> {
  let summary = "Asynchronous DMA load or store";
  let arguments = (ins Atlas_StateType:$state, StrAttr:$direction,
                   I32Attr:$channel, I32Attr:$reg, I32Attr:$dram,
                   I32Attr:$size);
  let results = (outs Atlas_StateType:$next);
  let hasVerifier = 1;
}

def Atlas_DMAWaitOp : Atlas_MachineOp<"dma_wait"> {
  let summary = "Wait for the selected DMA channel";
  let arguments = (ins Atlas_StateType:$state, I32Attr:$channel);
  let results = (outs Atlas_StateType:$next);
  let hasVerifier = 1;
}

def Atlas_DMAConfigOp : Atlas_MachineOp<"dma_config"> {
  let summary = "Set DMA base from a scalar register";
  let arguments = (ins Atlas_StateType:$state, I32Attr:$channel,
                   I32Attr:$base_reg);
  let results = (outs Atlas_StateType:$next);
  let hasVerifier = 1;
}

def Atlas_MXUPushOp : Atlas_MachineOp<"mxu_push"> {
  let summary = "Push tensor register into an MXU-local weight or accumulator slot";
  let arguments = (ins Atlas_StateType:$state, StrAttr:$kind,
                   I32Attr:$unit, I32Attr:$src, I32Attr:$slot);
  let results = (outs Atlas_StateType:$next);
  let hasVerifier = 1;
}

def Atlas_MXUMatmulOp : Atlas_MachineOp<"mxu_matmul"> {
  let summary = "Launch an MXU matmul, reset or accumulate";
  let arguments = (ins Atlas_StateType:$state, I32Attr:$unit,
                   I32Attr:$src, I32Attr:$weight_slot,
                   I32Attr:$acc_slot, BoolAttr:$accumulate);
  let results = (outs Atlas_StateType:$next);
  let hasVerifier = 1;
}

def Atlas_MXUPopOp : Atlas_MachineOp<"mxu_pop"> {
  let summary = "Pop an MXU-local accumulator to a tensor register";
  let arguments = (ins Atlas_StateType:$state, StrAttr:$format,
                   I32Attr:$unit, I32Attr:$dst, I32Attr:$slot,
                   I32Attr:$scale_reg);
  let results = (outs Atlas_StateType:$next);
  let hasVerifier = 1;
}

def Atlas_VPUBinaryOp : Atlas_MachineOp<"vpu_binary"> {
  let summary = "Physical BF16 binary VPU operation";
  let arguments = (ins Atlas_StateType:$state, StrAttr:$kind,
                   I32Attr:$dst, I32Attr:$lhs, I32Attr:$rhs);
  let results = (outs Atlas_StateType:$next);
  let hasVerifier = 1;
}

def Atlas_VPUUnaryOp : Atlas_MachineOp<"vpu_unary"> {
  let summary = "Physical unary VPU operation";
  let arguments = (ins Atlas_StateType:$state, StrAttr:$kind,
                   I32Attr:$dst, I32Attr:$src);
  let results = (outs Atlas_StateType:$next);
  let hasVerifier = 1;
}

def Atlas_VPUPackOp : Atlas_MachineOp<"vpu_pack"> {
  let summary = "Physical BF16/FP8 pack or unpack with explicit E8M0 scale register";
  let arguments = (ins Atlas_StateType:$state, StrAttr:$direction,
                   I32Attr:$dst, I32Attr:$src, I32Attr:$scale_reg);
  let results = (outs Atlas_StateType:$next);
  let hasVerifier = 1;
}

def Atlas_VPUReduceOp : Atlas_MachineOp<"vpu_reduce"> {
  let summary = "Physical row or column VPU reduction";
  let arguments = (ins Atlas_StateType:$state, StrAttr:$kind,
                   I32Attr:$dst, I32Attr:$src);
  let results = (outs Atlas_StateType:$next);
  let hasVerifier = 1;
}

def Atlas_VLIOp : Atlas_MachineOp<"vli"> {
  let summary = "Load immediate into tensor register, with broadcast mode";
  let arguments = (ins Atlas_StateType:$state, StrAttr:$mode,
                   I32Attr:$dst, I32Attr:$immediate);
  let results = (outs Atlas_StateType:$next);
  let hasVerifier = 1;
}

def Atlas_XLUTransposeOp : Atlas_MachineOp<"xlu_transpose"> {
  let summary = "Physical XLU transpose";
  let arguments = (ins Atlas_StateType:$state, I32Attr:$dst, I32Attr:$src);
  let results = (outs Atlas_StateType:$next);
  let hasVerifier = 1;
}

#endif
