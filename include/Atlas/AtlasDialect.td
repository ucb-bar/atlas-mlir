#ifndef ATLAS_DIALECT
#define ATLAS_DIALECT

include "mlir/IR/OpBase.td"

def Atlas_Dialect : Dialect {
  let name = "atlas";
  let cppNamespace = "::mlir::atlas";
  let summary = "Atlas virtual register candidates and physical instruction streams";
  let description = [{
    Hand-authored reference for the selected Atlas RTL. Virtual BF16 VPU
    candidates carry SSA register values and are not encodable. Machine operations
    have an explicit physical state chain; physical registers and slots remain
    checked integer attributes. This dialect does not prove timing or arithmetic
    equivalence.
  }];
  let useDefaultTypePrinterParser = 1;
  let extraClassDeclaration = [{ void registerTypes(); }];
}

class Atlas_Op<string mnemonic, list<Trait> traits = []> :
    Op<Atlas_Dialect, mnemonic, traits>;

#endif
