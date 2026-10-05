#ifndef ATLAS_DIALECT
#define ATLAS_DIALECT

include "mlir/IR/OpBase.td"

def Atlas_Dialect : Dialect {
  let name = "atlas";
  let cppNamespace = "::mlir::atlas";
  let summary = "Physical Atlas instruction stream with explicit state order";
  let description = [{
    Hand-authored machine-stage reference for the selected Atlas RTL. Each operation
    has one input and output state token so instruction issue order is explicit.
    Physical tensor registers and slots are encoded as checked integer attributes.
    This dialect does not claim to prove execution timing or arithmetic equivalence.
  }];
  let useDefaultTypePrinterParser = 1;
  let extraClassDeclaration = [{ void registerTypes(); }];
}

class Atlas_Op<string mnemonic, list<Trait> traits = []> :
    Op<Atlas_Dialect, mnemonic, traits>;

#endif
