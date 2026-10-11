#ifndef ATLAS_CONTRACT_ATTR_H
#define ATLAS_CONTRACT_ATTR_H

#include "Atlas/AtlasTiming.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/DenseSet.h"
#include <initializer_list>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

namespace mlir::atlas {

// Contract record readers: a missing or mistyped field reads as absent.
inline std::optional<int32_t> contractI32(DictionaryAttr d, StringRef key) {
  auto v = d.getAs<IntegerAttr>(key);
  if (!v || !v.getType().isSignlessInteger(32)) return std::nullopt;
  return int32_t(v.getInt());
}
// A nonnegative signless i32 instruction tag.
inline std::optional<int32_t> contractTag(Operation *op, StringRef key) {
  auto v = op->getAttrOfType<IntegerAttr>(key);
  if (!v || !v.getType().isSignlessInteger(32) || v.getInt() < 0) return std::nullopt;
  return int32_t(v.getInt());
}
inline StringRef contractString(DictionaryAttr d, StringRef key) {
  auto a = d.getAs<StringAttr>(key);
  return a ? a.getValue() : StringRef{};
}
inline std::vector<int32_t> contractI32Array(DictionaryAttr d, StringRef key) {
  auto a = d.getAs<DenseI32ArrayAttr>(key);
  return a ? std::vector<int32_t>(a.asArrayRef().begin(), a.asArrayRef().end()) : std::vector<int32_t>{};
}

// Reads required signless i32 fields in order, sign-extending into int32_t and
// zero-extending into uint32_t. `prefix` names the contract in diagnostics.
template <typename T>
LogicalResult readI32Fields(ModuleOp module, DictionaryAttr d, StringRef prefix,
                            std::initializer_list<std::pair<StringRef, T *>> fields) {
  static_assert(std::is_same_v<T, int32_t> || std::is_same_v<T, uint32_t>);
  for (auto [name, field] : fields) {
    auto integer = d.getAs<IntegerAttr>(name);
    if (!integer || !integer.getType().isSignlessInteger(32))
      return module.emitOpError(prefix) << " contract field requires signless i32: " << name;
    *field = std::is_signed_v<T> ? T(integer.getValue().getSExtValue()) : T(integer.getValue().getZExtValue());
  }
  return success();
}

// A signless i32 contract field; int32_t values keep their bit pattern.
inline NamedAttribute namedI32(Builder &b, StringRef name, uint32_t bits) {
  return b.getNamedAttr(name, b.getIntegerAttr(b.getI32Type(), llvm::APInt(32, bits)));
}

inline LogicalResult checkCaptured(Operation *op, StringRef prefix, StringRef field,
                                   std::optional<uint64_t> actual, uint64_t expected) {
  if (!actual)
    return op->emitOpError(prefix) << " contract cannot prove captured " << field;
  if (*actual != expected)
    return op->emitOpError(prefix) << " contract captured " << field << " mismatch: expected " << expected << ", got " << *actual;
  return success();
}

// AtlasCore uses wordAddr[18:3], size[12:0]; DMA consumes complete
// 32-byte beats. Full FP8/BF16 tiles are 1/2 KiB in selected VMEM.
inline bool validDMATileGeometry(uint64_t vmemByte, uint64_t dramByte, uint64_t bytes) {
  return (bytes == 1024 || bytes == 2048) && vmemByte % 1024 == 0 &&
         vmemByte + bytes <= timing::kVmemBytes && dramByte >= 0x80000000u &&
         dramByte % 32 == 0 && dramByte + bytes <= (uint64_t(1) << 32);
}

// Every block argument and operation result of `function`.
inline llvm::DenseSet<Value> sourceValues(func::FuncOp function) {
  llvm::DenseSet<Value> values;
  for (Block &block : function.getBody()) {
    values.insert(block.getArguments().begin(), block.getArguments().end());
    for (Operation &op : block)
      values.insert(op.getResults().begin(), op.getResults().end());
  }
  return values;
}

} // namespace mlir::atlas

#endif
