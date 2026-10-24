#pragma once
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"

namespace model_timing_test {
inline std::string encode(llvm::json::Object object) {
  std::string bytes;
  llvm::raw_string_ostream stream(bytes);
  stream << llvm::json::Value(std::move(object));
  return bytes;
}
inline std::string hash(llvm::StringRef bytes) {
  auto digest = llvm::SHA256::hash(llvm::ArrayRef<uint8_t>(
      reinterpret_cast<const uint8_t *>(bytes.data()), bytes.size()));
  return llvm::toHex(llvm::ArrayRef<uint8_t>(digest), true);
}
} // namespace model_timing_test
