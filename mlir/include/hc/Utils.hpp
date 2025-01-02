// SPDX-FileCopyrightText: 2024 The HC Authors
// SPDX-FileCopyrightText: 2025 The HC Authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <utility>

#include <llvm/ADT/ScopeExit.h>

namespace mlir {
class ConversionTarget;
class RewritePatternSet;
class TypeConverter;
} // namespace mlir

namespace hc {
template <typename T, typename H, typename F>
inline auto scopedDiagHandler(T &ctx, H &&diag_handler, F &&func) {
  auto &diagEngine = ctx.getDiagEngine();
  auto diagId = diagEngine.registerHandler(std::forward<H>(diag_handler));
  auto diagGuard =
      llvm::make_scope_exit([&]() { diagEngine.eraseHandler(diagId); });
  return func();
}

void populateFuncPatternsAndTypeConversion(mlir::RewritePatternSet &patterns,
                                           mlir::ConversionTarget &target,
                                           mlir::TypeConverter &converter);
} // namespace hc
