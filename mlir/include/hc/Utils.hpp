// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <utility>

#include <llvm/ADT/ScopeExit.h>

namespace hc {
template <typename T, typename H, typename F>
inline auto scopedDiagHandler(T &ctx, H &&diag_handler, F &&func) {
  auto &diagEngine = ctx.getDiagEngine();
  auto diagId = diagEngine.registerHandler(std::forward<H>(diag_handler));
  auto diagGuard =
      llvm::make_scope_exit([&]() { diagEngine.eraseHandler(diagId); });
  return func();
}
} // namespace hc
