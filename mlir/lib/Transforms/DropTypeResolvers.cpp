// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hc/Transforms/Passes.hpp"

#include "hc/Dialect/Typing/IR/TypingOps.hpp"

namespace hc {
#define GEN_PASS_DEF_DROPTYPERESOLVERSPASS
#include "hc/Transforms/Passes.h.inc"
} // namespace hc

namespace {
struct DropTypeResolversPass final
    : public hc::impl::DropTypeResolversPassBase<DropTypeResolversPass> {

  void runOnOperation() override {
    unsigned dropped = 0;

    getOperation()->walk<mlir::WalkOrder::PreOrder>(
        [&](hc::typing::TypeResolverOp op) -> mlir::WalkResult {
          op->erase();
          ++dropped;
          return mlir::WalkResult::skip();
        });

    if (dropped == 0)
      markAllAnalysesPreserved();
  }
};
} // namespace
