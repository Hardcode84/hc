// SPDX-FileCopyrightText: 2024 The HC Authors
// SPDX-FileCopyrightText: 2025 The HC Authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hc/Transforms/Passes.hpp"

#include "hc/Dialect/HKernel/IR/HKernelOps.hpp"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace hc {
#define GEN_PASS_DEF_DECOMPOSEHKERNELOPSPASS
#include "hc/Transforms/Passes.h.inc"
} // namespace hc

namespace {
struct DecomposeResolveSlice final
    : public mlir::OpRewritePattern<hc::hk::ResolveSliceOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(hc::hk::ResolveSliceOp op,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    mlir::Value lower = op.getLower();
    if (!lower)
      lower = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);

    mlir::Value upper = op.getUpper();
    if (!upper)
      upper = op.getSrcSize();

    mlir::Value step = op.getStep();
    if (!step)
      step = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);

    mlir::Value retSize =
        rewriter.create<mlir::arith::SubIOp>(loc, upper, lower);
    rewriter.replaceOp(op, {lower, retSize, step});
    return mlir::success();
  }
};
} // namespace

static void populateDecomposePatterns(mlir::RewritePatternSet &patterns) {
  patterns.insert<DecomposeResolveSlice>(patterns.getContext());
}

namespace {
struct DecomposeHKernelOpsPass final
    : public hc::impl::DecomposeHKernelOpsPassBase<DecomposeHKernelOpsPass> {

  void runOnOperation() override {
    mlir::MLIRContext *ctx = &getContext();
    mlir::RewritePatternSet patterns(ctx);
    populateDecomposePatterns(patterns);

    if (mlir::failed(
            applyPatternsGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
  }
};
} // namespace
