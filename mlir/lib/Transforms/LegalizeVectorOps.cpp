// SPDX-FileCopyrightText: 2025 The HC Authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hc/Transforms/Passes.hpp"

#include "hc/Dialect/HKernel/IR/HKernelOps.hpp"

#include <mlir/Dialect/UB/IR/UBOps.h>
#include <mlir/Dialect/Vector/Transforms/LoweringPatterns.h>
#include <mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace hc {
#define GEN_PASS_DEF_LEGALIZEVECTOROPSPASS
#include "hc/Transforms/Passes.h.inc"
} // namespace hc

namespace {
struct FlattenPoison : public mlir::OpRewritePattern<mlir::ub::PoisonOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::ub::PoisonOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto resType = mlir::dyn_cast<mlir::VectorType>(op.getType());
    if (!resType || resType.getRank() <= 1 || resType.isScalable())
      return mlir::failure();

    int64_t newShape[] = {resType.getNumElements()};
    auto newResType = mlir::VectorType::get(newShape, resType.getElementType());
    mlir::Value res =
        rewriter.create<mlir::ub::PoisonOp>(op.getLoc(), newResType);
    rewriter.replaceOpWithNewOp<mlir::vector::ShapeCastOp>(op, resType, res);
    return mlir::success();
  }
};

struct FlattenPtrLoad : public mlir::OpRewritePattern<hc::hk::PtrLoadOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(hc::hk::PtrLoadOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto resType = mlir::dyn_cast<mlir::VectorType>(op.getType());
    if (!resType || resType.getRank() <= 1 || resType.isScalable())
      return mlir::failure();

    auto origShape = resType.getShape();

    auto getShape = [](mlir::Type type) -> mlir::ArrayRef<int64_t> {
      if (auto shaped = mlir::dyn_cast<mlir::ShapedType>(type))
        return shaped.getShape();

      return {};
    };

    mlir::Value mask = op.getMask();
    if (mask && getShape(mask.getType()) != origShape)
      return mlir::failure();

    mlir::Value passThru = op.getPassThru();
    if (passThru && getShape(passThru.getType()) != origShape)
      return mlir::failure();

    int64_t newShape[] = {resType.getNumElements()};
    auto newResType = mlir::VectorType::get(newShape, resType.getElementType());

    mlir::Location loc = op.getLoc();
    if (mask) {
      auto elemType =
          mlir::cast<mlir::VectorType>(mask.getType()).getElementType();
      auto newType = mlir::VectorType::get(newShape, elemType);
      mask = rewriter.create<mlir::vector::ShapeCastOp>(loc, newType, mask);
    }

    if (passThru) {
      auto elemType =
          mlir::cast<mlir::VectorType>(passThru.getType()).getElementType();
      auto newType = mlir::VectorType::get(newShape, elemType);
      passThru =
          rewriter.create<mlir::vector::ShapeCastOp>(loc, newType, passThru);
    }

    mlir::Value result = rewriter.create<hc::hk::PtrLoadOp>(
        loc, newResType, op.getBase(), op.getOffset(), mask, passThru);
    rewriter.replaceOpWithNewOp<mlir::vector::ShapeCastOp>(op, resType, result);
    return mlir::success();
  }
};

struct FlattenPtrStore : public mlir::OpRewritePattern<hc::hk::PtrStoreOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(hc::hk::PtrStoreOp op,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::Value val = op.getValue();
    auto resType = mlir::dyn_cast<mlir::VectorType>(val.getType());
    if (!resType || resType.getRank() <= 1 || resType.isScalable())
      return mlir::failure();

    auto origShape = resType.getShape();

    auto getShape = [](mlir::Type type) -> mlir::ArrayRef<int64_t> {
      if (auto shaped = mlir::dyn_cast<mlir::ShapedType>(type))
        return shaped.getShape();

      return {};
    };

    mlir::Value mask = op.getMask();
    if (mask && getShape(mask.getType()) != origShape)
      return mlir::failure();

    int64_t newShape[] = {resType.getNumElements()};
    auto newResType = mlir::VectorType::get(newShape, resType.getElementType());

    mlir::Location loc = op.getLoc();
    val = rewriter.create<mlir::vector::ShapeCastOp>(loc, newResType, val);

    if (mask) {
      auto elemType =
          mlir::cast<mlir::VectorType>(mask.getType()).getElementType();
      auto newType = mlir::VectorType::get(newShape, elemType);
      mask = rewriter.create<mlir::vector::ShapeCastOp>(loc, newType, mask);
    }

    rewriter.create<hc::hk::PtrStoreOp>(loc, val, op.getBase(), op.getOffset(),
                                        mask);
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

struct LegalizeVectorOpsPass final
    : public hc::impl::LegalizeVectorOpsPassBase<LegalizeVectorOpsPass> {

  void runOnOperation() override {
    mlir::RewritePatternSet patterns(&getContext());
    mlir::vector::populateVectorMaskOpLoweringPatterns(patterns);
    mlir::vector::populateVectorMaskMaterializationPatterns(patterns, false);

    patterns.insert<FlattenPoison, FlattenPtrLoad, FlattenPtrStore>(
        patterns.getContext());

    if (mlir::failed(
            applyPatternsGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
  }
};
} // namespace
