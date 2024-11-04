// SPDX-FileCopyrightText: 2024 The HC Authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hc/Transforms/Passes.hpp"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Vector/IR/VectorOps.h>
#include <mlir/Transforms/DialectConversion.h>

namespace hc {
#define GEN_PASS_DEF_LEGALIZEBOOLMEMREFSPASS
#include "hc/Transforms/Passes.h.inc"
} // namespace hc

static bool isI1Memref(mlir::MemRefType type) {
  auto elemType = mlir::dyn_cast<mlir::IntegerType>(type.getElementType());
  return elemType && elemType.getWidth() == 1;
}

static bool isTrivialOp(mlir::Operation *op) {
  if (op->hasTrait<mlir::OpTrait::IsTerminator>())
    return true;

  if (mlir::isa<mlir::ViewLikeOpInterface>(op))
    return true;

  if (mlir::isa<mlir::memref::AllocOp, mlir::memref::AllocaOp,
                mlir::memref::DeallocOp>(op))
    return true;

  return false;
}

namespace {
struct ConvertTypes final : mlir::ConversionPattern {

  ConvertTypes(const mlir::TypeConverter &converter, mlir::MLIRContext *ctx,
               mlir::PatternBenefit benefit = 0)
      : mlir::ConversionPattern(converter, mlir::Pattern::MatchAnyOpTypeTag{},
                                benefit, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op, mlir::ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    if (!isTrivialOp(op))
      return rewriter.notifyMatchFailure(op, "Not a trivial op");

    mlir::FailureOr<mlir::Operation *> newOp =
        mlir::convertOpResultTypes(op, operands, *getTypeConverter(), rewriter);
    if (failed(newOp))
      return mlir::failure();

    rewriter.replaceOp(op, (*newOp)->getResults());
    return mlir::success();
  }
};

struct ConvertVecStore final
    : public mlir::OpConversionPattern<mlir::vector::StoreOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::vector::StoreOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    if (!isI1Memref(op.getBase().getType()))
      return rewriter.notifyMatchFailure(op, "Not a i1 memref");

    mlir::Value base = adaptor.getBase();
    mlir::Value valToStore = adaptor.getValueToStore();
    auto newElemType =
        mlir::cast<mlir::MemRefType>(base.getType()).getElementType();
    auto newVecType =
        mlir::cast<mlir::VectorType>(valToStore.getType()).clone(newElemType);

    mlir::Location loc = op.getLoc();
    valToStore =
        rewriter.create<mlir::arith::ExtUIOp>(loc, newVecType, valToStore);

    rewriter.replaceOpWithNewOp<mlir::vector::StoreOp>(
        op, valToStore, base, adaptor.getIndices(), adaptor.getNontemporal());
    return mlir::success();
  }
};

} // namespace

static void populateTypeConverter(mlir::MLIRContext *ctx,
                                  mlir::TypeConverter &converter) {
  auto i8 = mlir::IntegerType::get(ctx, 8);
  converter.addConversion(
      [i8](mlir::MemRefType type) -> std::optional<mlir::Type> {
        if (!isI1Memref(type))
          return std::nullopt;

        return type.clone(i8);
      });
}

namespace {
struct LegalizeBoolMemrefsPass final
    : public hc::impl::LegalizeBoolMemrefsPassBase<LegalizeBoolMemrefsPass> {

  void runOnOperation() override {
    auto mod = getOperation();

    auto *ctx = &getContext();
    mlir::ConversionTarget target(*ctx);
    mlir::TypeConverter converter;

    // Convert unknown types to itself
    converter.addConversion([](mlir::Type type) { return type; });

    populateTypeConverter(ctx, converter);

    auto materialize = [](mlir::OpBuilder &builder, mlir::Type type,
                          mlir::ValueRange inputs,
                          mlir::Location loc) -> mlir::Value {
      return builder.create<mlir::UnrealizedConversionCastOp>(loc, type, inputs)
          .getResult(0);
    };
    converter.addArgumentMaterialization(materialize);
    converter.addSourceMaterialization(materialize);
    converter.addTargetMaterialization(materialize);

    mlir::RewritePatternSet patterns(ctx);

    mlir::populateAnyFunctionOpInterfaceTypeConversionPattern(patterns,
                                                              converter);
    patterns.insert<ConvertTypes, ConvertVecStore>(converter, ctx);

    target.addDynamicallyLegalOp<mlir::func::FuncOp>(
        [&](mlir::func::FuncOp op) {
          return converter.isSignatureLegal(op.getFunctionType()) &&
                 converter.isLegal(&op.getBody());
        });

    target.markUnknownOpDynamicallyLegal(
        [&](mlir::Operation *op) -> bool { return converter.isLegal(op); });

    if (mlir::failed(
            mlir::applyFullConversion(mod, target, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace
