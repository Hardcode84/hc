// SPDX-FileCopyrightText: 2024 The HC Authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hc/Transforms/Passes.hpp"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Vector/IR/VectorOps.h>
#include <mlir/Transforms/DialectConversion.h>

namespace hc {
#define GEN_PASS_DEF_LEGALIZEBOOLMEMREFSPASS
#include "hc/Transforms/Passes.h.inc"
} // namespace hc

namespace {
struct ConvertTypes final : mlir::ConversionPattern {

  ConvertTypes(const mlir::TypeConverter &converter, mlir::MLIRContext *ctx,
               mlir::PatternBenefit benefit = 0)
      : mlir::ConversionPattern(converter, mlir::Pattern::MatchAnyOpTypeTag{},
                                benefit, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op, mlir::ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::FailureOr<mlir::Operation *> newOp =
        mlir::convertOpResultTypes(op, operands, *getTypeConverter(), rewriter);
    if (failed(newOp))
      return mlir::failure();

    rewriter.replaceOp(op, (*newOp)->getResults());
    return mlir::success();
  }
};
} // namespace

static void populateTypeConverter(mlir::MLIRContext *ctx,
                                  mlir::TypeConverter &converter) {
  auto i8 = mlir::IntegerType::get(ctx, 8);
  converter.addConversion(
      [i8](mlir::MemRefType type) -> std::optional<mlir::Type> {
        auto elemType =
            mlir::dyn_cast<mlir::IntegerType>(type.getElementType());
        if (!elemType || elemType.getWidth() != 1)
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
    patterns.insert<ConvertTypes>(converter, ctx);

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
