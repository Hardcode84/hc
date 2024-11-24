// SPDX-FileCopyrightText: 2024 The HC Authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hc/Transforms/Passes.hpp"

#include "hc/Dialect/HKernel/IR/HKernelOps.hpp"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Transforms/DialectConversion.h>

namespace hc {
#define GEN_PASS_DEF_DECOMPOSEPOINTERSPASS
#include "hc/Transforms/Passes.h.inc"
} // namespace hc

static void populateTypeConverter(mlir::MLIRContext *ctx,
                                  mlir::TypeConverter &converter) {
  converter.addConversion(
      [](hc::hk::PtrType type) -> std::optional<mlir::Type> {
        auto logical = mlir::dyn_cast_if_present<hc::hk::LogicalPtrAttr>(
            type.getMemorySpace());
        if (!logical)
          return std::nullopt;

        auto newPtrType = hc::hk::PtrType::get(type.getPointerType(),
                                               logical.getMemorySpace());
        return mlir::TupleType::get(type.getContext(),
                                    {newPtrType, logical.getOffsetType()});
      });
}

namespace {
struct ConvertAlloca final : mlir::OpConversionPattern<hc::hk::PtrAllocaOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(hc::hk::PtrAllocaOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto resType =
        getTypeConverter()->convertType<mlir::TupleType>(op.getType());
    if (!resType || resType.size() != 2 ||
        !mlir::isa<hc::hk::PtrType>(resType.getType(0)))
      return rewriter.notifyMatchFailure(op, "Invalid result type");

    auto ptrType = mlir::cast<hc::hk::PtrType>(resType.getType(0));
    auto offType = resType.getType(1);
    mlir::Location loc = op.getLoc();
    mlir::Value newPtr =
        rewriter.create<hc::hk::PtrAllocaOp>(loc, ptrType, adaptor.getSize());
    mlir::Value offset = rewriter.create<mlir::arith::ConstantOp>(
        loc, offType, rewriter.getIntegerAttr(offType, 0));
    mlir::Value args[] = {newPtr, offset};
    rewriter.replaceOpWithNewOp<hc::hk::MakeTupleOp>(op, resType, args);
    return mlir::success();
  }
};

struct ConvertReturn final : mlir::OpConversionPattern<mlir::func::ReturnOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::func::ReturnOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::func::ReturnOp>(op,
                                                      adaptor.getOperands());
    return mlir::success();
  }
};

struct DecomposePointersPass final
    : public hc::impl::DecomposePointersPassBase<DecomposePointersPass> {

  void runOnOperation() override {
    auto mod = getOperation();

    auto *ctx = &getContext();
    mlir::ConversionTarget target(*ctx);
    mlir::TypeConverter converter;

    // Convert unknown types to itself
    converter.addConversion([](mlir::Type type) { return type; });
    converter.addConversion(
        [&](mlir::TupleType type) -> std::optional<mlir::Type> {
          llvm::SmallVector<mlir::Type> newTypes(type.size());
          for (auto i : llvm::seq<size_t>(0, newTypes.size())) {
            mlir::Type newType = converter.convertType(type.getType(i));
            if (!newType)
              return std::nullopt;

            newTypes[i] = newType;
          }
          return mlir::TupleType::get(type.getContext(), newTypes);
        });

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
    patterns.insert<ConvertReturn>(converter, ctx);

    patterns.insert<ConvertAlloca>(converter, ctx);

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
