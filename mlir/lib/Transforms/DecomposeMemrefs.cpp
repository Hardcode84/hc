// SPDX-FileCopyrightText: 2024 The HC Authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hc/Transforms/Passes.hpp"

#include "hc/Dialect/HKernel/IR/HKernelOps.hpp"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Transforms/DialectConversion.h>

namespace hc {
#define GEN_PASS_DEF_DECOMPOSEMEMREFSPASS
#include "hc/Transforms/Passes.h.inc"
} // namespace hc

static mlir::FailureOr<size_t> getNumDynamicFields(mlir::MemRefType type) {
  size_t count = type.getNumDynamicDims();
  auto layout = type.getLayout();
  if (layout.isIdentity())
    return count;

  auto strided = mlir::dyn_cast<mlir::StridedLayoutAttr>(layout);
  if (!strided)
    return mlir::failure();

  for (auto s : strided.getStrides()) {
    if (mlir::ShapedType::isDynamic(s))
      ++count;
  }
  return count;
}

static void populateTypeConverter(mlir::MLIRContext *ctx,
                                  mlir::TypeConverter &converter) {
  auto index = mlir::IndexType::get(ctx);
  converter.addConversion(
      [index](mlir::MemRefType type) -> std::optional<mlir::Type> {
        llvm::SmallVector<mlir::Type> types;
        types.emplace_back(
            hc::hk::PtrType::get(type.getElementType(), type.getMemorySpace()));
        auto numVars = getNumDynamicFields(type);
        if (mlir::failed(numVars))
          return std::nullopt;

        types.append(*numVars, index);
        return mlir::TupleType::get(type.getContext(), types);
      });
}

namespace {
struct ConvertAlloca final : mlir::OpConversionPattern<mlir::memref::AllocaOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::memref::AllocaOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::MemRefType memrefType = op.getType();
    if (!memrefType.getLayout().isIdentity())
      return rewriter.notifyMatchFailure(
          op, "Only identity layout is supported for alloca");

    auto resultType =
        getTypeConverter()->convertType<mlir::TupleType>(memrefType);
    if (!resultType)
      return rewriter.notifyMatchFailure(op, "Unable to convert result type");

    mlir::ValueRange dynSizes = adaptor.getDynamicSizes();
    mlir::Location loc = op.getLoc();
    mlir::Value size = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);
    for (auto s : memrefType.getShape()) {
      mlir::Value dim;
      if (mlir::ShapedType::isDynamic(s)) {
        dim = dynSizes.front();
        dynSizes = dynSizes.drop_front();
      } else {
        dim = rewriter.create<mlir::arith::ConstantIndexOp>(loc, s);
      }
      size = rewriter.create<mlir::arith::MulIOp>(loc, size, dim);
    }
    mlir::Value ptr =
        rewriter.create<hc::hk::PtrAllocaOp>(loc, resultType.getType(0), size);
    llvm::SmallVector<mlir::Value> results;
    results.emplace_back(ptr);
    llvm::append_range(results, adaptor.getDynamicSizes());
    rewriter.replaceOpWithNewOp<hc::hk::MakeTupleOp>(op, resultType, results);
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

struct DecomposeMemrefsPass final
    : public hc::impl::DecomposeMemrefsPassBase<DecomposeMemrefsPass> {

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
