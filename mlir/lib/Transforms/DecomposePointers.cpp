// SPDX-FileCopyrightText: 2024 The HC Authors
// SPDX-FileCopyrightText: 2025 The HC Authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hc/Transforms/Passes.hpp"

#include "hc/Dialect/HKernel/IR/HKernelOps.hpp"
#include "hc/Utils.hpp"

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

        auto newPtrType = hc::hk::PtrType::get(type.getElementType(),
                                               logical.getMemorySpace());
        return mlir::TupleType::get(type.getContext(),
                                    {newPtrType, logical.getOffsetType()});
      });
}

static mlir::Value doCast(mlir::OpBuilder &builder, mlir::Location loc,
                          mlir::Value src, mlir::Type dstType) {
  mlir::Type srcType = src.getType();
  if (srcType == dstType)
    return src;

  assert(srcType.isIntOrIndex());
  assert(dstType.isIntOrIndex());
  if (mlir::isa<mlir::IndexType>(srcType) ||
      mlir::isa<mlir::IndexType>(dstType))
    return builder.create<mlir::arith::IndexCastOp>(loc, dstType, src);

  if (dstType.getIntOrFloatBitWidth() < srcType.getIntOrFloatBitWidth()) {
    return builder.create<mlir::arith::TruncIOp>(loc, dstType, src);
  } else {
    return builder.create<mlir::arith::ExtSIOp>(loc, dstType, src);
  }
}

namespace {
struct ConvertPtrAlloca final : mlir::OpConversionPattern<hc::hk::PtrAllocaOp> {
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

struct ConvertPtrAdd final : mlir::OpConversionPattern<hc::hk::PtrAddOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(hc::hk::PtrAddOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto resType =
        getTypeConverter()->convertType<mlir::TupleType>(op.getType());
    if (!resType || resType.size() != 2 ||
        !mlir::isa<hc::hk::PtrType>(resType.getType(0)))
      return rewriter.notifyMatchFailure(op, "Invalid result type");

    auto ptrType = mlir::cast<hc::hk::PtrType>(resType.getType(0));
    auto offType = resType.getType(1);
    mlir::Location loc = op.getLoc();
    mlir::Value zero = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
    mlir::Value one = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);
    mlir::Value ptr = rewriter.create<hc::hk::TupleExtractOp>(
        loc, ptrType, adaptor.getBase(), zero);
    mlir::Value srcOffset = rewriter.create<hc::hk::TupleExtractOp>(
        loc, offType, adaptor.getBase(), one);
    mlir::Value dstOffset = doCast(rewriter, loc, adaptor.getOffset(), offType);
    auto ovfFlags = mlir::arith::IntegerOverflowFlags::nsw |
                    mlir::arith::IntegerOverflowFlags::nuw;
    mlir::Value offset = rewriter.create<mlir::arith::AddIOp>(
        loc, srcOffset, dstOffset, ovfFlags);
    mlir::Value args[] = {ptr, offset};
    rewriter.replaceOpWithNewOp<hc::hk::MakeTupleOp>(op, resType, args);
    return mlir::success();
  }
};

struct ConvertPtrLoad final : mlir::OpConversionPattern<hc::hk::PtrLoadOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(hc::hk::PtrLoadOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto resType = getTypeConverter()->convertType(op.getType());
    if (!resType)
      return rewriter.notifyMatchFailure(op, "Invalid result type");

    mlir::Value base = adaptor.getBase();
    auto baseType = mlir::dyn_cast<mlir::TupleType>(base.getType());
    if (!baseType || baseType.size() != 2 ||
        !mlir::isa<hc::hk::PtrType>(baseType.getType(0)))
      return rewriter.notifyMatchFailure(op, "Invalid base type");

    auto ptrType = mlir::cast<hc::hk::PtrType>(baseType.getType(0));
    auto offType = baseType.getType(1);
    mlir::Location loc = op.getLoc();
    mlir::Value zero = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
    mlir::Value one = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);
    mlir::Value ptr = rewriter.create<hc::hk::TupleExtractOp>(
        loc, ptrType, adaptor.getBase(), zero);
    mlir::Value offset = rewriter.create<hc::hk::TupleExtractOp>(
        loc, offType, adaptor.getBase(), one);
    if (adaptor.getOffset()) {
      mlir::Value dstOffset =
          doCast(rewriter, loc, adaptor.getOffset(), offType);
      auto ovfFlags = mlir::arith::IntegerOverflowFlags::nsw |
                      mlir::arith::IntegerOverflowFlags::nuw;
      offset = rewriter.create<mlir::arith::AddIOp>(loc, offset, dstOffset,
                                                    ovfFlags);
    }

    rewriter.replaceOpWithNewOp<hc::hk::PtrLoadOp>(
        op, resType, ptr, offset, adaptor.getMask(), adaptor.getPassThru());
    return mlir::success();
  }
};

struct ConvertPtrStore final : mlir::OpConversionPattern<hc::hk::PtrStoreOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(hc::hk::PtrStoreOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Value base = adaptor.getBase();
    auto baseType = mlir::dyn_cast<mlir::TupleType>(base.getType());
    if (!baseType || baseType.size() != 2 ||
        !mlir::isa<hc::hk::PtrType>(baseType.getType(0)))
      return rewriter.notifyMatchFailure(op, "Invalid base type");

    auto ptrType = mlir::cast<hc::hk::PtrType>(baseType.getType(0));
    auto offType = baseType.getType(1);
    mlir::Location loc = op.getLoc();
    mlir::Value zero = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
    mlir::Value one = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);
    mlir::Value ptr = rewriter.create<hc::hk::TupleExtractOp>(
        loc, ptrType, adaptor.getBase(), zero);
    mlir::Value offset = rewriter.create<hc::hk::TupleExtractOp>(
        loc, offType, adaptor.getBase(), one);
    if (adaptor.getOffset()) {
      mlir::Value dstOffset =
          doCast(rewriter, loc, adaptor.getOffset(), offType);
      auto ovfFlags = mlir::arith::IntegerOverflowFlags::nsw |
                      mlir::arith::IntegerOverflowFlags::nuw;
      offset = rewriter.create<mlir::arith::AddIOp>(loc, offset, dstOffset,
                                                    ovfFlags);
    }

    rewriter.replaceOpWithNewOp<hc::hk::PtrStoreOp>(op, adaptor.getValue(), ptr,
                                                    offset, adaptor.getMask());
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

    hc::populateFuncPatternsAndTypeConversion(patterns, target, converter);

    patterns.insert<ConvertPtrAlloca, ConvertPtrAdd, ConvertPtrLoad,
                    ConvertPtrStore>(converter, ctx);

    target.markUnknownOpDynamicallyLegal(
        [&](mlir::Operation *op) -> bool { return converter.isLegal(op); });

    if (mlir::failed(
            mlir::applyFullConversion(mod, target, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace
