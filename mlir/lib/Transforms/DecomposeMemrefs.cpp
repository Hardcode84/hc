// SPDX-FileCopyrightText: 2024 The HC Authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hc/Transforms/Passes.hpp"

#include "hc/Dialect/HKernel/IR/HKernelOps.hpp"

#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Arith/Utils/Utils.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Utils/IndexingUtils.h>
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

static mlir::LogicalResult getStrides(mlir::MemRefType type,
                                      llvm::SmallVectorImpl<int64_t> &strides) {
  int64_t offset; // unused
  return mlir::getStridesAndOffset(type, strides, offset);
}

static mlir::LogicalResult
materializeStrides(mlir::OpBuilder &builder, mlir::Location loc,
                   mlir::MemRefType type, mlir::Value srcPacked,
                   llvm::SmallVectorImpl<mlir::OpFoldResult> &ret) {
  auto tupleType = mlir::dyn_cast<mlir::TupleType>(srcPacked.getType());
  int currentIndex = 1; // 0th is ptr
  auto getNextVal = [&]() -> mlir::OpFoldResult {
    mlir::Value idx =
        builder.create<mlir::arith::ConstantIndexOp>(loc, currentIndex);
    auto elemType = tupleType.getType(currentIndex);
    ++currentIndex;
    return builder.create<hc::hk::TupleExtractOp>(loc, elemType, srcPacked, idx)
        .getResult();
  };

  auto createConst = [&](int64_t val) -> mlir::OpFoldResult {
    return builder.getIndexAttr(val);
  };

  auto getDim = [&](int64_t val) -> mlir::OpFoldResult {
    if (mlir::ShapedType::isDynamic(val)) {
      return getNextVal();
    } else {
      return createConst(val);
    }
  };

  if (type.getLayout().isIdentity()) {
    mlir::Value stride =
        mlir::getValueOrCreateConstantIndexOp(builder, loc, createConst(1));
    ret.emplace_back(stride);
    for (auto s : llvm::reverse(type.getShape().drop_front())) {
      mlir::Value dimVal =
          mlir::getValueOrCreateConstantIndexOp(builder, loc, getDim(s));
      stride = builder.create<mlir::arith::MulIOp>(loc, stride, dimVal);
      ret.emplace_back(stride);
    }
    std::reverse(ret.end() - type.getRank(), ret.end()); // inplace
  } else {
    llvm::SmallVector<int64_t> strides;
    if (mlir::failed(getStrides(type, strides)))
      return mlir::failure();

    currentIndex += type.getNumDynamicDims();
    for (auto s : strides) {
      ret.emplace_back(getDim(s));
    }
  }
  return mlir::success();
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

struct ConvertSubview final
    : mlir::OpConversionPattern<mlir::memref::SubViewOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::memref::SubViewOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::MemRefType srcMemrefType = op.getSourceType();
    mlir::MemRefType resMemrefType = op.getType();
    auto resultType =
        getTypeConverter()->convertType<mlir::TupleType>(resMemrefType);
    if (!resultType)
      return rewriter.notifyMatchFailure(op, "Unable to convert result type");

    llvm::SmallVector<int64_t> outStrides;
    if (mlir::failed(getStrides(resMemrefType, outStrides)))
      return rewriter.notifyMatchFailure(op,
                                         "Unable to convert result strides");

    mlir::Location loc = op.getLoc();
    llvm::SmallVector<mlir::OpFoldResult> strides;
    if (mlir::failed(materializeStrides(rewriter, loc, srcMemrefType,
                                        adaptor.getSource(), strides)))
      return rewriter.notifyMatchFailure(op, "Unable to materialize strides");

    auto mixedOffsets = mlir::getMixedValues(adaptor.getStaticOffsets(),
                                             adaptor.getOffsets(), rewriter);
    auto mixedSizes = mlir::getMixedValues(adaptor.getStaticSizes(),
                                           adaptor.getSizes(), rewriter);
    auto mixedStrides = mlir::getMixedValues(adaptor.getStaticStrides(),
                                             adaptor.getStrides(), rewriter);
    auto &&[expr, values] = mlir::computeLinearIndex(rewriter.getIndexAttr(0),
                                                     strides, mixedOffsets);
    mlir::OpFoldResult finalOffset =
        mlir::affine::makeComposedFoldedAffineApply(rewriter, loc, expr,
                                                    values);
    mlir::Value finalOffsetVal =
        mlir::getValueOrCreateConstantIndexOp(rewriter, loc, finalOffset);

    mlir::Type ptrType = resultType.getType(0);
    mlir::Value zero = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
    mlir::Value ptr = rewriter.create<hc::hk::TupleExtractOp>(
        loc, ptrType, adaptor.getSource(), zero);
    ptr = rewriter.create<hc::hk::PtrAddOp>(loc, ptrType, ptr, finalOffsetVal);

    llvm::SmallVector<mlir::Value> results;
    results.emplace_back(ptr);

    auto droppedDims = op.getDroppedDims();
    for (auto &&[i, dim, dynDim] :
         llvm::enumerate(resMemrefType.getShape(), mixedSizes)) {
      if (droppedDims[i] || !mlir::ShapedType::isDynamic(dim))
        continue;

      results.emplace_back(
          mlir::getValueOrCreateConstantIndexOp(rewriter, loc, dynDim));
    }

    llvm::ArrayRef outStridesRange(outStrides);
    for (auto &&[i, dynStride] : llvm::enumerate(mixedStrides)) {
      if (droppedDims[i])
        continue;

      auto stride = outStridesRange.front();
      outStridesRange = outStridesRange.drop_front();
      if (!mlir::ShapedType::isDynamic(stride))
        continue;

      mlir::Value srcStride =
          mlir::getValueOrCreateConstantIndexOp(rewriter, loc, strides[i]);
      mlir::Value dstStride =
          mlir::getValueOrCreateConstantIndexOp(rewriter, loc, dynStride);
      mlir::Value newStride =
          rewriter.create<mlir::arith::MulIOp>(loc, srcStride, dstStride);
      results.emplace_back(newStride);
    }

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

    patterns.insert<ConvertAlloca, ConvertSubview>(converter, ctx);

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
