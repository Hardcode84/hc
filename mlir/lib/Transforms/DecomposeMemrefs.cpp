// SPDX-FileCopyrightText: 2024 The HC Authors
// SPDX-FileCopyrightText: 2025 The HC Authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hc/Transforms/Passes.hpp"

#include "hc/Dialect/HKernel/IR/HKernelOps.hpp"
#include "hc/Utils.hpp"

#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Arith/Utils/Utils.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Utils/IndexingUtils.h>
#include <mlir/Dialect/Vector/IR/VectorOps.h>
#include <mlir/Transforms/DialectConversion.h>

namespace hc {
#define GEN_PASS_DEF_DECOMPOSEMEMREFSPASS
#include "hc/Transforms/Passes.h.inc"
#define GEN_PASS_DEF_LEGALIZEMEMREFABIPASS
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
struct ConvertDim final : mlir::OpConversionPattern<mlir::memref::DimOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::memref::DimOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto memrefType = mlir::cast<mlir::MemRefType>(op.getSource().getType());
    auto idx = op.getConstantIndex();
    if (!idx || *idx >= memrefType.getRank())
      return rewriter.notifyMatchFailure(op, "Invalid dim index");

    mlir::Value src = adaptor.getSource();
    auto packedType = mlir::dyn_cast<mlir::TupleType>(src.getType());
    if (!packedType)
      return rewriter.notifyMatchFailure(op, "Invalid packed type");

    mlir::Location loc = op.getLoc();
    if (!memrefType.isDynamicDim(*idx)) {
      rewriter.replaceOpWithNewOp<mlir::arith::ConstantIndexOp>(
          op, memrefType.getDimSize(*idx));
      return mlir::success();
    }

    int packedIdx = 1;
    for (auto i : llvm::seq<int64_t>(0, *idx)) {
      if (memrefType.isDynamicDim(i))
        ++packedIdx;
    }
    mlir::Value packedIdxVal =
        rewriter.create<mlir::arith::ConstantIndexOp>(loc, packedIdx);
    rewriter.replaceOpWithNewOp<hc::hk::TupleExtractOp>(
        op, rewriter.getIndexType(), src, packedIdxVal);
    return mlir::success();
  }
};

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

struct ConvertLoad final : mlir::OpConversionPattern<mlir::memref::LoadOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::memref::LoadOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::MemRefType memrefType = op.getMemRefType();
    mlir::Value base = adaptor.getMemref();
    auto tupleType = mlir::dyn_cast<mlir::TupleType>(base.getType());
    if (!tupleType)
      return rewriter.notifyMatchFailure(op, "Invalid memref type");

    auto resultType = getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(op, "Unable to convert result type");

    mlir::Location loc = op.getLoc();
    llvm::SmallVector<mlir::OpFoldResult> strides;
    if (mlir::failed(
            materializeStrides(rewriter, loc, memrefType, base, strides)))
      return rewriter.notifyMatchFailure(op, "Unable to materialize strides");

    auto mixedOffsets = mlir::getAsOpFoldResult(adaptor.getIndices());
    auto &&[expr, values] = mlir::computeLinearIndex(rewriter.getIndexAttr(0),
                                                     strides, mixedOffsets);

    mlir::OpFoldResult finalOffset =
        mlir::affine::makeComposedFoldedAffineApply(rewriter, loc, expr,
                                                    values);
    mlir::Value finalOffsetVal =
        mlir::getValueOrCreateConstantIndexOp(rewriter, loc, finalOffset);

    mlir::Type ptrType = tupleType.getType(0);
    mlir::Value zero = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
    mlir::Value ptr =
        rewriter.create<hc::hk::TupleExtractOp>(loc, ptrType, base, zero);
    rewriter.replaceOpWithNewOp<hc::hk::PtrLoadOp>(
        op, resultType, ptr, finalOffsetVal, /*mask*/ nullptr,
        /*pass_thru*/ nullptr);
    return mlir::success();
  }
};

struct ConvertStore final : mlir::OpConversionPattern<mlir::memref::StoreOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::memref::StoreOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::MemRefType memrefType = op.getMemRefType();
    mlir::Value base = adaptor.getMemref();
    auto tupleType = mlir::dyn_cast<mlir::TupleType>(base.getType());
    if (!tupleType)
      return rewriter.notifyMatchFailure(op, "Invalid memref type");

    mlir::Location loc = op.getLoc();
    llvm::SmallVector<mlir::OpFoldResult> strides;
    if (mlir::failed(
            materializeStrides(rewriter, loc, memrefType, base, strides)))
      return rewriter.notifyMatchFailure(op, "Unable to materialize strides");

    auto mixedOffsets = mlir::getAsOpFoldResult(adaptor.getIndices());
    auto &&[expr, values] = mlir::computeLinearIndex(rewriter.getIndexAttr(0),
                                                     strides, mixedOffsets);

    mlir::OpFoldResult finalOffset =
        mlir::affine::makeComposedFoldedAffineApply(rewriter, loc, expr,
                                                    values);
    mlir::Value finalOffsetVal =
        mlir::getValueOrCreateConstantIndexOp(rewriter, loc, finalOffset);

    mlir::Type ptrType = tupleType.getType(0);
    mlir::Value zero = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
    mlir::Value ptr =
        rewriter.create<hc::hk::TupleExtractOp>(loc, ptrType, base, zero);
    rewriter.replaceOpWithNewOp<hc::hk::PtrStoreOp>(
        op, adaptor.getValue(), ptr, finalOffsetVal, /*mask*/ nullptr);
    return mlir::success();
  }
};

struct ConvertVecLoad final : mlir::OpConversionPattern<mlir::vector::LoadOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::vector::LoadOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::MemRefType memrefType = op.getMemRefType();
    mlir::Value base = adaptor.getBase();
    auto tupleType = mlir::dyn_cast<mlir::TupleType>(base.getType());
    if (!tupleType)
      return rewriter.notifyMatchFailure(op, "Invalid memref type");

    auto resultType = getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(op, "Unable to convert result type");

    mlir::Location loc = op.getLoc();
    llvm::SmallVector<mlir::OpFoldResult> strides;
    if (mlir::failed(
            materializeStrides(rewriter, loc, memrefType, base, strides)))
      return rewriter.notifyMatchFailure(op, "Unable to materialize strides");

    auto mixedOffsets = mlir::getAsOpFoldResult(adaptor.getIndices());
    auto &&[expr, values] = mlir::computeLinearIndex(rewriter.getIndexAttr(0),
                                                     strides, mixedOffsets);

    mlir::OpFoldResult finalOffset =
        mlir::affine::makeComposedFoldedAffineApply(rewriter, loc, expr,
                                                    values);
    mlir::Value finalOffsetVal =
        mlir::getValueOrCreateConstantIndexOp(rewriter, loc, finalOffset);

    mlir::Type ptrType = tupleType.getType(0);
    mlir::Value zero = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
    mlir::Value ptr =
        rewriter.create<hc::hk::TupleExtractOp>(loc, ptrType, base, zero);
    rewriter.replaceOpWithNewOp<hc::hk::PtrLoadOp>(
        op, resultType, ptr, finalOffsetVal, /*mask*/ nullptr,
        /*pass_thru*/ nullptr);
    return mlir::success();
  }
};

struct ConvertVecStore final
    : mlir::OpConversionPattern<mlir::vector::StoreOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::vector::StoreOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::MemRefType memrefType = op.getMemRefType();
    mlir::Value base = adaptor.getBase();
    auto tupleType = mlir::dyn_cast<mlir::TupleType>(base.getType());
    if (!tupleType)
      return rewriter.notifyMatchFailure(op, "Invalid memref type");

    mlir::Location loc = op.getLoc();
    llvm::SmallVector<mlir::OpFoldResult> strides;
    if (mlir::failed(
            materializeStrides(rewriter, loc, memrefType, base, strides)))
      return rewriter.notifyMatchFailure(op, "Unable to materialize strides");

    auto mixedOffsets = mlir::getAsOpFoldResult(adaptor.getIndices());
    auto &&[expr, values] = mlir::computeLinearIndex(rewriter.getIndexAttr(0),
                                                     strides, mixedOffsets);

    mlir::OpFoldResult finalOffset =
        mlir::affine::makeComposedFoldedAffineApply(rewriter, loc, expr,
                                                    values);
    mlir::Value finalOffsetVal =
        mlir::getValueOrCreateConstantIndexOp(rewriter, loc, finalOffset);

    mlir::Type ptrType = tupleType.getType(0);
    mlir::Value zero = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
    mlir::Value ptr =
        rewriter.create<hc::hk::TupleExtractOp>(loc, ptrType, base, zero);
    rewriter.replaceOpWithNewOp<hc::hk::PtrStoreOp>(
        op, adaptor.getValueToStore(), ptr, finalOffsetVal, /*mask*/ nullptr);
    return mlir::success();
  }
};

struct ConvertMaskedVecLoad final
    : mlir::OpConversionPattern<mlir::vector::MaskedLoadOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::vector::MaskedLoadOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::MemRefType memrefType = op.getMemRefType();
    mlir::Value base = adaptor.getBase();
    auto tupleType = mlir::dyn_cast<mlir::TupleType>(base.getType());
    if (!tupleType)
      return rewriter.notifyMatchFailure(op, "Invalid memref type");

    auto resultType = getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(op, "Unable to convert result type");

    mlir::Location loc = op.getLoc();
    llvm::SmallVector<mlir::OpFoldResult> strides;
    if (mlir::failed(
            materializeStrides(rewriter, loc, memrefType, base, strides)))
      return rewriter.notifyMatchFailure(op, "Unable to materialize strides");

    auto mixedOffsets = mlir::getAsOpFoldResult(adaptor.getIndices());
    auto &&[expr, values] = mlir::computeLinearIndex(rewriter.getIndexAttr(0),
                                                     strides, mixedOffsets);

    mlir::OpFoldResult finalOffset =
        mlir::affine::makeComposedFoldedAffineApply(rewriter, loc, expr,
                                                    values);
    mlir::Value finalOffsetVal =
        mlir::getValueOrCreateConstantIndexOp(rewriter, loc, finalOffset);

    mlir::Type ptrType = tupleType.getType(0);
    mlir::Value zero = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
    mlir::Value ptr =
        rewriter.create<hc::hk::TupleExtractOp>(loc, ptrType, base, zero);
    rewriter.replaceOpWithNewOp<hc::hk::PtrLoadOp>(
        op, resultType, ptr, finalOffsetVal, adaptor.getMask(),
        adaptor.getPassThru());
    return mlir::success();
  }
};

struct ConvertMaskedVecStore final
    : mlir::OpConversionPattern<mlir::vector::MaskedStoreOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::vector::MaskedStoreOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::MemRefType memrefType = op.getMemRefType();
    mlir::Value base = adaptor.getBase();
    auto tupleType = mlir::dyn_cast<mlir::TupleType>(base.getType());
    if (!tupleType)
      return rewriter.notifyMatchFailure(op, "Invalid memref type");

    mlir::Location loc = op.getLoc();
    llvm::SmallVector<mlir::OpFoldResult> strides;
    if (mlir::failed(
            materializeStrides(rewriter, loc, memrefType, base, strides)))
      return rewriter.notifyMatchFailure(op, "Unable to materialize strides");

    auto mixedOffsets = mlir::getAsOpFoldResult(adaptor.getIndices());
    auto &&[expr, values] = mlir::computeLinearIndex(rewriter.getIndexAttr(0),
                                                     strides, mixedOffsets);

    mlir::OpFoldResult finalOffset =
        mlir::affine::makeComposedFoldedAffineApply(rewriter, loc, expr,
                                                    values);
    mlir::Value finalOffsetVal =
        mlir::getValueOrCreateConstantIndexOp(rewriter, loc, finalOffset);

    mlir::Type ptrType = tupleType.getType(0);
    mlir::Value zero = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
    mlir::Value ptr =
        rewriter.create<hc::hk::TupleExtractOp>(loc, ptrType, base, zero);
    rewriter.replaceOpWithNewOp<hc::hk::PtrStoreOp>(
        op, adaptor.getValueToStore(), ptr, finalOffsetVal, adaptor.getMask());
    return mlir::success();
  }
};

struct ConvertDescCast final
    : mlir::OpConversionPattern<hc::hk::MemrefDescriptorCastOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(hc::hk::MemrefDescriptorCastOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    if (op->getNumResults() != 1)
      return rewriter.notifyMatchFailure(op, "Expected 1 result");

    auto dstType = getTypeConverter()->convertType<mlir::TupleType>(
        op.getResult(0).getType());
    if (!dstType)
      return rewriter.notifyMatchFailure(op, "Failed to convert dest type");

    mlir::Location loc = op.getLoc();
    auto cast = rewriter.create<hc::hk::MemrefDescriptorCastOp>(
        loc, dstType.getTypes(), adaptor.getSource());
    rewriter.replaceOpWithNewOp<hc::hk::MakeTupleOp>(op, dstType,
                                                     cast.getResults());
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

    hc::populateFuncPatternsAndTypeConversion(patterns, target, converter);

    patterns
        .insert<ConvertDim, ConvertAlloca, ConvertSubview, ConvertLoad,
                ConvertStore, ConvertVecLoad, ConvertVecStore,
                ConvertMaskedVecLoad, ConvertMaskedVecStore, ConvertDescCast>(
            converter, ctx);

    target.markUnknownOpDynamicallyLegal(
        [&](mlir::Operation *op) -> bool { return converter.isLegal(op); });

    if (mlir::failed(
            mlir::applyFullConversion(mod, target, std::move(patterns))))
      signalPassFailure();
  }
};

struct LegalizeMemrefABIPass final
    : public hc::impl::LegalizeMemrefABIPassBase<LegalizeMemrefABIPass> {

  void runOnOperation() override {
    auto mod = getOperation();

    mlir::OpBuilder builder(&getContext());
    mod->walk([&](mlir::func::FuncOp func) -> mlir::WalkResult {
      if (func.isExternal() ||
          !func->hasAttr(hc::hk::getKernelEntryPointAttrName()))
        return mlir::WalkResult::skip();

      auto funcType = func.getFunctionType();
      llvm::SmallVector<mlir::Type> argTypes =
          llvm::to_vector(funcType.getInputs());

      mlir::Block *entryBlock = &func.getFunctionBody().front();
      builder.setInsertionPointToStart(entryBlock);

      for (auto i : llvm::seq<size_t>(0, argTypes.size())) {
        auto argType = mlir::dyn_cast<mlir::MemRefType>(argTypes[i]);
        if (!argType)
          continue;

        auto newType = hc::hk::MemrefDescriptorType::get(argType);
        argTypes[i] = newType;

        auto arg = entryBlock->getArgument(i);
        arg.setType(newType);
        auto cast = builder.create<hc::hk::MemrefDescriptorCastOp>(
            arg.getLoc(), argType, arg);
        arg.replaceAllUsesExcept(cast.getResult(0), cast);
      }
      func.setType(funcType.clone(argTypes, funcType.getResults()));
      return mlir::WalkResult::skip();
    });
  }
};
} // namespace
