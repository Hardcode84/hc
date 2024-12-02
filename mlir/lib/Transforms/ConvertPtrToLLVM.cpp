// SPDX-FileCopyrightText: 2024 The HC Authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hc/Transforms/ConvertPtrToLLVM.hpp"

#include "hc/Dialect/HKernel/IR/HKernelOps.hpp"

#include <mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h>
#include <mlir/Conversion/LLVMCommon/MemRefBuilder.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Transforms/DialectConversion.h>

namespace {
struct ConvertDescriptorCast final
    : public mlir::OpConversionPattern<hc::hk::MemrefDescriptorCastOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(hc::hk::MemrefDescriptorCastOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto origSrcType =
        mlir::dyn_cast<hc::hk::MemrefDescriptorType>(op.getSource().getType());
    if (!origSrcType ||
        !mlir::isa<mlir::MemRefType>(origSrcType.getMemrefType()))
      return rewriter.notifyMatchFailure(op, "Invalid src type");

    auto origDstType = mlir::dyn_cast<mlir::TupleType>(op.getType());
    if (!origDstType)
      return rewriter.notifyMatchFailure(op, "Invalid dst type");

    auto resType =
        getTypeConverter()->convertType<mlir::TupleType>(origDstType);
    if (!resType)
      return rewriter.notifyMatchFailure(op, "Cannot convert result type");

    auto memrefType = mlir::cast<mlir::MemRefType>(origSrcType.getMemrefType());

    mlir::MemRefDescriptor desc(adaptor.getSource());

    mlir::Location loc = op.getLoc();
    mlir::Value ptr = desc.alignedPtr(rewriter, loc);
    mlir::Value offset = desc.offset(rewriter, loc);
    ptr = rewriter.create<mlir::LLVM::GEPOp>(loc, ptr.getType(),
                                             rewriter.getI8Type(), ptr, offset,
                                             /*inbounds*/ true);

    llvm::SmallVector<mlir::Value> results;
    results.emplace_back(ptr);
    for (auto &&[i, s] : llvm::enumerate(memrefType.getShape())) {
      if (!mlir::ShapedType::isDynamic(s))
        continue;

      results.emplace_back(desc.size(rewriter, loc, i));
    }

    if (!memrefType.getLayout().isIdentity()) {
      int64_t offset; // unused
      llvm::SmallVector<int64_t> strides;
      if (mlir::failed(mlir::getStridesAndOffset(memrefType, strides, offset)))
        return rewriter.notifyMatchFailure(op, "Failed to get strides");

      for (auto &&[i, s] : llvm::enumerate(strides)) {
        if (!mlir::ShapedType::isDynamic(s))
          continue;

        results.emplace_back(desc.stride(rewriter, loc, i));
      }
    }

    rewriter.replaceOpWithNewOp<hc::hk::MakeTupleOp>(op, resType, results);
    return mlir::success();
  }
};

struct ConvertPtrAdd final
    : public mlir::OpConversionPattern<hc::hk::PtrAddOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(hc::hk::PtrAddOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Value src = adaptor.getBase();
    auto srcType = mlir::dyn_cast<mlir::LLVM::LLVMPointerType>(src.getType());
    if (!srcType)
      return rewriter.notifyMatchFailure(op, "Invalid src type");

    auto converter = getTypeConverter();
    auto opType = mlir::cast<hc::hk::PtrType>(op.getType());
    auto dstType = converter->convertType<mlir::LLVM::LLVMPointerType>(opType);
    if (!dstType)
      return rewriter.notifyMatchFailure(op, "Invalid dst type");

    mlir::Type elemType = converter->convertType(opType.getElementType());
    if (!elemType)
      return rewriter.notifyMatchFailure(op, "Invalid element type");

    mlir::Value offset = adaptor.getOffset();

    rewriter.replaceOpWithNewOp<mlir::LLVM::GEPOp>(op, dstType, elemType, src,
                                                   offset);
    return mlir::success();
  }
};

struct ConvertPtrAlloca final
    : public mlir::OpConversionPattern<hc::hk::PtrAllocaOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(hc::hk::PtrAllocaOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto converter = getTypeConverter();
    auto opType = mlir::cast<hc::hk::PtrType>(op.getType());
    auto resType = converter->convertType(opType);
    if (!resType)
      return rewriter.notifyMatchFailure(op, "Invalid result type");

    mlir::Type elemType = converter->convertType(opType.getElementType());
    if (!elemType)
      return rewriter.notifyMatchFailure(op, "Invalid element type");

    rewriter.replaceOpWithNewOp<mlir::LLVM::AllocaOp>(op, resType, elemType,
                                                      adaptor.getSize());
    return mlir::success();
  }
};

struct ConvertPtrLoad final
    : public mlir::OpConversionPattern<hc::hk::PtrLoadOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(hc::hk::PtrLoadOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Value src = adaptor.getBase();
    auto srcType = mlir::dyn_cast<mlir::LLVM::LLVMPointerType>(src.getType());
    if (!srcType)
      return rewriter.notifyMatchFailure(op, "Invalid src type");

    auto converter = getTypeConverter();
    auto resType = converter->convertType(op.getType());
    if (!resType)
      return rewriter.notifyMatchFailure(op, "Invalid result type");

    if (mlir::Value offset = adaptor.getOffset()) {
      mlir::Type elemType = converter->convertType(
          mlir::cast<hc::hk::PtrType>(op.getBase().getType()).getElementType());
      if (!elemType)
        return rewriter.notifyMatchFailure(op, "Invalid element type");

      if (mlir::isa<mlir::IntegerType>(offset.getType())) {
        src = rewriter.create<mlir::LLVM::GEPOp>(op.getLoc(), srcType, elemType,
                                                 src, offset);
      } else {
        return rewriter.notifyMatchFailure(op, "Invalid offset type");
      }
    }

    if (mlir::Value mask = adaptor.getMask()) {
      // TODO: Annotate ptrs with alignment
      unsigned align =
          mlir::getElementTypeOrSelf(resType).getIntOrFloatBitWidth() / 8;
      mlir::Value passThru = adaptor.getPassThru();
      rewriter.replaceOpWithNewOp<mlir::LLVM::MaskedLoadOp>(
          op, resType, src, mask, passThru, align);
    } else {
      rewriter.replaceOpWithNewOp<mlir::LLVM::LoadOp>(op, resType, src);
    }
    return llvm::success();
  }
};

struct ConvertPtrStore final
    : public mlir::OpConversionPattern<hc::hk::PtrStoreOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(hc::hk::PtrStoreOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Value src = adaptor.getBase();
    auto srcType = mlir::dyn_cast<mlir::LLVM::LLVMPointerType>(src.getType());
    if (!srcType)
      return rewriter.notifyMatchFailure(op, "Invalid src type");

    auto converter = getTypeConverter();

    if (mlir::Value offset = adaptor.getOffset()) {
      mlir::Type elemType = converter->convertType(
          mlir::cast<hc::hk::PtrType>(op.getBase().getType()).getElementType());
      if (!elemType)
        return rewriter.notifyMatchFailure(op, "Invalid element type");

      if (mlir::isa<mlir::IntegerType>(offset.getType())) {
        src = rewriter.create<mlir::LLVM::GEPOp>(op.getLoc(), srcType, elemType,
                                                 src, offset);
      } else {
        return rewriter.notifyMatchFailure(op, "Invalid offset type");
      }
    }

    mlir::Value value = adaptor.getValue();
    if (mlir::Value mask = adaptor.getMask()) {
      // TODO: Annotate ptrs with alignment
      mlir::Type elemType =
          mlir::cast<hc::hk::PtrType>(op.getBase().getType()).getElementType();
      elemType = converter->convertType(elemType);
      if (!elemType)
        return rewriter.notifyMatchFailure(op, "Invalid element type");

      unsigned align =
          mlir::getElementTypeOrSelf(elemType).getIntOrFloatBitWidth() / 8;
      rewriter.replaceOpWithNewOp<mlir::LLVM::MaskedStoreOp>(op, value, src,
                                                             mask, align);
    } else {
      rewriter.replaceOpWithNewOp<mlir::LLVM::StoreOp>(op, value, src);
    }
    return llvm::success();
  }
};
} // namespace

static mlir::FailureOr<unsigned>
getPtrAddressSpace(mlir::LLVMTypeConverter &converter, mlir::Attribute attr) {
  if (!attr)
    return 0;

  // Fake memref type.
  auto type =
      mlir::MemRefType::get(1, mlir::IntegerType::get(attr.getContext(), 32),
                            mlir::MemRefLayoutAttrInterface{}, attr);
  std::optional<mlir::Attribute> converted =
      converter.convertTypeAttribute(type, attr);
  if (!converted)
    return mlir::failure();
  if (!(*converted)) // Conversion to default is 0.
    return 0;
  if (auto explicitSpace =
          mlir::dyn_cast_if_present<mlir::IntegerAttr>(*converted)) {
    if (explicitSpace.getType().isIndex() ||
        explicitSpace.getType().isSignlessInteger())
      return explicitSpace.getInt();
  }
  return mlir::failure();
}

void hc::populatePtrToLLVMConversionPatterns(
    mlir::LLVMTypeConverter &converter, mlir::RewritePatternSet &patterns) {
  converter.addConversion(
      [&converter](hc::hk::PtrType type) -> std::optional<mlir::Type> {
        auto addrSpace = getPtrAddressSpace(converter, type.getMemorySpace());
        if (mlir::failed(addrSpace))
          return std::nullopt;

        return mlir::LLVM::LLVMPointerType::get(type.getContext(), *addrSpace);
      });

  converter.addConversion([&converter](hc::hk::MemrefDescriptorType type)
                              -> std::optional<mlir::Type> {
    auto innerType = converter.convertType(type.getMemrefType());
    if (!innerType)
      return std::nullopt;

    return innerType;
  });

  converter.addConversion(
      [&converter](mlir::TupleType type) -> std::optional<mlir::Type> {
        llvm::SmallVector<mlir::Type> types(type.size());
        for (auto &&[i, t] : llvm::enumerate(type.getTypes())) {
          auto converted = converter.convertType(t);
          if (!converted)
            return std::nullopt;

          types[i] = converted;
        }

        return mlir::TupleType::get(type.getContext(), types);
      });
  patterns.insert<ConvertDescriptorCast, ConvertPtrAdd, ConvertPtrAlloca,
                  ConvertPtrLoad, ConvertPtrStore>(converter,
                                                   patterns.getContext());
}

namespace {
using namespace mlir;

struct PtrToLLVMDialectInterface : public ConvertToLLVMPatternInterface {
  using ConvertToLLVMPatternInterface::ConvertToLLVMPatternInterface;
  void loadDependentDialects(MLIRContext *context) const final {
    context->loadDialect<mlir::LLVM::LLVMDialect>();
  }

  void populateConvertToLLVMConversionPatterns(
      ConversionTarget &target, LLVMTypeConverter &typeConverter,
      RewritePatternSet &patterns) const final {
    hc::populatePtrToLLVMConversionPatterns(typeConverter, patterns);
  }
};
} // namespace

void hc::registerConvertPtrToLLVMInterface(mlir::DialectRegistry &registry) {
  registry.addExtension(
      +[](mlir::MLIRContext *ctx, hc::hk::HKernelDialect *dialect) {
        dialect->addInterfaces<PtrToLLVMDialectInterface>();
      });
}
