// SPDX-FileCopyrightText: 2024 The HC Authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hc/Transforms/ConvertPtrToLLVM.hpp"

#include "hc/Dialect/HKernel/IR/HKernelOps.hpp"

#include <mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Transforms/DialectConversion.h>

namespace {
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

  patterns
      .insert<ConvertPtrAdd, ConvertPtrAlloca, ConvertPtrLoad, ConvertPtrStore>(
          converter, patterns.getContext());
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
