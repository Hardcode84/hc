// SPDX-FileCopyrightText: 2024 The HC Authors
// SPDX-FileCopyrightText: 2025 The HC Authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hc/Transforms/ConvertPtrToLLVM.hpp"

#include "hc/Dialect/HKernel/IR/HKernelOps.hpp"

#include <mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h>
#include <mlir/Conversion/LLVMCommon/MemRefBuilder.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Transforms/DialectConversion.h>

#include <llvm/ADT/StringSet.h>

static mlir::FailureOr<unsigned>
getPtrAddressSpace(const mlir::TypeConverter &converter, mlir::Attribute attr) {
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

namespace {
struct ConvertGetPyArgOp final
    : public mlir::OpConversionPattern<hc::hk::GetPyArgOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(hc::hk::GetPyArgOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto module = op->getParentOfType<mlir::ModuleOp>();
    if (!module)
      return rewriter.notifyMatchFailure(op, "No parent module");

    mlir::Type i32Type = rewriter.getI32Type();
    auto func = op->getParentOfType<mlir::FunctionOpInterface>();
    if (!func || func.getResultTypes() != mlir::ArrayRef(i32Type))
      return rewriter.notifyMatchFailure(op, "Invalid parent func");

    mlir::Type origType = op.getType();
    mlir::Type resType = getTypeConverter()->convertType(origType);
    if (!resType)
      return rewriter.notifyMatchFailure(op, "Cannot convert result type");

    auto ptrType = mlir::LLVM::LLVMPointerType::get(rewriter.getContext());
    mlir::Location loc = op.getLoc();

    auto getFunction = [&](mlir::StringRef functionName, mlir::Type returnType,
                           mlir::ArrayRef<mlir::Type> argTypes) {
      auto funcType = mlir::LLVM::LLVMFunctionType::get(returnType, argTypes);
      if (auto function =
              module.lookupSymbol<mlir::LLVM::LLVMFuncOp>(functionName))
        return function;

      mlir::OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPointToEnd(module.getBody());
      return rewriter.create<mlir::LLVM::LLVMFuncOp>(rewriter.getUnknownLoc(),
                                                     functionName, funcType);
    };

    mlir::Value argIndex = rewriter.create<mlir::LLVM::ConstantOp>(
        loc, i32Type, adaptor.getIndex());
    mlir::Value ptr = rewriter.create<mlir::LLVM::GEPOp>(
        loc, ptrType, ptrType, adaptor.getArgs(), argIndex,
        /*inbounds*/ true);
    mlir::Value arg = rewriter.create<mlir::LLVM::LoadOp>(loc, ptrType, ptr);

    mlir::Value resPtr = [&] {
      mlir::OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPointToStart(&func.getFunctionBody().front());
      mlir::Value size = rewriter.create<mlir::LLVM::ConstantOp>(
          loc, rewriter.getI64Type(), 1);
      return rewriter.create<mlir::LLVM::AllocaOp>(loc, ptrType, resType, size);
    }();

    mlir::Value convertRes;
    if (auto memrefDesc =
            mlir::dyn_cast<hc::hk::MemrefDescriptorType>(origType)) {
      auto rank =
          mlir::cast<mlir::MemRefType>(memrefDesc.getMemrefType()).getRank();
      mlir::Value rankVal =
          rewriter.create<mlir::LLVM::ConstantOp>(loc, i32Type, rank);
      auto convertArgFunc = getFunction("hcgpuConvertPyArray", i32Type,
                                        {ptrType, ptrType, i32Type, ptrType});
      mlir::Value convertArgs[] = {adaptor.getErrorContext(), arg, rankVal,
                                   resPtr};
      convertRes =
          rewriter.create<mlir::LLVM::CallOp>(loc, convertArgFunc, convertArgs)
              .getResult();
    } else if (auto intType = mlir::dyn_cast<mlir::IntegerType>(origType)) {
      std::string funcName =
          ("hcgpuConvertPyInt" + llvm::Twine(intType.getWidth())).str();
      auto convertArgFunc =
          getFunction(funcName, i32Type, {ptrType, ptrType, ptrType});
      mlir::Value convertArgs[] = {adaptor.getErrorContext(), arg, resPtr};
      convertRes =
          rewriter.create<mlir::LLVM::CallOp>(loc, convertArgFunc, convertArgs)
              .getResult();
    } else {
      return rewriter.notifyMatchFailure(op, "Unsupported return type");
    }

    mlir::Value zero = rewriter.create<mlir::LLVM::ConstantOp>(loc, i32Type, 0);
    mlir::Value success = rewriter.create<mlir::LLVM::ICmpOp>(
        loc, rewriter.getI1Type(), mlir::LLVM::ICmpPredicate::eq, convertRes,
        zero);

    auto &&[successBlock, failureBlock] = [&] {
      mlir::Block *successBlock = rewriter.splitBlock(
          rewriter.getBlock(), rewriter.getInsertionPoint());
      mlir::Block *failureBlock = rewriter.createBlock(successBlock);
      rewriter.create<mlir::LLVM::ReturnOp>(loc, convertRes);
      return std::pair(successBlock, failureBlock);
    }();

    rewriter.setInsertionPointAfter(success.getDefiningOp());
    rewriter.create<mlir::LLVM::CondBrOp>(loc, success, successBlock,
                                          failureBlock);
    mlir::OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointToStart(successBlock);
    mlir::Value result =
        rewriter.create<mlir::LLVM::LoadOp>(loc, resType, resPtr);
    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};
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

    llvm::SmallVector<mlir::Type> resTypes;
    if (mlir::failed(
            getTypeConverter()->convertTypes(op.getResultTypes(), resTypes)))
      return rewriter.notifyMatchFailure(op, "Failed convert result types");

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
      if (mlir::failed(memrefType.getStridesAndOffset(strides, offset)))
        return rewriter.notifyMatchFailure(op, "Failed to get strides");

      for (auto &&[i, s] : llvm::enumerate(strides)) {
        if (!mlir::ShapedType::isDynamic(s))
          continue;

        results.emplace_back(desc.stride(rewriter, loc, i));
      }
    }

    if (results.size() != resTypes.size())
      return rewriter.notifyMatchFailure(op, "Results count mismatch");

    rewriter.replaceOp(op, results);
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

static unsigned getAlignment(mlir::Type type) {
  type = mlir::getElementTypeOrSelf(type);
  if (mlir::isa<mlir::IntegerType, mlir::FloatType>(type))
    return type.getIntOrFloatBitWidth() / 8;

  // TODO: check datalayout.
  return 4;
}

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
      unsigned align = getAlignment(resType);
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

      unsigned align = getAlignment(elemType);
      rewriter.replaceOpWithNewOp<mlir::LLVM::MaskedStoreOp>(op, value, src,
                                                             mask, align);
    } else {
      rewriter.replaceOpWithNewOp<mlir::LLVM::StoreOp>(op, value, src);
    }
    return llvm::success();
  }
};

static mlir::LLVM::GlobalOp getDynamicSharedMemorySymbol(
    mlir::OpBuilder &rewriter, mlir::gpu::GPUModuleOp moduleOp,
    hc::hk::PtrDynamicSharedMemOp op, const mlir::TypeConverter *typeConverter,
    unsigned alignmentBit) {
  uint64_t alignmentByte = alignmentBit / 8;

  auto ptrType = mlir::cast<hc::hk::PtrType>(op.getType());
  mlir::FailureOr<unsigned> addressSpace =
      getPtrAddressSpace(*typeConverter, ptrType.getMemorySpace());
  if (failed(addressSpace))
    return nullptr;

  llvm::StringSet<> existingGlobalNames;
  for (auto globalOp : moduleOp.getBody()->getOps<mlir::LLVM::GlobalOp>()) {
    existingGlobalNames.insert(globalOp.getSymName());
    if (auto arrayType =
            mlir::dyn_cast<mlir::LLVM::LLVMArrayType>(globalOp.getType())) {
      if (globalOp.getAddrSpace() == addressSpace.value() &&
          arrayType.getNumElements() == 0 &&
          globalOp.getAlignment().value_or(0) == alignmentByte) {
        return globalOp;
      }
    }
  }

  unsigned uniquingCounter = 0;
  llvm::SmallString<128> symName = mlir::SymbolTable::generateSymbolName<128>(
      "__dynamic_shmem_",
      [&](mlir::StringRef candidate) {
        return existingGlobalNames.contains(candidate);
      },
      uniquingCounter);

  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(moduleOp.getBody());

  auto zeroSizedArrayType = mlir::LLVM::LLVMArrayType::get(
      typeConverter->convertType(ptrType.getElementType()), 0);

  return rewriter.create<mlir::LLVM::GlobalOp>(
      op->getLoc(), zeroSizedArrayType, /*isConstant=*/false,
      mlir::LLVM::Linkage::Internal, symName, /*value=*/mlir::Attribute(),
      alignmentByte, addressSpace.value());
}

struct ConvertDynamicSHmem final
    : public mlir::OpConversionPattern<hc::hk::PtrDynamicSharedMemOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(hc::hk::PtrDynamicSharedMemOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto moduleOp = op->getParentOfType<mlir::gpu::GPUModuleOp>();
    if (!moduleOp)
      return rewriter.notifyMatchFailure(op, "No parent module");

    mlir::LLVM::GlobalOp shmemOp = getDynamicSharedMemorySymbol(
        rewriter, moduleOp, op, getTypeConverter(), 0);
    if (!shmemOp)
      return rewriter.notifyMatchFailure(op, "Failed to create shmem global");

    mlir::Location loc = op.getLoc();
    auto basePtr = rewriter.create<mlir::LLVM::AddressOfOp>(loc, shmemOp);
    mlir::Type baseType = basePtr->getResultTypes().front();

    auto elementType =
        mlir::cast<hc::hk::PtrType>(op.getType()).getElementType();
    mlir::LLVM::GEPArg gepArgs[] = {0};
    mlir::Value shmemPtr = rewriter.create<mlir::LLVM::GEPOp>(
        loc, baseType, elementType, basePtr, gepArgs);
    rewriter.replaceOp(op, shmemPtr);
    return mlir::success();
  }
};

struct ConvertPtrCast final
    : public mlir::OpConversionPattern<hc::hk::PtrCastOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(hc::hk::PtrCastOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto resType = getTypeConverter()->convertType(op.getType());
    if (!resType)
      return rewriter.notifyMatchFailure(op, "Failed to convert result type");

    mlir::Value src = adaptor.getValue();
    if (resType == src.getType()) {
      rewriter.replaceOp(op, src);
      return mlir::success();
    }

    return mlir::failure();
  }
};
} // namespace

void hc::populatePtrToLLVMTypeConverter(mlir::LLVMTypeConverter &converter) {
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

  auto ptrType = mlir::LLVM::LLVMPointerType::get(&converter.getContext());
  converter.addConversion(
      [ptrType](hc::hk::CurrentGroupType) -> std::optional<mlir::Type> {
        return ptrType;
      });
  converter.addConversion(
      [ptrType](hc::hk::ErrorContextType) -> std::optional<mlir::Type> {
        return ptrType;
      });
  converter.addConversion(
      [ptrType](hc::hk::PyArgsType) -> std::optional<mlir::Type> {
        return ptrType;
      });
}

void hc::populatePtrToLLVMConversionPatterns(
    mlir::LLVMTypeConverter &converter, mlir::RewritePatternSet &patterns) {
  patterns.insert<ConvertGetPyArgOp, ConvertDescriptorCast, ConvertPtrAdd,
                  ConvertPtrAlloca, ConvertPtrLoad, ConvertPtrStore,
                  ConvertDynamicSHmem, ConvertPtrCast>(converter,
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
    hc::populatePtrToLLVMTypeConverter(typeConverter);
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
