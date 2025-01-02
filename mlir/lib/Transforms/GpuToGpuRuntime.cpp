// SPDX-FileCopyrightText: 2024 The HC Authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hc/Transforms/Passes.hpp"

#include "hc/Dialect/HKernel/IR/HKernelOps.hpp"
#include "hc/Transforms/ConvertPtrToLLVM.hpp"

#include <mlir/Conversion/LLVMCommon/ConversionTarget.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Transforms/DialectConversion.h>

namespace hc {
#define GEN_PASS_DEF_GPUTOGPURUNTIME
#include "hc/Transforms/Passes.h.inc"
} // namespace hc

namespace {
static std::string getUniqueLLVMGlobalName(mlir::ModuleOp mod,
                                           llvm::Twine srcName) {
  for (int i = 0;; ++i) {
    auto name =
        (i == 0 ? srcName.str() : (srcName + "_" + llvm::Twine(i)).str());
    if (!mod.lookupSymbol(name))
      return name;
  }
}

using namespace mlir;

static gpu::ObjectAttr getSelectedObject(gpu::BinaryOp op) {
  ArrayRef<Attribute> objects = op.getObjectsAttr().getValue();

  // Obtain the index of the object to select.
  int64_t index = -1;
  if (Attribute target =
          cast<gpu::SelectObjectAttr>(op.getOffloadingHandlerAttr())
              .getTarget()) {
    // If the target attribute is a number it is the index. Otherwise compare
    // the attribute to every target inside the object array to find the index.
    if (auto indexAttr = mlir::dyn_cast<IntegerAttr>(target)) {
      index = indexAttr.getInt();
    } else {
      for (auto [i, attr] : llvm::enumerate(objects)) {
        auto obj = mlir::dyn_cast<gpu::ObjectAttr>(attr);
        if (obj.getTarget() == target) {
          index = i;
        }
      }
    }
  } else {
    // If the target attribute is null then it's selecting the first object in
    // the object array.
    index = 0;
  }

  if (index < 0 || index >= static_cast<int64_t>(objects.size())) {
    op->emitError("the requested target object couldn't be found");
    return nullptr;
  }
  return mlir::dyn_cast<gpu::ObjectAttr>(objects[index]);
}

static mlir::Value createGlobal(mlir::OpBuilder &builder, mlir::Type globaType,
                                mlir::ModuleOp mod, mlir::StringRef name) {
  mlir::Type ptrType = mlir::LLVM::LLVMPointerType::get(builder.getContext());
  mlir::Location loc = builder.getUnknownLoc();
  mlir::LLVM::GlobalOp handle;
  {
    OpBuilder::InsertionGuard g(builder);
    builder.setInsertionPointToEnd(mod.getBody());
    auto handleName = getUniqueLLVMGlobalName(mod, "kernel_handle");
    handle = builder.create<mlir::LLVM::GlobalOp>(
        loc, globaType, /*isConstant*/ false, mlir::LLVM::Linkage::Internal,
        handleName, mlir::Attribute());
  }
  return builder.create<mlir::LLVM::AddressOfOp>(loc, ptrType,
                                                 handle.getSymName());
}

static mlir::Value allocArray(mlir::OpBuilder &builder, mlir::Location loc,
                              mlir::Type elemType, mlir::ValueRange values) {
  auto arrayType = mlir::LLVM::LLVMArrayType::get(elemType, values.size());
  mlir::Value array = builder.create<mlir::LLVM::PoisonOp>(loc, arrayType);
  for (auto &&[i, val] : llvm::enumerate(values))
    array = builder.create<mlir::LLVM::InsertValueOp>(loc, array, val, i);

  auto ptrType = mlir::LLVM::LLVMPointerType::get(builder.getContext());
  mlir::Value size = builder.create<mlir::LLVM::ConstantOp>(
      loc, builder.getI64Type(), values.size());
  mlir::Value res =
      builder.create<LLVM::AllocaOp>(loc, ptrType, elemType, size, 0);
  builder.create<mlir::LLVM::StoreOp>(loc, array, res);
  return res;
}

struct FunctionCallBuilder {
  FunctionCallBuilder(StringRef functionName, Type returnType,
                      ArrayRef<Type> argumentTypes)
      : functionName(functionName),
        functionType(LLVM::LLVMFunctionType::get(returnType, argumentTypes)) {}
  LLVM::CallOp create(Location loc, OpBuilder &builder,
                      ArrayRef<Value> arguments) const {
    auto module = builder.getBlock()->getParent()->getParentOfType<ModuleOp>();
    auto function = [&] {
      OpBuilder::InsertionGuard g(builder);
      builder.setInsertionPointToEnd(module.getBody());
      if (auto function = module.lookupSymbol<LLVM::LLVMFuncOp>(functionName))
        return function;
      return builder.create<LLVM::LLVMFuncOp>(loc, functionName, functionType);
    }();
    return builder.create<LLVM::CallOp>(loc, function, arguments);
  }

  StringRef functionName;
  LLVM::LLVMFunctionType functionType;
};

struct ConvertGpuLaunch final
    : public mlir::OpConversionPattern<mlir::gpu::LaunchFuncOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::gpu::LaunchFuncOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto mod = op->getParentOfType<mlir::ModuleOp>();
    if (!mod)
      return rewriter.notifyMatchFailure(op, "No parent module");

    if (op->getNumResults() != 0)
      return rewriter.notifyMatchFailure(op, "Cannot replace op with results");

    auto kernelBinary = SymbolTable::lookupNearestSymbolFrom<gpu::BinaryOp>(
        op, op.getKernelModuleName());
    if (!kernelBinary)
      return rewriter.notifyMatchFailure(
          op, "Couldn't find the binary holding the kernel: " +
                  op.getKernelModuleName().getValue());

    gpu::ObjectAttr object = getSelectedObject(kernelBinary);
    if (!object)
      return rewriter.notifyMatchFailure(op, "Cannot get gpu object");

    StringRef objData = object.getObject();

    mlir::Location loc = op.getLoc();
    mlir::Value handlePtr =
        createGlobal(rewriter, llvmPointerType, mod, "kernel_handle");
    mlir::Value dataPtr = mlir::LLVM::createGlobalString(
        loc, rewriter, getUniqueLLVMGlobalName(mod, "kernel_data"), objData,
        mlir::LLVM::Linkage::Internal);
    mlir::Value dataSize = rewriter.create<mlir::LLVM::ConstantOp>(
        loc, llvmIndexType, objData.size());

    mlir::Value kernel =
        getKernelBuilder.create(loc, rewriter, {handlePtr, dataPtr, dataSize})
            .getResult();

    mlir::SmallVector<mlir::Value> suggestedBlockSizes;

    auto createConst = [&](int64_t val) -> mlir::Value {
      return rewriter.create<mlir::LLVM::ConstantOp>(
          loc, llvmIndexType, rewriter.getI64IntegerAttr(val));
    };

    auto createAlloca = [&](mlir::Type elemType, int64_t size) -> mlir::Value {
      mlir::Value sizeVal = createConst(size);
      return rewriter.create<LLVM::AllocaOp>(loc, llvmPointerType, elemType,
                                             sizeVal, 0);
    };

    constexpr unsigned ndims = 3;
    mlir::Value ndimsVal = createConst(ndims);
    auto sizesType = mlir::LLVM::LLVMArrayType::get(llvmIndexType, ndims);
    mlir::Value blockSizesOrig[ndims] = {op.getBlockSizeX(), op.getBlockSizeY(),
                                         op.getBlockSizeZ()};
    mlir::Value blockSizesBefore[ndims] = {adaptor.getBlockSizeX(),
                                           adaptor.getBlockSizeY(),
                                           adaptor.getBlockSizeZ()};
    mlir::Value blockSizes[ndims];
    for (auto &&[i, origArg, arg] :
         llvm::enumerate(blockSizesOrig, blockSizesBefore)) {
      auto suggested = origArg.getDefiningOp<hc::hk::SuggestBlockSizeOp>();
      if (!suggested) {
        blockSizes[i] = arg;
        continue;
      }

      if (suggestedBlockSizes.empty()) {
        mlir::Value globalSizesArray =
            rewriter.create<mlir::LLVM::PoisonOp>(loc, sizesType);
        mlir::ValueRange workSize = suggested.getWorkSize();
        for (auto i : llvm::seq(0u, ndims)) {
          mlir::Value val;
          if (i < workSize.size()) {
            val = rewriter.getRemappedValue(workSize[i]);
            if (!val)
              return rewriter.notifyMatchFailure(op,
                                                 "Cannot get remapped value");
          } else {
            val = createConst(1);
          }

          globalSizesArray = rewriter.create<mlir::LLVM::InsertValueOp>(
              loc, globalSizesArray, val, i);
        }
        mlir::Value globalSizes = createAlloca(llvmIndexType, ndims);
        rewriter.create<mlir::LLVM::StoreOp>(loc, globalSizesArray,
                                             globalSizes);

        mlir::Value blockSizes = createAlloca(llvmIndexType, ndims);
        suggestBlockSizeBuilder.create(
            loc, rewriter, {kernel, globalSizes, blockSizes, ndimsVal});
        mlir::Value blockSizeArray =
            rewriter.create<mlir::LLVM::LoadOp>(loc, sizesType, blockSizes);

        suggestedBlockSizes.resize(ndims);
        for (auto i : llvm::seq(0u, ndims)) {
          mlir::Value blockSize = rewriter.create<mlir::LLVM::ExtractValueOp>(
              loc, llvmIndexType, blockSizeArray, i);
          suggestedBlockSizes[i] = blockSize;
        }
      }

      blockSizes[i] = suggestedBlockSizes[i];
    }

    mlir::Value gridSizes[ndims] = {
        adaptor.getGridSizeX(), adaptor.getGridSizeY(), adaptor.getGridSizeZ()};

    mlir::Value gridSizesPtr =
        allocArray(rewriter, loc, llvmIndexType, gridSizes);
    mlir::Value blockSizesPtr =
        allocArray(rewriter, loc, llvmIndexType, blockSizes);

    mlir::ValueRange args = adaptor.getKernelOperands();

    auto argsPtrArrayType =
        mlir::LLVM::LLVMArrayType::get(llvmPointerType, args.size());
    mlir::Value argsArray =
        rewriter.create<mlir::LLVM::PoisonOp>(loc, argsPtrArrayType);
    for (auto &&[i, arg] : llvm::enumerate(args)) {
      mlir::Value argData = createAlloca(arg.getType(), 1);
      rewriter.create<mlir::LLVM::StoreOp>(loc, arg, argData);
      argsArray = rewriter.create<mlir::LLVM::InsertValueOp>(loc, argsArray,
                                                             argData, i);
    }
    mlir::Value argsArrayPtr = createAlloca(llvmPointerType, args.size());
    rewriter.create<mlir::LLVM::StoreOp>(loc, argsArray, argsArrayPtr);

    mlir::Value nargs = createConst(args.size());
    mlir::Value sharedMemSize = adaptor.getDynamicSharedMemorySize();
    if (!sharedMemSize)
      sharedMemSize = rewriter.create<mlir::LLVM::ConstantOp>(
          loc, llvmIndexType, rewriter.getI64IntegerAttr(0));

    mlir::Value params[] = {
        kernel,       gridSizesPtr, blockSizesPtr, ndimsVal,
        argsArrayPtr, nargs,        sharedMemSize,
    };

    launchKernelBuilder.create(loc, rewriter, params);
    rewriter.eraseOp(op);
    return mlir::success();
  }

private:
  MLIRContext *context = this->getContext();
  Type llvmVoidType = LLVM::LLVMVoidType::get(context);
  Type llvmPointerType = LLVM::LLVMPointerType::get(context);
  Type llvmIndexType =
      this->getTypeConverter()->convertType(mlir::IndexType::get(context));

  FunctionCallBuilder getKernelBuilder = {"hcgpuGetKernel",
                                          llvmPointerType,
                                          {
                                              llvmPointerType, // globalHandle
                                              llvmPointerType, // data
                                              llvmIndexType,   // data size
                                          }};
  FunctionCallBuilder suggestBlockSizeBuilder = {
      "hcgpuSuggestBlockSize",
      llvmVoidType,
      {
          llvmPointerType, // kernel
          llvmPointerType, // global sizes
          llvmPointerType, // block sizes ret
          llvmIndexType,   // ndim
      }};
  FunctionCallBuilder launchKernelBuilder = {
      "hcgpuLaunchKernel",
      llvmVoidType,
      {
          llvmPointerType, // kernel
          llvmPointerType, // grid sizes
          llvmPointerType, // blocks sizes
          llvmIndexType,   // ndim
          llvmPointerType, // args
          llvmIndexType,   // nargs
          llvmIndexType,   // shared mem size
      }};
};

struct GPUToGPURuntime final
    : public hc::impl::GPUToGPURuntimeBase<GPUToGPURuntime> {

  void runOnOperation() override {
    auto mod = getOperation();

    auto *ctx = &getContext();
    mlir::LLVMConversionTarget target(*ctx);
    mlir::LLVMTypeConverter converter(ctx);
    mlir::RewritePatternSet patterns(ctx);

    hc::populatePtrToLLVMTypeConverter(converter);

    target.addIllegalOp<mlir::gpu::LaunchFuncOp>();
    patterns.insert<ConvertGpuLaunch>(converter, ctx);

    if (mlir::failed(
            mlir::applyPartialConversion(mod, target, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace
