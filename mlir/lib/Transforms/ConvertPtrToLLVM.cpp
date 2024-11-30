// SPDX-FileCopyrightText: 2024 The HC Authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hc/Transforms/ConvertPtrToLLVM.hpp"

#include "hc/Dialect/HKernel/IR/HKernelOps.hpp"

#include <mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>

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
