// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hc/Dialect/HKernel/IR/HKernelOps.hpp"

#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>

#include <llvm/ADT/TypeSwitch.h>

void hc::hk::HKernelDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "hc/Dialect/HKernel/IR/HKernelOps.cpp.inc"
      >();

  addTypes<
#define GET_TYPEDEF_LIST
#include "hc/Dialect/HKernel/IR/HKernelOpsTypes.cpp.inc"
      >();

  addAttributes<
#define GET_ATTRDEF_LIST
#include "hc/Dialect/HKernel/IR/HKernelOpsAttributes.cpp.inc"
      >();
}

hc::hk::SymbolicallyShapedType
hc::hk::BufferType::cloneWith(std::optional<llvm::ArrayRef<mlir::Type>> shape,
                              mlir::Type elementType) const {
  return BufferType::get(getContext(), shape ? *shape : getShape(),
                         elementType ? elementType : getElementType());
}

#include "hc/Dialect/HKernel/IR/HKernelOpsDialect.cpp.inc"

#include "hc/Dialect/HKernel/IR/HKernelOpsTypeInterfaces.cpp.inc"

#define GET_OP_CLASSES
#include "hc/Dialect/HKernel/IR/HKernelOps.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "hc/Dialect/HKernel/IR/HKernelOpsAttributes.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "hc/Dialect/HKernel/IR/HKernelOpsTypes.cpp.inc"

#include "hc/Dialect/HKernel/IR/HKernelOpsEnums.cpp.inc"
