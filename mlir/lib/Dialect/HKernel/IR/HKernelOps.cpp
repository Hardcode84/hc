// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hc/Dialect/HKernel/IR/HKernelOps.hpp"

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

#include "hc/Dialect/HKernel/IR/HKernelOpsDialect.cpp.inc"

#define GET_OP_CLASSES
#include "hc/Dialect/HKernel/IR/HKernelOps.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "hc/Dialect/HKernel/IR/HKernelOpsAttributes.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "hc/Dialect/HKernel/IR/HKernelOpsTypes.cpp.inc"

#include "hc/Dialect/HKernel/IR/HKernelOpsEnums.cpp.inc"
