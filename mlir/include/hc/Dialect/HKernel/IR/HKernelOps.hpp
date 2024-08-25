// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <mlir/IR/Dialect.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/OpImplementation.h>

#include "hc/Dialect/HKernel/IR/HKernelOpsTypeInterfaces.h.inc"

#include "hc/Dialect/HKernel/IR/HKernelOpsDialect.h.inc"
#include "hc/Dialect/HKernel/IR/HKernelOpsEnums.h.inc"

#define GET_TYPEDEF_CLASSES
#include "hc/Dialect/HKernel/IR/HKernelOpsTypes.h.inc"

#define GET_ATTRDEF_CLASSES
#include "hc/Dialect/HKernel/IR/HKernelOpsAttributes.h.inc"

#define GET_OP_CLASSES
#include "hc/Dialect/HKernel/IR/HKernelOps.h.inc"
