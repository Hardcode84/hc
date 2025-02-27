// SPDX-FileCopyrightText: 2024 The HC Authors
// SPDX-FileCopyrightText: 2025 The HC Authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <mlir/Bytecode/BytecodeOpInterface.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/Interfaces/CastInterfaces.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>
#include <mlir/Interfaces/InferIntRangeInterface.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Interfaces/ViewLikeInterface.h>

namespace mlir {
class RewriterBase;
}

#include "hc/Dialect/Typing/IR/TypingOpsInterfaces.hpp"

#include "hc/Dialect/HKernel/IR/HKernelOpsTypeInterfaces.h.inc"

#include "hc/Dialect/HKernel/IR/HKernelOpsDialect.h.inc"
#include "hc/Dialect/HKernel/IR/HKernelOpsEnums.h.inc"

#define GET_TYPEDEF_CLASSES
#include "hc/Dialect/HKernel/IR/HKernelOpsTypes.h.inc"

#define GET_ATTRDEF_CLASSES
#include "hc/Dialect/HKernel/IR/HKernelOpsAttributes.h.inc"

#define GET_OP_CLASSES
#include "hc/Dialect/HKernel/IR/HKernelOps.h.inc"

namespace hc::hk {
llvm::StringRef getKernelEntryPointAttrName();
llvm::StringRef getKernelWorkShapeAttrName();
llvm::StringRef getKernelGroupShapeAttrName();
llvm::StringRef getKernelGroupCountAttrName();
llvm::StringRef getKernelGroupIdAttrName();
llvm::StringRef getKernelLocalIdAttrName();
llvm::StringRef getKernelSubgroupSizeAttrName();
llvm::StringRef getKernelSubgroupIdAttrName();

llvm::StringRef getKernelMetadataAttrName();

llvm::StringRef getKernelBackendAttrName();
llvm::StringRef getKernelDeviceAttrName();

llvm::StringRef getKernelAllocExpandAttrName();
} // namespace hc::hk
