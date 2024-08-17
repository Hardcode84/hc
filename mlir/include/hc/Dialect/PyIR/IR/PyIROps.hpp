// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "mlir/IR/SymbolTable.h"
#include <mlir/Bytecode/BytecodeOpInterface.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>
#include <mlir/Interfaces/FunctionInterfaces.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

#include "hc/Dialect/Typing/IR/TypingOpsInterfaces.hpp"

#include "hc/Dialect/PyIR/IR/PyIROpsDialect.h.inc"
#include "hc/Dialect/PyIR/IR/PyIROpsEnums.h.inc"

#define GET_TYPEDEF_CLASSES
#include "hc/Dialect/PyIR/IR/PyIROpsTypes.h.inc"

#define GET_ATTRDEF_CLASSES
#include "hc/Dialect/PyIR/IR/PyIROpsAttributes.h.inc"

#define GET_OP_CLASSES
#include "hc/Dialect/PyIR/IR/PyIROps.h.inc"
