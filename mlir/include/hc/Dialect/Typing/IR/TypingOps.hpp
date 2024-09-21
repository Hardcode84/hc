// SPDX-FileCopyrightText: 2024 The HC Authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/OpImplementation.h>

#include <mlir/Bytecode/BytecodeOpInterface.h>
#include <mlir/Interfaces/CastInterfaces.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

#include "hc/Dialect/Typing/IR/TypingOpsInterfaces.hpp"

namespace hc::typing {
class SymbolicTypeBase : public mlir::Type {
public:
  using Type::Type;

  static bool classof(Type type);

  static SymbolicTypeBase foldExpr(SymbolicTypeBase src);

  SymbolicTypeBase operator+(SymbolicTypeBase rhs) const;
  SymbolicTypeBase operator-(SymbolicTypeBase rhs) const;
  SymbolicTypeBase operator*(SymbolicTypeBase rhs) const;
  SymbolicTypeBase operator%(SymbolicTypeBase rhs) const;
  SymbolicTypeBase floorDiv(SymbolicTypeBase rhs) const;
  SymbolicTypeBase ceilDiv(SymbolicTypeBase rhs) const;
};
} // namespace hc::typing

#include "hc/Dialect/Typing/IR/TypingOpsDialect.h.inc"
#include "hc/Dialect/Typing/IR/TypingOpsEnums.h.inc"

#define GET_TYPEDEF_CLASSES
#include "hc/Dialect/Typing/IR/TypingOpsTypes.h.inc"

#define GET_ATTRDEF_CLASSES
#include "hc/Dialect/Typing/IR/TypingOpsAttributes.h.inc"

#define GET_OP_CLASSES
#include "hc/Dialect/Typing/IR/TypingOps.h.inc"
