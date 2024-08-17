// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/OpImplementation.h>

#include <mlir/Bytecode/BytecodeOpInterface.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

namespace hc::py_ast {
template <typename ConcreteType>
class NoReturn : public mlir::OpTrait::TraitBase<ConcreteType, NoReturn> {};
} // namespace hc::py_ast

#include "hc/Dialect/PyAST/IR/PyASTOpsDialect.h.inc"
#include "hc/Dialect/PyAST/IR/PyASTOpsEnums.h.inc"

#define GET_TYPEDEF_CLASSES
#include "hc/Dialect/PyAST/IR/PyASTOpsTypes.h.inc"

#define GET_ATTRDEF_CLASSES
#include "hc/Dialect/PyAST/IR/PyASTOpsAttributes.h.inc"

#define GET_OP_CLASSES
#include "hc/Dialect/PyAST/IR/PyASTOps.h.inc"
