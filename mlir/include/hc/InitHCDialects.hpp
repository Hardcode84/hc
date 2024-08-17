// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>

#include <hc/Dialect/PyAST/IR/PyASTOps.hpp>
#include <hc/Dialect/PyIR/IR/PyIROps.hpp>
#include <hc/Dialect/Typing/IR/TypingOps.hpp>

namespace hc {
inline void registerAllDialects(mlir::DialectRegistry &registry) {
  registry.insert<hc::py_ast::PyASTDialect, hc::py_ir::PyIRDialect,
                  hc::typing::TypingDialect>();
}

inline void registerAllDialects(mlir::MLIRContext &context) {
  mlir::DialectRegistry registry;
  ::hc::registerAllDialects(registry);
  context.appendDialectRegistry(registry);
}

} // namespace hc
