// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "hc/Dialect/Typing/IR/TypingOps.hpp"

namespace hc::typing {
class Interpreter final {
public:
  mlir::FailureOr<bool> run(mlir::Operation *rootOp, TypeResolverOp resolver,
                            mlir::TypeRange types,
                            llvm::SmallVectorImpl<mlir::Type> &result);

private:
  InterpreterState state;
};
} // namespace hc::typing
