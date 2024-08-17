// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hc/Dialect/Typing/Transforms/Interpreter.hpp"

#include "hc/Dialect/Typing/IR/TypingOpsInterfaces.hpp"

static mlir::FailureOr<bool> handleOp(hc::typing::InterpreterState &state,
                                      mlir::Operation &op) {
  if (auto iface = mlir::dyn_cast<hc::typing::TypingInterpreterInterface>(op))
    return iface.interpret(state);

  return op.emitError("Type interpreter: unsupported op");
}

mlir::FailureOr<bool>
hc::typing::Interpreter::run(mlir::Operation *rootOp, TypeResolverOp resolver,
                             mlir::TypeRange types,
                             llvm::SmallVectorImpl<mlir::Type> &result) {
  assert(!resolver.getBodyRegion().empty());
  state.init(rootOp, resolver.getBodyRegion().front(), types);

  while (true) {
    mlir::Operation &op = state.getNextOp();
    auto res = handleOp(state, op);
    if (mlir::failed(res))
      return mlir::failure();

    if (!*res)
      return false;

    if (state.isCompleted()) {
      getTypes(state, op.getOperands(), result);
      return true;
    }
  }
  llvm_unreachable("Unreachable");
}
