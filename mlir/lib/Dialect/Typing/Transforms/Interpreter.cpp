// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hc/Dialect/Typing/Transforms/Interpreter.hpp"

#include "hc/Dialect/Typing/IR/TypingOpsInterfaces.hpp"

static mlir::FailureOr<hc::typing::InterpreterResult>
handleOp(hc::typing::InterpreterState &state, mlir::Operation &op) {
  if (auto iface = mlir::dyn_cast<hc::typing::TypingInterpreterInterface>(op))
    return iface.interpret(state);

  return op.emitError("Type interpreter: unsupported op");
}

mlir::FailureOr<bool>
hc::typing::Interpreter::run(mlir::Operation *rootOp, TypeResolverOp resolver,
                             mlir::TypeRange types,
                             llvm::SmallVectorImpl<mlir::Type> &result) {
  assert(!resolver.getBodyRegion().empty());
  state.init(rootOp, resolver.getBodyRegion().front(), types, result);

  while (true) {
    mlir::Operation &op = state.getNextOp();
    auto res = handleOp(state, op);
    if (mlir::failed(res))
      return mlir::failure();

    switch (*res) {
    case InterpreterResult::MatchFail:
      return false;
    case InterpreterResult::MatchSuccess:
      return true;
    case InterpreterResult::Advance:
      continue;
    default:
      llvm_unreachable("Invalid type interpreter state");
    }
  }
  llvm_unreachable("Unreachable");
}
