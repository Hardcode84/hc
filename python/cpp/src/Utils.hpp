// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "llvm/ADT/STLFunctionalExtras.h"

namespace llvm {
class Twine;
struct LogicalResult;
} // namespace llvm

namespace mlir {
class Operation;
class PassManager;
} // namespace mlir

struct Context;

[[noreturn]] void reportError(const llvm::Twine &msg);

llvm::LogicalResult runUnderDiag(mlir::PassManager &pm,
                                 mlir::Operation *module);

void runPipeline(Context &context, mlir::Operation *op,
                 llvm::function_ref<void(mlir::PassManager &)> populateFunc);
