// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <mlir/Support/LogicalResult.h>

namespace llvm {
class StringRef;
}

namespace mlir {
class Operation;
}

namespace hc {
mlir::FailureOr<mlir::Operation *>
importPyModule(llvm::StringRef str, mlir::Operation *module, bool dumpAST);
} // namespace hc
