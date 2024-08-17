// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

namespace mlir {
class Operation;
struct LogicalResult;
} // namespace mlir

namespace hc {
mlir::LogicalResult linkModules(mlir::Operation *dest, mlir::Operation *toLink);
}
