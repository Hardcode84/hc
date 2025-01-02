// SPDX-FileCopyrightText: 2024 The HC Authors
// SPDX-FileCopyrightText: 2025 The HC Authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

namespace llvm {
struct LogicalResult;
}

namespace mlir {
class Operation;
}

namespace hc {
llvm::LogicalResult linkModules(mlir::Operation *dest, mlir::Operation *toLink);
}
