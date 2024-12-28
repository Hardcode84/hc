// SPDX-FileCopyrightText: 2024 The HC Authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <nanobind/nanobind.h>

namespace llvm {
class StringRef;
}

namespace mlir {
class MLIRContext;
}

void pushContext(mlir::MLIRContext *ctx);
void popContext(mlir::MLIRContext *ctx);

void populateMlirModule(nanobind::module_ &m);
llvm::StringRef toString(nanobind::handle h);
