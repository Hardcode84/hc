// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <pybind11/pybind11.h>

namespace mlir {
class MLIRContext;
}

void pushContext(mlir::MLIRContext *ctx);
void popContext(mlir::MLIRContext *ctx);

void populateMlirModule(pybind11::module &m);
