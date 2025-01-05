// SPDX-FileCopyrightText: 2024 The HC Authors
// SPDX-FileCopyrightText: 2025 The HC Authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <mlir/Pass/Pass.h>

#include <mlir/Conversion/GPUToROCDL/Runtimes.h> // TODO: remove

namespace mlir {
namespace cf {
class ControlFlowDialect;
}
namespace func {
class FuncDialect;
}

namespace ROCDL {
class ROCDLDialect;
}

namespace gpu {
class GPUModuleOp;
}
} // namespace mlir

namespace hc {
#define GEN_PASS_DECL
#include "hc/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "hc/Transforms/Passes.h.inc"

void populateSimplifyASTPatterns(mlir::RewritePatternSet &patterns);
void populateConvertPyASTToIRPatterns(mlir::RewritePatternSet &patterns);
void populatePyIRPromoteFuncsToStaticPatterns(
    mlir::RewritePatternSet &patterns);

std::unique_ptr<mlir::Pass> createSelectPass(
    std::string name, std::string selectCondName,
    mlir::ArrayRef<
        std::pair<mlir::StringRef, std::function<void(mlir::OpPassManager &)>>>
        populateFuncs);
} // namespace hc
