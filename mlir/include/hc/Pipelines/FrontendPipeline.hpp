// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

namespace mlir {
class PassManager;
}

namespace hc {
void populateImportPipeline(mlir::PassManager &pm);
void populateFrontendPipeline(mlir::PassManager &pm);
} // namespace hc
