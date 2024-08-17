// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

namespace mlir {
class PassManager;
}

void populateTypingPipeline(mlir::PassManager &pm);
