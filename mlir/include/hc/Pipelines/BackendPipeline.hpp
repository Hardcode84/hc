// SPDX-FileCopyrightText: 2024 The HC Authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

namespace mlir {
class PassManager;
}

namespace hc {
void populateBackendPipeline(mlir::PassManager &pm);
}
