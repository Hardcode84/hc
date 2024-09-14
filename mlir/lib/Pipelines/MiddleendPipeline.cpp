// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hc/Pipelines/MiddleendPipeline.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

#include "hc/Transforms/Passes.hpp"

void hc::populateMiddleendPipeline(mlir::PassManager &pm) {
  pm.addPass(hc::createLowerWorkgroupScopePass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(hc::createLowerSubgroupScopePass());
  pm.addPass(mlir::createCanonicalizerPass());
}
