// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hc/Pipelines/MiddleendPipeline.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

#include "hc/Transforms/Passes.hpp"

static void populateOptPasses(mlir::PassManager &pm) {
  pm.addPass(mlir::createCompositeFixedPointPass(
      "OptPass", [](mlir::OpPassManager &p) {
        p.addPass(mlir::createCanonicalizerPass());
        p.addPass(mlir::createCSEPass());
      }));
}

void hc::populateMiddleendPipeline(mlir::PassManager &pm) {
  pm.addPass(hc::createLowerWorkgroupScopePass());
  populateOptPasses(pm);
  pm.addPass(hc::createLowerSubgroupScopePass());
  populateOptPasses(pm);
  pm.addPass(hc::createResolveArgsPass());
  pm.addPass(hc::createLowerHKernelOpsPass());
  populateOptPasses(pm);
}
