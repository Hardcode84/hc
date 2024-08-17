// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hc/Pipelines/FrontendPipeline.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

#include "hc/Transforms/Passes.hpp"

static void populatePyIROptPasses(mlir::PassManager &pm) {
  pm.addPass(mlir::createCompositeFixedPointPass(
      "PyIROptPass", [](mlir::OpPassManager &p) {
        p.addPass(mlir::createCanonicalizerPass());
        p.addPass(mlir::createCSEPass());
        p.addPass(hc::createCleanupPySetVarPass());
      }));
}

void hc::populateImportPipeline(mlir::PassManager &pm) {
  pm.addPass(hc::createSimplifyASTPass());
  pm.addPass(hc::createConvertPyASTToIRPass());
  populatePyIROptPasses(pm);
  pm.addPass(hc::createReconstuctPySSAPass());
  populatePyIROptPasses(pm);
  pm.addPass(hc::createPyIRPromoteFuncsToStaticPass());
  pm.addPass(mlir::createSymbolDCEPass());
}

void hc::populateFrontendPipeline(mlir::PassManager &pm) {
  pm.addPass(hc::createPyTypeInferencePass());
  pm.addPass(hc::createDropTypeResolversPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createSymbolDCEPass());
}
