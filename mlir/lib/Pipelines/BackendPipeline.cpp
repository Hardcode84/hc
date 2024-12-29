// SPDX-FileCopyrightText: 2024 The HC Authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hc/Pipelines/BackendPipeline.hpp"

#include <mlir/Conversion/AffineToStandard/AffineToStandard.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/GPU/Transforms/Passes.h>
#include <mlir/Interfaces/FunctionInterfaces.h>
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

void hc::populateBackendPipeline(mlir::PassManager &pm) {
  pm.addPass(hc::createLegalizeMemrefABIPass());
  pm.addPass(hc::createDecomposeMemrefsPass());
  populateOptPasses(pm);

  pm.addPass(mlir::createGpuLauchSinkIndexComputationsPass());
  pm.addPass(mlir::createGpuKernelOutliningPass());
}
