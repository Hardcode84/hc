// SPDX-FileCopyrightText: 2024 The HC Authors
// SPDX-FileCopyrightText: 2025 The HC Authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hc/Pipelines/MiddleendPipeline.hpp"

#include <mlir/Conversion/AffineToStandard/AffineToStandard.h>
#include <mlir/Dialect/Arith/Transforms/Passes.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Interfaces/FunctionInterfaces.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

#include "hc/Transforms/Passes.hpp"

static void populateOptPasses(mlir::PassManager &pm) {
  pm.addPass(mlir::createCompositeFixedPointPass(
      "OptPass", [](mlir::OpPassManager &p) {
        p.addPass(mlir::arith::createIntRangeOptimizationsPass());
        p.addPass(mlir::createCanonicalizerPass());
        p.addPass(mlir::createCSEPass());
      }));
}

void hc::populateMiddleendPipeline(mlir::PassManager &pm) {
  pm.addPass(hc::createLowerWorkgroupScopePass());
  populateOptPasses(pm);
  pm.addPass(hc::createLowerSubgroupScopePass());
  populateOptPasses(pm);
  pm.addPass(hc::createLowerHKernelOpsPass());
  pm.addPass(hc::createResolveArgsPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(hc::createExpandSharedAllocsPass());
  pm.addPass(hc::createLegalizeBoolMemrefsPass());

  auto &func = pm.nest<mlir::func::FuncOp>();
  func.addPass(hc::createDecomposeHKernelOpsPass());
  func.addPass(mlir::createLowerAffinePass());

  populateOptPasses(pm);
  pm.addPass(hc::createLegalizeDynamicSharedMemPass());
}
