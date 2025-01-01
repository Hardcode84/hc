// SPDX-FileCopyrightText: 2024 The HC Authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hc/Pipelines/BackendPipeline.hpp"

#include <mlir/Conversion/AffineToStandard/AffineToStandard.h>
#include <mlir/Conversion/Passes.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/Dialect/GPU/Transforms/Passes.h>
#include <mlir/Interfaces/FunctionInterfaces.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

#include "hc/Transforms/Passes.hpp"

static void populateOptPasses(mlir::OpPassManager &pm) {
  pm.addPass(mlir::createCompositeFixedPointPass(
      "OptPass", [](mlir::OpPassManager &p) {
        p.addPass(mlir::createCanonicalizerPass());
        p.addPass(mlir::createCSEPass());
      }));
}

void hc::populateBackendPipeline(mlir::PassManager &pm,
                                 llvm::StringRef llvmBinDir) {
  pm.addPass(hc::createLegalizeMemrefABIPass());
  pm.addPass(hc::createDecomposeMemrefsPass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createLowerAffinePass());
  populateOptPasses(pm);

  pm.addPass(mlir::createGpuLauchSinkIndexComputationsPass());
  pm.addPass(mlir::createGpuKernelOutliningPass());

  auto &gpuPm = pm.nest<mlir::gpu::GPUModuleOp>();
  gpuPm.addPass(hc::createConvertGpuOpsToROCDLOps());
  populateOptPasses(gpuPm);

  pm.addPass(mlir::createGpuROCDLAttachTarget());

  mlir::GpuModuleToBinaryPassOptions toBinaryOpts;
  toBinaryOpts.toolkitPath = llvmBinDir;
  pm.addPass(mlir::createGpuModuleToBinaryPass(toBinaryOpts));
}
