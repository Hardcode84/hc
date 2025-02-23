// SPDX-FileCopyrightText: 2024 The HC Authors
// SPDX-FileCopyrightText: 2025 The HC Authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hc/Pipelines/BackendPipeline.hpp"

#include "hc/Dialect/HKernel/IR/HKernelOps.hpp"

#include <mlir/Conversion/AffineToStandard/AffineToStandard.h>
#include <mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h>
#include <mlir/Conversion/GPUToROCDL/GPUToROCDLPass.h>
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
        p.addPass(mlir::createLoopInvariantCodeMotionPass());
      }));
}

void hc::populateBackendPipeline(mlir::PassManager &pm) {
  pm.addPass(hc::createLegalizeMemrefABIPass());
  pm.addPass(hc::createCreatePyWrapperPass());
  pm.addPass(hc::createDecomposeMemrefsPass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createLowerAffinePass());
  pm.addNestedPass<mlir::func::FuncOp>(hc::createLegalizeVectorOpsPass());
  populateOptPasses(pm.nest<mlir::func::FuncOp>());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createSCFToControlFlowPass());

  pm.addPass(mlir::createGpuLauchSinkIndexComputationsPass());
  pm.addPass(mlir::createGpuKernelOutliningPass());
  pm.addPass(mlir::createCanonicalizerPass());

  using FuncT = std::function<void(mlir::OpPassManager &)>;
  std::pair<mlir::StringRef, FuncT> lowerings[] = {
      {"rocm",
       [](mlir::OpPassManager &pm) {
         auto &gpuPm = pm.nest<mlir::gpu::GPUModuleOp>();
         gpuPm.addPass(mlir::createLowerGpuOpsToROCDLOpsPass());
         populateOptPasses(gpuPm);

         pm.addPass(mlir::createGpuROCDLAttachTarget());
       }},
      {"nvvm", [](mlir::OpPassManager &pm) {
         auto &gpuPm = pm.nest<mlir::gpu::GPUModuleOp>();
         gpuPm.addPass(mlir::createConvertGpuOpsToNVVMOps());
         populateOptPasses(gpuPm);

         pm.addPass(mlir::createGpuNVVMAttachTarget());
       }}};

  pm.addPass(hc::createSelectPass(
      "KernelLowering", hc::hk::getKernelBackendAttrName().str(), lowerings));

  pm.addPass(mlir::createGpuModuleToBinaryPass());

  pm.addPass(hc::createGPUToGPURuntimePass());
  pm.addPass(mlir::createConvertToLLVMPass());
  pm.addPass(mlir::createReconcileUnrealizedCastsPass());
  pm.addPass(mlir::createSymbolDCEPass());
  pm.addPass(hc::createLegalizeLLVMABIPass());
  populateOptPasses(pm.nest<mlir::gpu::GPUModuleOp>());
}
