// SPDX-FileCopyrightText: 2024 The HC Authors
// SPDX-FileCopyrightText: 2025 The HC Authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hc/Transforms/Passes.hpp"

#include <mlir/Conversion/AMDGPUToROCDL/AMDGPUToROCDL.h>
#include <mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h>
#include <mlir/Conversion/ConvertToLLVM/ToLLVMPass.h>
#include <mlir/Conversion/GPUCommon/GPUCommonPass.h>
#include <mlir/Conversion/GPUToROCDL/GPUToROCDLPass.h>
#include <mlir/Conversion/LLVMCommon/ConversionTarget.h>
#include <mlir/Conversion/LLVMCommon/LoweringOptions.h>
#include <mlir/Conversion/LLVMCommon/Pattern.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h>
#include <mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h>
#include <mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h>
#include <mlir/Dialect/AMDGPU/Utils/Chipset.h>
#include <mlir/Dialect/Arith/Transforms/Passes.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/Dialect/GPU/Transforms/Passes.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/LLVMIR/ROCDLDialect.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace hc {
#define GEN_PASS_DEF_CONVERTGPUOPSTOROCDLOPS
#include "hc/Transforms/Passes.h.inc"
} // namespace hc

using namespace mlir;

/// Returns true if the given `gpu.func` can be safely called using the bare
/// pointer calling convention.
static bool canBeCalledWithBarePointers(gpu::GPUFuncOp func) {
  bool canBeBare = true;
  for (Type type : func.getArgumentTypes())
    if (auto memrefTy = dyn_cast<BaseMemRefType>(type))
      canBeBare &= LLVMTypeConverter::canConvertToBarePtr(memrefTy);
  return canBeBare;
}

static constexpr StringLiteral amdgcnDataLayout =
    "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32"
    "-p7:160:256:256:32-p8:128:128-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:"
    "32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:"
    "64-S32-A5-G1-ni:7:8:9";

namespace {

// A pass that replaces all occurrences of GPU device operations with their
// corresponding ROCDL equivalent.
//
// This pass only handles device code and is not meant to be run on GPU host
// code.
struct LowerGpuOpsToROCDLOpsPass
    : public hc::impl::ConvertGpuOpsToROCDLOpsBase<LowerGpuOpsToROCDLOpsPass> {
  LowerGpuOpsToROCDLOpsPass() = default;
  LowerGpuOpsToROCDLOpsPass(const hc::ConvertGpuOpsToROCDLOpsOptions options) {
    if (this->chipset.getNumOccurrences() == 0)
      this->chipset = options.chipset;
    if (this->indexBitwidth.getNumOccurrences() == 0)
      this->indexBitwidth = options.indexBitwidth;
    if (this->useBarePtrCallConv.getNumOccurrences() == 0)
      this->useBarePtrCallConv = options.useBarePtrCallConv;
    if (this->runtime.getNumOccurrences() == 0)
      this->runtime = options.runtime;
  }

  void getDependentDialects(DialectRegistry &registry) const override final {
    Base::getDependentDialects(registry);
    registerConvertToLLVMDependentDialectLoading(registry);
  }

  void runOnOperation() override {
    gpu::GPUModuleOp m = getOperation();
    MLIRContext *ctx = m.getContext();

    auto llvmDataLayout = m->getAttrOfType<StringAttr>(
        LLVM::LLVMDialect::getDataLayoutAttrName());
    if (!llvmDataLayout) {
      llvmDataLayout = StringAttr::get(ctx, amdgcnDataLayout);
      m->setAttr(LLVM::LLVMDialect::getDataLayoutAttrName(), llvmDataLayout);
    }
    // Request C wrapper emission.
    for (auto func : m.getOps<func::FuncOp>()) {
      func->setAttr(LLVM::LLVMDialect::getEmitCWrapperAttrName(),
                    UnitAttr::get(ctx));
    }

    FailureOr<amdgpu::Chipset> maybeChipset = amdgpu::Chipset::parse(chipset);
    if (failed(maybeChipset)) {
      emitError(UnknownLoc::get(ctx), "Invalid chipset name: " + chipset);
      return signalPassFailure();
    }

    /// Customize the bitwidth used for the device side index computations.
    LowerToLLVMOptions options(
        ctx, DataLayout(cast<DataLayoutOpInterface>(m.getOperation())));
    options.dataLayout = llvm::DataLayout(llvmDataLayout.getValue());
    if (indexBitwidth != kDeriveIndexBitwidthFromDataLayout)
      options.overrideIndexBitwidth(indexBitwidth);

    if (useBarePtrCallConv) {
      options.useBarePtrCallConv = true;
      WalkResult canUseBarePointers =
          m.walk([](gpu::GPUFuncOp func) -> WalkResult {
            if (canBeCalledWithBarePointers(func))
              return WalkResult::advance();
            return WalkResult::interrupt();
          });
      if (canUseBarePointers.wasInterrupted()) {
        emitError(UnknownLoc::get(ctx),
                  "bare pointer calling convention requires all memrefs to "
                  "have static shape and use the identity map");
        return signalPassFailure();
      }
    }

    // Apply in-dialect lowering. In-dialect lowering will replace
    // ops which need to be lowered further, which is not supported by a
    // single conversion pass.
    {
      RewritePatternSet patterns(ctx);
      populateGpuRewritePatterns(patterns);
      arith::populateExpandBFloat16Patterns(patterns);
      (void)applyPatternsGreedily(m, std::move(patterns));
    }

    LLVMTypeConverter converter(ctx, options);
    populateGpuMemorySpaceAttributeConversions(
        converter, [](gpu::AddressSpace space) {
          switch (space) {
          case gpu::AddressSpace::Global:
            return 1;
          case gpu::AddressSpace::Workgroup:
            return 3;
          case gpu::AddressSpace::Private:
            return 5;
          }
          llvm_unreachable("unknown address space enum value");
          return 0;
        });

    LLVMConversionTarget target(getContext());
    RewritePatternSet llvmPatterns(ctx);

    for (Dialect *dialect : ctx->getLoadedDialects()) {
      auto iface = dyn_cast<ConvertToLLVMPatternInterface>(dialect);
      if (!iface)
        continue;

      iface->populateConvertToLLVMConversionPatterns(target, converter,
                                                     llvmPatterns);
    }

    // TODO: These aren't covered by the ConvertToLLVMPatternInterface right
    // now.
    populateVectorToLLVMConversionPatterns(converter, llvmPatterns);
    populateAMDGPUToROCDLConversionPatterns(converter, llvmPatterns,
                                            *maybeChipset);
    populateGpuToROCDLConversionPatterns(converter, llvmPatterns, runtime);
    populateFinalizeMemRefToLLVMConversionPatterns(converter, llvmPatterns);
    configureGpuToROCDLConversionLegality(target);
    if (failed(applyPartialConversion(m, target, std::move(llvmPatterns))))
      signalPassFailure();
    auto *rocdlDialect = getContext().getLoadedDialect<ROCDL::ROCDLDialect>();
    auto reqdWorkGroupSizeAttrHelper =
        rocdlDialect->getReqdWorkGroupSizeAttrHelper();
    auto flatWorkGroupSizeAttrHelper =
        rocdlDialect->getFlatWorkGroupSizeAttrHelper();
    // Manually rewrite known block size attributes so the LLVMIR translation
    // infrastructure can pick them up.
    m.walk([&](LLVM::LLVMFuncOp op) {
      if (reqdWorkGroupSizeAttrHelper.isAttrPresent(op)) {
        auto blockSizes = reqdWorkGroupSizeAttrHelper.getAttr(op);
        // Also set up the rocdl.flat_work_group_size attribute to prevent
        // conflicting metadata.
        uint32_t flatSize = 1;
        for (uint32_t size : blockSizes.asArrayRef()) {
          flatSize *= size;
        }
        StringAttr flatSizeAttr =
            StringAttr::get(ctx, Twine(flatSize) + "," + Twine(flatSize));
        flatWorkGroupSizeAttrHelper.setAttr(op, flatSizeAttr);
      }
    });
  }
};

} // namespace
