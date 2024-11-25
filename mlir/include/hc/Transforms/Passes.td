// SPDX-FileCopyrightText: 2024 The HC Authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef _HC_TRANSFORMS_PASSES_TD_INCLUDED_
#define _HC_TRANSFORMS_PASSES_TD_INCLUDED_

include "mlir/Pass/PassBase.td"

def SimplifyASTPass : Pass<"hc-simplify-ast-pass"> {
  let summary = "TBD";
  let description = [{
    TBD
  }];
}

def ConvertPyASTToIRPass : Pass<"hc-convert-py-ast-to-ir-pass"> {
  let summary = "TBD";
  let description = [{
    TBD
  }];

  let dependentDialects = [
    "::hc::py_ir::PyIRDialect",
    "::mlir::cf::ControlFlowDialect",
    "::hc::typing::TypingDialect",
  ];
}

def ReconstuctPySSAPass : Pass<"hc-reconstruct-py-ssa-pass"> {
  let summary = "TBD";
  let description = [{
    TBD
  }];

  let dependentDialects = [
    "::hc::typing::TypingDialect"
  ];
}

def CleanupPySetVarPass : Pass<"hc-cleanup-py-setvar-pass"> {
  let summary = "TBD";
  let description = [{
    TBD
  }];
}

def PyTypeInferencePass : Pass<"hc-py-type-inference-pass"> {
  let summary = "TBD";
  let description = [{
    TBD
  }];

  let dependentDialects = [
    "::hc::typing::TypingDialect"
  ];
}

def DropTypeResolversPass : Pass<"hc-drop-type-resolvers-pass"> {
  let summary = "TBD";
  let description = [{
    TBD
  }];

  let dependentDialects = [
    "::hc::typing::TypingDialect"
  ];
}

def PyIRPromoteFuncsToStaticPass : Pass<"hc-pyir-promote-funcs-to-static-pass"> {
  let summary = "TBD";
  let description = [{
    TBD
  }];

  let dependentDialects = [
    "::hc::py_ir::PyIRDialect",
    "::hc::typing::TypingDialect"
  ];
}

def ConverPyFuncToFuncPass : Pass<"hc-convert-py-func-to-func-pass"> {
  let summary = "TBD";
  let description = [{
    TBD
  }];

  let dependentDialects = [
    "::hc::py_ir::PyIRDialect",
    "::mlir::func::FuncDialect",
  ];
}

def ConverPyFuncToKernelFuncPass : Pass<"hc-convert-py-func-to-kernel-func-pass", "::mlir::ModuleOp"> {
  let summary = "TBD";
  let description = [{
    TBD
  }];

  let dependentDialects = [
    "::hc::hk::HKernelDialect",
    "::hc::py_ir::PyIRDialect",
    "::mlir::func::FuncDialect",
  ];
}

def ConverPyIRToKernelPass : Pass<"hc-convert-py-IR-to-kernel-pass", "::mlir::ModuleOp"> {
  let summary = "TBD";
  let description = [{
    TBD
  }];

  let dependentDialects = [
    "::hc::hk::HKernelDialect",
    "::hc::py_ir::PyIRDialect",
    "::hc::typing::TypingDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::func::FuncDialect",
  ];
}

def LowerWorkgroupScopePass : Pass<"hc-lower-workgroup-scope-pass"> {
  let summary = "TBD";
  let description = [{
    TBD
  }];

  let dependentDialects = [
    "::hc::hk::HKernelDialect",
    "::hc::typing::TypingDialect",
  ];
}

def LowerSubgroupScopePass : Pass<"hc-lower-subgroup-scope-pass"> {
  let summary = "TBD";
  let description = [{
    TBD
  }];

  let dependentDialects = [
    "::hc::hk::HKernelDialect",
    "::hc::typing::TypingDialect",
  ];
}

def ResolveArgsPass : Pass<"hc-resolve-args-pass"> {
  let summary = "TBD";
  let description = [{
    TBD
  }];

  let dependentDialects = [
    "::hc::hk::HKernelDialect",
    "::hc::typing::TypingDialect",
    "::mlir::affine::AffineDialect",
    "::mlir::gpu::GPUDialect",
    "::mlir::memref::MemRefDialect",
  ];
}

def LowerHKernelOpsPass : Pass<"hc-lower-hkernel-ops-pass"> {
  let summary = "TBD";
  let description = [{
    TBD
  }];

  let dependentDialects = [
    "::hc::hk::HKernelDialect",
    "::hc::typing::TypingDialect",
    "::mlir::gpu::GPUDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::ub::UBDialect",
    "::mlir::vector::VectorDialect",
  ];
}

def DecomposeHKernelOpsPass : Pass<"hc-decompose-hkernel-ops-pass"> {
  let summary = "TBD";
  let description = [{
    TBD
  }];

  let dependentDialects = [
    "::hc::hk::HKernelDialect",
    "::mlir::arith::ArithDialect",
  ];
}

def LegalizeBoolMemrefsPass : Pass<"hc-legalize-bool-memrefs-pass"> {
  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::vector::VectorDialect",
  ];
}

def LegalizeMemrefABIPass : Pass<"hc-legalize-memref-abi-pass"> {
  let dependentDialects = [
    "::hc::hk::HKernelDialect",
  ];
}

def DecomposeMemrefsPass : Pass<"hc-decompose-memrefs-pass"> {
  let dependentDialects = [
    "::hc::hk::HKernelDialect",
    "::mlir::affine::AffineDialect",
    "::mlir::arith::ArithDialect",
  ];
}

def DecomposePointersPass : Pass<"hc-decompose-pointers-pass"> {
  let dependentDialects = [
    "::hc::hk::HKernelDialect",
    "::mlir::arith::ArithDialect",
  ];
}

#endif // _HC_TRANSFORMS_PASSES_TD_INCLUDED_
