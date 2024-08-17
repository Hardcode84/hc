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



#endif // _HC_TRANSFORMS_PASSES_TD_INCLUDED_
