// RUN: hc-opt -split-input-file %s --hc-convert-py-func-to-kernel-func-pass | FileCheck %s

// CHECK-LABEL: func.func @func()
//       CHECK: return

py_ir.module {
  %0 = py_ir.func "func" () capture () -> !py_ir.undefined {
    %1 = py_ir.none
    py_ir.return %1 : none
  }
  py_ir.module_end %0 : !py_ir.undefined
}
