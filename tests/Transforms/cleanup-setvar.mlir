// RUN: hc-opt -allow-unregistered-dialect -split-input-file %s --hc-cleanup-py-setvar-pass | FileCheck %s

// CHECK-LABEL: py_ir.module
//       CHECK:  %[[B:.*]] = py_ir.loadvar "B" : none
//       CHECK:  py_ir.storevar "A" %[[B]] : none
//   CHECK-NOT:  py_ir.storevar "C" %{{.*}} : none
//       CHECK:  %[[A:.*]] = py_ir.loadvar "A" : none
//       CHECK:  py_ir.return %[[A]] : none
py_ir.module {
  %f = py_ir.func "foo" () capture () -> !py_ir.undefined {
    %0 = py_ir.loadvar "B" : none
    py_ir.storevar "A" %0 : none
    py_ir.storevar "C" %0 : none
    %1 = py_ir.loadvar "A" : none
    py_ir.return %1 : none
  }
}

// -----

// CHECK-LABEL: py_ir.module
//       CHECK:  %[[B:.*]] = py_ir.loadvar "B" : none
//       CHECK:  py_ir.storevar "A" %[[B]] : none
//   CHECK-NOT:  py_ir.storevar "A" %[[B]] : none
//       CHECK:  %[[A:.*]] = py_ir.loadvar "A" : none
//       CHECK:  py_ir.return %[[A]] : none
py_ir.module {
  %f = py_ir.func "foo" () capture () -> !py_ir.undefined {
    %0 = py_ir.loadvar "B" : none
    py_ir.storevar "A" %0 : none
    py_ir.storevar "A" %0 : none
    %1 = py_ir.loadvar "A" : none
    py_ir.return %1 : none
  }
}
