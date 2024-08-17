// RUN: hc-opt -allow-unregistered-dialect -split-input-file %s --hc-reconstruct-py-ssa-pass | FileCheck %s

// CHECK-LABEL: py_ir.module
//       CHECK:  %[[B_C:.*]] = py_ir.loadvar "B" : none
//       CHECK:  py_ir.func "foo" () capture (B:%[[B_C]]) : none -> !py_ir.undefined
//       CHECK:  ^bb0(%[[B:.*]]: none):
//       CHECK:  py_ir.return %[[B]] : none
py_ir.module {
  %f = py_ir.func "foo" () capture () -> !py_ir.undefined {
    %0 = py_ir.loadvar "B" : none
    py_ir.storevar "A" %0 : none
    %1 = py_ir.loadvar "A" : none
    py_ir.return %1 : none
  }
}

// -----

// CHECK-LABEL: py_ir.module
//       CHECK:  %[[B_C:.*]] = py_ir.loadvar "B" : none
//       CHECK:  py_ir.func "foo" () capture (B:%[[B_C]]) : none -> !py_ir.undefined
//       CHECK:  ^bb0(%[[B:.*]]: none):
//       CHECK:  cf.br ^bb1(%[[B]] : none)
//       CHECK:  ^bb1(%[[B1:.*]]: none):
//       CHECK:  py_ir.return %[[B1]] : none
py_ir.module {
  %f = py_ir.func "foo" () capture () -> !py_ir.undefined {
   ^bb0:
    %0 = py_ir.loadvar "B" : none
    py_ir.storevar "A" %0 : none
    cf.br ^bb1
   ^bb1:
    %1 = py_ir.loadvar "A" : none
    py_ir.return %1 : none
  }
}

// -----

// CHECK-LABEL: py_ir.module
//       CHECK:  %[[B_C:.*]] = py_ir.loadvar "B" : none
//       CHECK:  py_ir.func "foo" () capture (B:%[[B_C]]) : none -> !py_ir.undefined
//       CHECK:  ^bb0(%[[B:.*]]: none):
//       CHECK:  cf.br ^bb1(%[[B]] : none)
//       CHECK:  ^bb1(%[[B1:.*]]: none):
//       CHECK:  cf.br ^bb2(%[[B1]] : none)
//       CHECK:  ^bb2(%[[B2:.*]]: none):
//       CHECK:  py_ir.return %[[B2]] : none
py_ir.module {
  %f = py_ir.func "foo" () capture () -> !py_ir.undefined {
   ^bb0:
    %0 = py_ir.loadvar "B" : none
    py_ir.storevar "A" %0 : none
    cf.br ^bb1
   ^bb1:
    cf.br ^bb2
   ^bb2:
    %1 = py_ir.loadvar "A" : none
    py_ir.return %1 : none
  }
}

// -----

// CHECK-LABEL: py_ir.module
//       CHECK:  %[[B_C:.*]] = py_ir.loadvar "B" : none
//       CHECK:  py_ir.func "foo" () capture (B:%[[B_C]]) : none -> !py_ir.undefined
//       CHECK:  ^bb0(%[[B:.*]]: none):
//       CHECK:  cf.br ^bb1(%[[B]] : none)
//       CHECK:  ^bb1(%[[B1:.*]]: none):
//       CHECK:  cf.cond_br %{{.*}}, ^bb1(%[[B1]] : none), ^bb2(%[[B1]] : none)
//       CHECK:  ^bb2(%[[B2:.*]]: none):
//       CHECK:  py_ir.return %[[B2]] : none
py_ir.module {
  %f = py_ir.func "foo" () capture () -> !py_ir.undefined {
   ^bb0:
    %0 = py_ir.loadvar "B" : none
    py_ir.storevar "A" %0 : none
    cf.br ^bb1
   ^bb1:
    %n = py_ir.none
    %c = typing.cast %n : none to i1
    cf.cond_br %c, ^bb1, ^bb2
   ^bb2:
    %1 = py_ir.loadvar "A" : none
    py_ir.return %1 : none
  }
}

// -----

// CHECK-LABEL: py_ir.module
//       CHECK:  %[[B_C:.*]] = py_ir.loadvar "B" : none
//       CHECK:  py_ir.func "foo" () capture (B:%[[B_C]]) : none -> !py_ir.undefined
//       CHECK:  ^bb0(%[[B:.*]]: none):
//       CHECK:  cf.br ^bb1(%[[B]] : none)
//       CHECK:  ^bb1(%[[B1:.*]]: none):
//       CHECK:  cf.cond_br %{{.*}}, ^bb2(%[[B1]] : none), ^bb2(%[[B1]] : none)
//       CHECK:  ^bb2(%[[B2:.*]]: none):
//       CHECK:  py_ir.return %[[B2]] : none
py_ir.module {
  %f = py_ir.func "foo" () capture () -> !py_ir.undefined {
   ^bb0:
    %0 = py_ir.loadvar "B" : none
    py_ir.storevar "A" %0 : none
    cf.br ^bb1
   ^bb1:
    %n = py_ir.none
    %c = typing.cast %n : none to i1
    cf.cond_br %c, ^bb2, ^bb2
   ^bb2:
    %1 = py_ir.loadvar "A" : none
    py_ir.return %1 : none
  }
}

// -----

// CHECK-LABEL: py_ir.module
//       CHECK:  py_ir.func
//       CHECK:    cf.br ^[[CONDBR:.*]](%{{.*}})
//       CHECK:  ^[[CONDBR]](%[[BV1:.*]]: [[BV1T:.*]]):
//       CHECK:    %[[COND:.*]] = typing.cast %[[BV1]] : [[BV1T]] to i1
//       CHECK:    cf.cond_br %[[COND]], ^[[THENBR:.*]](%[[BV1]] : [[BV1T]]), ^[[ELSEBR:.*]]
//       CHECK:  ^[[THENBR]](%[[BV2:.*]]: [[BV2T:.*]]):
//       CHECK:    %[[R:.*]] = py_ir.binop %[[BV2]] : [[BV2T]] add {{.*}} -> [[RT:.*]]
//       CHECK:    py_ir.storevar "A" %[[R]]
//       CHECK:    cf.br ^[[CONDBR]](%[[R]] : [[RT]])
//       CHECK:  ^[[ELSEBR]]:
//       CHECK:    py_ir.return
py_ir.module {
  %1 = py_ir.func "func" () capture () -> !py_ir.undefined {
    %0 = py_ir.constant 1 : i64
    cf.br ^bb1
  ^bb1:  // 2 preds: ^bb0, ^bb2
    %4 = py_ir.loadvar "A" : !py_ir.undefined
    %5 = typing.cast %4 : !py_ir.undefined to i1
    cf.cond_br %5, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %6 = py_ir.loadvar "A" : !py_ir.undefined
    %7 = py_ir.binop %6 : !py_ir.undefined add %0 : i64 -> !py_ir.undefined
    py_ir.storevar "A" %7 : !py_ir.undefined
    cf.br ^bb1
  ^bb3:  // pred: ^bb1
    %8 = py_ir.none
    py_ir.return %8 : none
  }
  py_ir.storevar "foo" %1 : !py_ir.undefined
  %2 = py_ir.loadvar "foo" : !py_ir.undefined
  %3 = py_ir.call %2 : !py_ir.undefined  () -> !py_ir.undefined
}

// -----

// CHECK-LABEL: py_ir.module
//   CHECK-DAG:  %[[A:.*]] = py_ir.loadvar "A" : !py_ir.undefined
//   CHECK-DAG:  %[[B:.*]] = py_ir.loadvar "B" : !py_ir.undefined
//       CHECK:  py_ir.func "func" () capture (A:%[[A]], B:%[[B]]) : !py_ir.undefined, !py_ir.undefined -> !py_ir.undefined {
//  CHECK-NEXT:  ^bb0(%{{.*}}: !py_ir.undefined, %{{.*}}: !py_ir.undefined):
py_ir.module {
  %0 = py_ir.func "func" () capture () -> !py_ir.undefined {
    cf.br ^bb1
  ^bb1:  // 2 preds: ^bb0, ^bb2
    %3 = py_ir.loadvar "A" : !py_ir.undefined
    %4 = typing.cast %3 : !py_ir.undefined to i1
    cf.cond_br %4, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %5 = py_ir.loadvar "B" : !py_ir.undefined
    %6 = typing.cast %5 : !py_ir.undefined to i1
    cf.cond_br %6, ^bb3, ^bb1
  ^bb3:  // 2 preds: ^bb1, ^bb2
    %7 = py_ir.none
    py_ir.return %7 : none
  }
  py_ir.storevar "func" %0 : !py_ir.undefined
  %1 = py_ir.loadvar "func" : !py_ir.undefined
  %2 = py_ir.call %1 : !py_ir.undefined  () -> !py_ir.undefined
}
