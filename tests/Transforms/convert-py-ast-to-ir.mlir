// RUN: hc-opt -allow-unregistered-dialect -split-input-file %s --hc-convert-py-ast-to-ir-pass | FileCheck %s

// CHECK-LABEL: py_ir.module {
//       CHECK: }
py_ast.module {
}

// -----

// CHECK-LABEL: py_ir.module {
//       CHECK:  %[[R:.*]] = py_ir.loadvar "Val" : !py_ir.undefined
//       CHECK:  py_ir.module_end %[[R]] : !py_ir.undefined
py_ast.module {
  py_ast.capture_val "Val"
}

// -----

// CHECK-LABEL: py_ir.module {
//       CHECK:  py_ir.func "func"
//       CHECK:  %[[R:.*]] = py_ir.loadvar "A" : !py_ir.undefined
//       CHECK:  py_ir.return %[[R]] : !py_ir.undefined
py_ast.module {
  py_ast.func "func"() {
    %0 = py_ast.name "A"
    py_ast.return %0
  }
}

// -----

// CHECK-LABEL: py_ir.module
//       CHECK:  py_ir.func "func"
//       CHECK:  %[[R:.*]] = py_ir.none
//       CHECK:  py_ir.return %[[R]] : none
py_ast.module {
  py_ast.func "func"() {
    %0 = py_ast.constant #py_ast.none
    py_ast.return %0
  }
}

// -----

// CHECK-LABEL: py_ir.module
//       CHECK:  py_ir.func "func"
//       CHECK:  %[[R:.*]] = py_ir.constant 42 : i64
//       CHECK:  %[[R1:.*]] = typing.cast %[[R]] : i64 to !py_ir.undefined
//       CHECK:  py_ir.return %[[R1]] : !py_ir.undefined
py_ast.module {
  py_ast.func "func"() {
    %0 = py_ast.constant 42 : i64
    py_ast.return %0
  }
}

// -----

// CHECK-LABEL: py_ir.module
//       CHECK:  %[[B:.*]] = py_ir.loadvar "B" : !py_ir.undefined
//       CHECK:  %[[C:.*]] = py_ir.loadvar "C" : !py_ir.undefined
//       CHECK:  %[[E:.*]] = py_ir.loadvar "E" : !py_ir.undefined
//       CHECK:  %[[R:.*]] = py_ir.call %[[B]] : !py_ir.undefined (%[[C]], D:%[[E]]) : !py_ir.undefined, !py_ir.undefined -> !py_ir.undefined
//       CHECK:  py_ir.storevar "A" %[[R]] : !py_ir.undefined
py_ast.module {
  %0 = py_ast.name "A"
  %1 = py_ast.name "C"
  %2 = py_ast.name "E"
  %3 = py_ast.keyword "D" = %2
  %4 = py_ast.name "B"
  %5 = py_ast.call %4(%1 keywords %3)
  py_ast.assign(%0) = %5
}

// -----

// CHECK-LABEL: py_ir.module
//       CHECK:  %[[A:.*]] = py_ir.loadvar "A" : !py_ir.undefined
//       CHECK:  %{{.*}} = py_ir.call %[[A]] : !py_ir.undefined () -> !py_ir.undefined
py_ast.module {
  %0 = py_ast.name "A"
  %1 = py_ast.call %0( keywords )
  py_ast.expr %1
}

// -----

// CHECK-LABEL: py_ir.module
//       CHECK:  py_ir.func "func"
//       CHECK:  ^bb0(%[[ARG1:.*]]: !py_ir.undefined, %[[ARG2:.*]]: !py_ir.undefined, %[[ARG3:.*]]: !py_ir.undefined):
//       CHECK:  py_ir.storevar "a" %[[ARG1]] : !py_ir.undefined
//       CHECK:  py_ir.storevar "b" %[[ARG2]] : !py_ir.undefined
//       CHECK:  py_ir.storevar "c" %[[ARG3]] : !py_ir.undefined
//       CHECK:  %[[R:.*]] = py_ir.none
//       CHECK:  py_ir.return %[[R]] : none
py_ast.module {
  %0 = py_ast.arg "a"
  %1 = py_ast.arg "b"
  %2 = py_ast.arg "c"
  py_ast.func "func"(%0, %1, %2) {
    %3 = py_ast.constant #py_ast.none
    py_ast.return %3
  }
}

// -----

// CHECK-LABEL: py_ir.module
//       CHECK:  %[[A:.*]] = py_ir.empty_annotation
//       CHECK:  py_ir.func "func" (a:%[[A]]) : !py_ir.undefined capture () -> !py_ir.undefined
//       CHECK:  ^bb0(%[[ARG1:.*]]: !py_ir.undefined):
//       CHECK:  py_ir.storevar "a" %[[ARG1]] : !py_ir.undefined
//       CHECK:  %[[R:.*]] = py_ir.none
//       CHECK:  py_ir.return %[[R]] : none
py_ast.module {
  %1 = py_ast.arg "a"
  py_ast.func "func"(%1) {
    %3 = py_ast.constant #py_ast.none
    py_ast.return %3
  }
}

// -----

// CHECK-LABEL: py_ir.module
//       CHECK:  %[[A:.*]] = py_ir.loadvar "Foo" : !py_ir.undefined
//       CHECK:  py_ir.func "func" (a:%[[A]]) : !py_ir.undefined capture () -> !py_ir.undefined
//       CHECK:  ^bb0(%[[ARG1:.*]]: !py_ir.undefined):
//       CHECK:  py_ir.storevar "a" %[[ARG1]] : !py_ir.undefined
//       CHECK:  %[[R:.*]] = py_ir.none
//       CHECK:  py_ir.return %[[R]] : none
py_ast.module {
  %0 = py_ast.name "Foo"
  %1 = py_ast.arg "a" : %0
  py_ast.func "func"(%1) {
    %3 = py_ast.constant #py_ast.none
    py_ast.return %3
  }
}

// -----

// CHECK-LABEL: py_ir.module
//   CHECK-DAG:  %[[A:.*]] = py_ir.loadvar "Foo" : !py_ir.undefined
//   CHECK-DAG:  %[[B:.*]] = py_ir.loadvar "Bar" : !py_ir.undefined
//       CHECK:  %[[C:.*]] = py_ir.getitem %[[A]] : !py_ir.undefined[%[[B]] : !py_ir.undefined] -> !py_ir.undefined
//       CHECK:  py_ir.func "func" (a:%[[C]]) : !py_ir.undefined capture () -> !py_ir.undefined
//       CHECK:  ^bb0(%[[ARG1:.*]]: !py_ir.undefined):
//       CHECK:  py_ir.storevar "a" %[[ARG1]] : !py_ir.undefined
//       CHECK:  %[[R:.*]] = py_ir.none
//       CHECK:  py_ir.return %[[R]] : none
py_ast.module {
  %0 = py_ast.name "Foo"
  %1 = py_ast.name "Bar"
  %2 = py_ast.subscript %0 [%1]
  %3 = py_ast.arg "a" : %2
  py_ast.func "func"(%3) {
    %4 = py_ast.constant #py_ast.none
    py_ast.return %4
  }
}

// -----

// CHECK-LABEL: py_ir.module
//       CHECK:  %[[C:.*]] = py_ir.constant 42 : i64
//       CHECK:  %[[C1:.*]] = typing.cast %[[C]] : i64 to !py_ir.undefined
//       CHECK:  py_ir.func "func" (a:%[[C1]]) : !py_ir.undefined capture () -> !py_ir.undefined
//       CHECK:  ^bb0(%[[ARG1:.*]]: !py_ir.undefined):
//       CHECK:  py_ir.storevar "a" %[[ARG1]] : !py_ir.undefined
//       CHECK:  %[[R:.*]] = py_ir.none
//       CHECK:  py_ir.return %[[R]] : none
py_ast.module {
  %0 = py_ast.constant 42 : i64
  %1 = py_ast.arg "a" : %0
  py_ast.func "func"(%1) {
    %3 = py_ast.constant #py_ast.none
    py_ast.return %3
  }
}

// -----

// CHECK-LABEL: py_ir.module
//       CHECK:  py_ir.func "func"
//       CHECK:  %[[C1:.*]] = py_ir.loadvar "Cond" : !py_ir.undefined
//       CHECK:  %[[C2:.*]] = typing.cast %[[C1]] : !py_ir.undefined to i1
//       CHECK:  cf.cond_br %[[C2]], ^bb1, ^bb2
//       CHECK:  ^bb1
//       CHECK:  "test.test1"() : () -> ()
//       CHECK:  cf.br ^bb2
//       CHECK:  ^bb2
//       CHECK:  %[[R:.*]] = py_ir.none
//       CHECK:  py_ir.return %[[R]] : none
py_ast.module {
  py_ast.func "func"() {
    %0 = py_ast.name "Cond"
    py_ast.if %0 {
      "test.test1"() : () -> ()
    } {}
    %1 = py_ast.constant #py_ast.none
    py_ast.return %1
  }
}

// -----

// CHECK-LABEL: py_ir.module
//       CHECK:  py_ir.func "func"
//       CHECK:  %[[C1:.*]] = py_ir.loadvar "Cond" : !py_ir.undefined
//       CHECK:  %[[C2:.*]] = typing.cast %[[C1]] : !py_ir.undefined to i1
//       CHECK:  cf.cond_br %[[C2]], ^bb1, ^bb2
//       CHECK:  ^bb1
//       CHECK:  "test.test1"() : () -> ()
//       CHECK:  cf.br ^bb3
//       CHECK:  ^bb2
//       CHECK:  "test.test2"() : () -> ()
//       CHECK:  cf.br ^bb3
//       CHECK:  ^bb3
//       CHECK:  %[[R:.*]] = py_ir.none
//       CHECK:  py_ir.return %[[R]] : none
py_ast.module {
  py_ast.func "func"() {
    %0 = py_ast.name "Cond"
    py_ast.if %0 {
      "test.test1"() : () -> ()
    } {
      "test.test2"() : () -> ()
    }
    %1 = py_ast.constant #py_ast.none
    py_ast.return %1
  }
}

// -----

// CHECK-LABEL: py_ir.module
//       CHECK:  %[[V:.*]] = py_ir.loadvar "B" : !py_ir.undefined
//       CHECK:  py_ir.storevar "A" %[[V]] : !py_ir.undefined
py_ast.module {
    %0 = py_ast.name "A"
    %1 = py_ast.name "B"
    py_ast.assign(%0) = %1
}

// -----

// CHECK-LABEL: py_ir.module
//       CHECK:  %[[D:.*]] = py_ir.loadvar "D" : !py_ir.undefined
//       CHECK:  %[[C:.*]] = py_ir.loadvar "C" : !py_ir.undefined
//       CHECK:  %[[T:.*]] = py_ir.getitem %[[C]] : !py_ir.undefined[%[[D]] : !py_ir.undefined] -> !py_ir.undefined
//       CHECK:  %[[B:.*]] = py_ir.loadvar "B" : !py_ir.undefined
//       CHECK:  %[[A:.*]] = py_ir.loadvar "A" : !py_ir.undefined
//       CHECK:  py_ir.setitem %[[A]] : !py_ir.undefined[%[[B]] : !py_ir.undefined] = %[[T]] : !py_ir.undefined
py_ast.module {
  %0 = py_ast.name "A"
  %1 = py_ast.name "B"
  %2 = py_ast.subscript %0[%1]
  %3 = py_ast.name "C"
  %4 = py_ast.name "D"
  %5 = py_ast.subscript %3[%4]
  py_ast.assign(%2) = %5
}

// -----

// CHECK-LABEL: py_ir.module
//       CHECK:  %[[C:.*]] = py_ir.loadvar "C" : !py_ir.undefined
//       CHECK:  %[[D:.*]] = py_ir.loadvar "D" : !py_ir.undefined
//       CHECK:  %[[E:.*]] = py_ir.loadvar "E" : !py_ir.undefined
//       CHECK:  %[[S:.*]] = py_ir.slice(%[[C]] !py_ir.undefined : %[[D]] !py_ir.undefined : %[[E]] !py_ir.undefined) -> !py_ir.undefined
//       CHECK:  %[[B:.*]] = py_ir.loadvar "B" : !py_ir.undefined
//       CHECK:  %[[R:.*]] = py_ir.getitem %[[B]] : !py_ir.undefined[%[[S]] : !py_ir.undefined] -> !py_ir.undefined
//       CHECK:  py_ir.storevar "A" %[[R]] : !py_ir.undefined
py_ast.module {
  %0 = py_ast.name "A"
  %1 = py_ast.name "B"
  %2 = py_ast.name "C"
  %3 = py_ast.name "D"
  %4 = py_ast.name "E"
  %5 = py_ast.slice(%2 : %3 : %4)
  %6 = py_ast.subscript %1[%5]
  py_ast.assign(%0) = %6
}

// -----

// CHECK-LABEL: py_ir.module
//       CHECK:  %[[C:.*]] = py_ir.loadvar "C" : !py_ir.undefined
//       CHECK:  %[[E:.*]] = py_ir.loadvar "E" : !py_ir.undefined
//       CHECK:  %[[S:.*]] = py_ir.slice(%[[C]] !py_ir.undefined : : %[[E]] !py_ir.undefined) -> !py_ir.undefined
//       CHECK:  %[[B:.*]] = py_ir.loadvar "B" : !py_ir.undefined
//       CHECK:  %[[R:.*]] = py_ir.getitem %[[B]] : !py_ir.undefined[%[[S]] : !py_ir.undefined] -> !py_ir.undefined
//       CHECK:  py_ir.storevar "A" %[[R]] : !py_ir.undefined
py_ast.module {
  %0 = py_ast.name "A"
  %1 = py_ast.name "B"
  %2 = py_ast.name "C"
  %4 = py_ast.name "E"
  %5 = py_ast.slice(%2 : : %4)
  %6 = py_ast.subscript %1[%5]
  py_ast.assign(%0) = %6
}

// -----

// CHECK-LABEL: py_ir.module
//       CHECK:  %[[C:.*]] = py_ir.loadvar "C" : !py_ir.undefined
//       CHECK:  %[[T:.*]] = py_ir.getattr %0 : !py_ir.undefined attr "D" -> !py_ir.undefined
//       CHECK:  %[[A:.*]] = py_ir.loadvar "A" : !py_ir.undefined
//       CHECK:  py_ir.setattr %[[A]] : !py_ir.undefined attr "B" = %[[T]] : !py_ir.undefined
py_ast.module {
  %0 = py_ast.name "A"
  %1 = py_ast.attribute %0 attr "B"
  %2 = py_ast.name "C"
  %3 = py_ast.attribute %2 attr "D"
  py_ast.assign(%1) = %3
}
// -----

// CHECK-LABEL: py_ir.module
//       CHECK:  %[[A:.*]] = py_ir.loadvar "A" : !py_ir.undefined
//       CHECK:  %[[B:.*]] = py_ir.loadvar "B" : !py_ir.undefined
//       CHECK:  %[[L:.*]] = py_ir.make_list %[[A]], %[[B]] : !py_ir.undefined, !py_ir.undefined -> !py_ir.undefined
//       CHECK:  py_ir.storevar "C" %[[L]] : !py_ir.undefined
py_ast.module {
  %0 = py_ast.name "A"
  %1 = py_ast.name "B"
  %2 = py_ast.list %0, %1
  %3 = py_ast.name "C"
  py_ast.assign(%3) = %2
}

// -----

// CHECK-LABEL: py_ir.module
//       CHECK:  %[[C:.*]] = py_ir.loadvar "C" : !py_ir.undefined
//       CHECK:  %[[D:.*]] = py_ir.loadvar "D" : !py_ir.undefined
//       CHECK:  %[[T1:.*]] = py_ir.tuple_pack %[[C]], %[[D]] : !py_ir.undefined, !py_ir.undefined -> !py_ir.undefined
//       CHECK:  %[[T2:.*]]:2 = py_ir.tuple_unpack %2 : !py_ir.undefined -> !py_ir.undefined, !py_ir.undefined
//       CHECK:  py_ir.storevar "A" %[[T2]]#0 : !py_ir.undefined
//       CHECK:  py_ir.storevar "B" %[[T2]]#1 : !py_ir.undefined
py_ast.module {
  %0 = py_ast.name "A"
  %1 = py_ast.name "B"
  %2 = py_ast.tuple %0, %1
  %3 = py_ast.name "C"
  %4 = py_ast.name "D"
  %5 = py_ast.tuple %3, %4
  py_ast.assign(%2) = %5
}

// -----

// CHECK-LABEL: py_ir.module
//   CHECK-DAG:  %[[A:.*]] = py_ir.loadvar "A" : !py_ir.undefined
//   CHECK-DAG:  %[[B:.*]] = py_ir.loadvar "B" : !py_ir.undefined
//       CHECK:  %[[R:.*]] = py_ir.binop %[[A]] : !py_ir.undefined add %[[B]] : !py_ir.undefined -> !py_ir.undefined
//       CHECK:  py_ir.storevar "C" %[[R]] : !py_ir.undefined
py_ast.module {
  %0 = py_ast.name "C"
  %1 = py_ast.name "A"
  %2 = py_ast.name "B"
  %3 = py_ast.binop %1 add %2
  py_ast.assign(%0) = %3
}

// -----

// CHECK-LABEL: py_ir.module
//   CHECK-DAG:  %[[A:.*]] = py_ir.loadvar "A" : !py_ir.undefined
//   CHECK-DAG:  %[[B:.*]] = py_ir.loadvar "B" : !py_ir.undefined
//       CHECK:  %[[R:.*]] = py_ir.cmp %[[A]] : !py_ir.undefined lt %[[B]] : !py_ir.undefined -> !py_ir.undefined
//       CHECK:  py_ir.storevar "C" %[[R]] : !py_ir.undefined
py_ast.module {
  %0 = py_ast.name "C"
  %1 = py_ast.name "A"
  %2 = py_ast.name "B"
  %3 = py_ast.compare %1 [2] %2
  py_ast.assign(%0) = %3
}

// -----

// CHECK-LABEL: py_ir.module
//   CHECK-DAG:  %[[A:.*]] = py_ir.loadvar "A" : !py_ir.undefined
//   CHECK-DAG:  %[[B:.*]] = py_ir.loadvar "B" : !py_ir.undefined
//       CHECK:  %[[R1:.*]] = py_ir.cmp %[[A]] : !py_ir.undefined lt %[[B]] : !py_ir.undefined -> !py_ir.undefined
//       CHECK:  %[[C:.*]] = py_ir.loadvar "C" : !py_ir.undefined
//       CHECK:  %[[R2:.*]] = py_ir.cmp %[[B]] : !py_ir.undefined le %[[C]] : !py_ir.undefined -> !py_ir.undefined
//       CHECK:  %[[R:.*]] = py_ir.binop %[[R1]] : !py_ir.undefined bool_and %[[R2]] : !py_ir.undefined -> !py_ir.undefined
//       CHECK:  py_ir.storevar "D" %[[R]] : !py_ir.undefined
py_ast.module {
  %0 = py_ast.name "D"
  %1 = py_ast.name "A"
  %2 = py_ast.name "B"
  %3 = py_ast.name "C"
  %4 = py_ast.compare %1 [2, 3] %2, %3
  py_ast.assign(%0) = %4
}

// -----

// CHECK-LABEL: py_ir.module
//   CHECK-DAG:  %[[A:.*]] = py_ir.loadvar "A" : !py_ir.undefined
//   CHECK-DAG:  %[[B:.*]] = py_ir.loadvar "B" : !py_ir.undefined
//       CHECK:  %[[R:.*]] = py_ir.binop %[[A]] : !py_ir.undefined bool_and %[[B]] : !py_ir.undefined -> !py_ir.undefined
//       CHECK:  py_ir.storevar "C" %[[R]] : !py_ir.undefined
py_ast.module {
  %1 = py_ast.name "C"
  %2 = py_ast.name "A"
  %3 = py_ast.name "B"
  %4 = py_ast.bool_op and, %2, %3
  py_ast.assign(%1) = %4
}

// -----

// CHECK-LABEL: py_ir.module
//   CHECK-DAG:  %[[A:.*]] = py_ir.loadvar "A" : !py_ir.undefined
//   CHECK-DAG:  %[[B:.*]] = py_ir.loadvar "B" : !py_ir.undefined
//       CHECK:  %[[R1:.*]] = py_ir.binop %[[A]] : !py_ir.undefined bool_or %[[B]] : !py_ir.undefined -> !py_ir.undefined
//       CHECK:  %[[C:.*]] = py_ir.loadvar "C" : !py_ir.undefined
//       CHECK:  %[[R2:.*]] = py_ir.binop %[[R1]] : !py_ir.undefined bool_or %[[C]] : !py_ir.undefined -> !py_ir.undefined
//       CHECK:  py_ir.storevar "D" %[[R2]] : !py_ir.undefined
py_ast.module {
  %2 = py_ast.name "D"
  %3 = py_ast.name "A"
  %4 = py_ast.name "B"
  %5 = py_ast.name "C"
  %6 = py_ast.bool_op or, %3, %4, %5
  py_ast.assign(%2) = %6
}

// -----

// CHECK-LABEL: py_ir.module
//   CHECK-DAG:  %[[A:.*]] = py_ir.loadvar "A" : !py_ir.undefined
//   CHECK-DAG:  %[[B:.*]] = py_ir.loadvar "B" : !py_ir.undefined
//       CHECK:  %[[R:.*]] = py_ir.inplace_binop %[[A]] : !py_ir.undefined add %[[B]] : !py_ir.undefined -> !py_ir.undefined
//       CHECK:  py_ir.storevar "A" %[[R]] : !py_ir.undefined
py_ast.module {
  %0 = py_ast.name "B"
  %1 = py_ast.name "A"
  py_ast.aug_assign %1 add %0
}

// -----

// CHECK-LABEL: py_ir.module
//       CHECK: %[[A:.*]] = py_ir.loadvar "A" : !py_ir.undefined
//       CHECK: %[[R:.*]] = py_ir.unaryop usub %[[A]] : !py_ir.undefined -> !py_ir.undefined
py_ast.module {
  %0 = py_ast.name "A"
  %1 = py_ast.unaryop usub %0
  py_ast.expr %1
}

// -----

// CHECK-LABEL: py_ir.module
//       CHECK-DAG: %[[A:.*]] = py_ir.loadvar "A"
//       CHECK-DAG: %[[B:.*]] = py_ir.loadvar "B"
//       CHECK-DAG: %[[C:.*]] = py_ir.loadvar "C"
//       CHECK: %[[OpR:.*]] = py_ir.ifexp %[[A]] : !py_ir.undefined if %[[B]] : !py_ir.undefined else %[[C]] : !py_ir.undefined -> !py_ir.undefined
py_ast.module {
  %0 = py_ast.name "C"
  %1 = py_ast.name "A"
  %2 = py_ast.name "B"
  %3 = py_ast.ifexp %1 if %2 else %0
  py_ast.expr %3
}

// -----

// CHECK-LABEL: py_ir.module
//       CHECK:  py_ir.func "func"
//       CHECK:  %[[C1_I:.*]] = py_ir.constant 1 : i64
//       CHECK:  cf.br ^[[CONDBR:.*]]
//       CHECK:  ^[[CONDBR]]:
//       CHECK:  %[[A1:.*]] = py_ir.loadvar "A" : [[A1T:.*]]
//       CHECK:  %[[COND:.*]] = typing.cast %[[A1]] : [[A1T]] to i1
//       CHECK:  cf.cond_br %[[COND]], ^[[THENBR:.*]], ^[[ELSEBR:.*]]
//       CHECK:  ^[[THENBR]]:
//   CHECK-DAG:  %[[A2:.*]] = py_ir.loadvar "A" : [[A2T:.*]]
//   CHECK-DAG:  %[[C1:.*]] = typing.cast %[[C1_I]] : i64 to !py_ir.undefined
//       CHECK:  %[[R:.*]] = py_ir.binop %[[A2]] : [[A2T]] add %[[C1]] : !py_ir.undefined
//       CHECK:  py_ir.storevar "A" %[[R]]
//       CHECK:  cf.br ^[[CONDBR]]
//       CHECK:  ^[[ELSEBR]]:
//       CHECK:  py_ir.return
py_ast.module {
  py_ast.func "func"() {
    %2 = py_ast.name "A"
    py_ast.while %2 {
      %4 = py_ast.name "A"
      %5 = py_ast.name "A"
      %6 = py_ast.constant 1 : i64
      %7 = py_ast.binop %5 add %6
      py_ast.assign(%4) = %7
    }
    %3 = py_ast.constant #py_ast.none
    py_ast.return %3
  }
  %0 = py_ast.name "func"
  %1 = py_ast.call %0( keywords )
  py_ast.expr %1
}

// -----

// CHECK-LABEL: py_ir.module
//       CHECK:  py_ir.func
//       CHECK:    %[[A:.*]] = py_ir.loadvar "A" : !py_ir.undefined
//       CHECK:    %[[Iter:.*]] = py_ir.iter %[[A]] : !py_ir.undefined -> !py_ir.undefined
//       CHECK:    cf.br ^[[CondBr:.*]](%[[Iter]] : !py_ir.undefined)
//       CHECK:  ^[[CondBr]](%[[CondIter:.*]]: !py_ir.undefined)
//       CHECK:    %[[Value:.*]], %[[Valid:.*]], %[[NIter:.*]] = py_ir.next %[[CondIter]] : !py_ir.undefined -> !py_ir.undefined, i1, !py_ir.undefined
//       CHECK:    cf.cond_br %[[Valid]], ^[[BodyBr:.*]](%[[Value]], %[[NIter]] : !py_ir.undefined, !py_ir.undefined), ^[[RestBr:.*]]
//       CHECK:  ^[[BodyBr]](%[[BValue:.*]]: !py_ir.undefined, %[[BIter:.*]]: !py_ir.undefined)
//       CHECK:    py_ir.storevar "i" %[[BValue:.*]] : !py_ir.undefined
//       CHECK:    cf.br ^[[CondBr]](%[[BIter]] : !py_ir.undefined)
//       CHECK:  ^[[RestBr]]
py_ast.module {
  py_ast.func "func"() {
    %2 = py_ast.name "A"
    %3 = py_ast.name "i"
    py_ast.for %3 in %2 {
    }
    %4 = py_ast.constant #py_ast.none
    py_ast.return %4
  }
  %0 = py_ast.name "func"
  %1 = py_ast.call %0( keywords )
  py_ast.expr %1
}

// -----

// CHECK-LABEL: py_ir.module
//       CHECK:  py_ir.func
//       CHECK:    %[[A:.*]] = py_ir.loadvar "A" : !py_ir.undefined
//       CHECK:    %[[Iter:.*]] = py_ir.iter %[[A]] : !py_ir.undefined -> !py_ir.undefined
//       CHECK:    cf.br ^[[CondBr:.*]](%[[Iter]] : !py_ir.undefined)
//       CHECK:  ^[[CondBr]](%[[CIter:.*]]: !py_ir.undefined):
//       CHECK:    %[[Value:.*]], %[[Valid:.*]], %[[NIter:.*]] = py_ir.next %[[CIter]] : !py_ir.undefined -> !py_ir.undefined, i1, !py_ir.undefined
//       CHECK:    cf.cond_br %[[Valid]], ^[[BodyBr:.*]](%[[Value]], %[[NIter]] : !py_ir.undefined, !py_ir.undefined), ^bb3
//       CHECK:  ^[[BodyBr]](%[[IValue:.*]]: !py_ir.undefined, %[[BIter:.*]]: !py_ir.undefined):
//       CHECK:    py_ir.storevar "i" %[[IValue]] : !py_ir.undefined
//       CHECK:    %[[B:.*]] = py_ir.loadvar "B" : !py_ir.undefined
//       CHECK:    %[[BCond:.*]] = typing.cast %[[B]] : !py_ir.undefined to i1
//       CHECK:    cf.cond_br %[[BCond]], ^[[RestBr:.*]], ^[[CondBr]](%[[BIter:.*]] : !py_ir.undefined)
//       CHECK:  ^[[RestBr]]:
py_ast.module {
  py_ast.func "func"() {
    %2 = py_ast.name "A"
    %3 = py_ast.name "i"
    py_ast.for %3 in %2 {
      %5 = py_ast.name "B"
      py_ast.if %5 {
        py_ast.break
      } {
      }
    }
    %4 = py_ast.constant #py_ast.none
    py_ast.return %4
  }
  %0 = py_ast.name "func"
  %1 = py_ast.call %0( keywords )
  py_ast.expr %1
}

// -----

// CHECK-LABEL: py_ir.module
//       CHECK: py_ir.func
//       CHECK:   %[[A:.*]] = py_ir.loadvar "A" : !py_ir.undefined
//       CHECK:   %[[ITER:.*]] = py_ir.iter %[[A]] : !py_ir.undefined -> !py_ir.undefined
//       CHECK:   cf.br ^[[BB1:.*]](%[[ITER]] : !py_ir.undefined)
//       CHECK: ^[[BB1]](%[[ITER_ARG:.*]]: !py_ir.undefined):  // 3 preds: ^[[BB0:.*]], ^[[BB2:.*]], ^[[BB3:.*]]
//       CHECK:   %[[VALUE:.*]], %[[VALID:.*]], %[[NEXTITER:.*]] = py_ir.next %[[ITER_ARG]] : !py_ir.undefined -> !py_ir.undefined, i1, !py_ir.undefined
//       CHECK:   cf.cond_br %[[VALID]], ^[[BB2]](%[[VALUE]], %[[NEXTITER]] : !py_ir.undefined, !py_ir.undefined), ^[[BB4:.*]]
//       CHECK: ^[[BB2]](%[[STORE_ARG:.*]]: !py_ir.undefined, %[[CAST_ARG:.*]]: !py_ir.undefined):  // pred: ^[[BB1]]
//       CHECK:   py_ir.storevar "i" %[[STORE_ARG]] : !py_ir.undefined
//       CHECK:   %[[B:.*]] = py_ir.loadvar "B" : !py_ir.undefined
//       CHECK:   %[[CAST_RESULT:.*]] = typing.cast %[[B]] : !py_ir.undefined to i1
//       CHECK:   cf.cond_br %[[CAST_RESULT]], ^[[BB3]], ^[[BB1]](%[[CAST_ARG]] : !py_ir.undefined)
//       CHECK: ^[[BB3]]:  // pred: ^[[BB2]]
//       CHECK:   %[[C:.*]] = py_ir.loadvar "C" : !py_ir.undefined
//       CHECK:   %[[CAST_C:.*]] = typing.cast %[[C]] : !py_ir.undefined to i1
//       CHECK:   cf.cond_br %[[CAST_C]], ^[[BB4]], ^[[BB1]](%[[CAST_ARG]] : !py_ir.undefined)
//       CHECK: ^[[BB4]]:  // 2 preds: ^[[BB1]], ^[[BB3]]
py_ast.module {
  py_ast.func "func"() {
    %2 = py_ast.name "A"
    %3 = py_ast.name "i"
    py_ast.for %3 in %2 {
      %5 = py_ast.name "B"
      py_ast.if %5 {
        %6 = py_ast.name "C"
        py_ast.if %6 {
          py_ast.break
        } {
        }
      } {
      }
    }
    %4 = py_ast.constant #py_ast.none
    py_ast.return %4
  }
  %0 = py_ast.name "foo"
  %1 = py_ast.call %0( keywords )
  py_ast.expr %1
}

// -----

// CHECK-LABEL: py_ir.module
//       CHECK: py_ir.func
//       CHECK:   cf.br ^[[BB1:.*]]
//       CHECK: ^[[BB1]]:  // 2 preds: ^[[BB0:.*]], ^[[BB2:.*]]
//       CHECK:   %[[LOAD_A:.*]] = py_ir.loadvar "A" : !py_ir.undefined
//       CHECK:   %[[CAST_A:.*]] = typing.cast %[[LOAD_A]] : !py_ir.undefined to i1
//       CHECK:   cf.cond_br %[[CAST_A]], ^[[BB2]], ^[[BB3:.*]]
//       CHECK: ^[[BB2]]:  // pred: ^[[BB1]]
//       CHECK:   %[[LOAD_B:.*]] = py_ir.loadvar "B" : !py_ir.undefined
//       CHECK:   %[[CAST_B:.*]] = typing.cast %[[LOAD_B]] : !py_ir.undefined to i1
//       CHECK:   cf.cond_br %[[CAST_B]], ^[[BB3]], ^[[BB1]]
//       CHECK: ^[[BB3]]:  // 2 preds: ^[[BB1]], ^[[BB2]]
py_ast.module {
  py_ast.func "func"() {
    %2 = py_ast.name "A"
    py_ast.while %2 {
      %4 = py_ast.name "B"
      py_ast.if %4 {
        py_ast.break
      } {
      }
    }
    %3 = py_ast.constant #py_ast.none
    py_ast.return %3
  }
  %0 = py_ast.name "func"
  %1 = py_ast.call %0( keywords )
  py_ast.expr %1
}

// -----

// CHECK-LABEL: py_ir.module
//       CHECK: %[[FUNC:.*]] = py_ir.func "func"
//       CHECK:   cf.br ^[[BB1:.*]]
//       CHECK: ^[[BB1]]:  // 3 preds: ^[[BB0:.*]], ^[[BB2:.*]], ^[[BB3:.*]]
//       CHECK:   %[[LOAD_A:.*]] = py_ir.loadvar "A" : !py_ir.undefined
//       CHECK:   %[[CAST_A:.*]] = typing.cast %[[LOAD_A]] : !py_ir.undefined to i1
//       CHECK:   cf.cond_br %[[CAST_A]], ^[[BB2]], ^[[BB4:.*]]
//       CHECK: ^[[BB2]]:  // pred: ^[[BB1]]
//       CHECK:   %[[LOAD_B:.*]] = py_ir.loadvar "B" : !py_ir.undefined
//       CHECK:   %[[CAST_B:.*]] = typing.cast %[[LOAD_B]] : !py_ir.undefined to i1
//       CHECK:   cf.cond_br %[[CAST_B]], ^[[BB3]], ^[[BB1]]
//       CHECK: ^[[BB3]]:  // pred: ^[[BB2]]
//       CHECK:   %[[LOAD_C:.*]] = py_ir.loadvar "C" : !py_ir.undefined
//       CHECK:   %[[CAST_C:.*]] = typing.cast %[[LOAD_C]] : !py_ir.undefined to i1
//       CHECK:   cf.cond_br %[[CAST_C]], ^[[BB4]], ^[[BB1]]
//       CHECK: ^[[BB4]]:  // 2 preds: ^[[BB1]], ^[[BB3]]
py_ast.module {
  py_ast.func "func"() {
    %2 = py_ast.name "A"
    py_ast.while %2 {
      %4 = py_ast.name "B"
      py_ast.if %4 {
        %5 = py_ast.name "C"
        py_ast.if %5 {
          py_ast.break
        } {
        }
      } {
      }
    }
    %3 = py_ast.constant #py_ast.none
    py_ast.return %3
  }
  %0 = py_ast.name "func"
  %1 = py_ast.call %0( keywords )
  py_ast.expr %1
}

// -----

// CHECK-LABEL: py_ir.module
//       CHECK: %[[FUNC:.*]] = py_ir.func "func"
//       CHECK:   %[[CONST_I:.*]] = py_ir.constant 1 : i64
//       CHECK:   cf.br ^[[BB1:.*]]
//       CHECK: ^[[BB1]]:  // 3 preds: ^[[BB0:.*]], ^[[BB2:.*]], ^[[BB3:.*]]
//       CHECK:   %[[LOAD_A:.*]] = py_ir.loadvar "A" : !py_ir.undefined
//       CHECK:   %[[CAST_A:.*]] = typing.cast %[[LOAD_A]] : !py_ir.undefined to i1
//       CHECK:   cf.cond_br %[[CAST_A]], ^[[BB2]], ^[[BB4:.*]]
//       CHECK: ^[[BB2]]:  // pred: ^[[BB1]]
//       CHECK:   %[[LOAD_B:.*]] = py_ir.loadvar "B" : !py_ir.undefined
//       CHECK:   %[[CAST_B:.*]] = typing.cast %[[LOAD_B]] : !py_ir.undefined to i1
//       CHECK:   cf.cond_br %[[CAST_B]], ^[[BB1]], ^[[BB3]]
//       CHECK: ^[[BB3]]:  // pred: ^[[BB2]]
//   CHECK-DAG:   %[[LOAD_A2:.*]] = py_ir.loadvar "A" : !py_ir.undefined
//   CHECK-DAG:   %[[CONST:.*]] = typing.cast %[[CONST_I]] : i64 to !py_ir.undefined
//       CHECK:   %[[BINOP:.*]] = py_ir.binop %[[LOAD_A2]] : !py_ir.undefined sub %[[CONST]] : !py_ir.undefined -> !py_ir.undefined
//       CHECK:   py_ir.storevar "A" %[[BINOP]] : !py_ir.undefined
//       CHECK:   cf.br ^[[BB1]]
//       CHECK: ^[[BB4]]:  // pred: ^[[BB1]]
py_ast.module {
  py_ast.func "func"() {
    %2 = py_ast.name "A"
    py_ast.while %2 {
      %4 = py_ast.name "B"
      py_ast.if %4 {
        py_ast.continue
      } {
      }
      %5 = py_ast.name "A"
      %6 = py_ast.name "A"
      %7 = py_ast.constant 1 : i64
      %8 = py_ast.binop %6 sub %7
      py_ast.assign(%5) = %8
    }
    %3 = py_ast.constant #py_ast.none
    py_ast.return %3
  }
  %0 = py_ast.name "func"
  %1 = py_ast.call %0( keywords )
  py_ast.expr %1
}

// -----

// CHECK-LABEL: py_ir.module
//       CHECK: %[[FUNC:.*]] = py_ir.func "func"
//       CHECK:   %[[CONST_I:.*]] = py_ir.constant 1 : i64
//       CHECK:   %[[LOAD_A:.*]] = py_ir.loadvar "A" : !py_ir.undefined
//       CHECK:   %[[ITER:.*]] = py_ir.iter %[[LOAD_A]] : !py_ir.undefined -> !py_ir.undefined
//       CHECK:   cf.br ^[[BB1:.*]](%[[ITER]] : !py_ir.undefined)
//       CHECK: ^[[BB1]](%[[ITER_ARG:.*]]: !py_ir.undefined):  // 3 preds: ^[[BB0:.*]], ^[[BB2:.*]], ^[[BB3:.*]]
//       CHECK:   %[[VALUE:.*]], %[[VALID:.*]], %[[NEXTITER:.*]] = py_ir.next %[[ITER_ARG]] : !py_ir.undefined -> !py_ir.undefined, i1, !py_ir.undefined
//       CHECK:   cf.cond_br %[[VALID]], ^[[BB2]](%[[VALUE]], %[[NEXTITER]] : !py_ir.undefined, !py_ir.undefined), ^[[BB4:.*]]
//       CHECK: ^[[BB2]](%[[STORE_ARG:.*]]: !py_ir.undefined, %[[CAST_ARG:.*]]: !py_ir.undefined):  // pred: ^[[BB1]]
//       CHECK:   py_ir.storevar "i" %[[STORE_ARG]] : !py_ir.undefined
//       CHECK:   %[[LOAD_B:.*]] = py_ir.loadvar "B" : !py_ir.undefined
//       CHECK:   %[[CAST_B:.*]] = typing.cast %[[LOAD_B]] : !py_ir.undefined to i1
//       CHECK:   cf.cond_br %[[CAST_B]], ^[[BB1]](%[[CAST_ARG]] : !py_ir.undefined), ^[[BB3]]
//       CHECK: ^[[BB3]]:  // pred: ^[[BB2]]
//   CHECK-DAG:   %[[LOAD_C:.*]] = py_ir.loadvar "C" : !py_ir.undefined
//   CHECK-DAG:   %[[CONST:.*]] = typing.cast %[[CONST_I]] : i64 to !py_ir.undefined
//       CHECK:   %[[BINOP:.*]] = py_ir.binop %[[LOAD_C]] : !py_ir.undefined add %[[CONST]] : !py_ir.undefined -> !py_ir.undefined
//       CHECK:   cf.br ^[[BB1]](%[[CAST_ARG]] : !py_ir.undefined)
//       CHECK: ^[[BB4]]:  // pred: ^[[BB1]]
py_ast.module {
  py_ast.func "func"() {
    %2 = py_ast.name "A"
    %3 = py_ast.name "i"
    py_ast.for %3 in %2 {
      %5 = py_ast.name "B"
      py_ast.if %5 {
        py_ast.continue
      } {
      }
      %6 = py_ast.name "C"
      %7 = py_ast.constant 1 : i64
      %8 = py_ast.binop %6 add %7
      py_ast.expr %8
    }
    %4 = py_ast.constant #py_ast.none
    py_ast.return %4
  }
  %0 = py_ast.name "func"
  %1 = py_ast.call %0( keywords )
  py_ast.expr %1
}

// -----

// CHECK-LABEL: py_ir.module
//       CHECK:  py_ir.func "func"
//       CHECK:  %[[A:.*]] = py_ir.loadvar "A" : !py_ir.undefined
//       CHECK:  %[[COND:.*]] = typing.cast %[[A]] : !py_ir.undefined to i1
//       CHECK:  cf.cond_br %[[COND]], ^bb1, ^bb2
//       CHECK:  ^bb1:
//       CHECK:  %[[B:.*]] = py_ir.loadvar "B" : !py_ir.undefined
//       CHECK:  py_ir.return %[[B]] : !py_ir.undefined
//       CHECK:  ^bb2:
//       CHECK:  %[[C:.*]] = py_ir.loadvar "C" : !py_ir.undefined
//       CHECK:  py_ir.return %[[C]] : !py_ir.undefined
py_ast.module {
  py_ast.func "func"() {
    %3 = py_ast.name "A"
    py_ast.if %3 {
      %5 = py_ast.name "B"
      py_ast.return %5
    } {
      %5 = py_ast.name "C"
      py_ast.return %5
    }
    %4 = py_ast.constant #py_ast.none
    py_ast.return %4
  }
  %1 = py_ast.name "test"
  %2 = py_ast.call %1( keywords )
  py_ast.expr %2
}
