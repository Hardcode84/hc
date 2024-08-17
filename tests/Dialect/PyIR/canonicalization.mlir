// RUN: hc-opt -allow-unregistered-dialect %s -pass-pipeline='builtin.module(py_ir.module(canonicalize{test-convergence}))' -split-input-file | FileCheck %s

// CHECK-LABEL: py_ir.module {
//       CHECK: py_ir.func "func" (a:%[[A:.*]], b:%[[A]]) : !py_ir.undefined, !py_ir.undefined capture (c:%[[C:.*]], d:%[[D:.*]], e:%[[E:.*]])
//       CHECK: ^bb0(%[[ARG0:.*]]: !py_ir.undefined, %[[ARG1:.*]]: !py_ir.undefined, %[[ARG2:.*]]: !py_ir.undefined, %[[ARG3:.*]]: !py_ir.undefined, %[[ARG4:.*]]: !py_ir.undefined):
//       CHECK: py_ir.tuple_pack %[[ARG0]], %[[ARG1]], %[[ARG2]], %[[ARG3]], %[[ARG4]] : !py_ir.undefined, !py_ir.undefined, !py_ir.undefined, !py_ir.undefined, !py_ir.undefined -> !py_ir.undefined
//       CHECK: py_ir.module_end
py_ir.module {
  %0 = py_ir.empty_annotation
  %1 = typing.make_ident "i1" []
  %2 = typing.make_ident "i8" []
  %3 = typing.make_ident "i16" []
  %4 = py_ir.func "func" (a:%0, b:%0) : !py_ir.undefined, !py_ir.undefined capture (c:%1, d:%2, e:%3) : !typing.value, !typing.value, !typing.value -> !py_ir.undefined {
  ^bb0(%arg0: !py_ir.undefined, %arg1: !py_ir.undefined, %arg2: !py_ir.undefined, %arg3: !py_ir.undefined, %arg4: !py_ir.undefined):
    %5 = py_ir.tuple_pack %arg0, %arg1, %arg2, %arg3, %arg4 : !py_ir.undefined, !py_ir.undefined, !py_ir.undefined, !py_ir.undefined, !py_ir.undefined -> !py_ir.undefined
    py_ir.return %5 : !py_ir.undefined
  }
  py_ir.module_end %4 : !py_ir.undefined
}

// -----

// CHECK-LABEL: py_ir.module {
//       CHECK: py_ir.func "func" (a:%[[A:.*]], b:%[[A]]) : !py_ir.undefined, !py_ir.undefined capture (c:%[[C:.*]], d:%[[D:.*]], e:%[[E:.*]])
//       CHECK: ^bb0(%[[ARG0:.*]]: !py_ir.undefined, %[[ARG1:.*]]: !py_ir.undefined, %[[ARG2:.*]]: !py_ir.undefined, %[[ARG3:.*]]: !py_ir.undefined, %[[ARG4:.*]]: !py_ir.undefined):
//       CHECK: py_ir.tuple_pack %[[ARG0]], %[[ARG2]], %[[ARG3]], %[[ARG4]] : !py_ir.undefined, !py_ir.undefined, !py_ir.undefined, !py_ir.undefined -> !py_ir.undefined
//       CHECK: py_ir.module_end
py_ir.module {
  %0 = py_ir.empty_annotation
  %1 = typing.make_ident "i1" []
  %2 = typing.make_ident "i8" []
  %3 = typing.make_ident "i16" []
  %4 = py_ir.func "func" (a:%0, b:%0) : !py_ir.undefined, !py_ir.undefined capture (c:%1, d:%2, e:%3) : !typing.value, !typing.value, !typing.value -> !py_ir.undefined {
  ^bb0(%arg0: !py_ir.undefined, %arg1: !py_ir.undefined, %arg2: !py_ir.undefined, %arg3: !py_ir.undefined, %arg4: !py_ir.undefined):
    %5 = py_ir.tuple_pack %arg0, %arg2, %arg3, %arg4 : !py_ir.undefined, !py_ir.undefined, !py_ir.undefined, !py_ir.undefined -> !py_ir.undefined
    py_ir.return %5 : !py_ir.undefined
  }
  py_ir.module_end %4 : !py_ir.undefined
}

// -----

// CHECK-LABEL: py_ir.module {
//       CHECK: %{{.*}} = py_ir.func "func" (a:%[[A:.*]], b:%[[A]]) : !py_ir.undefined, !py_ir.undefined capture (c:%[[C:.*]], d:%[[D:.*]])
//       CHECK: ^bb0(%[[ARG0:.*]]: !py_ir.undefined, %[[ARG1:.*]]: !py_ir.undefined, %[[ARG2:.*]]: !py_ir.undefined, %[[ARG3:.*]]: !py_ir.undefined):
//       CHECK: %{{.*}} = py_ir.tuple_pack %[[ARG0]], %[[ARG1]], %[[ARG2]], %[[ARG3]] : !py_ir.undefined, !py_ir.undefined, !py_ir.undefined, !py_ir.undefined -> !py_ir.undefined
//       CHECK: py_ir.return
//       CHECK: py_ir.module_end
py_ir.module {
  %0 = py_ir.empty_annotation
  %1 = typing.make_ident "i1" []
  %2 = typing.make_ident "i8" []
  %3 = typing.make_ident "i16" []
  %4 = py_ir.func "func" (a:%0, b:%0) : !py_ir.undefined, !py_ir.undefined capture (c:%1, d:%2, e:%3) : !typing.value, !typing.value, !typing.value -> !py_ir.undefined {
  ^bb0(%arg0: !py_ir.undefined, %arg1: !py_ir.undefined, %arg2: !py_ir.undefined, %arg3: !py_ir.undefined, %arg4: !py_ir.undefined):
    %5 = py_ir.tuple_pack %arg0, %arg1, %arg2, %arg3 : !py_ir.undefined, !py_ir.undefined, !py_ir.undefined, !py_ir.undefined -> !py_ir.undefined
    py_ir.return %5 : !py_ir.undefined
  }
  py_ir.module_end %4 : !py_ir.undefined
}

// -----

// CHECK-LABEL: py_ir.module {
//       CHECK: py_ir.func "func" (a:%[[A:.*]], b:%[[A]]) : !py_ir.undefined, !py_ir.undefined capture (c:%[[C:.*]], e:%[[E:.*]])
//       CHECK: ^bb0(%[[ARG0:.*]]: !py_ir.undefined, %[[ARG1:.*]]: !py_ir.undefined, %[[ARG2:.*]]: !py_ir.undefined, %[[ARG4:.*]]: !py_ir.undefined):
//       CHECK: py_ir.tuple_pack %[[ARG0]], %[[ARG1]], %[[ARG2]], %[[ARG4]] : !py_ir.undefined, !py_ir.undefined, !py_ir.undefined, !py_ir.undefined -> !py_ir.undefined
//       CHECK: py_ir.module_end
py_ir.module {
  %0 = py_ir.empty_annotation
  %1 = typing.make_ident "i1" []
  %2 = typing.make_ident "i8" []
  %3 = typing.make_ident "i16" []
  %4 = py_ir.func "func" (a:%0, b:%0) : !py_ir.undefined, !py_ir.undefined capture (c:%1, d:%2, e:%3) : !typing.value, !typing.value, !typing.value -> !py_ir.undefined {
  ^bb0(%arg0: !py_ir.undefined, %arg1: !py_ir.undefined, %arg2: !py_ir.undefined, %arg3: !py_ir.undefined, %arg4: !py_ir.undefined):
    %5 = py_ir.tuple_pack %arg0, %arg1, %arg2, %arg4 : !py_ir.undefined, !py_ir.undefined, !py_ir.undefined, !py_ir.undefined -> !py_ir.undefined
    py_ir.return %5 : !py_ir.undefined
  }
  py_ir.module_end %4 : !py_ir.undefined
}

// -----

// CHECK-LABEL: py_ir.module {
//       CHECK: py_ir.func "func" (a:%[[A:.*]], b:%[[A]]) : !py_ir.undefined, !py_ir.undefined capture (d:%[[D:.*]], e:%[[E:.*]])
//       CHECK: ^bb0(%[[ARG0:.*]]: !py_ir.undefined, %[[ARG1:.*]]: !py_ir.undefined, %[[ARG3:.*]]: !py_ir.undefined, %[[ARG4:.*]]: !py_ir.undefined):
//       CHECK: py_ir.tuple_pack %[[ARG0]], %[[ARG1]], %[[ARG3]], %[[ARG4]] : !py_ir.undefined, !py_ir.undefined, !py_ir.undefined, !py_ir.undefined -> !py_ir.undefined
//       CHECK: py_ir.module_end
py_ir.module {
  %0 = py_ir.empty_annotation
  %1 = typing.make_ident "i1" []
  %2 = typing.make_ident "i8" []
  %3 = typing.make_ident "i16" []
  %4 = py_ir.func "func" (a:%0, b:%0) : !py_ir.undefined, !py_ir.undefined capture (c:%1, d:%2, e:%3) : !typing.value, !typing.value, !typing.value -> !py_ir.undefined {
  ^bb0(%arg0: !py_ir.undefined, %arg1: !py_ir.undefined, %arg2: !py_ir.undefined, %arg3: !py_ir.undefined, %arg4: !py_ir.undefined):
    %5 = py_ir.tuple_pack %arg0, %arg1, %arg3, %arg4 : !py_ir.undefined, !py_ir.undefined, !py_ir.undefined, !py_ir.undefined -> !py_ir.undefined
    py_ir.return %5 : !py_ir.undefined
  }
  py_ir.module_end %4 : !py_ir.undefined
}

// -----

// CHECK-LABEL: py_ir.module {
//       CHECK: py_ir.func "func" (a:%[[A:.*]], b:%[[A]]) : !py_ir.undefined, !py_ir.undefined capture (c:%[[C:.*]])
//       CHECK: ^bb0(%[[ARG0:.*]]: !py_ir.undefined, %[[ARG1:.*]]: !py_ir.undefined, %[[ARG2:.*]]: !py_ir.undefined):
//       CHECK: py_ir.tuple_pack %[[ARG0]], %[[ARG1]], %[[ARG2]] : !py_ir.undefined, !py_ir.undefined, !py_ir.undefined -> !py_ir.undefined
//       CHECK: py_ir.module_end
py_ir.module {
  %0 = py_ir.empty_annotation
  %1 = typing.make_ident "i1" []
  %2 = typing.make_ident "i8" []
  %3 = typing.make_ident "i16" []
  %4 = py_ir.func "func" (a:%0, b:%0) : !py_ir.undefined, !py_ir.undefined capture (c:%1, d:%2, e:%3) : !typing.value, !typing.value, !typing.value -> !py_ir.undefined {
  ^bb0(%arg0: !py_ir.undefined, %arg1: !py_ir.undefined, %arg2: !py_ir.undefined, %arg3: !py_ir.undefined, %arg4: !py_ir.undefined):
    %5 = py_ir.tuple_pack %arg0, %arg1, %arg2 : !py_ir.undefined, !py_ir.undefined, !py_ir.undefined -> !py_ir.undefined
    py_ir.return %5 : !py_ir.undefined
  }
  py_ir.module_end %4 : !py_ir.undefined
}

// -----

// CHECK-LABEL: py_ir.module {
//       CHECK: py_ir.func "func" (a:%[[A:.*]], b:%[[A]]) : !py_ir.undefined, !py_ir.undefined capture (e:%[[E:.*]])
//       CHECK: ^bb0(%[[ARG0:.*]]: !py_ir.undefined, %[[ARG1:.*]]: !py_ir.undefined, %[[ARG4:.*]]: !py_ir.undefined):
//       CHECK: py_ir.tuple_pack %[[ARG0]], %[[ARG1]], %[[ARG4]] : !py_ir.undefined, !py_ir.undefined, !py_ir.undefined -> !py_ir.undefined
//       CHECK: py_ir.module_end
py_ir.module {
  %0 = py_ir.empty_annotation
  %1 = typing.make_ident "i1" []
  %2 = typing.make_ident "i8" []
  %3 = typing.make_ident "i16" []
  %4 = py_ir.func "func" (a:%0, b:%0) : !py_ir.undefined, !py_ir.undefined capture (c:%1, d:%2, e:%3) : !typing.value, !typing.value, !typing.value -> !py_ir.undefined {
  ^bb0(%arg0: !py_ir.undefined, %arg1: !py_ir.undefined, %arg2: !py_ir.undefined, %arg3: !py_ir.undefined, %arg4: !py_ir.undefined):
    %5 = py_ir.tuple_pack %arg0, %arg1, %arg4 : !py_ir.undefined, !py_ir.undefined, !py_ir.undefined -> !py_ir.undefined
    py_ir.return %5 : !py_ir.undefined
  }
  py_ir.module_end %4 : !py_ir.undefined
}

// -----

// CHECK-LABEL: py_ir.module {
//       CHECK: py_ir.func "func" (a:%[[A:.*]], b:%[[A]]) : !py_ir.undefined, !py_ir.undefined capture ()
//       CHECK: ^bb0(%[[ARG0:.*]]: !py_ir.undefined, %[[ARG1:.*]]: !py_ir.undefined):
//       CHECK: py_ir.tuple_pack %[[ARG0]], %[[ARG1]] : !py_ir.undefined, !py_ir.undefined -> !py_ir.undefined
//       CHECK: py_ir.module_end
py_ir.module {
  %0 = py_ir.empty_annotation
  %1 = typing.make_ident "i1" []
  %2 = typing.make_ident "i8" []
  %3 = typing.make_ident "i16" []
  %4 = py_ir.func "func" (a:%0, b:%0) : !py_ir.undefined, !py_ir.undefined capture (c:%1, d:%2, e:%3) : !typing.value, !typing.value, !typing.value -> !py_ir.undefined {
  ^bb0(%arg0: !py_ir.undefined, %arg1: !py_ir.undefined, %arg2: !py_ir.undefined, %arg3: !py_ir.undefined, %arg4: !py_ir.undefined):
    %5 = py_ir.tuple_pack %arg0, %arg1 : !py_ir.undefined, !py_ir.undefined -> !py_ir.undefined
    py_ir.return %5 : !py_ir.undefined
  }
  py_ir.module_end %4 : !py_ir.undefined
}

// -----

// CHECK-LABEL: py_ir.module {
//       CHECK: py_ir.func "func" () capture () -> !py_ir.undefined {
//       CHECK:   %[[R:.*]] = typing.type_constant #typing.type_attr<i32> : !typing.value
//       CHECK:   py_ir.return %[[R]] : !typing.value
//       CHECK: py_ir.module_end
py_ir.module {
  %2 = typing.type_constant #typing.type_attr<i32> : !typing.value
  %9 = py_ir.func "func" () capture (i32:%2) : !typing.value -> !py_ir.undefined {
  ^bb0(%arg0: !typing.value):
    py_ir.return %arg0 : !typing.value
  }
  py_ir.module_end %9 : !py_ir.undefined
}

// -----

// CHECK-LABEL: py_ir.module {
//       CHECK: py_ir.func "func" () capture () -> !py_ir.undefined {
//       CHECK:   %[[R:.*]] = typing.type_constant #typing.type_attr<i32> : !typing.value
//       CHECK:   %[[C:.*]] = typing.cast %[[R]] : !typing.value to none
//       CHECK:   py_ir.return %[[C]] : none
//       CHECK: py_ir.module_end
py_ir.module {
  %2 = typing.type_constant #typing.type_attr<i32> : !typing.value
  %3 = typing.cast %2 : !typing.value to none
  %9 = py_ir.func "func" () capture (i32:%3) : none -> !py_ir.undefined {
  ^bb0(%arg0: none):
    py_ir.return %arg0 : none
  }
  py_ir.module_end %9 : !py_ir.undefined
}

// -----

// CHECK-LABEL: py_ir.module {
//       CHECK: py_ir.static_func @func () type () -> none {
//       CHECK: %[[R:.*]] = py_ir.none
//       CHECK: py_ir.return %[[R]] : none
//       CHECK: }
//       CHECK: %{{.*}} = py_ir.static_call @func : () -> none
py_ir.module {
  py_ir.static_func @func () type () -> none {
    %0 = py_ir.none
    py_ir.return %0 : none
  }
  %1 = py_ir.sym_constant @func : !py_ir.undefined
  %2 = py_ir.call %1 : !py_ir.undefined () -> none
}
