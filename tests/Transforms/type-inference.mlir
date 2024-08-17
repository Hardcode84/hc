// RUN: hc-opt -split-input-file %s --hc-py-type-inference-pass | FileCheck %s


typing.type_resolver ["py_ir.loadvar", "CurrentGroup"] {
  %0 = typing.make_ident "CurrentGroup" []
  typing.type_resolver_return %0
}

//       CHECK: ![[ID:.*]] = !typing<ident "CurrentGroup">
// CHECK-LABEL: py_ir.module {
//       CHECK:  py_ir.func "func"
//       CHECK:  ^bb0(%[[ARG:.*]]: ![[ID]]):
//       CHECK:  %[[R:.*]] = py_ir.getattr %[[ARG]] : ![[ID]] attr "foo" -> !py_ir.undefined
//       CHECK:  py_ir.return %[[R]] : !py_ir.undefined

py_ir.module {
  %0 = py_ir.loadvar "CurrentGroup" : !py_ir.undefined
  %1 = py_ir.func "func" (group:%0) : !py_ir.undefined capture () -> !py_ir.undefined {
  ^bb0(%arg0: !py_ir.undefined):
    %2 = py_ir.getattr %arg0 : !py_ir.undefined attr "foo" -> !py_ir.undefined
    py_ir.return %2 : !py_ir.undefined
  }
  %3 = py_ir.call %1 : !py_ir.undefined  () -> !py_ir.undefined
}

// -----

typing.type_resolver ["py_ir.loadvar", "i64"] {
  %0 = typing.type_constant #typing.type_attr<i64> : !typing.value
  typing.type_resolver_return %0
}

// CHECK-LABEL: py_ir.module {
//       CHECK:  py_ir.func "func"
//       CHECK:  ^bb0(%[[ARG:.*]]: i64):
//       CHECK:  %[[R:.*]] = py_ir.getattr %[[ARG]] : i64 attr "foo" -> !py_ir.undefined
//       CHECK:  py_ir.return %[[R]] : !py_ir.undefined

py_ir.module {
  %0 = py_ir.loadvar "i64" : !py_ir.undefined
  %1 = py_ir.func "func" (group:%0) : !py_ir.undefined capture () -> !py_ir.undefined {
  ^bb0(%arg0: !py_ir.undefined):
    %2 = py_ir.getattr %arg0 : !py_ir.undefined attr "foo" -> !py_ir.undefined
    py_ir.return %2 : !py_ir.undefined
  }
  %3 = py_ir.call %1 : !py_ir.undefined  () -> !py_ir.undefined
}

// -----

typing.type_resolver ["py_ir.loadvar"] {
  %0 = typing.get_attr "name"
  typing.type_resolver_return %0
}

//       CHECK:  ![[LIT:.*]] = !typing<literal "i64">
// CHECK-LABEL: py_ir.module {
//       CHECK:  py_ir.func "func"
//       CHECK:  ^bb0(%[[ARG:.*]]: ![[LIT]]):
//       CHECK:  %[[R:.*]] = py_ir.getattr %[[ARG]] : ![[LIT]] attr "foo" -> !py_ir.undefined
//       CHECK:  py_ir.return %[[R]] : !py_ir.undefined

py_ir.module {
  %0 = py_ir.loadvar "i64" : !py_ir.undefined
  %1 = py_ir.func "func" (group:%0) : !py_ir.undefined capture () -> !py_ir.undefined {
  ^bb0(%arg0: !py_ir.undefined):
    %2 = py_ir.getattr %arg0 : !py_ir.undefined attr "foo" -> !py_ir.undefined
    py_ir.return %2 : !py_ir.undefined
  }
  %3 = py_ir.call %1 : !py_ir.undefined  () -> !py_ir.undefined
}

// -----

func.func @test_func(%0: !typing.value) -> (!typing.value) {
  return %0 : !typing.value
}

typing.type_resolver ["py_ir.loadvar", "CurrentGroup"] {
  %0 = typing.make_ident "CurrentGroup" []
  %1 = func.call @test_func(%0) : (!typing.value) -> (!typing.value)
  typing.type_resolver_return %1
}

//       CHECK: ![[ID:.*]] = !typing<ident "CurrentGroup">
// CHECK-LABEL: py_ir.module {
//       CHECK:  py_ir.func "func"
//       CHECK:  ^bb0(%[[ARG:.*]]: ![[ID]]):
//       CHECK:  %[[R:.*]] = py_ir.getattr %[[ARG]] : ![[ID]] attr "foo" -> !py_ir.undefined
//       CHECK:  py_ir.return %[[R]] : !py_ir.undefined

py_ir.module {
  %0 = py_ir.loadvar "CurrentGroup" : !py_ir.undefined
  %1 = py_ir.func "func" (group:%0) : !py_ir.undefined capture () -> !py_ir.undefined {
  ^bb0(%arg0: !py_ir.undefined):
    %2 = py_ir.getattr %arg0 : !py_ir.undefined attr "foo" -> !py_ir.undefined
    py_ir.return %2 : !py_ir.undefined
  }
  %3 = py_ir.call %1 : !py_ir.undefined  () -> !py_ir.undefined
}

// -----

//       CHECK: ![[LIT:.*]] = !typing<literal 1 : i64>
// CHECK-LABEL: py_ir.module {
//       CHECK:  py_ir.func "func"
//       CHECK:  ^bb0(%[[ARG:.*]]: ![[LIT]]):

py_ir.module {
  %0 = py_ir.constant 1 : i64
  %1 = typing.cast %0 : i64 to !py_ir.undefined
  %2 = py_ir.func "func" (A:%1) : !py_ir.undefined capture () -> !py_ir.undefined {
  ^bb0(%arg0: !py_ir.undefined):
    %4 = py_ir.none
    py_ir.return %4 : none
  }
  %3 = py_ir.call %2 : !py_ir.undefined  () -> !py_ir.undefined
}

// -----

typing.type_resolver ["py_ir.loadvar", "Foo"] {
  %0 = typing.make_ident "Foo" []
  typing.type_resolver_return %0
}

typing.type_resolver ["py_ir.loadvar", "Bar"] {
  %str = typing.type_constant #typing.type_attr<!typing<literal "Bar"> > : !typing.value
  %0 = typing.make_symbol %str
  typing.type_resolver_return %0
}

typing.type_resolver ["py_ir.loadvar", "Baz"] {
  %str = typing.type_constant #typing.type_attr<!typing<literal "Baz"> > : !typing.value
  %0 = typing.make_symbol %str
  typing.type_resolver_return %0
}

typing.type_resolver ["py_ir.tuple_pack"] {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %n = typing.get_num_args
  %seq = typing.create_seq
  cf.br ^bb1(%c0, %seq : index, !typing.value)
^bb1(%idx: index, %s1: !typing.value):
  %cond = arith.cmpi slt, %idx, %n : index
  %next = arith.addi %idx, %c1 : index
  cf.cond_br %cond, ^bb2, ^bb3(%s1: !typing.value)
^bb2:
  %arg = typing.get_arg %idx
  %new_seq = typing.append_seq %s1 %arg
  cf.br ^bb1(%next, %new_seq : index, !typing.value)
^bb3(%s2: !typing.value):
  %0 = typing.make_ident "Tuple" ["Elements"] : %s2
  typing.type_resolver_return %0
}

typing.type_resolver ["py_ir.getitem"] {
  %c1 = arith.constant 1 : index
  %tuple = typing.get_arg %c1
  %shape = typing.get_ident_param %tuple "Elements"
  %0 = typing.make_ident "Foo" ["Elements"] : %shape
  typing.type_resolver_return %0
}

//       CHECK: ![[SYM1:.*]] = !typing<symbol "Bar">
//       CHECK: ![[SYM2:.*]] = !typing<symbol "Baz">
//       CHECK: ![[SEQ:.*]] = !typing<sequence ![[SYM1]], ![[SYM2]]>
//       CHECK: ![[ID:.*]] = !typing<ident "Foo" : "Elements" -> ![[SEQ]]>
// CHECK-LABEL: py_ir.module {
//       CHECK:  py_ir.func "func"
//       CHECK:  ^bb0(%[[ARG:.*]]: ![[ID]]):

py_ir.module {
  %0 = py_ir.loadvar "Bar" : !py_ir.undefined
  %1 = py_ir.loadvar "Baz" : !py_ir.undefined
  %2 = py_ir.tuple_pack %0, %1 : !py_ir.undefined, !py_ir.undefined -> !py_ir.undefined
  %3 = py_ir.loadvar "Foo" : !py_ir.undefined
  %4 = py_ir.getitem %3 : !py_ir.undefined[%2 : !py_ir.undefined] -> !py_ir.undefined
  %5 = py_ir.func "func" (A:%4) : !py_ir.undefined capture () -> !py_ir.undefined {
  ^bb0(%arg0: !py_ir.undefined):
    %7 = py_ir.none
    py_ir.return %7 : none
  }
  %6 = py_ir.call %5 : !py_ir.undefined  () -> !py_ir.undefined
}

// -----

typing.type_resolver ["py_ir.loadvar", "B"] {
  %0 = typing.make_ident "B" []
  typing.type_resolver_return %0
}

typing.type_resolver ["py_ir.loadvar", "C"] {
  %0 = typing.make_ident "C" []
  typing.type_resolver_return %0
}

typing.type_resolver "join_types" {
  %0 = typing.make_ident "D" []
  typing.type_resolver_return %0
}

//   CHECK-DAG: ![[ID1:.*]] = !typing<ident "B">
//   CHECK-DAG: ![[ID2:.*]] = !typing<ident "C">
//   CHECK-DAG: ![[ID3:.*]] = !typing<ident "D">
// CHECK-LABEL: py_ir.module {
//       CHECK:  py_ir.func "func"
//       CHECK:  ^bb0(%{{.*}}: !py_ir.undefined, %[[B:.*]]: ![[ID1]], %[[C:.*]]: ![[ID2]]):
//       CHECK:  cf.cond_br %{{.*}}, ^bb1(%[[B]] : ![[ID1]]), ^bb2(%[[C]] : ![[ID2]])
//       CHECK:  ^bb1(%[[C1:.*]]: ![[ID1]]):
//       CHECK:  %[[C2:.*]] = typing.cast %[[C1]] : ![[ID1]] to ![[ID3]]
//       CHECK:  cf.br ^bb3(%[[C2]] : ![[ID3]])
//       CHECK:  ^bb2(%[[B1:.*]]: ![[ID2]]):
//       CHECK:  %[[B2:.*]] = typing.cast %[[B1]] : ![[ID2]] to ![[ID3]]
//       CHECK:  cf.br ^bb3(%[[B2]] : ![[ID3]])
//       CHECK:  ^bb3(%[[RES:.*]]: ![[ID3]]):
//       CHECK:  py_ir.return %[[RES]] : ![[ID3]]

py_ir.module {
  %0 = py_ir.loadvar "A" : !py_ir.undefined
  %2 = py_ir.loadvar "B" : !py_ir.undefined
  %3 = py_ir.loadvar "C" : !py_ir.undefined
  %4 = py_ir.func "func" () capture (A:%0, B:%2, C:%3) : !py_ir.undefined, !py_ir.undefined, !py_ir.undefined -> !py_ir.undefined {
  ^bb0(%arg0: !py_ir.undefined, %arg2: !py_ir.undefined, %arg3: !py_ir.undefined):
    %6 = typing.cast %arg0 : !py_ir.undefined to i1
    cf.cond_br %6, ^bb1(%arg2 : !py_ir.undefined), ^bb2(%arg3 : !py_ir.undefined)
  ^bb1(%8: !py_ir.undefined):
    cf.br ^bb3(%8 : !py_ir.undefined)
  ^bb2(%10: !py_ir.undefined):
    cf.br ^bb3(%10 : !py_ir.undefined)
  ^bb3(%11: !py_ir.undefined):
    py_ir.return %11 : !py_ir.undefined
  }
  %5 = py_ir.call %4 : !py_ir.undefined  () -> !py_ir.undefined
}

// -----

typing.type_resolver ["py_ir.loadvar", "B"] {
  %0 = typing.make_ident "B" []
  typing.type_resolver_return %0
}

typing.type_resolver ["py_ir.loadvar", "C"] {
  %0 = typing.make_ident "C" []
  typing.type_resolver_return %0
}

typing.type_resolver "join_types" {
  %0 = typing.make_ident "D" []
  typing.type_resolver_return %0
}

//   CHECK-DAG: ![[ID1:.*]] = !typing<ident "B">
//   CHECK-DAG: ![[ID2:.*]] = !typing<ident "C">
//   CHECK-DAG: ![[ID3:.*]] = !typing<ident "D">
// CHECK-LABEL: py_ir.module {
//       CHECK:  py_ir.func "func"
//       CHECK:  ^bb0(%{{.*}}: !py_ir.undefined, %[[ARG1:.*]]: ![[ID1]], %[[ARG2:.*]]: ![[ID2]]):
//       CHECK:  %[[RES:.*]] = typing.resolve %{{.*}}, %[[ARG1]], %[[ARG2]] : i1, ![[ID1]], ![[ID2]] -> ![[ID3]] {
//       CHECK:  %[[R:.*]] = arith.select %{{.*}}, %{{.*}}, %{{.*}} : !py_ir.undefined
//       CHECK:  typing.resolve_yield %[[R]] : !py_ir.undefined
//       CHECK:  py_ir.return %[[RES]] : ![[ID3]]

py_ir.module {
  %0 = py_ir.loadvar "A" : !py_ir.undefined
  %1 = py_ir.loadvar "B" : !py_ir.undefined
  %2 = py_ir.loadvar "C" : !py_ir.undefined
  %3 = py_ir.func "func" () capture (A:%0, B:%1, C:%2) : !py_ir.undefined, !py_ir.undefined, !py_ir.undefined -> !py_ir.undefined {
  ^bb0(%arg0: !py_ir.undefined, %arg1: !py_ir.undefined, %arg2: !py_ir.undefined):
    %5 = typing.cast %arg0 : !py_ir.undefined to i1
    %6 = arith.select %5, %arg1, %arg2 : !py_ir.undefined
    py_ir.return %6 : !py_ir.undefined
  }
  %4 = py_ir.call %3 : !py_ir.undefined  () -> !py_ir.undefined
}

// -----

typing.type_resolver ["py_ir.loadvar", "i64"] {
  %0 = typing.type_constant #typing.type_attr<!typing<sequence i32, i64, i16>> : !typing.value
  %c1 = arith.constant 1 : index
  %1 = typing.get_seq_element %0 [ %c1 ]
  typing.type_resolver_return %1
}

// CHECK-LABEL: py_ir.module {
//       CHECK:  py_ir.func "func"
//       CHECK:  ^bb0(%[[ARG:.*]]: i64):
//       CHECK:  %[[R:.*]] = py_ir.getattr %[[ARG]] : i64 attr "foo" -> !py_ir.undefined
//       CHECK:  py_ir.return %[[R]] : !py_ir.undefined

py_ir.module {
  %0 = py_ir.loadvar "i64" : !py_ir.undefined
  %1 = py_ir.func "func" (group:%0) : !py_ir.undefined capture () -> !py_ir.undefined {
  ^bb0(%arg0: !py_ir.undefined):
    %2 = py_ir.getattr %arg0 : !py_ir.undefined attr "foo" -> !py_ir.undefined
    py_ir.return %2 : !py_ir.undefined
  }
  %3 = py_ir.call %1 : !py_ir.undefined  () -> !py_ir.undefined
}

// -----

typing.type_resolver ["py_ir.loadvar", "i64"] {
  %0 = typing.type_constant #typing.type_attr<!typing<sequence i32, i64, i16>> : !typing.value
  %1 = typing.get_seq_size %0
  %c2 = arith.constant 2 : index
  %2 = arith.subi %1, %c2 : index
  %3 = typing.get_seq_element %0 [ %2 ]
  typing.type_resolver_return %3
}

// CHECK-LABEL: py_ir.module {
//       CHECK:  py_ir.func "func"
//       CHECK:  ^bb0(%[[ARG:.*]]: i64):
//       CHECK:  %[[R:.*]] = py_ir.getattr %[[ARG]] : i64 attr "foo" -> !py_ir.undefined
//       CHECK:  py_ir.return %[[R]] : !py_ir.undefined

py_ir.module {
  %0 = py_ir.loadvar "i64" : !py_ir.undefined
  %1 = py_ir.func "func" (group:%0) : !py_ir.undefined capture () -> !py_ir.undefined {
  ^bb0(%arg0: !py_ir.undefined):
    %2 = py_ir.getattr %arg0 : !py_ir.undefined attr "foo" -> !py_ir.undefined
    py_ir.return %2 : !py_ir.undefined
  }
  %3 = py_ir.call %1 : !py_ir.undefined  () -> !py_ir.undefined
}

// -----

typing.type_resolver ["py_ir.loadvar", "B"] {
  %0 = typing.make_ident "B" []
  typing.type_resolver_return %0
}

typing.type_resolver ["py_ir.loadvar", "C"] {
  %0 = typing.make_ident "C" []
  typing.type_resolver_return %0
}

typing.type_resolver "join_types" {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = typing.get_arg %c0
  %1 = typing.get_arg %c1
  %2 = typing.make_union %0, %1
  typing.type_resolver_return %2
}

//   CHECK-DAG: ![[ID1:.*]] = !typing<ident "B">
//   CHECK-DAG: ![[ID2:.*]] = !typing<ident "C">
//   CHECK-DAG: ![[ID3:.*]] = !typing<union ![[ID1]], ![[ID2]]>
// CHECK-LABEL: py_ir.module {
//       CHECK:  py_ir.func "func"
//       CHECK:  ^bb0(%{{.*}}: !py_ir.undefined, %[[ARG1:.*]]: ![[ID1]], %[[ARG2:.*]]: ![[ID2]]):
//       CHECK:  %[[RES:.*]] = typing.resolve %{{.*}}, %[[ARG1]], %[[ARG2]] : i1, ![[ID1]], ![[ID2]] -> ![[ID3]] {
//       CHECK:  %[[R:.*]] = arith.select %{{.*}}, %{{.*}}, %{{.*}} : !py_ir.undefined
//       CHECK:  typing.resolve_yield %[[R]] : !py_ir.undefined
//       CHECK:  py_ir.return %[[RES]] : ![[ID3]]

py_ir.module {
  %0 = py_ir.loadvar "A" : !py_ir.undefined
  %1 = py_ir.loadvar "B" : !py_ir.undefined
  %2 = py_ir.loadvar "C" : !py_ir.undefined
  %3 = py_ir.func "func" () capture (A:%0, B:%1, C:%2) : !py_ir.undefined, !py_ir.undefined, !py_ir.undefined -> !py_ir.undefined {
  ^bb0(%arg0: !py_ir.undefined, %arg1: !py_ir.undefined, %arg2: !py_ir.undefined):
    %5 = typing.cast %arg0 : !py_ir.undefined to i1
    %6 = arith.select %5, %arg1, %arg2 : !py_ir.undefined
    py_ir.return %6 : !py_ir.undefined
  }
  %4 = py_ir.call %3 : !py_ir.undefined  () -> !py_ir.undefined
}

// -----

typing.type_resolver ["py_ir.loadvar", "B"] {
  %0 = typing.make_ident "B" []
  typing.type_resolver_return %0
}

typing.type_resolver ["py_ir.loadvar", "D"] {
  %0 = typing.make_ident "D" []
  typing.type_resolver_return %0
}

typing.type_resolver ["py_ir.loadvar", "E"] {
  %0 = typing.make_ident "E" []
  typing.type_resolver_return %0
}

typing.type_resolver "join_types" {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = typing.get_arg %c0
  %1 = typing.get_arg %c1
  %2 = typing.make_union %0, %1
  typing.type_resolver_return %2
}

//   CHECK-DAG: ![[ID1:.*]] = !typing<ident "B">
//   CHECK-DAG: ![[ID2:.*]] = !typing<ident "D">
//   CHECK-DAG: ![[ID3:.*]] = !typing<ident "E">
//   CHECK-DAG: ![[U:.*]] = !typing<union ![[ID1]], ![[ID3]], ![[ID2]]>
// CHECK-LABEL: py_ir.module {
//       CHECK:  py_ir.func "func"
//       CHECK:  ^bb0(%[[ARG0:.*]]: !py_ir.undefined, %[[ARG1:.*]]: ![[ID1]], %[[ARG2:.*]]: !py_ir.undefined, %[[ARG3:.*]]: ![[ID2]], %[[ARG4:.*]]: ![[ID3]]):
//       CHECK:  %[[COND:.*]] = typing.cast %[[ARG0]] : !py_ir.undefined to i1
//       CHECK:  %[[U1:.*]] = typing.cast %[[ARG1]] : ![[ID1]] to ![[U]]
//       CHECK:  cf.cond_br %[[COND]], ^bb1(%[[U1]] : ![[U]]), ^bb2(%[[ARG2]], %[[ARG3]], %[[ARG4]] : !py_ir.undefined, ![[ID2]], ![[ID3]])
//       CHECK:  ^bb1(%[[U2:.*]]: ![[U]]):
//       CHECK:  py_ir.return %[[U2]] : ![[U]]
//       CHECK:  ^bb2(%[[COND2:.*]]: !py_ir.undefined, %[[ARG5:.*]]: ![[ID2]], %[[ARG6:.*]]: ![[ID3]]):
//       CHECK:  %[[COND3:.*]] = typing.cast %[[COND2]] : !py_ir.undefined to i1
//       CHECK:  %[[A1:.*]] = typing.cast %11 : ![[ID2]] to ![[U]]
//       CHECK:  %[[A2:.*]] = typing.cast %12 : ![[ID3]] to ![[U]]
//       CHECK:  cf.cond_br %[[COND3]], ^bb1(%[[A1]] : ![[U]]), ^bb1(%[[A2]] : ![[U]])
//       CHECK:  }

py_ir.module {
  %0 = py_ir.loadvar "A" : !py_ir.undefined
  %1 = py_ir.loadvar "B" : !py_ir.undefined
  %2 = py_ir.loadvar "C" : !py_ir.undefined
  %3 = py_ir.loadvar "D" : !py_ir.undefined
  %4 = py_ir.loadvar "E" : !py_ir.undefined
  %5 = py_ir.func "func" () capture (A:%0, B:%1, C:%2, D:%3, E:%4) : !py_ir.undefined, !py_ir.undefined, !py_ir.undefined, !py_ir.undefined, !py_ir.undefined -> !py_ir.undefined {
  ^bb0(%arg0: !py_ir.undefined, %arg1: !py_ir.undefined, %arg2: !py_ir.undefined, %arg3: !py_ir.undefined, %arg4: !py_ir.undefined):
    %7 = typing.cast %arg0 : !py_ir.undefined to i1
    cf.cond_br %7, ^bb1(%arg1 : !py_ir.undefined), ^bb2(%arg2, %arg3, %arg4 : !py_ir.undefined, !py_ir.undefined, !py_ir.undefined)
  ^bb1(%8: !py_ir.undefined):  // 3 preds: ^bb0, ^bb2, ^bb2
    py_ir.return %8 : !py_ir.undefined
  ^bb2(%9: !py_ir.undefined, %10: !py_ir.undefined, %11: !py_ir.undefined):  // pred: ^bb0
    %12 = typing.cast %9 : !py_ir.undefined to i1
    cf.cond_br %12, ^bb1(%10 : !py_ir.undefined), ^bb1(%11 : !py_ir.undefined)
  }
  %6 = py_ir.call %5 : !py_ir.undefined  () -> !py_ir.undefined
}

// -----

module attributes {test.test = #typing.type_attr<i32> : !typing.value} {
typing.type_resolver ["py_ir.loadvar", "CurrentGroup"] {
  %0 = typing.get_global_attr "test.test"
  typing.type_resolver_return %0
}

// CHECK-LABEL: py_ir.module {
//       CHECK:  py_ir.func "func"
//       CHECK:  ^bb0(%[[ARG:.*]]: i32):
//       CHECK:  %[[R:.*]] = py_ir.getattr %[[ARG]] : i32 attr "foo" -> !py_ir.undefined
//       CHECK:  py_ir.return %[[R]] : !py_ir.undefined

py_ir.module {
  %0 = py_ir.loadvar "CurrentGroup" : !py_ir.undefined
  %1 = py_ir.func "func" (group:%0) : !py_ir.undefined capture () -> !py_ir.undefined {
  ^bb0(%arg0: !py_ir.undefined):
    %2 = py_ir.getattr %arg0 : !py_ir.undefined attr "foo" -> !py_ir.undefined
    py_ir.return %2 : !py_ir.undefined
  }
  %3 = py_ir.call %1 : !py_ir.undefined  () -> !py_ir.undefined
}

}
