// RUN: hc-opt -split-input-file %s --hc-lower-workgroup-scope-pass | FileCheck %s


!literal = !typing<literal "DT">
!sym = !typing<symbol "$GROUP_ID0">
!sym1 = !typing<symbol "$GROUP_ID1">
!sym2 = !typing<symbol "$GROUP_SHAPE0">
!sym3 = !typing<symbol "$GROUP_SHAPE1">
!sym4 = !typing<symbol "$LOCAL_ID0">
!sym5 = !typing<symbol "$LOCAL_ID1">
!sym6 = !typing<symbol "$SUBGROUP_SIZE">
!expr = !typing<expr (!sym5, !sym6) -> s0 floordiv s1>
!expr1 = !typing<expr (!sym, !sym2) -> s0 * s1>
!expr2 = !typing<expr (!sym1, !sym3) -> s0 * s1>
!seq = !typing<sequence !sym, !sym1>
!seq1 = !typing<sequence !sym2, !sym3>
!seq2 = !typing<sequence !sym4, !sym5>
!seq3 = !typing<sequence !expr1, !expr2>
module attributes {kernel.group_id = #typing.type_attr<!seq> : !typing.value, kernel.group_shape = #typing.type_attr<!seq1> : !typing.value, kernel.local_id = #typing.type_attr<!seq2> : !typing.value, kernel.subgroup_id = #typing.type_attr<!expr> : !typing.value, kernel.subgroup_size = #typing.type_attr<!sym6> : !typing.value, kernel.work_offset = #typing.type_attr<!seq3> : !typing.value} {
  func.func @test(%arg0: !hkernel<current_group 2>, %arg1: !hkernel<buffer <"W" x "H"> x !literal>, %arg2: !hkernel<buffer <"W" x "H"> x !literal>) {
    %0 = hkernel.materialize_expr !sym3
    %1 = hkernel.materialize_expr !sym2
    %2 = hkernel.materialize_expr !expr2
    %3 = hkernel.materialize_expr !expr1
    hkernel.env_region #hkernel.workgroup_scope {
      %4 = hkernel.make_slice(%3  !expr1 :    :   ) -> !hkernel<slice !expr1 : none : none>
      %5 = hkernel.make_slice(%2  !expr2 :    :   ) -> !hkernel<slice !expr2 : none : none>
      %6 = hkernel.subview %arg1 : !hkernel<buffer <"W" x "H"> x !literal>[%4, %5] : !hkernel<slice !expr1 : none : none>, !hkernel<slice !expr2 : none : none> -> !hkernel<buffer <("W" - "$GROUP_ID0" * "$GROUP_SHAPE0") x ("H" - "$GROUP_ID1" * "$GROUP_SHAPE1")> x !literal>
      %7 = hkernel.load %6 : !hkernel<buffer <("W" - "$GROUP_ID0" * "$GROUP_SHAPE0") x ("H" - "$GROUP_ID1" * "$GROUP_SHAPE1")> x !literal>[%1, %0] : !sym2, !sym3 -> !hkernel<tensor <"$GROUP_SHAPE0" x "$GROUP_SHAPE1"> x !literal>
      %8 = hkernel.subview %arg2 : !hkernel<buffer <"W" x "H"> x !literal>[%4, %5] : !hkernel<slice !expr1 : none : none>, !hkernel<slice !expr2 : none : none> -> !hkernel<buffer <("W" - "$GROUP_ID0" * "$GROUP_SHAPE0") x ("H" - "$GROUP_ID1" * "$GROUP_SHAPE1")> x !literal>
      hkernel.store %8 : !hkernel<buffer <("W" - "$GROUP_ID0" * "$GROUP_SHAPE0") x ("H" - "$GROUP_ID1" * "$GROUP_SHAPE1")> x !literal> = %7 : !hkernel<tensor <"$GROUP_SHAPE0" x "$GROUP_SHAPE1"> x !literal>
    }
    return
  }
}


//   CHECK-DAG: ![[LIT:.*]] = !typing<literal "DT">
//   CHECK-DAG: ![[LIT1:.*]] = !typing<literal 1 : index>
//   CHECK-DAG: ![[SYM0:.*]] = !typing<symbol "$GROUP_ID0">
//   CHECK-DAG: ![[SYM1:.*]] = !typing<symbol "$GROUP_ID1">
//   CHECK-DAG: ![[SYM2:.*]] = !typing<symbol "$GROUP_SHAPE0">
//   CHECK-DAG: ![[SYM3:.*]] = !typing<symbol "$GROUP_SHAPE1">
//   CHECK-DAG: ![[SYM4:.*]] = !typing<symbol "$LOCAL_ID0">
//   CHECK-DAG: ![[SYM5:.*]] = !typing<symbol "$LOCAL_ID1">
//   CHECK-DAG: ![[SYM6:.*]] = !typing<symbol "$SUBGROUP_SIZE">
//   CHECK-DAG: ![[EXPR0:.*]] = !typing<expr (![[SYM5]], ![[SYM6]]) -> s0 floordiv s1>
//   CHECK-DAG: ![[EXPR1:.*]] = !typing<expr (![[SYM0]], ![[SYM2]]) -> s0 * s1>
//   CHECK-DAG: ![[EXPR2:.*]] = !typing<expr (![[SYM1]], ![[SYM3]]) -> s0 * s1>
//   CHECK-DAG: ![[EXPR3:.*]] = !typing<expr (![[SYM5]], ![[SYM6]]) -> (s0 floordiv s1) * s1>
// CHECK-LABEL: func @test
//  CHECK-SAME: (%{{.*}}: !hkernel<current_group 2>, %[[ARG1:.*]]: !hkernel<buffer <"W" x "H"> x ![[LIT]]>, %[[ARG2:.*]]: !hkernel<buffer <"W" x "H"> x ![[LIT]]>) {
//   CHECK-DAG:   %[[V0:.*]] = hkernel.materialize_expr ![[SYM6]]
//   CHECK-DAG:   %[[V1:.*]] = hkernel.materialize_expr ![[LIT1]]
//   CHECK-DAG:   %[[V2:.*]] = hkernel.materialize_expr ![[EXPR0]]
//   CHECK-DAG:   %[[V3:.*]] = hkernel.materialize_expr ![[SYM4]]
//   CHECK-DAG:   %[[V4:.*]] = hkernel.materialize_expr ![[EXPR2]]
//   CHECK-DAG:   %[[V5:.*]] = hkernel.materialize_expr ![[EXPR1]]
//       CHECK:   hkernel.env_region #hkernel.subgroup_scope {
//       CHECK:     %[[V6:.*]] = hkernel.make_slice(%[[V5]]  ![[EXPR1]] :    :   ) -> !hkernel<slice ![[EXPR1]] : none : none>
//       CHECK:     %[[V7:.*]] = hkernel.make_slice(%[[V4]]  ![[EXPR2]] :    :   ) -> !hkernel<slice ![[EXPR2]] : none : none>
//       CHECK:     %[[V8:.*]] = hkernel.subview %[[ARG1]] : !hkernel<buffer <"W" x "H"> x ![[LIT]]>[%[[V6]], %[[V7]]] : !hkernel<slice ![[EXPR1]] : none : none>, !hkernel<slice ![[EXPR2]] : none : none> -> !hkernel<buffer <("W" - "$GROUP_ID0" * "$GROUP_SHAPE0") x ("H" - "$GROUP_ID1" * "$GROUP_SHAPE1")> x ![[LIT]]>
//       CHECK:     %[[V9:.*]] = hkernel.make_slice(%[[V3]]  ![[SYM4]] :    :   ) -> !hkernel<slice ![[SYM4]] : none : none>
//       CHECK:     %[[V10:.*]] = hkernel.make_slice(%[[V2]]  ![[EXPR3]] :    :   ) -> !hkernel<slice ![[EXPR3]] : none : none>
//       CHECK:     %[[V11:.*]] = hkernel.subview %[[V8]] : !hkernel<buffer <("W" - "$GROUP_ID0" * "$GROUP_SHAPE0") x ("H" - "$GROUP_ID1" * "$GROUP_SHAPE1")> x ![[LIT]]>[%[[V9]], %[[V10]]] : !hkernel<slice ![[SYM4]] : none : none>, !hkernel<slice ![[EXPR3]] : none : none> -> !hkernel<buffer <(-("$GROUP_ID0" * "$GROUP_SHAPE0") - "$LOCAL_ID0" + "W") x ("H" - "$GROUP_ID1" * "$GROUP_SHAPE1" - ("$LOCAL_ID1" floordiv "$SUBGROUP_SIZE") * "$SUBGROUP_SIZE")> x ![[LIT]]>
//       CHECK:     %[[V12:.*]] = hkernel.load %[[V11]] : !hkernel<buffer <(-("$GROUP_ID0" * "$GROUP_SHAPE0") - "$LOCAL_ID0" + "W") x ("H" - "$GROUP_ID1" * "$GROUP_SHAPE1" - ("$LOCAL_ID1" floordiv "$SUBGROUP_SIZE") * "$SUBGROUP_SIZE")> x ![[LIT]]>[%[[V1]], %[[V0]]] : ![[LIT1]], ![[SYM6]] -> !hkernel<tensor <1 x "$SUBGROUP_SIZE"> x ![[LIT]]>
//       CHECK:     %[[V13:.*]] = hkernel.subview %[[ARG2]] : !hkernel<buffer <"W" x "H"> x ![[LIT]]>[%[[V6]], %[[V7]]] : !hkernel<slice ![[EXPR1]] : none : none>, !hkernel<slice ![[EXPR2]] : none : none> -> !hkernel<buffer <("W" - "$GROUP_ID0" * "$GROUP_SHAPE0") x ("H" - "$GROUP_ID1" * "$GROUP_SHAPE1")> x ![[LIT]]>
//       CHECK:     %[[V14:.*]] = hkernel.make_slice(%[[V3]]  ![[SYM4]] :    :   ) -> !hkernel<slice ![[SYM4]] : none : none>
//       CHECK:     %[[V15:.*]] = hkernel.make_slice(%[[V2]]  ![[EXPR3]] :    :   ) -> !hkernel<slice ![[EXPR3]] : none : none>
//       CHECK:     %[[V16:.*]] = hkernel.subview %[[V13]] : !hkernel<buffer <("W" - "$GROUP_ID0" * "$GROUP_SHAPE0") x ("H" - "$GROUP_ID1" * "$GROUP_SHAPE1")> x ![[LIT]]>[%[[V14]], %[[V15]]] : !hkernel<slice ![[SYM4]] : none : none>, !hkernel<slice ![[EXPR3]] : none : none> -> !hkernel<buffer <(-("$GROUP_ID0" * "$GROUP_SHAPE0") - "$LOCAL_ID0" + "W") x ("H" - "$GROUP_ID1" * "$GROUP_SHAPE1" - ("$LOCAL_ID1" floordiv "$SUBGROUP_SIZE") * "$SUBGROUP_SIZE")> x ![[LIT]]>
//       CHECK:     hkernel.store %[[V16]] : !hkernel<buffer <(-("$GROUP_ID0" * "$GROUP_SHAPE0") - "$LOCAL_ID0" + "W") x ("H" - "$GROUP_ID1" * "$GROUP_SHAPE1" - ("$LOCAL_ID1" floordiv "$SUBGROUP_SIZE") * "$SUBGROUP_SIZE")> x ![[LIT]]> = %[[V12]] : !hkernel<tensor <1 x "$SUBGROUP_SIZE"> x ![[LIT]]>
//       CHECK:   }
//       CHECK:   return
//       CHECK: }
