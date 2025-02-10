// RUN: hc-opt -split-input-file %s --hc-lower-subgroup-scope-pass | FileCheck %s


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
!expr3 = !typing<expr (!sym1, !sym3, !sym5, !sym6) -> s0 * s1 + s2 floordiv s3>
!expr4 = !typing<expr (!sym, !sym2, !sym4) -> s0 * s1 + s2>
!expr5 = !typing<expr (!sym2) -> 1>
!seq = !typing<sequence !sym, !sym1>
!seq1 = !typing<sequence !sym2, !sym3>
!seq2 = !typing<sequence !sym4, !sym5>
!seq3 = !typing<sequence !expr1, !expr2>
module attributes {kernel.group_id = #typing.type_attr<!seq> : !typing.value, kernel.group_shape = #typing.type_attr<!seq1> : !typing.value, kernel.local_id = #typing.type_attr<!seq2> : !typing.value, kernel.subgroup_id = #typing.type_attr<!expr> : !typing.value, kernel.subgroup_size = #typing.type_attr<!sym6> : !typing.value, kernel.work_offset = #typing.type_attr<!seq3> : !typing.value} {
  func.func @test(%arg0: !hkernel<current_group 2>, %arg1: !hkernel<buffer <"W" x "H"> x !literal>, %arg2: !hkernel<buffer <"W" x "H"> x !literal>) {
    %0 = hkernel.materialize_expr !expr3
    %1 = hkernel.materialize_expr !expr4
    %2 = hkernel.materialize_expr !sym6
    %3 = hkernel.materialize_expr !expr5
    hkernel.env_region #hkernel.subgroup_scope {
      %4 = hkernel.make_slice(%1  !expr4 :    :   ) -> !hkernel<slice !expr4 : none : none>
      %5 = hkernel.make_slice(%0  !expr3 :    :   ) -> !hkernel<slice !expr3 : none : none>
      %6 = hkernel.subview %arg1 : !hkernel<buffer <"W" x "H"> x !literal>[%4, %5] : !hkernel<slice !expr4 : none : none>, !hkernel<slice !expr3 : none : none> -> !hkernel<buffer <("W" - "$GROUP_ID0" * "$GROUP_SHAPE0" - "$LOCAL_ID0") x ("H" - "$GROUP_ID1" * "$GROUP_SHAPE1" - "$LOCAL_ID1" floordiv "$SUBGROUP_SIZE")> x !literal>
      %7 = hkernel.load %6 : !hkernel<buffer <("W" - "$GROUP_ID0" * "$GROUP_SHAPE0" - "$LOCAL_ID0") x ("H" - "$GROUP_ID1" * "$GROUP_SHAPE1" - "$LOCAL_ID1" floordiv "$SUBGROUP_SIZE")> x !literal>[%3, %2] : !expr5, !sym6 -> !hkernel<tensor <(1) x "$SUBGROUP_SIZE"> x !literal>
      %8 = hkernel.make_slice(%1  !expr4 :    :   ) -> !hkernel<slice !expr4 : none : none>
      %9 = hkernel.make_slice(%0  !expr3 :    :   ) -> !hkernel<slice !expr3 : none : none>
      %10 = hkernel.subview %arg2 : !hkernel<buffer <"W" x "H"> x !literal>[%8, %9] : !hkernel<slice !expr4 : none : none>, !hkernel<slice !expr3 : none : none> -> !hkernel<buffer <("W" - "$GROUP_ID0" * "$GROUP_SHAPE0" - "$LOCAL_ID0") x ("H" - "$GROUP_ID1" * "$GROUP_SHAPE1" - "$LOCAL_ID1" floordiv "$SUBGROUP_SIZE")> x !literal>
      hkernel.store %10 : !hkernel<buffer <("W" - "$GROUP_ID0" * "$GROUP_SHAPE0" - "$LOCAL_ID0") x ("H" - "$GROUP_ID1" * "$GROUP_SHAPE1" - "$LOCAL_ID1" floordiv "$SUBGROUP_SIZE")> x !literal> = %7 : !hkernel<tensor <(1) x "$SUBGROUP_SIZE"> x !literal>
    }
    return
  }
}


//   CHECK-DAG: ![[LIT0:.*]] = !typing<literal "DT">
//   CHECK-DAG: ![[LIT1:.*]] = !typing<literal 1 : index>
// CHECK-LABEL: func @test
//   CHECK-DAG: %[[V0:.*]] = hkernel.materialize_expr ![[LIT1]]
//       CHECK:   hkernel.env_region #hkernel.workitem_scope
//       CHECK:     hkernel.subview
//       CHECK:     %[[R:.*]] = hkernel.load %{{.*}} : !hkernel<buffer <("W" - "$GROUP_ID0" * "$GROUP_SHAPE0" - "$LOCAL_ID0") x (-("$GROUP_ID1" * "$GROUP_SHAPE1") - "$LOCAL_ID1" mod "$SUBGROUP_SIZE" - "$LOCAL_ID1" floordiv "$SUBGROUP_SIZE" + "H")> x ![[LIT0]]>[%[[V0]], %[[V0]]] : ![[LIT1]], ![[LIT1]] -> !hkernel<tensor <1 x 1> x ![[LIT0]]>
//       CHECK:     hkernel.subview
//       CHECK:     hkernel.store %{{.*}} : !hkernel<buffer <("W" - "$GROUP_ID0" * "$GROUP_SHAPE0" - "$LOCAL_ID0") x (-("$GROUP_ID1" * "$GROUP_SHAPE1") - "$LOCAL_ID1" mod "$SUBGROUP_SIZE" - "$LOCAL_ID1" floordiv "$SUBGROUP_SIZE" + "H")> x ![[LIT0]]> = %[[R]] : !hkernel<tensor <1 x 1> x ![[LIT0]]>
