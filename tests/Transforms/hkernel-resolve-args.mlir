// RUN: hc-opt -allow-unregistered-dialect -split-input-file %s --hc-resolve-args-pass | FileCheck %s

!sym = !typing<symbol "$GROUP_SHAPE0">
!sym1 = !typing<symbol "W">
!sym2 = !typing<symbol "$GROUP_SHAPE1">
!sym3 = !typing<symbol "H">
!sym4 = !typing<symbol "$GROUP_ID0">
!sym5 = !typing<symbol "$GROUP_ID1">
!sym6 = !typing<symbol "$LOCAL_ID0">
!sym7 = !typing<symbol "$LOCAL_ID1">
!sym8 = !typing<symbol "$SUBGROUP_SIZE">
!expr = !typing<expr (!sym, !sym1) -> s1 ceildiv s0>
!expr1 = !typing<expr (!sym2, !sym3) -> s1 ceildiv s0>
!expr2 = !typing<expr (!sym7, !sym8) -> s0 floordiv s1>
!expr3 = !typing<expr (!sym4, !sym) -> s0 * s1>
!expr4 = !typing<expr (!sym5, !sym2) -> s0 * s1>
!expr6 = !typing<expr (!sym4, !sym, !sym6) -> s0 * s1 + s2>
!seq = !typing<sequence !sym4, !sym5>
!seq1 = !typing<sequence !sym, !sym2>
!seq2 = !typing<sequence !sym6, !sym7>
!seq3 = !typing<sequence !sym1, !sym3>
!seq4 = !typing<sequence !expr, !expr1>
!seq5 = !typing<sequence !expr3, !expr4>
module attributes {kernel.group_count = #typing.type_attr<!seq4> : !typing.value, kernel.group_id = #typing.type_attr<!seq> : !typing.value, kernel.group_shape = #typing.type_attr<!seq1> : !typing.value, kernel.local_id = #typing.type_attr<!seq2> : !typing.value, kernel.subgroup_id = #typing.type_attr<!expr2> : !typing.value, kernel.subgroup_size = #typing.type_attr<!sym8> : !typing.value, kernel.work_offset = #typing.type_attr<!seq5> : !typing.value, kernel.work_shape = #typing.type_attr<!seq3> : !typing.value} {
  func.func @test(%arg1: !hkernel<buffer <"W" x "H"> x f16>) {
    %1 = hkernel.materialize_expr !expr6
    hkernel.env_region #hkernel.workitem_scope {
      "test.test1"(%arg1) : (!hkernel<buffer <"W" x "H"> x f16>) -> ()
      "test.test2"(%1) : (!expr6) -> ()
    }
    return
  }
}


//   CHECK-DAG: #[[MAP0:.*]] = affine_map<()[s0, s1] -> (s1 ceildiv s0)>
//   CHECK-DAG: #[[MAP1:.*]] = affine_map<()[s0, s1, s2] -> (s0 * s1 + s2)>
//   CHECK-DAG: ![[SYM1:.*]] = !typing<symbol "$GROUP_ID0">
//   CHECK-DAG: ![[SYM2:.*]] = !typing<symbol "$GROUP_SHAPE0">
//   CHECK-DAG: ![[SYM3:.*]] = !typing<symbol "$LOCAL_ID0">
//   CHECK-DAG: ![[EXPR:.*]] = !typing<expr (![[SYM1]], ![[SYM2]], ![[SYM3]]) -> s0 * s1 + s2>
// CHECK-LABEL: func @test
//  CHECK-SAME: (%[[ARG_ORIG:.*]]: memref<?x?xf16, strided<[?, ?], offset: ?>>)
//   CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
//   CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
//       CHECK: %[[ARG:.*]] = builtin.unrealized_conversion_cast %[[ARG_ORIG]] : memref<?x?xf16, strided<[?, ?], offset: ?>> to !hkernel<buffer <"W" x "H"> x f16>
//       CHECK: %[[DIM0:.*]] = memref.dim %[[ARG_ORIG]], %[[C0]] : memref<?x?xf16, strided<[?, ?], offset: ?>>
//       CHECK: %[[DIM1:.*]] = memref.dim %[[ARG_ORIG]], %[[C1]] : memref<?x?xf16, strided<[?, ?], offset: ?>>
//       CHECK: %[[BLOCK:.*]]:2 = hkernel.suggest_block_size %[[DIM1]], %[[DIM0]] : index, index
//       CHECK: %[[Y_BLOCK:.*]] = affine.apply #[[MAP0]]()[%[[BLOCK]]#1, %[[DIM0]]]
//       CHECK: %[[X_BLOCK:.*]] = affine.apply #[[MAP0]]()[%[[BLOCK]]#0, %[[DIM1]]]
//       CHECK: gpu.launch blocks(%{{.*}}, %[[BLY:.*]], %{{.*}}) in (%{{.*}} = %[[X_BLOCK]], %{{.*}} = %[[Y_BLOCK]], %{{.*}} = %[[C1]]) threads(%{{.*}}, %[[THRY:.*]], %{{.*}}) in (%{{.*}} = %[[BLOCK]]#0, %{{.*}} = %[[BLOCK]]#1, %{{.*}} = %[[C1]])
//   CHECK-NOT: hkernel.env_region
//       CHECK: %[[V1:.*]] = affine.apply #map1()[%[[BLY]], %[[BLOCK]]#1, %[[THRY]]]
//       CHECK: %[[V2:.*]] = builtin.unrealized_conversion_cast %[[V1]] : index to ![[EXPR]]
//       CHECK: "test.test1"(%[[ARG]]) : (!hkernel<buffer <"W" x "H"> x f16>) -> ()
//       CHECK: "test.test2"(%[[V2]]) : (![[EXPR]]) -> ()
//       CHECK: gpu.terminator
//       CHECK: return
