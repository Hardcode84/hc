// RUN: hc-opt -split-input-file %s --hc-lower-hkernel-ops-pass --canonicalize | FileCheck %s



func.func @test(%arg1: !hkernel<tensor <"W" x "H"> x f16>) -> !hkernel<tensor <"W" x "H"> x f16> {
  return %arg1 : !hkernel<tensor <"W" x "H"> x f16>
}

// CHECK-LABEL: func @test
//  CHECK-SAME: (%[[ARG:.*]]: tuple<memref<?x?xf16, strided<[?, ?], offset: ?>, #gpu.address_space<workgroup>>, memref<?x?xi1, strided<[?, ?], offset: ?>, #gpu.address_space<workgroup>>>)
//       CHECK: return %[[ARG]] : tuple<memref<?x?xf16, strided<[?, ?], offset: ?>, #gpu.address_space<workgroup>>, memref<?x?xi1, strided<[?, ?], offset: ?>, #gpu.address_space<workgroup>>>

// -----

func.func @test(%arg1: !typing<symbol "W">) -> !hkernel<slice !typing<symbol "W"> : none : none> {
  %1 = hkernel.make_slice(%arg1  !typing<symbol "W"> :    :   ) -> !hkernel<slice !typing<symbol "W"> : none : none>
  return %1 : !hkernel<slice !typing<symbol "W"> : none : none>
}

// CHECK-LABEL: func @test
//  CHECK-SAME: (%[[ARG:.*]]: index)
//   CHECK-DAG:  %[[P:.*]] = ub.poison : none
//       CHECK:  %[[R:.*]] = hkernel.make_tuple %[[ARG]], %[[P]], %[[P]] : index, none, none -> tuple<index, none, none>
//       CHECK:  return %[[R]] : tuple<index, none, none>

// -----

func.func @test(%arg1: !hkernel<buffer <"W"> x f16>, %arg2: !hkernel<slice !typing<symbol "H"> : none : none>) -> !hkernel<buffer <"W1"> x f16> {
  %1 = hkernel.subview %arg1 : !hkernel<buffer <"W"> x f16>[%arg2] : !hkernel<slice !typing<symbol "H"> : none : none> -> !hkernel<buffer <"W1"> x f16>
  return %1 : !hkernel<buffer <"W1"> x f16>
}

//       CHECK: ![[SYM:.*]] = !typing<symbol "W">
// CHECK-LABEL: func @test
//  CHECK-SAME: (%[[ARG1:.*]]: memref<?xf16, strided<[?], offset: ?>>, %[[ARG2:.*]]: tuple<index, none, none>)
//   CHECK-DAG:  %[[C0:.*]] = arith.constant 0 : index
//   CHECK-DAG:  %[[DIM:.*]] = hkernel.materialize_expr ![[SYM]]
//   CHECK-DAG:  %[[DIM_IDX:.*]] = builtin.unrealized_conversion_cast %[[DIM]] : ![[SYM]] to index
//       CHECK:  %[[LOWER:.*]] = hkernel.tuple_extract %[[ARG2]] : tuple<index, none, none>[%[[C0]]] -> index
//       CHECK:  %[[OFFSET:.*]], %[[SIZE:.*]], %[[STRIDE:.*]] = hkernel.resolve_slice(%[[LOWER]] : : ) %[[DIM_IDX]]
//       CHECK:  %[[SUBVIEW:.*]] = memref.subview %[[ARG1]][%[[OFFSET]]] [%[[SIZE]]] [%[[STRIDE]]] : memref<?xf16, strided<[?], offset: ?>> to memref<?xf16, strided<[?], offset: ?>>
//       CHECK:  return %[[SUBVIEW]] : memref<?xf16, strided<[?], offset: ?>>

// -----

func.func @test(%arg1: !hkernel<tensor <"W"> x f16>, %arg2: !hkernel<slice !typing<symbol "H"> : none : none>) -> !hkernel<tensor <"W1"> x f16> {
  %1 = hkernel.subview %arg1 : !hkernel<tensor <"W"> x f16>[%arg2] : !hkernel<slice !typing<symbol "H"> : none : none> -> !hkernel<tensor <"W1"> x f16>
  return %1 : !hkernel<tensor <"W1"> x f16>
}

//       CHECK: ![[SYM:.*]] = !typing<symbol "W">
// CHECK-LABEL: func @test
//  CHECK-SAME: (%[[ARG1:.*]]:  tuple<memref<?xf16, strided<[?], offset: ?>, #gpu.address_space<workgroup>>, memref<?xi1, strided<[?], offset: ?>, #gpu.address_space<workgroup>>>, %[[ARG2:.*]]: tuple<index, none, none>)
//   CHECK-DAG:  %[[C0:.*]] = arith.constant 0 : index
//   CHECK-DAG:  %[[C1:.*]] = arith.constant 1 : index
//   CHECK-DAG:  %[[DIM:.*]] = hkernel.materialize_expr ![[SYM]]
//   CHECK-DAG:  %[[DIM_IDX:.*]] = builtin.unrealized_conversion_cast %[[DIM]] : ![[SYM]] to index
//       CHECK:  %[[LOWER:.*]] = hkernel.tuple_extract %[[ARG2]] : tuple<index, none, none>[%[[C0]]] -> index
//       CHECK:  %[[OFFSET:.*]], %[[SIZE:.*]], %[[STRIDE:.*]] = hkernel.resolve_slice(%[[LOWER]] : : ) %[[DIM_IDX]]
//       CHECK:  %[[MEM1:.*]] = hkernel.tuple_extract %[[ARG1]] : tuple<memref<?xf16, strided<[?], offset: ?>, #gpu.address_space<workgroup>>, memref<?xi1, strided<[?], offset: ?>, #gpu.address_space<workgroup>>>[%[[C0]]] -> memref<?xf16, strided<[?], offset: ?>, #gpu.address_space<workgroup>>
//       CHECK:  %[[MEM2:.*]] = hkernel.tuple_extract %[[ARG1]] : tuple<memref<?xf16, strided<[?], offset: ?>, #gpu.address_space<workgroup>>, memref<?xi1, strided<[?], offset: ?>, #gpu.address_space<workgroup>>>[%[[C1]]] -> memref<?xi1, strided<[?], offset: ?>, #gpu.address_space<workgroup>>
//       CHECK:  %[[SUBVIEW1:.*]] = memref.subview %[[MEM1]][%[[OFFSET]]] [%[[SIZE]]] [%[[STRIDE]]] : memref<?xf16, strided<[?], offset: ?>, #gpu.address_space<workgroup>> to memref<?xf16, strided<[?], offset: ?>, #gpu.address_space<workgroup>>
//       CHECK:  %[[SUBVIEW2:.*]] = memref.subview %[[MEM2]][%[[OFFSET]]] [%[[SIZE]]] [%[[STRIDE]]] : memref<?xi1, strided<[?], offset: ?>, #gpu.address_space<workgroup>> to memref<?xi1, strided<[?], offset: ?>, #gpu.address_space<workgroup>>
//       CHECK:  %[[RET:.*]] = hkernel.make_tuple %[[SUBVIEW1]], %[[SUBVIEW2]] : memref<?xf16, strided<[?], offset: ?>, #gpu.address_space<workgroup>>, memref<?xi1, strided<[?], offset: ?>, #gpu.address_space<workgroup>> -> tuple<memref<?xf16, strided<[?], offset: ?>, #gpu.address_space<workgroup>>, memref<?xi1, strided<[?], offset: ?>, #gpu.address_space<workgroup>>>
//       CHECK:  return %[[RET]] :  tuple<memref<?xf16, strided<[?], offset: ?>, #gpu.address_space<workgroup>>, memref<?xi1, strided<[?], offset: ?>, #gpu.address_space<workgroup>>>
