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
//  CHECK-SAME: (%[[ARG1:.*]]: tuple<memref<?xf16, strided<[?], offset: ?>, #gpu.address_space<workgroup>>, memref<?xi1, strided<[?], offset: ?>, #gpu.address_space<workgroup>>>, %[[ARG2:.*]]: tuple<index, none, none>)
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

// -----

func.func @test(%arg1: !hkernel<buffer <"W"> x f16>, %arg2: !typing<symbol "H">) -> !hkernel<tensor <"H"> x f16> {
  %1 = hkernel.load %arg1 : !hkernel<buffer <"W"> x f16>[%arg2] : !typing<symbol "H"> -> !hkernel<tensor <"H"> x f16>
  return %1 : !hkernel<tensor <"H"> x f16>
}

//   CHECK-DAG: ![[SYM1:.*]] = !typing<symbol "H">
//   CHECK-DAG: ![[SYM2:.*]] = !typing<symbol "W">
// CHECK-LABEL: func @test
//  CHECK-SAME: (%[[ARG1:.*]]: memref<?xf16, strided<[?], offset: ?>>, %[[ARG2:.*]]: index)
//   CHECK-DAG:  %[[P:.*]] = ub.poison : vector<1xf16>
//   CHECK-DAG:  %[[C0:.*]] = arith.constant 0 : index
//   CHECK-DAG:  %[[C1:.*]] = arith.constant 1 : index
//   CHECK-DAG:  %[[EXPR1:.*]] = hkernel.materialize_expr ![[SYM1]]
//   CHECK-DAG:  %[[EXPR2:.*]] = hkernel.materialize_expr ![[SYM2]]
//   CHECK-DAG:  %[[EXPR_VAL1:.*]] = builtin.unrealized_conversion_cast %[[EXPR1]] : ![[SYM1]] to index
//   CHECK-DAG:  %[[EXPR_VAL2:.*]] = builtin.unrealized_conversion_cast %[[EXPR2]] : ![[SYM2]] to index
//       CHECK:  %[[ALLOC1:.*]] = memref.alloca(%[[EXPR_VAL1]]) : memref<?xf16, #gpu.address_space<workgroup>>
//       CHECK:  %[[CAST1:.*]] = memref.cast %[[ALLOC1]] : memref<?xf16, #gpu.address_space<workgroup>> to memref<?xf16, strided<[?], offset: ?>, #gpu.address_space<workgroup>>
//       CHECK:  %[[ALLOC2:.*]] = memref.alloca(%[[EXPR_VAL1]]) : memref<?xi1, #gpu.address_space<workgroup>>
//       CHECK:  %[[CAST2:.*]] = memref.cast %[[ALLOC2]] : memref<?xi1, #gpu.address_space<workgroup>> to memref<?xi1, strided<[?], offset: ?>, #gpu.address_space<workgroup>>
//       CHECK:  scf.parallel (%[[I:.*]]) = (%[[C0]]) to (%[[EXPR_VAL1]]) step (%[[C1]]) {
//       CHECK:    %[[M1:.*]] = arith.cmpi slt, %[[I]], %[[EXPR_VAL2]] : index
//       CHECK:    %[[M2:.*]] = vector.splat %[[M1]] : vector<1xi1>
//       CHECK:    %[[R:.*]] = vector.maskedload %[[ARG1]][%[[I]]], %[[M2]], %[[P]] : memref<?xf16, strided<[?], offset: ?>>, vector<1xi1>, vector<1xf16> into vector<1xf16>
//       CHECK:    vector.store %[[R]], %[[ALLOC1]][%[[I]]] : memref<?xf16, #gpu.address_space<workgroup>>, vector<1xf16>
//       CHECK:    vector.store %[[M2]], %[[ALLOC2]][%[[I]]] : memref<?xi1, #gpu.address_space<workgroup>>, vector<1xi1>
//       CHECK:    scf.reduce
//       CHECK:  }
//       CHECK:  %[[RES:.*]] = hkernel.make_tuple %[[CAST1]], %[[CAST2]] :
//  CHECK-SAME:    memref<?xf16, strided<[?], offset: ?>, #gpu.address_space<workgroup>>, memref<?xi1, strided<[?], offset: ?>, #gpu.address_space<workgroup>> ->
//  CHECK-SAME:    tuple<memref<?xf16, strided<[?], offset: ?>, #gpu.address_space<workgroup>>, memref<?xi1, strided<[?], offset: ?>, #gpu.address_space<workgroup>>>
//       CHECK:  return %[[RES]] : tuple<memref<?xf16, strided<[?], offset: ?>, #gpu.address_space<workgroup>>, memref<?xi1, strided<[?], offset: ?>, #gpu.address_space<workgroup>>>

// -----

func.func @test(%arg1: !hkernel<buffer <"W"> x f16>, %arg2: !hkernel<tensor <"H"> x f16>) {
  hkernel.store %arg1 : !hkernel<buffer <"W"> x f16> = %arg2 : !hkernel<tensor <"H"> x f16>
  return
}

//       CHECK: ![[SYM:.*]] = !typing<symbol "H">
// CHECK-LABEL: func @test
//  CHECK-SAME: (%[[ARG1:.*]]: memref<?xf16, strided<[?], offset: ?>>, %[[ARG2:.*]]: tuple<memref<?xf16, strided<[?], offset: ?>, #gpu.address_space<workgroup>>, memref<?xi1, strided<[?], offset: ?>, #gpu.address_space<workgroup>>>)
//   CHECK-DAG:  %[[C0:.*]] = arith.constant 0 : index
//   CHECK-DAG:  %[[C1:.*]] = arith.constant 1 : index
//   CHECK-DAG:  %[[EXPR:.*]] = hkernel.materialize_expr ![[SYM]]
//       CHECK:  %[[MEM1:.*]] = hkernel.tuple_extract %[[ARG2]] : tuple<memref<?xf16, strided<[?], offset: ?>, #gpu.address_space<workgroup>>, memref<?xi1, strided<[?], offset: ?>, #gpu.address_space<workgroup>>>[%[[C0]]] -> memref<?xf16, strided<[?], offset: ?>, #gpu.address_space<workgroup>>
//       CHECK:  %[[MASK1:.*]] = hkernel.tuple_extract %[[ARG2]] : tuple<memref<?xf16, strided<[?], offset: ?>, #gpu.address_space<workgroup>>, memref<?xi1, strided<[?], offset: ?>, #gpu.address_space<workgroup>>>[%[[C1]]] -> memref<?xi1, strided<[?], offset: ?>, #gpu.address_space<workgroup>>
//       CHECK:  %[[SIZE:.*]] = builtin.unrealized_conversion_cast %[[EXPR]] : ![[SYM]] to index
//       CHECK:  scf.parallel (%[[I:.*]]) = (%[[C0]]) to (%[[SIZE]]) step (%[[C1]]) {
//       CHECK:    %[[M:.*]] = vector.load %[[MASK1]][%[[I]]] : memref<?xi1, strided<[?], offset: ?>, #gpu.address_space<workgroup>>, vector<1xi1>
//       CHECK:    %[[R:.*]] = vector.load %[[MEM1]][%[[I]]] : memref<?xf16, strided<[?], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf16>
//       CHECK:    vector.maskedstore %[[ARG1]][%[[I]]], %[[M]], %[[R]] : memref<?xf16, strided<[?], offset: ?>>, vector<1xi1>, vector<1xf16>
//       CHECK:    scf.reduce
//       CHECK:  }
//       CHECK:  return
