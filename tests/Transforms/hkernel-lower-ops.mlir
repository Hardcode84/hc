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

// CHECK-LABEL: func @test
//  CHECK-SAME: (%[[ARG1:.*]]: memref<?xf16, strided<[?], offset: ?>>, %[[ARG2:.*]]: tuple<index, none, none>)
//       CHECK:  %[[C0:.*]] = arith.constant 0 : index
//       CHECK:  %[[DIM:.*]] = memref.dim %[[ARG1]], %[[C0]] : memref<?xf16, strided<[?], offset: ?>>
//       CHECK:  %[[LOWER:.*]] = hkernel.tuple_extract %[[ARG2]] : tuple<index, none, none>[%[[C0]]] -> index
//       CHECK:  %[[OFFSET:.*]], %[[SIZE:.*]], %[[STRIDE:.*]] = hkernel.resolve_slice(%[[LOWER]] : : ) %dim
//       CHECK:  %[[SUBVIEW:.*]] = memref.subview %[[ARG1]][%[[OFFSET]]] [%[[SIZE]]] [%[[STRIDE]]] : memref<?xf16, strided<[?], offset: ?>> to memref<?xf16, strided<[?], offset: ?>>
//       CHECK:  return %[[SUBVIEW]] : memref<?xf16, strided<[?], offset: ?>>
