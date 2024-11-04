// RUN: hc-opt -split-input-file %s --hc-decompose-hkernel-ops-pass --canonicalize | FileCheck %s


func.func @test(%dim: index) -> (index, index, index) {
  %offset, %size, %stride = hkernel.resolve_slice( :  : ) %dim
  return %offset, %size, %stride : index, index, index
}

// CHECK-LABEL: func @test
//  CHECK-SAME:  (%[[DIM:.*]]: index)
//   CHECK-DAG:  %[[C0:.*]] = arith.constant 0 : index
//   CHECK-DAG:  %[[C1:.*]] = arith.constant 1 : index
//       CHECK:  return %[[C0]], %[[DIM]], %[[C1]] : index, index, index

// -----

func.func @test(%lower: index, %dim: index) -> (index, index, index) {
  %offset, %size, %stride = hkernel.resolve_slice(%lower :  : ) %dim
  return %offset, %size, %stride : index, index, index
}

// CHECK-LABEL: func @test
//  CHECK-SAME:  (%[[L:.*]]: index, %[[DIM:.*]]: index)
//   CHECK-DAG:  %[[C1:.*]] = arith.constant 1 : index
//       CHECK:  %[[S:.*]] = arith.subi %[[DIM]], %[[L]] : index
//       CHECK:  return %[[L]], %[[S]], %[[C1]] : index, index, index

// -----

func.func @test(%upper: index, %dim: index) -> (index, index, index) {
  %offset, %size, %stride = hkernel.resolve_slice( : %upper  : ) %dim
  return %offset, %size, %stride : index, index, index
}

// CHECK-LABEL: func @test
//  CHECK-SAME:  (%[[U:.*]]: index, %[[DIM:.*]]: index)
//   CHECK-DAG:  %[[C0:.*]] = arith.constant 0 : index
//   CHECK-DAG:  %[[C1:.*]] = arith.constant 1 : index
//       CHECK:  return %[[C0]], %[[U]], %[[C1]] : index, index, index
