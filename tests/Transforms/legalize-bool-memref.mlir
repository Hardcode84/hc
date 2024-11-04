// RUN: hc-opt -split-input-file %s --hc-legalize-bool-memrefs-pass | FileCheck %s

func.func @test(%arg: memref<?xi1>) -> memref<?xi1> {
  return %arg : memref<?xi1>
}

// CHECK-LABEL: func @test
//  CHECK-SAME:  (%[[ARG:.*]]: memref<?xi8>)
//       CHECK:  return %[[ARG]] : memref<?xi8>

// -----

func.func @test() -> memref<10xi1> {
  %1 = memref.alloc() : memref<10xi1>
  return %1 : memref<10xi1>
}

// CHECK-LABEL: func @test() -> memref<10xi8>
//       CHECK:  %[[R:.*]] = memref.alloc() : memref<10xi8>
//       CHECK:  return %[[R]] : memref<10xi8>

// -----

func.func @test(%arg: memref<?xi1>, %val: vector<1xi1>) {
  %c0 = arith.constant 0 : index
  vector.store %val, %arg[%c0] : memref<?xi1>, vector<1xi1>
  return
}

// CHECK-LABEL: func @test
//  CHECK-SAME: (%[[ARG:.*]]: memref<?xi8>, %[[VAL:.*]]: vector<1xi1>)
//       CHECK:  %[[C0:.*]] = arith.constant 0 : index
//       CHECK:  %[[CAST:.*]] = arith.extui %[[VAL]] : vector<1xi1> to vector<1xi8>
//       CHECK:  vector.store %[[CAST]], %[[ARG]][%[[C0]]] : memref<?xi8>, vector<1xi8>

// -----

func.func @test(%arg: memref<?xi1>) -> vector<1xi1> {
  %c0 = arith.constant 0 : index
  %ret = vector.load %arg[%c0] : memref<?xi1>, vector<1xi1>
  return %ret : vector<1xi1>
}

// CHECK-LABEL: func @test
//  CHECK-SAME: (%[[ARG:.*]]: memref<?xi8>) -> vector<1xi1>
//       CHECK:  %[[C0:.*]] = arith.constant 0 : index
//       CHECK:  %[[R:.*]] = vector.load %[[ARG]][%[[C0]]] : memref<?xi8>, vector<1xi8>
//       CHECK:  %[[CASTED:.*]] = arith.trunci %[[R]] : vector<1xi8> to vector<1xi1>
//       CHECK:  return %[[CASTED]] : vector<1xi1>
