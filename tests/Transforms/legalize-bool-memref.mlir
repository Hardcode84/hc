// RUN: hc-opt -split-input-file %s --hc-legalize-bool-memrefs-pass | FileCheck %s

func.func @test(%arg: memref<?xi1>) -> memref<?xi1> {
  return %arg : memref<?xi1>
}

// CHECK-LABEL: func @test
//  CHECK-SAME: (%[[ARG:.*]]: memref<?xi8>)
//       CHECK: return %[[ARG]] : memref<?xi8>
