// RUN: hc-opt -split-input-file %s --hc-decompose-memrefs-pass | FileCheck %s

// CHECK-LABEL: func @test
//  CHECK-SAME:  (%[[ARG:.*]]: tuple<!hkernel.ptr<f32>>)
//       CHECK:  return %[[ARG]] : tuple<!hkernel.ptr<f32>>
func.func @test(%arg: memref<10xf32>) -> memref<10xf32> {
  return %arg : memref<10xf32>
}

// -----

// CHECK-LABEL: func @test
//  CHECK-SAME:  (%[[ARG:.*]]: tuple<!hkernel.ptr<f32>>)
//       CHECK:  return %[[ARG]] : tuple<!hkernel.ptr<f32>>
func.func @test(%arg: memref<10xf32, strided<[1], offset: ?>>) -> memref<10xf32, strided<[1], offset: ?>> {
  return %arg : memref<10xf32, strided<[1], offset: ?>>
}

// -----

// CHECK-LABEL: func @test
//  CHECK-SAME:  (%[[ARG:.*]]: tuple<!hkernel.ptr<f32>, index>)
//       CHECK:  return %[[ARG]] : tuple<!hkernel.ptr<f32>, index>
func.func @test(%arg: memref<?xf32>) -> memref<?xf32> {
  return %arg : memref<?xf32>
}

// -----

// CHECK-LABEL: func @test
//  CHECK-SAME:  (%[[ARG:.*]]: tuple<!hkernel.ptr<f32>, index, index>)
//       CHECK:  return %[[ARG]] : tuple<!hkernel.ptr<f32>, index, index>
func.func @test(%arg: memref<?xf32, strided<[?], offset: ?>>) -> memref<?xf32, strided<[?], offset: ?>> {
  return %arg : memref<?xf32, strided<[?], offset: ?>>
}
