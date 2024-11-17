// RUN: hc-opt -split-input-file %s --hc-decompose-memrefs-pass --canonicalize | FileCheck %s

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

// -----

// CHECK-LABEL: func @test
//       CHECK:  %[[S:.*]] = arith.constant 10 : index
//       CHECK:  %[[P:.*]] = hkernel.ptr_alloca %[[S]] : index, !hkernel.ptr<f32>
//       CHECK:  %[[R:.*]] = hkernel.make_tuple %[[P]] : !hkernel.ptr<f32> -> tuple<!hkernel.ptr<f32>>
//       CHECK:  return %[[R]] : tuple<!hkernel.ptr<f32>>
func.func @test() -> memref<10xf32> {
  %0 = memref.alloca() : memref<10xf32>
  return %0 : memref<10xf32>
}
