// RUN: hc-opt -split-input-file %s --hc-decompose-pointers-pass --canonicalize | FileCheck %s

// CHECK-LABEL: func @test
//  CHECK-SAME:  (%[[ARG:.*]]: tuple<!hkernel.ptr<f32>, i32>)
//       CHECK:  return %[[ARG]] : tuple<!hkernel.ptr<f32>, i32>
func.func @test(%arg: !hkernel.ptr<f32, #hkernel.logical_ptr<i32>>) -> !hkernel.ptr<f32, #hkernel.logical_ptr<i32>> {
  return %arg : !hkernel.ptr<f32, #hkernel.logical_ptr<i32>>
}

// -----

// CHECK-LABEL: func @test
//  CHECK-SAME:  (%[[ARG:.*]]: tuple<tuple<!hkernel.ptr<f32>, i32>, index>)
//       CHECK:  return %[[ARG]] : tuple<tuple<!hkernel.ptr<f32>, i32>, index>
func.func @test(%arg: tuple<!hkernel.ptr<f32, #hkernel.logical_ptr<i32>>, index>) -> tuple<!hkernel.ptr<f32, #hkernel.logical_ptr<i32>>, index> {
  return %arg : tuple<!hkernel.ptr<f32, #hkernel.logical_ptr<i32>>, index>
}
