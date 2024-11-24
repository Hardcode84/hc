// RUN: hc-opt -split-input-file %s --hc-decompose-pointers-pass --canonicalize | FileCheck %s

// CHECK-LABEL: func @test
//  CHECK-SAME:  (%[[ARG:.*]]: tuple<!hkernel.ptr<f32>, i32>)
//       CHECK:  return %[[ARG]] : tuple<!hkernel.ptr<f32>, i32>
func.func @test(%arg: !hkernel.ptr<f32, #hkernel.logical_ptr<i32>>) -> !hkernel.ptr<f32, #hkernel.logical_ptr<i32>> {
  return %arg : !hkernel.ptr<f32, #hkernel.logical_ptr<i32>>
}
