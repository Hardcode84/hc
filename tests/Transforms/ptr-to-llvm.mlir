// RUN: hc-opt -split-input-file %s --convert-to-llvm | FileCheck %s

func.func @test(%arg: !hkernel.ptr<f32>) -> !hkernel.ptr<f32> {
  return %arg : !hkernel.ptr<f32>
}

// CHECK-LABEL: func @test
//  CHECK-SAME:  (%[[ARG:.*]]: !llvm.ptr)
//       CHECK:  return %[[ARG]] : !llvm.ptr
