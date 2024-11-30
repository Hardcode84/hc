// RUN: hc-opt -split-input-file %s --convert-to-llvm | FileCheck %s

func.func @test(%arg: !hkernel.ptr<f32>) -> !hkernel.ptr<f32> {
  return %arg : !hkernel.ptr<f32>
}

// CHECK-LABEL: func @test
//  CHECK-SAME:  (%[[ARG:.*]]: !llvm.ptr)
//       CHECK:  return %[[ARG]] : !llvm.ptr

// -----

func.func @test(%arg: !hkernel.ptr<f32>, %offset: index) -> !hkernel.ptr<f32> {
  %0 = hkernel.ptr_add %arg : !hkernel.ptr<f32>, %offset : index
  return %0 : !hkernel.ptr<f32>
}

// CHECK-LABEL: func @test
//  CHECK-SAME:  (%[[ARG:.*]]: !llvm.ptr, %[[OFF:.*]]: i64)
//       CHECK:  %[[RES:.*]] = llvm.getelementptr %[[ARG]][%[[OFF]]] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.ptr
//       CHECK:  return %[[RES]] : !llvm.ptr
