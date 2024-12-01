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
//       CHECK:  %[[RES:.*]] = llvm.getelementptr %[[ARG]][%[[OFF]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
//       CHECK:  return %[[RES]] : !llvm.ptr

// -----

func.func @test(%arg: !hkernel.ptr<f32>) -> f32 {
  %0 = hkernel.ptr_load %arg : !hkernel.ptr<f32> : f32
  return %0 : f32
}

// CHECK-LABEL: func @test
//  CHECK-SAME:  (%[[ARG:.*]]: !llvm.ptr)
//       CHECK:  %[[RES:.*]] = llvm.load %[[ARG]] : !llvm.ptr -> f32
//       CHECK:  return %[[RES]] : f32

// -----

func.func @test(%arg: !hkernel.ptr<f32>, %offset: index) -> f32 {
  %0 = hkernel.ptr_load %arg : !hkernel.ptr<f32>[%offset : index] : f32
  return %0 : f32
}

// CHECK-LABEL: func @test
//  CHECK-SAME:  (%[[ARG:.*]]: !llvm.ptr, %[[OFF:.*]]: i64)
//       CHECK:  %[[PTR:.*]] = llvm.getelementptr %[[ARG]][%[[OFF]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
//       CHECK:  %[[RES:.*]] = llvm.load %[[PTR]] : !llvm.ptr -> f32
//       CHECK:  return %[[RES]] : f32

// -----

func.func @test(%arg: !hkernel.ptr<f32>, %mask: vector<4xi1>, %pass_thru: vector<4xf32>) -> vector<4xf32> {
  %0 = hkernel.ptr_load %arg : !hkernel.ptr<f32> mask %mask : vector<4xi1>, %pass_thru : vector<4xf32> : vector<4xf32>
  return %0 : vector<4xf32>
}

// CHECK-LABEL: func @test
//  CHECK-SAME:  (%[[ARG:.*]]: !llvm.ptr, %[[MASK:.*]]: vector<4xi1>, %[[PASS_THRU:.*]]: vector<4xf32>)
//       CHECK:  %[[RES:.*]] = llvm.intr.masked.load %[[ARG]], %[[MASK]], %[[PASS_THRU]] {alignment = 32 : i32} : (!llvm.ptr, vector<4xi1>, vector<4xf32>) -> vector<4xf32>
//       CHECK:  return %[[RES]] : vector<4xf32>
