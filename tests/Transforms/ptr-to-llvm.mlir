// RUN: hc-opt -split-input-file %s --convert-to-llvm | FileCheck %s

func.func @test(%arg: !hkernel.ptr<f32>) -> !hkernel.ptr<f32> {
  return %arg : !hkernel.ptr<f32>
}

// CHECK-LABEL: func @test
//  CHECK-SAME:  (%[[ARG:.*]]: !llvm.ptr)
//       CHECK:  return %[[ARG]] : !llvm.ptr

// -----

func.func @test(%arg: index) -> !hkernel.ptr<f32> {
  %0 = hkernel.ptr_alloca %arg : index, !hkernel.ptr<f32>
  return %0 : !hkernel.ptr<f32>
}

// CHECK-LABEL: func @test
//  CHECK-SAME:  (%[[ARG:.*]]: i64)
//       CHECK:  %[[RES:.*]] = llvm.alloca %[[ARG]] x f32 : (i64) -> !llvm.ptr
//       CHECK:  return %[[RES]] : !llvm.ptr

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
//       CHECK:  %[[RES:.*]] = llvm.intr.masked.load %[[ARG]], %[[MASK]], %[[PASS_THRU]] {alignment = 4 : i32} : (!llvm.ptr, vector<4xi1>, vector<4xf32>) -> vector<4xf32>
//       CHECK:  return %[[RES]] : vector<4xf32>

// -----

func.func @test(%arg: !hkernel.ptr<f32>, %value: f32) {
  hkernel.ptr_store %value : f32 %arg : !hkernel.ptr<f32>
  return
}

// CHECK-LABEL: func @test
//  CHECK-SAME:  (%[[ARG:.*]]: !llvm.ptr, %[[VAL:.*]]: f32)
//       CHECK:  llvm.store %[[VAL]], %[[ARG]] : f32, !llvm.ptr

// -----

func.func @test(%arg: !hkernel.ptr<f32>, %value: f32, %offset: index)  {
  hkernel.ptr_store %value : f32 %arg : !hkernel.ptr<f32>[%offset : index]
  return
}

// CHECK-LABEL: func @test
//  CHECK-SAME:  (%[[ARG:.*]]: !llvm.ptr, %[[VAL:.*]]: f32, %[[OFF:.*]]: i64)
//       CHECK:  %[[PTR:.*]] = llvm.getelementptr %[[ARG]][%[[OFF]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
//       CHECK:  llvm.store %[[VAL]], %[[PTR]] : f32, !llvm.ptr

// -----

func.func @test(%arg: !hkernel.ptr<f32>, %mask: vector<4xi1>, %value: vector<4xf32>)  {
  hkernel.ptr_store %value : vector<4xf32> %arg : !hkernel.ptr<f32> mask %mask : vector<4xi1>
  return
}

// CHECK-LABEL: func @test
//  CHECK-SAME:  (%[[ARG:.*]]: !llvm.ptr, %[[MASK:.*]]: vector<4xi1>, %[[VAL:.*]]: vector<4xf32>)
//       CHECK:  llvm.intr.masked.store %[[VAL]], %[[ARG]], %[[MASK]] {alignment = 4 : i32} : vector<4xf32>, vector<4xi1> into !llvm.ptr

// -----

func.func @test(%arg: !hkernel.memref_descriptor<memref<?xf32>>) -> !hkernel.memref_descriptor<memref<?xf32>> {
  return %arg : !hkernel.memref_descriptor<memref<?xf32>>
}

// CHECK-LABEL: func @test
//  CHECK-SAME:  (%[[ARG:.*]]: !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)
//       CHECK:  return %[[ARG]] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>

// -----

func.func @test(%arg: !hkernel.memref_descriptor<memref<?x?xf32, strided<[?, ?], offset: ?>>>) -> !hkernel.memref_descriptor<memref<?x?xf32, strided<[?, ?], offset: ?>>> {
  return %arg : !hkernel.memref_descriptor<memref<?x?xf32, strided<[?, ?], offset: ?>>>
}

// CHECK-LABEL: func @test
//  CHECK-SAME:  (%[[ARG:.*]]: !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>)
//       CHECK:  return %[[ARG]] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
