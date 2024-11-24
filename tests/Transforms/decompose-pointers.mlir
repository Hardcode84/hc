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

// -----

// CHECK-LABEL: func @test
//  CHECK-SAME:  (%[[ARG:.*]]: index)
//       CHECK:  %[[C0:.*]] = arith.constant 0 : i32
//       CHECK:  %[[PTR:.*]] = hkernel.ptr_alloca %[[ARG]] : index, !hkernel.ptr<f32>
//       CHECK:  %[[RES:.*]] = hkernel.make_tuple %[[PTR]], %[[C0]] : !hkernel.ptr<f32>, i32 -> tuple<!hkernel.ptr<f32>, i32>
//       CHECK:  return %[[RES]] : tuple<!hkernel.ptr<f32>, i32>
func.func @test(%arg: index) -> !hkernel.ptr<f32, #hkernel.logical_ptr<i32>> {
  %0 = hkernel.ptr_alloca %arg : index, !hkernel.ptr<f32, #hkernel.logical_ptr<i32>>
  return %0 : !hkernel.ptr<f32, #hkernel.logical_ptr<i32>>
}

// -----

// CHECK-LABEL: func @test
//  CHECK-SAME:  (%[[ARG0:.*]]: tuple<!hkernel.ptr<f32>, i32>, %[[ARG1:.*]]: index)
//   CHECK-DAG:  %[[C0:.*]] = arith.constant 0 : index
//   CHECK-DAG:  %[[C1:.*]] = arith.constant 1 : index
//       CHECK:  %[[PTR:.*]] = hkernel.tuple_extract %[[ARG0]] : tuple<!hkernel.ptr<f32>, i32>[%[[C0]]] -> !hkernel.ptr<f32>
//       CHECK:  %[[OFF1:.*]] = hkernel.tuple_extract %[[ARG0]] : tuple<!hkernel.ptr<f32>, i32>[%[[C1]]] -> i32
//       CHECK:  %[[OFF2:.*]] = arith.index_cast %[[ARG1]] : index to i32
//       CHECK:  %[[RES_OFF:.*]] = arith.addi %[[OFF1]], %[[OFF2]] overflow<nsw, nuw> : i32
//       CHECK:  %[[RES:.*]] = hkernel.make_tuple %[[PTR]], %[[RES_OFF]] : !hkernel.ptr<f32>, i32 -> tuple<!hkernel.ptr<f32>, i32>
//       CHECK:  return %[[RES]] : tuple<!hkernel.ptr<f32>, i32>
func.func @test(%arg: !hkernel.ptr<f32, #hkernel.logical_ptr<i32>>, %offset: index) -> !hkernel.ptr<f32, #hkernel.logical_ptr<i32>> {
  %0 = hkernel.ptr_add %arg : !hkernel.ptr<f32, #hkernel.logical_ptr<i32>>, %offset : index
  return %0 : !hkernel.ptr<f32, #hkernel.logical_ptr<i32>>
}
