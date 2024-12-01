// RUN: hc-opt -split-input-file %s --hc-expand-tuple-pass | FileCheck %s

func.func @test(%arg: tuple<>) -> tuple<> {
  return %arg : tuple<>
}

// CHECK-LABEL: func @test()
//       CHECK:  return

// -----

func.func @test(%arg: tuple<i32, f32>) -> tuple<i32, f32> {
  return %arg : tuple<i32, f32>
}

// CHECK-LABEL: func @test
//  CHECK-SAME:  (%[[ARG0:.*]]: i32, %[[ARG1:.*]]: f32)
//       CHECK:  return %[[ARG0]], %[[ARG1]] : i32, f32

// -----

func.func @test(%arg: tuple<tuple<i32, i64>, f32>) -> tuple<tuple<i32, i64>, f32> {
  return %arg : tuple<tuple<i32, i64>, f32>
}

// CHECK-LABEL: func @test
//  CHECK-SAME:  (%[[ARG0:.*]]: i32, %[[ARG1:.*]]: i64, %[[ARG2:.*]]: f32)
//       CHECK:  return %[[ARG0]], %[[ARG1]], %[[ARG2]] : i32, i64, f32
