// RUN: hc-opt -split-input-file %s --hc-convert-py-ir-to-kernel-pass | FileCheck %s

func.func @test(%arg1: f32, %arg2: f32) -> f32 {
  %0 = py_ir.binop %arg1 : f32 mul %arg2 : f32 -> f32
  return %0 : f32
}

// CHECK-LABEL: func @test
//  CHECK-SAME: (%[[ARG1:.*]]: f32, %[[ARG2:.*]]: f32)
//       CHECK:   %[[RES:.*]] = arith.mulf %[[ARG1]], %[[ARG2]] : f32
//       CHECK:   return %[[RES]]

// -----

func.func @test(%arg1: !typing<symbol "W">, %arg2: !typing<symbol "H">) -> !typing<expr (!typing<symbol "W">, !typing<symbol "H">) -> s0 * s1> {
  %0 = py_ir.binop %arg1 : !typing<symbol "W"> mul %arg2 : !typing<symbol "H"> -> !typing<expr (!typing<symbol "W">, !typing<symbol "H">) -> s0 * s1>
  return %0 : !typing<expr (!typing<symbol "W">, !typing<symbol "H">) -> s0 * s1>
}

//   CHECK-DAG: ![[SYM:.*]] = !typing<symbol "W">
//   CHECK-DAG: ![[SYM1:.*]] = !typing<symbol "H">
//   CHECK-DAG: ![[EXPR:.*]] = !typing<expr (![[SYM1]], ![[SYM]]) -> s1 * s0>
// CHECK-LABEL: func @test(
//       CHECK:   %[[ARG1:.*]] = hkernel.materialize_expr ![[SYM]]
//       CHECK:   %[[LHS:.*]] = typing.cast %[[ARG1]] : ![[SYM]] to index
//       CHECK:   %[[ARG2:.*]] = hkernel.materialize_expr ![[SYM1]]
//       CHECK:   %[[RHS:.*]] = typing.cast %[[ARG2]] : ![[SYM1]] to index
//       CHECK:   %[[RES:.*]] = arith.muli %[[LHS]], %[[RHS]] : index
//       CHECK:   %[[RES1:.*]] = typing.cast %[[RES]] : index to ![[EXPR]]
//       CHECK:   return %[[RES1]]
