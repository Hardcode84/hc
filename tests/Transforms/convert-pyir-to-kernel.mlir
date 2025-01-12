// RUN: hc-opt -split-input-file %s --hc-convert-py-ir-to-kernel-pass | FileCheck %s

func.func @test(%arg1: !typing<symbol "W">, %arg2: !typing<symbol "H">) -> !typing<expr (!typing<symbol "W">, !typing<symbol "H">) -> s0 * s1> {
  %0 = py_ir.binop %arg1 : !typing<symbol "W"> mul %arg2 : !typing<symbol "H"> -> !typing<expr (!typing<symbol "W">, !typing<symbol "H">) -> s0 * s1>
  return %0 : !typing<expr (!typing<symbol "W">, !typing<symbol "H">) -> s0 * s1>
}

//   CHECK-DAG: ![[SYM:.*]] = !typing<symbol "W">
//   CHECK-DAG: ![[SYM1:.*]] = !typing<symbol "H">
//   CHECK-DAG: ![[EXPR:.*]] = !typing<expr (![[SYM1]], ![[SYM]]) -> s1 * s0>
// CHECK-LABEL: func @test
//  CHECK-SAME:   (%[[ARG1:.*]]: ![[SYM]], %[[ARG2:.*]]: ![[SYM1]])
//       CHECK:   %[[LHS:.*]] = typing.cast %[[ARG1]] : ![[SYM]] to index
//       CHECK:   %[[RHS:.*]] = typing.cast %[[ARG2]] : ![[SYM1]] to index
//       CHECK:   %[[RES:.*]] = arith.muli %[[LHS]], %[[RHS]] : index
//       CHECK:   %[[RES1:.*]] = typing.cast %[[RES]] : index to ![[EXPR]]
//       CHECK:   return %[[RES1]]
