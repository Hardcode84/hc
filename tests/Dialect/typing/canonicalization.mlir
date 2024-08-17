// RUN: hc-opt -allow-unregistered-dialect %s -pass-pipeline='builtin.module(func.func(canonicalize{test-convergence}))' -split-input-file | FileCheck %s

// CHECK: ![[EXPR:.*]] = !typing<expr 0>
// CHECK: func.func private @func(![[EXPR]])
func.func private @func(!typing<expr 0>)

// -----

// CHECK: ![[SYM:.*]] = !typing<symbol "A">
// CHECK: ![[EXPR:.*]] = !typing<expr (![[SYM]]) -> s0>
// CHECK: func.func private @func(![[EXPR]])
func.func private @func(!typing<expr ( !typing<symbol "A"> ) -> s0 >)

// -----

// CHECK-DAG: ![[SYM1:.*]] = !typing<symbol "A">
// CHECK-DAG: ![[SYM2:.*]] = !typing<symbol "B">
// CHECK: ![[EXPR:.*]] = !typing<expr (![[SYM1]], ![[SYM2]]) -> s0 + s1>
// CHECK: func.func private @func(![[EXPR]])
func.func private @func(!typing<expr ( !typing<symbol "A">, !typing<symbol "B"> ) -> s0 + s1 >)

// -----

// CHECK: ![[SYM:.*]] = !typing<symbol "B">
// CHECK: ![[EXPR:.*]] = !typing<expr (![[SYM]]) -> s0>
// CHECK: func.func private @func(![[EXPR]])
func.func private @func(!typing<expr ( !typing<symbol "A">, !typing<symbol "B"> ) -> s1 >)

// -----

// CHECK: ![[SYM:.*]] = !typing<symbol "A">
// CHECK: ![[EXPR:.*]] = !typing<expr (![[SYM]]) -> s0 * 2>
// CHECK: func.func private @func(![[EXPR]])
func.func private @func(!typing<expr ( !typing<symbol "A">, !typing<symbol "A"> ) -> s0 + s1 >)

// -----

// CHECK-DAG: ![[SYM1:.*]] = !typing<symbol "A">
// CHECK-DAG: ![[SYM2:.*]] = !typing<symbol "B">
// CHECK-DAG: ![[SYM3:.*]] = !typing<symbol "C">
// CHECK: ![[EXPR:.*]] = !typing<expr (![[SYM1]], ![[SYM2]], ![[SYM3]]) -> s0 * s1 + s2>
// CHECK: func.func private @func(![[EXPR]])
func.func private @func(!typing<expr ( !typing<expr ( !typing<symbol "A">, !typing<symbol "B"> ) -> s0 * s1 >, !typing<symbol "C"> ) -> s0 + s1 >)

// -----

// CHECK: ![[EXPR:.*]] = !typing<expr -3>
// CHECK: func.func private @func(![[EXPR]])
func.func private @func(!typing<expr ( !typing<literal 2 : i64>, !typing<literal -5 : i64> ) -> s0 + s1 >)
