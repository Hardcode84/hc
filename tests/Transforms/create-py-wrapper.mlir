// RUN: hc-opt -split-input-file %s --hc-create-py-wrapper-pass | FileCheck %s

// CHECK-LABEL: func @copy_kernel_pyabi
//  CHECK-SAME:   (%[[ARG0:.*]]: !hkernel.error_context, %[[ARG1:.*]]: !hkernel.py_args)
//       CHECK:   %[[A0:.*]] = ub.poison : !hkernel<current_group 2>
//       CHECK:   %[[A1:.*]] = hkernel.get_py_arg %[[ARG1]][1], %[[ARG0]] : !hkernel.memref_descriptor<memref<?x?xi32, strided<[?, ?], offset: ?>>>~
//       CHECK:   %[[A2:.*]] = hkernel.get_py_arg %[[ARG1]][2], %[[ARG0]] : !hkernel.memref_descriptor<memref<?x?xi32, strided<[?, ?], offset: ?>>>
//       CHECK:   call @copy_kernel(%[[A0]], %[[A1]], %[[A2]]) : (!hkernel<current_group 2>, !hkernel.memref_descriptor<memref<?x?xi32, strided<[?, ?], offset: ?>>>, !hkernel.memref_descriptor<memref<?x?xi32, strided<[?, ?], offset: ?>>>) -> ()
//       CHECK:   %[[RET:.*]] = arith.constant 0 : i32
//       CHECK:   return %[[RET]] : i32
//       CHECK:   asdsaassda
func.func @copy_kernel(%arg0: !hkernel<current_group 2>, %arg1: !hkernel.memref_descriptor<memref<?x?xi32, strided<[?, ?], offset: ?>>>, %arg2: !hkernel.memref_descriptor<memref<?x?xi32, strided<[?, ?], offset: ?>>>) attributes {kernel.entrypoint} {
  return
}
