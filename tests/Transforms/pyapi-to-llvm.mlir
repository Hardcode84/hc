// RUN: hc-opt -split-input-file %s --convert-to-llvm | FileCheck %s

// CHECK-LABEL: func @copy_kernel_pyabi
//  CHECK-SAME:   (%[[ARG0:.*]]: !llvm.ptr, %[[ARG1:.*]]: !llvm.ptr)
//       CHECK:   %[[D0:.*]] = llvm.mlir.constant(1 : i64) : i64
//       CHECK:   %[[D1:.*]] = llvm.alloca %[[D0]] x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
//       CHECK:   %[[D2:.*]] = llvm.mlir.constant(1 : i64) : i64
//       CHECK:   %[[D3:.*]] = llvm.alloca %[[D2]] x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
//       CHECK:   %[[D4:.*]] = llvm.mlir.poison : !llvm.ptr
//       CHECK:   %[[D5:.*]] = llvm.mlir.constant(1 : i32) : i32
//       CHECK:   %[[D6:.*]] = llvm.call @hcgpuGetPyArg(%[[ARG1]], %[[D5]]) : (!llvm.ptr, i32) -> !llvm.ptr
//       CHECK:   %[[D7:.*]] = llvm.call @hcgpuConvertPyArray(%[[ARG0]], %[[D6]], %[[D3]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
//       CHECK:   %[[D8:.*]] = llvm.mlir.constant(0 : i32) : i32
//       CHECK:   %[[D9:.*]] = llvm.icmp "eq" %[[D7]], %[[D8]] : i32
//       CHECK:   llvm.cond_br %9, ^bb2, ^bb1
//       CHECK: ^bb1:
//       CHECK:   llvm.return %[[D7]] : i32
//       CHECK: ^bb2:
//       CHECK:   %[[D10:.*]] = llvm.load %[[D3]] : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
//       CHECK:   %[[D11:.*]] = llvm.mlir.constant(2 : i32) : i32
//       CHECK:   %[[D12:.*]] = llvm.call @hcgpuGetPyArg(%[[ARG1]], %[[D11]]) : (!llvm.ptr, i32) -> !llvm.ptr
//       CHECK:   %[[D13:.*]] = llvm.call @hcgpuConvertPyArray(%[[ARG0]], %[[D12]], %[[D1]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
//       CHECK:   %[[D14:.*]] = llvm.mlir.constant(0 : i32) : i32
//       CHECK:   %[[D15:.*]] = llvm.icmp "eq" %[[D13]], %[[D14]] : i32
//       CHECK:   llvm.cond_br %[[D15]], ^bb4, ^bb3
//       CHECK: ^bb3:
//       CHECK:   llvm.return %[[D13]] : i32
//       CHECK: ^bb4:
//       CHECK:   %[[D16:.*]] = llvm.load %[[D1]] : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
//       CHECK:   llvm.call @copy_kernel(%[[D4]], %[[D10]], %[[D16]]) : (!llvm.ptr, !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>) -> ()
//       CHECK:   %[[D17:.*]] = llvm.mlir.constant(0 : i32) : i32
//       CHECK:   llvm.return %[[D17]] : i32

func.func @copy_kernel_pyabi(%arg0: !hkernel.error_context, %arg1: !hkernel.py_args) -> i32 {
  %0 = ub.poison : !hkernel<current_group 2>
  %1 = hkernel.get_py_arg %arg1[1], %arg0 : !hkernel.memref_descriptor<memref<?x?xi32, strided<[?, ?], offset: ?>>>
  %2 = hkernel.get_py_arg %arg1[2], %arg0 : !hkernel.memref_descriptor<memref<?x?xi32, strided<[?, ?], offset: ?>>>
  call @copy_kernel(%0, %1, %2) : (!hkernel<current_group 2>, !hkernel.memref_descriptor<memref<?x?xi32, strided<[?, ?], offset: ?>>>, !hkernel.memref_descriptor<memref<?x?xi32, strided<[?, ?], offset: ?>>>) -> ()
  %ret = arith.constant 0 : i32
  return %ret : i32
}

func.func @copy_kernel(%arg0: !hkernel<current_group 2>, %arg1: !hkernel.memref_descriptor<memref<?x?xi32, strided<[?, ?], offset: ?>>>, %arg2: !hkernel.memref_descriptor<memref<?x?xi32, strided<[?, ?], offset: ?>>>) attributes {kernel.entrypoint} {
  return
}
