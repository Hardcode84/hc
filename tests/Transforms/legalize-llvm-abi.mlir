// RUN: hc-opt -split-input-file %s --hc-legalize-llvm-abi-pass | FileCheck %s

// CHECK-LABEL: func @func1
//  CHECK-SAME:   (%[[ARG0:.*]]: !llvm.ptr, %[[ARG1:.*]]: !llvm.ptr)
//       CHECK:   %[[R1:.*]] = llvm.load %[[ARG1]] : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
//       CHECK:   llvm.return %[[R1]]
llvm.func @func1(%arg0: !llvm.ptr, %arg1: !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>) -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> {
  llvm.return %arg1 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
}

// CHECK-LABEL: func @func2
//  CHECK-SAME:   (%[[ARG2:.*]]: !llvm.ptr, %[[ARG3:.*]]: !llvm.ptr)
//       CHECK:   %[[R2:.*]] = llvm.load %[[ARG3]] : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
//       CHECK:   %[[ONE:.*]] = llvm.mlir.constant(1 : i64) : i64
//       CHECK:   %[[PTR:.*]] = llvm.alloca %[[ONE]] x !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr
//       CHECK:   llvm.store %[[R2]], %[[PTR]] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>, !llvm.ptr
//       CHECK:   %[[R3:.*]] = llvm.call @func1(%[[ARG2]], %[[PTR]]) : (!llvm.ptr, !llvm.ptr) -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
//       CHECK:   llvm.return
llvm.func @func2(%arg0: !llvm.ptr, %arg1: !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>) {
  %1 = llvm.call @func1(%arg0, %arg1) : (!llvm.ptr, !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>) -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  llvm.return
}
