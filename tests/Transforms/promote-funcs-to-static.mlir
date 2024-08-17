// RUN: hc-opt -split-input-file %s --hc-pyir-promote-funcs-to-static-pass | FileCheck %s

// CHECK-LABEL: py_ir.module {
//       CHECK: %[[SYM:.*]] = py_ir.sym_constant @func : !py_ir.undefined
//       CHECK: py_ir.static_func "private" @func () type () -> !typing.value {
//       CHECK: %[[R:.*]] = typing.type_constant #typing.type_attr<i32> : !typing.value
//       CHECK: py_ir.return %[[R]] : !typing.value
//       CHECK: py_ir.module_end %[[SYM]]
py_ir.module {
  %1 = py_ir.func "func" () capture () -> !py_ir.undefined {
    %2 = typing.type_constant #typing.type_attr<i32> : !typing.value
    py_ir.return %2 : !typing.value
  }
  py_ir.module_end %1 : !py_ir.undefined
}
