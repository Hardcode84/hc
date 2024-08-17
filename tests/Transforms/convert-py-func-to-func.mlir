// RUN: hc-opt -split-input-file %s --hc-convert-py-func-to-func-pass | FileCheck %s

// CHECK-LABEL: py_ir.module {
//       CHECK:  func.func private @func() -> !typing.value {
//       CHECK:  %[[R:.*]] = typing.type_constant #typing.type_attr<i32> : !typing.value
//       CHECK:  return %[[R]] : !typing.value
//       CHECK:  %[[F:.*]] = func.call @func() : () -> !typing.value
//       CHECK:  py_ir.module_end %[[F]] : !typing.value

py_ir.module {
  py_ir.static_func "private" @func () type () -> !typing.value {
    %2 = typing.type_constant #typing.type_attr<i32> : !typing.value
    py_ir.return %2 : !typing.value
  }
  %1 = py_ir.static_call @func : () -> !typing.value
  py_ir.module_end %1 : !typing.value
}

// -----

// CHECK-LABEL: py_ir.module {
//       CHECK:  func.func private @func() {
//       CHECK:  return
//       CHECK:  func.call @func() : () -> ()
//       CHECK:  %[[F:.*]] = py_ir.none
//       CHECK:  py_ir.module_end %[[F]] : none

py_ir.module {
  py_ir.static_func "private" @func () type () -> none {
    %2 = py_ir.none
    py_ir.return %2 : none
  }
  %1 = py_ir.static_call @func : () -> none
  py_ir.module_end %1 : none
}
