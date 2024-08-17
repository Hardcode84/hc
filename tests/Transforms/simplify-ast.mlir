// RUN: hc-opt -allow-unregistered-dialect -split-input-file %s --hc-simplify-ast-pass | FileCheck %s

// CHECK-LABEL: py_ast.func "func"()
//       CHECK: %[[R:.*]] = py_ast.constant #py_ast.none
//       CHECK: py_ast.return %[[R]]
//   CHECK-NOT: py_ast.return
py_ast.module {
  py_ast.func "func"() {
    py_ast.return
  }
}

// -----

// CHECK-LABEL: py_ast.func "func"()
//   CHECK-NOT: py_ast.pass
py_ast.module {
  py_ast.func "func"() {
    py_ast.pass
  }
}

// -----

// CHECK-LABEL: py_ast.func "func"()
//       CHECK: py_ast.return
//   CHECK-NOT: py_ast.return
py_ast.module {
  py_ast.func "func"() {
    py_ast.return
    py_ast.return
  }
}

// -----

// CHECK-LABEL: py_ast.func "func"()
//       CHECK: %[[R:.*]] = py_ast.constant #py_ast.none
//       CHECK: py_ast.return %[[R]]
py_ast.module {
  py_ast.func "func"() {
  }
}
