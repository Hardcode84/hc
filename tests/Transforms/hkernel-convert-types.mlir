// RUN: hc-opt -split-input-file %s --hc-convert-hkernel-types-pass | FileCheck %s

func.func @test(%arg1: !typing<symbol "W">) -> !hkernel<slice !typing<symbol "W"> : none : none> {
  %1 = hkernel.make_slice(%arg1  !typing<symbol "W"> :    :   ) -> !hkernel<slice !typing<symbol "W"> : none : none>
  return %1 : !hkernel<slice !typing<symbol "W"> : none : none>
}

// CHECK-LABEL: func @test
//  CHECK-SAME: (%[[ARG:.*]]: index)
//       CHECK: %[[R:.*]] = hkernel.make_slice(%[[ARG]] index : : ) -> !hkernel<slice index : none : none>
//       CHECK: return %[[R]] : !hkernel<slice index : none : none>
