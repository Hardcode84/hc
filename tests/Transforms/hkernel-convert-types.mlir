// RUN: hc-opt -split-input-file %s --hc-convert-hkernel-types-pass | FileCheck %s

func.func @test(%arg1: !typing<symbol "W">) -> !hkernel<slice !typing<symbol "W"> : none : none> {
  %1 = hkernel.make_slice(%arg1  !typing<symbol "W"> :    :   ) -> !hkernel<slice !typing<symbol "W"> : none : none>
  return %1 : !hkernel<slice !typing<symbol "W"> : none : none>
}

// CHECK-LABEL: func @test
//  CHECK-SAME: (%[[ARG:.*]]: index)
//       CHECK: %[[R:.*]] = hkernel.make_slice(%[[ARG]] index : : ) -> !hkernel<slice index : none : none>
//       CHECK: return %[[R]] : !hkernel<slice index : none : none>

// -----

func.func @test(%arg1: !hkernel<tensor <"W" x "H"> x f16>) -> !hkernel<tensor <"W" x "H"> x f16> {
  return %arg1 : !hkernel<tensor <"W" x "H"> x f16>
}

// CHECK-LABEL: func @test
//  CHECK-SAME: (%[[ARG:.*]]: tuple<memref<?x?xf16, strided<[?, ?], offset: ?>, #gpu.address_space<workgroup>>, memref<?x?xi1, strided<[?, ?], offset: ?>, #gpu.address_space<workgroup>>>)
//       CHECK: return %[[ARG]] : tuple<memref<?x?xf16, strided<[?, ?], offset: ?>, #gpu.address_space<workgroup>>, memref<?x?xi1, strided<[?, ?], offset: ?>, #gpu.address_space<workgroup>>>
