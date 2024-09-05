// RUN: hc-opt %s | hc-opt | FileCheck %s
// RUN: hc-opt %s --mlir-print-op-generic | hc-opt | FileCheck %s

// CHECK-LABEL: test_buffer_type
//  CHECK-SAME: (!hkernel<buffer <"A" x ("H" - "$GROUP_ID1" * "$GROUP_SHAPE1")> x f32>)
func.func private @test_buffer_type(%arg0: !hkernel<buffer <"A" x ("H" - "$GROUP_ID1" * "$GROUP_SHAPE1")> x f32>)
