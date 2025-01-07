// RUN: hc-opt -split-input-file %s --hc-expand-shared-allocs-pass | FileCheck %s

func.func private @use(%arg0: memref<1x1xi32, #gpu.address_space<workgroup>>)

// CHECK-LABEL: func @test
//       CHECK:   gpu.launch blocks(%{{.*}}, %{{.*}}, %{{.*}}) in (%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}) threads
//  CHECK-SAME:   (%[[T0:.*]], %[[T1:.*]], %[[T2:.*]]) in (%[[TS0:.*]] = %{{.*}}, %[[TS1:.*]] = %{{.*}}, %[[TS2:.*]] = %{{.*}})
//       CHECK:   %[[ALLOC1:.*]] = memref.alloc(%[[TS0]], %[[TS1]], %[[TS2]]) : memref<?x?x?x1x1xi32, #gpu.address_space<workgroup>>
//       CHECK:   %[[SUBVIEW1:.*]] = memref.subview %[[ALLOC1]][%[[T0]], %[[T1]], %[[T2]], 0, 0] [1, 1, 1, 1, 1] [1, 1, 1, 1, 1] : memref<?x?x?x1x1xi32, #gpu.address_space<workgroup>> to memref<1x1xi32, strided<[?, ?], offset: ?>, #gpu.address_space<workgroup>>
//       CHECK:   %[[CAST1:.*]] = memref.cast %[[SUBVIEW1]] : memref<1x1xi32, strided<[?, ?], offset: ?>, #gpu.address_space<workgroup>> to memref<1x1xi32, #gpu.address_space<workgroup>>
//       CHECK:   %[[ALLOC2:.*]] = memref.alloca(%[[TS0]], %[[TS1]], %[[TS2]]) : memref<?x?x?x1x1xi32, #gpu.address_space<workgroup>>
//       CHECK:   %[[SUBVIEW2:.*]] = memref.subview %[[ALLOC2]][%[[T0]], %[[T1]], %[[T2]], 0, 0] [1, 1, 1, 1, 1] [1, 1, 1, 1, 1] : memref<?x?x?x1x1xi32, #gpu.address_space<workgroup>> to memref<1x1xi32, strided<[?, ?], offset: ?>, #gpu.address_space<workgroup>>
//       CHECK:   %[[CAST2:.*]] = memref.cast %[[SUBVIEW2]] : memref<1x1xi32, strided<[?, ?], offset: ?>, #gpu.address_space<workgroup>> to memref<1x1xi32, #gpu.address_space<workgroup>>
//       CHECK:   func.call @use(%[[CAST1]]) : (memref<1x1xi32, #gpu.address_space<workgroup>>) -> ()
//       CHECK:   func.call @use(%[[CAST2]]) : (memref<1x1xi32, #gpu.address_space<workgroup>>) -> ()
func.func @test(%arg0: index, %arg1: index, %arg2: index, %arg3: index, %arg4: index, %arg5: index)  {
  gpu.launch blocks(%arg6, %arg7, %arg8) in (%arg9 = %arg0, %arg10 = %arg1, %arg11 = %arg2) threads(%arg12, %13, %arg14) in (%arg15 = %arg3, %arg16 = %arg4, %arg17 = %arg5) {
    %alloc1 = memref.alloc() {kernel.alloc_expand} : memref<1x1xi32, #gpu.address_space<workgroup>>
    %alloc2 = memref.alloca() {kernel.alloc_expand} : memref<1x1xi32, #gpu.address_space<workgroup>>
    func.call @use(%alloc1) : (memref<1x1xi32, #gpu.address_space<workgroup>>) -> ()
    func.call @use(%alloc2) : (memref<1x1xi32, #gpu.address_space<workgroup>>) -> ()
    gpu.terminator
  }
  return
}
