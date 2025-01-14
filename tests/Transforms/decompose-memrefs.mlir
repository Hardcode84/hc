// RUN: hc-opt -split-input-file %s --hc-decompose-memrefs-pass --canonicalize | FileCheck %s

// CHECK-LABEL: func @test
//  CHECK-SAME:  (%[[ARG:.*]]: tuple<!hkernel.ptr<f32>>)
//       CHECK:  return %[[ARG]] : tuple<!hkernel.ptr<f32>>
func.func @test(%arg: memref<10xf32>) -> memref<10xf32> {
  return %arg : memref<10xf32>
}

// -----

// CHECK-LABEL: func @test
//  CHECK-SAME:  (%[[ARG:.*]]: tuple<!hkernel.ptr<f32>>)
//       CHECK:  return %[[ARG]] : tuple<!hkernel.ptr<f32>>
func.func @test(%arg: memref<10xf32, strided<[1], offset: ?>>) -> memref<10xf32, strided<[1], offset: ?>> {
  return %arg : memref<10xf32, strided<[1], offset: ?>>
}

// -----

// CHECK-LABEL: func @test
//  CHECK-SAME:  (%[[ARG:.*]]: tuple<!hkernel.ptr<f32>, index>)
//       CHECK:  return %[[ARG]] : tuple<!hkernel.ptr<f32>, index>
func.func @test(%arg: memref<?xf32>) -> memref<?xf32> {
  return %arg : memref<?xf32>
}

// -----

// CHECK-LABEL: func @test
//  CHECK-SAME:  (%[[ARG:.*]]: tuple<!hkernel.ptr<f32>, index, index>)
//       CHECK:  return %[[ARG]] : tuple<!hkernel.ptr<f32>, index, index>
func.func @test(%arg: memref<?xf32, strided<[?], offset: ?>>) -> memref<?xf32, strided<[?], offset: ?>> {
  return %arg : memref<?xf32, strided<[?], offset: ?>>
}

// -----

// CHECK-LABEL: func @test
//  CHECK-SAME:  (%[[ARG:.*]]: !hkernel.memref_descriptor<memref<?xf32>>)
//       CHECK:  %[[R:.*]]:2 = hkernel.memref_descriptor_cast %[[ARG]] : !hkernel.memref_descriptor<memref<?xf32>> to !hkernel.ptr<f32>, index
//       CHECK:  %[[RES:.*]] = hkernel.make_tuple %[[R]]#0, %[[R]]#1 : !hkernel.ptr<f32>, index -> tuple<!hkernel.ptr<f32>, index>
//       CHECK:  return %[[RES]] : tuple<!hkernel.ptr<f32>, index>
func.func @test(%arg: !hkernel.memref_descriptor<memref<?xf32>>) -> memref<?xf32> {
  %0 = hkernel.memref_descriptor_cast %arg : !hkernel.memref_descriptor<memref<?xf32>> to memref<?xf32>
  return %0 : memref<?xf32>
}

// -----

// CHECK-LABEL: func @test
//  CHECK-SAME:  (%[[ARG:.*]]: tuple<!hkernel.ptr<f32>, index>)
//       CHECK:  %[[C1:.*]] = arith.constant 1 : index
//       CHECK:  %[[RES:.*]] = hkernel.tuple_extract %[[ARG]] : tuple<!hkernel.ptr<f32>, index>[%[[C1]]] -> index
//       CHECK:  return %[[RES]] : index
func.func @test(%arg: memref<10x?xf32>) -> index {
  %c1 = arith.constant 1 : index
  %dim = memref.dim %arg, %c1 : memref<10x?xf32>
  return %dim : index
}

// -----

// CHECK-LABEL: func @test
//       CHECK:  %[[S:.*]] = arith.constant 10 : index
//       CHECK:  %[[P:.*]] = hkernel.ptr_alloca %[[S]] : index, !hkernel.ptr<f32>
//       CHECK:  %[[R:.*]] = hkernel.make_tuple %[[P]] : !hkernel.ptr<f32> -> tuple<!hkernel.ptr<f32>>
//       CHECK:  return %[[R]] : tuple<!hkernel.ptr<f32>>
func.func @test() -> memref<10xf32> {
  %0 = memref.alloca() : memref<10xf32>
  return %0 : memref<10xf32>
}

// -----

// CHECK-LABEL: func @test
//  CHECK-SAME:  (%[[ARG:.*]]: index)
//       CHECK:  %[[P:.*]] = hkernel.ptr_alloca %[[ARG]] : index, !hkernel.ptr<f32>
//       CHECK:  %[[R:.*]] = hkernel.make_tuple %[[P]], %[[ARG]] : !hkernel.ptr<f32>, index -> tuple<!hkernel.ptr<f32>, index>
//       CHECK:  return %[[R]] : tuple<!hkernel.ptr<f32>, index>
func.func @test(%arg: index) -> memref<?xf32> {
  %0 = memref.alloca(%arg) : memref<?xf32>
  return %0 : memref<?xf32>
}

// -----

//       CHECK: #[[MAP:.*]] = affine_map<()[s0, s1] -> (s0 * 7 + s1)>
// CHECK-LABEL: func @test
//  CHECK-SAME:  (%[[ARG0:.*]]: tuple<!hkernel.ptr<f32>>, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index)
//   CHECK-DAG:  %[[C0:.*]] = arith.constant 0 : index
//       CHECK:  %[[OFFSET:.*]] = affine.apply #[[MAP]]()[%[[ARG1]], %[[ARG2]]]
//       CHECK:  %[[PTR:.*]] = hkernel.tuple_extract %[[ARG0]] : tuple<!hkernel.ptr<f32>>[%[[C0]]] -> !hkernel.ptr<f32>
//       CHECK:  %[[PTR2:.*]] = hkernel.ptr_add %[[PTR]] : !hkernel.ptr<f32>, %[[OFFSET]] : index
//       CHECK:  %[[RES:.*]] = hkernel.make_tuple %[[PTR2]] : !hkernel.ptr<f32> -> tuple<!hkernel.ptr<f32>>
//       CHECK:  return %[[RES]] : tuple<!hkernel.ptr<f32>>
func.func @test(%arg0: memref<5x7xf32>, %arg1: index, %arg2: index) -> memref<2x3xf32, strided<[7, 1], offset: ?>> {
  %1 = memref.subview %arg0[%arg1, %arg2] [2, 3] [1, 1] : memref<5x7xf32> to memref<2x3xf32, strided<[7, 1], offset: ?>>
  return %1 : memref<2x3xf32, strided<[7, 1], offset: ?>>
}

// -----

//       CHECK: #[[MAP:.*]] = affine_map<()[s0, s1, s2] -> (s0 * s1 + s2)>
// CHECK-LABEL: func @test
//  CHECK-SAME:  (%[[ARG0:.*]]: tuple<!hkernel.ptr<f32>, index, index>, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index)
//   CHECK-DAG:  %[[C0:.*]] = arith.constant 0 : index
//   CHECK-DAG:  %[[C2:.*]] = arith.constant 2 : index
//       CHECK:  %[[SIZE:.*]] = hkernel.tuple_extract %[[ARG0]] : tuple<!hkernel.ptr<f32>, index, index>[%[[C2]]] -> index
//       CHECK:  %[[OFFSET:.*]] = affine.apply #[[MAP]]()[%[[ARG1]], %[[SIZE]], %[[ARG2]]]
//       CHECK:  %[[PTR:.*]] = hkernel.tuple_extract %[[ARG0]] : tuple<!hkernel.ptr<f32>, index, index>[%[[C0]]] -> !hkernel.ptr<f32>
//       CHECK:  %[[PTR2:.*]] = hkernel.ptr_add %[[PTR]] : !hkernel.ptr<f32>, %[[OFFSET]] : index
//       CHECK:  %[[RES:.*]] = hkernel.make_tuple %[[PTR2]], %[[SIZE]] : !hkernel.ptr<f32>, index -> tuple<!hkernel.ptr<f32>, index>
//       CHECK:  return %[[RES]] : tuple<!hkernel.ptr<f32>, index>
func.func @test(%arg0: memref<?x?xf32>, %arg1: index, %arg2: index) -> memref<2x3xf32, strided<[?, 1], offset: ?>> {
  %1 = memref.subview %arg0[%arg1, %arg2] [2, 3] [1, 1] : memref<?x?xf32> to memref<2x3xf32, strided<[?, 1], offset: ?>>
  return %1 : memref<2x3xf32, strided<[?, 1], offset: ?>>
}

// -----

//       CHECK: #[[MAP:.*]] = affine_map<()[s0, s1, s2, s3] -> (s0 * s1 + s2 * s3)>
// CHECK-LABEL: func @test
//  CHECK-SAME:  (%[[ARG0:.*]]: tuple<!hkernel.ptr<f32>, index, index, index, index>, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index)
//   CHECK-DAG:  %[[C0:.*]] = arith.constant 0 : index
//   CHECK-DAG:  %[[C3:.*]] = arith.constant 3 : index
//   CHECK-DAG:  %[[C4:.*]] = arith.constant 4 : index
//       CHECK:  %[[STRIDE1:.*]] = hkernel.tuple_extract %[[ARG0]] : tuple<!hkernel.ptr<f32>, index, index, index, index>[%[[C3]]] -> index
//       CHECK:  %[[STRIDE2:.*]] = hkernel.tuple_extract %[[ARG0]] : tuple<!hkernel.ptr<f32>, index, index, index, index>[%[[C4]]] -> index
//       CHECK:  %[[OFFSET:.*]] = affine.apply #[[MAP]]()[%[[ARG1]], %[[STRIDE1]], %[[ARG2]], %[[STRIDE2]]]
//       CHECK:  %[[PTR:.*]] = hkernel.tuple_extract %[[ARG0]] : tuple<!hkernel.ptr<f32>, index, index, index, index>[%[[C0]]] -> !hkernel.ptr<f32>
//       CHECK:  %[[PTR2:.*]] = hkernel.ptr_add %[[PTR]] : !hkernel.ptr<f32>, %[[OFFSET]] : index
//       CHECK:  %[[RES:.*]] = hkernel.make_tuple %[[PTR2]], %[[STRIDE1]], %[[STRIDE2]] : !hkernel.ptr<f32>, index, index -> tuple<!hkernel.ptr<f32>, index, index>
//       CHECK:  return %[[RES]] : tuple<!hkernel.ptr<f32>, index, index>
func.func @test(%arg0: memref<?x?xf32, strided<[?, ?], offset: ?>>, %arg1: index, %arg2: index) -> memref<2x3xf32, strided<[?, ?], offset: ?>> {
  %1 = memref.subview %arg0[%arg1, %arg2] [2, 3] [1, 1] : memref<?x?xf32, strided<[?, ?], offset: ?>> to memref<2x3xf32, strided<[?, ?], offset: ?>>
  return %1 : memref<2x3xf32, strided<[?, ?], offset: ?>>
}

// -----

//       CHECK: #[[MAP:.*]] = affine_map<()[s0, s1] -> (s0 * 7 + s1)>
// CHECK-LABEL: func @test
//  CHECK-SAME:  (%[[ARG0:.*]]: tuple<!hkernel.ptr<f32>>, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index)
//   CHECK-DAG:  %[[C0:.*]] = arith.constant 0 : index
//       CHECK:  %[[OFFSET:.*]] = affine.apply #[[MAP]]()[%[[ARG1]], %[[ARG2]]]
//       CHECK:  %[[PTR:.*]] = hkernel.tuple_extract %[[ARG0]] : tuple<!hkernel.ptr<f32>>[%[[C0]]] -> !hkernel.ptr<f32>
//       CHECK:  %[[RES:.*]] = hkernel.ptr_load %[[PTR]] : !hkernel.ptr<f32>[%[[OFFSET]] : index] : f32
//       CHECK:  return %[[RES]] : f32
func.func @test(%arg0: memref<5x7xf32>, %arg1: index, %arg2: index) -> f32 {
  %1 = memref.load %arg0[%arg1, %arg2] : memref<5x7xf32>
  return %1 : f32
}

// -----

//       CHECK: #[[MAP:.*]] = affine_map<()[s0, s1] -> (s0 * 7 + s1)>
// CHECK-LABEL: func @test
//  CHECK-SAME:  (%[[ARG0:.*]]: tuple<!hkernel.ptr<f32>>, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index, %[[ARG3:.*]]: f32)
//   CHECK-DAG:  %[[C0:.*]] = arith.constant 0 : index
//       CHECK:  %[[OFFSET:.*]] = affine.apply #[[MAP]]()[%[[ARG1]], %[[ARG2]]]
//       CHECK:  %[[PTR:.*]] = hkernel.tuple_extract %[[ARG0]] : tuple<!hkernel.ptr<f32>>[%[[C0]]] -> !hkernel.ptr<f32>
//       CHECK:  hkernel.ptr_store %[[ARG3]] : f32 %[[PTR]] : !hkernel.ptr<f32>[%[[OFFSET]] : index]
func.func @test(%arg0: memref<5x7xf32>, %arg1: index, %arg2: index, %arg3: f32)  {
  memref.store %arg3, %arg0[%arg1, %arg2] : memref<5x7xf32>
  return
}

// -----

//       CHECK: #[[MAP:.*]] = affine_map<()[s0, s1] -> (s0 * 7 + s1)>
// CHECK-LABEL: func @test
//  CHECK-SAME:  (%[[ARG0:.*]]: tuple<!hkernel.ptr<f32>>, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index)
//   CHECK-DAG:  %[[C0:.*]] = arith.constant 0 : index
//       CHECK:  %[[OFFSET:.*]] = affine.apply #[[MAP]]()[%[[ARG1]], %[[ARG2]]]
//       CHECK:  %[[PTR:.*]] = hkernel.tuple_extract %[[ARG0]] : tuple<!hkernel.ptr<f32>>[%[[C0]]] -> !hkernel.ptr<f32>
//       CHECK:  %[[RES:.*]] = hkernel.ptr_load %[[PTR]] : !hkernel.ptr<f32>[%[[OFFSET]] : index] : vector<2xf32>
//       CHECK:  return %[[RES]] : vector<2xf32>
func.func @test(%arg0: memref<5x7xf32>, %arg1: index, %arg2: index) -> vector<2xf32> {
  %1 = vector.load %arg0[%arg1, %arg2] : memref<5x7xf32>, vector<2xf32>
  return %1 : vector<2xf32>
}

// -----

//       CHECK: #[[MAP:.*]] = affine_map<()[s0, s1] -> (s0 * 7 + s1)>
// CHECK-LABEL: func @test
//  CHECK-SAME:  (%[[ARG0:.*]]: tuple<!hkernel.ptr<f32>>, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index, %[[ARG3:.*]]: vector<2xf32>)
//   CHECK-DAG:  %[[C0:.*]] = arith.constant 0 : index
//       CHECK:  %[[OFFSET:.*]] = affine.apply #[[MAP]]()[%[[ARG1]], %[[ARG2]]]
//       CHECK:  %[[PTR:.*]] = hkernel.tuple_extract %[[ARG0]] : tuple<!hkernel.ptr<f32>>[%[[C0]]] -> !hkernel.ptr<f32>
//       CHECK:  hkernel.ptr_store %[[ARG3]] : vector<2xf32> %[[PTR]] : !hkernel.ptr<f32>[%[[OFFSET]] : index]
func.func @test(%arg0: memref<5x7xf32>, %arg1: index, %arg2: index, %arg3: vector<2xf32>)  {
  vector.store %arg3, %arg0[%arg1, %arg2] : memref<5x7xf32>, vector<2xf32>
  return
}

// -----

//       CHECK: #[[MAP:.*]] = affine_map<()[s0, s1] -> (s0 * 7 + s1)>
// CHECK-LABEL: func @test
//  CHECK-SAME:  (%[[ARG0:.*]]: tuple<!hkernel.ptr<f32>>, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index, %[[ARG3:.*]]: vector<2xi1>, %[[ARG4:.*]]: vector<2xf32>)
//   CHECK-DAG:  %[[C0:.*]] = arith.constant 0 : index
//       CHECK:  %[[OFFSET:.*]] = affine.apply #[[MAP]]()[%[[ARG1]], %[[ARG2]]]
//       CHECK:  %[[PTR:.*]] = hkernel.tuple_extract %[[ARG0]] : tuple<!hkernel.ptr<f32>>[%[[C0]]] -> !hkernel.ptr<f32>
//       CHECK:  %[[RES:.*]] = hkernel.ptr_load %[[PTR]] : !hkernel.ptr<f32>[%[[OFFSET]] : index] mask %[[ARG3]] : vector<2xi1>, %[[ARG4]] : vector<2xf32> : vector<2xf32>
//       CHECK:  return %[[RES]] : vector<2xf32>
func.func @test(%arg0: memref<5x7xf32>, %arg1: index, %arg2: index, %mask: vector<2xi1>, %passthru: vector<2xf32>) -> vector<2xf32> {
  %1 = vector.maskedload %arg0[%arg1, %arg2], %mask, %passthru : memref<5x7xf32>, vector<2xi1>, vector<2xf32> into vector<2xf32>
  return %1 : vector<2xf32>
}

// -----

//       CHECK: #[[MAP:.*]] = affine_map<()[s0, s1] -> (s0 * 7 + s1)>
// CHECK-LABEL: func @test
//  CHECK-SAME:  (%[[ARG0:.*]]: tuple<!hkernel.ptr<f32>>, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index, %[[ARG3:.*]]: vector<2xf32>, %[[ARG4:.*]]: vector<2xi1>)
//   CHECK-DAG:  %[[C0:.*]] = arith.constant 0 : index
//       CHECK:  %[[OFFSET:.*]] = affine.apply #[[MAP]]()[%[[ARG1]], %[[ARG2]]]
//       CHECK:  %[[PTR:.*]] = hkernel.tuple_extract %[[ARG0]] : tuple<!hkernel.ptr<f32>>[%[[C0]]] -> !hkernel.ptr<f32>
//       CHECK:  hkernel.ptr_store %[[ARG3]] : vector<2xf32> %[[PTR]] : !hkernel.ptr<f32>[%[[OFFSET]] : index] mask %[[ARG4]] : vector<2xi1>
func.func @test(%arg0: memref<5x7xf32>, %arg1: index, %arg2: index, %arg3: vector<2xf32>, %mask: vector<2xi1>)  {
  vector.maskedstore %arg0[%arg1, %arg2], %mask, %arg3 : memref<5x7xf32>, vector<2xi1>, vector<2xf32>
  return
}

// -----

// CHECK-LABEL: func @test
//       CHECK:   gpu.launch
//       CHECK:   %[[PTR:.*]] = hkernel.ptr_dynamic_shared_mem : !hkernel.ptr<i8, #gpu.address_space<workgroup>>
//       CHECK:   hkernel.ptr_store %{{.*}} : i8 %[[PTR]] : !hkernel.ptr<i8, #gpu.address_space<workgroup>>[%{{.*}} : index]
func.func @test(%arg0: index, %arg1: index, %arg2: index, %arg3: index, %arg4: index, %arg5: index, %shmem: i32) {
  gpu.launch blocks(%arg6, %arg7, %arg8) in (%arg9 = %arg0, %arg10 = %arg1, %arg11 = %arg2) threads(%arg12, %13, %arg14) in (%arg15 = %arg3, %arg16 = %arg4, %arg17 = %arg5) dynamic_shared_memory_size %shmem {
    %0 = gpu.dynamic_shared_memory : memref<?xi8, #gpu.address_space<workgroup>>
    %v = arith.constant 0 : i8
    %c0 = arith.constant 0 : index
    memref.store %v, %0[%c0] : memref<?xi8, #gpu.address_space<workgroup>>
    gpu.terminator
  }
  return
}

// -----

// CHECK-LABEL: func @test
//  CHECK-SAME:  (%[[ARG0:.*]]: tuple<!hkernel.ptr<i8, #gpu.address_space<workgroup>>, index>, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index, %[[ARG3:.*]]: index, %[[ARG4:.*]]: index)
//   CHECK-DAG:  %[[C0:.*]] = arith.constant 0 : index
//       CHECK:  %[[PTR:.*]] = hkernel.tuple_extract %[[ARG0]] : tuple<!hkernel.ptr<i8, #gpu.address_space<workgroup>>, index>[%[[C0]]] -> !hkernel.ptr<i8, #gpu.address_space<workgroup>>
//       CHECK:  %[[PTR1:.*]] = hkernel.ptr_add %[[PTR]] : !hkernel.ptr<i8, #gpu.address_space<workgroup>>, %[[ARG1]] : index
//       CHECK:  %[[CAST:.*]] = hkernel.cast %[[PTR1]] : !hkernel.ptr<i8, #gpu.address_space<workgroup>> to !hkernel.ptr<i32, #gpu.address_space<workgroup>>
//       CHECK:  %[[RES:.*]] = hkernel.make_tuple %[[CAST]], %[[ARG2]], %[[ARG3]], %[[ARG4]] : !hkernel.ptr<i32, #gpu.address_space<workgroup>>, index, index, index -> tuple<!hkernel.ptr<i32, #gpu.address_space<workgroup>>, index, index, index>
//       CHECK:  return %[[RES]] : tuple<!hkernel.ptr<i32, #gpu.address_space<workgroup>>, index, index, index>
func.func @test(%arg0: memref<?xi8, #gpu.address_space<workgroup>>, %arg1: index, %arg2: index, %arg3: index, %arg4: index) -> memref<?x?x1x1x?xi32, #gpu.address_space<workgroup>> {
  %view = memref.view %arg0[%arg1][%arg2, %arg3, %arg4] : memref<?xi8, #gpu.address_space<workgroup>> to memref<?x?x1x1x?xi32, #gpu.address_space<workgroup>>
  return %view : memref<?x?x1x1x?xi32, #gpu.address_space<workgroup>>
}
