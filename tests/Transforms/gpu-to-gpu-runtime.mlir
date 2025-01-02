// RUN: hc-opt -split-input-file %s --hc-gpu-to-gpu-runtime-pass | FileCheck %s

// CHECK-LABEL: func @copy_kernel
//       CHECK:   llvm.call @hcgpuGetKernel
//       CHECK:   llvm.call @hcgpuSuggestBlockSize
//       CHECK:   llvm.call @hcgpuLaunchKernel
//   CHECK-NOT:  gpu.binary

module attributes {gpu.container_module} {
  func.func @copy_kernel(%arg0: !hkernel<current_group 2>, %arg1: !hkernel.memref_descriptor<memref<?x?xi32, strided<[?, ?], offset: ?>>>, %arg2: !hkernel.memref_descriptor<memref<?x?xi32, strided<[?, ?], offset: ?>>>) attributes {kernel.entrypoint} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = ub.poison : vector<1xi32>
    %c-1 = arith.constant -1 : index
    %1:5 = hkernel.memref_descriptor_cast %arg1 : !hkernel.memref_descriptor<memref<?x?xi32, strided<[?, ?], offset: ?>>> to !hkernel.ptr<i32>, index, index, index, index
    %2:5 = hkernel.memref_descriptor_cast %arg2 : !hkernel.memref_descriptor<memref<?x?xi32, strided<[?, ?], offset: ?>>> to !hkernel.ptr<i32>, index, index, index, index
    %3:2 = hkernel.suggest_block_size %1#2, %1#1 : index, index
    %4 = arith.cmpi sle, %1#1, %c0 : index
    %5 = arith.subi %c0, %1#1 : index
    %6 = arith.subi %1#1, %c1 : index
    %7 = arith.select %4, %5, %6 : index
    %8 = arith.divsi %7, %3#1 : index
    %9 = arith.subi %c0, %8 : index
    %10 = arith.addi %8, %c1 : index
    %11 = arith.select %4, %9, %10 : index
    %12 = arith.cmpi sle, %1#2, %c0 : index
    %13 = arith.subi %c0, %1#2 : index
    %14 = arith.subi %1#2, %c1 : index
    %15 = arith.select %12, %13, %14 : index
    %16 = arith.divsi %15, %3#0 : index
    %17 = arith.subi %c0, %16 : index
    %18 = arith.addi %16, %c1 : index
    %19 = arith.select %12, %17, %18 : index
    gpu.launch_func  @copy_kernel_kernel::@copy_kernel_kernel blocks in (%19, %11, %c1) threads in (%3#0, %3#1, %c1)  args(%3#1 : index, %1#3 : index, %1#0 : !hkernel.ptr<i32>, %1#1 : index, %1#2 : index, %2#3 : index, %2#0 : !hkernel.ptr<i32>)
    return
  }
  gpu.binary @copy_kernel_kernel  [#gpu.object<#rocdl.target, kernels = <[#gpu.kernel_metadata<"copy_kernel_kernel", !llvm.func<void (i64, i64, ptr, i64, i64, i64, ptr)>, metadata = {agpr_count = 4294967295 : i64, group_segment_fixed_size = 0 : i64, max_flat_workgroup_size = 256 : i64, private_segment_fixed_size = 0 : i64, reqd_workgroup_size = array<i32: -1, -1, -1>, sgpr_count = 28 : i64, sgpr_spill_count = 0 : i64, vgpr_count = 6 : i64, vgpr_spill_count = 0 : i64, wavefront_size = 64 : i64, workgroup_size_hint = array<i32: -1, -1, -1>}>]>, bin = "\7FELF">]
}
