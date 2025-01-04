// SPDX-FileCopyrightText: 2025 The HC Authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "GpuRuntime.hpp"

void *getKernelImpl(void **handle, const void *data, size_t dataSize) noexcept {
  abort();
}

void suggestBlockSizeImpl(void *kernel, const size_t *globalSizes,
                          size_t *blockSizesRet, size_t nDim) noexcept {
  abort();
}

void launchKernelImpl(void *kernel, const size_t *gridSizes,
                      const size_t *blockSizes, size_t nDim, void **args,
                      size_t nArgs, size_t sharedMemSize) noexcept {
  abort();
}
