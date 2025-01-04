// SPDX-FileCopyrightText: 2025 The HC Authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "GpuRuntime.hpp"

#include <cstdio>

#include "offload_api.h"

static void errCallback(void * /*ctx*/, OlSeverity /*sev*/, const char *desc) {
  fprintf(stderr, "Offload error: %s\n", desc);
  fflush(stderr);
}

static OlDevice device = nullptr;
static OlQueue queue = nullptr;

void *getKernelImpl(void **handle, const void *data, size_t dataSize) noexcept {
  if (void *cached = *handle)
    return cached;

  if (!device) {
    device = olCreateDevice("hip:0", &errCallback, nullptr);
    if (!device)
      abort();

    queue = olCreateSyncQueue(device);
    if (!queue)
      abort();
  }

  OlModule module = olCreateModule(device, data, dataSize);
  if (!module)
    abort();

  OlKernel kernel = olGetKernel(module, "main");
  if (!kernel)
    abort();

  *handle = kernel;
  return kernel;
}

void suggestBlockSizeImpl(void *kernel, const size_t *globalSizes,
                          size_t *blockSizesRet, size_t nDim) noexcept {
  if (olSuggestBlockSize(static_cast<OlKernel>(kernel), globalSizes,
                         blockSizesRet, nDim))
    abort();
}

void launchKernelImpl(void *kernel, const size_t *gridSizes,
                      const size_t *blockSizes, size_t nDim, void **args,
                      size_t nArgs, size_t sharedMemSize) noexcept {
  if (olLaunchKernel(queue, kernel, gridSizes, blockSizes, nDim, args, nArgs,
                     sharedMemSize))
    abort();
}
