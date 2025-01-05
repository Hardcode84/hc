// SPDX-FileCopyrightText: 2025 The HC Authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "GpuRuntime.hpp"

#include <cstdio>

#include "offload_api.h"

static void errCallback(void * /*ctx*/, OlSeverity sev, const char *desc) {
  const char *types[]{"error", "warning", "message"};
  const char *type = types[static_cast<int>(sev)];
  fprintf(stderr, "Offload %s: %s\n", type, desc);
  fflush(stderr);
}

[[noreturn]] static void fatal(const char *desc) {
  errCallback(nullptr, OlSeverity::Error, desc);
  abort();
}

static OlDevice device = nullptr;
static OlQueue queue = nullptr;

void *getKernelImpl(void **handle, const void *data, size_t dataSize,
                    const char *kenrnelName) noexcept {
  if (void *cached = *handle)
    return cached;

  if (!device) {
    device = olCreateDevice("hip:0", &errCallback, nullptr);
    if (!device)
      fatal("olCreateDevice failed");

    queue = olCreateQueue(device);
    if (!queue)
      fatal("olCreateSyncQueue failed");
  }

  OlModule module = olCreateModule(device, data, dataSize);
  if (!module)
    fatal("olCreateModule failed");

  OlKernel kernel = olGetKernel(module, kenrnelName);
  if (!kernel)
    fatal("olGetKernel failed");

  *handle = kernel;
  return kernel;
}

void suggestBlockSizeImpl(void *kernel, const size_t *globalSizes,
                          size_t *blockSizesRet, size_t nDim) noexcept {
  if (olSuggestBlockSize(static_cast<OlKernel>(kernel), globalSizes,
                         blockSizesRet, nDim))
    fatal("olSuggestBlockSize failed");
}

void launchKernelImpl(void *kernel, const size_t *gridSizes,
                      const size_t *blockSizes, size_t nDim, void **args,
                      size_t nArgs, size_t sharedMemSize) noexcept {
  if (olLaunchKernel(queue, kernel, gridSizes, blockSizes, nDim, args, nArgs,
                     sharedMemSize))
    fatal("olLaunchKernel failed");

  if (olSyncQueue(queue))
    fatal("olSyncQueue failed");
}
