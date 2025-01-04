// SPDX-FileCopyrightText: 2025 The HC Authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hc-hip-runtime_export.h"

#include <memory>

#define OFFLOAD_API_EXPORT HC_HIP_RUNTIME_EXPORT
#include "offload_api.h"

OlDevice olCreateDevice(const char *desc, OlErrorCallback errCallback,
                        void *ctx) noexcept {
  abort();
}
void olReleaseDevice(OlDevice dev) noexcept { abort(); }

OlModule olCreateModule(OlDevice dev, const void *data, size_t len) noexcept {
  abort();
}
void olReleaseModule(OlModule mod) noexcept { abort(); }

OlKernel olGetKernel(OlModule mod, const char *name) noexcept { abort(); }
void olReleaseKernel(OlKernel k) noexcept { abort(); }

int olSuggestBlockSize(OlKernel k, const size_t *globalsSizes,
                       size_t *blockSizesRet, size_t nDims) noexcept {
  abort();
}

OlQueue olCreateSyncQueue(OlDevice dev) noexcept { abort(); }
void olReleaseQueue(OlQueue q) noexcept { abort(); }

void *olAllocDevice(OlQueue q, size_t size, size_t align) noexcept { abort(); }

void olDeallocDevice(OlQueue q, void *data) noexcept { abort(); }

int olLaunchKernel(OlQueue q, OlKernel k, const size_t *gridSizes,
                   const size_t *blockSizes, size_t nDims, void **args,
                   size_t nArgs, size_t sharedMemSize) noexcept {
  abort();
}
