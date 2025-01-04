// SPDX-FileCopyrightText: 2025 The HC Authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hc-gpu-runtime-loader_export.h"

#define OFFLOAD_API_EXPORT HC_GPU_RUNTIME_LOADER_EXPORT
#include "offload_api.h"

OlDevice olCreateDevice(const char *desc, OlErrorCallback errCallback,
                        void *ctx) {
  abort();
}
void olReleaseDevice(OlDevice dev) { abort(); }

OlModule olCreateModule(OlDevice dev, const void *data, size_t len) { abort(); }
void olReleaseModule(OlModule mod) { abort(); }

OlKernel olGetKernel(OlModule mod, const char *name) { abort(); }
void olReleaseKernel(OlKernel k) { abort(); }

bool olSuggestBlockSize(OlKernel k, const size_t *globalsSizes,
                        size_t *blockSizesRet, size_t nDims) {
  abort();
}

OlQueue olCreateSyncQueue(OlDevice dev) { abort(); }
void olReleaseQueue(OlQueue q) { abort(); }

void *olAllocDevice(OlQueue q, size_t size, size_t align) { abort(); }
void olDeallocDevice(OlQueue q, void *data) { abort(); }

bool olLaunchKernel(OlQueue q, OlKernel k, const size_t *gridSizes,
                    const size_t *blockSizes, size_t nDims, void **args,
                    size_t nArgs, size_t sharedMemSize) {
  abort();
}
