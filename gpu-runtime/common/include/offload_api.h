// SPDX-FileCopyrightText: 2025 The HC Authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cstdlib>

using OlDevice = void *;
using OlModule = void *;
using OlKernel = void *;
using OlQueue = void *;
using OlEvent = void *;

enum class OlSeverity : int {
  Error = 0,
  Warning = 1,
  Message = 2,
};

using OlErrorCallback = void (*)(void *ctx, OlSeverity sev, const char *desc);

#ifndef OFFLOAD_API_EXPORT
#define OFFLOAD_API_EXPORT
#endif

#define OFFLOAD_NOEXEPT noexcept

extern "C" {
// backend and device id encoded in `desc`
// All functions return either `false` or nullptr on error and call errCallback
// with details.
OFFLOAD_API_EXPORT OlDevice olCreateDevice(const char *desc,
                                           OlErrorCallback errCallback,
                                           void *ctx) OFFLOAD_NOEXEPT;
OFFLOAD_API_EXPORT void olReleaseDevice(OlDevice dev) OFFLOAD_NOEXEPT;

// Creates GPU module from backend-specific binary blob
OFFLOAD_API_EXPORT OlModule olCreateModule(OlDevice dev, const void *data,
                                           size_t len) OFFLOAD_NOEXEPT;
OFFLOAD_API_EXPORT void olReleaseModule(OlModule mod) OFFLOAD_NOEXEPT;

OFFLOAD_API_EXPORT OlKernel olGetKernel(OlModule mod,
                                        const char *name) OFFLOAD_NOEXEPT;
OFFLOAD_API_EXPORT void olReleaseKernel(OlKernel k) OFFLOAD_NOEXEPT;

OFFLOAD_API_EXPORT int olSuggestBlockSize(OlKernel k,
                                          const size_t *globalsSizes,
                                          size_t *blockSizesRet,
                                          size_t nDims) OFFLOAD_NOEXEPT;

// Sync API
OFFLOAD_API_EXPORT OlQueue olCreateSyncQueue(OlDevice dev) OFFLOAD_NOEXEPT;
OFFLOAD_API_EXPORT void olReleaseQueue(OlQueue q) OFFLOAD_NOEXEPT;

OFFLOAD_API_EXPORT void *olAllocDevice(OlQueue q, size_t size,
                                       size_t align) OFFLOAD_NOEXEPT;
OFFLOAD_API_EXPORT void olDeallocDevice(OlQueue q, void *data) OFFLOAD_NOEXEPT;

OFFLOAD_API_EXPORT int olLaunchKernel(OlQueue q, OlKernel k,
                                      const size_t *gridSizes,
                                      const size_t *blockSizes, size_t nDims,
                                      void **args, size_t nArgs,
                                      size_t sharedMemSize) OFFLOAD_NOEXEPT;
}
