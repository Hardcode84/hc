// SPDX-FileCopyrightText: 2025 The HC Authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cstdlib>

void *getKernelImpl(void **handle, const void *data, size_t dataSize,
                    const char *kenrnelName) noexcept;

void suggestBlockSizeImpl(void *kernel, const size_t *globalSizes,
                          size_t *blockSizesRet, size_t nDim) noexcept;

void launchKernelImpl(void *kernel, const size_t *gridSizes,
                      const size_t *blockSizes, size_t nDim, void **args,
                      size_t nArgs, size_t sharedMemSize) noexcept;
