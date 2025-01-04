// SPDX-FileCopyrightText: 2025 The HC Authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PyRuntimeShared.hpp"
#include "hc-python-runtime_export.h"

#include "GpuRuntime.hpp"
#include "PyABI.hpp"

namespace {
static bool TraceFunctions = false;

struct FuncScope {
  FuncScope(const char *funcName) : name(funcName), enable(TraceFunctions) {
    if (enable) {
      fprintf(stdout, "%s enter\n", name);
      fflush(stdout);
    }
  }
  FuncScope(const FuncScope &) = delete;
  ~FuncScope() {
    if (enable) {
      fprintf(stdout, "%s exit\n", name);
      fflush(stdout);
    }
  }

private:
  const char *name;
  bool enable;
};
} // namespace
#define LOG_FUNC() FuncScope _scope(__func__)

extern "C" HC_PYTHON_RUNTIME_EXPORT void
hcgpuEnableTraceFunctions(int val) noexcept {
  TraceFunctions = val;
}

extern "C" HC_PYTHON_RUNTIME_EXPORT int
hcgpuConvertPyArray(hc::ExceptionDesc *errorDesc, void *obj, int rank,
                    void *ret) noexcept {
  LOG_FUNC();
  try {
    convertPyArrayImpl(obj, rank, ret);
    return 0;
  } catch (const std::exception &e) {
    errorDesc->message = e.what();
    return 1;
  }
}

extern "C" HC_PYTHON_RUNTIME_EXPORT void *
hcgpuGetKernel(void **handle, const void *data, size_t dataSize) noexcept {
  LOG_FUNC();
  return getKernelImpl(handle, data, dataSize);
}

extern "C" HC_PYTHON_RUNTIME_EXPORT void
hcgpuSuggestBlockSize(void *kernel, const size_t *globalSizes,
                      size_t *blockSizesRet, size_t nDim) noexcept {
  LOG_FUNC();
  return suggestBlockSizeImpl(kernel, globalSizes, blockSizesRet, nDim);
}

extern "C" HC_PYTHON_RUNTIME_EXPORT void
hcgpuLaunchKernel(void *kernel, const size_t *gridSizes,
                  const size_t *blockSizes, size_t nDim, void **args,
                  size_t nArgs, size_t sharedMemSize) noexcept {
  LOG_FUNC();
  return launchKernelImpl(kernel, gridSizes, blockSizes, nDim, args, nArgs,
                          sharedMemSize);
}
