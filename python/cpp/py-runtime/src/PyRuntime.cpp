// SPDX-FileCopyrightText: 2025 The HC Authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <nanobind/nanobind.h>

#include "PyRuntimeShared.hpp"
#include "hc-python-runtime_export.h"

namespace {
static bool traceFunctions = true;

struct FuncScope {
  FuncScope(const char *funcName) : name(funcName), enable(traceFunctions) {
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

extern "C" HC_PYTHON_RUNTIME_EXPORT int
hcgpuConvertPyArray(hc::ExceptionDesc *errorDesc, PyObject *obj, int rank,
                    void *ret) {
  LOG_FUNC();
  errorDesc->message = "Not implemented";
  return 1;
}

extern "C" HC_PYTHON_RUNTIME_EXPORT void *
hcgpuGetKernel(void **handle, const void *data, size_t dataSize) {
  LOG_FUNC();
  abort();
}

extern "C" HC_PYTHON_RUNTIME_EXPORT void
hcgpuSuggestBlockSize(void *kernel, const size_t *globalSizes,
                      size_t *blockSizesRet, size_t nDim) {
  LOG_FUNC();
  abort();
}

extern "C" HC_PYTHON_RUNTIME_EXPORT void
hcgpuLaunchKernel(void *kernel, const size_t *gridSizes,
                  const size_t *blockSizes, size_t nDim, void **args,
                  size_t nArgs, size_t sharedMemSize) {
  LOG_FUNC();
  abort();
}
