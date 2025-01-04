// SPDX-FileCopyrightText: 2025 The HC Authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <nanobind/nanobind.h>

#include "PyRuntimeShared.hpp"
#include "hc-python-runtime_export.h"

extern "C" HC_PYTHON_RUNTIME_EXPORT PyObject *hcgpuGetPyArg(PyObject *args,
                                                            int index) {
  abort();
}

extern "C" HC_PYTHON_RUNTIME_EXPORT int
hcgpuConvertPyArray(hc::ExceptionDesc *errorDesc, PyObject *obj, void *ret) {
  abort();
}

extern "C" HC_PYTHON_RUNTIME_EXPORT void *
hcgpuGetKernel(void **handle, const void *data, size_t dataSize) {
  abort();
}

extern "C" HC_PYTHON_RUNTIME_EXPORT void
hcgpuSuggestBlockSize(void *kernel, const size_t *globalSizes,
                      size_t *blockSizesRet, size_t nDim) {
  abort();
}

extern "C" HC_PYTHON_RUNTIME_EXPORT void
hcgpuLaunchKernel(void *kernel, const size_t *gridSizes,
                  const size_t *blockSizes, size_t nDim, void **args,
                  size_t nArgs, size_t sharedMemSize) {
  abort();
}
