// SPDX-FileCopyrightText: 2025 The HC Authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

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

struct MemrefDescriptor {
  void *allocated;
  void *aligned;
  intptr_t offset;
  intptr_t sizesAndStrides[1];
};

namespace py = nanobind;

static void convertPyArrayImpl(hc::ExceptionDesc *errorDesc, PyObject *obj,
                               int rank, MemrefDescriptor *ret) {
  auto array = py::cast<py::ndarray<>>(py::handle(obj));
  if (array.ndim() != rank)
    throw std::runtime_error("Invalid rank");

  ret->allocated = array.data();
  ret->aligned = array.data();
  ret->offset = 0;
  for (int i = 0; i < rank; ++i) {
    ret->sizesAndStrides[i] = array.shape(i);
    ret->sizesAndStrides[i + rank] = array.stride(i);
  }
}

extern "C" HC_PYTHON_RUNTIME_EXPORT int
hcgpuConvertPyArray(hc::ExceptionDesc *errorDesc, PyObject *obj, int rank,
                    MemrefDescriptor *ret) noexcept {
  LOG_FUNC();
  try {
    convertPyArrayImpl(errorDesc, obj, rank, ret);
    return 0;
  } catch (const std::exception &e) {
    errorDesc->message = e.what();
    return 1;
  }
}

extern "C" HC_PYTHON_RUNTIME_EXPORT void *
hcgpuGetKernel(void **handle, const void *data, size_t dataSize) noexcept {
  LOG_FUNC();
  abort();
}

extern "C" HC_PYTHON_RUNTIME_EXPORT void
hcgpuSuggestBlockSize(void *kernel, const size_t *globalSizes,
                      size_t *blockSizesRet, size_t nDim) noexcept {
  LOG_FUNC();
  abort();
}

extern "C" HC_PYTHON_RUNTIME_EXPORT void
hcgpuLaunchKernel(void *kernel, const size_t *gridSizes,
                  const size_t *blockSizes, size_t nDim, void **args,
                  size_t nArgs, size_t sharedMemSize) noexcept {
  LOG_FUNC();
  abort();
}
