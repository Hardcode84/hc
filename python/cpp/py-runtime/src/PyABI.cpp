// SPDX-FileCopyrightText: 2025 The HC Authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PyABI.hpp"

#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace py = nanobind;

namespace {
struct MemrefDescriptor {
  void *allocated;
  void *aligned;
  intptr_t offset;
  intptr_t sizesAndStrides[1];
};
} // namespace

void convertPyArrayImpl(void *obj, int rank, void *ret) {
  auto array =
      py::cast<py::ndarray<>>(py::handle(static_cast<PyObject *>(obj)));
  if (array.ndim() != rank)
    throw std::runtime_error("Invalid rank");

  auto *desc = static_cast<MemrefDescriptor *>(ret);
  desc->allocated = array.data();
  desc->aligned = array.data();
  desc->offset = 0;
  for (int i = 0; i < rank; ++i) {
    desc->sizesAndStrides[i] = array.shape(i);
    desc->sizesAndStrides[i + rank] = array.stride(i);
  }
}

void convertPyInt64(void *obj, int64_t *ret) {
  *ret = py::cast<int64_t>(py::handle(static_cast<PyObject *>(obj)));
}

extern "C" void setGPULoaderSearchPaths(const char *paths[], size_t count);

void setRuntimeSearchPathsImpl(void *obj) {
  auto paths = py::cast<py::list>(py::handle(static_cast<PyObject *>(obj)));
  std::vector<const char *> res;
  for (auto path : paths) {
    auto tmp = py::cast<py::bytes>(path);
    res.emplace_back(tmp.c_str());
  }

  setGPULoaderSearchPaths(res.data(), res.size());
}
