// SPDX-FileCopyrightText: 2025 The HC Authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PyABI.hpp"

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
