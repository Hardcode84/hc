// SPDX-FileCopyrightText: 2024 The HC Authors
// SPDX-FileCopyrightText: 2025 The HC Authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "Dispatcher.hpp"

#include <llvm/ADT/Twine.h>
#include <mlir/IR/Operation.h>

#include "Context.hpp"
#include "Utils.hpp"

#include "hc/Pipelines/BackendPipeline.hpp"
#include "hc/Pipelines/FrontendPipeline.hpp"
#include "hc/Pipelines/MiddleendPipeline.hpp"

namespace py = nanobind;

void Dispatcher::definePyClass(py::module_ &m) {
  py::class_<Dispatcher, DispatcherBase>(m, "Dispatcher")
      .def(py::init<py::capsule, py::object>())
      .def("__call__", &Dispatcher::call);
}

void Dispatcher::call(py::args args, py::kwargs kwargs) {
  runFrontend()->dump();
  invokeFunc(args, kwargs);
}

void Dispatcher::populateImportPipeline(mlir::PassManager &pm) {
  hc::populateImportPipeline(pm);
}

void Dispatcher::populateFrontendPipeline(mlir::PassManager &pm) {
  hc::populateFrontendPipeline(pm);
}

void Dispatcher::populateInvokePipeline(mlir::PassManager &pm) {
  hc::populateMiddleendPipeline(pm);
  hc::populateBackendPipeline(pm, context.llvmBinPath);
}
