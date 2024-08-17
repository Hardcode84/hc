// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "Context.hpp"
#include "Dispatcher.hpp"
#include "PyWrappers.hpp"
#include "TypingDispatcher.hpp"

namespace py = pybind11;

PYBIND11_MODULE(compiler, m) {
  m.def("create_context", &createContext);

  DispatcherBase::definePyClass(m);
  Dispatcher::definePyClass(m);

  auto mlirMod = m.def_submodule("_mlir");
  populateMlirModule(mlirMod);

  auto typingMod = m.def_submodule("_typing");
  TypingDispatcher::definePyClass(typingMod);
}
