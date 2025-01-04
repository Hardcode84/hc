// SPDX-FileCopyrightText: 2024 The HC Authors
// SPDX-FileCopyrightText: 2025 The HC Authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <nanobind/nanobind.h>

#include "Context.hpp"
#include "Dispatcher.hpp"
#include "PyWrappers.hpp"
#include "TypingDispatcher.hpp"

namespace py = nanobind;

NB_MODULE(compiler, m) {
  m.def("create_context", &createContext);

  DispatcherBase::definePyClass(m);
  Dispatcher::definePyClass(m);

  auto mlirMod = m.def_submodule("_mlir");
  populateMlirModule(mlirMod);

  auto typingMod = m.def_submodule("_typing");
  TypingDispatcher::definePyClass(typingMod);
}
