// SPDX-FileCopyrightText: 2024 The HC Authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "Context.hpp"

#include "hc/Dialect/PyIR/IR/PyIROps.hpp"
#include "hc/Dialect/Typing/IR/TypingOps.hpp"

#include "PyWrappers.hpp"

#include <llvm/Support/Debug.h>
#include <llvm/Support/StringSaver.h>

namespace py = pybind11;

Context::Context() {
  context.loadDialect<hc::py_ir::PyIRDialect, hc::typing::TypingDialect>();
  pushContext(&context);
}

Context::~Context() { popContext(&context); }

static void readSettings(Settings &ret, py::dict &settings) {
  ret.dumpAST = settings["DUMP_AST"].cast<bool>();
  ret.dumpIR = settings["DUMP_IR"].cast<bool>();
}

static void readDebugTypes(py::dict &settings) {
  auto debugType = settings["DEBUG_TYPE"].cast<py::list>();
  auto debugTypeSize = debugType.size();
  if (debugTypeSize != 0) {
    llvm::DebugFlag = true;
    llvm::BumpPtrAllocator alloc;
    auto types = alloc.Allocate<const char *>(debugTypeSize);
    llvm::StringSaver strSaver(alloc);
    for (auto i : llvm::seq<size_t>(0, debugTypeSize))
      types[i] = strSaver.save(debugType[i].cast<std::string>()).data();

    llvm::setCurrentDebugTypes(types, static_cast<unsigned>(debugTypeSize));
  }
}

py::capsule createContext(py::dict settings) {
  auto ctx = std::make_unique<Context>();
  readSettings(ctx->settings, settings);
  readDebugTypes(settings);
  auto dtor = [](void *ptr) { delete static_cast<Context *>(ptr); };
  pybind11::capsule ret(ctx.get(), dtor);
  ctx.release();
  return ret;
}
