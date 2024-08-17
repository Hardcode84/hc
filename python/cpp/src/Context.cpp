// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "Context.hpp"

#include "hc/Dialect/PyIR/IR/PyIROps.hpp"
#include "hc/Dialect/Typing/IR/TypingOps.hpp"

#include "PyWrappers.hpp"

namespace py = pybind11;

Context::Context() {
  context.loadDialect<hc::py_ir::PyIRDialect, hc::typing::TypingDialect>();
  pushContext(&context);
}

Context::~Context() { popContext(&context); }

static void readSettings(Settings &ret, py::dict &dict) {
  ret.dumpAST = dict["DUMP_AST"].cast<bool>();
  ret.dumpIR = dict["DUMP_IR"].cast<bool>();
}

py::capsule createContext(py::dict settings) {
  auto ctx = std::make_unique<Context>();
  readSettings(ctx->settings, settings);
  auto dtor = [](void *ptr) { delete static_cast<Context *>(ptr); };
  pybind11::capsule ret(ctx.get(), dtor);
  ctx.release();
  return ret;
}
