// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <pybind11/pybind11.h>

#include <mlir/IR/MLIRContext.h>

struct Settings {
  bool dumpAST = false;
  bool dumpIR = false;
};

struct Context {
  Context();
  ~Context();

  mlir::MLIRContext context;
  Settings settings;
};

pybind11::capsule createContext(pybind11::dict settings);
