// SPDX-FileCopyrightText: 2024 The HC Authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <nanobind/nanobind.h>

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

nanobind::capsule createContext(nanobind::dict settings);
