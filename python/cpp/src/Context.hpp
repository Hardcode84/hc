// SPDX-FileCopyrightText: 2024 The HC Authors
// SPDX-FileCopyrightText: 2025 The HC Authors
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

  // TODO: We need lld for AMDGPU linking
  std::string llvmBinPath;
};

nanobind::capsule createContext(nanobind::dict settings);
