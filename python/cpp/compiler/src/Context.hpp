// SPDX-FileCopyrightText: 2024 The HC Authors
// SPDX-FileCopyrightText: 2025 The HC Authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "hc/ExecutionEngine/ExecutionEngine.hpp"

#include <nanobind/nanobind.h>

#include <mlir/IR/MLIRContext.h>

struct Settings {
  bool dumpAST = false;
  bool dumpIR = false;
  bool dumpLLVM = false;
  bool dumpOptLLVM = false;
  bool dumpASM = false;
};

struct Context {
  Context(nanobind::dict settings_);
  ~Context();

  mlir::MLIRContext context;
  Settings settings;
  hc::ExecutionEngine executionEngine;
};

nanobind::capsule createContext(nanobind::dict settings);

nanobind::bool_ enableDumpAST(nanobind::capsule context,
                              nanobind::bool_ enable);
nanobind::bool_ enableDumpIR(nanobind::capsule context, nanobind::bool_ enable);

nanobind::bool_ enableDumpLLVM(nanobind::capsule context,
                               nanobind::bool_ enable);
nanobind::bool_ enableDumpOptLLVM(nanobind::capsule context,
                                  nanobind::bool_ enable);
nanobind::bool_ enableDumpASM(nanobind::capsule context,
                              nanobind::bool_ enable);
