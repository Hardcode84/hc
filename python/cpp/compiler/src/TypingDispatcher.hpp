// SPDX-FileCopyrightText: 2024 The HC Authors
// SPDX-FileCopyrightText: 2025 The HC Authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "DispatcherBase.hpp"

class TypingDispatcher : public DispatcherBase {
public:
  static void definePyClass(nanobind::module_ &m);

  using DispatcherBase::DispatcherBase;

  nanobind::object compile();

protected:
  virtual void populateFrontendPipeline(mlir::PassManager &pm) override;
  virtual void populateImportPipeline(mlir::PassManager &pm) override;
  virtual void populateInvokePipeline(mlir::PassManager &pm) override;
};
