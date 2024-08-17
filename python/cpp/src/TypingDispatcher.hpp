// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "DispatcherBase.hpp"

class TypingDispatcher : public DispatcherBase {
public:
  static void definePyClass(pybind11::module_ &m);

  using DispatcherBase::DispatcherBase;

  pybind11::object compile();

protected:
  virtual void populateFrontendPipeline(mlir::PassManager &pm) override;
  virtual void populateImportPipeline(mlir::PassManager &pm) override;
  virtual void populateInvokePipeline(mlir::PassManager &pm) override;
};
