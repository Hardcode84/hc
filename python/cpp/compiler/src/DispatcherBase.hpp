// SPDX-FileCopyrightText: 2024 The HC Authors
// SPDX-FileCopyrightText: 2025 The HC Authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Types.h>

#include <nanobind/nanobind.h>

namespace mlir {
class Operation;
class PassManager;
} // namespace mlir

struct Context;

namespace hc {
struct ExceptionDesc;
}

class DispatcherBase {
public:
  DispatcherBase(nanobind::capsule ctx, nanobind::object getDesc);
  virtual ~DispatcherBase();

  static void definePyClass(nanobind::module_ &m);

protected:
  virtual void populateImportPipeline(mlir::PassManager &pm) = 0;
  virtual void populateFrontendPipeline(mlir::PassManager &pm) = 0;
  virtual void populateInvokePipeline(mlir::PassManager &pm) = 0;

  mlir::Operation *runFrontend();
  void invokeFunc(const nanobind::args &args, const nanobind::kwargs &kwargs);

  Context &context;

private:
  using OpRef = mlir::OwningOpRef<mlir::Operation *>;
  struct ArgsHandlerBuilder;

  nanobind::object contextRef; // to keep context alive
  nanobind::object getFuncDesc;
  OpRef mod;
  std::unique_ptr<ArgsHandlerBuilder> argsHandlerBuilder;

  struct ArgDesc {
    llvm::StringRef name;
    std::function<void(mlir::MLIRContext &, nanobind::handle,
                       llvm::SmallMapVector<mlir::Type, mlir::Type, 8> &,
                       llvm::SmallVectorImpl<PyObject *> &)>
        handler;
  };
  llvm::SmallVector<ArgDesc> argsHandlers;

  using FuncT = int (*)(hc::ExceptionDesc *exc, PyObject *args[]);

  std::string funcName;
  llvm::SmallDenseMap<const void *, FuncT> funcsCache;
  llvm::SmallVector<void *> compilerModules;

  void populateArgsHandlers(nanobind::handle args);
  mlir::Attribute processArgs(const nanobind::args &args,
                              const nanobind::kwargs &kwargs,
                              llvm::SmallVectorImpl<PyObject *> &retArgs) const;

  void linkModules(mlir::Operation *rootModule,
                   const nanobind::dict &currentDeps);
  OpRef importFuncForLinking(
      llvm::SmallVectorImpl<std::pair<DispatcherBase *, mlir::Operation *>>
          &unresolved);
};
