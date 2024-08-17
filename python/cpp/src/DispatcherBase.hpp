// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Types.h>

#include <pybind11/pybind11.h>

namespace mlir {
class Operation;
class PassManager;
} // namespace mlir

struct Context;

class DispatcherBase {
public:
  DispatcherBase(pybind11::capsule ctx, pybind11::object getDesc);
  virtual ~DispatcherBase();

  static void definePyClass(pybind11::module_ &m);

protected:
  virtual void populateImportPipeline(mlir::PassManager &pm) = 0;
  virtual void populateFrontendPipeline(mlir::PassManager &pm) = 0;
  virtual void populateInvokePipeline(mlir::PassManager &pm) = 0;

  mlir::Operation *runFrontend();
  void invokeFunc(const pybind11::args &args, const pybind11::kwargs &kwargs);

private:
  using OpRef = mlir::OwningOpRef<mlir::Operation *>;

  Context &context;
  pybind11::object contextRef; // to keep context alive
  pybind11::object getFuncDesc;
  OpRef mod;

  struct ArgDesc {
    llvm::StringRef name;
    std::function<void(mlir::MLIRContext &, pybind11::handle,
                       llvm::SmallVectorImpl<mlir::Type> &,
                       llvm::SmallVectorImpl<PyObject *> &)>
        handler;
  };
  llvm::SmallVector<ArgDesc> argsHandlers;

  struct ExceptionDesc {
    std::string message;
  };

  using FuncT = int (*)(ExceptionDesc *exc, PyObject *args[]);

  llvm::DenseMap<mlir::Type, FuncT> funcsCache;

  void populateArgsHandlers(pybind11::handle args);
  mlir::Type processArgs(const pybind11::args &args,
                         const pybind11::kwargs &kwargs,
                         llvm::SmallVectorImpl<PyObject *> &retArgs) const;

  void linkModules(mlir::Operation *rootModule,
                   const pybind11::dict &currentDeps);
  OpRef importFuncForLinking(
      llvm::SmallVectorImpl<std::pair<DispatcherBase *, mlir::Operation *>>
          &unresolved);
};
