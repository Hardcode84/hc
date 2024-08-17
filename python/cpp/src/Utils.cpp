// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "Utils.hpp"

#include <stdexcept>

#include <llvm/ADT/Twine.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Pass/PassManager.h>

#include "hc/Utils.hpp"

#include "Context.hpp"

[[noreturn]] void reportError(const llvm::Twine &msg) {
  throw std::runtime_error(msg.str());
}

static void printDiag(llvm::raw_ostream &os, const mlir::Diagnostic &diag) {
  os << diag;
  for (auto &note : diag.getNotes())
    os << "\n" << note;

  os << "\n";
}

mlir::LogicalResult runUnderDiag(mlir::PassManager &pm,
                                 mlir::Operation *module) {
  bool dumpDiag = true;
  std::string err;
  llvm::raw_string_ostream errStream(err);
  auto diagHandler = [&](const mlir::Diagnostic &diag) {
    if (dumpDiag)
      printDiag(llvm::errs(), diag);

    if (diag.getSeverity() == mlir::DiagnosticSeverity::Error)
      printDiag(errStream, diag);
  };

  auto getErr = [&]() -> const std::string & {
    errStream << "\n";
    module->print(errStream);
    errStream.flush();
    return err;
  };

  bool verify = true;
  return hc::scopedDiagHandler(*module->getContext(), diagHandler, [&]() {
    if (verify && mlir::failed(mlir::verify(module))) {
      llvm::errs() << "MLIR broken module\n" << getErr();
      return mlir::failure();
    }

    if (mlir::failed(pm.run(module))) {
      llvm::errs() << "MLIR pipeline failed\n" << getErr();
      return mlir::failure();
    }

    return mlir::success();
  });
}

void runPipeline(Context &context, mlir::Operation *op,
                 llvm::function_ref<void(mlir::PassManager &)> populateFunc) {
  mlir::PassManager pm(&context.context);
  if (context.settings.dumpIR) {
    context.context.disableMultithreading();
    pm.enableIRPrinting();
  }
  populateFunc(pm);
  if (mlir::failed(runUnderDiag(pm, op)))
    reportError("pipeline failed");
}
