// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <llvm/ADT/SmallString.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Verifier.h>
#include <mlir/InitAllDialects.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/FileUtilities.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Support/ToolUtilities.h>
#include <mlir/Tools/mlir-opt/MlirOptMain.h>

#include "hc/Pipelines/FrontendPipeline.hpp"
#include "hc/PyFront/Import.hpp"
#include "hc/Utils.hpp"

enum class Cmd {
  Front,
  Full,
};

using ChunkBufferHandler = llvm::function_ref<mlir::LogicalResult(
    std::unique_ptr<llvm::MemoryBuffer> chunkBuffer, llvm::raw_ostream &os)>;

static void printDiag(llvm::raw_ostream &os, const mlir::Diagnostic &diag) {
  os << diag;
  for (auto &note : diag.getNotes())
    os << "\n" << note;

  os << "\n";
}

static mlir::LogicalResult runUnderDiag(mlir::PassManager &pm,
                                        mlir::ModuleOp module) {
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
    module.print(errStream);
    errStream.flush();
    return err;
  };

  bool verify = true;
  return hc::scopedDiagHandler(*module.getContext(), diagHandler, [&]() {
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

static mlir::LogicalResult pyfrontMain(llvm::StringRef inputFilename, Cmd cmd) {
  std::string errorMessage;
  auto file = mlir::openInputFile(inputFilename, &errorMessage);
  if (!file) {
    llvm::errs() << errorMessage << "\n";
    return mlir::failure();
  }

  mlir::MLIRContext ctx;
  auto loc = mlir::OpBuilder(&ctx).getUnknownLoc();

  auto processBuffer = [&](std::unique_ptr<llvm::MemoryBuffer> chunkBuffer,
                           llvm::raw_ostream &os) -> mlir::LogicalResult {
    auto mod = mlir::ModuleOp::create(loc);
    auto res =
        hc::importPyModule(chunkBuffer->getBuffer(), mod, /*dumpAST*/ false);
    if (mlir::succeeded(res)) {
      if (mlir::failed(mlir::verify(mod)))
        res = mlir::failure();

      mod.print(os);

      if (cmd == Cmd::Full) {
        mlir::PassManager pm(&ctx);

        ctx.disableMultithreading();
        pm.enableIRPrinting();

        hc::populateImportPipeline(pm);
        hc::populateFrontendPipeline(pm);
        if (mlir::failed(runUnderDiag(pm, mod)))
          return mlir::failure();
      }
    }

    mod->erase();
    return res;
  };

  return mlir::splitAndProcessBuffer(std::move(file), processBuffer,
                                     llvm::outs(), "# -----", "// -----");
}

int main(int argc, char **argv) {
  if (argc != 3) {
    llvm::errs() << argv[0] << " front|full <filename>";
    return EXIT_FAILURE;
  }

  llvm::StringRef cmd = argv[1];
  Cmd cmdId;
  if (cmd == "front") {
    cmdId = Cmd::Front;
  } else if (cmd == "full") {
    cmdId = Cmd::Full;
  } else {
    llvm::errs() << "Unknown command: " << cmd;
    return EXIT_FAILURE;
  }

  llvm::StringRef file = argv[2];

  return mlir::asMainReturnCode(pyfrontMain(file, cmdId));
}
