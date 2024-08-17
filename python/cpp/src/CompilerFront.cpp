// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <llvm/ADT/SmallString.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Verifier.h>
#include <mlir/InitAllDialects.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/FileUtilities.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Tools/mlir-opt/MlirOptMain.h>

#include "hc/Dialect/PyAST/IR/PyASTOps.hpp"
#include "hc/Dialect/PyIR/IR/PyIROps.hpp"
#include "hc/PyFront/Import.hpp"

#include "CompilerFront.hpp"
#include "Context.hpp"
#include "Utils.hpp"

static void createModuleImport(mlir::OpBuilder &builder,
                               const ImportedSym &sym) {
  assert(!sym.modulePath.empty());
  mlir::Value mod;

  mlir::Location loc = builder.getUnknownLoc();
  auto type = hc::py_ir::UndefinedType::get(builder.getContext());
  for (auto &&path : sym.modulePath) {
    if (!mod) {
      mod = builder.create<hc::py_ir::LoadModuleOp>(loc, type, path);
      continue;
    }

    mod = builder.create<hc::py_ir::GetAttrOp>(loc, type, mod, path);
  }
  assert(mod);
  builder.create<hc::py_ir::StoreVarOp>(loc, sym.name, mod);
}

template <typename T> static std::string toStr(T &&val) {
  std::string ret;
  llvm::raw_string_ostream os(ret);
  os << val;
  os.flush();
  return ret;
}

static void createLiteral(mlir::OpBuilder &builder, const Literal &lit) {
  auto attr = mlir::cast<mlir::TypedAttr>(lit.attr);
  auto loc = builder.getUnknownLoc();
  auto op =
      attr.getDialect().materializeConstant(builder, attr, attr.getType(), loc);

  if (!op) {
    auto arith =
        attr.getContext()->getLoadedDialect<mlir::arith::ArithDialect>();
    op = arith->materializeConstant(builder, attr, attr.getType(), loc);
  }

  if (!op)
    reportError(llvm::Twine("Cannot materialize literal: ") + toStr(lit.attr));

  builder.create<hc::py_ir::StoreVarOp>(loc, lit.name, op->getResult(0));
}

static mlir::LogicalResult
importAST(mlir::Operation *mod, llvm::StringRef source,
          llvm::StringRef funcName, llvm::ArrayRef<ImportedSym> importedSymbols,
          llvm::ArrayRef<Literal> literals, bool dumpAST) {
  auto res = hc::importPyModule(source, mod, dumpAST);
  if (mlir::failed(res))
    return mlir::failure();

  auto pyMod = mlir::cast<hc::py_ast::PyModuleOp>(*res);

  mlir::OpBuilder builder(mod->getContext());
  mlir::Block *body = pyMod.getBody();

  builder.setInsertionPointToStart(body);
  for (auto &&sym : importedSymbols)
    createModuleImport(builder, sym);

  for (auto &&lit : literals)
    createLiteral(builder, lit);

  auto term = body->getTerminator();
  builder.setInsertionPoint(term);
  builder.create<hc::py_ast::CaptureValOp>(term->getLoc(), funcName);
  return mlir::success();
}

mlir::FailureOr<mlir::OwningOpRef<mlir::Operation *>>
compileAST(Context &ctx, llvm::StringRef source, llvm::StringRef funcName,
           llvm::ArrayRef<ImportedSym> importedSymbols,
           llvm::ArrayRef<Literal> literals) {
  auto *mlirContext = &ctx.context;
  auto loc = mlir::OpBuilder(mlirContext).getUnknownLoc();

  mlir::OwningOpRef<mlir::Operation *> mod(mlir::ModuleOp::create(loc));

  auto &settings = ctx.settings;
  if (mlir::failed(importAST(*mod, source, funcName, importedSymbols, literals,
                             settings.dumpAST)))
    return mlir::failure();

  return mod;
}
