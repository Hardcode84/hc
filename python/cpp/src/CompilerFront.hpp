// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <string>

#include <llvm/ADT/SmallVector.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/Support/LogicalResult.h>

namespace llvm {
class StringRef;
}

struct Context;

struct ImportedSym {
  std::string name;
  llvm::SmallVector<std::string> modulePath;
};

struct Literal {
  std::string name;
  mlir::Attribute attr;
};

mlir::FailureOr<mlir::OwningOpRef<mlir::Operation *>>
compileAST(Context &ctx, llvm::StringRef source, llvm::StringRef funcName,
           llvm::ArrayRef<ImportedSym> importedSymbols,
           llvm::ArrayRef<Literal> literals);
