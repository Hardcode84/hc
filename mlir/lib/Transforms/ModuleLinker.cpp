// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hc/Transforms/ModuleLinker.hpp"

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/IRMapping.h>

static mlir::LogicalResult checkModulesCompatible(mlir::Operation *dest,
                                                  mlir::Operation *toLink) {
  if (dest->getName() != toLink->getName())
    return dest->emitOpError("Incompatible op types");

  if (dest->getNumRegions() != 1 || !llvm::hasSingleElement(dest->getRegion(0)))
    return dest->emitOpError("Expected 1 region with 1 block");

  if (toLink->getNumRegions() != 1 ||
      !llvm::hasSingleElement(toLink->getRegion(0)))
    return toLink->emitOpError("Expected 1 region with 1 block");

  return mlir::success();
}

static mlir::LogicalResult mergeSymbols(mlir::Operation *dest,
                                        mlir::Operation *toLink) {
  mlir::SymbolTable symTable(dest);

  mlir::Block &toLinkBlock = toLink->getRegion(0).front();
  for (auto symbol : llvm::make_early_inc_range(
           toLinkBlock.getOps<mlir::SymbolOpInterface>())) {
    auto destSymbol = mlir::cast_if_present<mlir::SymbolOpInterface>(
        symTable.lookup(symbol.getNameAttr()));
    if (!destSymbol)
      continue;

    if (symbol->getName() != destSymbol->getName())
      return symbol->emitOpError("Incompatible symbol ops");

    if (symbol.getVisibility() != destSymbol.getVisibility())
      return symbol->emitOpError("Incompatible symbol visibility");

    if (symbol->getAttrDictionary() != destSymbol->getAttrDictionary())
      return symbol->emitOpError("Incompatible symbol attrs");

    if (symbol.isDeclaration()) {
      symbol->erase();
      continue;
    }

    if (destSymbol.isDeclaration()) {
      symbol->moveBefore(destSymbol);
      destSymbol->erase();
      continue;
    }

    if (!mlir::OperationEquivalence::isEquivalentTo(
            symbol, destSymbol,
            mlir::OperationEquivalence::Flags::IgnoreLocations))
      return symbol->emitOpError("Symbols are not equivalent");

    // Symbols are equivalent, drop the second one.
    symbol->erase();
  }

  return mlir::success();
}

mlir::LogicalResult hc::linkModules(mlir::Operation *dest,
                                    mlir::Operation *toLink) {
  if (mlir::failed(checkModulesCompatible(dest, toLink)))
    return mlir::failure();

  if (dest->hasTrait<mlir::OpTrait::SymbolTable>()) {
    if (mlir::failed(mergeSymbols(dest, toLink)))
      return mlir::failure();
  }

  mlir::Block &dstBlock = dest->getRegion(0).front();
  if (dstBlock.mightHaveTerminator()) {
    mlir::Operation *term = dstBlock.getTerminator();
    if (term->getNumOperands() != 0)
      return dest->emitError("Non-trivial terminators are not supported");

    term->erase();
  }

  mlir::Block &toLinkBlock = toLink->getRegion(0).front();
  if (toLinkBlock.mightHaveTerminator()) {
    mlir::Operation *term = toLinkBlock.getTerminator();
    if (term->getNumOperands() != 0)
      return dest->emitError("Non-trivial terminators are not supported");
  }

  auto builder = mlir::OpBuilder::atBlockEnd(&dstBlock);
  mlir::IRMapping mapper;
  for (mlir::Operation &op : toLinkBlock.getOperations())
    builder.clone(op, mapper);

  return mlir::success();
}
