// SPDX-FileCopyrightText: 2025 The HC Authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hc/Transforms/Passes.hpp"

#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Transforms/DialectConversion.h>

namespace hc {
#define GEN_PASS_DEF_LEGALIZELLVMABIPASS
#include "hc/Transforms/Passes.h.inc"
} // namespace hc

namespace {
struct LegalizeLLVMABIPass final
    : public hc::impl::LegalizeLLVMABIPassBase<LegalizeLLVMABIPass> {

  void runOnOperation() override {
    auto mod = getOperation();

    bool changed = false;
    llvm::SmallVector<mlir::Type> newArgTypes;
    mlir::OpBuilder builder(&getContext());
    mlir::Type ptrType = mlir::LLVM::LLVMPointerType::get(builder.getContext());
    auto visitor = [&](mlir::LLVM::LLVMFuncOp func) -> mlir::WalkResult {
      bool needLegalize = true;
      newArgTypes.clear();
      for (auto arg : func.getArgumentTypes()) {
        if (!mlir::isa<mlir::LLVM::LLVMStructType>(arg)) {
          newArgTypes.emplace_back(arg);
          continue;
        }

        needLegalize = true;
        newArgTypes.emplace_back(ptrType);
      }

      if (!needLegalize)
        return mlir::WalkResult::advance();

      changed = true;

      auto users = func.getSymbolUses(mod);
      if (!users) {
        func->emitError("Failed to get func users");
        return mlir::WalkResult::interrupt();
      }

      for (auto user : *users) {
        auto call = mlir::dyn_cast<mlir::LLVM::CallOp>(user.getUser());
        if (!call) {
          call->emitError("Unsupported user");
          return mlir::WalkResult::interrupt();
        }

        auto userFunc = call->getParentOfType<mlir::LLVM::LLVMFuncOp>();
        if (!userFunc) {
          call->emitError("No parent func");
          return mlir::WalkResult::interrupt();
        }

        mlir::Location loc = call.getLoc();
        mlir::Block *block = &userFunc.getBody().front();
        builder.setInsertionPointToStart(block);
        mlir::Value one = builder.create<mlir::LLVM::ConstantOp>(
            loc, builder.getI64Type(), 1);
        for (auto &&[i, origType, newType] :
             llvm::enumerate(call.getArgOperands().getTypes(), newArgTypes)) {
          if (origType == newType)
            continue;

          mlir::Value ptr =
              builder.create<mlir::LLVM::AllocaOp>(loc, ptrType, origType, one);

          mlir::OpBuilder::InsertionGuard g(builder);
          builder.setInsertionPoint(call);
          builder.create<mlir::LLVM::StoreOp>(loc, call.getArgOperands()[i],
                                              ptr);
          call->setOperand(i, ptr);
        }
      }

      auto newFuncType = mlir::LLVM::LLVMFunctionType::get(
          func.getFunctionType().getReturnType(), newArgTypes);
      func.setFunctionType(newFuncType);

      if (!func.getBody().empty()) {
        mlir::Block *block = &func.getBody().front();
        builder.setInsertionPointToStart(block);

        for (auto &&[i, origType, newType] :
             llvm::enumerate(block->getArgumentTypes(), newArgTypes)) {
          if (origType == newType)
            continue;

          mlir::Value arg = block->getArgument(i);
          arg.setType(newType);
          mlir::Location loc = arg.getLoc();
          auto newArg = builder.create<mlir::LLVM::LoadOp>(loc, origType, arg);
          arg.replaceAllUsesExcept(newArg.getResult(), newArg);
        }
      }

      return mlir::WalkResult::advance();
    };

    if (mod->walk(visitor).wasInterrupted())
      return signalPassFailure();

    if (!changed)
      markAllAnalysesPreserved();
  }
};
} // namespace
