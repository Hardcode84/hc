// SPDX-FileCopyrightText: 2025 The HC Authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hc/Transforms/Passes.hpp"

#include "hc/Dialect/HKernel/IR/HKernelOps.hpp"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/UB/IR/UBOps.h>

namespace hc {
#define GEN_PASS_DEF_CREATEPYWRAPPERPASS
#include "hc/Transforms/Passes.h.inc"
} // namespace hc

namespace {
struct CreatePyWrapperPass final
    : public hc::impl::CreatePyWrapperPassBase<CreatePyWrapperPass> {

  void runOnOperation() override {
    mlir::MLIRContext *ctx = &getContext();
    auto mod = getOperation();

    mlir::OpBuilder builder(mod.getBodyRegion());
    auto entrypointAttr =
        builder.getStringAttr(hc::hk::getKernelEntryPointAttrName());
    mlir::Location loc = builder.getUnknownLoc();

    mlir::Type i32 = builder.getI32Type();
    auto errType = hc::hk::ErrorContextType::get(ctx);
    auto argsType = hc::hk::PyArgsType::get(ctx);
    auto funcType = mlir::FunctionType::get(ctx, {errType, argsType}, i32);

    auto visitor = [&](mlir::func::FuncOp func) {
      if (!func->hasAttr(entrypointAttr))
        return;

      auto newName = (func.getName() + "_pyabi").str();
      auto wrapper = builder.create<mlir::func::FuncOp>(loc, newName, funcType);
      wrapper.setPublic();

      mlir::Block *block = builder.createBlock(
          &wrapper.getBody(), {}, funcType.getInputs(), {loc, loc});
      mlir::OpBuilder::InsertionGuard g(builder);
      builder.setInsertionPointToStart(block);
      mlir::Value errContext = block->getArgument(0);
      mlir::Value pyArgs = block->getArgument(1);

      llvm::SmallVector<mlir::Value> args;
      for (auto &&[i, argType] :
           llvm::enumerate(func.getFunctionType().getInputs())) {
        if (mlir::isa<hc::hk::CurrentGroupType>(argType)) {
          args.emplace_back(builder.create<mlir::ub::PoisonOp>(loc, argType));
          continue;
        }

        args.emplace_back(builder.create<hc::hk::GetPyArgOp>(
            loc, argType, pyArgs, i, errContext));
      }

      builder.create<mlir::func::CallOp>(loc, func, args);
      mlir::Value zero =
          builder.create<mlir::arith::ConstantIntOp>(loc, 0, i32);
      builder.create<mlir::func::ReturnOp>(loc, zero);
    };

    mod->walk(visitor);
  }
};
} // namespace
