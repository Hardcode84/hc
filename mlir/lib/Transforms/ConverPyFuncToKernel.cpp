// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hc/Transforms/Passes.hpp"

#include "hc/Dialect/HKernel/IR/HKernelOps.hpp"
#include "hc/Dialect/PyIR/IR/PyIROps.hpp"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/PatternMatch.h>

namespace hc {
#define GEN_PASS_DEF_CONVERPYFUNCTOKERNELPASS
#include "hc/Transforms/Passes.h.inc"
} // namespace hc

namespace {
struct ConverPyFuncToKernelPass final
    : public hc::impl::ConverPyFuncToKernelPassBase<ConverPyFuncToKernelPass> {

  void runOnOperation() override {
    auto op = getOperation();

    mlir::IRRewriter builder(&getContext());
    builder.setInsertionPointToStart(op.getBody());

    for (auto pyModule :
         llvm::make_early_inc_range(op.getOps<hc::py_ir::PyModuleOp>())) {
      auto term = mlir::cast<hc::py_ir::PyModuleEndOp>(
          pyModule.getBody()->getTerminator());
      mlir::ValueRange termResults = term.getResults();
      if (termResults.size() != 1) {
        term->emitError("Unepected py module results count: ")
            << termResults.size();
        return signalPassFailure();
      }

      auto pyFunc = termResults.front().getDefiningOp<hc::py_ir::PyFuncOp>();
      if (!pyFunc) {
        term->emitError("Expected a py func, but got: ") << termResults.front();
        return signalPassFailure();
      }

      if (!pyFunc.getCaptureArgs().empty()) {
        pyFunc->emitError("Cannot convert function with captures");
        return signalPassFailure();
      }

      mlir::Region &srcRegion = pyFunc.getBodyRegion();
      for (mlir::Block &block : srcRegion) {
        auto term = mlir::dyn_cast<hc::py_ir::ReturnOp>(block.getTerminator());
        if (!term)
          continue;

        if (!mlir::isa<mlir::NoneType>(term.getOperand().getType())) {
          term->emitError("kernel musr return none");
          return signalPassFailure();
        }

        mlir::OpBuilder::InsertionGuard g(builder);
        builder.setInsertionPoint(term);
        builder.replaceOpWithNewOp<mlir::func::ReturnOp>(term);
      }

      mlir::TypeRange argTypes =
          pyFunc.getBodyRegion().front().getArgumentTypes();
      auto funcType = mlir::FunctionType::get(&getContext(), argTypes, {});
      auto loc = pyFunc.getLoc();
      auto newFunc =
          builder.create<mlir::func::FuncOp>(loc, pyFunc.getName(), funcType);

      mlir::Region &dstRegion = newFunc.getBody();
      builder.inlineRegionBefore(srcRegion, dstRegion, dstRegion.begin());
      builder.eraseOp(pyModule);
    }
  }
};
} // namespace
