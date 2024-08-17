// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hc/Transforms/Passes.hpp"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/PatternMatch.h>

#include "hc/Dialect/PyIR/IR/PyIROps.hpp"

namespace hc {
#define GEN_PASS_DEF_CONVERPYFUNCTOFUNCPASS
#include "hc/Transforms/Passes.h.inc"
} // namespace hc

namespace {
struct ConverPyFuncToFuncPass final
    : public hc::impl::ConverPyFuncToFuncPassBase<ConverPyFuncToFuncPass> {

  void runOnOperation() override {
    auto root = getOperation();

    mlir::IRRewriter builder(&getContext());

    auto visitor = [&](mlir::Operation *op) -> mlir::WalkResult {
      if (mlir::isa<hc::py_ir::SymbolConstantOp>(op)) {
        op->emitOpError("This op is not supported");
        return mlir::WalkResult::interrupt();
      }

      if (auto func = mlir::dyn_cast<hc::py_ir::PyStaticFuncOp>(op)) {
        auto mod = op->getParentOfType<hc::py_ir::PyModuleOp>();
        if (!mod) {
          func->emitOpError("No PyIR module");
          return mlir::WalkResult::interrupt();
        }

        builder.setInsertionPointToStart(mod.getBody());
        auto symName = func.getSymName();
        auto funcType = func.getFunctionType();

        bool retNone = false;
        if (mlir::isa<mlir::NoneType>(funcType.getResult(0))) {
          retNone = true;
          funcType = funcType.clone(funcType.getInputs(), {});
        }

        auto visibility = func.getSymVisibilityAttr();
        auto loc = op->getLoc();

        auto newFunc = builder.create<mlir::func::FuncOp>(
            loc, symName, funcType, visibility, nullptr, nullptr);
        mlir::Region &newReg = newFunc.getFunctionBody();
        builder.inlineRegionBefore(func.getBodyRegion(), newReg,
                                   newReg.begin());
        builder.eraseOp(func);

        for (mlir::Block &block : newFunc.getFunctionBody()) {
          auto term =
              mlir::dyn_cast<hc::py_ir::ReturnOp>(block.getTerminator());
          if (!term)
            continue;

          builder.setInsertionPoint(term);
          if (retNone) {
            builder.replaceOpWithNewOp<mlir::func::ReturnOp>(term);
          } else {
            builder.replaceOpWithNewOp<mlir::func::ReturnOp>(term,
                                                             term.getOperand());
          }
        }
        return mlir::WalkResult::advance();
      }

      if (auto call = mlir::dyn_cast<hc::py_ir::StaticCallOp>(op)) {
        llvm::SmallVector<mlir::Type, 1> resTypes(call->getResultTypes());

        auto callee = call.getCalleeAttr();
        auto func = mlir::SymbolTable::lookupNearestSymbolFrom<
            mlir::FunctionOpInterface>(op, callee);
        if (!func) {
          op->emitOpError("Uresolved symbol");
          return mlir::WalkResult::interrupt();
        }

        auto funcType =
            mlir::dyn_cast<mlir::FunctionType>(func.getFunctionType());
        if (!funcType || call.getOperandTypes() != funcType.getInputs() ||
            !(llvm::equal(funcType.getResults(), resTypes) ||
              (funcType.getResults().empty() &&
               mlir::isa<mlir::NoneType>(resTypes.back())))) {
          op->emitOpError("Ivalid symbol type, expected ") << funcType;
          return mlir::WalkResult::interrupt();
        }

        assert(resTypes.size() == 1);
        builder.setInsertionPoint(call);
        if (mlir::isa<mlir::NoneType>(resTypes.back())) {
          resTypes.clear();
          builder.create<mlir::func::CallOp>(op->getLoc(), callee, resTypes,
                                             call.getArgs());
          if (!op->use_empty()) {
            builder.replaceOpWithNewOp<hc::py_ir::NoneOp>(op);
          } else {
            builder.eraseOp(op);
          }
        } else {
          builder.replaceOpWithNewOp<mlir::func::CallOp>(op, callee, resTypes,
                                                         call.getArgs());
        }

        return mlir::WalkResult::advance();
      }
      return mlir::WalkResult::advance();
    };

    if (root->walk<mlir::WalkOrder::PostOrder>(visitor).wasInterrupted())
      return signalPassFailure();
  }
};
} // namespace
