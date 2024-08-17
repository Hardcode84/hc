// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hc/Transforms/Passes.hpp"

#include "hc/Dialect/PyIR/IR/PyIROps.hpp"
#include "hc/Dialect/Typing/IR/TypingOps.hpp"

#include <queue>

#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/PatternMatch.h>

namespace hc {
#define GEN_PASS_DEF_RECONSTUCTPYSSAPASS
#include "hc/Transforms/Passes.h.inc"
} // namespace hc

namespace {
struct ReconstuctPySSA {
  struct BlockState {
    llvm::SmallDenseMap<mlir::StringAttr, mlir::Value> defs;
    bool filled = false;
  };

  void processBlock(mlir::Block *block) {
    auto &desc = blocksMap[block];
    if (desc.filled)
      return;

    if (auto func = mlir::dyn_cast_if_present<hc::py_ir::PyFuncOp>(
            block->getParentOp())) {
      if (block == func.getEntryBlock()) {
        for (auto &&[name, arg, blockArg] : func.getCaptureNamesAndArgs()) {
          (void)arg;
          desc.defs[name] = blockArg;
        }
      }
    }

    auto replaceOp = [&](auto op, mlir::Value val) {
      auto res = op.getResult();
      auto resType = res.getType();
      if (resType != val.getType()) {
        mlir::OpBuilder builder(op->getContext());
        builder.setInsertionPoint(op);
        val = builder.create<hc::typing::CastOp>(op.getLoc(), resType, val);
      }
      res.replaceAllUsesWith(val);
      op->erase();
    };

    for (auto &op : llvm::make_early_inc_range(block->without_terminator())) {
      if (auto store = mlir::dyn_cast<hc::py_ir::StoreVarOp>(op)) {
        desc.defs[store.getNameAttr()] = store.getValue();
      } else if (auto load = mlir::dyn_cast<hc::py_ir::LoadVarOp>(op)) {
        auto name = load.getNameAttr();
        auto it = desc.defs.find(name);
        if (it != desc.defs.end()) {
          replaceOp(load, it->second);
        } else if (block->hasNoPredecessors()) {
          desc.defs[name] = load.getResult();
        } else {
          mlir::Type type = load.getResult().getType();
          mlir::OpBuilder builder(load->getContext());
          mlir::Value arg = block->addArgument(type, builder.getUnknownLoc());
          desc.defs[name] = arg;
          replaceOp(load, arg);

          for (auto pred : block->getPredecessors())
            toProcess.emplace(name, pred, block);
        }
      }
    }

    desc.filled = true;
  }

  void processBlockRecursively(mlir::StringAttr name, mlir::Block *current,
                               mlir::Block *successor) {
    processBlock(current);
    mlir::Type type = successor->getArguments().back().getType();
    mlir::Value val;
    auto &desc = blocksMap[current];
    auto it = desc.defs.find(name);
    mlir::OpBuilder builder(name.getContext());
    mlir::Operation *term = current->getTerminator();
    builder.setInsertionPoint(term);

    if (it != desc.defs.end()) {
      val = it->second;
    } else if (current->hasNoPredecessors()) {
      val = builder.create<hc::py_ir::LoadVarOp>(term->getLoc(), type, name);
      desc.defs[name] = val;
    } else {
      val = current->addArgument(type, builder.getUnknownLoc());
      desc.defs[name] = val;

      for (auto pred : current->getPredecessors())
        toProcess.emplace(name, pred, current);
    }

    if (type != val.getType())
      val = builder.create<hc::typing::CastOp>(term->getLoc(), type, val);

    if (auto br = mlir::dyn_cast<mlir::cf::BranchOp>(term)) {
      assert(successor == br.getSuccessor());
      auto args = llvm::to_vector(br.getDestOperands());
      args.emplace_back(val);
      builder.create<mlir::cf::BranchOp>(br.getLoc(), successor, args);
      br->erase();
    } else if (auto condBr = mlir::dyn_cast<mlir::cf::CondBranchOp>(term)) {
      auto trueArgs = llvm::to_vector(condBr.getTrueDestOperands());
      auto falseArgs = llvm::to_vector(condBr.getFalseDestOperands());
      if (condBr.getTrueDest() == successor &&
          !(condBr.getFalseDest() == successor &&
            trueArgs.size() == successor->getNumArguments())) {
        trueArgs.emplace_back(val);
      } else {
        assert(condBr.getFalseDest() == successor);
        falseArgs.emplace_back(val);
      }
      builder.create<mlir::cf::CondBranchOp>(
          condBr.getLoc(), condBr.getCondition(), condBr.getTrueDest(),
          trueArgs, condBr.getFalseDest(), falseArgs);
      condBr->erase();
    } else {
      llvm::errs() << *term << "\n";
      llvm_unreachable("Unsupported terminator");
    }
  }

  void processQueuedBlocks() {
    while (!toProcess.empty()) {
      auto val = toProcess.front();
      toProcess.pop();
      auto &&[name, block, successor] = val;
      processBlockRecursively(name, block, successor);
    }
  }

  void processRegion(mlir::Region &region) {
    blocksMap.clear();
    for (mlir::Block &block : region) {
      processBlock(&block);
      processQueuedBlocks();
    }
  }

  void processOp(mlir::Operation *op) {
    for (mlir::Region &reg : op->getRegions())
      processRegion(reg);

    if (auto func = mlir::dyn_cast<hc::py_ir::PyFuncOp>(op)) {
      mlir::IRRewriter builder(op->getContext());
      builder.setInsertionPoint(func);
      mlir::Location loc = func.getLoc();

      llvm::SmallVector<mlir::StringRef> captureNames;
      llvm::SmallVector<mlir::Value> captureArgs;
      for (auto &&[name, arg, blockArg] : func.getCaptureNamesAndArgs()) {
        (void)blockArg;
        captureNames.emplace_back(name.getValue());
        captureArgs.emplace_back(arg);
      }

      mlir::Block &block = func.getBodyRegion().front();

      for (mlir::Operation &innerOp :
           llvm::make_early_inc_range(block.without_terminator())) {
        auto load = mlir::dyn_cast<hc::py_ir::LoadVarOp>(innerOp);
        if (!load)
          continue;

        llvm::StringRef name = load.getName();
        mlir::Value newArg =
            builder.create<hc::py_ir::LoadVarOp>(loc, load.getType(), name);

        captureNames.emplace_back(name);
        captureArgs.emplace_back(newArg);

        load.replaceAllUsesWith(
            block.addArgument(load.getType(), load.getLoc()));
        load->erase();
      }

      if (captureNames.size() != func.getCaptureNames().size()) {
        auto argNames = func.getArgsNamesArray();
        auto annotations = func.getAnnotations();
        auto decorators = func.getDecorators();

        auto newFunc = builder.create<hc::py_ir::PyFuncOp>(
            loc, func.getType(), func.getName(), argNames, annotations,
            captureNames, captureArgs, decorators);
        mlir::Region &reg = newFunc.getBodyRegion();
        reg.front().erase();

        builder.inlineRegionBefore(func.getBodyRegion(), reg, reg.begin());
        func.replaceAllUsesWith(newFunc.getResult());
        func->erase();
      }
    }
  }

  std::queue<std::tuple<mlir::StringAttr, mlir::Block *, mlir::Block *>>
      toProcess;
  llvm::SmallDenseMap<mlir::Block *, BlockState, 1> blocksMap;
};

struct ReconstuctPySSAPass final
    : public hc::impl::ReconstuctPySSAPassBase<ReconstuctPySSAPass> {

  void runOnOperation() override {
    ReconstuctPySSA state;
    getOperation()->walk<mlir::WalkOrder::PostOrder>(
        [&](mlir::Operation *op) { state.processOp(op); });
  }
};
} // namespace
