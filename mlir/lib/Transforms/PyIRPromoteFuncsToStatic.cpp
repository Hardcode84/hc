// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hc/Transforms/Passes.hpp"

#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include "hc/Dialect/PyIR/IR/PyIROps.hpp"
#include "hc/Dialect/Typing/IR/TypingOps.hpp"

namespace hc {
#define GEN_PASS_DEF_PYIRPROMOTEFUNCSTOSTATICPASS
#include "hc/Transforms/Passes.h.inc"
} // namespace hc

static llvm::SmallString<32> getUniqueSymbolName(mlir::Operation *symTableOp,
                                                 mlir::StringRef origName) {
  assert(!origName.empty());
  mlir::SymbolTable symTable(symTableOp);

  llvm::SmallString<32> str;
  for (unsigned i = 0;; ++i) {
    if (i == 0) {
      str = origName;
    } else {
      str.clear();
      (llvm::Twine(origName) + "_" + llvm::Twine(i)).toVector(str);
    }
    if (!symTable.lookup(str))
      return str;
  }
}

namespace {
class PromotePyFunc final : public mlir::OpRewritePattern<hc::py_ir::PyFuncOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(hc::py_ir::PyFuncOp op,
                  mlir::PatternRewriter &rewriter) const override {
    if (!op.getCaptureArgs().empty() || !op.getDecorators().empty())
      return mlir::failure();

    auto pyMod =
        mlir::dyn_cast_if_present<hc::py_ir::PyModuleOp>(op->getParentOp());
    if (!pyMod)
      return mlir::failure();

    mlir::Type resType;
    for (mlir::Block &block : op.getBodyRegion()) {
      auto term = mlir::dyn_cast<hc::py_ir::ReturnOp>(block.getTerminator());
      if (!term)
        continue;

      mlir::Type type = term.getOperand().getType();
      if (!resType) {
        resType = type;
      }
      if (resType != type) {
        return mlir::failure();
      }
    }

    if (!resType)
      resType = rewriter.getNoneType();

    auto funcType =
        rewriter.getFunctionType(op.getBlockArgs().getTypes(), resType);
    auto symName = getUniqueSymbolName(pyMod, op.getName());

    mlir::Location loc = op.getLoc();

    auto newFunc = rewriter.create<hc::py_ir::PyStaticFuncOp>(
        loc, symName, funcType, op.getArgsNamesArray(), op.getAnnotations());
    newFunc.setPrivate();
    rewriter.eraseBlock(newFunc.getEntryBlock());
    mlir::Region &newRegion = newFunc.getBodyRegion();
    rewriter.inlineRegionBefore(op.getBodyRegion(), newRegion,
                                newRegion.begin());

    auto sym = mlir::FlatSymbolRefAttr::get(rewriter.getContext(), symName);
    rewriter.replaceOpWithNewOp<hc::py_ir::SymbolConstantOp>(op, op.getType(),
                                                             sym);
    return mlir::success();
  }
};

struct PyIRPromoteFuncsToStaticPass final
    : public hc::impl::PyIRPromoteFuncsToStaticPassBase<
          PyIRPromoteFuncsToStaticPass> {

  void runOnOperation() override {
    auto *ctx = &getContext();
    mlir::RewritePatternSet patterns(ctx);
    hc::populatePyIRPromoteFuncsToStaticPatterns(patterns);
    hc::py_ir::PyFuncOp::getCanonicalizationPatterns(patterns, ctx);
    hc::py_ir::CallOp::getCanonicalizationPatterns(patterns, ctx);

    if (mlir::failed(
            applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
  }
};
} // namespace

void hc::populatePyIRPromoteFuncsToStaticPatterns(
    mlir::RewritePatternSet &patterns) {
  patterns.insert<PromotePyFunc>(patterns.getContext());
}
