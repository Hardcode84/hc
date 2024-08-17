// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hc/Transforms/Passes.hpp"

#include "hc/Dialect/PyAST/IR/PyASTOps.hpp"
#include "hc/Dialect/PyIR/IR/PyIROps.hpp"
#include "hc/Dialect/Typing/IR/TypingOps.hpp"

#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace hc {
#define GEN_PASS_DEF_CONVERTPYASTTOIRPASS
#include "hc/Transforms/Passes.h.inc"
} // namespace hc

static hc::py_ir::CmpOpVal convertCmpOpType(hc::py_ast::CmpOp val) {
  return static_cast<hc::py_ir::CmpOpVal>(val);
}

static hc::py_ir::BinOpVal convertBinOpType(hc::py_ast::BinOpVal val) {
  return static_cast<hc::py_ir::BinOpVal>(val);
}

static hc::py_ir::UnaryOpVal convertUnaryOpType(hc::py_ast::UnaryOpVal val) {
  return static_cast<hc::py_ir::UnaryOpVal>(val);
}

static hc::py_ir::BinOpVal convertBoolOpType(hc::py_ast::BoolOpType val) {
  switch (val) {
  case hc::py_ast::BoolOpType::and_:
    return hc::py_ir::BinOpVal::bool_and;
  case hc::py_ast::BoolOpType::or_:
    return hc::py_ir::BinOpVal::bool_or;
  default:
    llvm_unreachable("Invalid BoolOpType");
  };
}

static mlir::Value getVar(mlir::OpBuilder &builder, mlir::Location loc,
                          mlir::Value val) {
  mlir::Type type = hc::py_ir::UndefinedType::get(builder.getContext());
  if (auto const_ = val.getDefiningOp<hc::py_ast::ConstantOp>()) {
    auto attr = const_.getValue();
    if (mlir::isa<hc::py_ast::NoneAttr>(attr))
      return builder.create<hc::py_ir::NoneOp>(loc);

    mlir::Value val = builder.create<hc::py_ir::ConstantOp>(
        loc, mlir::cast<mlir::TypedAttr>(attr));
    return builder.create<hc::typing::CastOp>(loc, type, val);
  }

  if (auto name = val.getDefiningOp<hc::py_ast::NameOp>())
    return builder.create<hc::py_ir::LoadVarOp>(loc, type, name.getId());

  if (auto subscript = val.getDefiningOp<hc::py_ast::SubscriptOp>()) {
    mlir::Value slice = getVar(builder, loc, subscript.getSlice());
    mlir::Value tgt = getVar(builder, loc, subscript.getValue());
    return builder.create<hc::py_ir::GetItemOp>(loc, type, tgt, slice);
  }

  if (auto attr = val.getDefiningOp<hc::py_ast::AttributeOp>()) {
    mlir::Value tgt = getVar(builder, loc, attr.getValue());
    auto name = attr.getAttr();
    return builder.create<hc::py_ir::GetAttrOp>(loc, type, tgt, name);
  }

  if (auto tuple = val.getDefiningOp<hc::py_ast::ListOp>()) {
    llvm::SmallVector<mlir::Value> args;
    for (auto el : tuple.getElts())
      args.emplace_back(getVar(builder, loc, el));

    return builder.create<hc::py_ir::MakeListOp>(loc, type, args);
  }

  if (auto tuple = val.getDefiningOp<hc::py_ast::TupleOp>()) {
    llvm::SmallVector<mlir::Value> args;
    for (auto el : tuple.getElts())
      args.emplace_back(getVar(builder, loc, el));

    return builder.create<hc::py_ir::TuplePackOp>(loc, type, args);
  }

  if (auto binOp = val.getDefiningOp<hc::py_ast::BinOp>()) {
    mlir::Value left = getVar(builder, loc, binOp.getLeft());
    mlir::Value right = getVar(builder, loc, binOp.getRight());
    hc::py_ir::BinOpVal bop = convertBinOpType(binOp.getOp());

    return builder.create<::hc::py_ir::BinOp>(loc, type, left, bop, right);
  }

  if (auto unaryOp = val.getDefiningOp<hc::py_ast::UnaryOp>()) {
    mlir::Value operand = getVar(builder, loc, unaryOp.getOperand());
    hc::py_ir::UnaryOpVal uop = convertUnaryOpType(unaryOp.getOp());

    return builder.create<::hc::py_ir::UnaryOp>(loc, type, uop, operand);
  }

  if (auto cmpOp = val.getDefiningOp<hc::py_ast::CompareOp>()) {
    assert(!cmpOp.getOps().empty());

    mlir::Value result;
    mlir::Value leftVal = getVar(builder, loc, cmpOp.getLeft());
    for (auto &&[op, right] : llvm::zip_equal(
             cmpOp.getOps().getAsValueRange<hc::py_ast::CmpOpAttr>(),
             cmpOp.getComparators())) {
      mlir::Value rightVal = getVar(builder, loc, right);

      hc::py_ir::CmpOpVal cop = convertCmpOpType(op);
      mlir::Value res =
          builder.create<hc::py_ir::CmpOp>(loc, type, leftVal, cop, rightVal);
      leftVal = rightVal;
      if (!result) {
        result = res;
      } else {
        result = builder.create<hc::py_ir::BinOp>(
            loc, type, result, hc::py_ir::BinOpVal::bool_and, res);
      }
    }
    return result;
  }

  if (auto ifExp = val.getDefiningOp<hc::py_ast::IfExpOp>()) {
    mlir::Value test = getVar(builder, loc, ifExp.getTest());
    mlir::Value body = getVar(builder, loc, ifExp.getBody());
    mlir::Value orelse = getVar(builder, loc, ifExp.getOrelse());

    return builder.create<::hc::py_ir::IfExpOp>(loc, type, test, body, orelse);
  }

  if (auto boolOp = val.getDefiningOp<hc::py_ast::BoolOp>()) {
    mlir::ValueRange vals = boolOp.getValues();
    assert(vals.size() >= 2);
    hc::py_ir::BinOpVal bop = convertBoolOpType(boolOp.getOp());

    mlir::Value ret = getVar(builder, loc, vals.front());
    vals = vals.drop_front();
    do {
      auto right = getVar(builder, loc, vals.front());
      vals = vals.drop_front();

      ret = builder.create<::hc::py_ir::BinOp>(loc, type, ret, bop, right);
    } while (!vals.empty());

    return ret;
  }

  if (auto call = val.getDefiningOp<hc::py_ast::CallOp>()) {
    auto func = getVar(builder, loc, call.getFunc());

    llvm::SmallVector<mlir::Value> args;
    for (auto arg : call.getArgs())
      args.emplace_back(getVar(builder, loc, arg));

    llvm::SmallVector<mlir::Attribute> names(args.size(),
                                             builder.getStringAttr(""));

    for (auto arg : call.getKeywords()) {
      auto kwarg = arg.getDefiningOp<hc::py_ast::KeywordOp>();
      assert(kwarg);
      args.emplace_back(getVar(builder, loc, kwarg.getValue()));
      names.emplace_back(kwarg.getArgAttr());
    }

    auto namesAttr = builder.getArrayAttr(names);
    return builder.create<hc::py_ir::CallOp>(loc, type, func, args, namesAttr);
  }

  if (auto slice = val.getDefiningOp<hc::py_ast::SliceOp>()) {
    auto getVarOpt = [&](mlir::Value val) -> mlir::Value {
      if (val)
        return getVar(builder, loc, val);
      return {};
    };
    auto lower = getVarOpt(slice.getLower());
    auto upper = getVarOpt(slice.getUpper());
    auto step = getVarOpt(slice.getStep());

    return builder.create<hc::py_ir::SliceOp>(loc, type, lower, upper, step);
  }

  return val;
}

static void setVar(mlir::OpBuilder &builder, mlir::Location loc,
                   mlir::Value target, mlir::Value val) {
  if (auto name = target.getDefiningOp<hc::py_ast::NameOp>()) {
    builder.create<hc::py_ir::StoreVarOp>(loc, name.getId(), val);
    return;
  }

  if (auto subscript = target.getDefiningOp<hc::py_ast::SubscriptOp>()) {
    mlir::Value slice = getVar(builder, loc, subscript.getSlice());
    mlir::Value tgt = getVar(builder, loc, subscript.getValue());
    builder.create<hc::py_ir::SetItemOp>(loc, tgt, slice, val);
    return;
  }

  if (auto attr = target.getDefiningOp<hc::py_ast::AttributeOp>()) {
    mlir::Value tgt = getVar(builder, loc, attr.getValue());
    auto name = attr.getAttr();
    builder.create<hc::py_ir::SetAttrOp>(loc, tgt, name, val);
    return;
  }

  if (auto tuple = target.getDefiningOp<hc::py_ast::TupleOp>()) {
    llvm::SmallVector<mlir::Type> types(
        tuple.getElts().size(),
        hc::py_ir::UndefinedType::get(builder.getContext()));
    auto unpack = builder.create<hc::py_ir::TupleUnpackOp>(loc, types, val);

    for (auto &&[el, arg] : llvm::zip(tuple.getElts(), unpack.getResults()))
      setVar(builder, loc, el, arg);

    return;
  }

  llvm::errs() << target << "\n";
  llvm_unreachable("Unknown setvar node");
}

static std::pair<mlir::StringRef, mlir::Value>
getArg(mlir::OpBuilder &builder, mlir::Location loc, mlir::Value astNode) {
  auto argOp = astNode.getDefiningOp<hc::py_ast::ArgOp>();
  mlir::Value annotation = argOp.getAnnotation();
  if (!annotation) {
    annotation = builder.create<hc::py_ir::EmptyAnnotationOp>(loc);
  } else {
    annotation = getVar(builder, loc, annotation);
  }

  return {argOp.getName(), annotation};
}

static bool isTopLevel(mlir::Operation *op) {
  auto parent = op->getParentOp();
  return parent && mlir::isa<hc::py_ir::PyFuncOp>(op->getParentOp());
}

static mlir::Value boolCast(mlir::OpBuilder &builder, mlir::Location loc,
                            mlir::Value val) {
  auto type = builder.getIntegerType(1);
  return builder.create<hc::typing::CastOp>(loc, type, val);
}

namespace {
class ConvertModule final
    : public mlir::OpRewritePattern<hc::py_ast::PyModuleOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(hc::py_ast::PyModuleOp op,
                  mlir::PatternRewriter &rewriter) const override {
    {
      auto term =
          mlir::cast<hc::py_ast::BlockEndOp>(op.getBody(0)->getTerminator());
      mlir::OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPoint(term);
      rewriter.replaceOpWithNewOp<hc::py_ir::PyModuleEndOp>(term);
    }

    auto newMod = rewriter.create<hc::py_ir::PyModuleOp>(op.getLoc());

    mlir::Region &dstRegion = newMod.getRegion();
    rewriter.inlineRegionBefore(op.getRegion(), dstRegion, dstRegion.end());
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

static bool checkFuncReturn(mlir::Block *body) {
  auto termIt = body->getTerminator()->getIterator();
  if (body->begin() == termIt)
    return false;

  auto retOp = mlir::dyn_cast<hc::py_ast::ReturnOp>(*std::prev(termIt));
  return retOp && retOp.getValue();
}

class ConvertFunc final : public mlir::OpRewritePattern<hc::py_ast::PyFuncOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(hc::py_ast::PyFuncOp op,
                  mlir::PatternRewriter &rewriter) const override {
    if (!checkFuncReturn(op.getBody()))
      return mlir::failure();

    mlir::Location loc = op.getLoc();
    llvm::SmallVector<mlir::StringRef> argNames;
    llvm::SmallVector<mlir::Value> annotations;
    for (auto arg : op.getArgs()) {
      auto [name, type] = getArg(rewriter, loc, arg);
      argNames.emplace_back(name);
      annotations.emplace_back(type);
    }

    llvm::SmallVector<mlir::Value> decorators;
    for (auto decor : op.getDecorators())
      decorators.emplace_back(getVar(rewriter, loc, decor));

    auto name = op.getName();
    auto type = hc::py_ir::UndefinedType::get(rewriter.getContext());
    auto newOp = rewriter.create<hc::py_ir::PyFuncOp>(
        loc, type, name, argNames, annotations, /*captureNames*/ std::nullopt,
        /*captureArgs*/ std::nullopt, decorators);
    rewriter.create<hc::py_ir::StoreVarOp>(loc, name, newOp.getResult());

    mlir::Region &dstRegion = newOp.getRegion();
    rewriter.inlineRegionBefore(op.getRegion(), dstRegion, dstRegion.end());

    mlir::Block &entryBlock = dstRegion.front();
    mlir::Block &bodyBlock = dstRegion.back();

    mlir::OpBuilder::InsertionGuard g(rewriter);

    rewriter.setInsertionPointToEnd(&entryBlock);
    for (auto &&[name, arg] : llvm::zip(argNames, entryBlock.getArguments()))
      rewriter.create<hc::py_ir::StoreVarOp>(loc, name, arg);

    rewriter.create<mlir::cf::BranchOp>(loc, &bodyBlock);
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

class ConvertReturn final
    : public mlir::OpRewritePattern<hc::py_ast::ReturnOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(hc::py_ast::ReturnOp op,
                  mlir::PatternRewriter &rewriter) const override {
    if (!isTopLevel(op))
      return mlir::failure();

    mlir::Value val = op.getValue();
    if (!val)
      return mlir::failure();

    val = getVar(rewriter, op.getLoc(), val);
    auto newOp = rewriter.replaceOpWithNewOp<hc::py_ir::ReturnOp>(op, val);
    rewriter.splitBlock(newOp->getBlock(), std::next(newOp->getIterator()));
    return mlir::success();
  }
};

class ConvertIf final : public mlir::OpRewritePattern<hc::py_ast::IfOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(hc::py_ast::IfOp op,
                  mlir::PatternRewriter &rewriter) const override {
    if (!isTopLevel(op))
      return mlir::failure();

    mlir::Location loc = op.getLoc();
    mlir::Value cond = getVar(rewriter, loc, op.getTest());
    cond = boolCast(rewriter, loc, cond);

    mlir::Block *condBlock = rewriter.getInsertionBlock();
    auto opPosition = rewriter.getInsertionPoint();
    auto *remainingOpsBlock = rewriter.splitBlock(condBlock, opPosition);

    mlir::OpBuilder::InsertionGuard g(rewriter);
    mlir::Block *thenBlock = &op.getBodyRegion().front();

    mlir::Operation *term = thenBlock->getTerminator();
    if (mlir::isa<hc::py_ast::BlockEndOp>(term)) {
      rewriter.setInsertionPointToEnd(thenBlock);
      rewriter.replaceOpWithNewOp<mlir::cf::BranchOp>(term, remainingOpsBlock);
    }
    rewriter.inlineRegionBefore(op.getBodyRegion(), remainingOpsBlock);

    mlir::Block *elseBlock;
    if (op.getOrelseRegion().empty()) {
      elseBlock = remainingOpsBlock;
    } else {
      elseBlock = &op.getOrelseRegion().front();
      term = elseBlock->getTerminator();
      if (mlir::isa<hc::py_ast::BlockEndOp>(term)) {
        rewriter.setInsertionPointToEnd(elseBlock);
        rewriter.replaceOpWithNewOp<mlir::cf::BranchOp>(
            elseBlock->getTerminator(), remainingOpsBlock);
      }
      rewriter.inlineRegionBefore(op.getOrelseRegion(), remainingOpsBlock);
    }

    rewriter.setInsertionPointToEnd(condBlock);
    rewriter.create<mlir::cf::CondBranchOp>(loc, cond, thenBlock, elseBlock);
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

static void replaceBreakAndContinue(mlir::PatternRewriter &rewriter,
                                    mlir::Block *thenBlock,
                                    mlir::Block *condBlock,
                                    const mlir::ValueRange &condBlockArgs,
                                    mlir::Block *remainingBlock) {
  mlir::OpBuilder::InsertionGuard g(rewriter);

  llvm::SmallVector<mlir::Operation *> continues;
  llvm::SmallVector<mlir::Operation *> breaks;

  thenBlock->walk([&](mlir::Operation *op) {
    if (mlir::isa<hc::py_ast::ForOp>(*op) ||
        mlir::isa<hc::py_ast::WhileOp>(*op)) {
      return mlir::WalkResult::skip();
    } else if (mlir::isa<hc::py_ast::ContinueOp>(*op)) {
      continues.push_back(op);
    } else if (mlir::isa<hc::py_ast::BreakOp>(*op)) {
      breaks.push_back(op);
    }

    return mlir::WalkResult::advance();
  });

  for (auto &&cont : continues) {
    mlir::Block *opBlock = cont->getBlock();
    rewriter.setInsertionPointToEnd(opBlock);
    rewriter.replaceOpWithNewOp<mlir::cf::BranchOp>(opBlock->getTerminator(),
                                                    condBlock, condBlockArgs);
    rewriter.eraseOp(cont);
  }

  for (auto &&brk : breaks) {
    mlir::Block *opBlock = brk->getBlock();
    rewriter.setInsertionPointToEnd(opBlock);
    rewriter.replaceOpWithNewOp<mlir::cf::BranchOp>(opBlock->getTerminator(),
                                                    remainingBlock);
    rewriter.eraseOp(brk);
  }
}

class ConvertWhile final : public mlir::OpRewritePattern<hc::py_ast::WhileOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(hc::py_ast::WhileOp op,
                  mlir::PatternRewriter &rewriter) const override {
    if (!isTopLevel(op))
      return mlir::failure();

    mlir::Location loc = op.getLoc();

    mlir::Block *beforeBlock = rewriter.getInsertionBlock();

    auto opPosition = rewriter.getInsertionPoint();
    auto *remainingOpsBlock = rewriter.splitBlock(beforeBlock, opPosition);

    mlir::OpBuilder::InsertionGuard g(rewriter);
    mlir::Block *condBlock = rewriter.createBlock(remainingOpsBlock);
    rewriter.setInsertionPointToEnd(beforeBlock);
    rewriter.create<mlir::cf::BranchOp>(loc, condBlock);

    mlir::Block *thenBlock = &op.getBodyRegion().front();
    rewriter.setInsertionPointToEnd(thenBlock);
    rewriter.replaceOpWithNewOp<mlir::cf::BranchOp>(thenBlock->getTerminator(),
                                                    condBlock);
    rewriter.inlineRegionBefore(op.getBodyRegion(), remainingOpsBlock);

    replaceBreakAndContinue(rewriter, thenBlock, condBlock, mlir::ValueRange{},
                            remainingOpsBlock);

    rewriter.setInsertionPointToEnd(condBlock);

    mlir::Value cond = getVar(rewriter, loc, op.getTest());
    cond = boolCast(rewriter, loc, cond);

    rewriter.create<mlir::cf::CondBranchOp>(loc, cond, thenBlock,
                                            remainingOpsBlock);
    rewriter.eraseOp(op);

    return mlir::success();
  }
};

class ConvertFor final : public mlir::OpRewritePattern<hc::py_ast::ForOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(hc::py_ast::ForOp op,
                  mlir::PatternRewriter &rewriter) const override {
    if (!isTopLevel(op))
      return mlir::failure();

    mlir::Type undefType = hc::py_ir::UndefinedType::get(rewriter.getContext());

    mlir::OpBuilder::InsertionGuard g(rewriter);

    mlir::Location loc = op.getLoc();

    mlir::Block *beforeBlock = rewriter.getInsertionBlock();

    auto opPosition = rewriter.getInsertionPoint();
    auto *remainingOpsBlock = rewriter.splitBlock(beforeBlock, opPosition);

    mlir::Block *condBlock =
        rewriter.createBlock(remainingOpsBlock, undefType, loc);
    rewriter.setInsertionPointToEnd(beforeBlock);
    mlir::Value container = getVar(rewriter, loc, op.getIter());
    mlir::Value iter =
        rewriter.create<hc::py_ir::IterOp>(loc, undefType, container);
    rewriter.create<mlir::cf::BranchOp>(loc, condBlock, iter);

    mlir::Block *thenBlock = &op.getBodyRegion().front();
    mlir::Value thenValue =
        thenBlock->addArgument(undefType, rewriter.getUnknownLoc());
    mlir::Value thenIter =
        thenBlock->addArgument(undefType, rewriter.getUnknownLoc());
    rewriter.setInsertionPointToStart(thenBlock);
    setVar(rewriter, loc, op.getTarget(), thenValue);
    rewriter.setInsertionPointToEnd(thenBlock);
    rewriter.replaceOpWithNewOp<mlir::cf::BranchOp>(thenBlock->getTerminator(),
                                                    condBlock, thenIter);
    rewriter.inlineRegionBefore(op.getBodyRegion(), remainingOpsBlock);

    replaceBreakAndContinue(rewriter, thenBlock, condBlock,
                            mlir::ValueRange{thenIter}, remainingOpsBlock);

    rewriter.setInsertionPointToEnd(condBlock);
    mlir::Value condIter = condBlock->getArgument(0);
    auto next = rewriter.create<hc::py_ir::NextOp>(
        loc, undefType, rewriter.getIntegerType(1), undefType, condIter);
    mlir::Value cond = getVar(rewriter, loc, op.getTarget());

    rewriter.create<mlir::cf::CondBranchOp>(
        loc, next.getValid(), thenBlock,
        mlir::ValueRange{next.getValue(), next.getNextiter()},
        remainingOpsBlock, mlir::ValueRange{});
    rewriter.eraseOp(op);

    return mlir::success();
  }
};

class ConvertAssign final
    : public mlir::OpRewritePattern<hc::py_ast::AssignOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(hc::py_ast::AssignOp op,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    mlir::Value val = getVar(rewriter, loc, op.getValue());

    for (auto target : op.getTargets())
      setVar(rewriter, loc, target, val);

    rewriter.eraseOp(op);
    return mlir::success();
  }
};

class ConvertAugAssign final
    : public mlir::OpRewritePattern<hc::py_ast::AugAssignOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(hc::py_ast::AugAssignOp op,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    hc::py_ir::BinOpVal bop = convertBinOpType(op.getOp());
    mlir::Value left = getVar(rewriter, loc, op.getTarget());
    mlir::Value right = getVar(rewriter, loc, op.getValue());
    mlir::Type type = hc::py_ir::UndefinedType::get(rewriter.getContext());
    mlir::Value val =
        rewriter.create<hc::py_ir::InplaceBinOp>(loc, type, left, bop, right);
    setVar(rewriter, loc, op.getTarget(), val);

    rewriter.eraseOp(op);
    return mlir::success();
  }
};

class ConvertExpr final : public mlir::OpRewritePattern<hc::py_ast::ExprOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(hc::py_ast::ExprOp op,
                  mlir::PatternRewriter &rewriter) const override {
    (void)getVar(rewriter, op.getLoc(), op.getValue());
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

class ConvertCaptureVal final
    : public mlir::OpRewritePattern<hc::py_ast::CaptureValOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(hc::py_ast::CaptureValOp op,
                  mlir::PatternRewriter &rewriter) const override {
    // Direct parent must be a PyModuleOp.
    auto mod = mlir::dyn_cast<hc::py_ir::PyModuleOp>(op->getParentOp());
    if (!mod)
      return mlir::failure();

    auto term =
        mlir::cast<hc::py_ir::PyModuleEndOp>(mod.getBody()->getTerminator());

    mlir::Type type = hc::py_ir::UndefinedType::get(rewriter.getContext());
    mlir::OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(term);
    mlir::Value val = rewriter.create<hc::py_ir::LoadVarOp>(term.getLoc(), type,
                                                            op.getIdAttr());

    auto args = llvm::to_vector(term.getResults());
    args.emplace_back(val);
    rewriter.replaceOpWithNewOp<hc::py_ir::PyModuleEndOp>(term, args);

    rewriter.eraseOp(op);
    return mlir::success();
  }
};

struct ConvertPyASTToIRPass final
    : public hc::impl::ConvertPyASTToIRPassBase<ConvertPyASTToIRPass> {

  void runOnOperation() override {
    auto *ctx = &getContext();
    mlir::RewritePatternSet patterns(ctx);
    hc::populateConvertPyASTToIRPatterns(patterns);
    mlir::cf::BranchOp::getCanonicalizationPatterns(patterns, ctx);
    mlir::cf::CondBranchOp::getCanonicalizationPatterns(patterns, ctx);

    if (mlir::failed(
            applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();

    getOperation()->walk([&](mlir::Operation *op) {
      if (!mlir::isa_and_present<hc::py_ast::PyASTDialect>(op->getDialect()))
        return;

      op->emitError("Unconverted AST op");
      signalPassFailure();
    });
  }
};
} // namespace

void hc::populateConvertPyASTToIRPatterns(mlir::RewritePatternSet &patterns) {
  patterns.insert<ConvertModule, ConvertFunc, ConvertReturn, ConvertIf,
                  ConvertWhile, ConvertFor, ConvertAssign, ConvertAugAssign,
                  ConvertExpr, ConvertCaptureVal>(patterns.getContext());
}
