// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hc/Dialect/HKernel/IR/HKernelOps.hpp"

#include "hc/Dialect/Typing/IR/TypingOps.hpp"

#include <mlir/Dialect/Utils/StaticValueUtils.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/PatternMatch.h>

#include <llvm/ADT/TypeSwitch.h>

void hc::hk::HKernelDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "hc/Dialect/HKernel/IR/HKernelOps.cpp.inc"
      >();

  addTypes<
#define GET_TYPEDEF_LIST
#include "hc/Dialect/HKernel/IR/HKernelOpsTypes.cpp.inc"
      >();

  addAttributes<
#define GET_ATTRDEF_LIST
#include "hc/Dialect/HKernel/IR/HKernelOpsAttributes.cpp.inc"
      >();
}

hc::hk::SymbolicallyShapedType
hc::hk::BufferType::cloneWith(std::optional<llvm::ArrayRef<mlir::Type>> shape,
                              mlir::Type elementType) const {
  return BufferType::get(getContext(), shape ? *shape : getShape(),
                         elementType ? elementType : getElementType());
}

hc::hk::SymbolicallyShapedType
hc::hk::TensorType::cloneWith(std::optional<llvm::ArrayRef<mlir::Type>> shape,
                              mlir::Type elementType) const {
  return TensorType::get(getContext(), shape ? *shape : getShape(),
                         elementType ? elementType : getElementType());
}

void hc::hk::MakeSliceOp::build(::mlir::OpBuilder &odsBuilder,
                                ::mlir::OperationState &odsState,
                                mlir::Value lower, mlir::Value upper,
                                mlir::Value step) {
  auto type = SliceType::get(odsBuilder.getContext(),
                             lower ? lower.getType() : odsBuilder.getNoneType(),
                             upper ? upper.getType() : odsBuilder.getNoneType(),
                             step ? step.getType() : odsBuilder.getNoneType());
  build(odsBuilder, odsState, type, lower, upper, step);
}

mlir::OpFoldResult hc::hk::MaterializeExprOp::fold(FoldAdaptor adaptor) {
  return hc::typing::TypeAttr::get(getType());
}

mlir::OpFoldResult hc::hk::TupleExtractOp::fold(FoldAdaptor adaptor) {
  if (auto idx = mlir::getConstantIntValue(adaptor.getIndex())) {
    auto src = getSource();
    auto def = src.getDefiningOp<MakeTupleOp>();
    if (!def)
      return nullptr;

    mlir::ValueRange args = def.getArgs();

    auto i = *idx;
    auto tupleType = mlir::cast<mlir::TupleType>(src.getType());
    assert(args.getTypes() == tupleType.getTypes());
    if (i < 0 || static_cast<size_t>(i) >= tupleType.size() ||
        tupleType.getType(i) != getType())
      return nullptr;

    return args[i];
  }

  return nullptr;
}

namespace {
struct FoldSubviewChain : public mlir::OpRewritePattern<hc::hk::SubViewOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(hc::hk::SubViewOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto src = op.getSource().getDefiningOp<hc::hk::SubViewOp>();
    if (!src)
      return mlir::failure();

    if (src.getIndex().size() != op.getIndex().size())
      return mlir::failure();

    auto check = [](mlir::Value v) -> bool {
      auto slice = mlir::dyn_cast<hc::hk::SliceType>(v.getType());
      if (!slice)
        return false;

      return mlir::isa<hc::typing::SymbolicTypeBase>(slice.getLower()) &&
             mlir::isa<mlir::NoneType>(slice.getUpper()) &&
             mlir::isa<mlir::NoneType>(slice.getStep());
    };
    if (!llvm::all_of(src.getIndex(), check) ||
        !llvm::all_of(op.getIndex(), check))
      return mlir::failure();

    auto getLower = [&](mlir::Type t) -> hc::typing::SymbolicTypeBase {
      auto slice = mlir::cast<hc::hk::SliceType>(t);
      return mlir::cast<hc::typing::SymbolicTypeBase>(slice.getLower());
    };

    mlir::Location loc = op.getLoc();
    llvm::SmallVector<mlir::Value> index;
    for (auto &&[s, d] : llvm::zip_equal(src.getIndex(), op.getIndex())) {
      auto srcType = getLower(s.getType());
      auto dstType = getLower(d.getType());
      mlir::Value idx =
          rewriter.create<hc::hk::MaterializeExprOp>(loc, srcType + dstType);
      index.emplace_back(rewriter.create<hc::hk::MakeSliceOp>(loc, idx));
    }

    rewriter.replaceOpWithNewOp<hc::hk::SubViewOp>(op, op.getType(),
                                                   src.getSource(), index);
    return mlir::success();
  }
};
} // namespace

void hc::hk::SubViewOp::getCanonicalizationPatterns(
    mlir::RewritePatternSet &results, mlir::MLIRContext *context) {
  results.insert<FoldSubviewChain>(context);
}

/// Given the region at `index`, or the parent operation if `index` is None,
/// return the successor regions. These are the regions that may be selected
/// during the flow of control. `operands` is a set of optional attributes that
/// correspond to a constant value for each operand, or null if that operand is
/// not a constant.
void hc::hk::EnvironmentRegionOp::getSuccessorRegions(
    mlir::RegionBranchPoint point,
    mlir::SmallVectorImpl<mlir::RegionSuccessor> &regions) {
  // If the predecessor is the ExecuteRegionOp, branch into the body.
  if (point.isParent()) {
    regions.push_back(mlir::RegionSuccessor(&getRegion()));
    return;
  }

  // Otherwise, the region branches back to the parent operation.
  regions.push_back(mlir::RegionSuccessor(getResults()));
}

/// Propagate yielded values, defined outside region.
struct EnvRegionPropagateOutsideValues
    : public mlir::OpRewritePattern<hc::hk::EnvironmentRegionOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(hc::hk::EnvironmentRegionOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto oldResults = op.getResults();
    auto count = static_cast<unsigned>(oldResults.size());

    mlir::Block *body = &op.getRegion().front();
    auto term =
        mlir::cast<hc::hk::EnvironmentRegionYieldOp>(body->getTerminator());
    auto termArgs = term.getResults();
    assert(oldResults.size() == termArgs.size());

    // Build list of propagated and new yield args.
    llvm::SmallVector<mlir::Value> newResults(count);
    llvm::SmallVector<mlir::Value> newYieldArgs;
    for (auto i : llvm::seq(0u, count)) {
      auto arg = termArgs[i];
      if (!op.getRegion().isAncestor(arg.getParentRegion())) {
        // Value defined outside op region - use it directly instead of
        // yielding.
        newResults[i] = arg;
      } else {
        newYieldArgs.emplace_back(arg);
      }
    }

    // Same yield results count - nothing changed.
    if (newYieldArgs.size() == count)
      return mlir::failure();

    // Contruct new env region op, only yielding values that weren't propagated.
    mlir::ValueRange newYieldArgsRange(newYieldArgs);
    auto newOp = rewriter.create<hc::hk::EnvironmentRegionOp>(
        op->getLoc(), newYieldArgsRange.getTypes(), op.getEnvironment(),
        op.getArgs());
    mlir::Region &newRegion = newOp.getRegion();
    rewriter.inlineRegionBefore(op.getRegion(), newRegion, newRegion.end());
    {
      mlir::OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPoint(term);
      rewriter.replaceOpWithNewOp<hc::hk::EnvironmentRegionYieldOp>(
          term, newYieldArgs);
    }

    mlir::ValueRange newOpResults = newOp.getResults();

    // Fill results that weren't propagated with results of new op.
    for (auto i : llvm::seq(0u, count)) {
      if (!newResults[i]) {
        newResults[i] = newOpResults.front();
        newOpResults = newOpResults.drop_front();
      }
    }
    assert(newOpResults.empty() &&
           "Some values weren't consumed - yield args count mismatch?");

    rewriter.replaceOp(op, newResults);
    return mlir::success();
  }
};

namespace {
/// Merge nested env region if parent have same environment and args.
struct MergeNestedEnvRegion
    : public mlir::OpRewritePattern<hc::hk::EnvironmentRegionOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(hc::hk::EnvironmentRegionOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto parent = op->getParentOfType<hc::hk::EnvironmentRegionOp>();
    if (!parent)
      return mlir::failure();

    if (parent.getEnvironment() != op.getEnvironment() ||
        parent.getArgs() != op.getArgs())
      return mlir::failure();

    hc::hk::EnvironmentRegionOp::inlineIntoParent(rewriter, op);
    return mlir::success();
  }
};

/// Remove duplicated and unused env region yield args.
struct CleanupRegionYieldArgs
    : public mlir::OpRewritePattern<hc::hk::EnvironmentRegionOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(hc::hk::EnvironmentRegionOp op,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::Block *body = &op.getRegion().front();
    auto term =
        mlir::cast<hc::hk::EnvironmentRegionYieldOp>(body->getTerminator());

    auto results = op.getResults();
    auto yieldArgs = term.getResults();
    assert(results.size() == yieldArgs.size());
    auto count = static_cast<unsigned>(results.size());

    // Build new yield args list, and mapping between old and new results
    llvm::SmallVector<mlir::Value> newYieldArgs;
    llvm::SmallVector<int> newResultsMapping(count, -1);
    llvm::SmallDenseMap<mlir::Value, int> argsMap;
    for (auto i : llvm::seq(0u, count)) {
      auto res = results[i];

      // Unused result.
      if (res.getUses().empty())
        continue;

      auto arg = yieldArgs[i];
      auto it = argsMap.find_as(arg);
      if (it == argsMap.end()) {
        // Add new result, compute index mapping for it.
        auto ind = static_cast<int>(newYieldArgs.size());
        argsMap.insert({arg, ind});
        newYieldArgs.emplace_back(arg);
        newResultsMapping[i] = ind;
      } else {
        // Duplicated result, reuse prev result index.
        newResultsMapping[i] = it->second;
      }
    }

    // Same yield results count - nothing changed.
    if (newYieldArgs.size() == count)
      return mlir::failure();

    // Contruct new env region op, only yielding values we selected.
    mlir::ValueRange newYieldArgsRange(newYieldArgs);
    auto newOp = rewriter.create<hc::hk::EnvironmentRegionOp>(
        op->getLoc(), newYieldArgsRange.getTypes(), op.getEnvironment(),
        op.getArgs());
    mlir::Region &newRegion = newOp.getRegion();
    rewriter.inlineRegionBefore(op.getRegion(), newRegion, newRegion.end());
    {
      mlir::OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPoint(term);
      rewriter.replaceOpWithNewOp<hc::hk::EnvironmentRegionYieldOp>(
          term, newYieldArgs);
    }

    // Contruct new result list, using mapping previously constructed.
    auto newResults = newOp.getResults();
    llvm::SmallVector<mlir::Value> newResultsToTeplace(count);
    for (auto i : llvm::seq(0u, count)) {
      auto mapInd = newResultsMapping[i];
      if (mapInd != -1)
        newResultsToTeplace[i] = newResults[mapInd];
    }

    rewriter.replaceOp(op, newResultsToTeplace);
    return mlir::success();
  }
};

/// Merge adjacent env regions.
struct MergeAdjacentRegions
    : public mlir::OpRewritePattern<hc::hk::EnvironmentRegionOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(hc::hk::EnvironmentRegionOp op,
                  mlir::PatternRewriter &rewriter) const override {
    // Get next pos and check if it is also env region op, current op cannot be
    // last as it is not a terminator.
    auto opPos = op->getIterator();
    auto nextOp =
        mlir::dyn_cast<hc::hk::EnvironmentRegionOp>(*std::next(opPos));
    if (!nextOp)
      return mlir::failure();

    if (nextOp.getEnvironment() != op.getEnvironment() ||
        nextOp.getArgs() != op.getArgs())
      return mlir::failure();

    mlir::Block *body = &op.getRegion().front();
    auto term =
        mlir::cast<hc::hk::EnvironmentRegionYieldOp>(body->getTerminator());

    auto results = op.getResults();
    auto yieldArgs = term.getResults();
    assert(results.size() == yieldArgs.size());
    auto count = static_cast<unsigned>(results.size());

    // Check if any results from first op are being used in second one, we need
    // to replace them by direct values.
    for (auto i : llvm::seq(0u, count)) {
      auto res = results[i];
      for (auto &use : llvm::make_early_inc_range(res.getUses())) {
        auto *owner = use.getOwner();
        if (nextOp->isProperAncestor(owner)) {
          auto arg = yieldArgs[i];
          rewriter.modifyOpInPlace(owner, [&]() { use.set(arg); });
        }
      }
    }

    mlir::Block *nextBody = &nextOp.getRegion().front();
    auto nextTerm =
        mlir::cast<hc::hk::EnvironmentRegionYieldOp>(nextBody->getTerminator());
    auto nextYieldArgs = nextTerm.getResults();

    // Contruct merged yield args list, some of the results may become unused,
    // but they will be cleaned up by other pattern.
    llvm::SmallVector<mlir::Value> newYieldArgs(count + nextYieldArgs.size());
    llvm::copy(nextYieldArgs, llvm::copy(yieldArgs, newYieldArgs.begin()));

    {
      // Merge region from second op into ferst one.
      mlir::OpBuilder::InsertionGuard g(rewriter);
      rewriter.inlineBlockBefore(nextBody, term);
      rewriter.setInsertionPoint(term);
      rewriter.create<hc::hk::EnvironmentRegionYieldOp>(term->getLoc(),
                                                        newYieldArgs);
      rewriter.eraseOp(term);
      rewriter.eraseOp(nextTerm);
    }

    // Contruct new env region op and steal new merged region into it.
    mlir::ValueRange newYieldArgsRange(newYieldArgs);
    auto newOp = rewriter.create<hc::hk::EnvironmentRegionOp>(
        op->getLoc(), newYieldArgsRange.getTypes(), op.getEnvironment(),
        op.getArgs());
    mlir::Region &newRegion = newOp.getRegion();
    rewriter.inlineRegionBefore(op.getRegion(), newRegion, newRegion.end());

    auto newResults = newOp.getResults();

    rewriter.replaceOp(op, newResults.take_front(count));
    rewriter.replaceOp(nextOp, newResults.drop_front(count));
    return mlir::success();
  }
};
} // namespace

void hc::hk::EnvironmentRegionOp::getCanonicalizationPatterns(
    mlir::RewritePatternSet &results, mlir::MLIRContext *context) {
  results.insert<EnvRegionPropagateOutsideValues, MergeNestedEnvRegion,
                 CleanupRegionYieldArgs, MergeAdjacentRegions>(context);
}

void hc::hk::EnvironmentRegionOp::inlineIntoParent(
    mlir::PatternRewriter &builder, EnvironmentRegionOp op) {
  mlir::Block *block = &op.getRegion().front();
  auto term = mlir::cast<EnvironmentRegionYieldOp>(block->getTerminator());
  auto args = llvm::to_vector(term.getResults());
  builder.eraseOp(term);
  builder.inlineBlockBefore(block, op);
  builder.replaceOp(op, args);
}

void hc::hk::EnvironmentRegionOp::build(
    ::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState,
    ::mlir::Attribute environment, ::mlir::ValueRange args,
    ::mlir::TypeRange results,
    ::llvm::function_ref<void(::mlir::OpBuilder &, ::mlir::Location)>
        bodyBuilder) {
  build(odsBuilder, odsState, results, environment, args);
  mlir::Region *bodyRegion = odsState.regions.back().get();

  bodyRegion->push_back(new mlir::Block);
  mlir::Block &bodyBlock = bodyRegion->front();
  if (bodyBuilder) {
    mlir::OpBuilder::InsertionGuard guard(odsBuilder);
    odsBuilder.setInsertionPointToStart(&bodyBlock);
    bodyBuilder(odsBuilder, odsState.location);
  }
  ensureTerminator(*bodyRegion, odsBuilder, odsState.location);
}

void hc::hk::SuggestBlockSizeOp::build(::mlir::OpBuilder &odsBuilder,
                                       ::mlir::OperationState &odsState,
                                       mlir::ValueRange args) {
  llvm::SmallVector<mlir::Type> types(args.size(), odsBuilder.getIndexType());
  build(odsBuilder, odsState, types, args);
}

// TODO: Upstream changes in affine parser
namespace {
using namespace mlir;
/// Lower precedence ops (all at the same precedence level). LNoOp is false in
/// the boolean sense.
enum AffineLowPrecOp {
  /// Null value.
  LNoOp,
  Add,
  Sub
};

/// Higher precedence ops - all at the same precedence level. HNoOp is false
/// in the boolean sense.
enum AffineHighPrecOp {
  /// Null value.
  HNoOp,
  Mul,
  FloorDiv,
  CeilDiv,
  Mod
};

enum class BindingStrength {
  Weak,   // + and -
  Strong, // All other binary operators.
};

} // namespace

using SymbolMap = llvm::SmallMapVector<mlir::StringAttr, unsigned, 8>;

static mlir::AffineExpr parseAffineExprImpl(mlir::AsmParser &parser,
                                            SymbolMap &symbolMap);

static mlir::AffineExpr parseParentheticalExpr(mlir::AsmParser &parser,
                                               SymbolMap &symbolMap) {
  if (!parser.parseOptionalRParen())
    return parser.emitError(parser.getCurrentLocation(),
                            "no expression inside parentheses"),
           nullptr;

  auto expr = parseAffineExprImpl(parser, symbolMap);
  if (!expr)
    return nullptr;

  if (parser.parseRParen())
    return parser.emitError(parser.getCurrentLocation(), "expected ')'"),
           nullptr;

  return expr;
}

static mlir::AffineExpr parseAffineOperandExpr(mlir::AsmParser &parser,
                                               SymbolMap &symbolMap,
                                               AffineExpr lhs);

static mlir::AffineExpr parseNegateExpression(mlir::AsmParser &parser,
                                              SymbolMap &symbolMap,
                                              AffineExpr lhs) {
  AffineExpr operand = parseAffineOperandExpr(parser, symbolMap, lhs);
  // Since negation has the highest precedence of all ops (including high
  // precedence ops) but lower than parentheses, we are only going to use
  // parseAffineOperandExpr instead of parseAffineExpr here.
  if (!operand) {
    // Extra error message although parseAffineOperandExpr would have
    // complained. Leads to a better diagnostic.
    parser.emitError(parser.getCurrentLocation(),
                     "missing operand of negation");
    return nullptr;
  }
  return (-1) * operand;
}

static mlir::AffineExpr parseAffineOperandExpr(mlir::AsmParser &parser,
                                               SymbolMap &symbolMap,
                                               AffineExpr lhs) {
  int64_t val;
  if (parser.parseOptionalInteger(val).has_value())
    return mlir::getAffineConstantExpr(val, parser.getContext());

  std::string str;
  if (!parser.parseOptionalString(&str)) {
    auto attr = parser.getBuilder().getStringAttr(str);
    auto it = symbolMap.find(attr);
    if (it == symbolMap.end())
      it = symbolMap.insert({attr, static_cast<unsigned>(symbolMap.size())})
               .first;

    return mlir::getAffineSymbolExpr(it->second, parser.getContext());
  }

  if (!parser.parseOptionalLParen())
    return parseParentheticalExpr(parser, symbolMap);

  if (!parser.parseOptionalMinus())
    return parseNegateExpression(parser, symbolMap, lhs);

  if (!parser.parseOptionalKeyword("floordiv") ||
      !parser.parseOptionalKeyword("ceildiv") ||
      !parser.parseOptionalKeyword("mod") || !parser.parseOptionalPlus() ||
      !parser.parseOptionalStar()) {
    auto loc = parser.getCurrentLocation();
    if (lhs)
      parser.emitError(loc, "missing right operand of binary operator");
    else
      parser.emitError(loc, "missing left operand of binary operator");
    return nullptr;
  }

  auto loc = parser.getCurrentLocation();
  if (lhs)
    parser.emitError(loc, "missing right operand of binary operator");
  else
    parser.emitError(loc, "expected affine expression");
  return nullptr;
}

static AffineLowPrecOp consumeIfLowPrecOp(mlir::AsmParser &parser) {
  if (!parser.parseOptionalPlus())
    return AffineLowPrecOp::Add;

  if (!parser.parseOptionalMinus())
    return AffineLowPrecOp::Sub;

  return AffineLowPrecOp::LNoOp;
}

static AffineHighPrecOp consumeIfHighPrecOp(mlir::AsmParser &parser) {
  if (!parser.parseOptionalStar())
    return Mul;

  if (!parser.parseOptionalKeyword("floordiv"))
    return FloorDiv;

  if (!parser.parseOptionalKeyword("ceildiv"))
    return CeilDiv;

  if (!parser.parseOptionalKeyword("mod"))
    return Mod;

  return HNoOp;
}

static mlir::AffineExpr getAffineBinaryOpExpr(mlir::AsmParser &parser,
                                              AffineHighPrecOp op,
                                              AffineExpr lhs, AffineExpr rhs,
                                              SMLoc opLoc) {
  // TODO: make the error location info accurate.
  switch (op) {
  case Mul:
    if (!lhs.isSymbolicOrConstant() && !rhs.isSymbolicOrConstant()) {
      parser.emitError(opLoc,
                       "non-affine expression: at least one of the multiply "
                       "operands has to be either a constant or symbolic");
      return nullptr;
    }
    return lhs * rhs;
  case FloorDiv:
    if (!rhs.isSymbolicOrConstant()) {
      parser.emitError(opLoc,
                       "non-affine expression: right operand of floordiv "
                       "has to be either a constant or symbolic");
      return nullptr;
    }
    return lhs.floorDiv(rhs);
  case CeilDiv:
    if (!rhs.isSymbolicOrConstant()) {
      parser.emitError(opLoc, "non-affine expression: right operand of ceildiv "
                              "has to be either a constant or symbolic");
      return nullptr;
    }
    return lhs.ceilDiv(rhs);
  case Mod:
    if (!rhs.isSymbolicOrConstant()) {
      parser.emitError(opLoc, "non-affine expression: right operand of mod "
                              "has to be either a constant or symbolic");
      return nullptr;
    }
    return lhs % rhs;
  case HNoOp:
    llvm_unreachable("can't create affine expression for null high prec op");
    return nullptr;
  }
  llvm_unreachable("Unknown AffineHighPrecOp");
}

static mlir::AffineExpr getAffineBinaryOpExpr(AffineLowPrecOp op,
                                              AffineExpr lhs, AffineExpr rhs) {
  switch (op) {
  case AffineLowPrecOp::Add:
    return lhs + rhs;
  case AffineLowPrecOp::Sub:
    return lhs - rhs;
  case AffineLowPrecOp::LNoOp:
    llvm_unreachable("can't create affine expression for null low prec op");
    return nullptr;
  }
  llvm_unreachable("Unknown AffineLowPrecOp");
}

static mlir::AffineExpr parseAffineHighPrecOpExpr(mlir::AsmParser &parser,
                                                  SymbolMap &symbolMap,
                                                  AffineExpr llhs,
                                                  AffineHighPrecOp llhsOp,
                                                  SMLoc llhsOpLoc) {
  AffineExpr lhs = parseAffineOperandExpr(parser, symbolMap, llhs);
  if (!lhs)
    return nullptr;

  // Found an LHS. Parse the remaining expression.
  auto opLoc = parser.getCurrentLocation();
  if (AffineHighPrecOp op = consumeIfHighPrecOp(parser)) {
    if (llhs) {
      AffineExpr expr = getAffineBinaryOpExpr(parser, llhsOp, llhs, lhs, opLoc);
      if (!expr)
        return nullptr;
      return parseAffineHighPrecOpExpr(parser, symbolMap, expr, op, opLoc);
    }
    // No LLHS, get RHS
    return parseAffineHighPrecOpExpr(parser, symbolMap, lhs, op, opLoc);
  }

  // This is the last operand in this expression.
  if (llhs)
    return getAffineBinaryOpExpr(parser, llhsOp, llhs, lhs, llhsOpLoc);

  // No llhs, 'lhs' itself is the expression.
  return lhs;
}

static mlir::AffineExpr parseAffineLowPrecOpExpr(mlir::AsmParser &parser,
                                                 SymbolMap &symbolMap,
                                                 AffineExpr llhs,
                                                 AffineLowPrecOp llhsOp) {
  AffineExpr lhs;
  if (!(lhs = parseAffineOperandExpr(parser, symbolMap, llhs)))
    return nullptr;

  // Found an LHS. Deal with the ops.
  if (AffineLowPrecOp lOp = consumeIfLowPrecOp(parser)) {
    if (llhs) {
      AffineExpr sum = getAffineBinaryOpExpr(llhsOp, llhs, lhs);
      return parseAffineLowPrecOpExpr(parser, symbolMap, sum, lOp);
    }
    // No LLHS, get RHS and form the expression.
    return parseAffineLowPrecOpExpr(parser, symbolMap, lhs, lOp);
  }
  auto opLoc = parser.getCurrentLocation();
  if (AffineHighPrecOp hOp = consumeIfHighPrecOp(parser)) {
    // We have a higher precedence op here. Get the rhs operand for the llhs
    // through parseAffineHighPrecOpExpr.
    AffineExpr highRes =
        parseAffineHighPrecOpExpr(parser, symbolMap, lhs, hOp, opLoc);
    if (!highRes)
      return nullptr;

    // If llhs is null, the product forms the first operand of the yet to be
    // found expression. If non-null, the op to associate with llhs is llhsOp.
    AffineExpr expr =
        llhs ? getAffineBinaryOpExpr(llhsOp, llhs, highRes) : highRes;

    // Recurse for subsequent low prec op's after the affine high prec op
    // expression.
    if (AffineLowPrecOp nextOp = consumeIfLowPrecOp(parser))
      return parseAffineLowPrecOpExpr(parser, symbolMap, expr, nextOp);
    return expr;
  }
  // Last operand in the expression list.
  if (llhs)
    return getAffineBinaryOpExpr(llhsOp, llhs, lhs);
  // No llhs, 'lhs' itself is the expression.
  return lhs;
}

static mlir::AffineExpr parseAffineExprImpl(mlir::AsmParser &parser,
                                            SymbolMap &symbolMap) {
  return parseAffineLowPrecOpExpr(parser, symbolMap, nullptr,
                                  AffineLowPrecOp::LNoOp);
}

static mlir::Type parseExpr(mlir::AsmParser &parser) {
  SymbolMap symbolMap;
  auto expr = parseAffineExprImpl(parser, symbolMap);
  if (!expr)
    return nullptr;

  llvm::SmallVector<mlir::Type> args;
  args.reserve(symbolMap.size());
  for (auto &&[key, val] : symbolMap) {
    (void)val;
    args.emplace_back(hc::typing::SymbolType::get(parser.getContext(), key));
  }
  symbolMap.clear();
  auto symExpr = hc::typing::ExprType::get(parser.getContext(), args, expr);
  return hc::typing::SymbolicTypeBase::foldExpr(symExpr);
}

static mlir::LogicalResult
parseSymbolicShape(mlir::AsmParser &parser,
                   llvm::SmallVectorImpl<mlir::Type> &shape) {
  if (parser.parseLess())
    return parser.emitError(parser.getCurrentLocation(), "'<' expected");

  if (!parser.parseOptionalGreater())
    return mlir::success();

  do {
    auto expr = parseExpr(parser);
    if (!expr)
      return parser.emitError(parser.getCurrentLocation(),
                              "failed to parse expr");

    shape.emplace_back(expr);
  } while (!parser.parseOptionalKeyword("x"));
  return parser.parseGreater();
}

static void printAffineExprInternal(
    mlir::AsmPrinter &os, mlir::AffineExpr expr,
    BindingStrength enclosingTightness,
    llvm::function_ref<void(unsigned, bool)> printValueName) {
  using namespace mlir;
  const char *binopSpelling = nullptr;
  switch (expr.getKind()) {
  case AffineExprKind::SymbolId: {
    unsigned pos = cast<AffineSymbolExpr>(expr).getPosition();
    if (printValueName)
      printValueName(pos, /*isSymbol=*/true);
    else
      os << 's' << pos;
    return;
  }
  case AffineExprKind::DimId: {
    unsigned pos = cast<AffineDimExpr>(expr).getPosition();
    if (printValueName)
      printValueName(pos, /*isSymbol=*/false);
    else
      os << 'd' << pos;
    return;
  }
  case AffineExprKind::Constant:
    os << cast<AffineConstantExpr>(expr).getValue();
    return;
  case AffineExprKind::Add:
    binopSpelling = " + ";
    break;
  case AffineExprKind::Mul:
    binopSpelling = " * ";
    break;
  case AffineExprKind::FloorDiv:
    binopSpelling = " floordiv ";
    break;
  case AffineExprKind::CeilDiv:
    binopSpelling = " ceildiv ";
    break;
  case AffineExprKind::Mod:
    binopSpelling = " mod ";
    break;
  }

  auto binOp = cast<AffineBinaryOpExpr>(expr);
  AffineExpr lhsExpr = binOp.getLHS();
  AffineExpr rhsExpr = binOp.getRHS();

  // Handle tightly binding binary operators.
  if (binOp.getKind() != AffineExprKind::Add) {
    if (enclosingTightness == BindingStrength::Strong)
      os << '(';

    // Pretty print multiplication with -1.
    auto rhsConst = dyn_cast<AffineConstantExpr>(rhsExpr);
    if (rhsConst && binOp.getKind() == AffineExprKind::Mul &&
        rhsConst.getValue() == -1) {
      os << "-";
      printAffineExprInternal(os, lhsExpr, BindingStrength::Strong,
                              printValueName);
      if (enclosingTightness == BindingStrength::Strong)
        os << ')';
      return;
    }

    printAffineExprInternal(os, lhsExpr, BindingStrength::Strong,
                            printValueName);

    os << binopSpelling;
    printAffineExprInternal(os, rhsExpr, BindingStrength::Strong,
                            printValueName);

    if (enclosingTightness == BindingStrength::Strong)
      os << ')';
    return;
  }

  // Print out special "pretty" forms for add.
  if (enclosingTightness == BindingStrength::Strong)
    os << '(';

  // Pretty print addition to a product that has a negative operand as a
  // subtraction.
  if (auto rhs = dyn_cast<AffineBinaryOpExpr>(rhsExpr)) {
    if (rhs.getKind() == AffineExprKind::Mul) {
      AffineExpr rrhsExpr = rhs.getRHS();
      if (auto rrhs = dyn_cast<AffineConstantExpr>(rrhsExpr)) {
        if (rrhs.getValue() == -1) {
          printAffineExprInternal(os, lhsExpr, BindingStrength::Weak,
                                  printValueName);
          os << " - ";
          if (rhs.getLHS().getKind() == AffineExprKind::Add) {
            printAffineExprInternal(os, rhs.getLHS(), BindingStrength::Strong,
                                    printValueName);
          } else {
            printAffineExprInternal(os, rhs.getLHS(), BindingStrength::Weak,
                                    printValueName);
          }

          if (enclosingTightness == BindingStrength::Strong)
            os << ')';
          return;
        }

        if (rrhs.getValue() < -1) {
          printAffineExprInternal(os, lhsExpr, BindingStrength::Weak,
                                  printValueName);
          os << " - ";
          printAffineExprInternal(os, rhs.getLHS(), BindingStrength::Strong,
                                  printValueName);
          os << " * " << -rrhs.getValue();
          if (enclosingTightness == BindingStrength::Strong)
            os << ')';
          return;
        }
      }
    }
  }

  // Pretty print addition to a negative number as a subtraction.
  if (auto rhsConst = dyn_cast<AffineConstantExpr>(rhsExpr)) {
    if (rhsConst.getValue() < 0) {
      printAffineExprInternal(os, lhsExpr, BindingStrength::Weak,
                              printValueName);
      os << " - " << -rhsConst.getValue();
      if (enclosingTightness == BindingStrength::Strong)
        os << ')';
      return;
    }
  }

  printAffineExprInternal(os, lhsExpr, BindingStrength::Weak, printValueName);

  os << " + ";
  printAffineExprInternal(os, rhsExpr, BindingStrength::Weak, printValueName);

  if (enclosingTightness == BindingStrength::Strong)
    os << ')';
}

static void
printAffineExpr(mlir::AsmPrinter &os, mlir::AffineExpr expr,
                llvm::function_ref<void(unsigned, bool)> printValueName) {
  printAffineExprInternal(os, expr, BindingStrength::Weak, printValueName);
}

static void printSymbolicShape(mlir::AsmPrinter &printer,
                               mlir::ArrayRef<mlir::Type> shape) {
  printer << "<";
  llvm::interleave(
      shape, printer,
      [&](mlir::Type t) {
        if (auto sym = mlir::dyn_cast<hc::typing::SymbolType>(t)) {
          printer << sym.getName();
          return;
        }
        if (auto lit = mlir::dyn_cast<hc::typing::LiteralType>(t)) {
          printer << mlir::cast<mlir::IntegerAttr>(lit.getValue()).getInt();
          return;
        }
        auto expr = mlir::cast<hc::typing::ExprType>(t);
        auto params = expr.getParams();
        auto printSym = [&](unsigned pos, bool isSymbol) {
          (void)isSymbol;
          auto sym = mlir::cast<hc::typing::SymbolType>(params[pos]);
          printer << sym.getName();
        };
        printer << "(";
        printAffineExpr(printer, expr.getExpr(), printSym);
        printer << ")";
      },
      " x ");
  printer << ">";
}

#include "hc/Dialect/HKernel/IR/HKernelOpsDialect.cpp.inc"

#include "hc/Dialect/HKernel/IR/HKernelOpsTypeInterfaces.cpp.inc"

#define GET_OP_CLASSES
#include "hc/Dialect/HKernel/IR/HKernelOps.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "hc/Dialect/HKernel/IR/HKernelOpsAttributes.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "hc/Dialect/HKernel/IR/HKernelOpsTypes.cpp.inc"

#include "hc/Dialect/HKernel/IR/HKernelOpsEnums.cpp.inc"
