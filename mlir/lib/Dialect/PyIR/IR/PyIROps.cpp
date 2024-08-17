// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hc/Dialect/PyIR/IR/PyIROps.hpp"

#include "hc/Dialect/Typing/IR/TypingOps.hpp"

#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/TypeUtilities.h>

#include <llvm/ADT/TypeSwitch.h>

void hc::py_ir::PyIRDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "hc/Dialect/PyIR/IR/PyIROps.cpp.inc"
      >();

  addTypes<
#define GET_TYPEDEF_LIST
#include "hc/Dialect/PyIR/IR/PyIROpsTypes.cpp.inc"
      >();

  addAttributes<
#define GET_ATTRDEF_LIST
#include "hc/Dialect/PyIR/IR/PyIROpsAttributes.cpp.inc"
      >();
}

void hc::py_ir::LoadModuleOp::getTypingKeyArgs(
    llvm::SmallVectorImpl<mlir::Attribute> &args) {
  args.emplace_back(getNameAttr());
}

namespace {
struct CleanupUnusedCaptures final
    : public mlir::OpRewritePattern<hc::py_ir::PyFuncOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(hc::py_ir::PyFuncOp op,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::ValueRange capturesArgs = op.getCaptureArgs();
    mlir::ValueRange capturesBlockArgs = op.getCaptureBlockArgs();
    mlir::SmallVector<unsigned> indexes;
    mlir::SmallVector<mlir::Operation *> constOps;
    for (int i = capturesBlockArgs.size() - 1; i >= 0; --i) {
      if (capturesBlockArgs[i].use_empty()) {
        indexes.emplace_back(i);
        constOps.emplace_back(nullptr);
        continue;
      }

      mlir::Operation *def = capturesArgs[i].getDefiningOp();
      if (def && def->getNumOperands() == 0 && def->getNumResults() == 1 &&
          def->hasTrait<mlir::OpTrait::ConstantLike>()) {
        indexes.emplace_back(i);
        constOps.emplace_back(def);
        continue;
      }
    }

    if (indexes.empty())
      return mlir::failure();

    auto *entryBlock = op.getEntryBlock();
    mlir::OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointToStart(entryBlock);

    rewriter.modifyOpInPlace(op, [&] {
      auto allArgs = entryBlock->getArguments().size();
      auto captNames =
          llvm::to_vector(op.getCaptureNames().getAsRange<mlir::Attribute>());

      unsigned numArgs = op.getBlockArgs().size();
      mlir::BitVector toErase(allArgs);
      for (auto &&[i, idx] : llvm::enumerate(indexes)) {
        int blockArgIdx = idx + numArgs;
        if (auto def = constOps[i]) {
          mlir::Operation *newOp = rewriter.clone(*def);
          rewriter.replaceAllUsesWith(entryBlock->getArgument(blockArgIdx),
                                      newOp->getResult(0));
        }
        op.getCaptureArgsMutable().erase(idx);
        toErase.set(blockArgIdx);
        captNames.erase(captNames.begin() + idx);
      }

      auto attr =
          rewriter.getArrayAttr(mlir::ArrayRef<mlir::Attribute>(captNames));
      op.setCaptureNamesAttr(attr);
      entryBlock->eraseArguments(toErase);
    });

    return mlir::success();
  }
};

struct PullCastCapture final
    : public mlir::OpRewritePattern<hc::py_ir::PyFuncOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(hc::py_ir::PyFuncOp op,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::ValueRange captures = op.getCaptureArgs();
    if (captures.empty())
      return mlir::failure();

    mlir::ValueRange blockArgs = op.getCaptureBlockArgs();
    bool needPull = false;
    llvm::SmallBitVector pull(captures.size(), false);
    for (auto &&[i, capt, arg] : llvm::enumerate(captures, blockArgs)) {
      auto cast = capt.getDefiningOp<mlir::CastOpInterface>();
      if (!arg.use_empty() && cast && mlir::isPure(cast) &&
          cast->getNumOperands() == 1 && cast->getNumResults() == 1) {
        pull[i] = true;
        needPull = true;
      }
    }

    if (!needPull)
      return mlir::failure();

    mlir::OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointToStart(op.getEntryBlock());

    mlir::IRMapping mapper;
    rewriter.modifyOpInPlace(op, [&]() {
      for (auto &&[i, capt, arg] : llvm::enumerate(captures, blockArgs)) {
        if (!pull[i])
          continue;

        mlir::Operation *cast = capt.getDefiningOp();
        mlir::Value newArg = cast->getOperand(0);
        mapper.map(newArg, arg);
        mlir::Type newType = newArg.getType();
        rewriter.replaceUsesWithIf(capt, newArg, [&](mlir::OpOperand &o) {
          return o.getOwner() == op;
        });
        mlir::Operation *cloned = rewriter.clone(*cast, mapper);
        rewriter.replaceAllUsesExcept(arg, cloned->getResult(0), cloned);
        arg.setType(newType);
      }
    });

    return mlir::success();
  }
};

} // namespace

mlir::FailureOr<bool>
hc::py_ir::NoneOp::inferTypes(mlir::TypeRange types,
                              llvm::SmallVectorImpl<mlir::Type> &results) {
  if (!types.empty())
    return emitError("Invalid arg count");

  results.emplace_back(mlir::NoneType::get(getContext()));
  return true;
}

mlir::OpFoldResult hc::py_ir::ConstantOp::fold(FoldAdaptor /*adaptor*/) {
  return getValue();
}

mlir::FailureOr<bool>
hc::py_ir::ConstantOp::inferTypes(mlir::TypeRange types,
                                  llvm::SmallVectorImpl<mlir::Type> &results) {
  if (!types.empty())
    return emitError("Invalid arg count");

  results.emplace_back(hc::typing::LiteralType::get(this->getValue()));
  return true;
}

mlir::OpFoldResult hc::py_ir::SymbolConstantOp::fold(FoldAdaptor /*adaptor*/) {
  return getValueAttr();
}

void hc::py_ir::PyFuncOp::build(::mlir::OpBuilder &odsBuilder,
                                ::mlir::OperationState &odsState,
                                mlir::Type resultType, llvm::StringRef name,
                                llvm::ArrayRef<llvm::StringRef> argNames,
                                mlir::ValueRange annotations,
                                llvm::ArrayRef<llvm::StringRef> captureNames,
                                mlir::ValueRange captureArgs,
                                mlir::ValueRange decorators) {
  assert(argNames.size() == annotations.size());
  assert(captureNames.size() == captureArgs.size());
  odsState.addAttribute(getNameAttrName(odsState.name),
                        odsBuilder.getStringAttr(name));
  odsState.addAttribute(getArgNamesAttrName(odsState.name),
                        odsBuilder.getStrArrayAttr(argNames));
  odsState.addAttribute(getCaptureNamesAttrName(odsState.name),
                        odsBuilder.getStrArrayAttr(captureNames));
  odsState.addOperands(annotations);
  odsState.addOperands(captureArgs);
  odsState.addOperands(decorators);
  odsState.addTypes(resultType);

  int32_t segmentSizes[3] = {};
  segmentSizes[0] = static_cast<int32_t>(annotations.size());
  segmentSizes[1] = static_cast<int32_t>(captureArgs.size());
  segmentSizes[2] = static_cast<int32_t>(decorators.size());
  odsState.addAttribute(getOperandSegmentSizeAttr(),
                        odsBuilder.getDenseI32ArrayAttr(segmentSizes));

  mlir::Region *region = odsState.addRegion();

  mlir::OpBuilder::InsertionGuard g(odsBuilder);

  auto numArgs = argNames.size() + captureNames.size();
  llvm::SmallVector<mlir::Type> types(
      numArgs, UndefinedType::get(odsBuilder.getContext()));
  llvm::SmallVector<mlir::Location> locs(numArgs, odsBuilder.getUnknownLoc());
  odsBuilder.createBlock(region, {}, types, locs);
}

void hc::py_ir::PyFuncOp::getCanonicalizationPatterns(
    ::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context) {
  results.insert<CleanupUnusedCaptures, PullCastCapture>(context);
}

void hc::py_ir::PyStaticFuncOp::build(::mlir::OpBuilder &odsBuilder,
                                      ::mlir::OperationState &odsState,
                                      llvm::StringRef symName,
                                      mlir::FunctionType functionType,
                                      llvm::ArrayRef<llvm::StringRef> argNames,
                                      mlir::ValueRange annotations) {
  assert(argNames.size() == annotations.size());
  odsState.addAttribute(mlir::SymbolTable::getSymbolAttrName(),
                        odsBuilder.getStringAttr(symName));
  odsState.addAttribute(getFunctionTypeAttrName(odsState.name),
                        mlir::TypeAttr::get(functionType));
  odsState.addAttribute(getArgNamesAttrName(odsState.name),
                        odsBuilder.getStrArrayAttr(argNames));
  odsState.addOperands(annotations);

  mlir::Region *region = odsState.addRegion();

  mlir::OpBuilder::InsertionGuard g(odsBuilder);

  auto numArgs = argNames.size();
  llvm::SmallVector<mlir::Type> types(
      numArgs, UndefinedType::get(odsBuilder.getContext()));
  llvm::SmallVector<mlir::Location> locs(numArgs, odsBuilder.getUnknownLoc());
  odsBuilder.createBlock(region, {}, types, locs);
}

namespace {
struct MakeStaticCall final : public mlir::OpRewritePattern<hc::py_ir::CallOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(hc::py_ir::CallOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto def = op.getFunc().getDefiningOp<hc::py_ir::SymbolConstantOp>();
    if (!def)
      return mlir::failure();

    auto mod = op->getParentOfType<hc::py_ir::PyModuleOp>();
    if (!mod)
      return mlir::failure();

    auto fn = mod.lookupSymbol<hc::py_ir::PyStaticFuncOp>(def.getValueAttr());
    if (!fn)
      return mlir::failure();

    mlir::TypeRange resTypes = fn.getFunctionType().getResults();
    mlir::Location loc = op.getLoc();
    mlir::Value newRes = rewriter.create<hc::py_ir::StaticCallOp>(
        loc, resTypes, def.getValueAttr(), op.getArgs(), op.getArgsNames());

    mlir::Type newType = newRes.getType();
    if (newType != op.getType())
      newRes = rewriter.create<hc::typing::CastOp>(loc, newType, newRes);

    rewriter.replaceOp(op, newRes);
    return mlir::success();
  }
};
} // namespace

void hc::py_ir::CallOp::getCanonicalizationPatterns(
    ::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context) {
  results.insert<MakeStaticCall>(context);
}

mlir::FailureOr<mlir::Value>
hc::py_ir::CallOp::getNamedArg(mlir::StringRef name) {
  for (auto &&[argName, arg] : llvm::zip_equal(
           getArgsNames().getAsValueRange<mlir::StringAttr>(), getArgs())) {
    if (name == argName)
      return arg;
  }

  return mlir::Value{};
}

mlir::LogicalResult hc::py_ir::StaticCallOp::verifySymbolUses(
    mlir::SymbolTableCollection &symbolTable) {
  // Check that the callee attribute was specified.
  auto fnAttr = (*this)->getAttrOfType<mlir::FlatSymbolRefAttr>("callee");
  if (!fnAttr)
    return emitOpError("requires a 'callee' symbol reference attribute");
  auto fn = symbolTable.lookupNearestSymbolFrom<PyStaticFuncOp>(*this, fnAttr);
  if (!fn)
    return emitOpError() << "'" << fnAttr.getValue()
                         << "' does not reference a valid function";

  // Verify that the operand and result types match the callee.
  auto fnType = fn.getFunctionType();
  if (fnType.getNumInputs() != getNumOperands())
    return emitOpError("incorrect number of operands for callee");

  // TODO: args can be in different order due to named args
  //  for (unsigned i = 0, e = fnType.getNumInputs(); i != e; ++i)
  //    if (getOperand(i).getType() != fnType.getInput(i))
  //      return emitOpError("operand type mismatch: expected operand type ")
  //             << fnType.getInput(i) << ", but provided "
  //             << getOperand(i).getType() << " for operand number " << i;

  //  mlir::Type t = getResult().getType();
  //  mlir::TypeRange resultTypes = t;
  //  if (fnType.getNumResults() != resultTypes.size())
  //    return emitOpError("incorrect number of results for callee");

  //  for (unsigned i = 0, e = fnType.getNumResults(); i != e; ++i)
  //    if (resultTypes[i] != fnType.getResult(i)) {
  //      auto diag = emitOpError("result type mismatch at index ") << i;
  //      diag.attachNote() << "      op result types: " << resultTypes;
  //      diag.attachNote() << "function result types: " << fnType.getResults();
  //      return diag;
  //    }

  return mlir::success();
}

mlir::FunctionType hc::py_ir::StaticCallOp::getCalleeType() {
  return mlir::FunctionType::get(getContext(), getOperandTypes(),
                                 getResult().getType());
}

void hc::py_ir::LoadVarOp::getTypingKeyArgs(
    llvm::SmallVectorImpl<mlir::Attribute> &args) {
  args.emplace_back(getNameAttr());
}

void hc::py_ir::GetAttrOp::getTypingKeyArgs(
    llvm::SmallVectorImpl<mlir::Attribute> &args) {
  args.emplace_back(getNameAttr());
}

mlir::FailureOr<mlir::Value>
hc::py_ir::SliceOp::getNamedArg(mlir::StringRef name) {
  if (name == "lower")
    return getLower();

  if (name == "upper")
    return getUpper();

  if (name == "step")
    return getStep();

  return mlir::failure();
}

static bool parseArgList(
    mlir::OpAsmParser &parser,
    llvm::SmallVectorImpl<mlir::OpAsmParser::UnresolvedOperand> &argsOperands,
    mlir::ArrayAttr &args_namesAttr) {
  if (parser.parseLParen())
    return true;

  auto *context = parser.getContext();
  llvm::SmallVector<mlir::Attribute> names;
  if (parser.parseOptionalRParen()) {
    std::string name;
    while (true) {
      name.clear();
      if (!parser.parseOptionalKeywordOrString(&name)) {
        if (parser.parseColon())
          return true;
      }
      names.push_back(mlir::StringAttr::get(context, name));

      argsOperands.push_back({});
      if (parser.parseOperand(argsOperands.back()))
        return true;

      if (!parser.parseOptionalRParen())
        break;

      if (parser.parseComma())
        return true;
    }
  }

  assert(names.size() == argsOperands.size());
  args_namesAttr = mlir::ArrayAttr::get(context, names);
  return false;
}

template <typename Op>
static void printArgList(mlir::OpAsmPrinter &printer, Op /*op*/,
                         mlir::ValueRange args, mlir::ArrayAttr argsNames) {
  assert(args.size() == argsNames.size());
  printer << '(';
  bool first = true;
  for (auto &&[arg, name] : llvm::zip(args, argsNames)) {
    if (first) {
      first = false;
    } else {
      printer << ", ";
    }
    auto nameStr = (name ? mlir::cast<mlir::StringAttr>(name).getValue()
                         : llvm::StringRef());
    if (!nameStr.empty())
      printer << nameStr << ':';
    printer.printOperand(arg);
  }
  printer << ')';
}

#include "hc/Dialect/PyIR/IR/PyIROpsDialect.cpp.inc"

#define GET_OP_CLASSES
#include "hc/Dialect/PyIR/IR/PyIROps.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "hc/Dialect/PyIR/IR/PyIROpsAttributes.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "hc/Dialect/PyIR/IR/PyIROpsTypes.cpp.inc"

#include "hc/Dialect/PyIR/IR/PyIROpsEnums.cpp.inc"
