// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "TypingPipeline.hpp"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <mlir/Transforms/Passes.h>

#include "hc/Dialect/PyIR/IR/PyIROps.hpp"
#include "hc/Dialect/Typing/IR/TypingOps.hpp"
#include "hc/Transforms/Passes.hpp"

static mlir::Value doCast(mlir::OpBuilder &builder, mlir::Location loc,
                          mlir::Value val, mlir::Type type) {
  if (val.getType() == type)
    return val;

  return builder.create<hc::typing::ValueCastOp>(loc, type, val);
}

template <typename Op, int NumArgs, bool NoRes = false>
static mlir::Operation *
convertCall(mlir::OpBuilder &builder, mlir::Location loc, mlir::ValueRange args,
            mlir::Type resType, mlir::TypeRange expectedArgTypes) {
  static_assert(NumArgs == 0 || NumArgs == 1 || NumArgs == 2 || NumArgs == 3);
  assert(args.size() == NumArgs);
  assert(expectedArgTypes.size() == NumArgs);
  auto doCast = [&](mlir::Value val, mlir::Type type) -> mlir::Value {
    return ::doCast(builder, loc, val, type);
  };
  auto build = [&](auto &&...args) -> mlir::Operation * {
    if constexpr (NoRes) {
      return builder.create<Op>(loc, args...);
    } else {
      return builder.create<Op>(loc, resType, args...);
    }
  };

  if constexpr (NumArgs == 0) {
    return builder.create<Op>(loc, resType);
  } else if constexpr (NumArgs == 1) {
    auto arg0 = doCast(args[0], expectedArgTypes[0]);
    return build(arg0);
  } else if constexpr (NumArgs == 2) {
    auto arg0 = doCast(args[0], expectedArgTypes[0]);
    auto arg1 = doCast(args[1], expectedArgTypes[1]);
    return build(arg0, arg1);
  } else if constexpr (NumArgs == 3) {
    auto arg0 = doCast(args[0], expectedArgTypes[0]);
    auto arg1 = doCast(args[1], expectedArgTypes[1]);
    auto arg2 = doCast(args[2], expectedArgTypes[2]);
    return build(arg0, arg1, arg2);
  }
  llvm_unreachable("Unreachable");
}

static mlir::LogicalResult convertCall(mlir::PatternRewriter &builder,
                                       hc::py_ir::CallOp call,
                                       llvm::StringRef name) {
  builder.setInsertionPoint(call);
  mlir::Type callResType = call.getResult().getType();
  mlir::ValueRange args = call.getArgs();
  mlir::TypeRange callArgTypes = args.getTypes();

  mlir::Location loc = call.getLoc();
  auto doCast = [&](mlir::Value val, mlir::Type type) -> mlir::Value {
    return ::doCast(builder, loc, val, type);
  };

  if (name == "to_int" && args.size() == 1) {
    builder.replaceOp(call, doCast(args[0], callResType));
    return mlir::success();
  }

  auto checkFuncName = [&](llvm::StringRef funcName) -> bool {
    return funcName == name;
  };

  auto index = builder.getIndexType();
  auto i1 = builder.getIntegerType(1);
  auto none = builder.getNoneType();
  auto vt = hc::typing::ValueType::get(builder.getContext());
  using fptr =
      mlir::Operation *(*)(mlir::OpBuilder & builder, mlir::Location loc,
                           mlir::ValueRange args, mlir::Type resType,
                           mlir::TypeRange expectedArgTypes);

  const std::tuple<llvm::StringRef, mlir::Type,
                   llvm::SmallVector<mlir::Type, 3>, fptr>
      handlers[] = {
          {"is_same", i1, {vt, vt}, &convertCall<hc::typing::IsSameOp, 2>},
          {"check", none, {i1}, &convertCall<hc::typing::CheckOp, 1, true>},
          {"make_symbol", vt, {vt}, &convertCall<hc::typing::MakeSymbolOp, 1>},
          {"get_num_args",
           index,
           {},
           &convertCall<hc::typing::GetNumArgsOp, 0>},
          {"get_arg", vt, {index}, &convertCall<hc::typing::GetArgOp, 1>},
          {"create_seq", vt, {}, &convertCall<hc::typing::CreateSeqOp, 0>},
          {"append_seq",
           vt,
           {vt, vt},
           &convertCall<hc::typing::AppendSeqOp, 2>},
          {"get_seq_element",
           vt,
           {vt, index},
           &convertCall<hc::typing::GetSeqElementOp, 2>},
          {"get_seq_size",
           index,
           {vt},
           &convertCall<hc::typing::GetSeqSizeOp, 1>},
          {"get_type_name",
           vt,
           {vt},
           &convertCall<hc::typing::GetIdentNameOp, 1>},
      };

  for (auto &&[funcName, resType, argTypes, handler] : handlers) {
    if (!checkFuncName(funcName))
      continue;

    mlir::Operation *resOp = handler(builder, loc, args, resType, argTypes);
    if (resOp->getNumResults() == 0) {
      builder.replaceOpWithNewOp<hc::py_ir::NoneOp>(call);
    } else {
      assert(resOp->getNumResults() == 1);
      mlir::Value res = resOp->getResult(0);
      if (res.getType() != callResType)
        res = doCast(res, callResType);

      builder.replaceOp(call, res);
    }
    return mlir::success();
  }

  auto isStrLiteralArg = [&](llvm::StringRef funcName, mlir::Type resType,
                             mlir::TypeRange argTypes) -> mlir::StringAttr {
    if (callArgTypes.size() != argTypes.size() + 1)
      return nullptr;

    if (!checkFuncName(funcName))
      return nullptr;

    if (callResType != resType)
      return nullptr;

    if (!llvm::equal(callArgTypes.drop_back(), argTypes))
      return nullptr;

    auto literal = mlir::dyn_cast<hc::typing::LiteralType>(callArgTypes.back());
    if (!literal)
      return nullptr;

    return mlir::dyn_cast<mlir::StringAttr>(literal.getValue());
  };

  if (auto name = isStrLiteralArg("get_attr", vt, {})) {
    builder.replaceOpWithNewOp<hc::typing::GetAttrOp>(call, callResType, name);
    return mlir::success();
  }
  if (auto name = isStrLiteralArg("get_type_param", vt, {vt})) {
    builder.replaceOpWithNewOp<hc::typing::GetIdentParamOp>(call, callResType,
                                                            args[0], name);
    return mlir::success();
  }
  if (auto name = isStrLiteralArg("get_global_attr", vt, {})) {
    builder.replaceOpWithNewOp<hc::typing::GetGlobalAttrOp>(call, callResType,
                                                            name);
    return mlir::success();
  }
  if (auto name = isStrLiteralArg("get_named_arg", vt, {})) {
    builder.replaceOpWithNewOp<hc::typing::GetNamedArgOp>(call, callResType,
                                                          name);
    return mlir::success();
  }

  auto matchMakeType = [&]() -> bool {
    if (!checkFuncName("make_type"))
      return false;

    if (callResType != vt)
      return false;

    auto literal =
        mlir::dyn_cast<hc::typing::LiteralType>(callArgTypes.front());
    if (!literal)
      return false;

    auto nameAttr = mlir::dyn_cast<mlir::StringAttr>(literal.getValue());
    if (!nameAttr)
      return false;

    auto names = llvm::to_vector(
        call.getArgsNames().getAsValueRange<mlir::StringAttr>());
    if (names.empty())
      return false;

    if (!names[0].empty())
      return false;

    names.erase(names.begin());

    auto args = call.getArgs().drop_front();
    assert(args.size() == names.size());

    auto namesAttr = builder.getStrArrayAttr(names);
    builder.replaceOpWithNewOp<hc::typing::MakeIdentOp>(
        call, callResType, nameAttr, namesAttr, args);
    return true;
  };
  if (matchMakeType())
    return mlir::success();

  return mlir::failure();
}

namespace {
class ConvertCall final : public mlir::OpRewritePattern<hc::py_ir::CallOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(hc::py_ir::CallOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto funcType =
        mlir::dyn_cast<hc::typing::IdentType>(op.getFunc().getType());
    if (!funcType)
      return mlir::failure();

    llvm::StringRef funcName = funcType.getName().getValue();
    if (!funcName.consume_front("hckernel.typing."))
      return mlir::failure();

    return convertCall(rewriter, op, funcName);
  }
};

static mlir::Value doCast(mlir::OpBuilder &builder, mlir::Location loc,
                          mlir::Value val, mlir::Type dstType) {
  mlir::Type srcType = val.getType();
  if (srcType == dstType)
    return val;

  if (srcType.isIntOrIndex() && dstType.isIntOrIndex()) {
    return builder.create<mlir::arith::IndexCastOp>(loc, dstType, val);
  }

  return builder.create<hc::typing::CastOp>(loc, dstType, val);
}

template <typename Op>
static mlir::Value createBinOp(mlir::OpBuilder &builder, mlir::Location loc,
                               mlir::Value lhs, mlir::Value rhs,
                               mlir::Type dstType) {
  lhs = doCast(builder, loc, lhs, dstType);
  rhs = doCast(builder, loc, rhs, dstType);
  return builder.create<Op>(loc, lhs, rhs);
}

template <typename BinOp>
class ConvertBinop final : public mlir::OpRewritePattern<BinOp> {
public:
  using mlir::OpRewritePattern<BinOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(BinOp op, mlir::PatternRewriter &rewriter) const override {
    auto retType = op.getType();
    if (!retType.isIntOrIndex())
      return mlir::failure();

    using fptr =
        mlir::Value (*)(mlir::OpBuilder & builder, mlir::Location loc,
                        mlir::Value lhs, mlir::Value rhs, mlir::Type dstType);
    using Op = hc::py_ir::BinOpVal;
    namespace arith = mlir::arith;
    const std::pair<Op, fptr> handlers[] = {
        {Op::add, createBinOp<arith::AddIOp>},
        {Op::sub, createBinOp<arith::SubIOp>},
        {Op::mul, createBinOp<arith::MulIOp>},
        {Op::bool_and, createBinOp<arith::AndIOp>},
        {Op::bool_or, createBinOp<arith::OrIOp>},
        {Op::bit_and, createBinOp<arith::AndIOp>},
        {Op::bit_or, createBinOp<arith::OrIOp>},
    };

    auto opType = op.getOp();
    for (auto &&[type, handler] : handlers) {
      if (type == opType) {
        mlir::Value res = handler(rewriter, op.getLoc(), op.getLeft(),
                                  op.getRight(), retType);
        rewriter.replaceOp(op, res);
        return mlir::success();
      }
    }
    return mlir::failure();
  }
};

template <mlir::arith::CmpIPredicate Pred>
static mlir::Value createCmpOp(mlir::OpBuilder &builder, mlir::Location loc,
                               mlir::Value lhs, mlir::Value rhs,
                               mlir::Type dstType) {
  lhs = doCast(builder, loc, lhs, dstType);
  rhs = doCast(builder, loc, rhs, dstType);
  return builder.create<mlir::arith::CmpIOp>(loc, Pred, lhs, rhs);
}

static bool isIntType(mlir::Type type) {
  if (type.isIntOrIndex())
    return true;

  auto lit = mlir::dyn_cast<hc::typing::LiteralType>(type);
  return lit && mlir::isa<mlir::IntegerAttr>(lit.getValue());
}

class ConvertCmp final : public mlir::OpRewritePattern<hc::py_ir::CmpOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(hc::py_ir::CmpOp op,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::Value lhs = op.getLeft();
    mlir::Value rhs = op.getRight();
    if (!isIntType(lhs.getType()) || !isIntType(rhs.getType()))
      return mlir::failure();

    mlir::Type resType = rewriter.getIndexType();

    using fptr =
        mlir::Value (*)(mlir::OpBuilder & builder, mlir::Location loc,
                        mlir::Value lhs, mlir::Value rhs, mlir::Type dstType);
    using Op = hc::py_ir::CmpOpVal;
    using pred = mlir::arith::CmpIPredicate;
    const std::pair<Op, fptr> handlers[] = {
        {Op::le, createCmpOp<pred::sle>}, {Op::lt, createCmpOp<pred::slt>},
        {Op::ge, createCmpOp<pred::sge>}, {Op::gt, createCmpOp<pred::sgt>},
        {Op::eq, createCmpOp<pred::eq>},  {Op::ne, createCmpOp<pred::ne>},
    };

    auto opType = op.getOp();
    for (auto &&[type, handler] : handlers) {
      if (type == opType) {
        mlir::Value res = handler(rewriter, op.getLoc(), lhs, rhs, resType);
        rewriter.replaceOp(op, res);
        return mlir::success();
      }
    }
    return mlir::failure();
  }
};

class ConvertConst final
    : public mlir::OpRewritePattern<hc::py_ir::ConstantOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(hc::py_ir::ConstantOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto val = mlir::dyn_cast<mlir::IntegerAttr>(op.getValue());
    if (!val)
      return mlir::failure();

    mlir::Operation *opToReplace = op;
    mlir::OpBuilder::InsertionGuard g(rewriter);
    if (auto resolve =
            mlir::dyn_cast<hc::typing::ResolveOp>(op->getParentOp())) {
      rewriter.setInsertionPoint(resolve);
      opToReplace = resolve;
    }

    mlir::Type resType = opToReplace->getResult(0).getType();

    mlir::Location loc = op.getLoc();
    mlir::Value newVal =
        rewriter.create<mlir::arith::ConstantIndexOp>(loc, val.getInt());
    newVal = doCast(rewriter, loc, newVal, resType);
    rewriter.replaceOp(opToReplace, newVal);
    return mlir::success();
  }
};

struct GenResolversFuncsPass
    : public mlir::PassWrapper<GenResolversFuncsPass,
                               mlir::OperationPass<void>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GenResolversFuncsPass)

  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<hc::typing::TypingDialect>();
    registry.insert<mlir::arith::ArithDialect>();
  }

  void runOnOperation() override {
    auto *ctx = &getContext();
    mlir::RewritePatternSet patterns(ctx);
    patterns.insert<ConvertCall, ConvertBinop<hc::py_ir::BinOp>,
                    ConvertBinop<hc::py_ir::InplaceBinOp>, ConvertCmp,
                    ConvertConst>(ctx);

    if (mlir::failed(
            applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
  }
};

struct DropFuncDecoratorPass
    : public mlir::PassWrapper<DropFuncDecoratorPass,
                               mlir::OperationPass<void>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DropFuncDecoratorPass)

  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<hc::typing::TypingDialect>();
    registry.insert<mlir::arith::ArithDialect>();
  }

  void runOnOperation() override {
    auto type =
        hc::typing::IdentType::get(&getContext(), "hckernel.typing.func");
    llvm::SmallVector<mlir::Value> decorators;
    auto visitor = [&](hc::py_ir::PyFuncOp op) {
      auto filter = [&](mlir::Value val) -> bool {
        return val.getType() != type;
      };

      decorators.clear();
      llvm::append_range(decorators,
                         llvm::make_filter_range(op.getDecorators(), filter));
      op.getDecoratorsMutable().assign(decorators);
    };

    getOperation()->walk<mlir::WalkOrder::PostOrder>(visitor);
  }
};
} // namespace

static std::optional<mlir::ArrayAttr>
processDecorators(mlir::ValueRange decorators) {
  if (decorators.size() != 1)
    return std::nullopt;

  auto type =
      mlir::dyn_cast<hc::typing::IdentType>(decorators.front().getType());
  if (!type || type.getName() != "type_resolver_type" ||
      type.getParamNames().size() != 1 || type.getParamNames().front() != "key")
    return std::nullopt;

  auto seqType =
      mlir::dyn_cast<hc::typing::SequenceType>(type.getParams().front());
  if (!seqType)
    return std::nullopt;

  llvm::SmallVector<mlir::Attribute> ret;
  for (auto elem : seqType.getParams()) {
    auto lit = mlir::dyn_cast<hc::typing::LiteralType>(elem);
    if (!lit)
      return std::nullopt;

    ret.emplace_back(lit.getValue());
  }
  return mlir::ArrayAttr::get(type.getContext(), ret);
}

namespace {
struct GenResolversPass
    : public mlir::PassWrapper<GenResolversPass, mlir::OperationPass<void>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GenResolversPass)

  virtual void
  getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<hc::typing::TypingDialect>();
    registry.insert<mlir::arith::ArithDialect>();
  }

  void runOnOperation() override {
    mlir::IRRewriter builder(&getContext());
    auto visitor = [&](hc::py_ir::PyModuleOp mod) -> mlir::WalkResult {
      builder.setInsertionPoint(mod);
      for (mlir::Operation &nestedOp :
           llvm::make_early_inc_range(mod.getBody()->getOperations())) {
        auto func = mlir::dyn_cast<hc::py_ir::PyFuncOp>(nestedOp);
        if (!func) {
          if (auto funcLike =
                  mlir::dyn_cast<mlir::FunctionOpInterface>(nestedOp))
            funcLike->moveBefore(mod);

          continue;
        }

        if (!func.getCaptureArgs().empty()) {
          func->emitOpError("Cannot convert function with captures");
          return mlir::WalkResult::interrupt();
        }

        if (!llvm::all_of(func.getBlockArgs(), [](auto arg) {
              return mlir::isa<hc::typing::ValueType>(arg.getType());
            })) {
          func->emitOpError("Invalid arg types");
          return mlir::WalkResult::interrupt();
        }

        auto decorator = processDecorators(func.getDecorators());
        if (!decorator) {
          func->emitOpError("Invalid decorator");
          return mlir::WalkResult::interrupt();
        }

        auto funcVisitor = [&](mlir::Operation *op) -> mlir::WalkResult {
          if (!mlir::isa<hc::typing::TypingInterpreterInterface>(op) &&
              !mlir::isa<hc::py_ir::ReturnOp, hc::py_ir::PyFuncOp>(op)) {
            op->emitOpError("unsupported typing op");
            return mlir::WalkResult::interrupt();
          }
          return mlir::WalkResult::advance();
        };
        if (func.walk(funcVisitor).wasInterrupted())
          return mlir::WalkResult::interrupt();

        mlir::Location loc = func.getLoc();
        auto resolver =
            builder.create<hc::typing::TypeResolverOp>(loc, *decorator);
        mlir::Region &dstReg = resolver.getBodyRegion();
        builder.inlineRegionBefore(func.getBodyRegion(), dstReg,
                                   dstReg.begin());

        mlir::Block &entry = dstReg.front();
        mlir::OpBuilder::InsertionGuard g(builder);
        builder.setInsertionPointToStart(&entry);
        for (auto &&[i, arg] : llvm::enumerate(entry.getArguments())) {
          mlir::Value idx =
              builder.create<mlir::arith::ConstantIndexOp>(loc, i);
          mlir::Value newArg =
              builder.create<hc::typing::GetArgOp>(loc, arg.getType(), idx);
          arg.replaceAllUsesWith(newArg);
        }
        entry.eraseArguments(0, entry.getNumArguments());

        for (mlir::Block &block : dstReg) {
          auto term =
              mlir::dyn_cast<hc::py_ir::ReturnOp>(block.getTerminator());
          if (!term)
            continue;

          builder.setInsertionPoint(term);
          builder.replaceOpWithNewOp<hc::typing::TypeResolverReturnOp>(
              term, term.getOperand());
        }
      }

      mod->erase();
      return mlir::WalkResult::skip();
    };

    if (getOperation()
            ->walk<mlir::WalkOrder::PreOrder>(visitor)
            .wasInterrupted())
      return signalPassFailure();
  }
};
} // namespace

void populateTypingPipeline(mlir::PassManager &pm) {
  pm.addPass(mlir::createCompositeFixedPointPass(
      "TypingLoop", [](mlir::OpPassManager &p) {
        p.addPass(hc::createPyTypeInferencePass());
        p.addPass(mlir::createCanonicalizerPass());
        p.addPass(std::make_unique<GenResolversFuncsPass>());
        p.addPass(std::make_unique<DropFuncDecoratorPass>());
        p.addPass(hc::createPyIRPromoteFuncsToStaticPass());
      }));
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(hc::createPyIRPromoteFuncsToStaticPass());
  pm.addPass(hc::createPyTypeInferencePass());
  pm.addPass(hc::createDropTypeResolversPass());
  pm.addPass(hc::createConverPyFuncToFuncPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(std::make_unique<GenResolversPass>());
  pm.addPass(mlir::createCanonicalizerPass());
}
