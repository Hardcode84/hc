// SPDX-FileCopyrightText: 2024 The HC Authors
// SPDX-FileCopyrightText: 2025 The HC Authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hc/Dialect/Typing/IR/TypingOps.hpp"
#include "hc/Dialect/Typing/IR/TypingOpsInterfaces.hpp"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/TypeUtilities.h>

#include <llvm/ADT/TypeSwitch.h>

MLIR_DEFINE_EXPLICIT_TYPE_ID(hc::typing::SymbolicTypeBase)

namespace {
struct TypingAsmDialectInterface : public mlir::OpAsmDialectInterface {
  using OpAsmDialectInterface::OpAsmDialectInterface;

  AliasResult getAlias(mlir::Type type, llvm::raw_ostream &os) const final {
    if (llvm::isa<hc::typing::IdentType>(type)) {
      os << "ident";
      return AliasResult::OverridableAlias;
    }
    if (llvm::isa<hc::typing::SequenceType>(type)) {
      os << "seq";
      return AliasResult::OverridableAlias;
    }
    if (llvm::isa<hc::typing::SymbolType>(type)) {
      os << "sym";
      return AliasResult::OverridableAlias;
    }
    if (llvm::isa<hc::typing::LiteralType>(type)) {
      os << "literal";
      return AliasResult::OverridableAlias;
    }
    if (llvm::isa<hc::typing::UnionType>(type)) {
      os << "union";
      return AliasResult::OverridableAlias;
    }
    if (llvm::isa<hc::typing::ExprType>(type)) {
      os << "expr";
      return AliasResult::OverridableAlias;
    }
    return AliasResult::NoAlias;
  }
};
} // namespace

void hc::typing::TypingDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "hc/Dialect/Typing/IR/TypingOps.cpp.inc"
      >();

  addTypes<
#define GET_TYPEDEF_LIST
#include "hc/Dialect/Typing/IR/TypingOpsTypes.cpp.inc"
      >();

  addAttributes<
#define GET_ATTRDEF_LIST
#include "hc/Dialect/Typing/IR/TypingOpsAttributes.cpp.inc"
      >();

  addInterface<TypingAsmDialectInterface>();

  registerArithTypingInterpreter(*getContext());
}

mlir::Operation *hc::typing::TypingDialect::materializeConstant(
    mlir::OpBuilder &builder, mlir::Attribute value, mlir::Type type,
    mlir::Location loc) {
  auto typeAttr = mlir::dyn_cast<TypeAttr>(value);
  if (typeAttr && mlir::isa<ValueType>(type))
    return builder.create<TypeConstantOp>(loc, typeAttr);

  return nullptr;
}

mlir::OpFoldResult hc::typing::TypeConstantOp::fold(FoldAdaptor /*adaptor*/) {
  return getValue();
}

mlir::FailureOr<bool> hc::typing::TypeConstantOp::inferTypes(
    mlir::TypeRange types, llvm::SmallVectorImpl<mlir::Type> &results) {
  if (!types.empty())
    return emitError("Invalid arg count");

  results.emplace_back(ValueType::get(getContext()));
  return true;
}

mlir::Type hc::typing::IdentType::getParam(mlir::StringAttr paramName) const {
  for (auto &&[name, val] : llvm::zip_equal(getParamNames(), getParams())) {
    if (name == paramName)
      return val;
  }
  return nullptr;
}

void hc::typing::ResolveOp::build(::mlir::OpBuilder &odsBuilder,
                                  ::mlir::OperationState &odsState,
                                  mlir::TypeRange resultTypes,
                                  mlir::ValueRange args) {
  odsState.addOperands(args);
  odsState.addTypes(resultTypes);

  mlir::Region *region = odsState.addRegion();

  mlir::OpBuilder::InsertionGuard g(odsBuilder);

  llvm::SmallVector<mlir::Location> locs(args.size(),
                                         odsBuilder.getUnknownLoc());
  odsBuilder.createBlock(region, {}, mlir::TypeRange(args), locs);
}

namespace {
static mlir::Value makeCast(mlir::OpBuilder &builder, mlir::Location loc,
                            mlir::Value src, mlir::Type type) {
  if (src.getType() == type)
    return src;

  return builder.create<hc::typing::ValueCastOp>(loc, type, src);
}

struct ResolveSelect final
    : public mlir::OpRewritePattern<hc::typing::ResolveOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(hc::typing::ResolveOp op,
                  mlir::PatternRewriter &rewriter) const override {
    if (op->getNumResults() != 1)
      return mlir::failure();

    mlir::Block *body = op.getBody();
    if (!llvm::hasSingleElement(body->without_terminator()))
      return mlir::failure();

    auto select = mlir::dyn_cast<mlir::arith::SelectOp>(body->front());
    if (!select)
      return mlir::failure();

    mlir::Type resType = op.getResult(0).getType();
    auto getArg = [&](mlir::Value src) -> mlir::Value {
      auto idx = mlir::cast<mlir::BlockArgument>(src).getArgNumber();
      return op->getOperand(idx);
    };

    mlir::Location loc = select.getLoc();
    mlir::Value cond = getArg(select.getCondition());
    mlir::Value trueVal =
        makeCast(rewriter, loc, getArg(select.getTrueValue()), resType);
    mlir::Value falseVal =
        makeCast(rewriter, loc, getArg(select.getFalseValue()), resType);
    rewriter.replaceOpWithNewOp<mlir::arith::SelectOp>(op, cond, trueVal,
                                                       falseVal);
    return mlir::success();
  }
};
} // namespace

void hc::typing::ResolveOp::getCanonicalizationPatterns(
    ::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context) {
  results.insert<ResolveSelect>(context);
}

bool hc::typing::CastOp::areCastCompatible(mlir::TypeRange inputs,
                                           mlir::TypeRange outputs) {
  (void)inputs;
  (void)outputs;
  assert(inputs.size() == 1 && "expected one input");
  assert(outputs.size() == 1 && "expected one output");
  return true;
}

mlir::OpFoldResult hc::typing::CastOp::fold(FoldAdaptor /*adaptor*/) {
  mlir::Value arg = getValue();
  mlir::Type dstType = getType();
  if (arg.getType() == dstType)
    return arg;

  while (auto parent = arg.getDefiningOp<CastOp>()) {
    arg = parent.getValue();
    if (arg.getType() == dstType)
      return arg;
  }

  return nullptr;
}

mlir::FailureOr<bool>
hc::typing::CastOp::inferTypes(mlir::TypeRange types,
                               llvm::SmallVectorImpl<mlir::Type> &results) {
  if (types.size() != 1)
    return emitError("Invalid arg count");

  results.emplace_back(types.front());
  return true;
}

mlir::FailureOr<hc::typing::InterpreterResult>
hc::typing::CastOp::interpret(InterpreterState &state) {
  state.state[getResult()] = getVal(state, getValue());
  return InterpreterResult::Advance;
}

bool hc::typing::ValueCastOp::areCastCompatible(mlir::TypeRange inputs,
                                                mlir::TypeRange outputs) {
  (void)inputs;
  (void)outputs;
  assert(inputs.size() == 1 && "expected one input");
  assert(outputs.size() == 1 && "expected one output");
  return true;
}

mlir::OpFoldResult hc::typing::ValueCastOp::fold(FoldAdaptor /*adaptor*/) {
  mlir::Value arg = getValue();
  mlir::Type dstType = getType();
  if (arg.getType() == dstType)
    return arg;

  while (auto parent = arg.getDefiningOp<CastOp>()) {
    arg = parent.getValue();
    if (arg.getType() == dstType)
      return arg;
  }

  if (mlir::isa<ValueType>(getType())) {
    if (auto lit = mlir::dyn_cast<LiteralType>(getValue().getType())) {
      return TypeAttr::get(lit);
    }
  }

  return nullptr;
}

mlir::FailureOr<bool> hc::typing::ValueCastOp::inferTypes(
    mlir::TypeRange /*types*/, llvm::SmallVectorImpl<mlir::Type> &results) {
  results.emplace_back(getType());
  return true;
}

mlir::FailureOr<hc::typing::InterpreterResult>
hc::typing::ValueCastOp::interpret(InterpreterState &state) {
  mlir::Type dstType = getType();
  if (dstType.isIntOrIndex()) {
    auto val = getInt(state, getValue());
    if (!val)
      return emitOpError("Invalid src val");

    state.state[getResult()] = setInt(getContext(), *val);
    return InterpreterResult::Advance;
  }

  return emitOpError("Unsupported cast");
}

namespace {
using namespace hc::typing;

static mlir::LogicalResult jumpToBlock(mlir::Operation *op,
                                       InterpreterState &state,
                                       mlir::Block *newBlock,
                                       mlir::ValueRange args) {
  if (newBlock->getNumArguments() != args.size())
    return op->emitError("Block arg count mismatch");

  state.iter = newBlock->begin();

  // Make a temp copy so we won't overwrite values prematurely if we jump to the
  // same block.
  llvm::SmallVector<InterpreterValue> newValues(args.size());
  for (auto &&[i, arg] : llvm::enumerate(args))
    newValues[i] = state.state[arg];

  for (auto &&[i, arg] : llvm::enumerate(newBlock->getArguments()))
    state.state[arg] = newValues[i];

  return mlir::success();
};

struct BranchOpInterpreterInterface final
    : public hc::typing::TypingInterpreterInterface::ExternalModel<
          BranchOpInterpreterInterface, mlir::cf::BranchOp> {
  mlir::FailureOr<hc::typing::InterpreterResult>
  interpret(mlir::Operation *o, InterpreterState &state) const {
    auto op = mlir::cast<mlir::cf::BranchOp>(o);
    if (mlir::failed(
            jumpToBlock(op, state, op.getDest(), op.getDestOperands())))
      return mlir::failure();

    return InterpreterResult::Advance;
  }
};

struct CondBranchOpInterpreterInterface final
    : public hc::typing::TypingInterpreterInterface::ExternalModel<
          CondBranchOpInterpreterInterface, mlir::cf::CondBranchOp> {
  mlir::FailureOr<hc::typing::InterpreterResult>
  interpret(mlir::Operation *o, InterpreterState &state) const {
    auto op = mlir::cast<mlir::cf::CondBranchOp>(o);
    auto cond = getInt(state, op.getCondition());
    if (!cond)
      return op.emitError("Invalid cond val");

    mlir::Block *dest = (*cond ? op.getTrueDest() : op.getFalseDest());
    mlir::ValueRange destArgs =
        (*cond ? op.getTrueDestOperands() : op.getFalseDestOperands());

    if (mlir::failed(jumpToBlock(op, state, dest, destArgs)))
      return mlir::failure();

    return InterpreterResult::Advance;
  }
};

struct ConstantOpInterpreterInterface final
    : public hc::typing::TypingInterpreterInterface::ExternalModel<
          ConstantOpInterpreterInterface, mlir::arith::ConstantOp> {
  mlir::FailureOr<hc::typing::InterpreterResult>
  interpret(mlir::Operation *o, InterpreterState &state) const {
    auto op = mlir::cast<mlir::arith::ConstantOp>(o);
    auto attr = mlir::dyn_cast<mlir::IntegerAttr>(op.getValue());
    if (!attr)
      return op->emitError("Expected int attribute but got ") << op.getValue();

    state.state[op.getResult()] = setInt(op.getContext(), attr.getInt());
    return InterpreterResult::Advance;
  }
};

struct AddOpInterpreterInterface final
    : public hc::typing::TypingInterpreterInterface::ExternalModel<
          AddOpInterpreterInterface, mlir::arith::AddIOp> {
  mlir::FailureOr<hc::typing::InterpreterResult>
  interpret(mlir::Operation *o, InterpreterState &state) const {
    auto op = mlir::cast<mlir::arith::AddIOp>(o);
    auto lhs = getInt(state, op.getLhs());
    if (!lhs)
      return op->emitError("Invalid lhs val");

    auto rhs = getInt(state, op.getRhs());
    if (!rhs)
      return op->emitError("Invalid rhs val");

    state.state[op.getResult()] = setInt(op.getContext(), *lhs + *rhs);
    return InterpreterResult::Advance;
  }
};

struct SubOpInterpreterInterface final
    : public hc::typing::TypingInterpreterInterface::ExternalModel<
          SubOpInterpreterInterface, mlir::arith::SubIOp> {
  mlir::FailureOr<hc::typing::InterpreterResult>
  interpret(mlir::Operation *o, InterpreterState &state) const {
    auto op = mlir::cast<mlir::arith::SubIOp>(o);
    auto lhs = getInt(state, op.getLhs());
    if (!lhs)
      return op->emitError("Invalid lhs val");

    auto rhs = getInt(state, op.getRhs());
    if (!rhs)
      return op->emitError("Invalid rhs val");

    state.state[op.getResult()] = setInt(op.getContext(), *lhs - *rhs);
    return InterpreterResult::Advance;
  }
};

struct OrOpInterpreterInterface final
    : public hc::typing::TypingInterpreterInterface::ExternalModel<
          OrOpInterpreterInterface, mlir::arith::OrIOp> {
  mlir::FailureOr<hc::typing::InterpreterResult>
  interpret(mlir::Operation *o, InterpreterState &state) const {
    auto op = mlir::cast<mlir::arith::OrIOp>(o);
    auto lhs = getInt(state, op.getLhs());
    if (!lhs)
      return op->emitError("Invalid lhs val");

    auto rhs = getInt(state, op.getRhs());
    if (!rhs)
      return op->emitError("Invalid rhs val");

    state.state[op.getResult()] = setInt(op.getContext(), *lhs | *rhs);
    return InterpreterResult::Advance;
  }
};

struct CmpOpInterpreterInterface final
    : public hc::typing::TypingInterpreterInterface::ExternalModel<
          CmpOpInterpreterInterface, mlir::arith::CmpIOp> {
  mlir::FailureOr<hc::typing::InterpreterResult>
  interpret(mlir::Operation *o, InterpreterState &state) const {
    auto op = mlir::cast<mlir::arith::CmpIOp>(o);
    auto lhs = getInt(state, op.getLhs());
    if (!lhs)
      return op->emitError("Invalid lhs val");

    auto rhs = getInt(state, op.getRhs());
    if (!rhs)
      return op->emitError("Invalid rhs val");

    int64_t res;
    using Pred = mlir::arith::CmpIPredicate;
    switch (op.getPredicate()) {
    case Pred::eq:
      res = (*lhs == *rhs);
      break;
    case Pred::ne:
      res = (*lhs != *rhs);
      break;
    case Pred::slt:
      res = (*lhs < *rhs);
      break;
    case Pred::sle:
      res = (*lhs <= *rhs);
      break;
    case Pred::sgt:
      res = (*lhs > *rhs);
      break;
    case Pred::sge:
      res = (*lhs >= *rhs);
      break;
    case Pred::ult:
      res = (static_cast<uint64_t>(*lhs) < static_cast<uint64_t>(*rhs));
      break;
    case Pred::ule:
      res = (static_cast<uint64_t>(*lhs) <= static_cast<uint64_t>(*rhs));
      break;
    case Pred::ugt:
      res = (static_cast<uint64_t>(*lhs) > static_cast<uint64_t>(*rhs));
      break;
    case Pred::uge:
      res = (static_cast<uint64_t>(*lhs) >= static_cast<uint64_t>(*rhs));
      break;
    default:
      return op->emitError("Unsupported predicate: ") << op.getPredicateAttr();
    }

    state.state[op.getResult()] = setInt(op.getContext(), res);
    return InterpreterResult::Advance;
  }
};

struct CallOpInterpreterInterface final
    : public hc::typing::TypingInterpreterInterface::ExternalModel<
          CallOpInterpreterInterface, mlir::func::CallOp> {
  mlir::FailureOr<hc::typing::InterpreterResult>
  interpret(mlir::Operation *o, InterpreterState &state) const {
    auto op = mlir::cast<mlir::func::CallOp>(o);
    auto callee = op.getCalleeAttr();
    auto func = mlir::SymbolTable::lookupNearestSymbolFrom<mlir::func::FuncOp>(
        op->getParentOp(), callee);
    if (!func)
      return op->emitError("Function not found: ") << callee;

    auto ftype = func.getFunctionType();
    if (!llvm::equal(ftype.getInputs(), op.getOperandTypes()) ||
        !llvm::equal(ftype.getResults(), op.getResultTypes()))
      return op->emitError("Invalid  function type: ") << ftype;

    if (func.isDeclaration())
      return op->emitError("Function body is not available");

    mlir::Block &body = func.getFunctionBody().front();
    assert(body.getNumArguments() == op.getNumOperands());
    for (auto &&[src, dst] :
         llvm::zip_equal(op.getOperands(), body.getArguments())) {
      auto it = state.state.find(src);
      assert(it != state.state.end());
      state.state[dst] = it->second;
    }
    state.callstack.emplace_back(op);
    state.iter = body.begin();
    return InterpreterResult::Advance;
  }
};

struct ReturnOpInterpreterInterface final
    : public hc::typing::TypingInterpreterInterface::ExternalModel<
          ReturnOpInterpreterInterface, mlir::func::ReturnOp> {
  mlir::FailureOr<hc::typing::InterpreterResult>
  interpret(mlir::Operation *o, InterpreterState &state) const {
    auto op = mlir::cast<mlir::func::ReturnOp>(o);
    if (state.callstack.empty())
      return op->emitError("Callstack is empty");

    mlir::Operation *ret = state.callstack.pop_back_val();
    if (ret->getNumResults() != op.getNumOperands())
      return op->emitError("Operand count mismatch");

    for (auto &&[src, dst] :
         llvm::zip_equal(op.getOperands(), ret->getResults())) {
      auto it = state.state.find(src);
      assert(it != state.state.end());
      state.state[dst] = it->second;
    }
    state.iter = std::next(ret->getIterator());
    return InterpreterResult::Advance;
  }
};

struct SelectOpInterpreterInterface final
    : public hc::typing::TypingInterpreterInterface::ExternalModel<
          SelectOpInterpreterInterface, mlir::arith::SelectOp> {
  mlir::FailureOr<hc::typing::InterpreterResult>
  interpret(mlir::Operation *o, InterpreterState &state) const {
    auto op = mlir::cast<mlir::arith::SelectOp>(o);

    auto cond = getInt(state, op.getCondition());
    if (!cond)
      return op->emitError("Invalid cond value");

    auto it = state.state.find(*cond ? op.getTrueValue() : op.getFalseValue());
    assert(it != state.state.end());
    state.state[op.getResult()] = it->second;
    return InterpreterResult::Advance;
  }
};

struct SelectOpDataflowJoinInterface final
    : public hc::typing::DataflowJoinInterface::ExternalModel<
          SelectOpDataflowJoinInterface, mlir::arith::SelectOp> {

  void getArgsIndices(mlir::Operation * /*op*/, unsigned resultIndex,
                      llvm::SmallVectorImpl<unsigned> &argsIndices) const {
    assert(resultIndex == 0);
    argsIndices.emplace_back(1);
    argsIndices.emplace_back(2);
  }
};
} // namespace

void hc::typing::registerArithTypingInterpreter(mlir::MLIRContext &ctx) {
  ctx.loadDialect<mlir::arith::ArithDialect, mlir::cf::ControlFlowDialect,
                  mlir::func::FuncDialect>();

  mlir::cf::BranchOp::attachInterface<BranchOpInterpreterInterface>(ctx);
  mlir::cf::CondBranchOp::attachInterface<CondBranchOpInterpreterInterface>(
      ctx);

  mlir::arith::ConstantOp::attachInterface<ConstantOpInterpreterInterface>(ctx);
  mlir::arith::AddIOp::attachInterface<AddOpInterpreterInterface>(ctx);
  mlir::arith::SubIOp::attachInterface<SubOpInterpreterInterface>(ctx);
  mlir::arith::OrIOp::attachInterface<OrOpInterpreterInterface>(ctx);
  mlir::arith::CmpIOp::attachInterface<CmpOpInterpreterInterface>(ctx);

  mlir::func::CallOp::attachInterface<CallOpInterpreterInterface>(ctx);
  mlir::func::ReturnOp::attachInterface<ReturnOpInterpreterInterface>(ctx);

  mlir::arith::SelectOp::attachInterface<SelectOpInterpreterInterface>(ctx);
  mlir::arith::SelectOp::attachInterface<SelectOpDataflowJoinInterface>(ctx);
}

template <typename Dst, typename Src>
static auto castArrayRef(mlir::ArrayRef<Src> src) {
  return mlir::ArrayRef<Dst>(static_cast<const Dst *>(src.data()), src.size());
}

InterpreterValue hc::typing::getVal(const InterpreterState &state,
                                    mlir::Value val) {
  auto it = state.state.find(val);
  assert(it != state.state.end());
  return it->second;
}

static const constexpr int PackShift = 2;

std::optional<int64_t> hc::typing::getInt(InterpreterValue val) {
  if (mlir::isa<void *>(val))
    return reinterpret_cast<intptr_t>(mlir::cast<void *>(val)) >> PackShift;

  auto lit =
      mlir::dyn_cast<hc::typing::LiteralType>(mlir::cast<mlir::Type>(val));
  if (!lit)
    return std::nullopt;

  auto attr = mlir::dyn_cast<mlir::IntegerAttr>(lit.getValue());
  if (!attr)
    return std::nullopt;

  return attr.getInt();
}

std::optional<int64_t> hc::typing::getInt(InterpreterState &state,
                                          mlir::Value val) {
  auto it = state.state.find(val);
  assert(it != state.state.end());
  return getInt(it->second);
}

hc::typing::InterpreterValue hc::typing::setInt(mlir::MLIRContext *ctx,
                                                int64_t val) {
  if (((static_cast<intptr_t>(val) << PackShift) >> PackShift) == val)
    return reinterpret_cast<void *>(static_cast<intptr_t>(val) << PackShift);

  auto attr = mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64), val);
  return hc::typing::LiteralType::get(attr);
}

mlir::Type hc::typing::getType(const hc::typing::InterpreterState &state,
                               mlir::Value val) {
  return mlir::dyn_cast<mlir::Type>(getVal(state, val));
}

void hc::typing::getTypes(const hc::typing::InterpreterState &state,
                          mlir::ValueRange vals,
                          llvm::SmallVectorImpl<mlir::Type> &result) {
  result.reserve(result.size() + vals.size());
  for (auto val : vals)
    result.emplace_back(getType(state, val));
}

llvm::SmallVector<mlir::Type>
hc::typing::getTypes(const hc::typing::InterpreterState &state,
                     mlir::ValueRange vals) {
  llvm::SmallVector<mlir::Type> ret;
  getTypes(state, vals, ret);
  return ret;
}

mlir::FailureOr<hc::typing::InterpreterResult>
hc::typing::TypeConstantOp::interpret(InterpreterState &state) {
  mlir::Type type = mlir::cast<TypeAttr>(getValue()).getTypeVal();
  state.state[getResult()] = type;
  return InterpreterResult::Advance;
}

mlir::FailureOr<hc::typing::InterpreterResult>
hc::typing::TypeResolverReturnOp::interpret(InterpreterState &state) {
  assert(state.result);
  auto &result = *state.result;
  mlir::ValueRange args = getArgs();

  auto isSeq = [&]() -> typing::UnpackedSequenceType {
    if (args.size() != 1)
      return {};

    return mlir::dyn_cast_if_present<typing::UnpackedSequenceType>(
        getType(state, args.front()));
  };

  if (auto seq = isSeq()) {
    for (auto type : seq.getParams())
      result.emplace_back(type);
  } else {
    getTypes(state, args, result);
  }

  return InterpreterResult::MatchSuccess;
}

mlir::FailureOr<hc::typing::InterpreterResult>
hc::typing::MakeIdentOp::interpret(InterpreterState &state) {
  auto name = this->getNameAttr();
  auto paramNames =
      castArrayRef<mlir::StringAttr>(this->getParamNames().getValue());
  auto paramTypes = getTypes(state, this->getParams());
  state.state[getResult()] = hc::typing::IdentType::get(
      this->getContext(), name, paramNames, paramTypes);
  return InterpreterResult::Advance;
}

mlir::FailureOr<hc::typing::InterpreterResult>
hc::typing::MakeSymbolOp::interpret(InterpreterState &state) {
  auto name = hc::typing::getType(state, getName());
  auto nameLiteral = mlir::dyn_cast<LiteralType>(name);
  if (!nameLiteral || !mlir::isa<mlir::StringAttr>(nameLiteral.getValue()))
    return emitOpError("Invalid name value: ") << name;

  auto nameStr = mlir::cast<mlir::StringAttr>(nameLiteral.getValue());
  state.state[getResult()] =
      hc::typing::SymbolType::get(this->getContext(), nameStr.getValue());
  return InterpreterResult::Advance;
}

mlir::FailureOr<hc::typing::InterpreterResult>
hc::typing::MakeLiteralOp::interpret(InterpreterState &state) {
  state.state[getResult()] = hc::typing::LiteralType::get(getValue());
  return InterpreterResult::Advance;
}

mlir::FailureOr<hc::typing::InterpreterResult>
hc::typing::GetNumArgsOp::interpret(InterpreterState &state) {
  state.state[getResult()] =
      setInt(this->getContext(), static_cast<int64_t>(state.args.size()));
  return InterpreterResult::Advance;
}

mlir::FailureOr<hc::typing::InterpreterResult>
hc::typing::GetArgOp::interpret(InterpreterState &state) {
  auto index = getInt(state, getIndex());
  if (!index)
    return emitOpError("Invalid index val");

  auto id = *index;
  auto args = state.args;
  if (id < 0 || id >= static_cast<decltype(id)>(args.size()))
    return emitOpError("Index out of bounds: ")
           << id << " [0, " << args.size() << "]";

  state.state[getResult()] = args[id];
  return InterpreterResult::Advance;
}

mlir::FailureOr<hc::typing::InterpreterResult>
hc::typing::GetNamedArgOp::interpret(InterpreterState &state) {
  auto iface =
      mlir::dyn_cast_if_present<hc::typing::GetNamedArgInterface>(state.op);
  if (!iface)
    return emitError("GetNamedArgInterface is not available");

  auto res = iface.getNamedArg(getName());
  if (mlir::failed(res))
    return emitError("getNamedArg failed");

  mlir::Value val = *res;
  if (!val) {
    state.state[getResult()] = mlir::NoneType::get(getContext());
    return InterpreterResult::Advance;
  }

  for (auto &&[i, arg] : llvm::enumerate(state.op->getOperands())) {
    if (arg == val) {
      state.state[getResult()] = state.args[i];
      return InterpreterResult::Advance;
    }
  }

  return emitError("Invalid named arg");
}

mlir::FailureOr<hc::typing::InterpreterResult>
hc::typing::GetAttrOp::interpret(InterpreterState &state) {
  auto op = state.op;
  if (!op)
    return emitOpError("Root op is not set");

  auto name = getNameAttr();
  auto attr = op->getAttrOfType<mlir::TypedAttr>(name);
  if (!attr)
    return emitOpError("Invalid attr: ") << name.getValue() << " " << *op;

  state.state[getResult()] = hc::typing::LiteralType::get(attr);
  return InterpreterResult::Advance;
}

mlir::FailureOr<hc::typing::InterpreterResult>
hc::typing::GetIdentNameOp::interpret(InterpreterState &state) {
  auto type = hc::typing::getType(state, getIdent());
  auto ident = mlir::dyn_cast_if_present<hc::typing::IdentType>(type);
  if (!ident)
    return emitError("Invalid ident type, got: ") << type;

  auto name = ident.getName();
  state.state[getResult()] = hc::typing::LiteralType::get(name);
  return InterpreterResult::Advance;
}

mlir::FailureOr<hc::typing::InterpreterResult>
hc::typing::GetIdentParamOp::interpret(InterpreterState &state) {
  auto type = hc::typing::getType(state, getIdent());
  auto ident = mlir::dyn_cast_if_present<hc::typing::IdentType>(type);
  if (!ident)
    return emitError("Invalid ident type, got: ") << type;

  auto nameAttr = getNameAttr();
  auto paramVal = ident.getParam(nameAttr);
  if (!paramVal)
    return emitError("Invalid param name for ")
           << ident << " : " << nameAttr.getValue();

  state.state[getResult()] = paramVal;
  return InterpreterResult::Advance;
}

mlir::FailureOr<hc::typing::InterpreterResult>
hc::typing::GetMetatypeNameOp::interpret(InterpreterState &state) {
  auto *ctx = getContext();
  auto type = hc::typing::getType(state, getValue());
  mlir::StringRef name = type.getAbstractType().getName();

  state.state[getResult()] =
      hc::typing::LiteralType::get(mlir::StringAttr::get(ctx, name));
  return InterpreterResult::Advance;
}

mlir::FailureOr<hc::typing::InterpreterResult>
hc::typing::CreateSeqOp::interpret(InterpreterState &state) {
  state.state[getResult()] = SequenceType::get(getContext(), std::nullopt);
  return InterpreterResult::Advance;
}

mlir::FailureOr<hc::typing::InterpreterResult>
hc::typing::AppendSeqOp::interpret(InterpreterState &state) {
  auto seq = mlir::dyn_cast_if_present<SequenceType>(
      ::hc::typing::getType(state, getSeq()));
  if (!seq)
    return emitError("Invalid seq type");

  auto arg = ::hc::typing::getType(state, getArg());
  llvm::SmallVector<mlir::Type> newArgs;
  llvm::append_range(newArgs, seq.getParams());
  newArgs.emplace_back(arg);

  state.state[getResult()] = SequenceType::get(getContext(), newArgs);
  return InterpreterResult::Advance;
}

mlir::FailureOr<hc::typing::InterpreterResult>
hc::typing::GetSeqElementOp::interpret(InterpreterState &state) {
  auto seq = mlir::dyn_cast_if_present<SequenceType>(
      ::hc::typing::getType(state, getSeq()));
  if (!seq)
    return emitError("Invalid seq type");

  auto val = getInt(state, getIndex());
  if (!val)
    return emitError("Invalid index");

  auto idx = *val;
  mlir::TypeRange params = seq.getParams();
  if (idx < 0 || static_cast<size_t>(idx) >= params.size())
    return emitError("Index out of bounds: ") << idx;

  state.state[getResult()] = params[idx];
  return InterpreterResult::Advance;
}

mlir::FailureOr<hc::typing::InterpreterResult>
hc::typing::GetSeqSizeOp::interpret(InterpreterState &state) {
  auto seq = mlir::dyn_cast_if_present<SequenceType>(
      ::hc::typing::getType(state, getSeq()));
  if (!seq)
    return emitError("Invalid seq type");

  state.state[getResult()] =
      setInt(getContext(), static_cast<int64_t>(seq.getParams().size()));
  return InterpreterResult::Advance;
}

mlir::FailureOr<hc::typing::InterpreterResult>
hc::typing::UnpackSeqOp::interpret(InterpreterState &state) {
  auto seq = mlir::dyn_cast_if_present<SequenceType>(
      ::hc::typing::getType(state, getSeq()));
  if (!seq)
    return emitError("Invalid seq type");

  state.state[getResult()] =
      UnpackedSequenceType::get(getContext(), seq.getParams());
  return InterpreterResult::Advance;
}

mlir::FailureOr<hc::typing::InterpreterResult>
hc::typing::IsSameOp::interpret(InterpreterState &state) {
  auto lhs = hc::typing::getType(state, getLhs());
  auto rhs = hc::typing::getType(state, getRhs());
  state.state[getResult()] = setInt(getContext(), lhs == rhs);
  return InterpreterResult::Advance;
}

mlir::FailureOr<hc::typing::InterpreterResult>
hc::typing::CheckOp::interpret(InterpreterState &state) {
  auto val = getInt(state, getCondition());
  if (!val)
    return emitError("Inavlid condition val");
  return *val ? InterpreterResult::Advance : InterpreterResult::MatchFail;
}

mlir::FailureOr<hc::typing::InterpreterResult>
hc::typing::PrintOp::interpret(InterpreterState &state) {
  auto type = hc::typing::getType(state, getValue());
  llvm::errs() << type << "\n";
  return InterpreterResult::Advance;
}

mlir::FailureOr<hc::typing::InterpreterResult>
hc::typing::MakeUnionOp::interpret(InterpreterState &state) {
  llvm::SmallSetVector<mlir::Type, 8> types;
  for (mlir::Value arg : getArgs()) {
    auto type = hc::typing::getType(state, arg);
    if (!type)
      return emitError("Invalid arg");
    if (auto u = mlir::dyn_cast<hc::typing::UnionType>(type)) {
      for (mlir::Type p : u.getParams())
        types.insert(p);
    } else {
      types.insert(type);
    }
  }

  state.state[getResult()] =
      hc::typing::UnionType::get(getContext(), types.getArrayRef());
  return InterpreterResult::Advance;
}

mlir::FailureOr<hc::typing::InterpreterResult>
hc::typing::GetGlobalAttrOp::interpret(InterpreterState &state) {
  if (!state.op)
    return emitError("op is not set");

  auto mod = state.op->getParentOfType<mlir::ModuleOp>();
  if (!mod)
    return emitError("no module");

  auto name = getNameAttr();
  auto attr = mod->getAttr(name);
  if (!attr)
    return emitError("Attribute ") << name.getValue() << " not found";

  auto typeAttr = mlir::dyn_cast<hc::typing::TypeAttr>(attr);
  if (!typeAttr)
    return emitError("Attribute ") << attr << " is not TypeAttr";

  state.state[getResult()] = typeAttr.getTypeVal();
  return InterpreterResult::Advance;
}

mlir::FailureOr<hc::typing::InterpreterResult>
hc::typing::BinOp::interpret(InterpreterState &state) {
  auto lhs = ::getType<SymbolicTypeBase>(state, getLhs());
  if (!lhs)
    return emitError("Invalid lhs value");

  auto rhs = ::getType<SymbolicTypeBase>(state, getRhs());
  if (!rhs)
    return emitError("Invalid rhs value");

  InterpreterValue res;
  switch (getOp()) {
  case BinOpVal::add:
    res = lhs + rhs;
    break;
  case BinOpVal::sub:
    res = lhs - rhs;
    break;
  case BinOpVal::mul:
    res = lhs * rhs;
    break;
  case BinOpVal::ceil_div:
    res = lhs.ceilDiv(rhs);
    break;
  case BinOpVal::floor_div:
    res = lhs.floorDiv(rhs);
    break;
  case BinOpVal::mod:
    res = lhs % rhs;
    break;
  default:
    return emitError("Unsupported op: ") << getOpAttr();
  }

  state.state[getResult()] = res;
  return InterpreterResult::Advance;
}

static bool expandLiterals(llvm::SmallVectorImpl<mlir::Type> &params,
                           mlir::AffineExpr &expr) {
  bool changed = false;
  for (auto &&[i, param] : llvm::enumerate(params)) {
    auto lit = mlir::dyn_cast<hc::typing::LiteralType>(param);
    if (!lit)
      continue;

    auto attr = mlir::dyn_cast<mlir::IntegerAttr>(lit.getValue());
    if (!attr)
      continue;

    auto *ctx = expr.getContext();
    expr = expr.replace(
        mlir::getAffineSymbolExpr(i, ctx),
        mlir::getAffineConstantExpr(attr.getValue().getSExtValue(), ctx));
    changed = true;
  }
  return changed;
}

static bool expandNested(llvm::SmallVectorImpl<mlir::Type> &params,
                         mlir::AffineExpr &expr) {
  for (auto &&[i, param] : llvm::enumerate(params)) {
    auto nested = mlir::dyn_cast<hc::typing::ExprType>(param);
    if (!nested)
      continue;

    auto nestedParams = nested.getParams();
    auto nestedExpr = nested.getExpr();
    nestedExpr = nestedExpr.shiftSymbols(nestedParams.size(), params.size());
    params.append(nestedParams.begin(), nestedParams.end());
    auto *ctx = expr.getContext();
    params[i] = mlir::NoneType::get(ctx);
    expr = expr.replace(mlir::getAffineSymbolExpr(i, ctx), nestedExpr);
    return true;
  }
  return false;
}

static bool removeDuplicates(llvm::SmallVectorImpl<mlir::Type> &params,
                             mlir::AffineExpr &expr) {
  bool changed = false;
  auto *ctx = expr.getContext();
  llvm::SmallDenseMap<mlir::Type, unsigned> paramIdx;
  mlir::SmallVector<mlir::AffineExpr> replacement(params.size());
  for (auto &&[i, param] : llvm::enumerate(params)) {
    auto it = paramIdx.find(param);
    if (it == paramIdx.end()) {
      auto id = unsigned(i);
      paramIdx.insert({param, id});
      replacement[i] = mlir::getAffineSymbolExpr(id, ctx);
      continue;
    }

    replacement[i] = mlir::getAffineSymbolExpr(it->second, ctx);
    changed = true;
  }

  expr = expr.replaceSymbols(replacement);
  return changed;
}

static bool removeUnused(llvm::SmallVectorImpl<mlir::Type> &params,
                         mlir::AffineExpr &expr) {
  auto *ctx = expr.getContext();
  llvm::SmallBitVector used(params.size());
  expr.walk([&](mlir::AffineExpr e) {
    if (auto sym = mlir::dyn_cast<mlir::AffineSymbolExpr>(e)) {
      assert(sym.getPosition() < used.size());
      used.set(sym.getPosition());
    }
  });
  if (used.all())
    return false;

  for (auto i : llvm::reverse(llvm::seq<size_t>(0, used.size()))) {
    if (used[i])
      continue;

    params.erase(params.begin() + i);
  }

  unsigned offset = 0;
  mlir::SmallVector<mlir::AffineExpr> replacement(used.size());
  for (auto i : llvm::seq<size_t>(0, used.size())) {
    replacement[i] = mlir::getAffineSymbolExpr(offset, ctx);
    if (used[i])
      ++offset;
  }

  expr = expr.replaceSymbols(replacement);
  return true;
}

static void sortParams(llvm::SmallVectorImpl<mlir::Type> &params,
                       mlir::AffineExpr &expr) {
  llvm::SmallDenseMap<mlir::Type, unsigned> origPos;
  for (auto &&[i, param] : llvm::enumerate(params))
    origPos[param] = unsigned(i);

  auto cmp = [](mlir::Type lhs, mlir::Type rhs) -> bool {
    auto lhsSym = mlir::dyn_cast<hc::typing::SymbolType>(lhs);
    if (!lhsSym)
      return false;

    auto rhsSym = mlir::dyn_cast<hc::typing::SymbolType>(rhs);
    if (!rhsSym)
      return true;

    return lhsSym.getName().getValue() < rhsSym.getName().getValue();
  };

  std::stable_sort(params.begin(), params.end(), cmp);

  auto *ctx = expr.getContext();
  mlir::SmallVector<mlir::AffineExpr> replacement(params.size());
  for (auto &&[i, param] : llvm::enumerate(params)) {
    auto orig = origPos.find(param)->second;
    replacement[orig] = mlir::getAffineSymbolExpr(i, ctx);
  }
  expr = expr.replaceSymbols(replacement);
}

static std::pair<llvm::SmallVector<mlir::Type>, mlir::AffineExpr>
simplifyExpr(llvm::ArrayRef<mlir::Type> params, mlir::AffineExpr expr) {
  llvm::SmallVector<mlir::Type> retParams(params);
  for (auto i : llvm::seq<size_t>(0, retParams.size()))
    retParams[i] =
        SymbolicTypeBase::foldExpr(mlir::cast<SymbolicTypeBase>(retParams[i]));

  bool changed;
  do {
    changed = false;
    expr = mlir::simplifyAffineExpr(expr, 0, retParams.size());

    if (expandLiterals(retParams, expr))
      changed = true;

    if (expandNested(retParams, expr))
      changed = true;

    if (removeDuplicates(retParams, expr))
      changed = true;

    if (removeUnused(retParams, expr))
      changed = true;
  } while (changed);

  sortParams(retParams, expr);
  return {retParams, expr};
}

static mlir::ParseResult
parseExprType(mlir::AsmParser &parser,
              llvm::SmallVectorImpl<mlir::Type> &params,
              mlir::AffineExpr &expr) {
  if (!parser.parseLParen()) {
    auto result =
        mlir::FieldParser<llvm::SmallVector<mlir::Type>>::parse(parser);
    if (mlir::failed(result))
      return parser.emitError(parser.getCurrentLocation(),
                              "Failed to parse params list");

    params = *result;

    if (parser.parseRParen())
      return parser.emitError(parser.getCurrentLocation(), "\")\" expected");
    if (parser.parseArrow())
      return parser.emitError(parser.getCurrentLocation(), "\"->\" expected");
  }

  auto *ctx = parser.getContext();
  llvm::SmallVector<std::pair<llvm::StringRef, mlir::AffineExpr>> paramExprs;
  for (auto &&[i, p] : llvm::enumerate(params)) {
    auto str = mlir::StringAttr::get(ctx, "s" + llvm::Twine(i)).getValue();
    paramExprs.emplace_back(str, mlir::getAffineSymbolExpr(i, ctx));
  }

  if (parser.parseAffineExpr(paramExprs, expr))
    return parser.emitError(parser.getCurrentLocation(),
                            "affine expr expected");

  return mlir::success();
}

static void printExprType(mlir::AsmPrinter &printer,
                          llvm::ArrayRef<mlir::Type> params,
                          mlir::AffineExpr expr) {
  if (!params.empty()) {
    printer << "(";
    llvm::interleaveComma(params, printer);
    printer << ") -> ";
  }
  printer << expr;
}

bool SymbolicTypeBase::classof(mlir::Type type) {
  return llvm::isa<LiteralType, SymbolType, BitsizeType, ExprType>(type);
}

static SymbolicTypeBase getLiteral(mlir::MLIRContext *ctx, int64_t val) {
  auto index = mlir::IndexType::get(ctx);
  return LiteralType::get(mlir::IntegerAttr::get(index, val));
}

SymbolicTypeBase SymbolicTypeBase::foldExpr(SymbolicTypeBase src) {
  if (auto bitsize = mlir::dyn_cast<BitsizeType>(src)) {
    auto arg = bitsize.getArg();
    if (arg.isIntOrFloat())
      return getLiteral(src.getContext(), arg.getIntOrFloatBitWidth());

    return src;
  }
  auto expr = mlir::dyn_cast<ExprType>(src);
  if (!expr)
    return src;

  if (auto lit = mlir::dyn_cast<mlir::AffineConstantExpr>(expr.getExpr()))
    return getLiteral(src.getContext(), lit.getValue());

  auto params = expr.getParams();
  if (params.size() != 1)
    return expr;

  if (expr.getExpr() != mlir::getAffineSymbolExpr(0, expr.getContext()))
    return expr;

  return mlir::cast<SymbolicTypeBase>(params.front());
}

template <typename F>
static SymbolicTypeBase makeExpr(SymbolicTypeBase lhs, SymbolicTypeBase rhs,
                                 F &&func) {
  auto ctx = lhs.getContext();
  auto op = func(mlir::getAffineSymbolExpr(0, ctx),
                 mlir::getAffineSymbolExpr(1, ctx));
  return SymbolicTypeBase::foldExpr(
      hc::typing::ExprType::get(ctx, {lhs, rhs}, op));
}

SymbolicTypeBase SymbolicTypeBase::operator+(SymbolicTypeBase rhs) const {
  return makeExpr(*this, rhs, [](auto a, auto b) { return a + b; });
}

SymbolicTypeBase SymbolicTypeBase::operator-(SymbolicTypeBase rhs) const {
  return makeExpr(*this, rhs, [](auto a, auto b) { return a - b; });
}

SymbolicTypeBase SymbolicTypeBase::operator*(SymbolicTypeBase rhs) const {
  return makeExpr(*this, rhs, [](auto a, auto b) { return a * b; });
}

SymbolicTypeBase SymbolicTypeBase::operator%(SymbolicTypeBase rhs) const {
  return makeExpr(*this, rhs, [](auto a, auto b) { return a % b; });
}

SymbolicTypeBase SymbolicTypeBase::floorDiv(SymbolicTypeBase rhs) const {
  return makeExpr(*this, rhs, [](auto a, auto b) { return a.floorDiv(b); });
}

SymbolicTypeBase SymbolicTypeBase::ceilDiv(SymbolicTypeBase rhs) const {
  return makeExpr(*this, rhs, [](auto a, auto b) { return a.ceilDiv(b); });
}

#include "hc/Dialect/Typing/IR/TypingOpsDialect.cpp.inc"

#define GET_OP_CLASSES
#include "hc/Dialect/Typing/IR/TypingOps.cpp.inc"

#include "hc/Dialect/Typing/IR/TypingOpsInterfaces.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "hc/Dialect/Typing/IR/TypingOpsAttributes.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "hc/Dialect/Typing/IR/TypingOpsTypes.cpp.inc"

#include "hc/Dialect/Typing/IR/TypingOpsEnums.cpp.inc"
