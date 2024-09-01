// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hc/Dialect/HKernel/IR/HKernelOps.hpp"

#include "hc/Dialect/Typing/IR/TypingOps.hpp"

#include <mlir/Dialect/Utils/StaticValueUtils.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>

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
  if (!parser.parseRParen())
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

  // TODO: parse unary minus

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

  // TODO: Parse minus
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
