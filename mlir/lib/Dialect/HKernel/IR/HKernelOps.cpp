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

static mlir::LogicalResult
parseSymbolicShape(mlir::AsmParser &parser,
                   llvm::SmallVectorImpl<mlir::Type> &shape) {
  // TODO: parse
  return mlir::failure();
}

enum class BindingStrength {
  Weak,   // + and -
  Strong, // All other binary operators.
};

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
