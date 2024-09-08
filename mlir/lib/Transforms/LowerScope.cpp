// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hc/Transforms/Passes.hpp"

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include "hc/Dialect/HKernel/IR/HKernelOps.hpp"
#include "hc/Dialect/Typing/IR/TypingOps.hpp"

namespace hc {
#define GEN_PASS_DEF_LOWERWORKGROUPSCOPEPASS
#include "hc/Transforms/Passes.h.inc"
} // namespace hc

template <typename T>
static T getTypeAttr(mlir::Operation *op, mlir::StringRef name) {
  auto attr = op->getAttrOfType<hc::typing::TypeAttr>(name);
  if (!attr)
    return nullptr;

  return mlir::dyn_cast<T>(attr.getTypeVal());
}

static bool isDivisibleImpl(mlir::TypeRange args, mlir::AffineExpr expr,
                            mlir::Type sym) {
  if (auto idx = mlir::dyn_cast<mlir::AffineSymbolExpr>(expr))
    return args[idx.getPosition()] == sym;

  if (expr.getKind() == mlir::AffineExprKind::Mul) {
    auto mul = mlir::cast<mlir::AffineBinaryOpExpr>(expr);
    return isDivisibleImpl(args, mul.getLHS(), sym) ||
           isDivisibleImpl(args, mul.getRHS(), sym);
  }

  return false;
}

static bool isDivisible(mlir::Type src, mlir::Type sym) {
  if (src == sym)
    return true;

  if (auto expr = mlir::dyn_cast<hc::typing::ExprType>(src))
    return isDivisibleImpl(expr.getParams(), expr.getExpr(), sym);

  return false;
}

static bool checkShapeDivisible(mlir::TypeRange shape, mlir::TypeRange div) {
  if (div.size() > shape.size())
    div = div.take_back(shape.size());

  for (auto &&[s, d] : llvm::zip_equal(shape.take_back(div.size()), div)) {
    if (!isDivisible(s, d))
      return false;
  }

  return true;
}

static bool checkShapedDivisible(mlir::Type type, mlir::TypeRange div) {
  if (auto buffer = mlir::dyn_cast<hc::hk::SymbolicallyShapedType>(type))
    return checkShapeDivisible(buffer.getShape(), div);

  return false;
}

static mlir::LogicalResult lowerWGScope(hc::hk::EnvironmentRegionOp region) {
  auto mod = region->getParentOfType<mlir::ModuleOp>();
  if (!mod)
    return region.emitError("No parent module");

  auto groupShape =
      getTypeAttr<hc::typing::SequenceType>(mod, "kernel.group_shape");
  if (!groupShape)
    return region->emitError("Group shape is not defined");

  auto localId = getTypeAttr<hc::typing::SequenceType>(mod, "kernel.local_id");
  if (!localId)
    return region->emitError("Local ID is not defined");

  if (groupShape.size() != localId.size())
    return region->emitError("Invalid group shape");

  auto subgroupId =
      getTypeAttr<hc::typing::SymbolicTypeBase>(mod, "kernel.subgroup_id");
  if (!subgroupId)
    return region->emitError("Subgroup ID is not defined");

  auto subgroupSize =
      getTypeAttr<hc::typing::SymbolicTypeBase>(mod, "kernel.subgroup_size");
  if (!subgroupSize)
    return region->emitError("Subgroup size is not defined");

  llvm::SmallVector<mlir::Type> ids(localId.getParams());
  ids.back() = subgroupId;

  mlir::OpBuilder builder(region.getContext());

  auto distributeShapedType = [&](hc::hk::SymbolicallyShapedType shapedType)
      -> hc::hk::SymbolicallyShapedType {
    auto rank = shapedType.getShape().size();
    auto grCount = groupShape.size();
    llvm::SmallVector<mlir::Type> newShape(
        shapedType.getShape().drop_back(grCount));
    auto groupShapeRef = llvm::ArrayRef(groupShape.getParams()).take_back(rank);
    for (auto &&[s, g] :
         llvm::zip(shapedType.getShape().take_back(grCount), groupShapeRef)) {
      auto symbolic = mlir::cast<hc::typing::SymbolicTypeBase>(s);
      auto symbolicGr = mlir::cast<hc::typing::SymbolicTypeBase>(g);
      newShape.emplace_back(symbolic.floorDiv(symbolicGr));
    }
    newShape.back() =
        mlir::cast<hc::typing::SymbolicTypeBase>(newShape.back()) *
        subgroupSize;
    return shapedType.clone(newShape);
  };

  mlir::IRMapping mapping;
  auto getShaped = [&](mlir::Value val) -> mlir::Value {
    if (auto res = mapping.lookupOrNull(val))
      return res;

    auto shapedType = mlir::cast<hc::hk::SymbolicallyShapedType>(val.getType());
    if (shapedType.getShape().empty()) {
      mapping.map(val, val);
      return val;
    }

    auto rank = shapedType.getShape().size();
    auto grCount = groupShape.size();

    mlir::Location loc = builder.getUnknownLoc();
    llvm::SmallVector<mlir::Value> newIndices;
    llvm::SmallVector<mlir::Type> newShape(
        shapedType.getShape().drop_back(grCount));
    if (rank > grCount) {
      auto type = hc::typing::LiteralType::get(builder.getIndexAttr(0));
      mlir::Value zero = builder.create<hc::hk::MaterializeExprOp>(loc, type);
      newIndices.resize(rank - grCount, zero);
    }

    for (auto &&[ind, s] :
         llvm::zip(llvm::ArrayRef(ids).take_back(rank),
                   shapedType.getShape().take_back(grCount))) {
      mlir::Value idx = builder.create<hc::hk::MaterializeExprOp>(loc, ind);
      newIndices.emplace_back(builder.create<hc::hk::MakeSliceOp>(loc, idx));
      auto newDim = mlir::cast<hc::typing::SymbolicTypeBase>(s) -
                    mlir::cast<hc::typing::SymbolicTypeBase>(ind);
      newShape.emplace_back(newDim);
    }
    auto newShapedType = shapedType.clone(newShape);

    mlir::Value res =
        builder.create<hc::hk::SubViewOp>(loc, newShapedType, val, newIndices);
    mapping.map(val, res);
    return res;
  };

  auto genShapeArray = [&](hc::hk::SymbolicallyShapedType type)
      -> llvm::SmallVector<mlir::Value> {
    llvm::SmallVector<mlir::Value> ret;
    mlir::Location loc = builder.getUnknownLoc();
    for (auto t : type.getShape())
      ret.emplace_back(builder.create<hc::hk::MaterializeExprOp>(loc, t));

    return ret;
  };

  auto visitor = [&](mlir::Operation *op) -> mlir::WalkResult {
    if (auto reg = mlir::dyn_cast<hc::hk::EnvironmentRegionOp>(op)) {
      return mlir::isa<hc::hk::WorkgroupScopeAttr, hc::hk::SubgroupScopeAttr,
                       hc::hk::WorkitemScopeAttr>(reg.getEnvironment())
                 ? mlir::WalkResult::skip()
                 : mlir::WalkResult::advance();
    }

    if (auto load = mlir::dyn_cast<hc::hk::LoadOp>(op)) {
      if (!checkShapedDivisible(load.getType(), groupShape.getParams())) {
        load.emitError("load shape is not divisible by group size");
        return mlir::WalkResult::interrupt();
      }

      auto newResType = distributeShapedType(
          mlir::cast<hc::hk::SymbolicallyShapedType>(load.getType()));

      builder.setInsertionPoint(op);
      getShaped(load.getSource()); // Populate mapper

      mapping.map(load.getIndex(), genShapeArray(newResType));
      auto newOp = builder.clone(*op, mapping);
      newOp->getResult(0).setType(newResType);
      return mlir::WalkResult::advance();
    }

    if (auto store = mlir::dyn_cast<hc::hk::StoreOp>(op)) {
      if (!mapping.lookupOrNull(store.getSource()))
        return mlir::WalkResult::advance();

      builder.setInsertionPoint(op);
      getShaped(store.getTarget()); // Populate mapper
      builder.clone(*op, mapping);
      store->erase();
      return mlir::WalkResult::advance();
    }

    return mlir::WalkResult::advance();
  };

  if (region.getBody()
          ->walk<mlir::WalkOrder::PostOrder>(visitor)
          .wasInterrupted())
    return mlir::failure();

  region.setEnvironmentAttr(
      hc::hk::SubgroupScopeAttr::get(region.getContext()));
  return mlir::success();
}

namespace {
struct LowerWorkgroupScopePass final
    : public hc::impl::LowerWorkgroupScopePassBase<LowerWorkgroupScopePass> {

  void runOnOperation() override {

    auto visitor = [&](hc::hk::EnvironmentRegionOp region) -> mlir::WalkResult {
      if (!mlir::isa<hc::hk::WorkgroupScopeAttr>(region.getEnvironment()))
        return mlir::WalkResult::advance();

      if (mlir::failed(lowerWGScope(region)))
        return mlir::WalkResult::interrupt();

      return mlir::WalkResult::skip();
    };

    if (getOperation()
            ->walk<mlir::WalkOrder::PostOrder>(visitor)
            .wasInterrupted())
      return signalPassFailure();

    // DCE
    (void)applyPatternsAndFoldGreedily(getOperation(), {});
  }
};
} // namespace
