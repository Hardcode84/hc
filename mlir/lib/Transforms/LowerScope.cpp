// SPDX-FileCopyrightText: 2024 The HC Authors
// SPDX-FileCopyrightText: 2025 The HC Authors
//
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

namespace hc {
#define GEN_PASS_DEF_LOWERSUBGROUPSCOPEPASS
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

static const constexpr unsigned UnsetIdx = static_cast<unsigned>(-1);

static std::optional<llvm::SmallVector<unsigned, 3>>
findDivisibleShape(mlir::TypeRange shape, mlir::TypeRange div) {
  llvm::SmallVector<unsigned, 3> ret(div.size(), UnsetIdx);
  unsigned set = 0;

  while (!shape.empty()) {
    auto dim = shape.back();
    shape = shape.drop_back(1);
    auto idx = static_cast<unsigned>(shape.size());
    for (auto i : llvm::reverse(llvm::seq<size_t>(0, div.size()))) {
      if (ret[i] != UnsetIdx)
        continue;

      if (isDivisible(dim, div[i])) {
        ret[i] = idx;
        ++set;
        break;
      }
    }

    if (set == ret.size())
      break;
  }

  if (!set)
    return std::nullopt;

  return ret;
}

static std::optional<llvm::SmallVector<unsigned, 3>>
checkShapedDivisible(mlir::Type type, mlir::TypeRange div) {
  if (auto buffer = mlir::dyn_cast<hc::hk::SymbolicallyShapedType>(type))
    return findDivisibleShape(buffer.getShape(), div);

  return std::nullopt;
}

static mlir::LogicalResult lowerScope(hc::hk::EnvironmentRegionOp region,
                                      mlir::TypeRange groupShape,
                                      mlir::TypeRange ids,
                                      hc::typing::SymbolicTypeBase subgroupSize,
                                      bool wgScope) {
  mlir::OpBuilder builder(region.getContext());

  auto distributeShapedType = [&](mlir::Type type,
                                  mlir::ArrayRef<unsigned> distributeDims)
      -> hc::hk::SymbolicallyShapedType {
    auto shapedType = mlir::cast<hc::hk::SymbolicallyShapedType>(type);
    llvm::SmallVector<mlir::Type> newShape(shapedType.getShape());
    for (auto &&[i, d] : llvm::enumerate(distributeDims)) {
      if (d == UnsetIdx)
        continue;

      assert(d < newShape.size());
      auto symbolic = mlir::cast<hc::typing::SymbolicTypeBase>(newShape[d]);
      auto symbolicGr = mlir::cast<hc::typing::SymbolicTypeBase>(groupShape[i]);
      newShape[d] = symbolic.floorDiv(symbolicGr);
    }

    if (wgScope && distributeDims.back() != UnsetIdx) {
      auto i = distributeDims.back();
      auto elem = mlir::cast<hc::typing::SymbolicTypeBase>(newShape[i]);
      newShape[i] = elem * subgroupSize;
    }
    return shapedType.clone(newShape);
  };

  mlir::IRMapping mapping;
  auto getShaped = [&](mlir::Value val, mlir::TypeRange distShape,
                       mlir::ArrayRef<unsigned> distributeDims) -> mlir::Value {
    if (auto res = mapping.lookupOrNull(val))
      return res;

    auto shapedType = mlir::cast<hc::hk::SymbolicallyShapedType>(val.getType());
    if (shapedType.getShape().empty()) {
      mapping.map(val, val);
      return val;
    }

    llvm::SmallVector<mlir::Type> newShape(shapedType.getShape());

    mlir::Location loc = builder.getUnknownLoc();
    auto makeSlice = [&](mlir::Type type) -> mlir::Value {
      mlir::Value idx = builder.create<hc::hk::MaterializeExprOp>(loc, type);
      return builder.create<hc::hk::MakeSliceOp>(loc, idx);
    };

    auto zeroType = hc::typing::LiteralType::get(builder.getIndexAttr(0));
    llvm::SmallVector<mlir::Value> newIndices(newShape.size(),
                                              makeSlice(zeroType));
    for (auto &&[i, d] : llvm::enumerate(distributeDims)) {
      if (d == UnsetIdx)
        continue;

      assert(i < ids.size());
      assert(d < newIndices.size());
      auto ind = mlir::cast<hc::typing::SymbolicTypeBase>(ids[i]);
      if (i == (distributeDims.size() - 1) && !wgScope)
        ind = ind % subgroupSize;

      auto newDim = mlir::cast<hc::typing::SymbolicTypeBase>(distShape[d]);
      ind = ind * newDim;

      newIndices[d] = makeSlice(ind);

      auto dim = mlir::cast<hc::typing::SymbolicTypeBase>(newShape[d]);
      newShape[d] = dim - ind;
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
      auto div = checkShapedDivisible(load.getType(), groupShape);
      if (!div) {
        load.emitError("load shape is not divisible by group size");
        return mlir::WalkResult::interrupt();
      }

      auto newResType = distributeShapedType(load.getType(), *div);

      builder.setInsertionPoint(op);
      getShaped(load.getSource(), newResType.getShape(),
                *div); // Populate mapper

      mapping.map(load.getShape(), genShapeArray(newResType));
      auto newOp = builder.clone(*op, mapping);
      newOp->getResult(0).setType(newResType);
      return mlir::WalkResult::advance();
    }

    if (auto store = mlir::dyn_cast<hc::hk::StoreOp>(op)) {
      auto newSrc = mapping.lookupOrNull(store.getSource());
      if (!newSrc)
        return mlir::WalkResult::advance();

      auto srcType = store.getSource().getType();
      auto div = checkShapedDivisible(srcType, groupShape);
      if (!div)
        return mlir::WalkResult::advance();

      auto newSrcType = distributeShapedType(srcType, *div);
      if (newSrc.getType() != newSrcType)
        return mlir::WalkResult::advance();

      builder.setInsertionPoint(op);
      getShaped(store.getTarget(), newSrcType.getShape(),
                *div); // Populate mapper
      builder.clone(*op, mapping);
      store->erase();
      return mlir::WalkResult::advance();
    }

    return mlir::WalkResult::advance();
  };

  return mlir::failure(region.getBody()
                           ->walk<mlir::WalkOrder::PostOrder>(visitor)
                           .wasInterrupted());
}

static mlir::LogicalResult lowerWGScope(hc::hk::EnvironmentRegionOp region) {
  auto mod = region->getParentOfType<mlir::ModuleOp>();
  if (!mod)
    return region.emitError("No parent module");

  auto groupShape = getTypeAttr<hc::typing::SequenceType>(
      mod, hc::hk::getKernelGroupShapeAttrName());
  if (!groupShape)
    return region->emitError("Group shape is not defined");

  auto localId = getTypeAttr<hc::typing::SequenceType>(
      mod, hc::hk::getKernelLocalIdAttrName());
  if (!localId)
    return region->emitError("Local ID is not defined");

  if (groupShape.size() != localId.size())
    return region->emitError("Invalid group shape");

  auto subgroupId = getTypeAttr<hc::typing::SymbolicTypeBase>(
      mod, hc::hk::getKernelSubgroupIdAttrName());
  if (!subgroupId)
    return region->emitError("Subgroup ID is not defined");

  auto subgroupSize = getTypeAttr<hc::typing::SymbolicTypeBase>(
      mod, hc::hk::getKernelSubgroupSizeAttrName());
  if (!subgroupSize)
    return region->emitError("Subgroup size is not defined");

  llvm::SmallVector<mlir::Type> ids(localId.getParams());
  ids.back() = subgroupId;

  if (mlir::failed(
          lowerScope(region, groupShape.getParams(), ids, subgroupSize, true)))
    return mlir::failure();

  region.setEnvironmentAttr(
      hc::hk::SubgroupScopeAttr::get(region.getContext()));
  return mlir::success();
}

static mlir::LogicalResult lowerSGScope(hc::hk::EnvironmentRegionOp region) {
  auto mod = region->getParentOfType<mlir::ModuleOp>();
  if (!mod)
    return region.emitError("No parent module");

  auto localId = getTypeAttr<hc::typing::SequenceType>(
      mod, hc::hk::getKernelLocalIdAttrName());
  if (!localId)
    return region->emitError("Local ID is not defined");

  auto subgroupSize = getTypeAttr<hc::typing::SymbolicTypeBase>(
      mod, hc::hk::getKernelSubgroupSizeAttrName());
  if (!subgroupSize)
    return region->emitError("Subgroup size is not defined");

  if (mlir::failed(lowerScope(region, subgroupSize, localId.getParams().back(),
                              subgroupSize, false)))
    return mlir::failure();

  region.setEnvironmentAttr(
      hc::hk::WorkitemScopeAttr::get(region.getContext()));
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
    (void)applyPatternsGreedily(getOperation(), {});
  }
};

struct LowerSubgroupScopePass final
    : public hc::impl::LowerSubgroupScopePassBase<LowerSubgroupScopePass> {

  void runOnOperation() override {
    auto visitor = [&](hc::hk::EnvironmentRegionOp region) -> mlir::WalkResult {
      if (!mlir::isa<hc::hk::SubgroupScopeAttr>(region.getEnvironment()))
        return mlir::WalkResult::advance();

      if (mlir::failed(lowerSGScope(region)))
        return mlir::WalkResult::interrupt();

      return mlir::WalkResult::skip();
    };

    if (getOperation()
            ->walk<mlir::WalkOrder::PostOrder>(visitor)
            .wasInterrupted())
      return signalPassFailure();

    // DCE
    (void)applyPatternsGreedily(getOperation(), {});
  }
};
} // namespace
