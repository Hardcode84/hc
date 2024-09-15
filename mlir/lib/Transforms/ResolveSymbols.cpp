// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hc/Transforms/Passes.hpp"

#include "hc/Dialect/HKernel/IR/HKernelOps.hpp"
#include "hc/Dialect/Typing/IR/TypingOps.hpp"

#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Arith/Utils/Utils.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace hc {
#define GEN_PASS_DEF_RESOLVEARGSPASS
#include "hc/Transforms/Passes.h.inc"
} // namespace hc

static hc::typing::SymbolicTypeBase foldSymbolic(mlir::Type type) {
  if (auto symbolic = mlir::dyn_cast<hc::typing::SymbolicTypeBase>(type))
    return hc::typing::SymbolicTypeBase::foldExpr(symbolic);

  return nullptr;
}

static std::optional<int64_t> getSymbolicLiteral(mlir::Type type) {
  auto lit =
      mlir::dyn_cast_if_present<hc::typing::LiteralType>(foldSymbolic(type));
  if (!lit)
    return std::nullopt;

  auto attr = mlir::dyn_cast<mlir::IntegerAttr>(lit.getValue());
  if (!attr)
    return std::nullopt;

  return attr.getInt();
}

static mlir::SmallVector<int64_t> convertShape(mlir::TypeRange symbolic) {
  llvm::SmallVector<int64_t> shape(symbolic.size());
  for (auto &&[i, s] : llvm::enumerate(symbolic)) {
    if (auto lit = getSymbolicLiteral(s)) {
      shape[i] = *lit;
    } else {
      shape[i] = mlir::ShapedType::kDynamic;
    }
  }
  return shape;
}

static mlir::Type convertElemType(const mlir::TypeConverter &converter,
                                  mlir::Type type) {
  // TODO: hack until we have proper sym replacement
  if (mlir::isa<hc::typing::LiteralType>(type))
    return mlir::Float32Type::get(type.getContext());

  auto elemType = converter.convertType(type);
  if (!elemType || !mlir::MemRefType::isValidElementType(elemType))
    return nullptr;

  return elemType;
}

static void populateTypeConverter(mlir::TypeConverter &converter) {
  // Convert unknown types to itself
  converter.addConversion([](mlir::Type type) { return type; });

  converter.addConversion([&](hc::hk::BufferType type) -> mlir::Type {
    auto elemType = convertElemType(converter, type.getElementType());
    if (!elemType)
      return nullptr;

    return mlir::MemRefType::get(convertShape(type.getShape()), elemType);
  });

  converter.addConversion([](hc::typing::SymbolicTypeBase t) -> mlir::Type {
    return mlir::IndexType::get(t.getContext());
  });
}

static bool getSeq(mlir::Operation *op, mlir::StringRef name,
                   mlir::TypeRange &ret) {
  auto attr = op->getAttrOfType<hc::typing::TypeAttr>(name);
  if (!attr)
    return false;

  auto seq = mlir::dyn_cast<hc::typing::SequenceType>(attr.getTypeVal());
  if (!seq)
    return false;

  if (!llvm::all_of(seq.getParams(), [](mlir::Type t) {
        return mlir::isa<hc::typing::SymbolicTypeBase>(t);
      }))
    return false;

  ret = seq.getParams();
  return true;
}

static mlir::Type getTypeAttr(mlir::Operation *op, mlir::StringRef name) {
  auto attr = op->getAttrOfType<hc::typing::TypeAttr>(name);
  if (!attr)
    return nullptr;

  return attr.getTypeVal();
}

static mlir::Operation *doCast(mlir::OpBuilder &builder, mlir::Location loc,
                               mlir::Value src, mlir::Type newType) {
  return builder.create<mlir::UnrealizedConversionCastOp>(loc, newType, src);
}

using SymbolsMapType = llvm::SmallDenseMap<mlir::Type, mlir::Value>;

static void handleBufferType(mlir::OpBuilder &builder, mlir::Location loc,
                             mlir::Value arg, hc::hk::BufferType type,
                             SymbolsMapType &symbolsMap) {
  for (auto &&[i, s] : llvm::enumerate(type.getShape())) {
    auto sym = mlir::dyn_cast<hc::typing::SymbolType>(s);
    if (!sym)
      continue;

    if (symbolsMap.contains(sym))
      continue;

    mlir::Value val = builder.create<mlir::memref::DimOp>(loc, arg, i);
    symbolsMap[sym] = val;
  }
}

static void handleArgType(mlir::OpBuilder &builder, mlir::Location loc,
                          mlir::Value arg, mlir::Type type,
                          SymbolsMapType &symbolsMap) {
  if (auto buffer = mlir::dyn_cast<hc::hk::BufferType>(type))
    return handleBufferType(builder, loc, arg, buffer, symbolsMap);
}

static mlir::Value resolveSymbol(mlir::OpBuilder &builder, mlir::Location loc,
                                 mlir::Type type, SymbolsMapType &symbolsMap) {
  auto it = symbolsMap.find(type);
  if (it != symbolsMap.end())
    return it->second;

  auto folded = foldSymbolic(type);
  if (folded) {
    if (auto lit = mlir::dyn_cast<hc::typing::LiteralType>(folded)) {
      auto attr = mlir::dyn_cast<mlir::IntegerAttr>(lit.getValue());
      if (!attr) {
        mlir::emitError(builder.getUnknownLoc())
            << "Invalid attr type: " << lit;
        return nullptr;
      }

      mlir::Value ret =
          builder.create<mlir::arith::ConstantIndexOp>(loc, attr.getInt());
      symbolsMap[type] = ret;
      return ret;
    }

    if (auto expr = mlir::dyn_cast<hc::typing::ExprType>(folded)) {
      llvm::SmallVector<mlir::OpFoldResult> args;
      for (auto param : expr.getParams()) {
        auto res = resolveSymbol(builder, loc, param, symbolsMap);
        if (!res)
          return nullptr;

        args.emplace_back(res);
      }
      mlir::Value ret = mlir::getValueOrCreateConstantIndexOp(
          builder, loc,
          mlir::affine::makeComposedFoldedAffineApply(builder, loc,
                                                      expr.getExpr(), args));
      symbolsMap[type] = ret;
      return ret;
    }
  }
  mlir::emitError(builder.getUnknownLoc()) << "Invalid expr: " << folded;
  return nullptr;
}

static mlir::gpu::LaunchOp
createLauncOp(mlir::OpBuilder &builder, mlir::Location loc,
              mlir::TypeRange workShape, mlir::TypeRange groupShape,
              mlir::TypeRange groupCount, mlir::TypeRange groupId,
              mlir::TypeRange localId, SymbolsMapType &symbolsMap) {
  auto rank = workShape.size();
  assert(rank > 0 && rank <= 3 && "Invalid rank");
  mlir::ValueRange suggestedGroupSize;

  auto getSuggestGroupSize = [&](unsigned id) -> mlir::Value {
    if (suggestedGroupSize.empty()) {
      llvm::SmallVector<mlir::Value, 3> workShapeVals;
      for (auto w : llvm::reverse(workShape)) {
        mlir::Value res = resolveSymbol(builder, loc, w, symbolsMap);
        if (!res) {
          mlir::emitError(builder.getUnknownLoc())
              << "Failed to resolve workShape expr: " << w;
          return nullptr;
        }

        workShapeVals.emplace_back(res);
      }
      suggestedGroupSize =
          builder.create<hc::hk::SuggestBlockSizeOp>(loc, workShapeVals)
              .getResults();
    }

    return suggestedGroupSize[id];
  };

  llvm::SmallVector<mlir::Value, 3> gridSize;
  llvm::SmallVector<mlir::Value, 3> blockSize;
  for (auto &&[i, count, shape] : llvm::enumerate(groupCount, groupShape)) {
    mlir::Value shapeVal;
    if (mlir::isa<hc::typing::SymbolType>(shape) &&
        !symbolsMap.contains(shape)) {
      shapeVal = getSuggestGroupSize(rank - i - 1);
      symbolsMap[shape] = shapeVal;
    } else {
      shapeVal = resolveSymbol(builder, loc, shape, symbolsMap);
    }

    if (!shapeVal) {
      mlir::emitError(builder.getUnknownLoc())
          << "Failed to resolve shape expr: " << shape;
      return nullptr;
    }

    mlir::Value countVal = resolveSymbol(builder, loc, count, symbolsMap);
    if (!countVal) {
      mlir::emitError(builder.getUnknownLoc())
          << "Failed to resolve count expr: " << count;
      return nullptr;
    }

    gridSize.emplace_back(countVal);
    blockSize.emplace_back(shapeVal);
  }

  std::reverse(gridSize.begin(), gridSize.end());
  std::reverse(blockSize.begin(), blockSize.end());

  if (gridSize.size() < 3) {
    mlir::Value one = builder.create<mlir::arith::ConstantIndexOp>(loc, 1);
    gridSize.resize(3, one);
    blockSize.resize(3, one);
  }

  auto op = builder.create<mlir::gpu::LaunchOp>(loc, gridSize[0], gridSize[1],
                                                gridSize[2], blockSize[0],
                                                blockSize[1], blockSize[2]);
  auto dim3 = [](const mlir::gpu::KernelDim3 &k) -> std::array<mlir::Value, 3> {
    return {k.z, k.y, k.z};
  };
  auto launchGrIds = dim3(op.getBlockIds());
  auto launchLocIds = dim3(op.getThreadIds());
  for (auto &&[grId, locId, lgrId, llocId] :
       llvm::zip_first(llvm::reverse(groupId), llvm::reverse(localId),
                       launchGrIds, launchLocIds)) {
    symbolsMap[grId] = lgrId;
    symbolsMap[locId] = llocId;
  }
  return op;
}

template <typename T> T getParentOrSelf(mlir::Operation *op) {
  auto ret = mlir::dyn_cast<T>(op);
  if (ret)
    return ret;

  return op->getParentOfType<T>();
}

namespace {
struct ResolveArgsPass final
    : public hc::impl::ResolveArgsPassBase<ResolveArgsPass> {

  void runOnOperation() override {
    auto mod = getParentOrSelf<mlir::ModuleOp>(getOperation());
    if (!mod) {
      getOperation()->emitError("No parent module");
      return signalPassFailure();
    }

    mlir::TypeRange workShape;
    if (!getSeq(mod, "kernel.work_shape", workShape)) {
      mod.emitError("No work shape attr");
      return signalPassFailure();
    }

    mlir::TypeRange groupShape;
    if (!getSeq(mod, "kernel.group_shape", groupShape)) {
      mod.emitError("No group shape attr");
      return signalPassFailure();
    }

    mlir::TypeRange groupCount;
    if (!getSeq(mod, "kernel.group_count", groupCount)) {
      mod.emitError("No group vount attr");
      return signalPassFailure();
    }

    mlir::TypeRange groupId;
    if (!getSeq(mod, "kernel.group_id", groupId)) {
      mod.emitError("No group id attr");
      return signalPassFailure();
    }

    mlir::TypeRange localId;
    if (!getSeq(mod, "kernel.local_id", localId)) {
      mod.emitError("No local id attr");
      return signalPassFailure();
    }

    auto rank = workShape.size();
    if (rank < 1 || rank > 3 || groupShape.size() != rank ||
        groupShape.size() != rank || groupId.size() != rank ||
        localId.size() != rank) {
      mod.emitError("Invalid ids rank");
      return signalPassFailure();
    }

    auto subgroupSize = getTypeAttr(mod, "kernel.subgroup_size");
    if (!subgroupSize) {
      mod.emitError("No subgroup size");
      return signalPassFailure();
    }

    mlir::TypeConverter converter;
    populateTypeConverter(converter);

    SymbolsMapType symbolsMap;

    mlir::DominanceInfo dom;
    mlir::IRRewriter builder(&getContext());
    auto visitor = [&](mlir::func::FuncOp func) -> mlir::WalkResult {
      if (func.isDeclaration())
        return mlir::WalkResult::skip();

      if (!func.getResultTypes().empty()) {
        func.emitError("Function must not return values");
        return mlir::WalkResult::interrupt();
      }

      if (!llvm::hasSingleElement(func.getFunctionBody())) {
        func.emitError("Function must have single block");
        return mlir::WalkResult::interrupt();
      }

      mlir::Block &body = func.getFunctionBody().front();
      builder.setInsertionPointToStart(&body);
      for (auto &&[i, arg] : llvm::enumerate(func.getArguments())) {
        mlir::Type oldType = arg.getType();
        mlir::Type newType = converter.convertType(oldType);
        if (!newType) {
          func.emitError("Cannnot convert arg type: ") << i << " " << oldType;
          return mlir::WalkResult::interrupt();
        }
        mlir::Value newArg = arg;
        if (oldType != newType) {
          arg.setType(newType);
          mlir::Operation *op = doCast(builder, arg.getLoc(), arg, oldType);
          newArg = op->getResult(0);
          arg.replaceAllUsesExcept(newArg, op);
        }
        handleArgType(builder, arg.getLoc(), arg, oldType, symbolsMap);
      }

      auto newFuncType = func.getFunctionType().clone(
          mlir::ValueRange(func.getArguments()).getTypes(), {});
      func.setFunctionType(newFuncType);

      mlir::gpu::LaunchOp launch =
          createLauncOp(builder, func.getLoc(), workShape, groupShape,
                        groupCount, groupId, localId, symbolsMap);
      if (!launch) {
        func.emitError("Failed to create launch op");
        return mlir::WalkResult::interrupt();
      }

      auto termLoc = body.getTerminator()->getLoc();

      mlir::Block *newLaunchBody =
          builder.splitBlock(&body, std::next(launch->getIterator()));
      mlir::Block *oldLaunchBody = &launch.getBody().front();
      builder.mergeBlocks(newLaunchBody, oldLaunchBody);

      builder.setInsertionPointAfter(launch);
      builder.create<mlir::func::ReturnOp>(termLoc);

      builder.setInsertionPoint(oldLaunchBody->getTerminator());
      builder.create<mlir::gpu::TerminatorOp>(termLoc);
      builder.eraseOp(oldLaunchBody->getTerminator());

      // TODO: unhardcode
      if (!symbolsMap.contains(subgroupSize)) {
        builder.setInsertionPoint(launch);
        symbolsMap[subgroupSize] = builder.create<mlir::arith::ConstantIndexOp>(
            builder.getUnknownLoc(), 64);
      }

      auto replaceMaterialize =
          [&](hc::hk::MaterializeExprOp mat) -> mlir::WalkResult {
        auto type = mat.getType();
        mlir::Location loc = mat.getLoc();
        builder.setInsertionPoint(mat);
        mlir::Value expr = resolveSymbol(builder, loc, type, symbolsMap);
        if (!expr) {
          mat.emitError("Failed to materialize expr");
          return mlir::WalkResult::interrupt();
        }
        expr = doCast(builder, loc, expr, type)->getResult(0);
        builder.replaceAllUsesWith(mat, expr);
        builder.eraseOp(mat);
        return mlir::WalkResult::advance();
      };

      if (launch.getBody()
              .walk<mlir::WalkOrder::PostOrder>(replaceMaterialize)
              .wasInterrupted())
        return mlir::WalkResult::interrupt();

      return mlir::WalkResult::skip();
    };

    if (getOperation()
            ->walk<mlir::WalkOrder::PostOrder>(visitor)
            .wasInterrupted())
      return signalPassFailure();

    getOperation()->walk<mlir::WalkOrder::PostOrder>(
        [&](hc::hk::EnvironmentRegionOp reg) {
          if (!mlir::isa<hc::hk::WorkgroupScopeAttr, hc::hk::SubgroupScopeAttr,
                         hc::hk::WorkitemScopeAttr>(reg.getEnvironmentAttr()))
            return;

          hc::hk::EnvironmentRegionOp::inlineIntoParent(builder, reg);
        });

    // DCE
    (void)applyPatternsAndFoldGreedily(getOperation(), {});
  }
};
} // namespace
