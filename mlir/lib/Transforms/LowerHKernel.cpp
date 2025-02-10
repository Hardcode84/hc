// SPDX-FileCopyrightText: 2024 The HC Authors
// SPDX-FileCopyrightText: 2025 The HC Authors
//
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
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/UB/IR/UBOps.h>
#include <mlir/Dialect/Vector/IR/VectorOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace hc {
#define GEN_PASS_DEF_RESOLVEARGSPASS
#include "hc/Transforms/Passes.h.inc"

#define GEN_PASS_DEF_LOWERHKERNELOPSPASS
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

static mlir::SmallVector<int64_t> convertShape(mlir::TypeRange symbolic,
                                               mlir::TypeConverter &converter) {
  llvm::SmallVector<int64_t> shape(symbolic.size());
  for (auto &&[i, s] : llvm::enumerate(symbolic)) {
    if (auto lit = getSymbolicLiteral(s)) {
      shape[i] = *lit;
    } else if (auto lit = getSymbolicLiteral(converter.convertType(s))) {
      shape[i] = *lit;
    } else {
      shape[i] = mlir::ShapedType::kDynamic;
    }
  }
  return shape;
}

static mlir::StridedLayoutAttr getStridedLayout(mlir::MLIRContext *ctx,
                                                size_t numDims) {
  auto d = mlir::ShapedType::kDynamic;
  llvm::SmallVector<int64_t> strides(numDims, d);
  return mlir::StridedLayoutAttr::get(ctx, d, strides);
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

static mlir::gpu::AddressSpaceAttr getWGAddrSpace(mlir::MLIRContext *ctx) {
  return mlir::gpu::AddressSpaceAttr::get(
      ctx, mlir::gpu::GPUDialect::getWorkgroupAddressSpace());
}

static void populateTypeConverter(mlir::TypeConverter &converter) {
  converter.addConversion([&](hc::hk::BufferType type) -> mlir::Type {
    auto elemType = convertElemType(converter, type.getElementType());
    if (!elemType)
      return nullptr;

    mlir::TypeRange shape = type.getShape();
    auto layout = getStridedLayout(type.getContext(), shape.size());
    return mlir::MemRefType::get(convertShape(shape, converter), elemType,
                                 layout);
  });

  converter.addConversion([&](hc::hk::TensorType type) -> mlir::Type {
    auto elemType = convertElemType(converter, type.getElementType());
    if (!elemType)
      return nullptr;

    auto maskElemType = mlir::IntegerType::get(type.getContext(), 1);

    mlir::TypeRange shape = type.getShape();
    auto layout = getStridedLayout(type.getContext(), shape.size());
    auto addrSpace = getWGAddrSpace(type.getContext());
    auto convertedShape = convertShape(shape, converter);
    auto dataType =
        mlir::MemRefType::get(convertedShape, elemType, layout, addrSpace);
    auto maskType =
        mlir::MemRefType::get(convertedShape, maskElemType, layout, addrSpace);
    return mlir::TupleType::get(type.getContext(), {dataType, maskType});
  });

  converter.addConversion([&](hc::hk::VectorType type) -> mlir::Type {
    auto elemType = convertElemType(converter, type.getElementType());
    if (!elemType)
      return nullptr;

    mlir::TypeRange shape = type.getShape();
    auto convertedShape = convertShape(shape, converter);
    if (mlir::ShapedType::isDynamicShape(convertedShape))
      return nullptr;

    auto maskElemType = mlir::IntegerType::get(type.getContext(), 1);
    auto dataType = mlir::VectorType::get(convertedShape, elemType);
    auto maskType = mlir::VectorType::get(convertedShape, maskElemType);
    return mlir::TupleType::get(type.getContext(), {dataType, maskType});
  });

  converter.addConversion([](hc::typing::SymbolicTypeBase t) -> mlir::Type {
    return mlir::IndexType::get(t.getContext());
  });

  converter.addConversion([&](hc::hk::SliceType s) -> mlir::Type {
    auto lower = converter.convertType(s.getLower());
    if (!lower)
      return nullptr;

    auto upper = converter.convertType(s.getUpper());
    if (!upper)
      return nullptr;

    auto step = converter.convertType(s.getStep());
    if (!step)
      return nullptr;

    return mlir::TupleType::get(s.getContext(), {lower, upper, step});
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

  if (auto sym = mlir::dyn_cast<hc::typing::SymbolicTypeBase>(type)) {
    symbolsMap[sym] = arg;
    return;
  }
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

    mlir::Type indexType = builder.getIndexType();
    if (auto expr = mlir::dyn_cast<hc::typing::ExprType>(folded)) {
      llvm::SmallVector<mlir::OpFoldResult> args;
      for (auto param : expr.getParams()) {
        mlir::Value res = resolveSymbol(builder, loc, param, symbolsMap);
        if (!res)
          return nullptr;

        if (auto literal =
                mlir::dyn_cast<hc::typing::LiteralType>(res.getType())) {
          auto attr = literal.getValue();
          auto di = builder.getContext()
                        ->getLoadedDialect<mlir::arith::ArithDialect>();
          auto c = di->materializeConstant(builder, attr, attr.getType(), loc);
          if (!c || c->getNumResults() != 1)
            return nullptr;

          res = c->getResult(0);
        }

        if (!res.getType().isIntOrIndex())
          return nullptr;

        if (res.getType() != indexType)
          res = builder.create<mlir::arith::IndexCastOp>(loc, indexType, res);

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
    return {k.x, k.y, k.z};
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

static mlir::Type makeSignless(mlir::Type type) {
  auto intType = mlir::dyn_cast<mlir::IntegerType>(type);
  if (!intType || intType.isSignless())
    return type;

  return mlir::IntegerType::get(type.getContext(), intType.getWidth());
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
    if (!getSeq(mod, hc::hk::getKernelWorkShapeAttrName(), workShape)) {
      mod.emitError("No work shape attr");
      return signalPassFailure();
    }

    mlir::TypeRange groupShape;
    if (!getSeq(mod, hc::hk::getKernelGroupShapeAttrName(), groupShape)) {
      mod.emitError("No group shape attr");
      return signalPassFailure();
    }

    mlir::TypeRange groupCount;
    if (!getSeq(mod, hc::hk::getKernelGroupCountAttrName(), groupCount)) {
      mod.emitError("No group vount attr");
      return signalPassFailure();
    }

    mlir::TypeRange groupId;
    if (!getSeq(mod, hc::hk::getKernelGroupIdAttrName(), groupId)) {
      mod.emitError("No group id attr");
      return signalPassFailure();
    }

    mlir::TypeRange localId;
    if (!getSeq(mod, hc::hk::getKernelLocalIdAttrName(), localId)) {
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

    auto subgroupSize =
        getTypeAttr(mod, hc::hk::getKernelSubgroupSizeAttrName());
    if (!subgroupSize) {
      mod.emitError("No subgroup size");
      return signalPassFailure();
    }

    llvm::SmallDenseMap<mlir::Type, mlir::Type> metadataMap;
    if (auto metadata = mod->getAttrOfType<mlir::ArrayAttr>(
            hc::hk::getKernelMetadataAttrName())) {
      if (metadata.size() % 2 != 0) {
        mod.emitError("Invalid metadata size");
        return signalPassFailure();
      }

      for (auto i : llvm::seq<size_t>(0, metadata.size() / 2)) {
        auto keyAttr = mlir::dyn_cast<hc::typing::TypeAttr>(metadata[i * 2]);
        auto valAttr =
            mlir::dyn_cast<hc::typing::TypeAttr>(metadata[i * 2 + 1]);
        if (!keyAttr || !valAttr) {
          mod.emitError("Invalid metadata");
          return signalPassFailure();
        }
        metadataMap.insert({keyAttr.getTypeVal(), valAttr.getTypeVal()});
      }
    }

    mlir::TypeConverter converter;
    // Convert unknown types to itself
    converter.addConversion([](mlir::Type type) { return type; });
    populateTypeConverter(converter);
    converter.addConversion([&](mlir::Type type) -> std::optional<mlir::Type> {
      auto it = metadataMap.find(type);
      if (it != metadataMap.end())
        return makeSignless(it->second);

      return std::nullopt;
    });

    SymbolsMapType symbolsMap;

    mlir::DominanceInfo dom;
    mlir::IRRewriter builder(&getContext());

    auto entrypointAttrName =
        builder.getStringAttr(hc::hk::getKernelEntryPointAttrName());
    auto visitor = [&](mlir::func::FuncOp func) -> mlir::WalkResult {
      if (func.isDeclaration() || !func->hasAttr(entrypointAttrName))
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

      mlir::Type indexType = builder.getIndexType();
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
        if (expr.getType() != indexType)
          expr = builder.create<mlir::arith::IndexCastOp>(loc, indexType, expr);

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
    (void)applyPatternsGreedily(getOperation(), {});
  }
};
} // namespace

namespace {
struct ConvertTypes final : mlir::ConversionPattern {

  ConvertTypes(const mlir::TypeConverter &converter, mlir::MLIRContext *ctx,
               mlir::PatternBenefit benefit = 0)
      : mlir::ConversionPattern(converter, mlir::Pattern::MatchAnyOpTypeTag{},
                                benefit, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op, mlir::ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::FailureOr<mlir::Operation *> newOp =
        mlir::convertOpResultTypes(op, operands, *getTypeConverter(), rewriter);
    if (failed(newOp))
      return mlir::failure();

    rewriter.replaceOp(op, (*newOp)->getResults());
    return mlir::success();
  }
};

struct ConvertMakeSlice final
    : public mlir::OpConversionPattern<hc::hk::MakeSliceOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(hc::hk::MakeSliceOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    const mlir::TypeConverter *converter = getTypeConverter();
    auto resType = converter->convertType<mlir::TupleType>(op.getType());
    if (!resType || resType.size() != 3)
      return rewriter.notifyMatchFailure(op, "Invalid result type");

    auto loc = op.getLoc();
    mlir::Value lower = adaptor.getLower();
    if (!lower)
      lower = rewriter.create<mlir::ub::PoisonOp>(loc, resType.getType(0));

    mlir::Value upper = adaptor.getUpper();
    if (!upper)
      upper = rewriter.create<mlir::ub::PoisonOp>(loc, resType.getType(1));

    mlir::Value step = adaptor.getUpper();
    if (!step)
      step = rewriter.create<mlir::ub::PoisonOp>(loc, resType.getType(2));

    mlir::Value args[] = {lower, upper, step};
    rewriter.replaceOpWithNewOp<hc::hk::MakeTupleOp>(op, resType, args);
    return mlir::success();
  }
};

static bool checkIsMemref(mlir::Type type) {
  auto check = [](mlir::Type t) { return mlir::isa<mlir::MemRefType>(t); };

  if (auto tuple = mlir::dyn_cast<mlir::TupleType>(type))
    return llvm::all_of(tuple.getTypes(), check);

  return check(type);
}

static mlir::Value getDim(mlir::OpBuilder &builder, mlir::Location loc,
                          mlir::Value src, int64_t dim) {
  if (auto tuple = mlir::dyn_cast<mlir::TupleType>(src.getType())) {
    if (tuple.size() == 0)
      return nullptr;

    mlir::Value id = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
    src =
        builder.create<hc::hk::TupleExtractOp>(loc, tuple.getType(0), src, id);
  }

  if (mlir::isa<mlir::MemRefType>(src.getType()))
    return builder.create<mlir::memref::DimOp>(loc, src, dim);

  return nullptr;
}

static mlir::Value makeSubview(mlir::OpBuilder &builder, mlir::Location loc,
                               mlir::Value src,
                               mlir::ArrayRef<mlir::OpFoldResult> offsets,
                               mlir::ArrayRef<mlir::OpFoldResult> sizes,
                               mlir::ArrayRef<mlir::OpFoldResult> strides,
                               mlir::Type resType) {
  if (mlir::isa<mlir::MemRefType>(src.getType()) &&
      mlir::isa<mlir::MemRefType>(resType)) {
    auto srcType = mlir::cast<mlir::MemRefType>(src.getType());
    auto dstType = mlir::cast<mlir::MemRefType>(resType);

    mlir::Type subviewType;
    if (srcType.getRank() == dstType.getRank()) {
      subviewType = mlir::memref::SubViewOp::inferResultType(srcType, offsets,
                                                             sizes, strides);
    } else if (srcType.getRank() > dstType.getRank()) {
      subviewType = mlir::memref::SubViewOp::inferRankReducedResultType(
          dstType.getShape(), srcType, offsets, sizes, strides);
    } else {
      return nullptr;
    }
    mlir::Value res = builder.create<mlir::memref::SubViewOp>(
        loc, mlir::cast<mlir::MemRefType>(subviewType), src, offsets, sizes,
        strides);
    if (res.getType() != resType)
      res = builder.create<mlir::memref::CastOp>(loc, resType, res);

    return res;
  }
  return nullptr;
}

static std::tuple<mlir::Value, mlir::Value, mlir::Value>
createResolveSlice(mlir::OpBuilder &builder, mlir::Location loc,
                   mlir::Value size, mlir::Value src) {
  mlir::Value lower, upper, step;
  if (auto tuple = mlir::dyn_cast<mlir::TupleType>(src.getType())) {
    mlir::Value id0 = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
    mlir::Value id1 = builder.create<mlir::arith::ConstantIndexOp>(loc, 1);
    mlir::Value id2 = builder.create<mlir::arith::ConstantIndexOp>(loc, 2);

    if (!mlir::isa<mlir::NoneType>(tuple.getType(0)))
      lower = builder.create<hc::hk::TupleExtractOp>(loc, tuple.getType(0), src,
                                                     id0);
    if (!mlir::isa<mlir::NoneType>(tuple.getType(1)))
      upper = builder.create<hc::hk::TupleExtractOp>(loc, tuple.getType(1), src,
                                                     id1);
    if (!mlir::isa<mlir::NoneType>(tuple.getType(2)))
      step = builder.create<hc::hk::TupleExtractOp>(loc, tuple.getType(2), src,
                                                    id2);
  } else {
    lower = src;
  }

  auto op =
      builder.create<hc::hk::ResolveSliceOp>(loc, size, lower, upper, step);
  return {op.getOffset(), op.getSize(), op.getStride()};
}

struct ConvertSubview final
    : public mlir::OpConversionPattern<hc::hk::SubViewOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(hc::hk::SubViewOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Value src = adaptor.getSource();
    if (!checkIsMemref(src.getType()))
      return rewriter.notifyMatchFailure(op, "Invalid source type");

    auto srcSymbolic = mlir::dyn_cast<hc::hk::SymbolicallyShapedType>(
        op.getSource().getType());
    if (!srcSymbolic)
      return rewriter.notifyMatchFailure(op,
                                         "Failed to get source symbolic shape");

    mlir::TypeRange srcSymbolicShape = srcSymbolic.getShape();

    const mlir::TypeConverter *converter = getTypeConverter();
    assert(converter);

    mlir::Type resType = converter->convertType(op.getType());
    if (!resType)
      return rewriter.notifyMatchFailure(op, "Failed to convert result type");

    mlir::Location loc = op.getLoc();
    llvm::SmallVector<mlir::OpFoldResult> offsets;
    llvm::SmallVector<mlir::OpFoldResult> sizes;
    llvm::SmallVector<mlir::OpFoldResult> strides;
    for (auto &&[i, origIdx, idx] :
         llvm::enumerate(op.getIndex(), adaptor.getIndex())) {
      if (!mlir::isa<hc::hk::SliceType>(origIdx.getType()) &&
          !mlir::isa<mlir::IndexType>(idx.getType()))
        return rewriter.notifyMatchFailure(op, "Invalid slice type");

      mlir::Value dim;
      if (auto symbolcDim = mlir::dyn_cast<hc::typing::SymbolicTypeBase>(
              srcSymbolicShape[i])) {
        dim = rewriter.create<hc::hk::MaterializeExprOp>(loc, symbolcDim);
        dim = converter->materializeTargetConversion(
            rewriter, loc, rewriter.getIndexType(), dim);
      } else {
        dim = getDim(rewriter, loc, src, int64_t(i));
      }
      if (!dim)
        return rewriter.notifyMatchFailure(op, "Failed to get dim");

      auto &&[offset, size, stride] =
          createResolveSlice(rewriter, loc, dim, idx);
      offsets.emplace_back(offset);
      sizes.emplace_back(size);
      strides.emplace_back(stride);
    }

    mlir::TypeRange resTypes;
    llvm::SmallVector<mlir::Value> buffers;
    if (auto tuple = mlir::dyn_cast<mlir::TupleType>(src.getType())) {
      auto resTuple = mlir::dyn_cast<mlir::TupleType>(resType);
      if (!resTuple || resTuple.size() != tuple.size())
        return rewriter.notifyMatchFailure(op, "Invalid result type");

      resTypes = resTuple.getTypes();
      for (auto &&[i, type] : llvm::enumerate(tuple.getTypes())) {
        mlir::Value id =
            rewriter.create<mlir::arith::ConstantIndexOp>(loc, int64_t(i));
        mlir::Value val =
            rewriter.create<hc::hk::TupleExtractOp>(loc, type, src, id);
        buffers.emplace_back(val);
      }
    } else {
      resTypes = resType;
      buffers.emplace_back(src);
    }

    llvm::SmallVector<mlir::Value> results;
    for (auto &&[resType, buff] : llvm::zip_equal(resTypes, buffers)) {
      mlir::Value res =
          makeSubview(rewriter, loc, buff, offsets, sizes, strides, resType);
      if (!res)
        return rewriter.notifyMatchFailure(op, "Failed to create subview");

      results.emplace_back(res);
    }

    assert(!results.empty());
    mlir::Value res;
    if (results.size() > 1) {
      res = rewriter.create<hc::hk::MakeTupleOp>(loc, resType, results);
    } else {
      res = results.front();
    }
    rewriter.replaceOp(op, res);
    return mlir::success();
  }
};

template <typename T>
static llvm::SmallVector<mlir::TypedValue<T>>
unpackTuple(mlir::OpBuilder &builder, mlir::Location loc, mlir::Value src) {
  llvm::SmallVector<mlir::TypedValue<T>> ret;
  if (auto tuple = mlir::dyn_cast<mlir::TupleType>(src.getType())) {
    for (auto &&[i, type] : llvm::enumerate(tuple.getTypes())) {
      if (!mlir::isa<T>(type))
        return {};

      mlir::Value idx =
          builder.create<mlir::arith::ConstantIndexOp>(loc, int64_t(i));
      mlir::Value val =
          builder.create<hc::hk::TupleExtractOp>(loc, type, src, idx);
      ret.emplace_back(mlir::cast<mlir::TypedValue<T>>(val));
    }
  } else if (mlir::isa<T>(src.getType())) {
    ret.emplace_back(mlir::cast<mlir::TypedValue<T>>(src));
  }
  return ret;
}

static mlir::Value shapedCast(mlir::OpBuilder &builder, mlir::Location loc,
                              mlir::Value src, mlir::Type dstType) {
  auto srcType = src.getType();
  if (srcType == dstType)
    return src;

  if (mlir::isa<mlir::MemRefType>(srcType) &&
      mlir::isa<mlir::MemRefType>(dstType))
    return builder.create<mlir::memref::CastOp>(loc, dstType, src);

  return nullptr;
}

static mlir::Value packTuple(mlir::OpBuilder &builder, mlir::Location loc,
                             mlir::ValueRange args, mlir::Type type) {
  assert(!args.empty());
  if (args.size() == 1)
    return shapedCast(builder, loc, args.front(), type);

  auto tuple = mlir::dyn_cast<mlir::TupleType>(type);
  if (!tuple)
    return nullptr;

  llvm::SmallVector<mlir::Value> vals;
  for (auto &&[src, dstType] : llvm::zip_equal(args, tuple.getTypes())) {
    mlir::Value val = shapedCast(builder, loc, src, dstType);
    if (!val)
      return nullptr;

    vals.emplace_back(val);
  }
  return builder.create<hc::hk::MakeTupleOp>(loc, tuple, vals);
}

static llvm::SmallVector<mlir::Value> createAlloc(mlir::OpBuilder &builder,
                                                  mlir::Location loc,
                                                  mlir::Type resType,
                                                  mlir::ValueRange shape) {
  llvm::SmallVector<mlir::MemRefType> types;
  if (auto tuple = mlir::dyn_cast<mlir::TupleType>(resType)) {
    for (auto t : tuple.getTypes()) {
      auto mem = mlir::dyn_cast<mlir::MemRefType>(t);
      if (!mem)
        return {};

      types.emplace_back(mem);
    }
  } else if (auto mem = mlir::dyn_cast<mlir::MemRefType>(resType)) {
    types.emplace_back(mem);
  } else {
    return {};
  }

  auto expandAttrName =
      builder.getStringAttr(hc::hk::getKernelAllocExpandAttrName());
  auto unit = builder.getUnitAttr();
  auto wgSpace = getWGAddrSpace(builder.getContext());

  llvm::SmallVector<mlir::Value> ret;
  llvm::SmallVector<mlir::Value> args;
  for (auto type : types) {
    args.clear();
    for (auto &&[i, d] : llvm::enumerate(type.getShape())) {
      if (!mlir::ShapedType::isDynamic(d))
        continue;

      args.emplace_back(shape[i]);
    }
    auto allocType = mlir::MemRefType::get(
        type.getShape(), type.getElementType(),
        mlir::MemRefLayoutAttrInterface{}, type.getMemorySpace());
    auto op = builder.create<mlir::memref::AllocaOp>(loc, allocType, args);
    if (type.getMemorySpace() == wgSpace)
      op->setAttr(expandAttrName, unit);

    mlir::Value mem = op.getResult();
    if (mem.getType() != type)
      mem = builder.create<mlir::memref::CastOp>(loc, type, mem);

    ret.emplace_back(mem);
  }
  return ret;
}

struct ConvertLoad final : public mlir::OpConversionPattern<hc::hk::LoadOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(hc::hk::LoadOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto origSrcType = mlir::dyn_cast<hc::hk::SymbolicallyShapedType>(
        op.getSource().getType());
    if (!origSrcType)
      return rewriter.notifyMatchFailure(op, "Invalid source type");

    auto origResultType =
        mlir::dyn_cast<hc::hk::SymbolicallyShapedType>(op.getType());
    if (!origResultType)
      return rewriter.notifyMatchFailure(op, "Invalid result type");

    const mlir::TypeConverter *converter = getTypeConverter();
    assert(converter);

    auto newResultType = converter->convertType(origResultType);
    if (!newResultType)
      return rewriter.notifyMatchFailure(op, "Failed to convert result type");

    auto rank = origResultType.getShape().size();
    if (rank < 1 || origSrcType.getShape().size() != rank ||
        adaptor.getShape().size() != rank)
      return rewriter.notifyMatchFailure(op, "Rank mismatch");

    mlir::Value src = adaptor.getSource();

    mlir::Location loc = op.getLoc();
    llvm::SmallVector<mlir::Value> srcShape;
    for (auto &&[i, symElem] : llvm::enumerate(origSrcType.getShape())) {
      mlir::Value dim;
      if (auto symbolcDim =
              mlir::dyn_cast<hc::typing::SymbolicTypeBase>(symElem)) {
        dim = rewriter.create<hc::hk::MaterializeExprOp>(loc, symbolcDim);
        dim = converter->materializeTargetConversion(
            rewriter, loc, rewriter.getIndexType(), dim);
      } else {
        dim = getDim(rewriter, loc, src, int64_t(i));
      }

      if (!dim)
        return rewriter.notifyMatchFailure(op, "Failed to get src dim");

      srcShape.emplace_back(dim);
    }

    llvm::SmallVector<mlir::Value> dstShape;
    for (auto &&[symElem, elem] :
         llvm::zip_equal(origResultType.getShape(), adaptor.getShape())) {
      mlir::Value dim;
      if (auto symbolcDim =
              mlir::dyn_cast<hc::typing::SymbolicTypeBase>(symElem)) {
        dim = rewriter.create<hc::hk::MaterializeExprOp>(loc, symbolcDim);
        dim = converter->materializeTargetConversion(
            rewriter, loc, rewriter.getIndexType(), dim);
      } else {
        dim = elem;
      }

      if (!dim)
        return rewriter.notifyMatchFailure(op, "Failed to get dst dim");

      dstShape.emplace_back(dim);
    }

    auto unpackedSrc = unpackTuple<mlir::ShapedType>(rewriter, loc, src);
    if (unpackedSrc.empty())
      return rewriter.notifyMatchFailure(op, "Failed to unpack src");

    if (mlir::isa<hc::hk::TensorType>(origResultType)) {
      auto alloc = createAlloc(rewriter, loc, newResultType, dstShape);
      if (alloc.size() != 2)
        return rewriter.notifyMatchFailure(op, "Failed to create alloc");

      mlir::Value zero = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
      mlir::Value one = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);
      auto bodyBuilder = [&](mlir::OpBuilder &builder, mlir::Location loc,
                             mlir::ValueRange indices, mlir::ValueRange) {
        mlir::Value cond;
        for (auto &&[srcDim, index] : llvm::zip_equal(srcShape, indices)) {
          mlir::Value lt = builder.create<mlir::arith::CmpIOp>(
              loc, mlir::arith::CmpIPredicate::slt, index, srcDim);
          if (!cond) {
            cond = lt;
          } else {
            cond = builder.create<mlir::arith::AndIOp>(loc, cond, lt);
          }
        }
        auto maskType = mlir::VectorType::get(1, builder.getI1Type());
        mlir::Value mask =
            builder.create<mlir::vector::SplatOp>(loc, maskType, cond);
        if (unpackedSrc.size() >= 2) {
          mlir::Value f = builder.create<mlir::arith::ConstantIntOp>(
              loc, /*value*/ 0, /*width*/ 1);
          mlir::Value passthru =
              builder.create<mlir::vector::SplatOp>(loc, maskType, f);
          mask = builder.create<mlir::vector::MaskedLoadOp>(
              loc, maskType, unpackedSrc[1], indices, mask, passthru);
        }

        auto src = unpackedSrc.front();
        auto vecType = mlir::VectorType::get(1, src.getType().getElementType());
        mlir::Value passthru = builder.create<mlir::ub::PoisonOp>(loc, vecType);
        mlir::Value res = builder.create<mlir::vector::MaskedLoadOp>(
            loc, vecType, src, indices, mask, passthru);
        builder.create<mlir::vector::StoreOp>(loc, res, alloc[0], indices);
        builder.create<mlir::vector::StoreOp>(loc, mask, alloc[1], indices);
      };
      llvm::SmallVector<mlir::Value> lowerBounds(rank, zero);
      llvm::SmallVector<mlir::Value> steps(rank, one);
      rewriter.create<mlir::scf::ParallelOp>(loc, lowerBounds, dstShape, steps,
                                             std::nullopt, bodyBuilder);

      auto res = packTuple(rewriter, loc, alloc, newResultType);
      if (!res)
        return rewriter.notifyMatchFailure(op, "Failed to pack result");

      rewriter.replaceOp(op, res);
      return mlir::success();
    }
    if (mlir::isa<hc::hk::VectorType>(origResultType)) {
      auto resTupleType = mlir::dyn_cast<mlir::TupleType>(newResultType);
      if (!resTupleType || resTupleType.size() != 2 ||
          !mlir::isa<mlir::VectorType>(resTupleType.getType(0)) ||
          !mlir::isa<mlir::VectorType>(resTupleType.getType(1)))
        return rewriter.notifyMatchFailure(op, "Invalid result type");

      auto resType = mlir::cast<mlir::VectorType>(resTupleType.getType(0));
      auto maskType = mlir::cast<mlir::VectorType>(resTupleType.getType(1));
      mlir::Value mask =
          rewriter.create<mlir::vector::CreateMaskOp>(loc, maskType, srcShape);

      mlir::Value zero = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
      llvm::SmallVector<mlir::Value> indices(resType.getRank(), zero);
      if (unpackedSrc.size() >= 2)
        mask = rewriter.create<mlir::vector::MaskedLoadOp>(
            loc, maskType, unpackedSrc[1], indices, mask, mask);

      mlir::Value passthru = rewriter.create<mlir::ub::PoisonOp>(loc, resType);
      mlir::Value result = rewriter.create<mlir::vector::MaskedLoadOp>(
          loc, resType, unpackedSrc[0], indices, mask, passthru);
      auto res = packTuple(rewriter, loc, {result, mask}, newResultType);
      if (!res)
        return rewriter.notifyMatchFailure(op, "Failed to pack result");

      rewriter.replaceOp(op, res);
      return mlir::success();
    }

    return rewriter.notifyMatchFailure(op, "Invalid ret type");
  }
};

struct ConvertStore final : public mlir::OpConversionPattern<hc::hk::StoreOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(hc::hk::StoreOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto origSrcType = mlir::dyn_cast<hc::hk::SymbolicallyShapedType>(
        op.getSource().getType());
    if (!origSrcType)
      return rewriter.notifyMatchFailure(op, "Invalid source type");

    auto origDstType = mlir::dyn_cast<hc::hk::SymbolicallyShapedType>(
        op.getTarget().getType());
    if (!origDstType)
      return rewriter.notifyMatchFailure(op, "Invalid target type");

    if (origSrcType.getShape().size() != origDstType.getShape().size())
      return rewriter.notifyMatchFailure(op, "Shape mismatch");

    mlir::Location loc = op.getLoc();
    auto unpackedSrc =
        unpackTuple<mlir::ShapedType>(rewriter, loc, adaptor.getSource());
    if (unpackedSrc.empty())
      return rewriter.notifyMatchFailure(op, "Failed to unpack source");

    auto unpackedDst =
        unpackTuple<mlir::ShapedType>(rewriter, loc, adaptor.getTarget());
    if (unpackedDst.empty())
      return rewriter.notifyMatchFailure(op, "Failed to unpack target");

    const mlir::TypeConverter *converter = getTypeConverter();
    assert(converter);

    llvm::SmallVector<mlir::Value> srcShape;
    for (auto &&[i, symElem] : llvm::enumerate(origSrcType.getShape())) {
      mlir::Value dim;
      if (auto symbolcDim =
              mlir::dyn_cast<hc::typing::SymbolicTypeBase>(symElem)) {
        dim = rewriter.create<hc::hk::MaterializeExprOp>(loc, symbolcDim);
        dim = converter->materializeTargetConversion(
            rewriter, loc, rewriter.getIndexType(), dim);
      } else {
        dim = getDim(rewriter, loc, unpackedSrc.front(), int64_t(i));
      }

      if (!dim)
        return rewriter.notifyMatchFailure(op, "Failed to get dim");

      srcShape.emplace_back(dim);
    }

    if (mlir::isa<hc::hk::TensorType>(origSrcType)) {
      mlir::Value zero = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
      mlir::Value one = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);
      auto bodyBuilder = [&](mlir::OpBuilder &builder, mlir::Location loc,
                             mlir::ValueRange indices, mlir::ValueRange) {
        auto maskType = mlir::VectorType::get(1, builder.getI1Type());
        mlir::Value mask;
        if (unpackedSrc.size() > 1) {
          mask = builder.create<mlir::vector::LoadOp>(loc, maskType,
                                                      unpackedSrc[1], indices);
        } else {
          mask = builder.create<mlir::arith::ConstantIntOp>(loc, 1, 1);
          mask = builder.create<mlir::vector::SplatOp>(loc, maskType, mask);
        }

        auto src = unpackedSrc.front();
        auto vecType = mlir::VectorType::get(1, src.getType().getElementType());
        mlir::Value val =
            builder.create<mlir::vector::LoadOp>(loc, vecType, src, indices);

        auto dst = unpackedDst.front();
        builder.create<mlir::vector::MaskedStoreOp>(loc, dst, indices, mask,
                                                    val);
        if (unpackedDst.size() > 1)
          builder.create<mlir::vector::MaskedStoreOp>(loc, unpackedDst[1],
                                                      indices, mask, mask);
      };

      auto rank = srcShape.size();
      llvm::SmallVector<mlir::Value> lowerBounds(rank, zero);
      llvm::SmallVector<mlir::Value> steps(rank, one);
      rewriter.create<mlir::scf::ParallelOp>(loc, lowerBounds, srcShape, steps,
                                             std::nullopt, bodyBuilder);
      rewriter.eraseOp(op);
      return mlir::success();
    }
    if (mlir::isa<hc::hk::VectorType>(origSrcType)) {
      if (unpackedSrc.size() != 2 ||
          !mlir::isa<mlir::VectorType>(unpackedSrc[0].getType()) ||
          !mlir::isa<mlir::VectorType>(unpackedSrc[1].getType()))
        return rewriter.notifyMatchFailure(op, "Invalid source type");

      mlir::Value value = unpackedSrc[0];
      mlir::Value mask = unpackedSrc[1];

      mlir::Value zero = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
      llvm::SmallVector<mlir::Value> indices(srcShape.size(), zero);

      auto dst = unpackedDst.front();
      rewriter.create<mlir::vector::MaskedStoreOp>(loc, dst, indices, mask,
                                                   value);
      if (unpackedDst.size() > 1)
        rewriter.create<mlir::vector::MaskedStoreOp>(loc, unpackedDst[1],
                                                     indices, mask, mask);

      rewriter.eraseOp(op);
      return mlir::success();
    }
    return rewriter.notifyMatchFailure(op, "Invalid src type");
  }
};

struct LowerHKernelOpsPass final
    : public hc::impl::LowerHKernelOpsPassBase<LowerHKernelOpsPass> {

  void runOnOperation() override {
    auto *ctx = &getContext();
    auto mod = getParentOrSelf<mlir::ModuleOp>(getOperation());
    if (!mod) {
      getOperation()->emitError("No parent module");
      return signalPassFailure();
    }

    mlir::ConversionTarget target(*ctx);

    llvm::SmallDenseMap<mlir::Type, mlir::Type> metadataMap;
    if (auto metadata = mod->getAttrOfType<mlir::ArrayAttr>(
            hc::hk::getKernelMetadataAttrName())) {
      if (metadata.size() % 2 != 0) {
        mod.emitError("Invalid metadata size");
        return signalPassFailure();
      }

      for (auto i : llvm::seq<size_t>(0, metadata.size() / 2)) {
        auto keyAttr = mlir::dyn_cast<hc::typing::TypeAttr>(metadata[i * 2]);
        auto valAttr =
            mlir::dyn_cast<hc::typing::TypeAttr>(metadata[i * 2 + 1]);
        if (!keyAttr || !valAttr) {
          mod.emitError("Invalid metadata");
          return signalPassFailure();
        }
        metadataMap.insert({keyAttr.getTypeVal(), valAttr.getTypeVal()});
      }
    }

    mlir::TypeConverter converter;

    // Convert unknown types to itself
    converter.addConversion([](mlir::Type type) { return type; });
    populateTypeConverter(converter);
    converter.addConversion([&](mlir::Type type) -> std::optional<mlir::Type> {
      auto it = metadataMap.find(type);
      if (it != metadataMap.end())
        return makeSignless(it->second);

      return std::nullopt;
    });

    auto materialize = [](mlir::OpBuilder &builder, mlir::Type type,
                          mlir::ValueRange inputs,
                          mlir::Location loc) -> mlir::Value {
      if (inputs.size() != 1)
        return {};

      return doCast(builder, loc, inputs.front(), type)->getResult(0);
    };
    converter.addArgumentMaterialization(materialize);
    converter.addSourceMaterialization(materialize);
    converter.addTargetMaterialization(materialize);

    mlir::RewritePatternSet patterns(ctx);

    mlir::populateAnyFunctionOpInterfaceTypeConversionPattern(patterns,
                                                              converter);

    target.addDynamicallyLegalDialect<hc::hk::HKernelDialect,
                                      mlir::scf::SCFDialect>(
        [&](mlir::Operation *op) -> bool { return converter.isLegal(op); });

    auto entrypointAttrName = mlir::StringAttr::get(
        &getContext(), hc::hk::getKernelEntryPointAttrName());
    target.addDynamicallyLegalOp<mlir::func::FuncOp>(
        [&](mlir::func::FuncOp op) {
          if (op->hasAttr(entrypointAttrName))
            return true;

          return converter.isSignatureLegal(op.getFunctionType()) &&
                 converter.isLegal(&op.getBody());
        });
    target.addDynamicallyLegalOp<mlir::func::ReturnOp>(
        [&](mlir::func::ReturnOp op) {
          return converter.isLegal(op.getOperandTypes());
        });
    target.addLegalOp<mlir::UnrealizedConversionCastOp>();
    target.addLegalOp<hc::hk::MaterializeExprOp>();
    target.addLegalDialect<mlir::ub::UBDialect, mlir::arith::ArithDialect,
                           mlir::memref::MemRefDialect,
                           mlir::vector::VectorDialect>();

    patterns.insert<ConvertTypes, ConvertMakeSlice, ConvertSubview, ConvertLoad,
                    ConvertStore>(converter, ctx);

    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target,
                                                  std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace
