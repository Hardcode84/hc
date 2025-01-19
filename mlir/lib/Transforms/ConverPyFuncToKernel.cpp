// SPDX-FileCopyrightText: 2024 The HC Authors
// SPDX-FileCopyrightText: 2025 The HC Authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hc/Transforms/Passes.hpp"

#include "hc/Dialect/HKernel/IR/HKernelOps.hpp"
#include "hc/Dialect/PyIR/IR/PyIROps.hpp"
#include "hc/Dialect/Typing/IR/TypingOps.hpp"
#include "hc/Utils.hpp"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/DialectConversion.h>

namespace hc {
#define GEN_PASS_DEF_CONVERPYFUNCTOKERNELFUNCPASS
#include "hc/Transforms/Passes.h.inc"

#define GEN_PASS_DEF_CONVERPYIRTOKERNELPASS
#include "hc/Transforms/Passes.h.inc"
} // namespace hc

namespace {
struct ConverPyFuncToKernelFuncPass final
    : public hc::impl::ConverPyFuncToKernelFuncPassBase<
          ConverPyFuncToKernelFuncPass> {

  void runOnOperation() override {
    auto op = getOperation();

    mlir::IRRewriter builder(&getContext());
    builder.setInsertionPointToStart(op.getBody());

    auto entrypointAttrName =
        builder.getStringAttr(hc::hk::getKernelEntryPointAttrName());
    for (auto pyModule :
         llvm::make_early_inc_range(op.getOps<hc::py_ir::PyModuleOp>())) {
      auto term = mlir::cast<hc::py_ir::PyModuleEndOp>(
          pyModule.getBody()->getTerminator());
      mlir::ValueRange termResults = term.getResults();
      if (termResults.size() != 1) {
        term->emitError("Unepected py module results count: ")
            << termResults.size();
        return signalPassFailure();
      }

      auto pyFunc = termResults.front().getDefiningOp<hc::py_ir::PyFuncOp>();
      if (!pyFunc) {
        term->emitError("Expected a py func, but got: ") << termResults.front();
        return signalPassFailure();
      }

      if (!pyFunc.getCaptureArgs().empty()) {
        pyFunc->emitError("Cannot convert function with captures");
        return signalPassFailure();
      }

      mlir::Region &srcRegion = pyFunc.getBodyRegion();
      for (mlir::Block &block : srcRegion) {
        auto term = mlir::dyn_cast<hc::py_ir::ReturnOp>(block.getTerminator());
        if (!term)
          continue;

        if (!mlir::isa<mlir::NoneType>(term.getOperand().getType())) {
          term->emitError("kernel must return none");
          return signalPassFailure();
        }

        mlir::OpBuilder::InsertionGuard g(builder);
        builder.setInsertionPoint(term);
        builder.replaceOpWithNewOp<hc::hk::EnvironmentRegionYieldOp>(term);
      }

      mlir::TypeRange argTypes =
          pyFunc.getBodyRegion().front().getArgumentTypes();
      auto funcType = mlir::FunctionType::get(&getContext(), argTypes, {});
      auto loc = pyFunc.getLoc();
      auto newFunc =
          builder.create<mlir::func::FuncOp>(loc, pyFunc.getName(), funcType);
      newFunc->setAttr(entrypointAttrName, builder.getUnitAttr());

      llvm::SmallVector<mlir::Type> types;
      llvm::SmallVector<mlir::Location> locs;
      for (auto &&arg : srcRegion.getArguments()) {
        types.emplace_back(arg.getType());
        locs.emplace_back(arg.getLoc());
      }
      builder.createBlock(&newFunc.getBody(), {}, types, locs);
      auto env = hc::hk::WorkgroupScopeAttr::get(builder.getContext());
      auto envRegion = builder.create<hc::hk::EnvironmentRegionOp>(loc, env);
      mlir::Region &dstRegion = envRegion.getRegion();
      if (!dstRegion.empty())
        builder.eraseBlock(&dstRegion.front());

      builder.create<mlir::func::ReturnOp>(loc);

      builder.inlineRegionBefore(srcRegion, dstRegion, dstRegion.begin());
      for (auto &&[regArg, funcArg] :
           llvm::zip_equal(dstRegion.getArguments(), newFunc.getArguments()))
        regArg.replaceAllUsesWith(funcArg);

      dstRegion.front().eraseArguments(0, dstRegion.getNumArguments());
      builder.eraseOp(pyModule);
    }
  }
};

static void populateTypeConverter(mlir::MLIRContext *ctx,
                                  mlir::TypeConverter &converter) {
  // Convert unknown types to itself
  converter.addConversion([](mlir::Type type) { return type; });

  auto getStr = [&](mlir::StringRef str) -> mlir::StringAttr {
    return mlir::StringAttr::get(ctx, str);
  };

  auto currentGroup1Str = getStr("hckernel.kernel_api.CurrentGroup1");
  auto currentGroup2Str = getStr("hckernel.kernel_api.CurrentGroup2");
  auto currentGroup3Str = getStr("hckernel.kernel_api.CurrentGroup3");
  converter.addConversion(
      [=](hc::typing::IdentType type) -> std::optional<mlir::Type> {
        if (type.getName() == currentGroup1Str)
          return hc::hk::CurrentGroupType::get(ctx, 1);

        if (type.getName() == currentGroup2Str)
          return hc::hk::CurrentGroupType::get(ctx, 2);

        if (type.getName() == currentGroup3Str)
          return hc::hk::CurrentGroupType::get(ctx, 3);

        return std::nullopt;
      });

  auto bufferStr = getStr("Buffer");
  auto tensorStr = getStr("Tensor");
  auto vectorStr = getStr("Vector");
  auto dimsStr = getStr("dims");
  auto dtypeStr = getStr("dtype");
  auto nameStr = getStr("name");

  auto getDtype = [=](hc::typing::IdentType type) -> std::optional<mlir::Type> {
    mlir::Type dtype = type.getParam(dtypeStr);
    if (!dtype)
      return std::nullopt;

    if (auto ident = mlir::dyn_cast<hc::typing::IdentType>(dtype))
      dtype = ident.getParam(nameStr);

    if (auto literal = mlir::dyn_cast<hc::typing::LiteralType>(dtype)) {
      if (auto strAttr = mlir::dyn_cast<mlir::StringAttr>(literal.getValue())) {
        dtype =
            hc::typing::SymbolType::get(dtype.getContext(), strAttr.getValue());
      }
    }
    return dtype;
  };

  converter.addConversion(
      [=](hc::typing::IdentType type) -> std::optional<mlir::Type> {
        if (type.getName() != bufferStr)
          return std::nullopt;

        auto dims = type.getParam<hc::typing::SequenceType>(dimsStr);
        if (!dims)
          return std::nullopt;

        auto dtype = getDtype(type);
        if (!dtype)
          return std::nullopt;

        return hc::hk::BufferType::get(type.getContext(), dims.getParams(),
                                       *dtype);
      });
  converter.addConversion(
      [=](hc::typing::IdentType type) -> std::optional<mlir::Type> {
        if (type.getName() != tensorStr)
          return std::nullopt;

        auto dims = type.getParam<hc::typing::SequenceType>(dimsStr);
        if (!dims)
          return std::nullopt;

        auto dtype = getDtype(type);
        if (!dtype)
          return std::nullopt;

        return hc::hk::TensorType::get(type.getContext(), dims.getParams(),
                                       *dtype);
      });
  converter.addConversion(
      [=](hc::typing::IdentType type) -> std::optional<mlir::Type> {
        if (type.getName() != vectorStr)
          return std::nullopt;

        auto dims = type.getParam<hc::typing::SequenceType>(dimsStr);
        if (!dims)
          return std::nullopt;

        auto dtype = getDtype(type);
        if (!dtype)
          return std::nullopt;

        return hc::hk::VectorType::get(type.getContext(), dims.getParams(),
                                       *dtype);
      });

  auto tupleStr = getStr("Tuple");
  auto elementsStr = getStr("elements");
  converter.addConversion(
      [=, &converter](hc::typing::IdentType type) -> std::optional<mlir::Type> {
        if (type.getName() != tupleStr)
          return std::nullopt;

        auto elements = type.getParam<hc::typing::SequenceType>(elementsStr);
        if (!elements)
          return std::nullopt;

        llvm::SmallVector<mlir::Type> newTypes(elements.getParams().size());
        for (auto &&[i, type] : llvm::enumerate(elements.getParams())) {
          auto converted = converter.convertType(type);
          if (!converted)
            return std::nullopt;

          newTypes[i] = converted;
        }

        return mlir::TupleType::get(ctx, newTypes);
      });

  auto sliceStr = getStr("Slice");
  auto lowerStr = getStr("lower");
  auto upperStr = getStr("upper");
  auto stepStr = getStr("step");
  converter.addConversion(
      [=, &converter](hc::typing::IdentType type) -> std::optional<mlir::Type> {
        if (type.getName() != sliceStr)
          return std::nullopt;

        auto lower = converter.convertType(type.getParam(lowerStr));
        if (!lower)
          return std::nullopt;

        auto upper = converter.convertType(type.getParam(upperStr));
        if (!upper)
          return std::nullopt;

        auto step = converter.convertType(type.getParam(stepStr));
        if (!step)
          return std::nullopt;

        return hc::hk::SliceType::get(ctx, lower, upper, step);
      });
}

struct ConvertTuplePack final
    : public mlir::OpConversionPattern<hc::py_ir::TuplePackOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(hc::py_ir::TuplePackOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    const mlir::TypeConverter *converter = getTypeConverter();
    auto resType = converter->convertType<mlir::TupleType>(op.getType());
    if (!resType)
      return rewriter.notifyMatchFailure(op, "Invalid result type");

    rewriter.replaceOpWithNewOp<hc::hk::MakeTupleOp>(op, resType,
                                                     adaptor.getArgs());
    return mlir::success();
  }
};

struct ConvertTupleUnpack final
    : public mlir::OpConversionPattern<hc::py_ir::TupleUnpackOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(hc::py_ir::TupleUnpackOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Value src = adaptor.getArg();
    auto srcType = mlir::dyn_cast<mlir::TupleType>(src.getType());
    if (!srcType)
      return rewriter.notifyMatchFailure(op, "Invalid src type");

    mlir::Location loc = op.getLoc();
    const mlir::TypeConverter *converter = getTypeConverter();
    llvm::SmallVector<mlir::Value> results(op->getNumResults());
    for (auto &&[i, resType] : llvm::enumerate(op->getResultTypes())) {
      auto convertedType = converter->convertType(resType);
      if (convertedType != srcType.getType(i))
        return rewriter.notifyMatchFailure(op, "Invalid result type");

      mlir::Value idx = rewriter.create<mlir::arith::ConstantIndexOp>(
          loc, static_cast<int64_t>(i));
      results[i] =
          rewriter.create<hc::hk::TupleExtractOp>(loc, convertedType, src, idx);
    }

    rewriter.replaceOp(op, results);
    return mlir::success();
  }
};

struct ConvertSlice final
    : public mlir::OpConversionPattern<hc::py_ir::SliceOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(hc::py_ir::SliceOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    const mlir::TypeConverter *converter = getTypeConverter();
    auto resType = converter->convertType<hc::hk::SliceType>(op.getType());
    if (!resType)
      return rewriter.notifyMatchFailure(op, "Invalid result type");

    rewriter.replaceOpWithNewOp<hc::hk::MakeSliceOp>(
        op, resType, adaptor.getLower(), adaptor.getUpper(), adaptor.getStep());
    return mlir::success();
  }
};

template <typename GroupType>
struct ConvertGroupExpr final
    : public mlir::OpConversionPattern<hc::py_ir::GetAttrOp> {
  ConvertGroupExpr(mlir::StringRef name_,
                   const mlir::TypeConverter &typeConverter,
                   mlir::MLIRContext *context, mlir::PatternBenefit benefit = 1)
      : mlir::OpConversionPattern<hc::py_ir::GetAttrOp>(typeConverter, context,
                                                        benefit),
        name(mlir::StringAttr::get(context, name_)) {}

  mlir::LogicalResult
  matchAndRewrite(hc::py_ir::GetAttrOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    if (!mlir::isa<GroupType>(adaptor.getTarget().getType()))
      return rewriter.notifyMatchFailure(op, "Wrong target type");

    if (op.getNameAttr() != name)
      return rewriter.notifyMatchFailure(op, "Wrong name");

    const mlir::TypeConverter *converter = getTypeConverter();
    auto resType = converter->convertType(op.getType());
    if (!resType)
      return rewriter.notifyMatchFailure(op, "Invalid result type");

    if (auto tuple = mlir::dyn_cast<mlir::TupleType>(resType)) {
      mlir::Location loc = op.getLoc();
      llvm::SmallVector<mlir::Value> results(tuple.size());
      for (auto &&[i, type] : llvm::enumerate(tuple.getTypes()))
        results[i] = rewriter.create<hc::hk::MaterializeExprOp>(loc, type);

      rewriter.replaceOpWithNewOp<hc::hk::MakeTupleOp>(op, resType, results);
    } else {
      rewriter.replaceOpWithNewOp<hc::hk::MaterializeExprOp>(op, resType);
    }

    return mlir::success();
  }

private:
  mlir::StringAttr name;
};

static llvm::SmallVector<mlir::Value>
unpackTupleArg(mlir::OpBuilder &builder, mlir::Location loc, mlir::Value arg) {
  llvm::SmallVector<mlir::Value> ret;
  if (auto tuple = mlir::dyn_cast<mlir::TupleType>(arg.getType())) {
    ret.resize(tuple.size());
    for (auto &&[i, type] : llvm::enumerate(tuple.getTypes())) {
      mlir::Value idx = builder.create<mlir::arith::ConstantIndexOp>(loc, i);
      ret[i] = builder.create<hc::hk::TupleExtractOp>(loc, type, arg, idx);
    }
  } else {
    ret.emplace_back(arg);
  }

  return ret;
}

struct ConvertGetItem final
    : public mlir::OpConversionPattern<hc::py_ir::GetItemOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(hc::py_ir::GetItemOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Value target = adaptor.getTarget();
    if (!mlir::isa<hc::hk::SymbolicallyShapedType>(target.getType()))
      return rewriter.notifyMatchFailure(op, "Invalid source type");

    const mlir::TypeConverter *converter = getTypeConverter();
    auto resType =
        converter->convertType<hc::hk::SymbolicallyShapedType>(op.getType());
    if (!resType)
      return rewriter.notifyMatchFailure(op, "Invalid result type");

    auto index = unpackTupleArg(rewriter, op.getLoc(), adaptor.getIndex());
    rewriter.replaceOpWithNewOp<hc::hk::SubViewOp>(op, resType, target, index);
    return mlir::success();
  }
};

static bool checkIsIdent(mlir::Type type,
                         llvm::ArrayRef<mlir::StringAttr> names) {
  auto ident = mlir::dyn_cast<hc::typing::IdentType>(type);
  return ident && llvm::is_contained(names, ident.getName());
}

static llvm::SmallVector<mlir::Value>
decodeFuncArgs(hc::py_ir::CallOpAdaptor op,
               mlir::ArrayRef<mlir::StringAttr> resArgsNames) {
  llvm::SmallVector<mlir::Value> ret;
  ret.reserve(resArgsNames.size());

  mlir::ValueRange args = op.getArgs();
  mlir::ArrayRef<mlir::Attribute> argsNames = op.getArgsNames().getValue();
  assert(args.size() == argsNames.size());

  auto getCurrent = [&]() -> std::pair<mlir::Value, mlir::StringAttr> {
    return {args.front(), mlir::cast<mlir::StringAttr>(argsNames.front())};
  };

  auto getByName = [&](mlir::StringAttr argName) -> mlir::Value {
    for (auto &&[arg, name] : llvm::zip_equal(args, argsNames))
      if (name == argName)
        return arg;

    return nullptr;
  };

  auto popCurrent = [&]() {
    assert(!args.empty());
    args = args.drop_front();
    argsNames = argsNames.drop_front();
  };

  for (auto expectedName : resArgsNames) {
    assert(!expectedName.empty());

    if (args.empty()) {
      ret.emplace_back(nullptr);
      continue;
    }

    auto &&[arg, name] = getCurrent();
    if (name.empty()) {
      ret.emplace_back(arg);
      popCurrent();
      continue;
    }

    arg = getByName(expectedName);
    ret.emplace_back(arg);
  }
  return ret;
}

struct ConvertLoad final : public mlir::OpConversionPattern<hc::py_ir::CallOp> {
  ConvertLoad(const mlir::TypeConverter &typeConverter,
              mlir::MLIRContext *context, mlir::PatternBenefit benefit = 1)
      : mlir::OpConversionPattern<hc::py_ir::CallOp>(typeConverter, context,
                                                     benefit),
        loadName(mlir::StringAttr::get(
            context, "hckernel.kernel_api.CurrentGroup.load")),
        vloadName(mlir::StringAttr::get(
            context, "hckernel.kernel_api.CurrentGroup.vload")),
        arrayName(mlir::StringAttr::get(context, "array")),
        shapeName(mlir::StringAttr::get(context, "shape")) {}

  mlir::LogicalResult
  matchAndRewrite(hc::py_ir::CallOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    if (!checkIsIdent(adaptor.getFunc().getType(), {loadName, vloadName}))
      return rewriter.notifyMatchFailure(op, "Invalid func type");

    const mlir::TypeConverter *converter = getTypeConverter();
    auto resType =
        converter->convertType<hc::hk::SymbolicallyShapedType>(op.getType());
    if (!resType)
      return rewriter.notifyMatchFailure(op, "Invalid result type");

    auto args = decodeFuncArgs(adaptor, {arrayName, shapeName});
    mlir::Value array = args[0];
    mlir::Value shape = args[1];
    if (!mlir::isa_and_present<hc::hk::SymbolicallyShapedType>(array.getType()))
      return rewriter.notifyMatchFailure(op, "Invalid source type");

    if (!shape)
      return rewriter.notifyMatchFailure(op, "No shape");

    auto shapeArr = unpackTupleArg(rewriter, op.getLoc(), shape);
    rewriter.replaceOpWithNewOp<hc::hk::LoadOp>(op, resType, array, shapeArr);
    return mlir::success();
  }

private:
  mlir::StringAttr loadName;
  mlir::StringAttr vloadName;
  mlir::StringAttr arrayName;
  mlir::StringAttr shapeName;
};

struct ConvertStore final
    : public mlir::OpConversionPattern<hc::py_ir::CallOp> {
  ConvertStore(const mlir::TypeConverter &typeConverter,
               mlir::MLIRContext *context, mlir::PatternBenefit benefit = 1)
      : mlir::OpConversionPattern<hc::py_ir::CallOp>(typeConverter, context,
                                                     benefit),
        funcName(mlir::StringAttr::get(
            context, "hckernel.kernel_api.CurrentGroup.store")),
        dstName(mlir::StringAttr::get(context, "dst")),
        srcName(mlir::StringAttr::get(context, "src")) {}

  mlir::LogicalResult
  matchAndRewrite(hc::py_ir::CallOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    if (!op->use_empty())
      return rewriter.notifyMatchFailure(op, "Op still has uses");

    if (!checkIsIdent(adaptor.getFunc().getType(), funcName))
      return rewriter.notifyMatchFailure(op, "Invalid func type");

    const mlir::TypeConverter *converter = getTypeConverter();
    if (!converter->convertType<mlir::NoneType>(op.getType()))
      return rewriter.notifyMatchFailure(op, "Invalid result type");

    auto args = decodeFuncArgs(adaptor, {dstName, srcName});
    mlir::Value dst = args[0];
    mlir::Value src = args[1];
    if (!mlir::isa_and_present<hc::hk::SymbolicallyShapedType>(dst.getType()))
      return rewriter.notifyMatchFailure(op, "Invalid destination type");

    if (!mlir::isa_and_present<hc::hk::SymbolicallyShapedType>(src.getType()))
      return rewriter.notifyMatchFailure(op, "Invalid source type");

    rewriter.create<hc::hk::StoreOp>(op.getLoc(), dst, src);
    rewriter.eraseOp(op);
    return mlir::success();
  }

private:
  mlir::StringAttr funcName;
  mlir::StringAttr dstName;
  mlir::StringAttr srcName;
};

static bool isIntegral(mlir::Type type) {
  return mlir::isa<mlir::IntegerType, mlir::IndexType,
                   hc::typing::SymbolicTypeBase>(type);
}

static bool isFloat(mlir::Type type) {
  return mlir::isa<mlir::FloatType>(type);
}

template <typename Op>
static mlir::Value createBinOp(mlir::OpBuilder &builder, mlir::Location loc,
                               mlir::Value lhs, mlir::Value rhs) {
  return builder.create<Op>(loc, lhs, rhs);
}

struct ConvertScalarBinop final
    : public mlir::OpConversionPattern<hc::py_ir::BinOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(hc::py_ir::BinOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    const mlir::TypeConverter *converter = getTypeConverter();
    mlir::Type resType = converter->convertType(op.getType());
    if (!resType)
      return rewriter.notifyMatchFailure(op, "Invalid result type");

    mlir::Type origResType = resType;
    bool isInt; // else is float
    if (isIntegral(resType)) {
      isInt = true;
      resType = mlir::isa<hc::typing::SymbolicTypeBase>(resType)
                    ? rewriter.getIndexType()
                    : resType;
    } else if (isFloat(resType)) {
      isInt = false;
    } else {
      return rewriter.notifyMatchFailure(op, "Unsupported result type");
    }

    mlir::Value lhs = adaptor.getLeft();
    mlir::Value rhs = adaptor.getRight();
    mlir::Location loc = op.getLoc();

    using Handler = mlir::Value (*)(mlir::OpBuilder &, mlir::Location,
                                    mlir::Value, mlir::Value);
    std::tuple<hc::py_ir::BinOpVal, Handler, Handler> handlers[] = {
        {hc::py_ir::BinOpVal::mul, &createBinOp<mlir::arith::MulIOp>,
         &createBinOp<mlir::arith::MulFOp>},
    };

    auto convertArg = [&](mlir::Value val) -> mlir::Value {
      if (auto symbolic =
              mlir::dyn_cast<hc::typing::SymbolicTypeBase>(val.getType()))
        val = rewriter.create<hc::hk::MaterializeExprOp>(loc, symbolic);

      if (val.getType() != resType)
        val = rewriter.create<hc::typing::CastOp>(loc, resType, val);

      return val;
    };

    auto pred = op.getOp();
    for (auto &&[val, intHandler, floatHandler] : handlers) {
      if (val == pred) {
        auto handler = (isInt ? intHandler : floatHandler);
        if (!handler)
          break;

        lhs = convertArg(lhs);
        rhs = convertArg(rhs);

        mlir::Value res = handler(rewriter, loc, lhs, rhs);
        if (res.getType() != origResType)
          res = rewriter.create<hc::typing::CastOp>(loc, origResType, res);

        rewriter.replaceOp(op, res);
        return mlir::success();
      }
    }

    return rewriter.notifyMatchFailure(op, "N handler");
  }
};

struct ConvertTupleGetitem final
    : public mlir::OpConversionPattern<hc::py_ir::GetItemOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(hc::py_ir::GetItemOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Value src = adaptor.getTarget();
    auto srcType = mlir::dyn_cast<mlir::TupleType>(src.getType());
    if (!srcType)
      return rewriter.notifyMatchFailure(op, "Not a tuple type");

    const mlir::TypeConverter *converter = getTypeConverter();
    mlir::Type resType = converter->convertType(op.getType());
    if (!resType)
      return rewriter.notifyMatchFailure(op, "Invalid result type");

    mlir::Location loc = op.getLoc();
    mlir::Value index = adaptor.getIndex();
    if (auto symbolic =
            mlir::dyn_cast<hc::typing::SymbolicTypeBase>(index.getType()))
      index = rewriter.create<hc::hk::MaterializeExprOp>(loc, symbolic);

    mlir::Type indexType = rewriter.getIndexType();
    if (index.getType() != indexType)
      index = rewriter.create<hc::typing::CastOp>(loc, indexType, index);

    rewriter.replaceOpWithNewOp<hc::hk::TupleExtractOp>(op, resType, src,
                                                        index);
    return mlir::success();
  }
};

struct ConverPyIRToKernelPass final
    : public hc::impl::ConverPyIRToKernelPassBase<ConverPyIRToKernelPass> {

  void runOnOperation() override {
    auto mod = getOperation();

    auto *ctx = &getContext();
    mlir::ConversionTarget target(*ctx);
    mlir::TypeConverter converter;

    populateTypeConverter(ctx, converter);

    auto materialize = [](mlir::OpBuilder &builder, mlir::Type type,
                          mlir::ValueRange inputs,
                          mlir::Location loc) -> mlir::Value {
      auto cast =
          builder.create<mlir::UnrealizedConversionCastOp>(loc, type, inputs);
      return cast.getResult(0);
    };
    converter.addArgumentMaterialization(materialize);
    converter.addSourceMaterialization(materialize);
    converter.addTargetMaterialization(materialize);

    mlir::RewritePatternSet patterns(ctx);

    hc::populateFuncPatternsAndTypeConversion(patterns, target, converter);

    target.addIllegalOp<hc::py_ir::TuplePackOp, hc::py_ir::TuplePackOp,
                        hc::py_ir::BinOp>();
    target.addLegalDialect<mlir::arith::ArithDialect, hc::typing::TypingDialect,
                           hc::hk::HKernelDialect>();

    patterns.insert<ConvertTuplePack, ConvertTupleUnpack, ConvertSlice,
                    ConvertGetItem, ConvertLoad, ConvertStore,
                    ConvertScalarBinop, ConvertTupleGetitem>(converter, ctx);

    using ConvertCurrentGroup = ConvertGroupExpr<hc::hk::CurrentGroupType>;
    patterns.insert<ConvertCurrentGroup>("work_offset", converter, ctx);
    patterns.insert<ConvertCurrentGroup>("shape", converter, ctx);

    if (mlir::failed(
            mlir::applyPartialConversion(mod, target, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace
