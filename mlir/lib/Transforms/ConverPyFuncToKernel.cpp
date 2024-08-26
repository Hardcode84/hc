// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hc/Transforms/Passes.hpp"

#include "hc/Dialect/HKernel/IR/HKernelOps.hpp"
#include "hc/Dialect/PyIR/IR/PyIROps.hpp"
#include "hc/Dialect/Typing/IR/TypingOps.hpp"

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
          term->emitError("kernel musr return none");
          return signalPassFailure();
        }

        mlir::OpBuilder::InsertionGuard g(builder);
        builder.setInsertionPoint(term);
        builder.replaceOpWithNewOp<mlir::func::ReturnOp>(term);
      }

      mlir::TypeRange argTypes =
          pyFunc.getBodyRegion().front().getArgumentTypes();
      auto funcType = mlir::FunctionType::get(&getContext(), argTypes, {});
      auto loc = pyFunc.getLoc();
      auto newFunc =
          builder.create<mlir::func::FuncOp>(loc, pyFunc.getName(), funcType);

      mlir::Region &dstRegion = newFunc.getBody();
      builder.inlineRegionBefore(srcRegion, dstRegion, dstRegion.begin());
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
  auto dimsStr = getStr("dims");
  auto dtypeStr = getStr("dtype");
  auto nameStr = getStr("name");
  converter.addConversion([=](hc::typing::IdentType type)
                              -> std::optional<mlir::Type> {
    if (type.getName() != bufferStr)
      return std::nullopt;

    auto dims = type.getParam<hc::typing::SequenceType>(dimsStr);
    if (!dims)
      return std::nullopt;

    auto dtype = type.getParam(dtypeStr);
    if (auto ident = mlir::dyn_cast_if_present<hc::typing::IdentType>(dtype))
      dtype = ident.getParam(nameStr);

    if (!dtype)
      return std::nullopt;

    return hc::hk::BufferType::get(type.getContext(), dims.getParams(), dtype);
  });
  converter.addConversion([=](hc::typing::IdentType type)
                              -> std::optional<mlir::Type> {
    if (type.getName() != tensorStr)
      return std::nullopt;

    auto dims = type.getParam<hc::typing::SequenceType>(dimsStr);
    if (!dims)
      return std::nullopt;

    auto dtype = type.getParam(dtypeStr);
    if (auto ident = mlir::dyn_cast_if_present<hc::typing::IdentType>(dtype))
      dtype = ident.getParam(nameStr);

    if (!dtype)
      return std::nullopt;

    return hc::hk::TensorType::get(type.getContext(), dims.getParams(), dtype);
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

static bool checkIsIdent(mlir::Type type, mlir::StringAttr name) {
  auto ident = mlir::dyn_cast<hc::typing::IdentType>(type);
  return ident && ident.getName() == name;
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
        funcName(mlir::StringAttr::get(
            context, "hckernel.kernel_api.CurrentGroup.load")),
        arrayName(mlir::StringAttr::get(context, "array")),
        shapeName(mlir::StringAttr::get(context, "shape")) {}

  mlir::LogicalResult
  matchAndRewrite(hc::py_ir::CallOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    if (!checkIsIdent(adaptor.getFunc().getType(), funcName))
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
  mlir::StringAttr funcName;
  mlir::StringAttr arrayName;
  mlir::StringAttr shapeName;
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
                          mlir::Location loc) -> std::optional<mlir::Value> {
      auto cast =
          builder.create<mlir::UnrealizedConversionCastOp>(loc, type, inputs);
      return cast.getResult(0);
    };
    converter.addArgumentMaterialization(materialize);
    converter.addSourceMaterialization(materialize);
    converter.addTargetMaterialization(materialize);

    mlir::RewritePatternSet patterns(ctx);

    mlir::populateAnyFunctionOpInterfaceTypeConversionPattern(patterns,
                                                              converter);

    target.addDynamicallyLegalOp<mlir::func::FuncOp>(
        [&](mlir::func::FuncOp op) {
          return converter.isSignatureLegal(op.getFunctionType()) &&
                 converter.isLegal(&op.getBody());
        });
    target.addIllegalOp<hc::py_ir::TuplePackOp, hc::py_ir::TuplePackOp>();
    target.addLegalDialect<mlir::arith::ArithDialect, hc::hk::HKernelDialect>();

    patterns.insert<ConvertTuplePack, ConvertTupleUnpack, ConvertSlice,
                    ConvertGetItem, ConvertLoad>(converter, ctx);

    using ConvertCurrentGroup = ConvertGroupExpr<hc::hk::CurrentGroupType>;
    patterns.insert<ConvertCurrentGroup>("work_offset", converter, ctx);
    patterns.insert<ConvertCurrentGroup>("shape", converter, ctx);

    if (mlir::failed(
            mlir::applyPartialConversion(mod, target, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace
