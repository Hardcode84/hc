// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hc/Transforms/Passes.hpp"

#include "hc/Dialect/HKernel/IR/HKernelOps.hpp"
#include "hc/Dialect/PyIR/IR/PyIROps.hpp"
#include "hc/Dialect/Typing/IR/TypingOps.hpp"

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

static void popolateTypeConverter(mlir::MLIRContext *ctx,
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
}

struct ConverPyIRToKernelPass final
    : public hc::impl::ConverPyIRToKernelPassBase<ConverPyIRToKernelPass> {

  void runOnOperation() override {
    auto mod = getOperation();

    auto *ctx = &getContext();
    mlir::ConversionTarget target(*ctx);
    mlir::TypeConverter converter;

    popolateTypeConverter(ctx, converter);

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

    if (mlir::failed(
            mlir::applyPartialConversion(mod, target, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace
