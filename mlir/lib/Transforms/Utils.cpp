// SPDX-FileCopyrightText: 2024 The HC Authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hc/Utils.hpp"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Transforms/DialectConversion.h>

static llvm::SmallVector<mlir::Value>
flattenValues(llvm::ArrayRef<mlir::ValueRange> values) {
  llvm::SmallVector<mlir::Value> result;
  for (const auto &vals : values)
    llvm::append_range(result, vals);
  return result;
}

namespace {
struct ConvertReturn final : mlir::OpConversionPattern<mlir::func::ReturnOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::func::ReturnOp op, OneToNOpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::func::ReturnOp>(
        op, flattenValues(adaptor.getOperands()));
    return mlir::success();
  }
};
} // namespace

void hc::populateFuncPatternsAndTypeConversion(
    mlir::RewritePatternSet &patterns, mlir::ConversionTarget &target,
    mlir::TypeConverter &converter) {
  mlir::populateAnyFunctionOpInterfaceTypeConversionPattern(patterns,
                                                            converter);
  patterns.insert<ConvertReturn>(converter, patterns.getContext());

  target.addDynamicallyLegalOp<mlir::func::FuncOp>([&](mlir::func::FuncOp op) {
    return converter.isSignatureLegal(op.getFunctionType()) &&
           converter.isLegal(&op.getBody());
  });

  target.addDynamicallyLegalOp<mlir::func::ReturnOp>(
      [&](mlir::Operation *op) -> bool { return converter.isLegal(op); });
}
