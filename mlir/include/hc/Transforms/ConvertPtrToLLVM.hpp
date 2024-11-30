// SPDX-FileCopyrightText: 2024 The HC Authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

namespace mlir {
class DialectRegistry;
class LLVMTypeConverter;
class RewritePatternSet;
} // namespace mlir

namespace hc {
void populatePtrToLLVMConversionPatterns(mlir::LLVMTypeConverter &converter,
                                         mlir::RewritePatternSet &patterns);

void registerConvertPtrToLLVMInterface(mlir::DialectRegistry &registry);
} // namespace hc
