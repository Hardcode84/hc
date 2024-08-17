// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hc/Dialect/PyAST/IR/PyASTOps.hpp"

#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/TypeUtilities.h>

#include <llvm/ADT/TypeSwitch.h>

void hc::py_ast::PyASTDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "hc/Dialect/PyAST/IR/PyASTOps.cpp.inc"
      >();

  addTypes<
#define GET_TYPEDEF_LIST
#include "hc/Dialect/PyAST/IR/PyASTOpsTypes.cpp.inc"
      >();

  addAttributes<
#define GET_ATTRDEF_LIST
#include "hc/Dialect/PyAST/IR/PyASTOpsAttributes.cpp.inc"
      >();
}

void hc::py_ast::PyModuleOp::build(::mlir::OpBuilder &odsBuilder,
                                   ::mlir::OperationState &odsState) {
  ensureTerminator(*odsState.addRegion(), odsBuilder, odsState.location);
}

void hc::py_ast::PyFuncOp::build(::mlir::OpBuilder &odsBuilder,
                                 ::mlir::OperationState &odsState,
                                 llvm::StringRef name, mlir::ValueRange args,
                                 mlir::ValueRange decorators) {
  odsState.addAttribute(getNameAttrName(odsState.name),
                        odsBuilder.getStringAttr(name));
  odsState.addOperands(args);
  odsState.addOperands(decorators);

  int32_t segmentSizes[2] = {};
  segmentSizes[0] = static_cast<int32_t>(args.size());
  segmentSizes[1] = static_cast<int32_t>(decorators.size());
  odsState.addAttribute(getOperandSegmentSizeAttr(),
                        odsBuilder.getDenseI32ArrayAttr(segmentSizes));

  ensureTerminator(*odsState.addRegion(), odsBuilder, odsState.location);
}

void hc::py_ast::ArgOp::build(::mlir::OpBuilder &odsBuilder,
                              ::mlir::OperationState &odsState,
                              llvm::StringRef name, mlir::Value annotation) {
  auto type = NodeType::get(odsBuilder.getContext());
  build(odsBuilder, odsState, type, name, annotation);
}

void hc::py_ast::NameOp::build(::mlir::OpBuilder &odsBuilder,
                               ::mlir::OperationState &odsState,
                               llvm::StringRef id) {
  auto type = NodeType::get(odsBuilder.getContext());
  build(odsBuilder, odsState, type, id);
}

void hc::py_ast::SubscriptOp::build(::mlir::OpBuilder &odsBuilder,
                                    ::mlir::OperationState &odsState,
                                    mlir::Value value, mlir::Value slice) {
  auto type = NodeType::get(odsBuilder.getContext());
  build(odsBuilder, odsState, type, value, slice);
}

void hc::py_ast::ListOp::build(::mlir::OpBuilder &odsBuilder,
                               ::mlir::OperationState &odsState,
                               mlir::ValueRange elts) {
  auto type = NodeType::get(odsBuilder.getContext());
  build(odsBuilder, odsState, type, elts);
}

void hc::py_ast::TupleOp::build(::mlir::OpBuilder &odsBuilder,
                                ::mlir::OperationState &odsState,
                                mlir::ValueRange elts) {
  auto type = NodeType::get(odsBuilder.getContext());
  build(odsBuilder, odsState, type, elts);
}

void hc::py_ast::AttributeOp::build(::mlir::OpBuilder &odsBuilder,
                                    ::mlir::OperationState &odsState,
                                    mlir::Value value, llvm::StringRef attr) {
  auto type = NodeType::get(odsBuilder.getContext());
  build(odsBuilder, odsState, type, value, attr);
}

void hc::py_ast::ConstantOp::build(::mlir::OpBuilder &odsBuilder,
                                   ::mlir::OperationState &odsState,
                                   mlir::Attribute value) {
  auto type = NodeType::get(odsBuilder.getContext());
  build(odsBuilder, odsState, type, value);
}

void hc::py_ast::SliceOp::build(::mlir::OpBuilder &odsBuilder,
                                ::mlir::OperationState &odsState,
                                mlir::Value lower, mlir::Value upper,
                                mlir::Value step) {
  auto type = NodeType::get(odsBuilder.getContext());
  build(odsBuilder, odsState, type, lower, upper, step);
}

void hc::py_ast::CallOp::build(::mlir::OpBuilder &odsBuilder,
                               ::mlir::OperationState &odsState,
                               mlir::Value func, mlir::ValueRange args,
                               mlir::ValueRange keywods) {
  auto type = NodeType::get(odsBuilder.getContext());
  build(odsBuilder, odsState, type, func, args, keywods);
}

void hc::py_ast::KeywordOp::build(::mlir::OpBuilder &odsBuilder,
                                  ::mlir::OperationState &odsState,
                                  llvm::StringRef arg, mlir::Value value) {
  auto type = NodeType::get(odsBuilder.getContext());
  build(odsBuilder, odsState, type, arg, value);
}

void hc::py_ast::BoolOp::build(::mlir::OpBuilder &odsBuilder,
                               ::mlir::OperationState &odsState, BoolOpType op,
                               mlir::ValueRange values) {
  auto type = NodeType::get(odsBuilder.getContext());
  build(odsBuilder, odsState, type, op, values);
}

void hc::py_ast::IfOp::build(::mlir::OpBuilder &odsBuilder,
                             ::mlir::OperationState &odsState, mlir::Value test,
                             bool hasElse) {
  odsState.addOperands(test);
  auto thenRegion = odsState.addRegion();
  auto elseRegion = odsState.addRegion();
  ensureTerminator(*thenRegion, odsBuilder, odsState.location);
  if (hasElse)
    ensureTerminator(*elseRegion, odsBuilder, odsState.location);
}

void hc::py_ast::ForOp::build(::mlir::OpBuilder &odsBuilder,
                              ::mlir::OperationState &odsState,
                              mlir::Value target, mlir::Value iter) {
  odsState.addOperands(target);
  odsState.addOperands(iter);
  auto bodyRegion = odsState.addRegion();
  ensureTerminator(*bodyRegion, odsBuilder, odsState.location);
}

void hc::py_ast::WhileOp::build(::mlir::OpBuilder &odsBuilder,
                                ::mlir::OperationState &odsState,
                                mlir::Value test) {
  odsState.addOperands(test);
  auto bodyRegion = odsState.addRegion();
  ensureTerminator(*bodyRegion, odsBuilder, odsState.location);
}

void hc::py_ast::CompareOp::build(::mlir::OpBuilder &odsBuilder,
                                  ::mlir::OperationState &odsState,
                                  mlir::Value left, mlir::ArrayRef<CmpOp> ops,
                                  mlir::ValueRange comparators) {
  auto type = NodeType::get(odsBuilder.getContext());
  llvm::SmallVector<mlir::Attribute> opAttrs;
  opAttrs.reserve(ops.size());
  for (auto op : ops)
    opAttrs.emplace_back(CmpOpAttr::get(odsBuilder.getContext(), op));

  build(odsBuilder, odsState, type, left, odsBuilder.getArrayAttr(opAttrs),
        comparators);
}

void hc::py_ast::BinOp::build(::mlir::OpBuilder &odsBuilder,
                              ::mlir::OperationState &odsState,
                              mlir::Value left, BinOpVal op,
                              mlir::Value right) {
  auto type = NodeType::get(odsBuilder.getContext());
  build(odsBuilder, odsState, type, left, op, right);
}

void hc::py_ast::UnaryOp::build(::mlir::OpBuilder &odsBuilder,
                                ::mlir::OperationState &odsState, UnaryOpVal op,
                                mlir::Value operand) {
  auto type = NodeType::get(odsBuilder.getContext());
  build(odsBuilder, odsState, type, op, operand);
}

void hc::py_ast::IfExpOp::build(::mlir::OpBuilder &odsBuilder,
                                ::mlir::OperationState &odsState,
                                mlir::Value test, mlir::Value body,
                                mlir::Value oresle) {
  auto type = NodeType::get(odsBuilder.getContext());
  build(odsBuilder, odsState, type, test, body, oresle);
}

#include "hc/Dialect/PyAST/IR/PyASTOpsDialect.cpp.inc"

#define GET_OP_CLASSES
#include "hc/Dialect/PyAST/IR/PyASTOps.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "hc/Dialect/PyAST/IR/PyASTOpsAttributes.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "hc/Dialect/PyAST/IR/PyASTOpsTypes.cpp.inc"

#include "hc/Dialect/PyAST/IR/PyASTOpsEnums.cpp.inc"
