// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hc/Dialect/HKernel/IR/HKernelOps.hpp"

#include <mlir/Dialect/Utils/StaticValueUtils.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>

#include <llvm/ADT/TypeSwitch.h>

void hc::hk::HKernelDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "hc/Dialect/HKernel/IR/HKernelOps.cpp.inc"
      >();

  addTypes<
#define GET_TYPEDEF_LIST
#include "hc/Dialect/HKernel/IR/HKernelOpsTypes.cpp.inc"
      >();

  addAttributes<
#define GET_ATTRDEF_LIST
#include "hc/Dialect/HKernel/IR/HKernelOpsAttributes.cpp.inc"
      >();
}

hc::hk::SymbolicallyShapedType
hc::hk::BufferType::cloneWith(std::optional<llvm::ArrayRef<mlir::Type>> shape,
                              mlir::Type elementType) const {
  return BufferType::get(getContext(), shape ? *shape : getShape(),
                         elementType ? elementType : getElementType());
}

mlir::OpFoldResult hc::hk::TupleExtractOp::fold(FoldAdaptor adaptor) {
  if (auto idx = mlir::getConstantIntValue(adaptor.getIndex())) {
    auto src = getSrc();
    auto def = src.getDefiningOp<MakeTupleOp>();
    if (!def)
      return nullptr;

    mlir::ValueRange args = def.getArgs();

    auto i = *idx;
    auto tupleType = mlir::cast<mlir::TupleType>(src.getType());
    assert(args.getTypes() == tupleType.getTypes());
    if (i < 0 || static_cast<size_t>(i) >= tupleType.size() ||
        tupleType.getType(i) != getType())
      return nullptr;

    return args[i];
  }

  return nullptr;
}

#include "hc/Dialect/HKernel/IR/HKernelOpsDialect.cpp.inc"

#include "hc/Dialect/HKernel/IR/HKernelOpsTypeInterfaces.cpp.inc"

#define GET_OP_CLASSES
#include "hc/Dialect/HKernel/IR/HKernelOps.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "hc/Dialect/HKernel/IR/HKernelOpsAttributes.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "hc/Dialect/HKernel/IR/HKernelOpsTypes.cpp.inc"

#include "hc/Dialect/HKernel/IR/HKernelOpsEnums.cpp.inc"
