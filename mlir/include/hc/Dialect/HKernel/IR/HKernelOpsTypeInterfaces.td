// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef HKERNEL_TYPE_INTERFACES
#define HKERNEL_TYPE_INTERFACES

include "mlir/IR/OpBase.td"

def SymbolicallyShapedTypeInterface : TypeInterface<"SymbolicallyShapedType"> {
  let cppNamespace = "::hc::hk";
  let description = [{
    TBD
  }];
  let methods = [
    InterfaceMethod<[{
      Returns a clone of this type with the given shape and element type.

      If no shape is provided, the shape of this type is used.
    }],
    "::hc::hk::SymbolicallyShapedType", "cloneWith", (ins
      "::std::optional<::llvm::ArrayRef<::mlir::Type>>":$shape,
      "::mlir::Type":$elementType
    )>,

    InterfaceMethod<[{
      Returns the element type of this shaped type.
    }],
    "::mlir::Type", "getElementType">,

    InterfaceMethod<[{
      Returns the symbolic shape of this type.
    }],
    "::llvm::ArrayRef<::mlir::Type>", "getShape">,
  ];

  let extraClassDeclaration = [{
    /// Return a clone of this type with the given new shape and element type.
    auto clone(::llvm::ArrayRef<::mlir::Type> shape, ::mlir::Type elementType) {
      return cloneWith(shape, elementType);
    }

    /// Return a clone of this type with the given new shape.
    auto clone(::llvm::ArrayRef<::mlir::Type> shape) {
      return cloneWith(shape, getElementType());
    }
  }];


  let extraSharedClassDeclaration = [{
    /// Return a clone of this type with the given new element type.
    auto clone(::mlir::Type elementType) {
      return $_type.cloneWith(/*shape=*/std::nullopt, elementType);
    }
  }];
}

#endif // HKERNEL_TYPE_INTERFACES
