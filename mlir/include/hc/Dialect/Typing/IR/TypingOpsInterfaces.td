// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TYPING_OPS_INTERFACES
#define TYPING_OPS_INTERFACES

include "mlir/IR/OpBase.td"

def TypingKeyInterface : OpInterface<"TypingKeyInterface"> {
  let cppNamespace = "::hc::typing";

  let methods = [
    InterfaceMethod<
      "",
      "::llvm::SmallVector<::mlir::Attribute>",
      "getTypingKeys",
      (ins),
      [{}],
      [{
        ::llvm::SmallVector<::mlir::Attribute> ret;
        ::llvm::SmallVector<::mlir::Attribute> attrs;
        auto op = ::mlir::cast<ConcreteOp>(this->getOperation());
        ::mlir::Attribute name = op->getName().getIdentifier();
        attrs.emplace_back(name);
        op.getTypingKeyArgs(attrs);
        auto *ctx = op->getContext();
        if (attrs.size() > 1)
          ret.emplace_back(::mlir::ArrayAttr::get(ctx, name));

        ret.emplace_back(::mlir::ArrayAttr::get(ctx, attrs));
        return ret;
      }]
      >,
    InterfaceMethod<
      "",
      "void",
      "getTypingKeyArgs",
      (ins "::llvm::SmallVectorImpl<::mlir::Attribute>&":$args),
      [{}],
      [{
        // Nothing
      }]
      >
  ];
}

def TypingInterpreterInterface : OpInterface<"TypingInterpreterInterface"> {
  let cppNamespace = "::hc::typing";

  let methods = [
    InterfaceMethod<
      "",
      "::mlir::FailureOr<bool>",
      "interpret",
      (ins "::hc::typing::InterpreterState&":$state)>
  ];
}

def TypeInferenceInterface : OpInterface<"TypeInferenceInterface"> {
  let cppNamespace = "::hc::typing";

  let methods = [
    InterfaceMethod<
      "",
      "::mlir::FailureOr<bool>",
      "inferTypes",
      (ins
        "::mlir::TypeRange":$types,
        "::llvm::SmallVectorImpl<::mlir::Type>&":$results
      )>
  ];
}

def TypingUpdateInplaceInterface : OpInterface<"TypingUpdateInplaceInterface"> {
  let cppNamespace = "::hc::typing";

  let methods = [
    InterfaceMethod<
      "",
      "bool",
      "canUpdateArgTypeInplace",
      (ins "::mlir::Value":$arg),
      [{}],
      [{
        return true;
      }]
      >
  ];
}

def DataflowJoinInterface : OpInterface<"DataflowJoinInterface"> {
  let cppNamespace = "::hc::typing";

  let methods = [
    InterfaceMethod<
      "",
      "void",
      "getArgsIndices",
      (ins
        "unsigned":$resultIndex,
        "::llvm::SmallVectorImpl<unsigned>&":$argsIndices
      )>
  ];
}

def GetNamedArgInterface : OpInterface<"GetNamedArgInterface"> {
  let cppNamespace = "::hc::typing";

  let methods = [
    InterfaceMethod<
      "",
      "::mlir::FailureOr<::mlir::Value>",
      "getNamedArg",
      (ins
        "::mlir::StringRef":$name
      )>
  ];
}


#endif // TYPING_OPS_INTERFACES
