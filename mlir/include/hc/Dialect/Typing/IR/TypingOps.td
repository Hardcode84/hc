// SPDX-FileCopyrightText: 2024 The HC Authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef TYPING_OPS
#define TYPING_OPS

include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"

include "hc/Dialect/Typing/IR/TypingOpsInterfaces.td"

def Typing_Dialect : Dialect {
  let name = "typing";
  let cppNamespace = "::hc::typing";

  let summary = "TBD.";
  let description = [{
    TBD
  }];

  let useDefaultAttributePrinterParser = 1;
  let useDefaultTypePrinterParser = 1;
  let hasConstantMaterializer = 1;
}

class Typing_Type<string name, string typeMnemonic, list<Trait> traits = [],
                   string baseCppClass = "::mlir::Type">
    : TypeDef<Typing_Dialect, name, traits, baseCppClass> {
  let mnemonic = typeMnemonic;
}

class Typing_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<Typing_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

class Typing_OpBase<string mnemonic, list<Trait> traits = []>
    : Op<Typing_Dialect, mnemonic, traits>;

def Typing_TypeAttr : Typing_Attr<"Type", "type_attr", [TypedAttrInterface]> {
  let parameters = (ins
    "::mlir::Type":$type_val
  );

  let builders = [
    AttrBuilderWithInferredContext<(ins "::mlir::Type":$type), [{
      return $_get(type.getContext(), type);
    }]>
  ];

  let assemblyFormat = "`<` $type_val `>`";

  let extraClassDeclaration = [{
    ::mlir::Type getType() {
      return ::hc::typing::ValueType::get(getContext());
    }
  }];
}

def Typing_ValueType : Typing_Type<"Value", "value", [], "::mlir::Type">;

def Typing_IdentType : Typing_Type<"Ident", "ident", [], "::mlir::Type"> {
  let parameters = (ins
    "::mlir::StringAttr":$name,
    OptionalArrayRefParameter<"::mlir::StringAttr">:$paramNames,
    OptionalArrayRefParameter<"::mlir::Type">:$params
  );

  let builders = [
    TypeBuilderWithInferredContext<(ins
      "::mlir::MLIRContext*":$ctx,
      "::llvm::StringRef":$name,
      CArg<"::llvm::ArrayRef<::llvm::StringRef>", "{}">:$paramNames,
      CArg<"::llvm::ArrayRef<::mlir::Type>", "{}">:$params
    ), [{
      ::llvm::SmallVector<::mlir::StringAttr> arr;
      for (auto str : paramNames)
        arr.emplace_back(::mlir::StringAttr::get(ctx, str));
      return $_get(ctx, ::mlir::StringAttr::get(ctx, name), arr, params);
    }]>
  ];

  let assemblyFormat = "$name (`:` $paramNames^)? (`->` $params^)?";

  let extraClassDeclaration = [{
    ::mlir::Type getParam(::mlir::StringAttr name) const;

    template<typename T>
    T getParam(::mlir::StringAttr name) const {
      return ::mlir::dyn_cast_if_present<T>(getParam(name));
    }
  }];
}

def Typing_SequenceType : Typing_Type<"Sequence", "sequence", [], "::mlir::Type"> {
  let parameters = (ins
    ArrayRefParameter<"::mlir::Type">:$params
  );

  let assemblyFormat = "$params";

  let extraClassDeclaration = [{
    size_t size() const {
      return getParams().size();
    }
  }];
}

def Typing_UnpackedSequenceType : Typing_Type<"UnpackedSequence", "unpacked_sequence", [], "::mlir::Type"> {
  let parameters = (ins
    ArrayRefParameter<"::mlir::Type">:$params
  );

  let assemblyFormat = "$params";
}


def Typing_SymbolType : Typing_Type<"Symbol", "symbol", [], "::hc::typing::SymbolicTypeBase"> {
  let parameters = (ins "::mlir::StringAttr":$name);

  let builders = [
    TypeBuilderWithInferredContext<(ins
      "::mlir::MLIRContext*":$ctx,
      "::llvm::StringRef":$name
    ), [{
      return $_get(ctx, ::mlir::StringAttr::get(ctx, name));
    }]>
  ];

  let assemblyFormat = "$name";
}

def Typing_LiteralType : Typing_Type<"Literal", "literal", [], "::hc::typing::SymbolicTypeBase"> {
  let parameters = (ins "::mlir::TypedAttr":$value);

  let builders = [
    TypeBuilderWithInferredContext<(ins
      "::mlir::TypedAttr":$value
    ), [{
      return $_get(value.getContext(), value);
    }]>
  ];

  let assemblyFormat = "$value";
}

def Typing_BitsizeType : Typing_Type<"Bitsize", "bitsize", [], "::hc::typing::SymbolicTypeBase"> {
  let parameters = (ins "::mlir::Type":$arg);

  let builders = [
    TypeBuilderWithInferredContext<(ins
      "::mlir::Type":$value
    ), [{
      return $_get(value.getContext(), value);
    }]>
  ];

  let assemblyFormat = "$arg";
}

def Typing_ExprType : Typing_Type<"Expr", "expr", [], "::hc::typing::SymbolicTypeBase"> {
  let parameters = (ins
    ArrayRefParameter<"::mlir::Type">:$params,
    "::mlir::AffineExpr":$expr
  );

  let skipDefaultBuilders = 1;
  let assemblyFormat = "custom<ExprType>($params, $expr)";

  let builders = [
    TypeBuilder<(ins "::llvm::ArrayRef<::mlir::Type>":$params,
                     "::mlir::AffineExpr":$expr), [{
      auto [newParams, newExpr] = ::simplifyExpr(params, expr);
      return Base::get($_ctxt, std::move(newParams), std::move(newExpr));
    }]>
  ];
}

def Typing_UnionType : Typing_Type<"Union", "union", [], "::mlir::Type"> {
  let parameters = (ins
    ArrayRefParameter<"::mlir::Type">:$params
  );

  let assemblyFormat = "$params";
}


def Typing_TypeConstantOp : Typing_OpBase<"type_constant", [
    ConstantLike, Pure,
    DeclareOpInterfaceMethods<TypeInferenceInterface>,
    DeclareOpInterfaceMethods<TypingInterpreterInterface>
]> {

  let arguments = (ins TypedAttrInterface:$value);
  let results = (outs Typing_ValueType:$result);

  let builders = [
    OpBuilder<(ins "::mlir::TypedAttr":$value), [{
      build($_builder, $_state, value.getType(), value);
    }]>
  ];

  let assemblyFormat = "attr-dict $value";

  let hasFolder = 1;
}


def Typing_ResolveOp : Typing_OpBase<"resolve", [
  SingleBlockImplicitTerminator<"ResolveYieldOp">,
  RecursiveMemoryEffects,
  IsolatedFromAbove,
  DeclareOpInterfaceMethods<TypingUpdateInplaceInterface>
]> {

  let arguments = (ins Variadic<AnyType>:$args);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$bodyRegion);

  let builders = [
    OpBuilder<(ins
        "::mlir::TypeRange":$resultTypes,
        "::mlir::ValueRange":$args
      )>
  ];

  let skipDefaultBuilders = 1;
  let hasCanonicalizer = 1;

  let assemblyFormat = [{
    attr-dict-with-keyword ($args^ `:` type($args))?
    (`->` type($results)^)?
    $bodyRegion
  }];
}

def Typing_ResolveYieldOp : Typing_OpBase<"resolve_yield", [
  Terminator, HasParent<"ResolveOp ">, Pure
]> {
  let arguments = (ins Variadic<AnyType>:$args);
  let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];

  let assemblyFormat = "attr-dict ($args^ `:` type($args))?";
}

def Typing_TypeResolverOp : Typing_OpBase<"type_resolver",
    [IsolatedFromAbove]> {
  let arguments = (ins
    AnyAttr:$key
  );

  let regions = (region AnyRegion:$bodyRegion);

  let assemblyFormat = "attr-dict $key $bodyRegion";
}

def Typing_TypeResolverReturnOp : Typing_OpBase<"type_resolver_return", [
  Pure, HasParent<"TypeResolverOp">, ReturnLike, Terminator,
  DeclareOpInterfaceMethods<TypingInterpreterInterface>
]> {
  let arguments = (ins Variadic<Typing_ValueType>:$args);

  let assemblyFormat = "attr-dict ($args^)?";
}

def Typing_CastOp : Typing_OpBase<"cast", [
  Pure,
  DeclareOpInterfaceMethods<CastOpInterface>,
  DeclareOpInterfaceMethods<TypingKeyInterface>,
  DeclareOpInterfaceMethods<TypingUpdateInplaceInterface>,
  DeclareOpInterfaceMethods<TypeInferenceInterface>,
  DeclareOpInterfaceMethods<TypingInterpreterInterface>
]> {
  let arguments = (ins AnyType:$value);
  let results = (outs AnyType:$result);

  let hasFolder = 1;

  let assemblyFormat = "attr-dict $value `:` type($value) `to` type($result)";
}

def Typing_ValueCastOp : Typing_OpBase<"value_cast", [
  Pure,
  DeclareOpInterfaceMethods<CastOpInterface>,
  DeclareOpInterfaceMethods<TypingKeyInterface>,
  DeclareOpInterfaceMethods<TypingUpdateInplaceInterface>,
  DeclareOpInterfaceMethods<TypeInferenceInterface>,
  DeclareOpInterfaceMethods<TypingInterpreterInterface>
]> {
  let arguments = (ins AnyType:$value);
  let results = (outs AnyType:$result);

  let hasFolder = 1;

  let assemblyFormat = "attr-dict $value `:` type($value) `to` type($result)";
}


def Typing_MakeIdentOp : Typing_OpBase<"make_ident", [
  Pure, DeclareOpInterfaceMethods<TypingInterpreterInterface>
]> {
  let arguments = (ins
    StrAttr:$name,
    ArrayAttr:$paramNames,
    Variadic<Typing_ValueType>:$params
  );
  let results = (outs Typing_ValueType:$result);

  let assemblyFormat = "attr-dict $name $paramNames (`:` $params^)?";
}

def Typing_MakeSymbolOp : Typing_OpBase<"make_symbol", [
  Pure, DeclareOpInterfaceMethods<TypingInterpreterInterface>
]> {
  let arguments = (ins Typing_ValueType:$name);
  let results = (outs Typing_ValueType:$result);

  let assemblyFormat = "attr-dict $name";
}

def Typing_MakeLiteralOp : Typing_OpBase<"make_literal", [
  Pure, DeclareOpInterfaceMethods<TypingInterpreterInterface>
]> {
  let arguments = (ins TypedAttrInterface:$value);
  let results = (outs Typing_ValueType:$result);

  let assemblyFormat = "attr-dict $value";
}


def Typing_GetNumArgsOp : Typing_OpBase<"get_num_args", [
  Pure, DeclareOpInterfaceMethods<TypingInterpreterInterface>
]> {
  let results = (outs Index:$result);

  let assemblyFormat = "attr-dict";
}

def Typing_GetArgOp : Typing_OpBase<"get_arg", [
  Pure, DeclareOpInterfaceMethods<TypingInterpreterInterface>
]> {
  let arguments = (ins Index:$index);
  let results = (outs Typing_ValueType:$result);

  let assemblyFormat = "attr-dict $index";
}

def Typing_GetNamedArgOp : Typing_OpBase<"get_named_arg", [
  Pure, DeclareOpInterfaceMethods<TypingInterpreterInterface>
]> {
  let arguments = (ins StrAttr:$name);
  let results = (outs Typing_ValueType:$result);

  let assemblyFormat = "attr-dict $name";
}

def Typing_GetAttrOp : Typing_OpBase<"get_attr", [
  Pure, DeclareOpInterfaceMethods<TypingInterpreterInterface>
]> {
  let arguments = (ins StrAttr:$name);
  let results = (outs Typing_ValueType:$result);

  let assemblyFormat = "attr-dict $name";
}

def Typing_GetIdentNameOp : Typing_OpBase<"get_ident_name", [
  Pure, DeclareOpInterfaceMethods<TypingInterpreterInterface>,
  DeclareOpInterfaceMethods<TypingKeyInterface>
]> {
  let arguments = (ins Typing_ValueType:$ident);
  let results = (outs Typing_ValueType:$result);

  let assemblyFormat = "attr-dict $ident";
}

def Typing_GetIdentParamOp : Typing_OpBase<"get_ident_param", [
  Pure, DeclareOpInterfaceMethods<TypingInterpreterInterface>,
  DeclareOpInterfaceMethods<TypingKeyInterface>
]> {
  let arguments = (ins
    Typing_ValueType:$ident,
    StrAttr:$name);
  let results = (outs Typing_ValueType:$result);

  let assemblyFormat = "attr-dict $ident $name";
}

def Typing_GetMetatypeNameOp : Typing_OpBase<"get_metatype_name", [
  Pure, DeclareOpInterfaceMethods<TypingInterpreterInterface>,
  DeclareOpInterfaceMethods<TypingKeyInterface>
]> {
  let arguments = (ins Typing_ValueType:$value);
  let results = (outs Typing_ValueType:$result);

  let assemblyFormat = "attr-dict $value";
}

def Typing_CreateSeqOp : Typing_OpBase<"create_seq", [
  Pure,
  DeclareOpInterfaceMethods<TypingInterpreterInterface>,
  DeclareOpInterfaceMethods<TypingKeyInterface>
]> {
  let results = (outs Typing_ValueType:$result);

  let assemblyFormat = "attr-dict";
}

def Typing_AppendSeqOp : Typing_OpBase<"append_seq", [
  Pure, DeclareOpInterfaceMethods<TypingInterpreterInterface>
]> {
  let arguments = (ins
    Typing_ValueType:$seq,
    Typing_ValueType:$arg);
  let results = (outs Typing_ValueType:$result);

  let assemblyFormat = "attr-dict $seq $arg";
}

def Typing_GetSeqElementOp : Typing_OpBase<"get_seq_element", [
  Pure, DeclareOpInterfaceMethods<TypingInterpreterInterface>
]> {
  let arguments = (ins
    Typing_ValueType:$seq,
    Index:$index);
  let results = (outs Typing_ValueType:$result);

  let assemblyFormat = "attr-dict $seq `[` $index `]`";
}

def Typing_GetSeqSizeOp : Typing_OpBase<"get_seq_size", [
  Pure, DeclareOpInterfaceMethods<TypingInterpreterInterface>
]> {
  let arguments = (ins Typing_ValueType:$seq);
  let results = (outs Index:$result);

  let assemblyFormat = "attr-dict $seq";
}

def Typing_UnpackSeqOp : Typing_OpBase<"unpack_seq", [
  Pure,
  DeclareOpInterfaceMethods<TypingInterpreterInterface>
]> {
  let arguments = (ins Typing_ValueType:$seq);
  let results = (outs Typing_ValueType:$result);

  let assemblyFormat = "attr-dict $seq";
}

def Typing_IsSameOp : Typing_OpBase<"is_same", [
  Pure, DeclareOpInterfaceMethods<TypingInterpreterInterface>
]> {
  let arguments = (ins
    Typing_ValueType:$lhs,
    Typing_ValueType:$rhs);
  let results = (outs I1:$result);

  let assemblyFormat = "attr-dict $lhs $rhs";
}

def Typing_CheckOp : Typing_OpBase<"check", [
  DeclareOpInterfaceMethods<TypingInterpreterInterface>
]> {
  let arguments = (ins I1:$condition);

  let assemblyFormat = "attr-dict $condition";
}

def Typing_PrintOp : Typing_OpBase<"print", [
  DeclareOpInterfaceMethods<TypingInterpreterInterface>
]> {
  let arguments = (ins Typing_ValueType:$value);

  let assemblyFormat = "attr-dict $value";
}

def Typing_MakeUnionOp : Typing_OpBase<"make_union", [
  Pure, DeclareOpInterfaceMethods<TypingInterpreterInterface>
]> {
  let arguments = (ins Variadic<Typing_ValueType>:$args);
  let results = (outs Typing_ValueType:$result);

  let assemblyFormat = "attr-dict $args";
}

def Typing_GetGlobalAttrOp : Typing_OpBase<"get_global_attr", [
  Pure, DeclareOpInterfaceMethods<TypingInterpreterInterface>
]> {
  let arguments = (ins StrAttr:$name);
  let results = (outs Typing_ValueType:$result);

  let assemblyFormat = "attr-dict $name";
}

def Typing_BinOpAttr : I64EnumAttr<
    "BinOpVal", "",
    [
      I64EnumAttrCase<"add", 0>,
      I64EnumAttrCase<"sub", 1>,
      I64EnumAttrCase<"mul", 2>,
      I64EnumAttrCase<"ceil_div", 3>,
      I64EnumAttrCase<"floor_div", 4>,
      I64EnumAttrCase<"mod", 5>,
    ]> {
  let cppNamespace = "::hc::typing";
}

def Typing_BinOp : Typing_OpBase<"binop", [
  Pure, DeclareOpInterfaceMethods<TypingInterpreterInterface>
]> {
  let arguments = (ins
      Typing_ValueType:$lhs,
      Typing_BinOpAttr:$op,
      Typing_ValueType:$rhs);
  let results = (outs Typing_ValueType:$result);

  let assemblyFormat = "attr-dict $lhs $op $rhs";
}


#endif // TYPING_OPS
