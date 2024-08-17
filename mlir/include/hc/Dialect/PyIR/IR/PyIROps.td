// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef PYIR_OPS
#define PYIR_OPS

include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"

include "hc/Dialect/Typing/IR/TypingOpsInterfaces.td"

def PyIR_Dialect : Dialect {
  let name = "py_ir";
  let cppNamespace = "::hc::py_ir";

  let summary = "TBD.";
  let description = [{
    TBD
  }];

//  let useDefaultAttributePrinterParser = 1;
  let useDefaultTypePrinterParser = 1;
}

class PyIR_Type<string name, string typeMnemonic, list<Trait> traits = [],
                   string baseCppClass = "::mlir::Type">
    : TypeDef<PyIR_Dialect, name, traits, baseCppClass> {
  let mnemonic = typeMnemonic;
}

class PyIR_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<PyIR_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

class PyIR_OpBase<string mnemonic, list<Trait> traits = []>
    : Op<PyIR_Dialect, mnemonic, !listconcat(traits, [
      DeclareOpInterfaceMethods<TypingUpdateInplaceInterface>
    ])>;

class PyIR_OpBaseNoTypeUpdate<string mnemonic, list<Trait> traits = []>
    : Op<PyIR_Dialect, mnemonic, traits>;

def PyIR_UndefinedType : PyIR_Type<"Undefined", "undefined", [], "::mlir::Type">;

def PyIR_IdentType : PyIR_Type<"Ident", "ident", [], "::mlir::Type"> {
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

def PyIR_SubscriptType : PyIR_Type<"Subscript", "subscript", [], "::mlir::Type"> {
  let parameters = (ins
      "::mlir::Type":$value,
      "::mlir::Type":$slice);

  let builders = [
    TypeBuilderWithInferredContext<(ins
      "::mlir::Type":$value,
      "::mlir::Type":$slice
    ), [{
      auto ctx = value.getContext();
      return $_get(ctx, value, slice);
    }]>
  ];

  let assemblyFormat = "$value `[` $slice `]`";
}

def PyIR_ConstType : PyIR_Type<"Const", "const", [], "::mlir::Type"> {
  let parameters = (ins "::mlir::Attribute":$value);

  let builders = [
    TypeBuilderWithInferredContext<(ins
      "::mlir::Attribute":$value
    ), [{
      auto ctx = value.getContext();
      return $_get(ctx, value);
    }]>
  ];

  let assemblyFormat = "$value";
}

def PyIR_NoneOp : PyIR_OpBase<"none", [
  Pure, DeclareOpInterfaceMethods<TypeInferenceInterface>
]> {
  let results = (outs NoneType:$result);

  let builders = [
    OpBuilder<(ins), [{
      build($_builder, $_state, ::mlir::NoneType::get($_builder.getContext()));
    }]>
  ];

  let assemblyFormat = "attr-dict";
}

def PyIR_ConstantOp : PyIR_OpBaseNoTypeUpdate<"constant", [
    ConstantLike, Pure, AllTypesMatch<["value", "result"]>,
    DeclareOpInterfaceMethods<TypeInferenceInterface>
]> {
  let arguments = (ins TypedAttrInterface:$value);
  let results = (outs AnyType:$result);

  let builders = [
    OpBuilder<(ins "::mlir::TypedAttr":$value), [{
      build($_builder, $_state, value.getType(), value);
    }]>
  ];

  let assemblyFormat = "attr-dict $value";

  let hasFolder = 1;
}

def PyIR_SymbolConstantOp : PyIR_OpBaseNoTypeUpdate<"sym_constant", [
    ConstantLike, Pure
]> {
  let arguments = (ins FlatSymbolRefAttr:$value);
  let results = (outs AnyType:$result);

  let assemblyFormat = "attr-dict $value `:` type(results)";

  let hasFolder = 1;
}

def PyIR_PyModuleOp : PyIR_OpBase<"module", [
  IsolatedFromAbove, SymbolTable,
  SingleBlockImplicitTerminator<"PyModuleEndOp">
]> {

  let regions = (region SizedRegion<1>:$bodyRegion);

  let assemblyFormat = "attr-dict-with-keyword $bodyRegion";
}

def PyIR_PyModuleEndOp : PyIR_OpBase<"module_end", [
  Terminator, HasParent<"PyModuleOp">, Pure
]> {
  let arguments = (ins Variadic<AnyType>:$results);
  let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];

  let assemblyFormat = "attr-dict ($results^ `:` type($results))?";
}

def PyIR_LoadModuleOp : PyIR_OpBase<"load_module", [
  Pure,
  DeclareOpInterfaceMethods<TypingKeyInterface, ["getTypingKeyArgs"]>
]> {
  let arguments = (ins StrAttr:$name);
  let results = (outs AnyType:$result);

  let assemblyFormat = "attr-dict $name `:` type($result)";
}

def PyIR_EmptyAnnotationOp : PyIR_OpBase<"empty_annotation", [Pure]> {
  let results = (outs PyIR_UndefinedType:$result);

  let builders = [
    OpBuilder<(ins), [{
      build($_builder, $_state, ::hc::py_ir::UndefinedType::get($_builder.getContext()));
    }]>
  ];

  let assemblyFormat = "attr-dict";
}

def PyIR_PyFuncOp : PyIR_OpBase<"func", [
  Pure, IsolatedFromAbove, AttrSizedOperandSegments
]> {
  let arguments = (ins
      StrAttr:$name,
      ArrayAttr:$arg_names,
      Variadic<AnyType>:$annotations,
      ArrayAttr:$capture_names,
      Variadic<AnyType>:$capture_args,
      Variadic<AnyType>:$decorators);

  let results = (outs AnyType:$result);

  let regions = (region AnyRegion:$bodyRegion);

  let builders = [
    OpBuilder<(ins
        "::mlir::Type":$resultType,
        "::llvm::StringRef":$name,
        "::llvm::ArrayRef<::llvm::StringRef>":$argNames,
        "::mlir::ValueRange":$annotations,
        "::llvm::ArrayRef<::llvm::StringRef>":$captureNames,
        "::mlir::ValueRange":$captureArgs,
        "::mlir::ValueRange":$decorators)>
  ];

  let extraClassDeclaration = [{
    ::mlir::Block *getEntryBlock() { return &getBodyRegion().front(); }

    ::mlir::ValueRange getBlockArgs() {
      return getEntryBlock()->getArguments().take_front(getArgNames().size());
    }

    ::mlir::ValueRange getCaptureBlockArgs() {
      return getEntryBlock()->getArguments().take_back(getCaptureNames().size());
    }

    auto getCaptureNamesAndArgs() {
      return llvm::zip_equal(
        getCaptureNames().getAsRange<mlir::StringAttr>(),
        getCaptureArgs(), getCaptureBlockArgs());
    }

    auto getArgsNamesArray() {
      return ::llvm::to_vector(getArgNames().getAsValueRange<mlir::StringAttr>());
    }

    auto getCaptureNamesArray() {
      return ::llvm::to_vector(getCaptureNames().getAsValueRange<mlir::StringAttr>());
    }
  }];

  let skipDefaultBuilders = 1;
  let hasCanonicalizer = 1;

  let assemblyFormat = [{
    attr-dict $name
    custom<ArgList>($annotations, $arg_names) (`:` type($annotations)^)?
    `capture` custom<ArgList>($capture_args, $capture_names) (`:` type($capture_args)^)?
    ( `decorators` $decorators^ `:` type($decorators) )?
    `->` type($result) $bodyRegion
  }];
}

def PyIR_PyStaticFuncOp : PyIR_OpBase<"static_func", [
  FunctionOpInterface, IsolatedFromAbove
]> {
  let arguments = (ins
      SymbolNameAttr:$sym_name,
      OptionalAttr<StrAttr>:$sym_visibility,
      TypeAttrOf<FunctionType>:$function_type,
      ArrayAttr:$arg_names,
      Variadic<AnyType>:$annotations);

  let regions = (region AnyRegion:$bodyRegion);

  let builders = [
    OpBuilder<(ins
        "::llvm::StringRef":$symName,
        "::mlir::FunctionType":$functionType,
        "::llvm::ArrayRef<::llvm::StringRef>":$argNames,
        "::mlir::ValueRange":$annotations)>
  ];

  let extraClassDeclaration = [{
    ::mlir::Block *getEntryBlock() { return &getBodyRegion().front(); }

    ::mlir::ValueRange getBlockArgs() {
      return getEntryBlock()->getArguments().take_front(getArgNames().size());
    }

    auto getArgsNamesArray() {
      return ::llvm::to_vector(getArgNames().getAsValueRange<mlir::StringAttr>());
    }

    /// Returns the region on the current operation that is callable.
    ::mlir::Region *getCallableRegion() { return &getBodyRegion(); }

    /// Returns the argument types of this function.
    ::llvm::ArrayRef<::mlir::Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this function.
    ::llvm::ArrayRef<::mlir::Type> getResultTypes() { return getFunctionType().getResults(); }
  }];

  let skipDefaultBuilders = 1;

  let assemblyFormat = [{
    attr-dict ($sym_visibility^)? $sym_name
    custom<ArgList>($annotations, $arg_names) (`:` type($annotations)^)?
    `type` $function_type
    $bodyRegion
  }];
}

def PyIR_ReturnOp : PyIR_OpBase<"return", [
  Pure, ParentOneOf<["PyFuncOp", "PyStaticFuncOp"]>, ReturnLike, Terminator
]> {
  let arguments = (ins AnyType:$operand);

  let assemblyFormat = "attr-dict $operand `:` type($operand)";
}

def PyIR_CallOp : PyIR_OpBase<"call", [
  DeclareOpInterfaceMethods<TypingKeyInterface>,
  DeclareOpInterfaceMethods<GetNamedArgInterface>
]> {
  let arguments = (ins
    AnyType:$func,
    Variadic<AnyType>:$args,
    ArrayAttr:$args_names);

  let results = (outs AnyType:$result);

  let hasCanonicalizer = 1;

  let assemblyFormat = [{
    attr-dict $func `:` type($func) ` ` custom<ArgList>($args, $args_names)
    (`:` type($args)^)? `->` type($result)
  }];
}

def PyIR_StaticCallOp : PyIR_OpBase<"static_call", [
  CallOpInterface, DeclareOpInterfaceMethods<SymbolUserOpInterface>,
  DeclareOpInterfaceMethods<TypingKeyInterface>
]> {
  let arguments = (ins
    FlatSymbolRefAttr:$callee,
    Variadic<AnyType>:$args,
    ArrayAttr:$args_names);

  let results = (outs AnyType:$result);

  let extraClassDeclaration = [{
    ::mlir::FunctionType getCalleeType();

    /// Get the argument operands to the called function.
    operand_range getArgOperands() {
      return {arg_operand_begin(), arg_operand_end()};
    }

    ::mlir::MutableOperandRange getArgOperandsMutable() {
      return getArgsMutable();
    }

    operand_iterator arg_operand_begin() { return operand_begin(); }
    operand_iterator arg_operand_end() { return operand_end(); }

    /// Return the callee of this operation.
    ::mlir::CallInterfaceCallable getCallableForCallee() {
      return (*this)->getAttrOfType<::mlir::SymbolRefAttr>("callee");
    }

    /// Set the callee for this operation.
    void setCalleeFromCallable(::mlir::CallInterfaceCallable callee) {
      (*this)->setAttr("callee", callee.get<::mlir::SymbolRefAttr>());
    }
  }];

  let assemblyFormat = [{
    attr-dict $callee `:` custom<ArgList>($args, $args_names)
    (`:` type($args)^)? `->` type($result)
  }];
}

def PyIR_StoreVarOp : PyIR_OpBase<"storevar", [MemoryEffects<[MemWrite]>]> {
  let arguments = (ins
    StrAttr:$name,
    AnyType:$value);

  let assemblyFormat = "attr-dict $name $value `:` type($value)";
}

def PyIR_LoadVarOp : PyIR_OpBase<"loadvar", [
  MemoryEffects<[MemRead]>,
  DeclareOpInterfaceMethods<TypingKeyInterface, ["getTypingKeyArgs"]>
]> {
  let arguments = (ins StrAttr:$name);
  let results = (outs AnyType:$result);

  let assemblyFormat = "attr-dict $name `:` type($result)";
}

def PyIR_SetItemOp : PyIR_OpBase<"setitem"> {
  let arguments = (ins
    AnyType:$target,
    AnyType:$index,
    AnyType:$value);

  let assemblyFormat = "attr-dict $target `:` type($target) `[` $index `:` type($index) `]` `=` $value `:` type($value)";
}

def PyIR_GetItemOp : PyIR_OpBase<"getitem", [
  DeclareOpInterfaceMethods<TypingKeyInterface>
]> {
  let arguments = (ins
    AnyType:$target,
    AnyType:$index);
  let results = (outs AnyType:$result);

  let assemblyFormat = "attr-dict $target `:` type($target) `[` $index `:` type($index) `]` `->` type($result)";
}

def PyIR_SetAttrOp : PyIR_OpBase<"setattr"> {
  let arguments = (ins
    AnyType:$target,
    StrAttr:$name,
    AnyType:$value);

  let assemblyFormat = "attr-dict $target `:` type($target) `attr` $name `=` $value `:` type($value)";
}

def PyIR_GetAttrOp : PyIR_OpBase<"getattr", [
  Pure, DeclareOpInterfaceMethods<TypingKeyInterface, ["getTypingKeyArgs"]>
]> {
  let arguments = (ins
    AnyType:$target,
    StrAttr:$name);
  let results = (outs AnyType:$result);

  let assemblyFormat = "attr-dict $target `:` type($target) `attr` $name `->` type($result)";
}

def PyIR_MakeListOp : PyIR_OpBase<"make_list", [
  Pure, DeclareOpInterfaceMethods<TypingKeyInterface>
]> {
  let arguments = (ins Variadic<AnyType>:$args);

  let results = (outs AnyType:$result);

  let assemblyFormat = "attr-dict $args `:` type($args) `->` type($result)";
}

def PyIR_TuplePackOp : PyIR_OpBase<"tuple_pack", [
  Pure, DeclareOpInterfaceMethods<TypingKeyInterface>
]> {
  let arguments = (ins Variadic<AnyType>:$args);

  let results = (outs AnyType:$result);

  let assemblyFormat = "attr-dict $args `:` type($args) `->` type($result)";
}

def PyIR_TupleUnpackOp : PyIR_OpBase<"tuple_unpack"> {
  let arguments = (ins AnyType:$arg);

  let results = (outs Variadic<AnyType>:$results);

  let assemblyFormat = "attr-dict $arg `:` type($arg) `->` type($results)";
}

def PyIR_SliceOp : PyIR_OpBase<"slice", [
  Pure,
  AttrSizedOperandSegments,
  DeclareOpInterfaceMethods<TypingKeyInterface>,
  DeclareOpInterfaceMethods<GetNamedArgInterface>
]> {
  let arguments = (ins
      Optional<AnyType>:$lower,
      Optional<AnyType>:$upper,
      Optional<AnyType>:$step);

  let results = (outs AnyType:$result);

  let assemblyFormat = [{
    attr-dict `(` $lower ` ` type($lower) `:`
                  $upper ` ` type($upper) `:`
                  $step  ` ` type($step)  `)`
                  `->` type($result)
  }];
}

def PyIR_CmpOpAttr : I64EnumAttr<
    "CmpOpVal", "",
    [
      I64EnumAttrCase<"eq", 0>,
      I64EnumAttrCase<"ne", 1>,
      I64EnumAttrCase<"lt", 2>,
      I64EnumAttrCase<"le", 3>,
      I64EnumAttrCase<"gt", 4>,
      I64EnumAttrCase<"ge", 5>,
      I64EnumAttrCase<"is", 6>,
      I64EnumAttrCase<"isn", 7>,
      I64EnumAttrCase<"in", 8>,
      I64EnumAttrCase<"nin", 9>,
    ]> {
  let cppNamespace = "::hc::py_ir";
}

def PyIR_BinOpAttr : I64EnumAttr<
    "BinOpVal", "",
    [
      I64EnumAttrCase<"add", 0>,
      I64EnumAttrCase<"sub", 1>,
      I64EnumAttrCase<"mul", 2>,
      I64EnumAttrCase<"div", 3>,
      I64EnumAttrCase<"floor_div", 4>,
      I64EnumAttrCase<"mod", 5>,
      I64EnumAttrCase<"pow", 6>,
      I64EnumAttrCase<"lshift", 7>,
      I64EnumAttrCase<"rshift", 8>,
      I64EnumAttrCase<"bit_or", 9>,
      I64EnumAttrCase<"bit_xor", 10>,
      I64EnumAttrCase<"bit_and", 11>,
      I64EnumAttrCase<"matmul", 12>,

      I64EnumAttrCase<"bool_and", 13>,
      I64EnumAttrCase<"bool_or", 14>,
    ]> {
  let cppNamespace = "::hc::py_ir";
}

def PyIR_UnaryOpAttr : I64EnumAttr<
    "UnaryOpVal", "",
    [
      I64EnumAttrCase<"invert", 0>,
      I64EnumAttrCase<"unot", 1>,
      I64EnumAttrCase<"uadd", 2>,
      I64EnumAttrCase<"usub", 3>,
    ]> {
  let cppNamespace = "::hc::py_ir";
}

def PyIR_BinOp : PyIR_OpBase<"binop", [
  DeclareOpInterfaceMethods<TypingKeyInterface>
]> {
  let arguments = (ins
      AnyType:$left,
      PyIR_BinOpAttr:$op,
      AnyType:$right);
  let results = (outs AnyType:$result);

  let assemblyFormat = "attr-dict $left`:`type($left) $op $right`:`type($right) `->` type($result)";
}

def PyIR_InplaceBinOp : PyIR_OpBase<"inplace_binop", [
  DeclareOpInterfaceMethods<TypingKeyInterface>
]> {
  let arguments = (ins
      AnyType:$left,
      PyIR_BinOpAttr:$op,
      AnyType:$right);
  let results = (outs AnyType:$result);

  let assemblyFormat = "attr-dict $left`:`type($left) $op $right`:`type($right) `->` type($result)";
}

def PyIR_UnaryOp : PyIR_OpBase<"unaryop"> {
  let arguments = (ins
      PyIR_UnaryOpAttr:$op,
      AnyType:$operand);
  let results = (outs AnyType:$result);

  let assemblyFormat = "attr-dict $op $operand`:`type($operand) `->` type($result)";
}

def PyIR_CmpOp : PyIR_OpBase<"cmp", [
  DeclareOpInterfaceMethods<TypingKeyInterface>
]> {
  let arguments = (ins
      AnyType:$left,
      PyIR_CmpOpAttr:$op,
      AnyType:$right);
  let results = (outs AnyType:$result);

  let assemblyFormat = "attr-dict $left`:`type($left) $op $right`:`type($right) `->` type($result)";
}

def PyIR_IterOp : PyIR_OpBase<"iter"> {
  let arguments = (ins
      AnyType:$cont);
  let results = (outs AnyType:$iterator);

  let assemblyFormat = "attr-dict $cont`:`type($cont) `->` type($iterator)";
}

def PyIR_NextOp : PyIR_OpBase<"next"> {
  let arguments = (ins
      AnyType:$iter);
  let results = (outs AnyType:$value, AnyI1:$valid, AnyType:$nextiter);

  let assemblyFormat = [{
    attr-dict $iter`:`type($iter)
    `->` type($value)`,` type($valid)`,` type($nextiter)
  }];
}

def PyIR_IfExpOp : PyIR_OpBase<"ifexp"> {
  let arguments = (ins
      AnyType:$test,
      AnyType:$body,
      AnyType:$orelse);
  let results = (outs AnyType:$result);

  let assemblyFormat = [{
    attr-dict
    $body`:`type($body) `if` $test`:`type($test) `else` $orelse`:`type($orelse)
    `->` type($result)
  }];
}

#endif // PYIR_OPS
