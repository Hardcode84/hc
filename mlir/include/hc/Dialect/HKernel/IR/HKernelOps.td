// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef HKERNEL_OPS
#define HKERNEL_OPS

include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

include "hc/Dialect/HKernel/IR/HKernelOpsTypeInterfaces.td"

def HKernel_Dialect : Dialect {
  let name = "hkernel";
  let cppNamespace = "::hc::hk";

  let summary = "TBD.";
  let description = [{
    TBD
  }];

  let dependentDialects = ["::hc::typing::TypingDialect"];

  let useDefaultAttributePrinterParser = 1;
  let useDefaultTypePrinterParser = 1;
}

class HKernel_Type<string name, string typeMnemonic, list<Trait> traits = [],
                   string baseCppClass = "::mlir::Type">
    : TypeDef<HKernel_Dialect, name, traits, baseCppClass> {
  let mnemonic = typeMnemonic;
}

class HKernel_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<HKernel_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

class HKernel_OpBase<string mnemonic, list<Trait> traits = []>
    : Op<HKernel_Dialect, mnemonic, traits>;


def HKernel_WorkgroupScopeAttr : HKernel_Attr<"WorkgroupScope", "workgroup_scope">;

def HKernel_SubgroupScopeAttr : HKernel_Attr<"SubgroupScope", "subgroup_scope">;

def HKernel_WorkitemScopeAttr : HKernel_Attr<"WorkitemScope", "workitem_scope">;

def HKernel_CurrentGroupType : HKernel_Type<"CurrentGroup", "current_group", [], "::mlir::Type"> {
  let summary = "TBD";
  let description = [{
    TBD
  }];
  let parameters = (ins "int":$num_dims);

  let assemblyFormat = "$num_dims";
}

def HKernel_SliceType : HKernel_Type<"Slice", "slice", [], "::mlir::Type"> {
  let summary = "TBD";
  let description = [{
    TBD
  }];
  let parameters = (ins
    "::mlir::Type":$lower,
    "::mlir::Type":$upper,
    "::mlir::Type":$step
  );

  let assemblyFormat = "$lower `:` $upper `:` $step";
}

def HKernel_BufferType : HKernel_Type<"Buffer", "buffer", [
  SymbolicallyShapedTypeInterface
], "::mlir::Type"> {
  let summary = "TBD";
  let description = [{
    TBD
  }];
  let parameters = (ins
    ArrayRefParameter<"::mlir::Type">:$shape,
    "::mlir::Type":$elementType
  );

  let assemblyFormat = "custom<SymbolicShape>($shape) `x` $elementType";

  let extraClassDeclaration = [{
    ::hc::hk::SymbolicallyShapedType cloneWith(
      std::optional<llvm::ArrayRef<mlir::Type>> shape,
      mlir::Type elementType) const;
  }];
}

def HKernel_TensorType : HKernel_Type<"Tensor", "tensor", [
  SymbolicallyShapedTypeInterface
], "::mlir::Type"> {
  let summary = "TBD";
  let description = [{
    TBD
  }];
  let parameters = (ins
    ArrayRefParameter<"::mlir::Type">:$shape,
    "::mlir::Type":$elementType
  );

  let assemblyFormat = "custom<SymbolicShape>($shape) `x` $elementType";

  let extraClassDeclaration = [{
    ::hc::hk::SymbolicallyShapedType cloneWith(
      std::optional<llvm::ArrayRef<mlir::Type>> shape,
      mlir::Type elementType) const;
  }];
}

def HKernel_MakeTupleOp : HKernel_OpBase<"make_tuple", [Pure]> {
  let arguments = (ins Variadic<AnyType>:$args);
  let results = (outs AnyTuple:$result);

  let assemblyFormat = "attr-dict $args `:` type($args) `->` type($result)";
}

def HKernel_TupleExtractOp : HKernel_OpBase<"tuple_extract", [Pure]> {
  let arguments = (ins
    AnyTuple:$source,
    Index:$index
  );
  let results = (outs AnyType:$result);

  let assemblyFormat = "attr-dict $source `:` type($source) `[` $index `]` `->` type($result)";

  let hasFolder = 1;
}

def HKernel_MakeSliceOp : HKernel_OpBase<"make_slice", [
  Pure, AttrSizedOperandSegments
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

def HKernel_MaterializeExprOp : HKernel_OpBase<"materialize_expr", [Pure]> {
  let results = (outs AnyType:$result);

  let assemblyFormat = "attr-dict type($result)";
}

def HKernel_SubViewOp : HKernel_OpBase<"subview", [Pure]> {
  let arguments = (ins
    AnyType:$source,
    Variadic<AnyType>:$index
  );
  let results = (outs AnyType:$result);

  let assemblyFormat = "attr-dict $source `:` type($source) `[` $index `]` `:` type($index) `->` type($result)";
}

def HKernel_LoadOp : HKernel_OpBase<"load", []> {
  let arguments = (ins
    Arg<AnyType, "", [MemRead]>:$source,
    Variadic<AnyType>:$index
  );
  let results = (outs AnyType:$result);

  let assemblyFormat = "attr-dict $source `:` type($source) `[` $index `]` `:` type($index) `->` type($result)";
}

def HKernel_StoreOp : HKernel_OpBase<"store", []> {
  let arguments = (ins
    Arg<AnyType, "", [MemWrite]>:$target,
    Arg<AnyType, "", [MemRead]>:$source
  );

  let assemblyFormat = "attr-dict $target `:` type($target) `=` $source `:` type($source)";
}

def EnvironmentRegionOp : HKernel_OpBase<"env_region", [
    DeclareOpInterfaceMethods<RegionBranchOpInterface>,
    SingleBlockImplicitTerminator<"::hc::hk::EnvironmentRegionYieldOp">,
    RecursiveMemoryEffects
]> {
  let summary = "Operation that executes its region with a specific environment";
  let description = [{
    "env_region" executes operations inside its region within a specific
    enviroment. Operations are executed exactly once. All SSA values that
    dominate the op can be accessed inside the op.

    "env_region" takes "environment" attribute and zero or more SSA arguments.

    Actual intepretation of the "environment" attribute and arguments is not
    specified here and is left to the lowering passes.

    Values can be yielded from "env_region" region using "env_region_yield" op.
    Values yielded from this op's region define the op's results.
  }];


  let arguments = (ins AnyAttr:$environment, Variadic<AnyType>:$args);
  let results = (outs Variadic<AnyType>:$results);

  let regions = (region SizedRegion<1>:$region);

  let assemblyFormat =
      "attr-dict $environment ($args^ `:` type($args))? (`->` type($results)^)? $region";

  let extraClassDeclaration = [{
    /// Inline op body into parent region and erase the op.
    static void inlineIntoParent(::mlir::PatternRewriter &builder, EnvironmentRegionOp op);
  }];

  let builders = [
    OpBuilder<(ins "::mlir::Attribute":$environment,
      CArg<"::mlir::ValueRange", "std::nullopt">:$args,
      CArg<"::mlir::TypeRange", "std::nullopt">:$results,
      CArg<"::llvm::function_ref<void(::mlir::OpBuilder &, ::mlir::Location)>", "nullptr">)>
  ];

  let hasCanonicalizer = 1;
}

def EnvironmentRegionYieldOp : HKernel_OpBase<"env_region_yield", [
  Pure,
  ReturnLike,
  Terminator,
  HasParent<"::hc::hk::EnvironmentRegionOp">
]> {
  let summary = "Environment region yield and termination operation";
  let description = [{
    "env_region_yield" yields an SSA value from the "env_region" op region and
    terminates it.

    If "env_region_yield" has any operands, the operands must match the parent
    operation's results.
  }];

  let arguments = (ins Variadic<AnyType>:$results);
  let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];

  let assemblyFormat = "attr-dict ($results^ `:` type($results))?";
}

#endif // HKERNEL_OPS
