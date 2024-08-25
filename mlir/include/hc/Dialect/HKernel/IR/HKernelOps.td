// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef HKERNEL_OPS
#define HKERNEL_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

include "hc/Dialect/HKernel/IR/HKernelOpsTypeInterfaces.td"

def HKernel_Dialect : Dialect {
  let name = "hkernel";
  let cppNamespace = "::hc::hk";

  let summary = "TBD.";
  let description = [{
    TBD
  }];

//  let useDefaultAttributePrinterParser = 1;
  let useDefaultTypePrinterParser = 1;
}

class HKernel_Type<string name, string typeMnemonic, list<Trait> traits = [],
                   string baseCppClass = "::mlir::Type">
    : TypeDef<HKernel_Dialect, name, traits, baseCppClass> {
  let mnemonic = typeMnemonic;
}

class HKernel_OpBase<string mnemonic, list<Trait> traits = []>
    : Op<HKernel_Dialect, mnemonic, traits>;

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

  let assemblyFormat = "$shape `x` $elementType";

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
    AnyTuple:$src,
    Index:$index
  );
  let results = (outs AnyType:$result);

  let assemblyFormat = "attr-dict $src `:` type($src) `[` $index `]` `->` type($result)";
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

#endif // HKERNEL_OPS
