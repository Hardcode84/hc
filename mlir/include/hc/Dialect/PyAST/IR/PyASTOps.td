// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef PYAST_OPS
#define PYAST_OPS

include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/OpBase.td"

def PyAST_Dialect : Dialect {
  let name = "py_ast";
  let cppNamespace = "::hc::py_ast";

  let summary = "TBD.";
  let description = [{
    TBD
  }];

  let useDefaultAttributePrinterParser = 1;
  let useDefaultTypePrinterParser = 1;
}

class PyAST_Type<string name, string typeMnemonic, list<Trait> traits = [],
                   string baseCppClass = "::mlir::Type">
    : TypeDef<PyAST_Dialect, name, traits, baseCppClass> {
  let mnemonic = typeMnemonic;
}

class PyAST_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<PyAST_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

class PyAST_OpBase<string mnemonic, list<Trait> traits = []>
    : Op<PyAST_Dialect, mnemonic, traits>;

class PyAST_PureOpBase<string mnemonic, list<Trait> traits = []>
    : Op<PyAST_Dialect, mnemonic, !listconcat(traits, [Pure])> {}

def PyAST_BoolOpTypeAttr : I64EnumAttr<
    "BoolOpType", "",
    [
      I64EnumAttrCase<"and_", 0, "and">,
      I64EnumAttrCase<"or_", 1, "or">,
    ]> {
  let cppNamespace = "::hc::py_ast";
}

def PyAST_CmpOpAttr : I64EnumAttr<
    "CmpOp", "",
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
  let cppNamespace = "::hc::py_ast";
}

def PyAST_BinOpAttr : I64EnumAttr<
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
    ]> {
  let cppNamespace = "::hc::py_ast";
}

def PyAST_UnaryOpAttr : I64EnumAttr<
    "UnaryOpVal", "",
    [
      I64EnumAttrCase<"invert", 0>,
      I64EnumAttrCase<"unot", 1>,
      I64EnumAttrCase<"uadd", 2>,
      I64EnumAttrCase<"usub", 3>,
    ]> {
  let cppNamespace = "::hc::py_ast";
}

def PyAST_NoneAttr : PyAST_Attr<"None", "none">;

def PyAST_ASTNodeType : PyAST_Type<"Node", "node", [], "::mlir::Type">;

def NoReturn : NativeOpTrait<"NoReturn"> {
  let cppNamespace = "::hc::py_ast";
}

def PyAST_BlockEndOp : PyAST_OpBase<"block_end", [Terminator,
    ParentOneOf<["PyModuleOp", "PyFuncOp", "IfOp", "ForOp", "WhileOp"]>
]> {
  let assemblyFormat = "attr-dict";
}

def PyAST_PyModuleOp : PyAST_OpBase<"module", [
      IsolatedFromAbove, SingleBlockImplicitTerminator<"BlockEndOp">
    ]> {


  let regions = (region SizedRegion<1>:$bodyRegion);

  let builders = [
    OpBuilder<(ins)>
  ];

  let skipDefaultBuilders = 1;

  let assemblyFormat = "attr-dict-with-keyword $bodyRegion";
}

def PyAST_PyFuncOp : PyAST_OpBase<"func", [
      SingleBlockImplicitTerminator<"BlockEndOp">, AttrSizedOperandSegments
    ]> {

  let arguments = (ins
      StrAttr:$name,
      Variadic<PyAST_ASTNodeType>:$args,
      Variadic<PyAST_ASTNodeType>:$decorators);

  let regions = (region SizedRegion<1>:$bodyRegion);

  let builders = [
    OpBuilder<(ins
        "::llvm::StringRef":$name,
        "::mlir::ValueRange":$args,
        "::mlir::ValueRange":$decorators)>
  ];

  let skipDefaultBuilders = 1;

  let assemblyFormat = "attr-dict $name `(` $args `)` ( `decorators` $decorators^ )? $bodyRegion";
}

def PyAST_PassOp : PyAST_OpBase<"pass"> {
  let assemblyFormat = "attr-dict";
}

def PyAST_ContinueOp : PyAST_OpBase<"continue", [NoReturn]> {
  let assemblyFormat = "attr-dict";
}

def PyAST_BreakOp : PyAST_OpBase<"break", [NoReturn]> {
  let assemblyFormat = "attr-dict";
}

def PyAST_ArgOp : PyAST_PureOpBase<"arg"> {
  let arguments = (ins StrAttr:$name, Optional<PyAST_ASTNodeType>:$annotation);
  let results = (outs PyAST_ASTNodeType:$result);

  let builders = [
    OpBuilder<(ins
        "::llvm::StringRef":$name,
        CArg<"::mlir::Value", "nullptr">:$annotation)>
  ];

  let assemblyFormat = "attr-dict $name ( `:` $annotation^ )?";
}

def PyAST_NameOp : PyAST_PureOpBase<"name"> {
  let arguments = (ins StrAttr:$id);
  let results = (outs PyAST_ASTNodeType:$result);

  let builders = [
    OpBuilder<(ins"::llvm::StringRef":$id)>
  ];

  let assemblyFormat = "attr-dict $id";
}

def PyAST_SubscriptOp : PyAST_PureOpBase<"subscript"> {
  let arguments = (ins PyAST_ASTNodeType:$value, PyAST_ASTNodeType:$slice);
  let results = (outs PyAST_ASTNodeType:$result);

  let builders = [
    OpBuilder<(ins
        "::mlir::Value":$value,
        "::mlir::Value":$slice)>
  ];

  let assemblyFormat = "attr-dict $value `[` $slice `]`";
}

def PyAST_ExprOp : PyAST_OpBase<"expr"> {
  let arguments = (ins PyAST_ASTNodeType:$value);

  let assemblyFormat = "attr-dict $value";
}

def PyAST_ListOp : PyAST_PureOpBase<"list"> {
  let arguments = (ins Variadic<PyAST_ASTNodeType>:$elts);
  let results = (outs PyAST_ASTNodeType:$result);

  let builders = [
    OpBuilder<(ins "::mlir::ValueRange":$elts)>
  ];

  let assemblyFormat = "attr-dict $elts";
}

def PyAST_TupleOp : PyAST_PureOpBase<"tuple"> {
  let arguments = (ins Variadic<PyAST_ASTNodeType>:$elts);
  let results = (outs PyAST_ASTNodeType:$result);

  let builders = [
    OpBuilder<(ins "::mlir::ValueRange":$elts)>
  ];

  let assemblyFormat = "attr-dict $elts";
}

def PyAST_AttributeOp : PyAST_PureOpBase<"attribute"> {
  let arguments = (ins PyAST_ASTNodeType:$value, StrAttr:$attr);
  let results = (outs PyAST_ASTNodeType:$result);

  let builders = [
    OpBuilder<(ins
        "::mlir::Value":$value,
        "::llvm::StringRef":$attr)>
  ];

  let assemblyFormat = "attr-dict $value `attr` $attr";
}

def PyAST_ConstantOp : PyAST_PureOpBase<"constant"> {
  let arguments = (ins AnyAttr:$value);
  let results = (outs PyAST_ASTNodeType:$result);

  let builders = [
    OpBuilder<(ins "::mlir::Attribute":$value)>
  ];

  let assemblyFormat = "attr-dict $value";
}

def PyAST_SliceOp : PyAST_PureOpBase<"slice", [AttrSizedOperandSegments]> {
  let arguments = (ins
      Optional<PyAST_ASTNodeType>:$lower,
      Optional<PyAST_ASTNodeType>:$upper,
      Optional<PyAST_ASTNodeType>:$step);

  let results = (outs PyAST_ASTNodeType:$result);

  let builders = [
    OpBuilder<(ins
      "::mlir::Value":$lower,
      "::mlir::Value":$upper,
      "::mlir::Value":$step)>
  ];

  let assemblyFormat = "attr-dict `(` $lower `:` $upper `:` $step `)`";
}

def PyAST_AssignOp : PyAST_OpBase<"assign"> {
  let arguments = (ins
      Variadic<PyAST_ASTNodeType>:$targets,
      PyAST_ASTNodeType:$value);

  let assemblyFormat = "attr-dict `(` $targets `)` `=` $value";
}

def PyAST_CallOp : PyAST_PureOpBase<"call", [AttrSizedOperandSegments]> {
  let arguments = (ins
      PyAST_ASTNodeType:$func,
      Variadic<PyAST_ASTNodeType>:$args,
      Variadic<PyAST_ASTNodeType>:$keywords);

  let results = (outs PyAST_ASTNodeType:$result);

  let builders = [
    OpBuilder<(ins
      "::mlir::Value":$func,
      "::mlir::ValueRange":$args,
      "::mlir::ValueRange":$keywods)>
  ];

  let assemblyFormat = "attr-dict $func `(` $args `keywords` $keywords `)`";
}

def PyAST_KeywordOp : PyAST_PureOpBase<"keyword"> {
  let arguments = (ins StrAttr:$arg, PyAST_ASTNodeType:$value);
  let results = (outs PyAST_ASTNodeType:$result);

  let builders = [
    OpBuilder<(ins
        "::llvm::StringRef":$arg,
        "::mlir::Value":$value)>
  ];

  let assemblyFormat = "attr-dict $arg `=` $value";
}

def PyAST_BoolOp : PyAST_PureOpBase<"bool_op"> {
  let arguments = (ins
      PyAST_BoolOpTypeAttr:$op,
      Variadic<PyAST_ASTNodeType>:$values);
  let results = (outs PyAST_ASTNodeType:$result);

  let builders = [
    OpBuilder<(ins
        "BoolOpType":$arg,
        "::mlir::ValueRange":$values)>
  ];

  let assemblyFormat = "attr-dict $op `,` $values";
}

def PyAST_IfOp : PyAST_OpBase<"if", [
      SingleBlockImplicitTerminator<"BlockEndOp">
    ]> {

  let arguments = (ins PyAST_ASTNodeType:$test);

  let regions = (region SizedRegion<1>:$bodyRegion,
                        MaxSizedRegion<1>:$orelseRegion);

  let builders = [
    OpBuilder<(ins "::mlir::Value":$test, "bool":$hasElse)>
  ];

  let skipDefaultBuilders = 1;

  let assemblyFormat = "attr-dict $test $bodyRegion $orelseRegion";
}

def PyAST_ForOp : PyAST_OpBase<"for", [
      SingleBlockImplicitTerminator<"BlockEndOp">
    ]> {

  let arguments = (ins PyAST_ASTNodeType:$target, PyAST_ASTNodeType:$iter);

  let regions = (region SizedRegion<1>:$bodyRegion);

  let builders = [
    OpBuilder<(ins "::mlir::Value":$target, "::mlir::Value":$iter)>
  ];

  let skipDefaultBuilders = 1;

  let assemblyFormat = "attr-dict $target `in` $iter $bodyRegion";
}

def PyAST_WhileOp : PyAST_OpBase<"while", [
      SingleBlockImplicitTerminator<"BlockEndOp">
    ]> {

  let arguments = (ins PyAST_ASTNodeType:$test);

  let regions = (region SizedRegion<1>:$bodyRegion);

  let builders = [
    OpBuilder<(ins "::mlir::Value":$test)>
  ];

  let skipDefaultBuilders = 1;

  let assemblyFormat = "attr-dict $test $bodyRegion";
}

def PyAST_CompareOp : PyAST_PureOpBase<"compare"> {
  let arguments = (ins
      PyAST_ASTNodeType:$left,
      ArrayAttr:$ops,
      Variadic<PyAST_ASTNodeType>:$comparators);
  let results = (outs PyAST_ASTNodeType:$result);

  let builders = [
    OpBuilder<(ins
        "::mlir::Value":$left,
        "::mlir::ArrayRef<CmpOp>":$ops,
        "::mlir::ValueRange":$comparators)>
  ];

// TODO: fix enum printing/parsing
  let assemblyFormat = "attr-dict $left $ops $comparators";
}

def PyAST_BinOp : PyAST_PureOpBase<"binop"> {
  let arguments = (ins
      PyAST_ASTNodeType:$left,
      PyAST_BinOpAttr:$op,
      PyAST_ASTNodeType:$right);
  let results = (outs PyAST_ASTNodeType:$result);

  let builders = [
    OpBuilder<(ins
        "::mlir::Value":$left,
        "BinOpVal":$op,
        "::mlir::Value":$right)>
  ];

  let assemblyFormat = "attr-dict $left $op $right";
}

def PyAST_AugAssignOp : PyAST_OpBase<"aug_assign"> {
  let arguments = (ins
      PyAST_ASTNodeType:$target,
      PyAST_BinOpAttr:$op,
      PyAST_ASTNodeType:$value);

  let assemblyFormat = "attr-dict $target $op $value";
}

def PyAST_UnaryOp : PyAST_PureOpBase<"unaryop"> {
  let arguments = (ins
      PyAST_UnaryOpAttr:$op,
      PyAST_ASTNodeType:$operand);
  let results = (outs PyAST_ASTNodeType:$result);

  let builders = [
    OpBuilder<(ins
        "UnaryOpVal":$op,
        "::mlir::Value":$operand)>
  ];

  let assemblyFormat = "attr-dict $op $operand";
}

def PyAST_IfExpOp : PyAST_PureOpBase<"ifexp"> {
  let arguments = (ins
      PyAST_ASTNodeType:$test,
      PyAST_ASTNodeType:$body,
      PyAST_ASTNodeType:$orelse);
  let results = (outs PyAST_ASTNodeType:$result);

  let builders = [
    OpBuilder<(ins
        "::mlir::Value":$test,
        "::mlir::Value":$body,
        "::mlir::Value":$orelse)>
  ];

  let assemblyFormat = "attr-dict $body `if` $test `else` $orelse";
}

def PyAST_ReturnOp : PyAST_OpBase<"return", [NoReturn]> {
  let arguments = (ins Optional<PyAST_ASTNodeType>:$value);

  let assemblyFormat = "attr-dict $value";
}

def PyAST_CaptureValOp : PyAST_OpBase<"capture_val"> {
  let arguments = (ins StrAttr:$id);

  let assemblyFormat = "attr-dict $id";
}

#endif // PYAST_OPS
