// SPDX-FileCopyrightText: 2024 The HC Authors
// SPDX-FileCopyrightText: 2025 The HC Authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hc/Transforms/Passes.hpp"

#include "hc/Dialect/HKernel/IR/HKernelOps.hpp"
#include "hc/Utils.hpp"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Transforms/DialectConversion.h>

namespace hc {
#define GEN_PASS_DEF_EXPANDTUPLEPASS
#include "hc/Transforms/Passes.h.inc"
} // namespace hc

static void populateTypeConverter(mlir::MLIRContext *ctx,
                                  mlir::TypeConverter &converter) {
  converter.addConversion(
      [&converter](mlir::TupleType type, llvm::SmallVectorImpl<mlir::Type> &ret)
          -> std::optional<mlir::LogicalResult> {
        llvm::SmallVector<mlir::Type> expanded;
        for (auto t : type.getTypes()) {
          expanded.clear();
          if (mlir::failed(converter.convertTypes(t, expanded)))
            return std::nullopt;

          ret.append(expanded);
        }

        return mlir::success();
      });
}
using namespace mlir;
/// Creates a sequence of `test.get_tuple_element` ops for all elements of a
/// given tuple value. If some tuple elements are, in turn, tuples, the elements
/// of those are extracted recursively such that the returned values have the
/// same types as `resultTypes.getFlattenedTypes()`.
static SmallVector<Value> buildDecomposeTuple(OpBuilder &builder,
                                              TypeRange resultTypes,
                                              ValueRange inputs, Location loc) {
  // Skip materialization if the single input value is not a tuple.
  if (inputs.size() != 1)
    return {};
  Value tuple = inputs.front();
  auto tupleType = dyn_cast<TupleType>(tuple.getType());
  if (!tupleType)
    return {};
  // Skip materialization if the flattened types do not match the requested
  // result types.
  SmallVector<Type> flattenedTypes;
  tupleType.getFlattenedTypes(flattenedTypes);
  if (TypeRange(resultTypes) != TypeRange(flattenedTypes))
    return {};
  // Recursively decompose the tuple.
  SmallVector<Value> result;
  std::function<void(Value)> decompose = [&](Value tuple) {
    auto tupleType = dyn_cast<TupleType>(tuple.getType());
    if (!tupleType) {
      // This is not a tuple.
      result.push_back(tuple);
      return;
    }
    for (unsigned i = 0, e = tupleType.size(); i < e; ++i) {
      Type elementType = tupleType.getType(i);
      Value idx = builder.create<mlir::arith::ConstantIndexOp>(loc, i);
      Value element =
          builder.create<hc::hk::TupleExtractOp>(loc, elementType, tuple, idx);
      decompose(element);
    }
  };
  decompose(tuple);
  return result;
}

/// Creates a `test.make_tuple` op out of the given inputs building a tuple of
/// type `resultType`. If that type is nested, each nested tuple is built
/// recursively with another `test.make_tuple` op.
static Value buildMakeTupleOp(OpBuilder &builder, TupleType resultType,
                              ValueRange inputs, Location loc) {
  // Build one value for each element at this nesting level.
  SmallVector<Value> elements;
  elements.reserve(resultType.getTypes().size());
  ValueRange::iterator inputIt = inputs.begin();
  for (Type elementType : resultType.getTypes()) {
    if (auto nestedTupleType = dyn_cast<TupleType>(elementType)) {
      // Determine how many input values are needed for the nested elements of
      // the nested TupleType and advance inputIt by that number.
      // TODO: We only need the *number* of nested types, not the types itself.
      //       Maybe it's worth adding a more efficient overload?
      SmallVector<Type> nestedFlattenedTypes;
      nestedTupleType.getFlattenedTypes(nestedFlattenedTypes);
      size_t numNestedFlattenedTypes = nestedFlattenedTypes.size();
      ValueRange nestedFlattenedelements(inputIt,
                                         inputIt + numNestedFlattenedTypes);
      inputIt += numNestedFlattenedTypes;

      // Recurse on the values for the nested TupleType.
      Value res = buildMakeTupleOp(builder, nestedTupleType,
                                   nestedFlattenedelements, loc);
      if (!res)
        return Value();

      // The tuple constructed by the conversion is the element value.
      elements.push_back(res);
    } else {
      // Base case: take one input as is.
      elements.push_back(*inputIt++);
    }
  }

  // Assemble the tuple from the elements.
  return builder.create<hc::hk::MakeTupleOp>(loc, resultType, elements);
}

namespace {
struct ExpandTuplePass final
    : public hc::impl::ExpandTuplePassBase<ExpandTuplePass> {

  void runOnOperation() override {
    auto mod = getOperation();

    auto *ctx = &getContext();
    mlir::ConversionTarget target(*ctx);
    mlir::TypeConverter converter;

    // Convert unknown types to itself
    converter.addConversion([](mlir::Type type) { return type; });

    populateTypeConverter(ctx, converter);

    converter.addArgumentMaterialization(buildMakeTupleOp);
    converter.addTargetMaterialization(buildDecomposeTuple);

    mlir::RewritePatternSet patterns(ctx);

    hc::populateFuncPatternsAndTypeConversion(patterns, target, converter);

    target.markUnknownOpDynamicallyLegal(
        [&](mlir::Operation *op) -> bool { return converter.isLegal(op); });

    if (mlir::failed(
            mlir::applyFullConversion(mod, target, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace
