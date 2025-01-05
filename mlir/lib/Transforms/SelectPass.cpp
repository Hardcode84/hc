// SPDX-FileCopyrightText: 2025 The HC Authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hc/Transforms/Passes.hpp"

#include <mlir/Pass/PassManager.h>

namespace hc {
#define GEN_PASS_DEF_SELECTPASS
#include "hc/Transforms/Passes.h.inc"
} // namespace hc

using namespace mlir;

namespace {
struct SelectPass final : public hc::impl::SelectPassBase<SelectPass> {
  using SelectPassBase::SelectPassBase;

  SelectPass(std::string name_,
             mlir::ArrayRef<std::pair<
                 mlir::StringRef, std::function<void(mlir::OpPassManager &)>>>
                 populateFuncs) {
    name = std::move(name_);

    SmallVector<std::string> selectVals;
    SmallVector<std::string> selectPpls;
    selectVals.reserve(populateFuncs.size());
    selectPpls.reserve(populateFuncs.size());
    selectPassManagers.reserve(populateFuncs.size());
    for (auto &&[name, populate] : populateFuncs) {
      selectVals.emplace_back(name);

      auto &pm = selectPassManagers.emplace_back();
      populate(pm);

      llvm::raw_string_ostream os(selectPpls.emplace_back());
      pm.printAsTextualPipeline(os);
    }

    selectValues = selectVals;
    selectPipelines = selectPpls;
  }

  LogicalResult initializeOptions(
      StringRef options,
      function_ref<LogicalResult(const Twine &)> errorHandler) override {
    if (failed(SelectPassBase::initializeOptions(options, errorHandler)))
      return failure();

    if (selectCondName.empty())
      return errorHandler("Invalid select-cond-name");

    if (selectValues.size() != selectPipelines.size())
      return errorHandler("Values and pipelines size mismatch");

    selectPassManagers.resize(selectPipelines.size());

    for (auto &&[i, pipeline] : llvm::enumerate(selectPipelines)) {
      if (failed(parsePassPipeline(pipeline, selectPassManagers[i])))
        return errorHandler("Failed to parse pipeline");
    }

    return success();
  }

  LogicalResult initialize(MLIRContext *context) override {
    condAttrName = StringAttr::get(context, selectCondName);

    selectAttrs.reserve(selectAttrs.size());
    for (StringRef value : selectValues)
      selectAttrs.emplace_back(StringAttr::get(context, value));

    return success();
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    for (const OpPassManager &pipeline : selectPassManagers)
      pipeline.getDependentDialects(registry);
  }

  void runOnOperation() override {
    Operation *op = getOperation();
    Attribute condAttrValue = op->getAttr(condAttrName);
    if (!condAttrValue) {
      op->emitError("Condition attribute not present");
      return signalPassFailure();
    }

    for (auto &&[value, pm] :
         llvm::zip_equal(selectAttrs, selectPassManagers)) {
      if (value != condAttrValue)
        continue;

      if (failed(runPipeline(pm, op)))
        return signalPassFailure();

      return;
    }

    op->emitError("Unhandled condition value: ") << condAttrValue;
    return signalPassFailure();
  }

protected:
  StringRef getName() const override { return name; }

private:
  StringAttr condAttrName;
  SmallVector<Attribute> selectAttrs;
  SmallVector<OpPassManager> selectPassManagers;
};
} // namespace

std::unique_ptr<mlir::Pass> hc::createSelectPass(
    std::string name,
    mlir::ArrayRef<
        std::pair<mlir::StringRef, std::function<void(mlir::OpPassManager &)>>>
        populateFuncs) {
  return std::make_unique<SelectPass>(std::move(name), populateFuncs);
}
