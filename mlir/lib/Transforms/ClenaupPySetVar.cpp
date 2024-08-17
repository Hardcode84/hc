// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hc/Transforms/Passes.hpp"

#include "hc/Dialect/PyIR/IR/PyIROps.hpp"

namespace hc {
#define GEN_PASS_DEF_CLEANUPPYSETVARPASS
#include "hc/Transforms/Passes.h.inc"
} // namespace hc

namespace {
struct CleanupPySetVarPass final
    : public hc::impl::CleanupPySetVarPassBase<CleanupPySetVarPass> {

  void runOnOperation() override {
    llvm::SmallDenseSet<mlir::StringAttr> liveNames;
    getOperation()->walk([&](hc::py_ir::PyModuleOp mod) {
      liveNames.clear();
      mod.walk([&](hc::py_ir::LoadVarOp load) {
        liveNames.insert(load.getNameAttr());
      });

      mod.walk([&](hc::py_ir::StoreVarOp store) {
        if (!liveNames.contains(store.getNameAttr()))
          store->erase();
      });
    });

    llvm::SmallDenseMap<mlir::StringAttr, hc::py_ir::StoreVarOp> deadStores;
    getOperation()->walk([&](mlir::Block *block) {
      deadStores.clear();
      for (mlir::Operation &op : llvm::make_early_inc_range(*block)) {
        if (auto store = mlir::dyn_cast<hc::py_ir::StoreVarOp>(op)) {
          auto name = store.getNameAttr();
          auto it = deadStores.find(name);
          if (it != deadStores.end())
            it->second->erase();

          deadStores[name] = store;
        } else if (auto load = mlir::dyn_cast<hc::py_ir::LoadVarOp>(op)) {
          deadStores.erase(load.getNameAttr());
        }
      }
    });
  }
};
} // namespace
