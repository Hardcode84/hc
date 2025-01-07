// SPDX-FileCopyrightText: 2025 The HC Authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hc/Transforms/Passes.hpp"

#include "hc/Dialect/HKernel/IR/HKernelOps.hpp"

#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>

namespace hc {
#define GEN_PASS_DEF_EXPANDSHAREDALLOCSPASS
#include "hc/Transforms/Passes.h.inc"
} // namespace hc

namespace {
struct ExpandSharedAllocsPass final
    : public hc::impl::ExpandSharedAllocsPassBase<ExpandSharedAllocsPass> {

  void runOnOperation() override {
    mlir::OpBuilder builder(&getContext());

    auto attrName =
        builder.getStringAttr(hc::hk::getKernelAllocExpandAttrName());
    auto visitor = [&](mlir::Operation *op) -> mlir::WalkResult {
      if (!op->hasAttr(attrName))
        return mlir::WalkResult::advance();

      if (op->getNumResults() != 1 ||
          !mlir::isa<mlir::MemRefType>(op->getResult(0).getType())) {
        op->emitError("Invalid result type");
        return mlir::WalkResult::interrupt();
      }

      auto launch = op->getParentOfType<mlir::gpu::LaunchOp>();
      if (!launch) {
        op->emitError("No inside launch op");
        return mlir::WalkResult::interrupt();
      }

      auto resType = mlir::cast<mlir::MemRefType>(op->getResult(0).getType());
      mlir::SmallVector<int64_t> shape(3, mlir::ShapedType::kDynamic);
      llvm::append_range(shape, resType.getShape());
      auto newResType = mlir::MemRefType::get(
          shape, resType.getElementType(), nullptr, resType.getMemorySpace());

      auto blockSizes = launch.getBlockSize();
      auto threadIds = launch.getThreadIds();

      mlir::SmallVector<mlir::Value> args(
          {blockSizes.x, blockSizes.y, blockSizes.z});
      llvm::append_range(args, op->getOperands());

      builder.setInsertionPoint(op);
      mlir::Location loc = op->getLoc();

      mlir::SmallVector<mlir::OpFoldResult> offsets(
          {threadIds.x, threadIds.y, threadIds.z});
      mlir::SmallVector<mlir::OpFoldResult> sizes(3,
                                                  builder.getI64IntegerAttr(1));
      mlir::SmallVector<mlir::OpFoldResult> strides(
          newResType.getRank(), builder.getI64IntegerAttr(1));

      offsets.resize(newResType.getRank(), builder.getI64IntegerAttr(0));

      mlir::Value res;
      if (auto alloc = mlir::dyn_cast<mlir::memref::AllocOp>(op)) {
        llvm::append_range(sizes, alloc.getMixedSizes());
        res = builder.create<mlir::memref::AllocOp>(loc, newResType, args,
                                                    alloc.getAlignmentAttr());
      } else if (auto alloc = mlir::dyn_cast<mlir::memref::AllocaOp>(op)) {
        llvm::append_range(sizes, alloc.getMixedSizes());
        res = builder.create<mlir::memref::AllocaOp>(loc, newResType, args,
                                                     alloc.getAlignmentAttr());
      } else {
        op->emitError("Unsupported op");
        return mlir::WalkResult::interrupt();
      }

      auto subViewType = mlir::cast<mlir::MemRefType>(
          mlir::memref::SubViewOp::inferRankReducedResultType(
              resType.getShape(), newResType, offsets, sizes, strides));
      res = builder.create<mlir::memref::SubViewOp>(loc, subViewType, res,
                                                    offsets, sizes, strides);

      if (subViewType != resType)
        res = builder.create<mlir::memref::CastOp>(loc, resType, res);

      op->replaceAllUsesWith(mlir::ArrayRef(res));
      op->erase();
      return mlir::WalkResult::advance();
    };

    if (getOperation()->walk(visitor).wasInterrupted())
      return signalPassFailure();
  }
};
} // namespace
