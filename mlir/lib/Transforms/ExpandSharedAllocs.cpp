// SPDX-FileCopyrightText: 2025 The HC Authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hc/Transforms/Passes.hpp"

#include "hc/Dialect/HKernel/IR/HKernelOps.hpp"

#include <mlir/Analysis/DataLayoutAnalysis.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/Dominance.h>
#include <mlir/IR/IRMapping.h>

namespace hc {
#define GEN_PASS_DEF_EXPANDSHAREDALLOCSPASS
#include "hc/Transforms/Passes.h.inc"

#define GEN_PASS_DEF_LEGALIZEDYNAMICSHAREDMEMPASS
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

struct LegalizeDynamicSharedMemPass final
    : public hc::impl::LegalizeDynamicSharedMemPassBase<
          LegalizeDynamicSharedMemPass> {

  void runOnOperation() override {
    auto *ctx = &getContext();
    mlir::OpBuilder builder(ctx);

    auto &dl = getAnalysis<mlir::DataLayoutAnalysis>();
    auto wgAttr = mlir::gpu::AddressSpaceAttr::get(
        ctx, mlir::gpu::GPUDialect::getWorkgroupAddressSpace());
    auto i32 = builder.getI32Type();

    mlir::DominanceInfo dom;
    mlir::IRMapping mapping;
    auto visitor = [&](mlir::gpu::LaunchOp launch) -> mlir::WalkResult {
      mapping.clear();
      auto toRange =
          [&](const mlir::gpu::KernelDim3 &src) -> std::array<mlir::Value, 3> {
        return {src.x, src.y, src.z};
      };
      mapping.map(toRange(launch.getBlockSize()),
                  toRange(launch.getBlockSizeOperandValues()));
      mapping.map(toRange(launch.getGridSize()),
                  toRange(launch.getGridSizeOperandValues()));

      const mlir::DataLayout &layout = dl.getAtOrAbove(launch);

      mlir::Value dynamicMem;
      mlir::Value dynamicMemSize;
      auto getDynamicMem = [&](unsigned elemSize, mlir::ValueRange sizes,
                               mlir::MemRefType resType) -> mlir::Value {
        mlir::Location loc = builder.getUnknownLoc();
        if (!dynamicMem) {
          builder.setInsertionPoint(launch);
          dynamicMemSize = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
          mlir::OpBuilder::InsertionGuard g(builder);
          builder.setInsertionPointToStart(&launch.getBody().front());
          auto resType = mlir::MemRefType::get(
              mlir::ShapedType::kDynamic, builder.getI8Type(), nullptr, wgAttr);
          dynamicMem =
              builder.create<mlir::gpu::DynamicSharedMemoryOp>(loc, resType);
        }
        mlir::Value offset = dynamicMemSize;
        mlir::Value size =
            builder.create<mlir::arith::ConstantIndexOp>(loc, elemSize);

        auto flags = mlir::arith::IntegerOverflowFlags::nsw |
                     mlir::arith::IntegerOverflowFlags::nuw;
        int i = 0;
        for (auto s : resType.getShape()) {
          mlir::Value v;
          if (mlir::ShapedType::isDynamic(s)) {
            v = sizes[i++];
          } else {
            v = builder.create<mlir::arith::ConstantIndexOp>(loc, s);
          }

          size = builder.create<mlir::arith::MulIOp>(loc, size, v, flags);
        }
        dynamicMemSize = builder.create<mlir::arith::AddIOp>(
            loc, dynamicMemSize, size, flags);

        mlir::OpBuilder::InsertionGuard g(builder);
        builder.setInsertionPointAfter(dynamicMem.getDefiningOp());
        return builder.create<mlir::memref::ViewOp>(loc, resType, dynamicMem,
                                                    offset, sizes);
      };

      auto allocVisitor = [&](mlir::Operation *op) -> mlir::WalkResult {
        if (!mlir::isa<mlir::memref::AllocOp, mlir::memref::AllocaOp>(op))
          return mlir::WalkResult::advance();

        auto resType = mlir::cast<mlir::MemRefType>(op->getResult(0).getType());
        if (resType.getMemorySpace() != wgAttr || resType.hasStaticShape())
          return mlir::WalkResult::advance();

        auto width = layout.getTypeSize(resType.getElementType());
        if (!width || !width.isFixed()) {
          op->emitError("Invalid element type size");
          return mlir::WalkResult::interrupt();
        }

        mlir::SmallVector<mlir::Value> sizes;
        for (auto size : op->getOperands()) {
          if (dom.properlyDominates(size, launch)) {
            sizes.emplace_back(size);
          } else if (mlir::Value mapped = mapping.lookupOrNull(size)) {
            sizes.emplace_back(mapped);
          } else {
            op->emitError("Invalid shared mem size");
            return mlir::WalkResult::interrupt();
          }
        }

        mlir::Value ptr = getDynamicMem(width, sizes, resType);
        op->replaceAllUsesWith(mlir::ArrayRef(ptr));
        op->erase();

        return mlir::WalkResult::advance();
      };
      if (launch.walk(allocVisitor).wasInterrupted())
        return mlir::WalkResult::interrupt();

      if (dynamicMemSize) {
        if (launch.getDynamicSharedMemorySize()) {
          launch->emitError("Dynamic shared mem already set");
          return mlir::WalkResult::interrupt();
        }

        dynamicMemSize = builder.create<mlir::arith::IndexCastOp>(
            launch.getLoc(), i32, dynamicMemSize);
        launch.getDynamicSharedMemorySizeMutable().assign(dynamicMemSize);
      }

      return mlir::WalkResult::skip();
    };

    if (getOperation()
            ->walk<mlir::WalkOrder::PreOrder>(visitor)
            .wasInterrupted())
      return signalPassFailure();
  }
};
} // namespace
