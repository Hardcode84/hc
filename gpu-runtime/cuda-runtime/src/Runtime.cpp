// SPDX-FileCopyrightText: 2025 The HC Authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hc-cuda-runtime_export.h"

#include <algorithm>
#include <cassert>
#include <charconv>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <utility>

#define OFFLOAD_API_EXPORT HC_CUDA_RUNTIME_EXPORT
#include "offload_api.h"

#include "cuda.h"

namespace {

#define REPORT_IF_ERROR(expr) device->checkStatus(#expr, (expr))

struct Device {
  Device(int id, OlErrorCallback callback, void *c)
      : errCallback(callback), ctx(c), index(id) {}

  OlErrorCallback errCallback = nullptr;
  void *ctx = nullptr;
  int index = 0;
  CUcontext context = nullptr;

  int initCountext() {
    auto device = this;
    static int init = [&]() { return REPORT_IF_ERROR(cuInit(/*flags=*/0)); }();
    if (init)
      return init;

    CUdevice cuDev;
    if (REPORT_IF_ERROR(cuDeviceGet(&cuDev, /*ordinal=*/index)))
      return 1;

    return REPORT_IF_ERROR(cuDevicePrimaryCtxRetain(&context, cuDev));
  }

  template <typename... Args>
  void printf(OlSeverity sev, const char *fmt, Args... args) {
    if (!errCallback)
      return;

    char buffer[256] = {0};
    auto len = std::size(buffer);
    snprintf(buffer, len - 1, fmt, args...);
    errCallback(ctx, OlSeverity::Error, buffer);
  }

  template <typename... Args> void printfErr(const char *fmt, Args... args) {
    printf(OlSeverity::Error, fmt, std::forward<Args>(args)...);
  }

  int checkStatus(const char *expr, CUresult result) {
    if (!result)
      return 0;

    if (errCallback) {
      const char *name = nullptr;
      cuGetErrorName(result, &name);
      if (!name)
        name = "<unknown>";

      printfErr("'%s' failed with '%s'\n", expr, name);
    }
    return 1;
  }

  template <typename F> int executeInContext(F &&f) {
    auto device = this;
    if (REPORT_IF_ERROR(cuCtxPushCurrent(context)))
      return 1;

    int res = f();
    if (REPORT_IF_ERROR(cuCtxPopCurrent(nullptr)))
      return 1;

    return res;
  }
};

struct Module {
  Module(Device *dev, const void *data) : device(dev) {
    REPORT_IF_ERROR(cuModuleLoadData(&module, data));
  }

  ~Module() {
    if (module)
      REPORT_IF_ERROR(cuModuleUnload(module));
  }

  bool isValid() const { return module; }

  Device *device = nullptr;
  CUmodule module = nullptr;
};

struct Queue {
  Queue(Device *dev) : device(dev) {
    REPORT_IF_ERROR(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
  }

  ~Queue() {
    if (stream)
      REPORT_IF_ERROR(cuStreamDestroy(stream));
  }

  bool isValid() const { return stream; }

  Device *device = nullptr;
  CUstream stream = nullptr;
};
} // namespace

OlDevice olCreateDevice(const char *desc, OlErrorCallback errCallback,
                        void *ctx) noexcept {
  std::string_view start("cuda:");
  std::string_view descStr(desc);

  auto reportError = [&](const char *err) -> void * {
    if (errCallback)
      errCallback(ctx, OlSeverity::Error, err);

    return nullptr;
  };
  if (descStr.substr(0, start.size()) != start)
    return reportError("Invalid device desc");

  descStr = descStr.substr(start.size());

  int deviceId = 0;
  if (std::from_chars(descStr.data(), descStr.data() + descStr.size(), deviceId)
          .ec != std::errc{})
    return reportError("Invalid device id");

  auto device = std::make_unique<Device>(deviceId, errCallback, ctx);
  if (device->initCountext())
    return nullptr;

  return device.release();
}
void olReleaseDevice(OlDevice dev) noexcept {
  delete static_cast<Device *>(dev);
}

OlModule olCreateModule(OlDevice dev, const void *data,
                        size_t /*len*/) noexcept {
  auto device = static_cast<Device *>(dev);
  std::unique_ptr<Module> module;
  if (device->executeInContext([&]() {
        module = std::make_unique<Module>(device, data);
        if (!module->isValid())
          return 1;

        return 0;
      }))
    return nullptr;

  return module.release();
}
void olReleaseModule(OlModule mod) noexcept {
  delete static_cast<Module *>(mod);
}

OlKernel olGetKernel(OlModule mod, const char *name) noexcept {
  auto module = static_cast<Module *>(mod);
  auto device = module->device;
  CUfunction function = nullptr;
  if (REPORT_IF_ERROR(cuModuleGetFunction(&function, module->module, name)))
    return nullptr;

  return function;
}

void olReleaseKernel(OlKernel /*k*/) noexcept {
  // Nothing
}

static uint32_t upPow2(uint32_t x) {
  assert(x > 0);
  x--;
  x |= (x >> 1);
  x |= (x >> 2);
  x |= (x >> 4);
  x |= (x >> 8);
  x |= (x >> 16);
  x++;
  return x;
}

int olSuggestBlockSize(OlKernel k, const size_t *globalsSizes,
                       size_t *blockSizesRet, size_t nDims) noexcept {
  if (nDims < 1 || nDims > 3)
    return 1;

  size_t maxWgSize = std::numeric_limits<int>::max();
  for (size_t i = 0; i < nDims; ++i) {
    auto gsize = globalsSizes[i];
    if (gsize > 1) {
      auto lsize = std::min(gsize, maxWgSize);
      blockSizesRet[i] = lsize;
      maxWgSize /= upPow2(lsize);
    } else {
      blockSizesRet[i] = 1;
    }
  }

  return 0;
}

OlQueue olCreateQueue(OlDevice dev) noexcept {
  auto device = static_cast<Device *>(dev);

  std::unique_ptr<Queue> queue;
  if (device->executeInContext([&]() {
        queue = std::make_unique<Queue>(device);
        if (!queue->isValid())
          return 1;

        return 0;
      }))
    return nullptr;

  return queue.release();
}

void olReleaseQueue(OlQueue q) noexcept { delete static_cast<Queue *>(q); }

int olSyncQueue(OlQueue q) noexcept {
  auto queue = static_cast<Queue *>(q);
  auto device = queue->device;
  return REPORT_IF_ERROR(cuStreamSynchronize(queue->stream));
}

void *olAllocDevice(OlQueue q, size_t size, size_t /*align*/) noexcept {
  auto queue = static_cast<Queue *>(q);
  auto device = queue->device;

  CUdeviceptr ptr = 0;
  if (device->executeInContext([&]() {
        if (REPORT_IF_ERROR(cuMemAlloc(&ptr, size)))
          return 1;

        return 0;
      }))
    return nullptr;

  return reinterpret_cast<void *>(ptr);
}

void olDeallocDevice(OlQueue q, void *data) noexcept {
  auto queue = static_cast<Queue *>(q);
  auto device = queue->device;
  REPORT_IF_ERROR(cuMemFree(reinterpret_cast<CUdeviceptr>(data)));
}

int olLaunchKernel(OlQueue q, OlKernel k, const size_t *gridSizes,
                   const size_t *blockSizes, size_t nDims, void **args,
                   size_t /*nArgs*/, size_t sharedMemSize) noexcept {
  auto queue = static_cast<Queue *>(q);
  auto device = queue->device;
  if (nDims != 3) {
    device->printfErr("Invalid launch dims: %d", static_cast<int>(nDims));
    return 1;
  }

  auto kernel = static_cast<CUfunction>(k);
  return device->executeInContext([&]() {
    return REPORT_IF_ERROR(cuLaunchKernel(
        kernel, gridSizes[0], gridSizes[1], gridSizes[2], blockSizes[0],
        blockSizes[1], blockSizes[2], static_cast<unsigned>(sharedMemSize),
        queue->stream, args, /*extra*/ nullptr));
  });
}
