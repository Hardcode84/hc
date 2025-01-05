// SPDX-FileCopyrightText: 2025 The HC Authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hc-hip-runtime_export.h"

#include <algorithm>
#include <charconv>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <utility>

#define OFFLOAD_API_EXPORT HC_HIP_RUNTIME_EXPORT
#include "offload_api.h"

#include "hip/hip_runtime.h"

namespace {

#define HIP_REPORT_IF_ERROR(expr) device->checkHipStatus(#expr, (expr))

struct Device {
  Device(int id, OlErrorCallback callback, void *c)
      : errCallback(callback), ctx(c), index(id) {}

  OlErrorCallback errCallback = nullptr;
  void *ctx = nullptr;
  int index = 0;

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

  int checkHipStatus(const char *expr, hipError_t result) {
    if (!result)
      return 0;

    if (errCallback) {
      const char *name = hipGetErrorName(result);
      if (!name)
        name = "<unknown>";

      printfErr("'%s' failed with '%s'\n", expr, name);
    }
    return 1;
  }

  int setCurrent() {
    auto device = this;
    return HIP_REPORT_IF_ERROR(hipSetDevice(index));
  }
};

struct Module {
  Module(Device *dev, const void *data) : device(dev) {
    HIP_REPORT_IF_ERROR(hipModuleLoadData(&module, data));
  }

  ~Module() {
    if (module)
      HIP_REPORT_IF_ERROR(hipModuleUnload(module));
  }

  bool isValid() const { return module; }

  Device *device = nullptr;
  hipModule_t module = nullptr;
};

struct Queue {
  Queue(Device *dev) : device(dev) {
    HIP_REPORT_IF_ERROR(hipStreamCreate(&stream));
  }

  ~Queue() {
    if (stream)
      HIP_REPORT_IF_ERROR(hipStreamDestroy(stream));
  }

  bool isValid() const { return stream; }

  Device *device = nullptr;
  hipStream_t stream = nullptr;
};
} // namespace

OlDevice olCreateDevice(const char *desc, OlErrorCallback errCallback,
                        void *ctx) noexcept {
  std::string_view start("hip:");
  std::string_view descStr(desc);

  auto reportError = [&](const char *err) -> void * {
    if (errCallback)
      errCallback(ctx, OlSeverity::Error, err);

    return nullptr;
  };
  if (descStr.substr(0, start.size()) != start)
    return reportError("Invalid device desc");

  descStr = descStr.substr(start.size() + 1);

  int deviceId = 0;
  if (std::from_chars(descStr.data(), descStr.data() + descStr.size(), deviceId)
          .ec != std::errc{})
    return reportError("Invalid device id");

  auto device = std::make_unique<Device>(deviceId, errCallback, ctx);
  if (device->setCurrent() != 0)
    return nullptr;

  return device.release();
}
void olReleaseDevice(OlDevice dev) noexcept {
  delete static_cast<Device *>(dev);
}

OlModule olCreateModule(OlDevice dev, const void *data,
                        size_t /*len*/) noexcept {
  auto device = static_cast<Device *>(dev);
  if (device->setCurrent())
    return nullptr;

  auto module = std::make_unique<Module>(device, data);
  if (!module->isValid())
    return nullptr;

  return module.release();
}
void olReleaseModule(OlModule mod) noexcept {
  delete static_cast<Module *>(mod);
}

OlKernel olGetKernel(OlModule mod, const char *name) noexcept {
  auto module = static_cast<Module *>(mod);
  auto device = module->device;
  hipFunction_t function = nullptr;
  if (HIP_REPORT_IF_ERROR(
          hipModuleGetFunction(&function, module->module, name)))
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
  if (device->setCurrent())
    return nullptr;

  auto queue = std::make_unique<Queue>(device);

  if (!queue->isValid())
    return nullptr;

  return queue.release();
}

void olReleaseQueue(OlQueue q) noexcept { delete static_cast<Queue *>(q); }

int olSyncQueue(OlQueue q) noexcept {
  auto queue = static_cast<Queue *>(q);
  auto device = queue->device;
  return HIP_REPORT_IF_ERROR(hipStreamSynchronize(queue->stream));
}

void *olAllocDevice(OlQueue q, size_t size, size_t /*align*/) noexcept {
  auto queue = static_cast<Queue *>(q);
  auto device = queue->device;
  if (device->setCurrent())
    return nullptr;

  void *ptr;
  if (HIP_REPORT_IF_ERROR(hipMalloc(&ptr, size)))
    return nullptr;

  return ptr;
}

void olDeallocDevice(OlQueue q, void *data) noexcept {
  auto queue = static_cast<Queue *>(q);
  auto device = queue->device;
  HIP_REPORT_IF_ERROR(hipFree(data));
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

  auto kernel = static_cast<hipFunction_t>(k);
  return HIP_REPORT_IF_ERROR(hipModuleLaunchKernel(
      kernel, gridSizes[0], gridSizes[1], gridSizes[2], blockSizes[0],
      blockSizes[1], blockSizes[2], static_cast<unsigned>(sharedMemSize),
      queue->stream, args, /*extra*/ nullptr));
}
