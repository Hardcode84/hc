// SPDX-FileCopyrightText: 2025 The HC Authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hc-gpu-runtime-loader_export.h"

#include <cassert>
#include <memory>
#include <string_view>

#define OFFLOAD_API_EXPORT HC_GPU_RUNTIME_LOADER_EXPORT
#include "offload_api.h"

#include "SharedLib.hpp"

namespace {
struct ErrorContext {
  OlErrorCallback errorCallback = nullptr;
  void *ctx = nullptr;

  static void reportError(void *ctx, const char *str) {
    auto errCtx = static_cast<ErrorContext *>(ctx);
    errCtx->errorCallback(errCtx->ctx, OlSeverity::Error, str);
  }

  void reportError(const char *str) {
    errorCallback(ctx, OlSeverity::Error, str);
  }
};

#define INIT_FUNC(func)                                                        \
  do {                                                                         \
    this->func = this->lib.getSymbol<decltype(this->func)>(                    \
        #func, &ErrorContext::reportError, &errCtx);                           \
    if (!this->func)                                                           \
      failed = true;                                                           \
  } while (false)
#define DECL_FUNC(func) decltype(&ol##func) func;

struct API {
  API(const char *libName, ErrorContext &errCtx)
      : lib(libName, &ErrorContext::reportError, &errCtx) {
    if (!lib)
      return;

    bool failed = false;
    INIT_FUNC(CreateDevice);
    INIT_FUNC(ReleaseDevice);
    INIT_FUNC(CreateModule);
    INIT_FUNC(ReleaseModule);
    INIT_FUNC(GetKernel);
    INIT_FUNC(ReleaseKernel);
    INIT_FUNC(SuggestBlockSize);
    INIT_FUNC(CreateSyncQueue);
    INIT_FUNC(ReleaseQueue);
    INIT_FUNC(AllocDevice);
    INIT_FUNC(DeallocDevice);
    INIT_FUNC(LaunchKernel);
    if (failed)
      lib.reset();
  }

  void reset() { lib.reset(); }

  explicit operator bool() const { return static_cast<bool>(lib); }

  DynamicLibHelper lib;

  DECL_FUNC(CreateDevice);
  DECL_FUNC(ReleaseDevice);
  DECL_FUNC(CreateModule);
  DECL_FUNC(ReleaseModule);
  DECL_FUNC(GetKernel);
  DECL_FUNC(ReleaseKernel);
  DECL_FUNC(SuggestBlockSize);
  DECL_FUNC(CreateSyncQueue);
  DECL_FUNC(ReleaseQueue);
  DECL_FUNC(AllocDevice);
  DECL_FUNC(DeallocDevice);
  DECL_FUNC(LaunchKernel);
};

#undef INIT_FUNC
#undef DECL_FUNC

struct LoaderDevice {
  LoaderDevice(const char *lib, const char *desc, ErrorContext &errCtx)
      : api(lib, errCtx) {
    if (!api)
      return;

    device = api.CreateDevice(desc, errCtx.errorCallback, errCtx.ctx);
    if (!device)
      api.reset();
  }

  ~LoaderDevice() {
    if (device)
      api.ReleaseDevice(device);
  }

  bool isValid() const { return static_cast<bool>(api); }

  API api;
  OlDevice device = nullptr;
};

struct LoaderModule {
  LoaderModule(LoaderDevice *dev, const void *data, size_t len) : device(dev) {
    assert(device);
    module = getAPI().CreateModule(device->device, data, len);
  }

  ~LoaderModule() {
    if (module)
      getAPI().ReleaseModule(module);
  }

  bool isValid() const { return module; }

  const API &getAPI() const { return device->api; }

  LoaderDevice *device = nullptr;
  OlModule module = nullptr;
};

struct LoaderKernel {
  LoaderKernel(LoaderModule *mod, const char *name) : device(mod->device) {
    kernel = getAPI().GetKernel(mod->module, name);
  }

  ~LoaderKernel() {
    if (kernel)
      getAPI().ReleaseKernel(kernel);
  }

  bool isValid() const { return kernel; }

  const API &getAPI() const { return device->api; }

  LoaderDevice *device = nullptr;
  OlKernel kernel = nullptr;
};

struct LoaderQueue {
  LoaderQueue(LoaderDevice *dev) : device(dev) {
    assert(device);
    queue = getAPI().CreateSyncQueue(device->device);
  }

  ~LoaderQueue() {
    if (queue)
      getAPI().ReleaseQueue(queue);
  }

  bool isValid() const { return queue; }

  const API &getAPI() const { return device->api; }

  LoaderDevice *device = nullptr;
  OlModule queue = nullptr;
};
} // namespace

static std::string getLibName(std::string_view name) {
#ifdef __linux__
  return "lib" + std::string(name) + ".so";
#elif defined(_WIN32) || defined(_WIN64)
  return std::string(name) + ".dll";
#else
#error "Unsupported platform"
#endif
}

OlDevice olCreateDevice(const char *desc, OlErrorCallback errCallback,
                        void *ctx) noexcept {
  ErrorContext errCtx{errCallback, ctx};
  std::string_view descStr(desc);
  auto sep = descStr.find(':');
  if (sep == descStr.npos) {
    std::string str = "Invalid device desc: " + std::string(descStr);
    errCtx.reportError(str.c_str());
    return nullptr;
  }
  auto libname =
      getLibName("hc-" + std::string(descStr.substr(0, sep)) + "-runtime");
  auto device = std::make_unique<LoaderDevice>(libname.c_str(), desc, errCtx);
  if (!device->isValid())
    return nullptr;

  return device.release();
}
void olReleaseDevice(OlDevice dev) noexcept {
  delete static_cast<LoaderDevice *>(dev);
}

OlModule olCreateModule(OlDevice dev, const void *data, size_t len) noexcept {
  auto device = static_cast<LoaderDevice *>(dev);
  auto module = std::make_unique<LoaderModule>(device, data, len);
  if (!module->isValid())
    return nullptr;

  return module.release();
}
void olReleaseModule(OlModule mod) noexcept {
  delete static_cast<LoaderModule *>(mod);
}

OlKernel olGetKernel(OlModule mod, const char *name) noexcept {
  auto module = static_cast<LoaderModule *>(mod);
  auto kernel = std::make_unique<LoaderKernel>(module, name);
  if (!kernel->isValid())
    return nullptr;

  return kernel.release();
}
void olReleaseKernel(OlKernel k) noexcept {
  delete static_cast<LoaderKernel *>(k);
}

int olSuggestBlockSize(OlKernel k, const size_t *globalsSizes,
                       size_t *blockSizesRet, size_t nDims) noexcept {
  auto kernel = static_cast<LoaderKernel *>(k);
  return kernel->getAPI().SuggestBlockSize(kernel->kernel, globalsSizes,
                                           blockSizesRet, nDims);
}

OlQueue olCreateSyncQueue(OlDevice dev) noexcept {
  auto device = static_cast<LoaderDevice *>(dev);
  auto queue = std::make_unique<LoaderQueue>(device);
  if (!queue->isValid())
    return nullptr;

  return queue.release();
}
void olReleaseQueue(OlQueue q) noexcept {
  delete static_cast<LoaderQueue *>(q);
}

void *olAllocDevice(OlQueue q, size_t size, size_t align) noexcept {
  auto queue = static_cast<LoaderQueue *>(q);
  return queue->getAPI().AllocDevice(queue->queue, size, align);
}

void olDeallocDevice(OlQueue q, void *data) noexcept {
  auto queue = static_cast<LoaderQueue *>(q);
  queue->getAPI().DeallocDevice(queue->queue, data);
}

int olLaunchKernel(OlQueue q, OlKernel k, const size_t *gridSizes,
                   const size_t *blockSizes, size_t nDims, void **args,
                   size_t nArgs, size_t sharedMemSize) noexcept {
  auto queue = static_cast<LoaderQueue *>(q);
  auto kernel = static_cast<LoaderKernel *>(k);
  auto &api = queue->getAPI();
  return api.LaunchKernel(queue->queue, kernel->kernel, gridSizes, blockSizes,
                          nDims, args, nArgs, sharedMemSize);
}
