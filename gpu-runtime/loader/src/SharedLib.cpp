// SPDX-FileCopyrightText: 2025 The HC Authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "SharedLib.hpp"

#include <cassert>
#include <string>

#ifdef __linux__
#include <dlfcn.h>
#elif defined(_WIN32) || defined(_WIN64)
#define NOMINMAX
#include <windows.h>
#endif // __linux__

DynamicLibHelper::DynamicLibHelper(const char *libName,
                                   ErrorReporter reportError, void *ctx) {
  assert(libName);

#ifdef __linux__
  _handle = dlopen(libName, RTLD_NOLOAD | RTLD_NOW | RTLD_LOCAL);
  if (!_handle) {
    char *error = dlerror();
    reportError(
        ctx, ("Could not load library " + std::string(libName) +
              ". Error encountered: " + std::string(error ? error : "<null>"))
                 .c_str());
  }
#elif defined(_WIN32) || defined(_WIN64)
  _handle = LoadLibraryExA(libName, nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
  if (!_handle)
    reportError(ctx,
                ("Could not load library " + std::string(libName)).c_str());
#endif
}

DynamicLibHelper::~DynamicLibHelper() { reset(); }

void DynamicLibHelper::reset() {
  if (!_handle)
    return;

#ifdef __linux__
  dlclose(_handle);
#elif defined(_WIN32) || defined(_WIN64)
  FreeLibrary((HMODULE)_handle);
#endif
  _handle = nullptr;
}

void *DynamicLibHelper::getSymbol(const char *symName,
                                  ErrorReporter reportError, void *ctx) {
#ifdef __linux__
  void *sym = dlsym(_handle, symName);

  if (!sym) {
    char *error = dlerror();
    reportError(ctx, ("Could not retrieve symbol " + std::string(symName) +
                      ". Error encountered: " + std::string(error))
                         .c_str());
  }

#elif defined(_WIN32) || defined(_WIN64)
  void *sym = (void *)GetProcAddress((HMODULE)_handle, symName);

  if (!sym)
    reportError(ctx,
                ("Could not retrieve symbol " + std::string(symName)).c_str());
#endif

  return static_cast<void *>(sym);
}
