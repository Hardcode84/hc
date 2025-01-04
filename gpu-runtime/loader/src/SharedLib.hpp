// SPDX-FileCopyrightText: 2025 The HC Authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

class DynamicLibHelper final {
public:
  using ErrorReporter = void (*)(void *, const char *);
  DynamicLibHelper() = delete;
  DynamicLibHelper(const DynamicLibHelper &) = delete;
  DynamicLibHelper(const char *libName, ErrorReporter reportError, void *ctx);

  ~DynamicLibHelper();

  void reset();

  void *getSymbol(const char *symName, ErrorReporter reportError, void *ctx);

  template <typename T>
  T getSymbol(const char *symName, ErrorReporter reportError, void *ctx) {
    return reinterpret_cast<T>(this->getSymbol(symName, reportError, ctx));
  }

  explicit operator bool() const { return _handle; }

private:
  void *_handle = nullptr;
};
