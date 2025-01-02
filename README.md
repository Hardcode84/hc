<!--
SPDX-FileCopyrightText: 2024 The HC Authors
SPDX-FileCopyrightText: 2025 The HC Authors

SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

## Building

```
mkdir cmake_build && cd cmake_build
cmake -G Ninja -Dpybind11_DIR:PATH=<pybin11-install>/share/cmake/pybind11 -DLLVM_DIR:PATH=<llvm-install>/lib/cmake/llvm -DMLIR_DIR:PATH=<llvm-install>/lib/cmake/mlir -DCMAKE_BUILD_TYPE=Release -DHC_ENABLE_TESTS=ON -DLLVM_EXTERNAL_LIT=<path-to-repo>\hc\scripts\runlit.py ..
ninja check-hc
```
