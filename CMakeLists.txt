# SPDX-FileCopyrightText: 2024 The HC Authors
# SPDX-FileCopyrightText: 2025 The HC Authors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

cmake_minimum_required(VERSION 3.20)

project(hc LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN 1)

if(POLICY CMP0077)
    cmake_policy(SET CMP0077 NEW)
endif()

if(POLICY CMP0116)
    cmake_policy(SET CMP0116 OLD)
endif()

if(POLICY CMP0148)
    cmake_policy(SET CMP0148 OLD)
endif()

option(HC_ENABLE_TOOLS "Enable Tools" ON)
option(HC_ENABLE_PYTHON "Enable Python" OFF)
option(HC_ENABLE_TESTS "Enable Tests" OFF)
option(HC_ENABLE_GPU_LOADER "Enable GPU loader" OFF)
option(HC_ENABLE_HIP_RUNTIME "Enable GPU HIP runtime" OFF)
option(HC_ENABLE_CUDA_RUNTIME "Enable GPU Cuda runtime" OFF)

message(STATUS "HC_ENABLE_TOOLS ${HC_ENABLE_TOOLS}")
message(STATUS "HC_ENABLE_PYTHON ${HC_ENABLE_PYTHON}")
message(STATUS "HC_ENABLE_TESTS ${HC_ENABLE_TESTS}")
message(STATUS "HC_ENABLE_GPU_LOADER ${HC_ENABLE_GPU_LOADER}")
message(STATUS "HC_ENABLE_HIP_RUNTIME ${HC_ENABLE_HIP_RUNTIME}")
message(STATUS "HC_ENABLE_CUDA_RUNTIME ${HC_ENABLE_CUDA_RUNTIME}")

# Disable IPO to speed up pybind11 linking.
# TODO: Enable for non-develop builds
option(CMAKE_INTERPROCEDURAL_OPTIMIZATION "Interprocedural optimization" OFF)

message(STATUS "CMAKE_INTERPROCEDURAL_OPTIMIZATION ${CMAKE_INTERPROCEDURAL_OPTIMIZATION}")

macro(apply_llvm_compile_flags target)
    if (MSVC)
        target_compile_options(${target} PRIVATE /EHsc)
    endif ()
    target_compile_definitions(${target} PRIVATE ${LLVM_DEFINITIONS})
endmacro()

add_subdirectory(mlir)
add_subdirectory(frontend)

if (HC_ENABLE_GPU_LOADER)
  add_subdirectory(gpu-runtime/loader)
endif()

if (HC_ENABLE_HIP_RUNTIME)
  add_subdirectory(gpu-runtime/hip-runtime)
endif()

if (HC_ENABLE_CUDA_RUNTIME)
  add_subdirectory(gpu-runtime/cuda-runtime)
endif()

if (HC_ENABLE_PYTHON)
  add_subdirectory(python)
endif()

if (HC_ENABLE_TOOLS)
  add_subdirectory(tools/pyfront)
  add_subdirectory(tools/hc-opt)
endif()

if (HC_ENABLE_TESTS)
  add_subdirectory(tests)
endif()
