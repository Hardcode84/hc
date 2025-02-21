# SPDX-FileCopyrightText: 2024 The HC Authors
# SPDX-FileCopyrightText: 2025 The HC Authors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

project(pyfront LANGUAGES CXX C)

find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(nanobind REQUIRED)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

set(SOURCES_LIST
    Main.cpp
  )
set(HEADERS_LIST
  )

add_executable(${PROJECT_NAME} ${SOURCES_LIST} ${HEADERS_LIST})

apply_llvm_compile_flags(${PROJECT_NAME})

target_include_directories(${PROJECT_NAME} PRIVATE
    ${LLVM_INCLUDE_DIRS}
    ${MLIR_INCLUDE_DIRS}
  )

target_link_libraries(${PROJECT_NAME} PRIVATE
    hc_core
    hc_frontend
    Python::Python
    MLIRPass
  )
