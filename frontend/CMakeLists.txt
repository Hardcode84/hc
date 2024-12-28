# SPDX-FileCopyrightText: 2024 The HC Authors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

project(hc_frontend LANGUAGES CXX C)

find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(nanobind REQUIRED)

set(SOURCES_LIST
    lib/Import.cpp
  )
set(HEADERS_LIST
    include/hc/PyFront/Import.hpp
  )

add_library(${PROJECT_NAME} STATIC ${SOURCES_LIST} ${HEADERS_LIST})

target_link_libraries(${PROJECT_NAME} PRIVATE
    hc_core
    nanobind-static
  )

target_include_directories(${PROJECT_NAME} SYSTEM PRIVATE
    ${MLIR_INCLUDE_DIRS}
    )
target_include_directories(${PROJECT_NAME} PUBLIC
    ./include
  )

apply_llvm_compile_flags(${PROJECT_NAME})
