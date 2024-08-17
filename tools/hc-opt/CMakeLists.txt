# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

project(hc-opt LANGUAGES CXX C)

find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

get_property(mlir_dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(mlir_conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(mlir_extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

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
    ${mlir_dialect_libs}
    ${mlir_conversion_libs}
    ${mlir_extension_libs}
    MLIROptLib
    hc_core
    )
