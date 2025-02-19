# SPDX-FileCopyrightText: 2025 The HC Authors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

project(hc-cuda-runtime LANGUAGES CXX C)

find_package(CUDAToolkit 11.2 REQUIRED)

include(GenerateExportHeader)

set(SOURCES_LIST
    src/Runtime.cpp
  )
set(HEADERS_LIST
  )

add_library(${PROJECT_NAME} SHARED ${SOURCES_LIST} ${HEADERS_LIST})
generate_export_header(${PROJECT_NAME})

target_link_libraries(${PROJECT_NAME} PRIVATE
    CUDA::cuda_driver
  )

target_include_directories(${PROJECT_NAME} PRIVATE
    ${PROJECT_BINARY_DIR}
    ${CMAKE_SOURCE_DIR}/gpu-runtime/common/include
  )


install(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_INSTALL_PREFIX})
