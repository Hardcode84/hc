# SPDX-FileCopyrightText: 2025 The HC Authors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

project(hc-hip-runtime LANGUAGES CXX C)

# Configure ROCm support.
set(HIP_PLATFORM "amd")
if (DEFINED ENV{ROCM_PATH})
  set(ROCM_PATH $ENV{ROCM_PATH} CACHE PATH "Path to which ROCm has been installed")
elseif (NOT DEFINED ROCM_PATH)
  set(ROCM_PATH "/opt/rocm" CACHE PATH "Path to which ROCm has been installed")
endif()
message("ROCM_PATH $ENV{ROCM_PATH} ${ROCM_PATH}")
# A lot of the ROCm CMake files expect to find their own dependencies in
# CMAKE_PREFIX_PATH and don't respect PATHS or HINTS :( .
# Therefore, temporarily add the ROCm path to CMAKE_PREFIX_PATH so we can
# load HIP, then remove it
set(REAL_CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH}")
list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH} "${ROCM_PATH}/hip")
find_package(hip REQUIRED)
set(CMAKE_PREFIX_PATH "${REAL_CMAKE_PREFIX_PATH}")

include(GenerateExportHeader)

set(SOURCES_LIST
    src/Runtime.cpp
  )
set(HEADERS_LIST
  )

add_library(${PROJECT_NAME} SHARED ${SOURCES_LIST} ${HEADERS_LIST})
generate_export_header(${PROJECT_NAME})

target_link_libraries(${PROJECT_NAME} PRIVATE
    hip::host
    hip::amdhip64
  )

target_include_directories(${PROJECT_NAME} PRIVATE
    ${PROJECT_BINARY_DIR}
    ${CMAKE_SOURCE_DIR}/gpu-runtime/common/include
  )


install(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_INSTALL_PREFIX})
