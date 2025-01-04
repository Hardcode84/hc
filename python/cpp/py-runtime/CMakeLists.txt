# SPDX-FileCopyrightText: 2025 The HC Authors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

project(hc-python-runtime LANGUAGES CXX C)

find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(nanobind REQUIRED)

include(GenerateExportHeader)

set(SOURCES_LIST
    src/PyRuntime.cpp
    src/PyABI.cpp
  )
set(HEADERS_LIST
    src/PyABI.hpp
  )

add_library(${PROJECT_NAME} SHARED ${SOURCES_LIST} ${HEADERS_LIST})
generate_export_header(${PROJECT_NAME})

nanobind_compile_options(${PROJECT_NAME})
nanobind_link_options(${PROJECT_NAME})
nanobind_build_library(nanobind-static)
target_link_libraries(${PROJECT_NAME} PRIVATE
    nanobind-static
    Python::Python
    Python::Module
  )

target_include_directories(${PROJECT_NAME} PRIVATE
    ${PROJECT_BINARY_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../shared/include
  )


install(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_INSTALL_PREFIX})
