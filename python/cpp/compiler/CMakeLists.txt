# SPDX-FileCopyrightText: 2024 The HC Authors
# SPDX-FileCopyrightText: 2025 The HC Authors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

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

find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(nanobind REQUIRED)

# TODO: something better
set(_python_sources ${LLVM_MAIN_INCLUDE_DIR}/../src/python/)
if(NOT EXISTS ${_python_sources})
  message(FATAL_ERROR "MLIR python bindings not found")
endif()

set(BINDINGS_COMPONENTS
  MLIRPythonCAPI.HeaderSources
  MLIRPythonExtension.Core
  )

set(MLIR_BINDINGS_SOURCES)
foreach(_component IN LISTS BINDINGS_COMPONENTS)
   file(GLOB_RECURSE _tmp_src ${_python_sources}/${_component}/*.cpp ${_python_sources}/*.h )
   list(APPEND MLIR_BINDINGS_SOURCES ${_tmp_src})
endforeach()


set(SOURCES_LIST
    src/CompilerFront.cpp
    src/CompilerFront.hpp
    src/Context.cpp
    src/Context.hpp
    src/Dispatcher.cpp
    src/Dispatcher.hpp
    src/DispatcherBase.cpp
    src/DispatcherBase.hpp
    src/PyModule.cpp
    src/PyWrappers.cpp
    src/PyWrappers.hpp
    src/TypingDispatcher.cpp
    src/TypingDispatcher.hpp
    src/TypingPipeline.cpp
    src/TypingPipeline.hpp
    src/Utils.cpp
    src/Utils.hpp
  )

nanobind_add_module(${python_module_name} MODULE
  ${SOURCES_LIST}
  ${MLIR_BINDINGS_SOURCES}
)

target_include_directories(${python_module_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(${python_module_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_link_libraries(${python_module_name} PRIVATE
    hc_core
    hc_frontend
    Python::Python
    Python::Module
    LLVM${LLVM_NATIVE_ARCH}AsmParser
    LLVM${LLVM_NATIVE_ARCH}CodeGen
    LLVM${LLVM_NATIVE_ARCH}Desc
    LLVMOrcJIT
    LLVMTarget
    MLIRCAPIDebug
    MLIRCAPIIR
    MLIRCAPIInterfaces
    MLIRCAPITransforms
    MLIRPass
    MLIRToLLVMIRTranslationRegistration
    ${mlir_dialect_libs}
    ${mlir_conversion_libs}
    ${mlir_extension_libs}
  )

apply_llvm_compile_flags(${python_module_name})

target_include_directories(${python_module_name} PRIVATE
    ${LLVM_INCLUDE_DIRS}
    ${MLIR_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/../shared/include
    ${_python_sources}/MLIRPythonExtension.Core
  )

install(TARGETS ${python_module_name} DESTINATION ${CMAKE_INSTALL_PREFIX})
install(DIRECTORY ${LLVM_TOOLS_BINARY_DIR} DESTINATION "${CMAKE_INSTALL_PREFIX}/llvm" FILES_MATCHING PATTERN "ld.lld*")
